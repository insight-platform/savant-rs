//! [`Picasso`] — `PipelineMsg` → `EncodedMsg` rendering +
//! encoding stage wrapping a user-supplied
//! [`PicassoEngine`].
//!
//! The stage owns Picasso-engine lifecycle and surfaces the
//! configurable parts of rendering, encoding, and output routing:
//!
//! * Captures a [`Router<EncodedMsg>`] at build time (optionally
//!   with a default peer installed via
//!   [`PicassoBuilder::downstream`]) and hands it to the
//!   user-supplied [`PicassoEngineFactory`].  The stage itself
//!   never calls `router.send`; the user's
//!   [`OnEncodedFrame`](picasso::prelude::OnEncodedFrame) impl
//!   (installed on the engine via the factory) is the single send
//!   site for encoded output.
//! * Accepts the engine via factory (required): the factory receives
//!   the wired [`Router<EncodedMsg>`] so the user can build their
//!   own `OnEncodedFrame` and plug it into
//!   [`picasso::prelude::Callbacks::builder`].
//! * Accepts a *source-spec factory* that turns a per-source
//!   `(width, height, fps)` tuple into a
//!   [`SourceSpec`] the first time a
//!   given `source_id` appears.
//! * Optional per-pair delivery hook (`on_delivery`) — invoked
//!   **once per unsealed `(VideoFrame, SharedBuffer)` pair**
//!   before that pair is fed to the engine.  Same signature shape
//!   as
//!   [`DeepStreamFunctionInboxBuilder::on_delivery`](super::deepstream_function::DeepStreamFunctionInboxBuilder::on_delivery),
//!   so applications can mutate frames, attach overlay objects, or
//!   update per-source counters before each pair reaches Picasso.
//! * On [`PipelineMsg::SourceEos`] calls
//!   [`PicassoEngine::send_eos`](picasso::prelude::PicassoEngine::send_eos)
//!   — the downstream
//!   [`EncodedMsg::SourceEos`]
//!   is emitted by the user's own `OnEncodedFrame` callback on
//!   [`OutputMessage::EndOfStream`](picasso::prelude::OutputMessage::EndOfStream),
//!   preserving stream alignment.
//! * On [`Actor::stopping`] calls
//!   [`PicassoEngine::shutdown`](picasso::prelude::PicassoEngine::shutdown).
//!
//! # Grouped builder API
//!
//! Hooks are grouped into three bundles following the cross-stage
//! pattern:
//!
//! * [`PicassoInbox`] — batch-level / per-frame hooks
//!   (`on_delivery`, `on_crop_select`).
//! * [`PicassoResults`] — inbox-forward tee
//!   (`forward_inbox_delivery`, `forward_inbox_source_eos`); both
//!   hooks receive a `Router<PipelineMsg>` wired from
//!   [`PicassoBuilder::forward_inbox_downstream`] so the body owns
//!   the drop/send decision.  Defaults are debug-log + drop —
//!   picasso is a sink for `(VideoFrame, SharedBuffer)` traffic
//!   unless configured otherwise.
//! * [`PicassoCommon`] — loop-level knobs (`poll_timeout`) plus
//!   the user `stopping` hook.
//!
//! ```ignore
//! Picasso::builder(name, 16)
//!     .downstream(peer)
//!     .forward_inbox_downstream(inbox_tee_peer)
//!     .engine_factory(make_engine)
//!     .source_spec_factory(make_spec)
//!     .inbox(
//!         PicassoInbox::builder()
//!             .on_delivery(|ctx, frame, buffer| { /* ... */ Ok(()) })
//!             .on_crop_select(|ctx, frame| None)
//!             .build(),
//!     )
//!     .results(
//!         PicassoResults::builder()
//!             .forward_inbox_delivery(|ctx, router, pairs| {
//!                 // clone + reseal + router.send(...)
//!                 Ok(())
//!             })
//!             .forward_inbox_source_eos(|ctx, router, sid| {
//!                 let _ = router.send(PipelineMsg::SourceEos {
//!                     source_id: sid.to_string(),
//!                 });
//!                 Ok(())
//!             })
//!             .build(),
//!     )
//!     .common(PicassoCommon::builder().build())
//!     .build()?;
//! ```
//!
//! # Runtime invariant
//!
//! Every hook slot on the runtime [`Picasso`] struct is a
//! non-`Option` boxed closure: when the user does not install a
//! hook, the bundle builders substitute the matching
//! `Picasso::default_*` no-op at build time. The actor body
//! never has to branch on whether a hook is present.
//!
//! There is no framework-injected per-frame counter; if user
//! code needs a stable per-frame index it should maintain its
//! own (e.g. via an `Arc<AtomicU64>`).

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::{anyhow, Result};
use deepstream_buffers::{Rect, SharedBuffer, SkiaRenderer, SurfaceView};
use hashbrown::HashSet;
use picasso::prelude::{
    Callbacks, EvictionDecision, GeneralSpec, OnEncodedFrame, OnEviction, OnGpuMat,
    OnObjectDrawSpec, OnRender, OnStreamReset, OutputMessage, PicassoEngine, SourceSpec,
    StreamResetReason,
};
use picasso::EndOfStream;
use savant_core::draw::ObjectDraw;
use savant_core::primitives::frame::VideoFrame;
use savant_core::primitives::object::BorrowedVideoObject;

use crate::envelopes::{BatchDelivery, EncodedMsg, PipelineMsg, SingleDelivery};
use crate::instrument::enter_callback_span;
use crate::router::Router;
use crate::supervisor::StageName;
use crate::{
    Actor, ActorBuilder, BuildCtx, Context, Dispatch, Flow, Handler, HookCtx,
    RemoveSourcePayload, ShutdownPayload, SourceEosPayload, UpdateSourceSpecPayload,
};

/// Cheap, cloneable, deferred handle to the [`PicassoEngine`] built
/// by a [`Picasso`] stage.
///
/// Construct one with [`PicassoEngineHandle::new`], hand a clone to
/// [`PicassoBuilder::engine_handle`], and capture additional clones
/// in any user closure (inbox / results / engine-level) that needs
/// to call engine methods.  The stage binds the underlying `Arc`
/// when its actor factory builds the engine; before that point
/// [`Self::try_get`] returns `None`.
///
/// The handle is `Send + Sync + Clone` and uses [`OnceLock`] under
/// the hood — once bound, `try_get` is lock-free.
#[derive(Clone, Default)]
pub struct PicassoEngineHandle {
    inner: Arc<OnceLock<Arc<PicassoEngine>>>,
}

impl std::fmt::Debug for PicassoEngineHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PicassoEngineHandle")
            .field("bound", &self.inner.get().is_some())
            .finish()
    }
}

impl PicassoEngineHandle {
    /// Create an unbound handle.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return the bound engine, or `None` if the actor's factory
    /// hasn't run yet.
    pub fn try_get(&self) -> Option<Arc<PicassoEngine>> {
        self.inner.get().cloned()
    }

    /// Stage-internal: bind the engine.  Calling [`OnceLock::set`]
    /// twice would silently fail — we log the second attempt so a
    /// configuration error (same handle reused across two stages)
    /// is observable rather than silent.
    pub(crate) fn bind(&self, engine: Arc<PicassoEngine>) {
        if self.inner.set(engine).is_err() {
            log::warn!(
                "PicassoEngineHandle: bind called twice; \
                 the second engine will not be reachable through this handle"
            );
        }
    }
}

/// Default inbox receive-poll cadence.
pub const DEFAULT_POLL_TIMEOUT: Duration = Duration::from_millis(100);

/// Source-spec factory — builds a
/// [`SourceSpec`] for a
/// previously-unseen `source_id` the first time the picasso stage
/// sees a frame for it.  Invoked exactly once per `source_id` per
/// stage instance.
pub type SourceSpecFactory =
    Box<dyn FnMut(&str, u32, u32, i32, i32) -> Result<SourceSpec> + Send + 'static>;

/// Per-pair delivery hook — invoked **once per
/// `(VideoFrame, SharedBuffer)` pair** **before** the pair is fed
/// to
/// [`PicassoEngine::send_frame`](picasso::prelude::PicassoEngine::send_frame).
///
/// Both single-delivery and batched-delivery inbox variants funnel
/// through this loop: the stage unseals the batch on the actor
/// thread and calls the hook in a tight loop, so user code never
/// observes the batching.  The frame and buffer are passed by
/// **shared reference** — the hook body can inspect, clone, or
/// forward them to nested helpers (overlays, analytics, inference
/// taps, …) without consuming, and the stage drives the per-pair
/// engine send after this hook returns.
///
/// Always installed at runtime; the default is a no-op.  Matches
/// the shape of
/// [`DeepStreamFunctionInboxBuilder::on_delivery`](super::deepstream_function::DeepStreamFunctionInboxBuilder::on_delivery)
/// so the cross-stage hook vocabulary is uniform.
pub type OnDeliveryHook = Box<
    dyn FnMut(&mut Context<Picasso>, &VideoFrame, &SharedBuffer) -> Result<()>
        + Send
        + 'static,
>;

/// Inbox-forward hook for delivery batches — fires once per
/// incoming batch *after* [`PicassoInbox::on_delivery`] has run and
/// **before** the per-frame engine feed loop, with a
/// [`Router<PipelineMsg>`] supplied by the stage.  The hook body
/// decides whether to relay the batch (typically by re-sealing the
/// pairs and sending [`PipelineMsg::Deliveries`] / [`PipelineMsg::Delivery`]
/// down the inbox-forward router) or drop it.
///
/// The default — installed automatically when the user omits the
/// setter — emits a single `debug` log line and drops the batch,
/// matching the assumption that picasso is normally a sink for
/// `(VideoFrame, SharedBuffer)` traffic.
pub type OnForwardInboxDeliveryHook = Box<
    dyn FnMut(
            &mut Context<Picasso>,
            &Router<PipelineMsg>,
            &[(VideoFrame, SharedBuffer)],
        ) -> Result<()>
        + Send
        + 'static,
>;

/// Inbox-forward hook for source-EOS sentinels — fires for every
/// [`PipelineMsg::SourceEos`] *after* picasso has flushed the
/// matching engine source via
/// [`PicassoEngine::send_eos`](picasso::prelude::PicassoEngine::send_eos),
/// with a [`Router<PipelineMsg>`] supplied by the stage.
///
/// The default — installed automatically when the user omits the
/// setter — emits a single `debug` log line and drops the EOS,
/// again matching the sink-by-default contract.
pub type OnForwardInboxSourceEosHook = Box<
    dyn FnMut(&mut Context<Picasso>, &Router<PipelineMsg>, &str) -> Result<()>
        + Send
        + 'static,
>;

/// Engine spec factory — invoked once when the [`Picasso`] actor's
/// factory runs.  Returns just the [`GeneralSpec`]; the stage owns
/// `Callbacks` lifecycle and assembles them from the encoded
/// closures in [`PicassoResults`] plus the engine-level closures
/// installed via the top-level builder
/// (`on_render`, `on_gpu_mat`, …).
///
/// Typical implementation:
///
/// ```ignore
/// .engine_factory(|_bx| Ok(GeneralSpec::builder().name("demo").build()))
/// ```
pub type PicassoEngineFactory =
    Box<dyn FnOnce(&BuildCtx) -> Result<GeneralSpec> + Send>;

/// Encoded-frame result hook — fires from the engine worker
/// thread when [`PicassoEngine`] produces an encoded
/// [`OutputMessage::VideoFrame`].  Receives the encoded frame and
/// the [`Router<EncodedMsg>`] wired from
/// [`PicassoBuilder::downstream`] so the body owns the drop / send
/// decision.
///
/// Runs **off the actor thread** — no `Context` argument; closures
/// must be `Fn + Send + Sync`.  Capture an [`Arc<AtomicU64>`] (or a
/// [`PicassoEngineHandle`] if engine access is needed) for shared
/// state.
///
/// Default: forward as `EncodedMsg::Frame { frame, payload: None }`,
/// matching the impl every existing user wrote by hand.
pub type OnEncodedFrameHook =
    Arc<dyn Fn(&HookCtx, &Router<EncodedMsg>, VideoFrame) + Send + Sync + 'static>;

/// Encoded-EOS result hook — fires from the engine worker thread
/// when [`PicassoEngine`] emits an [`OutputMessage::EndOfStream`].
///
/// Default: forward as `EncodedMsg::SourceEos { source_id }`.
pub type OnEncodedSourceEosHook =
    Arc<dyn Fn(&HookCtx, &Router<EncodedMsg>, EndOfStream) + Send + Sync + 'static>;

/// Pre-flush canvas hook — closure form of
/// [`OnRender`](picasso::prelude::OnRender).  Optional; no default.
pub type OnRenderHook =
    Arc<dyn Fn(&HookCtx, &str, &mut SkiaRenderer, &VideoFrame) + Send + Sync + 'static>;

/// GPU-mat-ready hook — closure form of
/// [`OnGpuMat`](picasso::prelude::OnGpuMat).  Optional; no default.
pub type OnGpuMatHook =
    Arc<dyn Fn(&HookCtx, &str, &VideoFrame, &SurfaceView) + Send + Sync + 'static>;

/// Per-object draw-spec override — closure form of
/// [`OnObjectDrawSpec`](picasso::prelude::OnObjectDrawSpec).  Optional;
/// no default.
pub type OnObjectDrawSpecHook = Arc<
    dyn Fn(&HookCtx, &str, &BorrowedVideoObject, Option<&ObjectDraw>) -> Option<ObjectDraw>
        + Send
        + Sync
        + 'static,
>;

/// Source eviction decision — closure form of
/// [`OnEviction`](picasso::prelude::OnEviction).  Optional; no default.
pub type OnEvictionHook =
    Arc<dyn Fn(&HookCtx, &str) -> EvictionDecision + Send + Sync + 'static>;

/// Stream-reset notification — closure form of
/// [`OnStreamReset`](picasso::prelude::OnStreamReset).  Optional; no
/// default.
pub type OnStreamResetHook =
    Arc<dyn Fn(&HookCtx, &str, StreamResetReason) + Send + Sync + 'static>;

// ── Closure → trait adapters ────────────────────────────────────────
//
// `Callbacks::builder().on_*` accepts `impl OnX` (a trait object), so
// closures need a thin adapter that forwards `call` to the closure.
// All adapters are crate-private — the public surface is the closure
// API on `PicassoBuilder` / `PicassoResults`.

struct ClosureOnEncodedFrame {
    on_frame: OnEncodedFrameHook,
    on_eos: OnEncodedSourceEosHook,
    router: Router<EncodedMsg>,
    ctx: HookCtx,
}

impl OnEncodedFrame for ClosureOnEncodedFrame {
    fn call(&self, output: OutputMessage) {
        match output {
            OutputMessage::VideoFrame(frame) => {
                // Cross-thread propagation: encoder callback runs
                // on a worker thread.  `enter_callback_span` clones
                // the frame internally (cheap Arc clone), so the
                // guard's pop runs even after the user hook
                // consumes `frame`.
                let _span = enter_callback_span(
                    &frame,
                    "on_encoded_frame",
                    self.ctx.pipeline_name(),
                    &self.ctx.own_name().to_string(),
                );
                (self.on_frame)(&self.ctx, &self.router, frame);
            }
            OutputMessage::EndOfStream(eos) => {
                // No frame on EOS — no span.
                (self.on_eos)(&self.ctx, &self.router, eos)
            }
        }
    }
}

struct ClosureOnRender {
    hook: OnRenderHook,
    ctx: HookCtx,
}
impl OnRender for ClosureOnRender {
    fn call(&self, source_id: &str, renderer: &mut SkiaRenderer, frame: &VideoFrame) {
        let _g = enter_callback_span(
            frame,
            "on_render",
            self.ctx.pipeline_name(),
            &self.ctx.own_name().to_string(),
        );
        (self.hook)(&self.ctx, source_id, renderer, frame)
    }
}

struct ClosureOnGpuMat {
    hook: OnGpuMatHook,
    ctx: HookCtx,
}
impl OnGpuMat for ClosureOnGpuMat {
    fn call(&self, source_id: &str, frame: &VideoFrame, view: &SurfaceView) {
        let _g = enter_callback_span(
            frame,
            "on_gpu_mat",
            self.ctx.pipeline_name(),
            &self.ctx.own_name().to_string(),
        );
        (self.hook)(&self.ctx, source_id, frame, view)
    }
}

struct ClosureOnObjectDrawSpec {
    hook: OnObjectDrawSpecHook,
    ctx: HookCtx,
}
impl OnObjectDrawSpec for ClosureOnObjectDrawSpec {
    fn call(
        &self,
        source_id: &str,
        object: &BorrowedVideoObject,
        current_spec: Option<&ObjectDraw>,
    ) -> Option<ObjectDraw> {
        // Operates on a single object — no frame in scope, no span.
        (self.hook)(&self.ctx, source_id, object, current_spec)
    }
}

struct ClosureOnEviction {
    hook: OnEvictionHook,
    ctx: HookCtx,
}
impl OnEviction for ClosureOnEviction {
    fn call(&self, source_id: &str) -> EvictionDecision {
        // No frame — no span.
        (self.hook)(&self.ctx, source_id)
    }
}

struct ClosureOnStreamReset {
    hook: OnStreamResetHook,
    ctx: HookCtx,
}
impl OnStreamReset for ClosureOnStreamReset {
    fn call(&self, source_id: &str, reason: StreamResetReason) {
        // No frame — no span.
        (self.hook)(&self.ctx, source_id, reason)
    }
}

/// Per-frame crop-rectangle selector.  Returning `Some(rect)`
/// forwards the rectangle to
/// [`PicassoEngine::send_frame`](picasso::prelude::PicassoEngine::send_frame);
/// returning `None` leaves the engine's default in place (use the
/// whole source surface).  Always installed at runtime; the default
/// returns `None` for every frame.
///
/// Runs on the actor receive loop, between
/// [`PicassoInbox::on_delivery`] and the per-frame engine feed.
/// Argument order is `(ctx, frame)` — the frame's `source_id` is
/// available via [`VideoFrame::get_source_id`] when needed.
pub type OnCropSelectHook = Box<
    dyn FnMut(&mut Context<Picasso>, &VideoFrame) -> Option<Rect> + Send + 'static,
>;

/// User shutdown hook invoked from [`Actor::stopping`] *after* the
/// stage's built-in cleanup (guarded
/// [`PicassoEngine::shutdown`](picasso::prelude::PicassoEngine::shutdown))
/// has completed.
///
/// Runs on the actor thread with full access to the [`Context`].
/// Ideal for final metrics flushes, bespoke log lines, or custom
/// bookkeeping that must observe the shut-down engine state.  The
/// stage's load-bearing cleanup cannot be skipped — users that
/// need to replace `engine.shutdown()` entirely should implement
/// [`Actor`] directly on a bespoke struct.
pub type OnStoppingHook = Box<dyn FnMut(&mut Context<Picasso>) + Send + 'static>;

/// `PipelineMsg` → `EncodedMsg` actor wrapping a
/// [`PicassoEngine`].  Hook slots
/// are non-`Option` by construction — see the runtime invariant in
/// the module docs.
pub struct Picasso {
    engine: Arc<PicassoEngine>,
    source_spec: SourceSpecFactory,
    on_delivery: OnDeliveryHook,
    on_crop_select: OnCropSelectHook,
    forward_inbox: Router<PipelineMsg>,
    forward_inbox_delivery: OnForwardInboxDeliveryHook,
    forward_inbox_source_eos: OnForwardInboxSourceEosHook,
    registered: HashSet<String>,
    poll_timeout: Duration,
    on_stopping: OnStoppingHook,
    shutdown_done: bool,
}

impl Picasso {
    /// Start a fluent builder for a picasso actor registered under
    /// `name` with inbox capacity `capacity`.
    pub fn builder(name: StageName, capacity: usize) -> PicassoBuilder {
        PicassoBuilder::new(name, capacity)
    }

    /// Default per-pair delivery hook — a no-op.  Always installed
    /// when the user omits the inbox setter so the runtime struct
    /// never needs to inspect an `Option`.
    pub fn default_on_delivery(
    ) -> impl FnMut(&mut Context<Picasso>, &VideoFrame, &SharedBuffer) -> Result<()>
           + Send
           + 'static {
        |_ctx, _frame, _buffer| Ok(())
    }

    /// Default `on_crop_select` hook — returns `None` for every
    /// frame, so Picasso uses the whole source surface.  Always
    /// installed when the user omits the inbox setter.
    pub fn default_on_crop_select(
    ) -> impl FnMut(&mut Context<Picasso>, &VideoFrame) -> Option<Rect> + Send + 'static
    {
        |_ctx, _frame| None
    }

    /// Default `forward_inbox_delivery` hook — emits a single
    /// `debug` log line and drops the batch.  Picasso is a sink
    /// for `(VideoFrame, SharedBuffer)` traffic by default; users
    /// who need to tee deliveries downstream override this hook.
    pub fn default_forward_inbox_delivery() -> impl FnMut(
        &mut Context<Picasso>,
        &Router<PipelineMsg>,
        &[(VideoFrame, SharedBuffer)],
    ) -> Result<()>
           + Send
           + 'static {
        |ctx, _router, pairs| {
            log::debug!(
                "[{}] forward_inbox_delivery: dropping {} pair(s)",
                ctx.own_name(),
                pairs.len()
            );
            Ok(())
        }
    }

    /// Default `on_encoded_frame` hook — forwards as
    /// `EncodedMsg::Frame { frame, payload: None }`.  Matches the
    /// `OnEncodedFrame` impl every existing user wrote by hand;
    /// switching to defaults removes the boilerplate.
    pub fn default_on_encoded_frame(
    ) -> impl Fn(&HookCtx, &Router<EncodedMsg>, VideoFrame) + Send + Sync + 'static {
        |_ctx, router, frame| {
            let _ = router.send(EncodedMsg::frame(frame, None));
        }
    }

    /// Default `on_encoded_source_eos` hook — forwards as
    /// `EncodedMsg::SourceEos { source_id }`.
    pub fn default_on_encoded_source_eos(
    ) -> impl Fn(&HookCtx, &Router<EncodedMsg>, EndOfStream) + Send + Sync + 'static {
        |_ctx, router, eos| {
            let _ = router.send(EncodedMsg::source_eos(eos.source_id));
        }
    }

    /// Default `forward_inbox_source_eos` hook — emits a single
    /// `debug` log line and drops the EOS.  Override to forward
    /// the sentinel to a downstream stage that also tracks the
    /// per-source drain.
    pub fn default_forward_inbox_source_eos(
    ) -> impl FnMut(&mut Context<Picasso>, &Router<PipelineMsg>, &str) -> Result<()>
           + Send
           + 'static {
        |ctx, _router, source_id| {
            log::debug!(
                "[{}] forward_inbox_source_eos: dropping eos for {source_id}",
                ctx.own_name()
            );
            Ok(())
        }
    }

    /// Default user shutdown hook — a no-op.  The stage's own
    /// [`Actor::stopping`] body always runs the guarded
    /// [`PicassoEngine::shutdown`](picasso::prelude::PicassoEngine::shutdown)
    /// before this hook fires, so omitting the bundle setter
    /// simply means "don't add any extra cleanup on top of the
    /// built-in engine shutdown".
    pub fn default_on_stopping() -> impl FnMut(&mut Context<Picasso>) + Send + 'static {
        |_ctx| {}
    }
}

impl Actor for Picasso {
    type Msg = PipelineMsg;

    fn handle(&mut self, msg: PipelineMsg, ctx: &mut Context<Self>) -> Result<Flow> {
        msg.dispatch(self, ctx)
    }

    fn poll_timeout(&self) -> Duration {
        self.poll_timeout
    }

    fn started(&mut self, ctx: &mut Context<Self>) -> Result<()> {
        log::info!("[{}] picasso started", ctx.own_name());
        Ok(())
    }

    fn stopping(&mut self, ctx: &mut Context<Self>) {
        if !self.shutdown_done {
            self.engine.shutdown();
            self.shutdown_done = true;
        }
        log::info!("[{}] picasso stopping", ctx.own_name());
        (self.on_stopping)(ctx);
    }
}

impl Handler<SingleDelivery> for Picasso {
    fn handle(&mut self, msg: SingleDelivery, ctx: &mut Context<Self>) -> Result<Flow> {
        let pairs = PipelineMsg::delivery(msg.0).into_pairs();
        self.send_pairs(pairs, ctx)
    }
}

impl Handler<BatchDelivery> for Picasso {
    fn handle(&mut self, msg: BatchDelivery, ctx: &mut Context<Self>) -> Result<Flow> {
        let pairs = msg.0.unseal();
        self.send_pairs(pairs, ctx)
    }
}

impl Handler<SourceEosPayload> for Picasso {
    fn handle(&mut self, msg: SourceEosPayload, ctx: &mut Context<Self>) -> Result<Flow> {
        log::info!(
            "[{}] SourceEos {}: flushing picasso",
            ctx.own_name(),
            msg.source_id
        );
        if let Err(e) = self.engine.send_eos(&msg.source_id) {
            log::warn!("[{}] send_eos({}): {e}", ctx.own_name(), msg.source_id);
        }
        // No frame on a SourceEos — no span.
        if let Err(e) = (self.forward_inbox_source_eos)(ctx, &self.forward_inbox, &msg.source_id) {
            log::warn!(
                "[{}] forward_inbox_source_eos({}) failed: {e}",
                ctx.own_name(),
                msg.source_id
            );
        }
        Ok(Flow::Cont)
    }
}

impl Handler<ShutdownPayload> for Picasso {}

impl Handler<RemoveSourcePayload> for Picasso {
    /// Tear down the Picasso worker for `source_id` via
    /// [`PicassoEngine::remove_source_spec`].  The local
    /// `registered` set is also cleared so the next delivery for the
    /// same `source_id` re-triggers the `source_spec_factory`.
    fn handle(&mut self, msg: RemoveSourcePayload, ctx: &mut Context<Self>) -> Result<Flow> {
        log::info!(
            "[{}] remove_source_spec({}) dispatched",
            ctx.own_name(),
            msg.source_id
        );
        self.engine.remove_source_spec(&msg.source_id);
        self.registered.remove(&msg.source_id);
        Ok(Flow::Cont)
    }
}

impl Handler<UpdateSourceSpecPayload> for Picasso {
    /// Install (or hot-swap) the [`SourceSpec`] bound to
    /// `source_id` via [`PicassoEngine::set_source_spec`].  Also
    /// marks the `source_id` as registered so the next delivery
    /// does not re-run the `source_spec_factory`.
    fn handle(&mut self, msg: UpdateSourceSpecPayload, ctx: &mut Context<Self>) -> Result<Flow> {
        log::info!(
            "[{}] set_source_spec({}) dispatched",
            ctx.own_name(),
            msg.source_id
        );
        let UpdateSourceSpecPayload { source_id, spec } = msg;
        self.engine
            .set_source_spec(&source_id, *spec)
            .map_err(|e| anyhow!("set_source_spec({source_id}): {e}"))?;
        self.registered.insert(source_id);
        Ok(Flow::Cont)
    }
}

impl Picasso {
    fn send_pairs(
        &mut self,
        pairs: Vec<(VideoFrame, SharedBuffer)>,
        ctx: &mut Context<Self>,
    ) -> Result<Flow> {
        // Register any previously-unseen source ids first so the
        // user hook observes a fully-registered engine state.
        for (frame, _buffer) in &pairs {
            let sid = frame.get_source_id();
            if self.registered.insert(sid.clone()) {
                let w = frame.get_width().max(1) as u32;
                let h = frame.get_height().max(1) as u32;
                let (fps_num, fps_den) = frame.get_fps();
                log::info!(
                    "[{}] registering source_id={sid} {w}x{h} fps={fps_num}/{fps_den}",
                    ctx.own_name()
                );
                let spec = (self.source_spec)(&sid, w, h, fps_num as i32, fps_den as i32)?;
                self.engine
                    .set_source_spec(&sid, spec)
                    .map_err(|e| anyhow!("set_source_spec({sid}): {e}"))?;
            }
        }

        // Inbox-forward tee — fires before the engine feed loop so
        // a downstream stage that also wants the batch is not held
        // behind picasso's rendering.  Errors are logged but
        // non-fatal: a failed forward must not drop the frame.
        // The hook receives all pairs at once; per-frame nesting
        // happens through the per-pair `on_delivery` span below.
        if let Err(e) = (self.forward_inbox_delivery)(ctx, &self.forward_inbox, &pairs) {
            log::warn!("[{}] forward_inbox_delivery failed: {e}", ctx.own_name());
        }

        for (frame, buffer) in pairs {
            // User per-pair hook — fires before the engine feed.
            // Errors are logged but non-fatal so a failed observer
            // does not drop the frame by default.
            let pipeline_name = ctx.pipeline_name().to_string();
            let stage_name = ctx.own_name().to_string();
            {
                let _g = enter_callback_span(
                    &frame,
                    "on_delivery",
                    &pipeline_name,
                    &stage_name,
                );
                if let Err(e) = (self.on_delivery)(ctx, &frame, &buffer) {
                    log::warn!("[{}] on_delivery failed: {e}", ctx.own_name());
                }
            }
            let sid = frame.get_source_id();
            let view = SurfaceView::from_buffer(&buffer, 0)
                .map_err(|e| anyhow!("SurfaceView::from_buffer: {e}"))?;
            let crop = {
                let _g = enter_callback_span(
                    &frame,
                    "on_crop_select",
                    &pipeline_name,
                    &stage_name,
                );
                (self.on_crop_select)(ctx, &frame)
            };
            if let Err(e) = self.engine.send_frame(&sid, frame, view, crop) {
                log::error!("[{}] send_frame failed: {e}", ctx.own_name());
                return Err(anyhow!("picasso send_frame: {e}"));
            }
        }
        Ok(Flow::Cont)
    }
}

/// Inbox / per-frame hook bundle for [`Picasso`].  Built through
/// [`PicassoInbox::builder`] and handed to [`PicassoBuilder::inbox`].
/// Omitted branches auto-install the matching `Picasso::default_*`
/// at build time.
pub struct PicassoInbox {
    on_delivery: OnDeliveryHook,
    on_crop_select: OnCropSelectHook,
}

impl PicassoInbox {
    /// Start a builder that auto-installs every default on
    /// [`PicassoInboxBuilder::build`].
    pub fn builder() -> PicassoInboxBuilder {
        PicassoInboxBuilder::new()
    }
}

impl Default for PicassoInbox {
    fn default() -> Self {
        PicassoInboxBuilder::new().build()
    }
}

/// Fluent builder for [`PicassoInbox`].
pub struct PicassoInboxBuilder {
    on_delivery: Option<OnDeliveryHook>,
    on_crop_select: Option<OnCropSelectHook>,
}

impl PicassoInboxBuilder {
    /// Empty bundle — every hook defaults to its matching
    /// `Picasso::default_*` at [`build`](Self::build) time.
    pub fn new() -> Self {
        Self {
            on_delivery: None,
            on_crop_select: None,
        }
    }

    /// Install a per-pair delivery hook.  The hook receives a
    /// single unsealed `(VideoFrame, SharedBuffer)` pair by
    /// **shared reference** before that pair is fed to the engine
    /// — when a batched delivery arrives, this hook is called once
    /// per pair in the batch.  The body can clone or route the
    /// pair into nested helpers (overlays, analytics, …) without
    /// consuming.  Returning `Err` is logged but does not abort
    /// the loop; see [`Picasso::send_pairs`](Picasso).
    pub fn on_delivery<F>(mut self, f: F) -> Self
    where
        F: FnMut(&mut Context<Picasso>, &VideoFrame, &SharedBuffer) -> Result<()>
            + Send
            + 'static,
    {
        self.on_delivery = Some(Box::new(f));
        self
    }

    /// Install a per-frame crop-rectangle selector.  Returning
    /// `Some(rect)` feeds the rectangle to
    /// [`PicassoEngine::send_frame`]; returning `None` leaves the
    /// engine's default behaviour (use the whole source surface).
    /// The frame's `source_id` is available via
    /// [`VideoFrame::get_source_id`] when needed.
    pub fn on_crop_select<F>(mut self, f: F) -> Self
    where
        F: FnMut(&mut Context<Picasso>, &VideoFrame) -> Option<Rect> + Send + 'static,
    {
        self.on_crop_select = Some(Box::new(f));
        self
    }

    /// Finalise the bundle, substituting defaults for every omitted
    /// setter.
    pub fn build(self) -> PicassoInbox {
        let PicassoInboxBuilder {
            on_delivery,
            on_crop_select,
        } = self;
        PicassoInbox {
            on_delivery: on_delivery.unwrap_or_else(|| Box::new(Picasso::default_on_delivery())),
            on_crop_select: on_crop_select
                .unwrap_or_else(|| Box::new(Picasso::default_on_crop_select())),
        }
    }
}

impl Default for PicassoInboxBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Result-path hook bundle for [`Picasso`] — the four routed
/// callbacks the stage owns.  Built through
/// [`PicassoResults::builder`] and handed to
/// [`PicassoBuilder::results`].  Each hook receives the matching
/// [`Router<_>`] so its body owns the drop/send decision; defaults
/// substitute the typical-case behaviour at build time.
///
/// Encoded callbacks (`on_encoded_frame`, `on_encoded_source_eos`)
/// fire from the engine worker thread and are `Fn + Send + Sync`
/// (no `Context`).  Inbox-forward callbacks
/// (`forward_inbox_delivery`, `forward_inbox_source_eos`) fire from
/// the actor receive loop and take `&mut Context<Picasso>`.
pub struct PicassoResults {
    on_encoded_frame: OnEncodedFrameHook,
    on_encoded_source_eos: OnEncodedSourceEosHook,
    forward_inbox_delivery: OnForwardInboxDeliveryHook,
    forward_inbox_source_eos: OnForwardInboxSourceEosHook,
}

impl PicassoResults {
    /// Start a builder that auto-installs every default on
    /// [`PicassoResultsBuilder::build`].
    pub fn builder() -> PicassoResultsBuilder {
        PicassoResultsBuilder::new()
    }
}

impl Default for PicassoResults {
    fn default() -> Self {
        PicassoResultsBuilder::new().build()
    }
}

/// Fluent builder for [`PicassoResults`].
pub struct PicassoResultsBuilder {
    on_encoded_frame: Option<OnEncodedFrameHook>,
    on_encoded_source_eos: Option<OnEncodedSourceEosHook>,
    forward_inbox_delivery: Option<OnForwardInboxDeliveryHook>,
    forward_inbox_source_eos: Option<OnForwardInboxSourceEosHook>,
}

impl PicassoResultsBuilder {
    /// Empty bundle — every hook defaults to its matching
    /// `Picasso::default_*` at [`build`](Self::build) time.
    pub fn new() -> Self {
        Self {
            on_encoded_frame: None,
            on_encoded_source_eos: None,
            forward_inbox_delivery: None,
            forward_inbox_source_eos: None,
        }
    }

    /// Install an encoded-frame result hook.  Receives the
    /// [`Router<EncodedMsg>`] (wired from [`PicassoBuilder::downstream`])
    /// and the encoded [`VideoFrame`] from the engine worker thread.
    /// Default: forward as `EncodedMsg::Frame { frame, payload: None }`.
    pub fn on_encoded_frame<F>(mut self, f: F) -> Self
    where
        F: Fn(&HookCtx, &Router<EncodedMsg>, VideoFrame) + Send + Sync + 'static,
    {
        self.on_encoded_frame = Some(Arc::new(f));
        self
    }

    /// Install an encoded source-EOS result hook.  Default: forward
    /// as `EncodedMsg::SourceEos { source_id }`.
    pub fn on_encoded_source_eos<F>(mut self, f: F) -> Self
    where
        F: Fn(&HookCtx, &Router<EncodedMsg>, EndOfStream) + Send + Sync + 'static,
    {
        self.on_encoded_source_eos = Some(Arc::new(f));
        self
    }

    /// Install a delivery-batch inbox-forward hook.  Fires once per
    /// incoming batch (after [`PicassoInbox::on_delivery`] and before
    /// the engine feed loop) with a [`Router<PipelineMsg>`] supplied
    /// by the stage; typical bodies clone the unsealed pairs, mint a
    /// fresh already-released
    /// [`ReleaseSeal`](savant_core::utils::release_seal::ReleaseSeal),
    /// wrap them in
    /// [`SealedDeliveries`](deepstream_buffers::SealedDeliveries),
    /// and call `router.send(PipelineMsg::Deliveries(sealed))`.
    /// Default: debug-log + drop.
    pub fn forward_inbox_delivery<F>(mut self, f: F) -> Self
    where
        F: FnMut(
                &mut Context<Picasso>,
                &Router<PipelineMsg>,
                &[(VideoFrame, SharedBuffer)],
            ) -> Result<()>
            + Send
            + 'static,
    {
        self.forward_inbox_delivery = Some(Box::new(f));
        self
    }

    /// Install a source-EOS inbox-forward hook.  Fires for every
    /// [`PipelineMsg::SourceEos`] *after* picasso has flushed the
    /// matching engine source, with a [`Router<PipelineMsg>`]
    /// supplied by the stage so the body can
    /// `router.send(PipelineMsg::SourceEos { source_id })` to relay
    /// the sentinel.  Default: debug-log + drop.
    pub fn forward_inbox_source_eos<F>(mut self, f: F) -> Self
    where
        F: FnMut(&mut Context<Picasso>, &Router<PipelineMsg>, &str) -> Result<()>
            + Send
            + 'static,
    {
        self.forward_inbox_source_eos = Some(Box::new(f));
        self
    }

    /// Finalise the bundle, substituting defaults for every omitted
    /// setter.
    pub fn build(self) -> PicassoResults {
        let PicassoResultsBuilder {
            on_encoded_frame,
            on_encoded_source_eos,
            forward_inbox_delivery,
            forward_inbox_source_eos,
        } = self;
        PicassoResults {
            on_encoded_frame: on_encoded_frame
                .unwrap_or_else(|| Arc::new(Picasso::default_on_encoded_frame())),
            on_encoded_source_eos: on_encoded_source_eos
                .unwrap_or_else(|| Arc::new(Picasso::default_on_encoded_source_eos())),
            forward_inbox_delivery: forward_inbox_delivery
                .unwrap_or_else(|| Box::new(Picasso::default_forward_inbox_delivery())),
            forward_inbox_source_eos: forward_inbox_source_eos
                .unwrap_or_else(|| Box::new(Picasso::default_forward_inbox_source_eos())),
        }
    }
}

impl Default for PicassoResultsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Loop-level common knobs + user shutdown hook for [`Picasso`].
/// Built through [`PicassoCommon::builder`] and handed to
/// [`PicassoBuilder::common`].
pub struct PicassoCommon {
    poll_timeout: Duration,
    on_stopping: OnStoppingHook,
}

impl PicassoCommon {
    /// Start a builder seeded with [`DEFAULT_POLL_TIMEOUT`] and the
    /// no-op stopping hook.
    pub fn builder() -> PicassoCommonBuilder {
        PicassoCommonBuilder::new()
    }
}

impl Default for PicassoCommon {
    fn default() -> Self {
        PicassoCommonBuilder::new().build()
    }
}

/// Fluent builder for [`PicassoCommon`].
pub struct PicassoCommonBuilder {
    poll_timeout: Option<Duration>,
    on_stopping: Option<OnStoppingHook>,
}

impl PicassoCommonBuilder {
    /// Empty bundle — `poll_timeout` defaults to
    /// [`DEFAULT_POLL_TIMEOUT`] and `stopping` to a no-op.
    pub fn new() -> Self {
        Self {
            poll_timeout: None,
            on_stopping: None,
        }
    }

    /// Inbox receive-poll cadence (default [`DEFAULT_POLL_TIMEOUT`]).
    pub fn poll_timeout(mut self, d: Duration) -> Self {
        self.poll_timeout = Some(d);
        self
    }

    /// Override the user shutdown hook — fired from
    /// [`Actor::stopping`] **after** the stage's built-in
    /// cleanup (guarded
    /// [`PicassoEngine::shutdown`](picasso::prelude::PicassoEngine::shutdown))
    /// has completed.  The built-in cleanup is **load-bearing** and
    /// always runs first; it cannot be skipped through this hook.
    pub fn on_stopping<F>(mut self, f: F) -> Self
    where
        F: FnMut(&mut Context<Picasso>) + Send + 'static,
    {
        self.on_stopping = Some(Box::new(f));
        self
    }

    /// Finalise the bundle, substituting defaults for every omitted
    /// setter.
    pub fn build(self) -> PicassoCommon {
        let PicassoCommonBuilder {
            poll_timeout,
            on_stopping,
        } = self;
        PicassoCommon {
            poll_timeout: poll_timeout.unwrap_or(DEFAULT_POLL_TIMEOUT),
            on_stopping: on_stopping
                .unwrap_or_else(|| Box::new(Picasso::default_on_stopping())),
        }
    }
}

impl Default for PicassoCommonBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Fluent builder for [`Picasso`].
///
/// Wiring-level configuration (`downstream`, `engine_factory`,
/// `source_spec_factory`) stays on the top-level builder.  Inbox
/// hooks live on [`PicassoInbox`]; loop-level knobs and the user
/// shutdown hook live on [`PicassoCommon`].  Install them via
/// [`PicassoBuilder::inbox`] / [`PicassoBuilder::common`].
pub struct PicassoBuilder {
    name: StageName,
    capacity: usize,
    downstream: Option<StageName>,
    forward_inbox_downstream: Option<StageName>,
    engine_factory: Option<PicassoEngineFactory>,
    source_spec: Option<SourceSpecFactory>,
    engine_handle: Option<PicassoEngineHandle>,
    on_render: Option<OnRenderHook>,
    on_gpu_mat: Option<OnGpuMatHook>,
    on_object_draw_spec: Option<OnObjectDrawSpecHook>,
    on_eviction: Option<OnEvictionHook>,
    on_stream_reset: Option<OnStreamResetHook>,
    inbox: Option<PicassoInbox>,
    results: Option<PicassoResults>,
    common: Option<PicassoCommon>,
}

impl PicassoBuilder {
    /// Start a builder with framework poll defaults.
    pub fn new(name: StageName, capacity: usize) -> Self {
        Self {
            name,
            capacity,
            downstream: None,
            forward_inbox_downstream: None,
            engine_factory: None,
            source_spec: None,
            engine_handle: None,
            on_render: None,
            on_gpu_mat: None,
            on_object_draw_spec: None,
            on_eviction: None,
            on_stream_reset: None,
            inbox: None,
            results: None,
            common: None,
        }
    }

    /// Optional default peer installed on the
    /// [`Router<EncodedMsg>`] handed to the engine factory.  The
    /// user's `OnEncodedFrame` calls `router.send(msg)` to route to
    /// this peer and `router.send_to(&peer, msg)` to address any
    /// other registered actor by name.
    pub fn downstream(mut self, peer: StageName) -> Self {
        self.downstream = Some(peer);
        self
    }

    /// Optional default peer installed on the
    /// [`Router<PipelineMsg>`] handed to the inbox-forward hooks
    /// ([`PicassoInboxBuilder::forward_inbox_delivery`] and
    /// [`PicassoInboxBuilder::forward_inbox_source_eos`]).  Omit when
    /// picasso is the terminal sink for `(VideoFrame, SharedBuffer)`
    /// traffic — the default hooks log-and-drop and never touch the
    /// router.
    pub fn forward_inbox_downstream(mut self, peer: StageName) -> Self {
        self.forward_inbox_downstream = Some(peer);
        self
    }

    /// Required: engine spec factory.  See [`PicassoEngineFactory`].
    /// Returns just the [`GeneralSpec`]; the stage owns `Callbacks`
    /// lifecycle and assembles them from the encoded closures in
    /// [`PicassoResults`] plus any engine-level closures installed
    /// on the top-level builder.
    pub fn engine_factory<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&BuildCtx) -> Result<GeneralSpec> + Send + 'static,
    {
        self.engine_factory = Some(Box::new(f));
        self
    }

    /// Required: source-spec factory.  See [`SourceSpecFactory`].
    pub fn source_spec_factory<F>(mut self, f: F) -> Self
    where
        F: FnMut(&str, u32, u32, i32, i32) -> Result<SourceSpec> + Send + 'static,
    {
        self.source_spec = Some(Box::new(f));
        self
    }

    /// Hand a [`PicassoEngineHandle`] for the stage to bind once
    /// the engine is built.  Cheap to clone — pass clones into
    /// closures that need to call engine methods (`set_source_spec`,
    /// `send_eos`, `shutdown`, …).  Omit when no out-of-band engine
    /// access is required.
    pub fn engine_handle(mut self, handle: PicassoEngineHandle) -> Self {
        self.engine_handle = Some(handle);
        self
    }

    /// Install an [`OnRender`](picasso::prelude::OnRender) callback
    /// (closure form).  Fires from the engine worker thread before
    /// each Skia flush.  No default — omitting leaves the
    /// engine slot empty.
    pub fn on_render<F>(mut self, f: F) -> Self
    where
        F: Fn(&HookCtx, &str, &mut SkiaRenderer, &VideoFrame) + Send + Sync + 'static,
    {
        self.on_render = Some(Arc::new(f));
        self
    }

    /// Install an [`OnGpuMat`](picasso::prelude::OnGpuMat) callback
    /// (closure form).  Fires from the engine worker thread once
    /// the destination [`SurfaceView`] is ready.  No default.
    pub fn on_gpu_mat<F>(mut self, f: F) -> Self
    where
        F: Fn(&HookCtx, &str, &VideoFrame, &SurfaceView) + Send + Sync + 'static,
    {
        self.on_gpu_mat = Some(Arc::new(f));
        self
    }

    /// Install an [`OnObjectDrawSpec`](picasso::prelude::OnObjectDrawSpec)
    /// callback (closure form).  Returning `Some(spec)` overrides
    /// the static draw spec for the object; returning `None` keeps
    /// the resolved default.  No default.
    pub fn on_object_draw_spec<F>(mut self, f: F) -> Self
    where
        F: Fn(&HookCtx, &str, &BorrowedVideoObject, Option<&ObjectDraw>) -> Option<ObjectDraw>
            + Send
            + Sync
            + 'static,
    {
        self.on_object_draw_spec = Some(Arc::new(f));
        self
    }

    /// Install an [`OnEviction`](picasso::prelude::OnEviction)
    /// callback (closure form).  Returns the
    /// [`EvictionDecision`](picasso::prelude::EvictionDecision) for
    /// an idle source.  No default.
    pub fn on_eviction<F>(mut self, f: F) -> Self
    where
        F: Fn(&HookCtx, &str) -> EvictionDecision + Send + Sync + 'static,
    {
        self.on_eviction = Some(Arc::new(f));
        self
    }

    /// Install an [`OnStreamReset`](picasso::prelude::OnStreamReset)
    /// callback (closure form).  Fires when the engine resets its
    /// per-source encoder.  No default.
    pub fn on_stream_reset<F>(mut self, f: F) -> Self
    where
        F: Fn(&HookCtx, &str, StreamResetReason) + Send + Sync + 'static,
    {
        self.on_stream_reset = Some(Arc::new(f));
        self
    }

    /// Install a [`PicassoInbox`] bundle.  Omitting this call is
    /// equivalent to `.inbox(PicassoInbox::default())`, which
    /// wires every slot to the matching `Picasso::default_*`.
    pub fn inbox(mut self, i: PicassoInbox) -> Self {
        self.inbox = Some(i);
        self
    }

    /// Install a [`PicassoResults`] bundle — the inbox-forward tee
    /// (`forward_inbox_delivery`, `forward_inbox_source_eos`).
    /// Omitting this call is equivalent to
    /// `.results(PicassoResults::default())`, which wires both slots
    /// to the matching `Picasso::default_forward_inbox_*` (debug-log
    /// + drop).
    pub fn results(mut self, r: PicassoResults) -> Self {
        self.results = Some(r);
        self
    }

    /// Install a [`PicassoCommon`] bundle.  Omitting this call is
    /// equivalent to `.common(PicassoCommon::default())`.
    pub fn common(mut self, c: PicassoCommon) -> Self {
        self.common = Some(c);
        self
    }

    /// Finalise the stage.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `source_spec_factory` or `engine_factory` is
    /// missing.  `downstream` is optional — callers that route
    /// exclusively via [`Router::send_to`] may omit it.
    pub fn build(self) -> Result<ActorBuilder<Picasso>> {
        let PicassoBuilder {
            name,
            capacity,
            downstream,
            forward_inbox_downstream,
            engine_factory,
            source_spec,
            engine_handle,
            on_render,
            on_gpu_mat,
            on_object_draw_spec,
            on_eviction,
            on_stream_reset,
            inbox,
            results,
            common,
        } = self;
        let source_spec =
            source_spec.ok_or_else(|| anyhow!("Picasso: missing source_spec_factory"))?;
        let engine_factory =
            engine_factory.ok_or_else(|| anyhow!("Picasso: missing engine_factory"))?;
        let PicassoInbox {
            on_delivery,
            on_crop_select,
        } = inbox.unwrap_or_default();
        let PicassoResults {
            on_encoded_frame,
            on_encoded_source_eos,
            forward_inbox_delivery,
            forward_inbox_source_eos,
        } = results.unwrap_or_default();
        let PicassoCommon {
            poll_timeout,
            on_stopping,
        } = common.unwrap_or_default();
        Ok(ActorBuilder::new(name, capacity)
            .poll_timeout(poll_timeout)
            .factory(move |bx| {
                let router: Router<EncodedMsg> = bx.router(downstream.as_ref())?;
                let forward_inbox: Router<PipelineMsg> =
                    bx.router(forward_inbox_downstream.as_ref())?;
                let spec = (engine_factory)(bx)?;
                let ctx = bx.hook_ctx();

                // Stage owns the OnEncodedFrame impl now — it
                // delegates to the user closures stored in
                // PicassoResults.  The encoded router and a HookCtx
                // clone are moved into the sink so the worker
                // thread doesn't need to touch any actor-thread
                // state.
                let encoded_sink = ClosureOnEncodedFrame {
                    on_frame: on_encoded_frame,
                    on_eos: on_encoded_source_eos,
                    router,
                    ctx: ctx.clone(),
                };
                let mut cb = Callbacks::builder().on_encoded_frame(encoded_sink);
                if let Some(h) = on_render {
                    cb = cb.on_render(ClosureOnRender {
                        hook: h,
                        ctx: ctx.clone(),
                    });
                }
                if let Some(h) = on_gpu_mat {
                    cb = cb.on_gpumat(ClosureOnGpuMat {
                        hook: h,
                        ctx: ctx.clone(),
                    });
                }
                if let Some(h) = on_object_draw_spec {
                    cb = cb.on_object_draw_spec(ClosureOnObjectDrawSpec {
                        hook: h,
                        ctx: ctx.clone(),
                    });
                }
                if let Some(h) = on_eviction {
                    cb = cb.on_eviction(ClosureOnEviction {
                        hook: h,
                        ctx: ctx.clone(),
                    });
                }
                if let Some(h) = on_stream_reset {
                    cb = cb.on_stream_reset(ClosureOnStreamReset {
                        hook: h,
                        ctx: ctx.clone(),
                    });
                }
                let engine = Arc::new(PicassoEngine::new(spec, cb.build()));

                if let Some(handle) = engine_handle.as_ref() {
                    handle.bind(engine.clone());
                }

                Ok(Picasso {
                    engine,
                    source_spec,
                    on_delivery,
                    on_crop_select,
                    forward_inbox,
                    forward_inbox_delivery,
                    forward_inbox_source_eos,
                    registered: HashSet::new(),
                    poll_timeout,
                    on_stopping,
                    shutdown_done: false,
                })
            }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::supervisor::{StageKind, StageName};
    use picasso::prelude::GeneralSpec;

    fn dummy_spec(_sid: &str, _w: u32, _h: u32, _fn_: i32, _fd: i32) -> Result<SourceSpec> {
        Ok(SourceSpec::default())
    }

    /// Builds a minimal [`GeneralSpec`].  The stage now owns
    /// `Callbacks` lifecycle, so the factory only returns the spec.
    fn make_engine_factory() -> PicassoEngineFactory {
        Box::new(|_bx| Ok(GeneralSpec::builder().name("demo").build()))
    }

    #[test]
    fn builder_requires_source_spec_and_engine_factory() {
        let name = StageName::unnamed(StageKind::Render);
        let err = Picasso::builder(name.clone(), 4).build().err().unwrap();
        assert!(err.to_string().contains("missing source_spec_factory"));

        let err = Picasso::builder(name, 4)
            .source_spec_factory(dummy_spec)
            .build()
            .err()
            .unwrap();
        assert!(err.to_string().contains("missing engine_factory"));
    }

    /// `downstream` is optional — the builder accepts the
    /// no-downstream configuration.  The resulting actor's router
    /// logs once the first time the user's `OnEncodedFrame` tries
    /// to forward via the default peer and silently drops subsequent
    /// default-peer sends.
    #[test]
    fn builder_without_downstream_is_accepted() {
        let name = StageName::unnamed(StageKind::Render);
        let _ = Picasso::builder(name, 4)
            .source_spec_factory(dummy_spec)
            .engine_factory(make_engine_factory())
            .build()
            .unwrap();
    }

    #[test]
    fn picasso_implements_source_lifecycle_handlers() {
        fn assert_remove<H: Handler<RemoveSourcePayload>>() {}
        fn assert_update<H: Handler<UpdateSourceSpecPayload>>() {}
        assert_remove::<Picasso>();
        assert_update::<Picasso>();

        let remove = RemoveSourcePayload {
            source_id: "cam-0".to_string(),
        };
        assert_eq!(remove.source_id, "cam-0");

        let update = UpdateSourceSpecPayload {
            source_id: "cam-1".to_string(),
            spec: Box::new(SourceSpec::default()),
        };
        assert_eq!(update.source_id, "cam-1");
    }

    #[test]
    fn builder_accepts_full_config() {
        let name = StageName::unnamed(StageKind::Render);
        let _ = Picasso::builder(name, 4)
            .downstream(StageName::unnamed(StageKind::BitstreamSink))
            .engine_factory(make_engine_factory())
            .source_spec_factory(dummy_spec)
            .inbox(
                PicassoInbox::builder()
                    .on_crop_select(|_ctx, _frame| None)
                    .on_delivery(|_ctx, _frame, _buffer| Ok(()))
                    .build(),
            )
            .common(
                PicassoCommon::builder()
                    .poll_timeout(Duration::from_millis(50))
                    .on_stopping(Picasso::default_on_stopping())
                    .build(),
            )
            .build()
            .unwrap();
    }

    /// The [`PicassoCommon`] bundle accepts a user-supplied
    /// `.stopping(F)` closure.  Compile-only verification of the
    /// hook bound; integration coverage should exercise the ordering
    /// invariant that built-in `engine.shutdown()` runs before the
    /// user hook.
    #[test]
    fn builder_accepts_user_stopping() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let flag = Arc::new(AtomicBool::new(false));
        let flag_hook = flag.clone();
        let name = StageName::unnamed(StageKind::Render);
        let _ = Picasso::builder(name, 4)
            .engine_factory(make_engine_factory())
            .source_spec_factory(dummy_spec)
            .common(
                PicassoCommon::builder()
                    .on_stopping(move |_ctx| {
                        flag_hook.store(true, Ordering::SeqCst);
                    })
                    .build(),
            )
            .build()
            .unwrap();
        assert!(!flag.load(Ordering::SeqCst));
    }

    /// Runtime invariant: the `on_delivery` and `on_crop_select`
    /// slots are no longer `Option<_>`.  A minimal builder without
    /// any `.inbox(...)` call still produces a runtime [`Picasso`]
    /// whose `on_delivery` and `on_crop_select` are populated with
    /// the matching no-op defaults.
    #[test]
    fn runtime_invariant_all_hooks_populated() {
        use crate::context::BuildCtx;
        use crate::registry::Registry;
        use crate::shared::SharedStore;

        let name = StageName::unnamed(StageKind::Render);
        let ab = Picasso::builder(name.clone(), 4)
            .engine_factory(make_engine_factory())
            .source_spec_factory(dummy_spec)
            .build()
            .unwrap();
        let parts = ab.into_parts();
        let reg = Arc::new(Registry::new());
        let shared = Arc::new(SharedStore::new());
        let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let pn: Arc<str> = Arc::from("test");
        let sm = crate::stage_metrics::StageMetrics::new(parts.name.to_string());
        let bx = BuildCtx::new(&parts.name, &pn, &reg, &shared, &stop_flag, &sm);
        let pic = (parts.factory)(&bx).expect("factory resolves");
        let Picasso {
            on_delivery: _,
            on_crop_select: _,
            forward_inbox: _,
            forward_inbox_delivery: _,
            forward_inbox_source_eos: _,
            on_stopping: _,
            ..
        } = pic;
    }

    /// Builder accepts user-supplied inbox-forward hooks together
    /// with an explicit `forward_inbox_downstream` peer.  Compile-only
    /// check that the closure shapes match the published types and
    /// that the [`PicassoResults`] bundle plumbs through cleanly.
    #[test]
    fn builder_accepts_forward_inbox_hooks() {
        let name = StageName::unnamed(StageKind::Render);
        let _ = Picasso::builder(name, 4)
            .downstream(StageName::unnamed(StageKind::BitstreamSink))
            .forward_inbox_downstream(StageName::unnamed(StageKind::Custom("inbox-tee")))
            .engine_factory(make_engine_factory())
            .source_spec_factory(dummy_spec)
            .results(
                PicassoResults::builder()
                    .forward_inbox_delivery(|_ctx, _router, pairs| {
                        let _ = pairs.len();
                        Ok(())
                    })
                    .forward_inbox_source_eos(|_ctx, _router, _sid| Ok(()))
                    .build(),
            )
            .build()
            .unwrap();
    }

    /// Builder accepts user-supplied encoded result hooks alongside
    /// the inbox-forward ones.  Compile-only verification of the
    /// closure shapes; runtime behaviour is covered by integration
    /// tests against the picasso engine.
    #[test]
    fn builder_accepts_encoded_result_hooks() {
        let name = StageName::unnamed(StageKind::Render);
        let _ = Picasso::builder(name, 4)
            .downstream(StageName::unnamed(StageKind::BitstreamSink))
            .engine_factory(make_engine_factory())
            .source_spec_factory(dummy_spec)
            .results(
                PicassoResults::builder()
                    .on_encoded_frame(|_ctx, router, frame| {
                        let _ = router.send(EncodedMsg::frame(frame, None));
                    })
                    .on_encoded_source_eos(|_ctx, router, eos| {
                        let _ = router.send(EncodedMsg::source_eos(eos.source_id));
                    })
                    .build(),
            )
            .build()
            .unwrap();
    }

    /// Builder accepts the five top-level engine-level callbacks as
    /// closures.  No defaults — omitted callbacks leave their slot
    /// on the engine empty.
    #[test]
    fn builder_accepts_engine_level_callbacks() {
        use picasso::prelude::EvictionDecision;
        let name = StageName::unnamed(StageKind::Render);
        let _ = Picasso::builder(name, 4)
            .engine_factory(make_engine_factory())
            .source_spec_factory(dummy_spec)
            .on_eviction(|_ctx, _sid| EvictionDecision::KeepFor(60))
            .on_object_draw_spec(|_ctx, _sid, _obj, _current| None)
            .on_stream_reset(|_ctx, _sid, _reason| {})
            // on_render and on_gpu_mat reference the rendering /
            // GPU-buffer surfaces; closures still type-check even
            // without exercising them.
            .build()
            .unwrap();
    }

    /// `PicassoEngineHandle` starts unbound and becomes bound the
    /// moment the actor's factory runs.  Cloning shares the bind
    /// state across holders.
    #[test]
    fn engine_handle_binds_after_factory_runs() {
        use crate::context::BuildCtx;
        use crate::registry::Registry;
        use crate::shared::SharedStore;

        let handle = PicassoEngineHandle::new();
        let observer = handle.clone();
        assert!(observer.try_get().is_none());

        let name = StageName::unnamed(StageKind::Render);
        let ab = Picasso::builder(name.clone(), 4)
            .engine_factory(make_engine_factory())
            .source_spec_factory(dummy_spec)
            .engine_handle(handle)
            .build()
            .unwrap();
        let parts = ab.into_parts();
        let reg = Arc::new(Registry::new());
        let shared = Arc::new(SharedStore::new());
        let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let pn: Arc<str> = Arc::from("test");
        let sm = crate::stage_metrics::StageMetrics::new(parts.name.to_string());
        let bx = BuildCtx::new(&parts.name, &pn, &reg, &shared, &stop_flag, &sm);
        let _pic = (parts.factory)(&bx).expect("factory resolves");

        // Same handle, observed through a clone.
        assert!(
            observer.try_get().is_some(),
            "engine_handle must be bound after the factory runs"
        );
    }
}
