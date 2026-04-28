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
//! * Optional batch-level delivery hook (`on_delivery`) — receives
//!   the unsealed `(VideoFrameProxy, SharedBuffer)` batch by
//!   **shared slice reference** before frames are fed to the
//!   engine.  Same signature shape as
//!   [`FunctionInboxBuilder::on_delivery`](super::function::FunctionInboxBuilder::on_delivery),
//!   so applications can mutate frames, attach overlay objects, or
//!   update per-source counters before the batch reaches Picasso.
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
//! Hooks are grouped into two bundles following the cross-stage
//! pattern:
//!
//! * [`PicassoInbox`] — batch-level / per-frame hooks
//!   (`on_delivery`, `src_rect_for`).
//! * [`PicassoCommon`] — loop-level knobs (`poll_timeout`) plus
//!   the user `stopping` hook.
//!
//! ```ignore
//! Picasso::builder(name, 16)
//!     .downstream(peer)
//!     .engine_factory(make_engine)
//!     .source_spec_factory(make_spec)
//!     .inbox(
//!         PicassoInbox::builder()
//!             .on_delivery(|pairs, ctx| { /* ... */ Ok(()) })
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

use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use deepstream_buffers::{Rect, SharedBuffer, SurfaceView};
use hashbrown::HashSet;
use picasso::prelude::{PicassoEngine, SourceSpec};
use savant_core::primitives::frame::VideoFrameProxy;

use crate::envelopes::{BatchDelivery, EncodedMsg, PipelineMsg, SingleDelivery};
use crate::router::Router;
use crate::supervisor::StageName;
use crate::{
    Actor, ActorBuilder, BuildCtx, Context, Dispatch, Flow, Handler, RemoveSourcePayload,
    ShutdownPayload, SourceEosPayload, UpdateSourceSpecPayload,
};

/// Default inbox receive-poll cadence.
pub const DEFAULT_POLL_TIMEOUT: Duration = Duration::from_millis(100);

/// Source-spec factory — builds a
/// [`SourceSpec`] for a
/// previously-unseen `source_id` the first time the picasso stage
/// sees a frame for it.  Invoked exactly once per `source_id` per
/// stage instance.
pub type SourceSpecFactory =
    Box<dyn FnMut(&str, u32, u32, i32, i32) -> Result<SourceSpec> + Send + 'static>;

/// Batch-level delivery hook — called once per incoming
/// `(VideoFrameProxy, SharedBuffer)` batch **before** the frames
/// are fed to
/// [`PicassoEngine::send_frame`](picasso::prelude::PicassoEngine::send_frame).
///
/// The batch is passed by **shared slice reference** so the hook
/// body can iterate, inspect, and forward frames to nested
/// helpers (overlays, analytics, inference taps, …) without
/// consuming the batch — the stage still owns the underlying
/// `Vec` and drives the per-frame send loop after this hook
/// returns.
///
/// Always installed at runtime; the default is a no-op.  Matches
/// the shape of
/// [`FunctionInboxBuilder::on_delivery`](super::function::FunctionInboxBuilder::on_delivery)
/// so the cross-stage hook vocabulary is uniform.
pub type DeliveryHook = Box<
    dyn FnMut(&[(VideoFrameProxy, SharedBuffer)], &mut Context<Picasso>) -> Result<()>
        + Send
        + 'static,
>;

/// Engine factory — invoked once when
/// [`System::run`](super::super::system::System::run) builds the
/// [`Picasso`] actor.  Receives the already-wired
/// [`Router<EncodedMsg>`] so the user can construct whatever
/// [`OnEncodedFrame`](picasso::prelude::OnEncodedFrame) impl
/// they want (typically one that calls `router.send(...)` for
/// `OutputMessage::VideoFrame` and `OutputMessage::EndOfStream`).
/// Typical implementation:
///
/// ```ignore
/// struct MySink { router: Router<EncodedMsg> }
/// impl OnEncodedFrame for MySink {
///     fn call(&self, output: OutputMessage) {
///         match output {
///             OutputMessage::VideoFrame(frame) => {
///                 let _ = self.router.send(EncodedMsg::Frame { frame, payload: None });
///             }
///             OutputMessage::EndOfStream(eos) => {
///                 let _ = self.router.send(EncodedMsg::SourceEos { source_id: eos.source_id });
///             }
///         }
///     }
/// }
///
/// .engine_factory(|_bx, router| {
///     let callbacks = Callbacks::builder()
///         .on_encoded_frame(MySink { router })
///         .build();
///     let general = GeneralSpec::builder().name("demo").build();
///     Ok(Arc::new(PicassoEngine::new(general, callbacks)))
/// })
/// ```
pub type PicassoEngineFactory =
    Box<dyn FnOnce(&BuildCtx, Router<EncodedMsg>) -> Result<Arc<PicassoEngine>> + Send>;

/// Per-frame `src_rect` provider.  Returning `Some(rect)` forwards
/// the rectangle to
/// [`PicassoEngine::send_frame`](picasso::prelude::PicassoEngine::send_frame);
/// returning `None` leaves the existing `None` default in place
/// (use the whole source surface).  Always installed at runtime;
/// the default returns `None` for every frame.
pub type SrcRectProvider =
    Arc<dyn Fn(&str, &VideoFrameProxy) -> Option<Rect> + Send + Sync + 'static>;

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
    on_delivery: DeliveryHook,
    src_rect: SrcRectProvider,
    registered: HashSet<String>,
    poll_timeout: Duration,
    stopping: OnStoppingHook,
    shutdown_done: bool,
}

impl Picasso {
    /// Start a fluent builder for a picasso actor registered under
    /// `name` with inbox capacity `capacity`.
    pub fn builder(name: StageName, capacity: usize) -> PicassoBuilder {
        PicassoBuilder::new(name, capacity)
    }

    /// Default batch-level delivery hook — a no-op.  Always
    /// installed when the user omits the inbox setter so the
    /// runtime struct never needs to inspect an `Option`.
    pub fn default_on_delivery(
    ) -> impl FnMut(&[(VideoFrameProxy, SharedBuffer)], &mut Context<Picasso>) -> Result<()>
           + Send
           + 'static {
        |_pairs, _ctx| Ok(())
    }

    /// Default `src_rect` provider — returns `None` for every
    /// frame, so Picasso uses the whole source surface.  Always
    /// installed when the user omits the inbox setter.
    pub fn default_src_rect_for() -> SrcRectProvider {
        Arc::new(|_sid: &str, _frame: &VideoFrameProxy| None)
    }

    /// Default user shutdown hook — a no-op.  The stage's own
    /// [`Actor::stopping`] body always runs the guarded
    /// [`PicassoEngine::shutdown`](picasso::prelude::PicassoEngine::shutdown)
    /// before this hook fires, so omitting the bundle setter
    /// simply means "don't add any extra cleanup on top of the
    /// built-in engine shutdown".
    pub fn default_stopping() -> impl FnMut(&mut Context<Picasso>) + Send + 'static {
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
        (self.stopping)(ctx);
    }
}

impl Handler<SingleDelivery> for Picasso {
    fn handle(&mut self, msg: SingleDelivery, ctx: &mut Context<Self>) -> Result<Flow> {
        let pairs = PipelineMsg::Delivery(msg.0).into_pairs();
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
        pairs: Vec<(VideoFrameProxy, SharedBuffer)>,
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

        // User batch-level hook — observes/forwards the batch by
        // shared reference.  Errors are logged but non-fatal so a
        // failed observer does not drop the frame by default.
        if let Err(e) = (self.on_delivery)(&pairs, ctx) {
            log::warn!("[{}] on_delivery failed: {e}", ctx.own_name());
        }

        for (frame, buffer) in pairs {
            let sid = frame.get_source_id();
            let view = SurfaceView::from_buffer(&buffer, 0)
                .map_err(|e| anyhow!("SurfaceView::from_buffer: {e}"))?;
            let src_rect = (self.src_rect)(&sid, &frame);
            if let Err(e) = self.engine.send_frame(&sid, frame, view, src_rect) {
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
    on_delivery: DeliveryHook,
    src_rect: SrcRectProvider,
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
    on_delivery: Option<DeliveryHook>,
    src_rect: Option<SrcRectProvider>,
}

impl PicassoInboxBuilder {
    /// Empty bundle — every hook defaults to its matching
    /// `Picasso::default_*` at [`build`](Self::build) time.
    pub fn new() -> Self {
        Self {
            on_delivery: None,
            src_rect: None,
        }
    }

    /// Install a batch-level delivery hook.  The hook receives the
    /// unsealed `(VideoFrameProxy, SharedBuffer)` batch by
    /// **shared slice reference** before frames are fed to the
    /// engine, so the body can iterate and route frames into
    /// nested helpers (overlays, analytics, …) without consuming
    /// the batch.  Returning `Err` is logged but does not abort
    /// the loop; see [`Picasso::send_pairs`](Picasso).
    pub fn on_delivery<F>(mut self, f: F) -> Self
    where
        F: FnMut(&[(VideoFrameProxy, SharedBuffer)], &mut Context<Picasso>) -> Result<()>
            + Send
            + 'static,
    {
        self.on_delivery = Some(Box::new(f));
        self
    }

    /// Install a per-frame `src_rect` provider.  Returning
    /// `Some(rect)` feeds the rectangle to
    /// [`PicassoEngine::send_frame`]; returning `None` leaves the
    /// default `None` behaviour (use the whole source surface).
    pub fn src_rect_for<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &VideoFrameProxy) -> Option<Rect> + Send + Sync + 'static,
    {
        self.src_rect = Some(Arc::new(f));
        self
    }

    /// Finalise the bundle, substituting defaults for every omitted
    /// setter.
    pub fn build(self) -> PicassoInbox {
        let PicassoInboxBuilder {
            on_delivery,
            src_rect,
        } = self;
        PicassoInbox {
            on_delivery: on_delivery.unwrap_or_else(|| Box::new(Picasso::default_on_delivery())),
            src_rect: src_rect.unwrap_or_else(Picasso::default_src_rect_for),
        }
    }
}

impl Default for PicassoInboxBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Loop-level common knobs + user shutdown hook for [`Picasso`].
/// Built through [`PicassoCommon::builder`] and handed to
/// [`PicassoBuilder::common`].
pub struct PicassoCommon {
    poll_timeout: Duration,
    stopping: OnStoppingHook,
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
    stopping: Option<OnStoppingHook>,
}

impl PicassoCommonBuilder {
    /// Empty bundle — `poll_timeout` defaults to
    /// [`DEFAULT_POLL_TIMEOUT`] and `stopping` to a no-op.
    pub fn new() -> Self {
        Self {
            poll_timeout: None,
            stopping: None,
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
    pub fn stopping<F>(mut self, f: F) -> Self
    where
        F: FnMut(&mut Context<Picasso>) + Send + 'static,
    {
        self.stopping = Some(Box::new(f));
        self
    }

    /// Finalise the bundle, substituting defaults for every omitted
    /// setter.
    pub fn build(self) -> PicassoCommon {
        let PicassoCommonBuilder {
            poll_timeout,
            stopping,
        } = self;
        PicassoCommon {
            poll_timeout: poll_timeout.unwrap_or(DEFAULT_POLL_TIMEOUT),
            stopping: stopping.unwrap_or_else(|| Box::new(Picasso::default_stopping())),
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
    engine_factory: Option<PicassoEngineFactory>,
    source_spec: Option<SourceSpecFactory>,
    inbox: Option<PicassoInbox>,
    common: Option<PicassoCommon>,
}

impl PicassoBuilder {
    /// Start a builder with framework poll defaults.
    pub fn new(name: StageName, capacity: usize) -> Self {
        Self {
            name,
            capacity,
            downstream: None,
            engine_factory: None,
            source_spec: None,
            inbox: None,
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

    /// Required: engine factory.  See [`PicassoEngineFactory`].
    pub fn engine_factory<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&BuildCtx, Router<EncodedMsg>) -> Result<Arc<PicassoEngine>> + Send + 'static,
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

    /// Install a [`PicassoInbox`] bundle.  Omitting this call is
    /// equivalent to `.inbox(PicassoInbox::default())`, which
    /// wires every slot to the matching `Picasso::default_*`.
    pub fn inbox(mut self, i: PicassoInbox) -> Self {
        self.inbox = Some(i);
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
            engine_factory,
            source_spec,
            inbox,
            common,
        } = self;
        let source_spec =
            source_spec.ok_or_else(|| anyhow!("Picasso: missing source_spec_factory"))?;
        let engine_factory =
            engine_factory.ok_or_else(|| anyhow!("Picasso: missing engine_factory"))?;
        let PicassoInbox {
            on_delivery,
            src_rect,
        } = inbox.unwrap_or_default();
        let PicassoCommon {
            poll_timeout,
            stopping,
        } = common.unwrap_or_default();
        Ok(ActorBuilder::new(name, capacity)
            .poll_timeout(poll_timeout)
            .factory(move |bx| {
                let router: Router<EncodedMsg> = bx.router(downstream.as_ref())?;
                let engine = (engine_factory)(bx, router)?;
                Ok(Picasso {
                    engine,
                    source_spec,
                    on_delivery,
                    src_rect,
                    registered: HashSet::new(),
                    poll_timeout,
                    stopping,
                    shutdown_done: false,
                })
            }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::supervisor::{StageKind, StageName};
    use picasso::prelude::{Callbacks, GeneralSpec, OnEncodedFrame, OutputMessage};

    fn dummy_spec(_sid: &str, _w: u32, _h: u32, _fn_: i32, _fd: i32) -> Result<SourceSpec> {
        Ok(SourceSpec::default())
    }

    /// Smallest `OnEncodedFrame` impl that owns a router and does
    /// its own sends — the shape users are expected to provide.
    struct TestSink {
        router: Router<EncodedMsg>,
    }

    impl OnEncodedFrame for TestSink {
        fn call(&self, output: OutputMessage) {
            match output {
                OutputMessage::VideoFrame(frame) => {
                    let _ = self.router.send(EncodedMsg::Frame {
                        frame,
                        payload: None,
                    });
                }
                OutputMessage::EndOfStream(eos) => {
                    let _ = self.router.send(EncodedMsg::SourceEos {
                        source_id: eos.source_id,
                    });
                }
            }
        }
    }

    fn make_engine_factory() -> PicassoEngineFactory {
        Box::new(|_bx, router| {
            let callbacks = Callbacks::builder()
                .on_encoded_frame(TestSink { router })
                .build();
            let general = GeneralSpec::builder().name("demo").build();
            Ok(Arc::new(PicassoEngine::new(general, callbacks)))
        })
    }

    #[test]
    fn builder_requires_source_spec_and_engine_factory() {
        let name = StageName::unnamed(StageKind::Picasso);
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
        let name = StageName::unnamed(StageKind::Picasso);
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
        let name = StageName::unnamed(StageKind::Picasso);
        let _ = Picasso::builder(name, 4)
            .downstream(StageName::unnamed(StageKind::Mp4Mux))
            .engine_factory(make_engine_factory())
            .source_spec_factory(dummy_spec)
            .inbox(
                PicassoInbox::builder()
                    .src_rect_for(|_sid, _frame| None)
                    .on_delivery(|_pairs, _ctx| Ok(()))
                    .build(),
            )
            .common(
                PicassoCommon::builder()
                    .poll_timeout(Duration::from_millis(50))
                    .stopping(Picasso::default_stopping())
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
        let name = StageName::unnamed(StageKind::Picasso);
        let _ = Picasso::builder(name, 4)
            .engine_factory(make_engine_factory())
            .source_spec_factory(dummy_spec)
            .common(
                PicassoCommon::builder()
                    .stopping(move |_ctx| {
                        flag_hook.store(true, Ordering::SeqCst);
                    })
                    .build(),
            )
            .build()
            .unwrap();
        assert!(!flag.load(Ordering::SeqCst));
    }

    /// Runtime invariant: the `on_delivery` and `src_rect` slots
    /// are no longer `Option<_>`.  A minimal builder without any
    /// `.inbox(...)` call still produces a runtime [`Picasso`]
    /// whose `on_delivery` and `src_rect` are populated with the
    /// matching no-op defaults.
    #[test]
    fn runtime_invariant_all_hooks_populated() {
        use crate::context::BuildCtx;
        use crate::registry::Registry;
        use crate::shared::SharedStore;

        let name = StageName::unnamed(StageKind::Picasso);
        let ab = Picasso::builder(name.clone(), 4)
            .engine_factory(make_engine_factory())
            .source_spec_factory(dummy_spec)
            .build()
            .unwrap();
        let parts = ab.into_parts();
        let reg = Arc::new(Registry::new());
        let shared = Arc::new(SharedStore::new());
        let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let bx = BuildCtx::new(&parts.name, &reg, &shared, &stop_flag);
        let pic = (parts.factory)(&bx).expect("factory resolves");
        let Picasso {
            on_delivery: _,
            src_rect: _,
            stopping: _,
            ..
        } = pic;
    }
}
