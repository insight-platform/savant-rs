//! [`NvInfer`] — multiplexed `PipelineMsg` → `PipelineMsg` stage
//! wrapping a user-supplied
//! [`NvInferBatchingOperator`].
//!
//! The stage owns batching-operator lifecycle and routes operator
//! outputs into user-provided result hooks:
//!
//! * Captures a [`Router<PipelineMsg>`] at build time — optionally
//!   bound to a default peer via [`NvInferBuilder::downstream`] — and
//!   hands it to three user-supplied per-variant hooks
//!   ([`on_inference`](NvInferResultsBuilder::on_inference),
//!   [`on_source_eos`](NvInferResultsBuilder::on_source_eos),
//!   [`on_error`](NvInferResultsBuilder::on_error)).  The stage dispatches
//!   each [`OperatorOutput`] variant to its matching hook both from
//!   the operator's result callback and from the final
//!   `graceful_shutdown` drain, preserving stream-aligned
//!   `SourceEos` ordering.  Inside each hook use `router.send(msg)`
//!   to route to the configured default peer, or
//!   `router.send_to(&peer, msg)` to route to any registered stage by
//!   name.
//! * Accepts the operator by **factory closure** — the factory
//!   receives the ready-to-install
//!   [`OperatorResultCallback`]
//!   so the user can plug it into whatever their nvinfer
//!   construction needs (warmup, cached TRT engines, custom batch
//!   formation, …) without the stage having an opinion.
//! * Routes
//!   [`PipelineMsg::Delivery`]
//!   and
//!   [`PipelineMsg::Deliveries`]
//!   through
//!   [`NvInferBatchingOperator::add_frame`](deepstream_nvinfer::prelude::NvInferBatchingOperator::add_frame)
//!   on a flattened `Vec<(frame, buffer)>` basis — matching the
//!   "one code path per stage" contract of [`PipelineMsg::into_pairs`](crate::envelopes::PipelineMsg::into_pairs).
//! * Routes
//!   [`PipelineMsg::SourceEos`]
//!   through
//!   [`NvInferBatchingOperator::send_eos`](deepstream_nvinfer::prelude::NvInferBatchingOperator::send_eos)
//!   — the downstream sentinel is emitted by the operator callback
//!   on `OperatorOutput::Eos`, preserving stream alignment.
//! * Calls
//!   [`NvInferBatchingOperator::flush_idle`](deepstream_nvinfer::prelude::NvInferBatchingOperator::flush_idle)
//!   every [`Actor::poll_timeout`] when the inbox is idle.
//! * On [`Actor::stopping`] calls
//!   [`NvInferBatchingOperator::graceful_shutdown`](deepstream_nvinfer::prelude::NvInferBatchingOperator::graceful_shutdown)
//!   and dispatches every drained output through the same three
//!   hooks.
//!
//! # Grouped builder surface
//!
//! The [`NvInferBuilder`] exposes two bundle setters —
//! [`results`](NvInferBuilder::results) and
//! [`common`](NvInferBuilder::common) — instead of one setter per
//! hook.  Build the bundles via [`NvInferResults::builder`] and
//! [`NvInferCommon::builder`]:
//!
//! ```ignore
//! NvInfer::builder(name, cap)
//!     .downstream(tracker_name)
//!     .operator_factory(|_bx, cb| { /* build operator */ })
//!     .results(
//!         NvInferResults::builder()
//!             .on_inference(|inf, router, ctx| { /* custom post */ })
//!             .build(), // on_source_eos + on_error use defaults
//!     )
//!     .common(
//!         NvInferCommon::builder()
//!             .poll_timeout(Duration::from_millis(50))
//!             .build(),
//!     )
//!     .build()?;
//! ```
//!
//! Omitting `.results(...)` / `.common(...)` entirely is
//! equivalent to installing their `::default()` values — every
//! hook and knob has a sensible out-of-the-box default (see
//! below).
//!
//! # Default forwarders
//!
//! Every hook has an out-of-the-box default — if the matching
//! [`NvInferResults`] / [`NvInferCommon`] field is not explicitly
//! set, [`NvInferBuilder::build`] auto-installs it:
//!
//! * [`NvInfer::default_on_inference`] — for every output tensor
//!   of every frame in the batch, attach a persistent attribute to
//!   the frame with `namespace = <stage name>` (the actor's
//!   [`StageName`] via its [`std::fmt::Display`] impl) and
//!   `name = <tensor layer name>` carrying
//!   [`AttributeValue::bytes(&dims_i64, host_bytes, None)`](savant_core::primitives::attribute_value::AttributeValue::bytes).
//!   Tensor dims are stored inside the `Bytes` variant alongside the
//!   raw host payload (one attribute per output layer, per frame).
//!   After attachment, forwards
//!   [`PipelineMsg::Deliveries`]
//!   downstream via `router.send(..)`.  Tensors with host data
//!   unavailable (e.g. `host_copy_enabled=false`) are logged and
//!   skipped.
//! * [`NvInfer::default_on_source_eos`] — forwards
//!   [`PipelineMsg::SourceEos`]
//!   via `router.send(..)`.
//! * [`NvInfer::default_on_error`] — logs the operator error at
//!   `error` level, prefixed by the stage name.
//! * [`NvInfer::default_stopping`] — a no-op user shutdown hook.
//!   See [`NvInferCommonBuilder::stopping`] for the composition
//!   contract (built-in `graceful_shutdown` + drain-dispatch
//!   always run first; the user hook runs after).
//!
//! Override any branch to add stage-specific processing (e.g. YOLO
//! post-processing, custom metrics) — the builder is additive and
//! each setter replaces exactly one branch's behaviour.
//!
//! # Runtime invariant
//!
//! [`NvInfer`] stores its three hooks as non-`Option`
//! `Arc<dyn Fn>`.  The builder's `Option<...>` fields exist purely
//! as an internal "was the setter called?" marker;
//! [`NvInferBuilder::build`] always substitutes the matching
//! `NvInfer::default_on_*` before the actor value is
//! constructed.  There is therefore no runtime code path where an
//! [`OperatorOutput`] variant is dispatched to `None` — every
//! branch (`Inference`, `Eos`, `Error`) is always handled, whether
//! by a user-supplied closure or by the auto-installed default.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use deepstream_nvinfer::prelude::NvInferBatchingOperator;
use deepstream_nvinfer::{
    NvInferError, OperatorInferenceOutput, OperatorOutput, OperatorResultCallback,
};
use savant_core::primitives::attribute_value::AttributeValue;
use savant_core::primitives::WithAttributes;

use crate::envelopes::{BatchDelivery, PipelineMsg, SingleDelivery};
use crate::router::Router;
use crate::supervisor::StageName;
use crate::{
    Actor, ActorBuilder, BuildCtx, Context, Dispatch, ErrorAction, Flow, Handler, HookCtx,
    ShutdownPayload, SourceEosPayload,
};

/// Default graceful-shutdown drain timeout.
pub const DEFAULT_DRAIN_TIMEOUT: Duration = Duration::from_secs(30);
/// Default inbox receive-poll cadence.
pub const DEFAULT_POLL_TIMEOUT: Duration = Duration::from_millis(100);

/// Hook fired for every
/// [`OperatorOutput::Inference`]
/// payload.  Receives the
/// [`OperatorInferenceOutput`]
/// (which owns the output buffer + tensor pointers) and the stage's
/// [`Router<PipelineMsg>`].  Typical use:
///
/// * Attach per-tensor attributes / decoded detections to
///   `inf.frames()[i].frame`.
/// * Call [`OperatorInferenceOutput::take_deliveries`](deepstream_nvinfer::OperatorInferenceOutput::take_deliveries),
///   drop `inf`, then send
///   [`PipelineMsg::Deliveries`]
///   downstream via `router.send(..)` or `router.send_to(..)`.
///
/// See [`NvInfer::default_on_inference`] for the ready-made
/// "attach raw tensors + forward" default.  The third parameter
/// is the off-loop [`HookCtx`] for shared-state / registry access
/// from the operator callback thread.
pub type OnInferenceHook =
    Arc<dyn Fn(OperatorInferenceOutput, &Router<PipelineMsg>, &HookCtx) + Send + Sync + 'static>;

/// Hook fired for every
/// [`OperatorOutput::Eos`]
/// payload.  Receives the source identifier, the router, and the
/// off-loop [`HookCtx`].  Typical use:
/// `router.send(PipelineMsg::SourceEos { source_id: source_id.into() })`
/// followed by `Ok(Flow::Cont)`.
///
/// Return semantics — unified with every stage's
/// `on_source_eos`:
///
/// * `Ok(Flow::Cont)` — keep the actor loop running (default).
/// * `Ok(Flow::Stop)` — ask the loop to wind down cooperatively
///   (stage requests a stop via [`HookCtx::request_stop`]).
/// * `Err(_)` — log the error and request a cooperative stop;
///   equivalent to `Ok(Flow::Stop)` modulo the `error!` log line
///   emitted by the stage.
pub type OnSourceEosHook =
    Arc<dyn Fn(&str, &Router<PipelineMsg>, &HookCtx) -> Result<Flow> + Send + Sync + 'static>;

/// Hook fired for every
/// [`OperatorOutput::Error`]
/// payload.  Receives the operator error, the router, and the
/// off-loop [`HookCtx`] (use
/// [`HookCtx::own_name`](crate::HookCtx::own_name) for
/// log prefixing).  The router is handed over for callers that want
/// to translate the error into a downstream control message.
///
/// Return semantics — unified with every stage's `on_error`
/// hook via [`ErrorAction`]:
///
/// * [`ErrorAction::LogAndContinue`] — keep processing subsequent
///   outputs.  Default.
/// * [`ErrorAction::Fatal`] — stage escalates: requests a
///   cooperative stop through [`HookCtx::request_stop`] so the
///   actor loop exits through its normal
///   [`Actor::stopping`] path.
/// * [`ErrorAction::Swallow`] — same as `LogAndContinue` for an
///   actor (no latched error semantics here); provided for
///   vocabulary parity with
///   [`Mp4DemuxerSource`](super::mp4_demuxer::Mp4DemuxerSource).
///
/// The stage emits a single `error!` log line (prefixed by the
/// stage name) *before* invoking the hook, so user code can focus
/// on classification rather than logging.
pub type OnErrorHook = Arc<
    dyn Fn(&NvInferError, &Router<PipelineMsg>, &HookCtx) -> ErrorAction + Send + Sync + 'static,
>;

/// User shutdown hook invoked from [`Actor::stopping`] *after* the
/// stage's built-in cleanup (guarded
/// [`NvInferBatchingOperator::graceful_shutdown`] and drained output
/// dispatch) has completed.
///
/// Runs on the actor thread with full access to the [`Context`].
/// Ideal for final metrics flushes, bespoke log lines, or custom
/// bookkeeping that must observe the drained operator state.  The
/// stage's load-bearing cleanup (drain + fan drained outputs
/// through the per-variant hooks) cannot be skipped — users that
/// need to replace it entirely should implement [`Actor`] directly
/// on a bespoke struct.
pub type OnStoppingHook = Box<dyn FnMut(&mut Context<NvInfer>) + Send + 'static>;

/// Factory closure: builds the [`NvInferBatchingOperator`] once
/// the stage has wired the result callback.  Invoked once
/// when [`System::run`](super::super::system::System::run)
/// constructs the [`NvInfer`] actor; it receives the
/// fully-populated [`BuildCtx`] so it can resolve sibling stages
/// and shared state (e.g. cached converters) without capturing
/// them from the builder's definition site.
pub type NvInferOperatorFactory =
    Box<dyn FnOnce(&BuildCtx, OperatorResultCallback) -> Result<NvInferBatchingOperator> + Send>;

/// `PipelineMsg` → `PipelineMsg` actor stage wrapping an
/// [`NvInferBatchingOperator`].
pub struct NvInfer {
    operator: NvInferBatchingOperator,
    router: Router<PipelineMsg>,
    hook_ctx: HookCtx,
    on_inference: OnInferenceHook,
    on_source_eos: OnSourceEosHook,
    on_error: OnErrorHook,
    stopping: OnStoppingHook,
    drain_timeout: Duration,
    poll_timeout: Duration,
    drain_done: bool,
}

impl NvInfer {
    /// Start a fluent builder for an infer actor registered under
    /// `name` with inbox capacity `capacity`.
    pub fn builder(name: StageName, capacity: usize) -> NvInferBuilder {
        NvInferBuilder::new(name, capacity)
    }

    /// Default `on_inference` forwarder.
    ///
    /// For every `(frame, element, tensor)` tuple in `inf.frames()`,
    /// attaches a persistent attribute to the frame with
    /// `namespace = <stage name>` (stringified via [`StageName`]'s
    /// [`std::fmt::Display`] impl) and `name = <tensor layer name>`,
    /// carrying
    /// [`AttributeValue::bytes`](savant_core::primitives::attribute_value::AttributeValue::bytes)
    /// whose `dims` field mirrors
    /// [`TensorView::dims`](deepstream_nvinfer::TensorView::dims) and
    /// whose payload is the raw host byte slice from
    /// `tensor.host_ptr[..tensor.byte_length]`.  One attribute per
    /// tensor per frame.  Tensors with host data unavailable
    /// (`host_copy_enabled=false`, null pointer, or zero byte
    /// length) are logged at `warn` level and skipped.
    ///
    /// After attachment, sends
    /// [`PipelineMsg::Deliveries`]
    /// downstream via `router.send(..)`.
    ///
    /// # Why this is the default
    ///
    /// The "raw tensor attributes" representation is universal —
    /// independent of model family, dtype, or post-processing
    /// strategy.  Application-specific decoders (YOLO, DETR, …) are
    /// encouraged to replace this with a custom `on_inference`
    /// closure that translates tensors into
    /// [`VideoObject`](savant_core::primitives::object::VideoObject)
    /// detections.
    ///
    /// # Behavioural testing
    ///
    /// A unit test for this helper would require a fabricated
    /// [`OperatorInferenceOutput`]
    /// with backing `GstBuffer` memory, which the public API does
    /// not expose.  Exercise this path with an integration pipeline
    /// that drives real tensors through the stage.
    pub fn default_on_inference(
    ) -> impl Fn(OperatorInferenceOutput, &Router<PipelineMsg>, &HookCtx) + Send + Sync + 'static
    {
        |mut inf, router, ctx| {
            let ns = ctx.own_name().to_string();
            for frame_output in inf.frames() {
                for element in &frame_output.elements {
                    for tensor in &element.tensors {
                        if !tensor.host_copy_enabled
                            || tensor.host_ptr.is_null()
                            || tensor.byte_length == 0
                        {
                            log::warn!(
                                "[{ns}] tensor '{}' host data unavailable (host_copy_enabled={}, null={}, byte_length={}); attribute not attached",
                                tensor.name,
                                tensor.host_copy_enabled,
                                tensor.host_ptr.is_null(),
                                tensor.byte_length,
                            );
                            continue;
                        }
                        let dims: Vec<i64> =
                            tensor.dims.dimensions.iter().map(|d| *d as i64).collect();
                        // SAFETY: `host_copy_enabled` is true, the
                        // pointer is non-null, and `byte_length`
                        // bytes are valid: the owning
                        // `OperatorInferenceOutput` (`inf`) keeps
                        // the output GstBuffer alive for the
                        // duration of this callback — raw host-side
                        // data is stable until `inf` is dropped.
                        let bytes = unsafe {
                            std::slice::from_raw_parts(
                                tensor.host_ptr as *const u8,
                                tensor.byte_length,
                            )
                        };
                        let value = AttributeValue::bytes(&dims, bytes, None);
                        let mut frame = frame_output.frame.clone();
                        frame.set_persistent_attribute(
                            &ns,
                            &tensor.name,
                            &None,
                            false,
                            vec![value],
                        );
                    }
                }
            }
            if let Some(sealed) = inf.take_deliveries() {
                drop(inf);
                if !router.send(PipelineMsg::Deliveries(sealed)) {
                    log::warn!("[{ns}] downstream closed; dropping sealed batch");
                }
            }
        }
    }

    /// Default `on_source_eos` forwarder.  Sends
    /// [`PipelineMsg::SourceEos { source_id }`](crate::envelopes::PipelineMsg::SourceEos)
    /// via `router.send(..)` and returns `Ok(Flow::Cont)` — the
    /// stage keeps running to serve subsequent sources.  The emit
    /// site preserves the stream-aligned contract: the operator
    /// guarantees
    /// [`OperatorOutput::Eos`]
    /// fires strictly after the last
    /// [`OperatorOutput::Inference`]
    /// for the same `source_id`.
    pub fn default_on_source_eos(
    ) -> impl Fn(&str, &Router<PipelineMsg>, &HookCtx) -> Result<Flow> + Send + Sync + 'static {
        |source_id, router, _ctx| {
            log::info!("OperatorOutput::Eos for source_id={source_id}; propagating");
            if !router.send(PipelineMsg::SourceEos {
                source_id: source_id.to_string(),
            }) {
                log::warn!("downstream closed; dropping SourceEos({source_id})");
            }
            Ok(Flow::Cont)
        }
    }

    /// Default `on_error` classifier.  Returns
    /// [`ErrorAction::LogAndContinue`] unconditionally — a
    /// misbehaving model should not tear the pipeline down.  The
    /// stage emits a single `error!` line prefixed by the
    /// stage name *before* the hook fires, so this classifier has
    /// no I/O of its own.
    pub fn default_on_error(
    ) -> impl Fn(&NvInferError, &Router<PipelineMsg>, &HookCtx) -> ErrorAction + Send + Sync + 'static
    {
        |_err, _router, _ctx| ErrorAction::LogAndContinue
    }

    /// Default user shutdown hook — a no-op.  The stage's own
    /// [`Actor::stopping`] body always runs the guarded
    /// [`NvInferBatchingOperator::graceful_shutdown`] drain and fans
    /// the drained outputs through the per-variant hooks before this
    /// hook fires, so omitting `.stopping(...)` simply means "don't
    /// add any extra cleanup on top of the built-in drain".
    pub fn default_stopping() -> impl FnMut(&mut Context<NvInfer>) + Send + 'static {
        |_ctx| {}
    }
}

impl Actor for NvInfer {
    type Msg = PipelineMsg;

    fn handle(&mut self, msg: PipelineMsg, ctx: &mut Context<Self>) -> Result<Flow> {
        msg.dispatch(self, ctx)
    }

    fn poll_timeout(&self) -> Duration {
        self.poll_timeout
    }

    fn started(&mut self, ctx: &mut Context<Self>) -> Result<()> {
        log::info!("[{}] nvinfer started", ctx.own_name());
        Ok(())
    }

    fn on_tick(&mut self, ctx: &mut Context<Self>) -> Result<Flow> {
        if let Err(e) = self.operator.flush_idle() {
            log::warn!("[{}] flush_idle failed: {e}", ctx.own_name());
        }
        Ok(Flow::Cont)
    }

    fn stopping(&mut self, ctx: &mut Context<Self>) {
        if self.drain_done {
            return;
        }
        match self.operator.graceful_shutdown(self.drain_timeout) {
            Ok(drained) => {
                log::info!(
                    "[{}] drained {} remaining outputs",
                    ctx.own_name(),
                    drained.len()
                );
                for out in drained {
                    dispatch_output(
                        out,
                        &self.on_inference,
                        &self.on_source_eos,
                        &self.on_error,
                        &self.router,
                        &self.hook_ctx,
                    );
                }
            }
            Err(e) => {
                log::error!("[{}] graceful_shutdown failed: {e}", ctx.own_name());
            }
        }
        self.drain_done = true;
        log::info!("[{}] nvinfer stopping", ctx.own_name());
        (self.stopping)(ctx);
    }
}

impl Handler<SingleDelivery> for NvInfer {
    fn handle(&mut self, msg: SingleDelivery, ctx: &mut Context<Self>) -> Result<Flow> {
        let pairs = PipelineMsg::Delivery(msg.0).into_pairs();
        submit_pairs(&mut self.operator, pairs, ctx.own_name())
    }
}

impl Handler<BatchDelivery> for NvInfer {
    fn handle(&mut self, msg: BatchDelivery, ctx: &mut Context<Self>) -> Result<Flow> {
        let pairs = msg.0.unseal();
        submit_pairs(&mut self.operator, pairs, ctx.own_name())
    }
}

impl Handler<SourceEosPayload> for NvInfer {
    fn handle(&mut self, msg: SourceEosPayload, ctx: &mut Context<Self>) -> Result<Flow> {
        log::info!(
            "[{}] SourceEos {}: initiating operator drain",
            ctx.own_name(),
            msg.source_id
        );
        if let Err(e) = self.operator.send_eos(&msg.source_id) {
            log::warn!(
                "[{}] send_eos({}) failed: {e}",
                ctx.own_name(),
                msg.source_id
            );
        }
        Ok(Flow::Cont)
    }
}

/// Default no-op — the loop driver has already consumed the
/// shutdown hint from `PipelineMsg::Shutdown`.
impl Handler<ShutdownPayload> for NvInfer {}

fn submit_pairs(
    operator: &mut NvInferBatchingOperator,
    pairs: Vec<(
        savant_core::primitives::frame::VideoFrame,
        deepstream_buffers::SharedBuffer,
    )>,
    stage: &StageName,
) -> Result<Flow> {
    for (frame, buffer) in pairs {
        if let Err(e) = operator.add_frame(frame, buffer) {
            log::error!("[{stage}] add_frame failed: {e}");
            return Err(anyhow!("infer add_frame: {e}"));
        }
    }
    Ok(Flow::Cont)
}

/// Shared three-way dispatcher used by both the operator result
/// callback and the `stopping()` drain path.  Keeps the variant →
/// hook mapping authored once.
///
/// The dispatcher owns the uniform error-logging and flow-control
/// translation contract documented on [`OnSourceEosHook`] /
/// [`OnErrorHook`]:
///
/// * `on_inference` — pure fan-out, no return type.
/// * `on_source_eos` — `Ok(Flow::Stop)` or `Err(_)` calls
///   [`HookCtx::request_stop`] so the actor loop exits through
///   its [`Actor::stopping`] path.  `Err(_)` also logs the error.
/// * `on_error` — the stage emits the `error!` log line, then
///   the hook classifies.  [`ErrorAction::Fatal`] triggers
///   [`HookCtx::request_stop`] and aborts the default sink so no
///   further drops slip downstream; `LogAndContinue` / `Swallow`
///   keep the stage running.
fn dispatch_output(
    out: OperatorOutput,
    on_inference: &OnInferenceHook,
    on_source_eos: &OnSourceEosHook,
    on_error: &OnErrorHook,
    router: &Router<PipelineMsg>,
    ctx: &HookCtx,
) {
    match out {
        OperatorOutput::Inference(inf) => on_inference(inf, router, ctx),
        OperatorOutput::Eos { source_id } => match on_source_eos(&source_id, router, ctx) {
            Ok(Flow::Cont) => {}
            Ok(Flow::Stop) => {
                log::info!(
                    "[{}] on_source_eos requested stop (source_id={source_id})",
                    ctx.own_name()
                );
                ctx.request_stop();
            }
            Err(e) => {
                log::error!(
                    "[{}] on_source_eos error (source_id={source_id}): {e}",
                    ctx.own_name()
                );
                ctx.request_stop();
            }
        },
        OperatorOutput::Error(err) => {
            log::error!("[{}] operator error: {err}", ctx.own_name());
            match on_error(&err, router, ctx) {
                ErrorAction::LogAndContinue | ErrorAction::Swallow => {}
                ErrorAction::Fatal => {
                    log::error!(
                        "[{}] on_error returned Fatal; aborting default sink and requesting stop",
                        ctx.own_name()
                    );
                    if let Some(sink) = router.default_sink() {
                        sink.abort();
                    }
                    ctx.request_stop();
                }
            }
        }
    }
}

/// Per-variant operator-output hook bundle — one branch per
/// [`OperatorOutput`] variant.
///
/// Built through [`NvInferResults::builder`] and handed to
/// [`NvInferBuilder::results`].  Omitted branches auto-install
/// the matching `NvInfer::default_on_*` at build time, so the
/// runtime invariant "every `OperatorOutput` variant is always
/// dispatched to a user-supplied or auto-installed hook" holds
/// regardless of how much the user overrides.
pub struct NvInferResults {
    on_inference: OnInferenceHook,
    on_source_eos: OnSourceEosHook,
    on_error: OnErrorHook,
}

impl NvInferResults {
    /// Start a builder that auto-installs every default on
    /// [`NvInferResultsBuilder::build`].
    pub fn builder() -> NvInferResultsBuilder {
        NvInferResultsBuilder::new()
    }
}

impl Default for NvInferResults {
    /// Every branch wired to its matching `NvInfer::default_on_*`.
    fn default() -> Self {
        NvInferResultsBuilder::new().build()
    }
}

/// Fluent builder for [`NvInferResults`] — one setter per
/// `OperatorOutput` variant.
pub struct NvInferResultsBuilder {
    on_inference: Option<OnInferenceHook>,
    on_source_eos: Option<OnSourceEosHook>,
    on_error: Option<OnErrorHook>,
}

impl NvInferResultsBuilder {
    /// Empty bundle — every hook defaults to its
    /// `NvInfer::default_on_*` equivalent at
    /// [`build`](Self::build) time.
    pub fn new() -> Self {
        Self {
            on_inference: None,
            on_source_eos: None,
            on_error: None,
        }
    }

    /// Override the `on_inference` hook.  Omitting this setter is
    /// equivalent to calling
    /// `.on_inference(NvInfer::default_on_inference())` — attach
    /// raw tensor attributes, then forward
    /// [`PipelineMsg::Deliveries`]
    /// via `router.send(..)`.
    ///
    /// The closure receives the inference output, the stage's
    /// [`Router<PipelineMsg>`], and a [`HookCtx`] for shared-state
    /// / registry access from the operator callback thread.
    pub fn on_inference<F>(mut self, f: F) -> Self
    where
        F: Fn(OperatorInferenceOutput, &Router<PipelineMsg>, &HookCtx) + Send + Sync + 'static,
    {
        self.on_inference = Some(Arc::new(f));
        self
    }

    /// Override the `on_source_eos` hook.  Omitting this setter is
    /// equivalent to calling
    /// `.on_source_eos(NvInfer::default_on_source_eos())` —
    /// forward [`PipelineMsg::SourceEos`]
    /// downstream and return `Ok(Flow::Cont)`.
    ///
    /// Return `Ok(Flow::Stop)` to request cooperative shutdown of
    /// the stage once this EOS is processed; `Err(_)` logs the
    /// error and also requests shutdown.
    pub fn on_source_eos<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &Router<PipelineMsg>, &HookCtx) -> Result<Flow> + Send + Sync + 'static,
    {
        self.on_source_eos = Some(Arc::new(f));
        self
    }

    /// Override the `on_error` hook.  Omitting this setter is
    /// equivalent to calling
    /// `.on_error(NvInfer::default_on_error())` — classify every
    /// error as [`ErrorAction::LogAndContinue`].
    ///
    /// The stage always emits an `error!` log line
    /// (stage-prefixed) *before* invoking this hook; the hook only
    /// classifies.  Return [`ErrorAction::Fatal`] to abort the
    /// default sink and request cooperative shutdown.
    pub fn on_error<F>(mut self, f: F) -> Self
    where
        F: Fn(&NvInferError, &Router<PipelineMsg>, &HookCtx) -> ErrorAction + Send + Sync + 'static,
    {
        self.on_error = Some(Arc::new(f));
        self
    }

    /// Finalise the bundle.  Every omitted branch collapses into
    /// the matching `NvInfer::default_on_*` so the resulting
    /// [`NvInferResults`] is always complete.
    pub fn build(self) -> NvInferResults {
        let NvInferResultsBuilder {
            on_inference,
            on_source_eos,
            on_error,
        } = self;
        NvInferResults {
            on_inference: on_inference.unwrap_or_else(|| Arc::new(NvInfer::default_on_inference())),
            on_source_eos: on_source_eos
                .unwrap_or_else(|| Arc::new(NvInfer::default_on_source_eos())),
            on_error: on_error.unwrap_or_else(|| Arc::new(NvInfer::default_on_error())),
        }
    }
}

impl Default for NvInferResultsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Loop-level common knobs + user shutdown hook.  Built through
/// [`NvInferCommon::builder`] and handed to
/// [`NvInferBuilder::common`].
pub struct NvInferCommon {
    drain_timeout: Duration,
    poll_timeout: Duration,
    stopping: OnStoppingHook,
}

impl NvInferCommon {
    /// Start a builder seeded with
    /// [`DEFAULT_DRAIN_TIMEOUT`] / [`DEFAULT_POLL_TIMEOUT`] / no-op
    /// stopping hook.
    pub fn builder() -> NvInferCommonBuilder {
        NvInferCommonBuilder::new()
    }
}

impl Default for NvInferCommon {
    /// `DEFAULT_DRAIN_TIMEOUT`, `DEFAULT_POLL_TIMEOUT`, and a no-op
    /// `stopping` hook.
    fn default() -> Self {
        NvInferCommonBuilder::new().build()
    }
}

/// Fluent builder for [`NvInferCommon`].
pub struct NvInferCommonBuilder {
    drain_timeout: Option<Duration>,
    poll_timeout: Option<Duration>,
    stopping: Option<OnStoppingHook>,
}

impl NvInferCommonBuilder {
    /// Empty bundle — build-time defaults are
    /// [`DEFAULT_DRAIN_TIMEOUT`] / [`DEFAULT_POLL_TIMEOUT`] /
    /// no-op stopping hook.
    pub fn new() -> Self {
        Self {
            drain_timeout: None,
            poll_timeout: None,
            stopping: None,
        }
    }

    /// Override the graceful-shutdown drain timeout (default
    /// [`DEFAULT_DRAIN_TIMEOUT`]).
    pub fn drain_timeout(mut self, d: Duration) -> Self {
        self.drain_timeout = Some(d);
        self
    }

    /// Inbox receive-poll cadence (default [`DEFAULT_POLL_TIMEOUT`]).
    pub fn poll_timeout(mut self, d: Duration) -> Self {
        self.poll_timeout = Some(d);
        self
    }

    /// Override the user shutdown hook — fired from
    /// [`Actor::stopping`] **after** the stage's built-in
    /// cleanup (guarded
    /// [`NvInferBatchingOperator::graceful_shutdown`] + drained
    /// output dispatch) has completed.
    ///
    /// Runs on the actor thread with full access to the
    /// [`Context`].  Use for final metrics flushes, bespoke log
    /// lines, or custom bookkeeping that must observe the drained
    /// operator state.  The built-in cleanup is **load-bearing**
    /// and always runs first; it cannot be skipped through this
    /// hook.
    pub fn stopping<F>(mut self, f: F) -> Self
    where
        F: FnMut(&mut Context<NvInfer>) + Send + 'static,
    {
        self.stopping = Some(Box::new(f));
        self
    }

    /// Finalise the bundle, substituting defaults for every
    /// omitted setter.
    pub fn build(self) -> NvInferCommon {
        let NvInferCommonBuilder {
            drain_timeout,
            poll_timeout,
            stopping,
        } = self;
        NvInferCommon {
            drain_timeout: drain_timeout.unwrap_or(DEFAULT_DRAIN_TIMEOUT),
            poll_timeout: poll_timeout.unwrap_or(DEFAULT_POLL_TIMEOUT),
            stopping: stopping.unwrap_or_else(|| Box::new(NvInfer::default_stopping())),
        }
    }
}

impl Default for NvInferCommonBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Fluent builder for [`NvInfer`].
///
/// The builder only exposes wiring-level configuration at the top
/// level (`downstream`, `operator_factory`).  Per-variant
/// operator-output hooks live on [`NvInferResults`]; loop-level
/// knobs and the user shutdown hook live on [`NvInferCommon`].
/// Install them via [`NvInferBuilder::results`] and
/// [`NvInferBuilder::common`].
pub struct NvInferBuilder {
    name: StageName,
    capacity: usize,
    downstream: Option<StageName>,
    operator_factory: Option<NvInferOperatorFactory>,
    results: Option<NvInferResults>,
    common: Option<NvInferCommon>,
}

impl NvInferBuilder {
    /// Start a builder for an infer actor registered under `name`
    /// with inbox capacity `capacity`.
    ///
    /// Every per-variant hook defaults to its
    /// `NvInfer::default_on_*` equivalent; loop-level knobs
    /// default to [`DEFAULT_DRAIN_TIMEOUT`] /
    /// [`DEFAULT_POLL_TIMEOUT`].  The user only needs to supply an
    /// `operator_factory`; `results` / `common` are optional and
    /// composable.
    pub fn new(name: StageName, capacity: usize) -> Self {
        Self {
            name,
            capacity,
            downstream: None,
            operator_factory: None,
            results: None,
            common: None,
        }
    }

    /// Optional default peer for `router.send(msg)` inside the
    /// per-variant hooks (typically the tracker).  When omitted,
    /// the hooks must use `router.send_to(&peer, msg)` for
    /// explicit name-based routing; a default-less
    /// `router.send(msg)` silently drops the message and logs
    /// once.
    pub fn downstream(mut self, peer: StageName) -> Self {
        self.downstream = Some(peer);
        self
    }

    /// Required: factory that builds the
    /// [`NvInferBatchingOperator`] given the already-wired result
    /// callback.  Invoked once by
    /// [`System::run`](super::super::system::System::run) when the
    /// [`NvInfer`] actor is constructed.
    pub fn operator_factory<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&BuildCtx, OperatorResultCallback) -> Result<NvInferBatchingOperator>
            + Send
            + 'static,
    {
        self.operator_factory = Some(Box::new(f));
        self
    }

    /// Install a [`NvInferResults`] bundle — one branch per
    /// `OperatorOutput` variant (`inference`, `source_eos`,
    /// `error`).  Omitting this call is equivalent to
    /// `.results(NvInferResults::default())`, which wires every
    /// branch to its matching `NvInfer::default_on_*`.
    pub fn results(mut self, r: NvInferResults) -> Self {
        self.results = Some(r);
        self
    }

    /// Install a [`NvInferCommon`] bundle — loop-level knobs
    /// (`drain_timeout`, `poll_timeout`) plus the user `stopping`
    /// hook.  Omitting this call is equivalent to
    /// `.common(NvInferCommon::default())`.
    pub fn common(mut self, c: NvInferCommon) -> Self {
        self.common = Some(c);
        self
    }

    /// Finalise the stage and obtain the Layer-A
    /// [`ActorBuilder<NvInfer>`].
    ///
    /// # Errors
    ///
    /// Returns `Err` if `operator_factory` is missing.  All three
    /// per-variant hooks and all loop knobs have defaults
    /// auto-installed when the matching `results` / `common`
    /// bundle is omitted.  `downstream` is optional — callers that
    /// route via [`Router::send_to`] exclusively may omit it.
    pub fn build(self) -> Result<ActorBuilder<NvInfer>> {
        let NvInferBuilder {
            name,
            capacity,
            downstream,
            operator_factory,
            results,
            common,
        } = self;
        let operator_factory =
            operator_factory.ok_or_else(|| anyhow!("NvInfer: missing operator_factory"))?;
        let NvInferResults {
            on_inference,
            on_source_eos,
            on_error,
        } = results.unwrap_or_default();
        let NvInferCommon {
            drain_timeout,
            poll_timeout,
            stopping,
        } = common.unwrap_or_default();
        Ok(ActorBuilder::new(name, capacity)
            .poll_timeout(poll_timeout)
            .factory(move |bx| {
                let hook_ctx = bx.hook_ctx();
                let router: Router<PipelineMsg> = bx.router(downstream.as_ref())?;
                let router_for_cb = router.clone();
                let on_inference_cb = on_inference.clone();
                let on_source_eos_cb = on_source_eos.clone();
                let on_error_cb = on_error.clone();
                let hook_ctx_cb = hook_ctx.clone();
                let result_cb: OperatorResultCallback = Box::new(move |out| {
                    dispatch_output(
                        out,
                        &on_inference_cb,
                        &on_source_eos_cb,
                        &on_error_cb,
                        &router_for_cb,
                        &hook_ctx_cb,
                    );
                });
                let operator = (operator_factory)(bx, result_cb)?;
                Ok(NvInfer {
                    operator,
                    router,
                    hook_ctx,
                    on_inference,
                    on_source_eos,
                    on_error,
                    stopping,
                    drain_timeout,
                    poll_timeout,
                    drain_done: false,
                })
            }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::addr::Addr;
    use crate::operator_sink::OperatorSink;
    use crate::registry::Registry;
    use crate::supervisor::{StageKind, StageName};
    use crossbeam::channel::{bounded, Receiver};

    fn noop_on_inference(_: OperatorInferenceOutput, _: &Router<PipelineMsg>, _: &HookCtx) {}
    fn noop_on_source_eos(_: &str, _: &Router<PipelineMsg>, _: &HookCtx) -> Result<Flow> {
        Ok(Flow::Cont)
    }
    fn noop_on_error(_: &NvInferError, _: &Router<PipelineMsg>, _: &HookCtx) -> ErrorAction {
        ErrorAction::LogAndContinue
    }

    /// Synthesise a bare [`HookCtx`] for the default-hook unit
    /// tests.  The shared store and registry are empty — the
    /// default forwarders only read `own_name` from the ctx, so
    /// empty-but-valid backing state is enough.
    fn test_hook_ctx() -> HookCtx {
        HookCtx::new(
            StageName::unnamed(StageKind::Infer),
            Arc::new(Registry::new()),
            Arc::new(crate::shared::SharedStore::new()),
            Arc::new(std::sync::atomic::AtomicBool::new(false)),
        )
    }

    /// Build a `Router<PipelineMsg>` whose default peer is a freshly
    /// registered `PipelineMsg` inbox.  Returns the router and the
    /// receiver so tests can observe forwarded messages.
    fn router_with_default_peer() -> (Router<PipelineMsg>, Receiver<PipelineMsg>) {
        let peer = StageName::unnamed(StageKind::Tracker);
        let (tx, rx) = bounded::<PipelineMsg>(4);
        let mut reg = Registry::new();
        reg.insert::<PipelineMsg>(peer.clone(), Addr::new(peer.clone(), tx));
        let addr: Addr<PipelineMsg> = reg.get::<PipelineMsg>(&peer).unwrap();
        let owner = StageName::unnamed(StageKind::Infer);
        let default = OperatorSink::new(owner.clone(), addr);
        let router = Router::new(owner, Arc::new(reg), Some(default));
        (router, rx)
    }

    #[test]
    fn builder_requires_operator_factory() {
        let name = StageName::unnamed(StageKind::Infer);
        let err = NvInfer::builder(name.clone(), 4).build().err().unwrap();
        assert!(err.to_string().contains("missing operator_factory"));
    }

    #[test]
    fn builder_accepts_full_config() {
        let name = StageName::unnamed(StageKind::Infer);
        let _ = NvInfer::builder(name, 4)
            .downstream(StageName::unnamed(StageKind::Tracker))
            .operator_factory(|_bx, _cb| unreachable!("not invoked in this test"))
            .results(
                NvInferResults::builder()
                    .on_inference(noop_on_inference)
                    .on_source_eos(noop_on_source_eos)
                    .on_error(noop_on_error)
                    .build(),
            )
            .common(
                NvInferCommon::builder()
                    .drain_timeout(Duration::from_secs(10))
                    .poll_timeout(Duration::from_millis(50))
                    .build(),
            )
            .build()
            .unwrap();
    }

    /// Downstream is optional — a builder without `.downstream()`
    /// still produces a valid `ActorBuilder`.  The resulting actor's
    /// `router.send(msg)` inside any hook returns `false` and warns
    /// once; routed sends still work.
    #[test]
    fn builder_without_downstream_is_accepted() {
        let name = StageName::unnamed(StageKind::Infer);
        let _ = NvInfer::builder(name, 4)
            .operator_factory(|_bx, _cb| unreachable!("not invoked in this test"))
            .results(
                NvInferResults::builder()
                    .on_inference(noop_on_inference)
                    .on_source_eos(noop_on_source_eos)
                    .on_error(noop_on_error)
                    .build(),
            )
            .build()
            .unwrap();
    }

    /// All three per-variant hooks have defaults that
    /// [`NvInferBuilder::build`] auto-installs — a builder with only
    /// `.operator_factory(...)` set (no `.results(..)` / `.common(..)`)
    /// still produces a valid `ActorBuilder`.
    #[test]
    fn builder_accepts_default_forwarders() {
        let name = StageName::unnamed(StageKind::Infer);
        let _ = NvInfer::builder(name, 4)
            .downstream(StageName::unnamed(StageKind::Tracker))
            .operator_factory(|_bx, _cb| unreachable!("not invoked in this test"))
            .build()
            .unwrap();
    }

    /// The three `default_on_*` associated functions slot into the
    /// `NvInferResults` builder's generic hook bounds as-is.
    #[test]
    fn builder_accepts_explicit_defaults() {
        let name = StageName::unnamed(StageKind::Infer);
        let _ = NvInfer::builder(name.clone(), 4)
            .downstream(StageName::unnamed(StageKind::Tracker))
            .operator_factory(|_bx, _cb| unreachable!("not invoked in this test"))
            .results(
                NvInferResults::builder()
                    .on_inference(NvInfer::default_on_inference())
                    .on_source_eos(NvInfer::default_on_source_eos())
                    .on_error(NvInfer::default_on_error())
                    .build(),
            )
            .common(
                NvInferCommon::builder()
                    .stopping(NvInfer::default_stopping())
                    .build(),
            )
            .build()
            .unwrap();
    }

    /// The builder accepts a user-supplied `.common(...)` bundle
    /// that provides a `stopping` closure.  This is a compile-only
    /// verification of the hook bound; the ordering invariant
    /// (built-in `graceful_shutdown` + drain dispatch runs before
    /// the user hook) is covered by end-to-end smoke runs.
    #[test]
    fn builder_accepts_user_stopping() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let flag = Arc::new(AtomicBool::new(false));
        let flag_hook = flag.clone();
        let name = StageName::unnamed(StageKind::Infer);
        let _ = NvInfer::builder(name, 4)
            .downstream(StageName::unnamed(StageKind::Tracker))
            .operator_factory(|_bx, _cb| unreachable!("not invoked in this test"))
            .common(
                NvInferCommon::builder()
                    .stopping(move |_ctx| {
                        flag_hook.store(true, Ordering::SeqCst);
                    })
                    .build(),
            )
            .build()
            .unwrap();
        assert!(!flag.load(Ordering::SeqCst));
    }

    /// `default_on_source_eos` forwards a `PipelineMsg::SourceEos`
    /// carrying the supplied `source_id` via `router.send(..)` and
    /// returns `Ok(Flow::Cont)`.  This is the only place in the
    /// stage where `SourceEos` leaves, so the contract must stay
    /// covered.
    #[test]
    fn default_on_source_eos_forwards_via_router() {
        let hook = NvInfer::default_on_source_eos();
        let (router, rx) = router_with_default_peer();
        let ctx = test_hook_ctx();
        let flow = hook("cam-1", &router, &ctx).expect("default hook is infallible");
        assert_eq!(flow, Flow::Cont);
        match rx.try_recv().expect("SourceEos should be forwarded") {
            PipelineMsg::SourceEos { source_id } => assert_eq!(source_id, "cam-1"),
            other => panic!("expected SourceEos, got {other:?}"),
        }
        assert!(rx.try_recv().is_err(), "exactly one message forwarded");
    }

    /// `default_on_error` classifies every error as
    /// [`ErrorAction::LogAndContinue`] and does not route anything
    /// downstream (the stage emits the log line itself before
    /// invoking the hook).
    #[test]
    fn default_on_error_does_not_route() {
        let hook = NvInfer::default_on_error();
        let (router, rx) = router_with_default_peer();
        let ctx = test_hook_ctx();
        let err = NvInferError::PipelineError("synthetic".to_string());
        let action = hook(&err, &router, &ctx);
        assert_eq!(action, ErrorAction::LogAndContinue);
        assert!(rx.try_recv().is_err(), "errors must not be routed");
    }

    /// Dispatcher escalates [`ErrorAction::Fatal`] via
    /// [`HookCtx::request_stop`] so the actor loop can observe it
    /// through its existing `should_quit` path.
    #[test]
    fn dispatch_output_fatal_error_requests_stop() {
        let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let ctx = HookCtx::new(
            StageName::unnamed(StageKind::Infer),
            Arc::new(Registry::new()),
            Arc::new(crate::shared::SharedStore::new()),
            Arc::clone(&stop_flag),
        );
        let (router, _rx) = router_with_default_peer();
        let on_inference: OnInferenceHook = Arc::new(noop_on_inference);
        let on_source_eos: OnSourceEosHook = Arc::new(noop_on_source_eos);
        let on_error: OnErrorHook = Arc::new(
            |_err: &NvInferError, _r: &Router<PipelineMsg>, _c: &HookCtx| ErrorAction::Fatal,
        );
        dispatch_output(
            OperatorOutput::Error(NvInferError::PipelineError("boom".to_string())),
            &on_inference,
            &on_source_eos,
            &on_error,
            &router,
            &ctx,
        );
        assert!(ctx.stop_requested(), "Fatal must flip the shared stop flag");
    }

    /// Dispatcher translates `Ok(Flow::Stop)` from `on_source_eos`
    /// into a cooperative stop request on the shared flag.
    #[test]
    fn dispatch_output_source_eos_stop_requests_stop() {
        let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let ctx = HookCtx::new(
            StageName::unnamed(StageKind::Infer),
            Arc::new(Registry::new()),
            Arc::new(crate::shared::SharedStore::new()),
            Arc::clone(&stop_flag),
        );
        let (router, _rx) = router_with_default_peer();
        let on_inference: OnInferenceHook = Arc::new(noop_on_inference);
        let on_source_eos: OnSourceEosHook =
            Arc::new(|_sid: &str, _r: &Router<PipelineMsg>, _c: &HookCtx| Ok(Flow::Stop));
        let on_error: OnErrorHook = Arc::new(noop_on_error);
        dispatch_output(
            OperatorOutput::Eos {
                source_id: "cam".to_string(),
            },
            &on_inference,
            &on_source_eos,
            &on_error,
            &router,
            &ctx,
        );
        assert!(ctx.stop_requested());
    }
}
