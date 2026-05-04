//! [`NvTracker`] — multiplexed `PipelineMsg` → `PipelineMsg`
//! stage wrapping an
//! [`NvTrackerBatchingOperator`].
//!
//! Mirror of the [`NvInfer`](super::nvinfer::NvInfer)
//! stage — structurally identical, only the operator and output
//! types differ.
//!
//! # Grouped builder API
//!
//! The stage exposes two hook bundles that match the
//! [`NvInfer`](super::nvinfer::NvInfer) pattern:
//!
//! * [`NvTrackerResults`] — per-variant
//!   [`TrackerOperatorOutput`] hooks (`on_tracking`,
//!   `on_source_eos`, `on_error`).
//! * [`NvTrackerCommon`] — loop-level knobs (`drain_timeout`,
//!   `poll_timeout`) plus the user `stopping` hook.
//!
//! Build the bundles via [`NvTrackerResults::builder`] and
//! [`NvTrackerCommon::builder`]:
//!
//! ```ignore
//! NvTracker::builder(name, 16)
//!     .downstream(picasso)
//!     .batch_formation(my_tracker_batch_formation_cb)
//!     .operator_factory(|bx, batch_formation, cb| { /* ... */ })
//!     .results(
//!         NvTrackerResults::builder()
//!             .on_source_eos(|sid, router, ctx| { /* ... */ Ok(Flow::Cont) })
//!             .build(),
//!     )
//!     .common(
//!         NvTrackerCommon::builder()
//!             .drain_timeout(Duration::from_secs(10))
//!             .build(),
//!     )
//!     .build()?;
//! ```
//!
//! Any field left unset is auto-installed from the matching
//! `NvTracker::default_*` at build time, so a minimal builder
//! without either bundle is a valid configuration: every unset
//! [`NvTrackerResults`] / [`NvTrackerCommon`] field collapses
//! into its default. There is no flat-setter fallback on
//! [`NvTrackerBuilder`]; `results` / `common` are the sole
//! interface for hook and knob configuration.
//!
//! # Default forwarders
//!
//! Every per-variant hook has an out-of-the-box default — if the
//! matching [`NvTrackerResults`] / [`NvTrackerCommon`] field is not
//! explicitly set, [`NvTrackerBuilder::build`] auto-installs it:
//!
//! * [`NvTracker::default_on_tracking`] — for every frame
//!   output, calls
//!   [`TrackerOperatorFrameOutput::apply_to_frame`](deepstream_nvtracker::TrackerOperatorFrameOutput::apply_to_frame)
//!   to reconcile tracker-side updates with the frame's objects, logs
//!   unmatched updates at `warn` level, then forwards
//!   [`PipelineMsg::Deliveries`]
//!   via `router.send(..)`.  Tracker outputs are object/track updates
//!   rather than tensor layers, so no attribute attachment is done.
//! * [`NvTracker::default_on_source_eos`] — forwards
//!   [`PipelineMsg::SourceEos`]
//!   and returns `Ok(Flow::Cont)`.
//! * [`NvTracker::default_on_error`] — returns
//!   [`ErrorAction::LogAndContinue`]; the stage emits the
//!   `error!` log line itself *before* invoking the hook.
//! * [`NvTracker::default_on_stopping`] — a no-op user shutdown hook.
//!   See [`NvTrackerCommonBuilder::stopping`] for the composition
//!   contract (built-in `graceful_shutdown` + drain-dispatch always
//!   run first; the user hook runs after).
//!
//! # Runtime invariant
//!
//! [`NvTracker`] stores its three hooks as non-`Option`
//! `Arc<dyn Fn>`.  The [`NvTrackerResults`] / [`NvTrackerCommon`]
//! bundles carry `Option<...>` fields only in their builders as
//! internal "was the setter called?" markers;
//! [`NvTrackerResultsBuilder::build`] and
//! [`NvTrackerCommonBuilder::build`] always substitute the matching
//! `NvTracker::default_*` before the [`NvTracker`] actor value is
//! constructed.  There is therefore no runtime code path where a
//! [`TrackerOperatorOutput`] variant is dispatched to `None` —
//! every branch (`Tracking`, `Eos`, `Error`) is always handled,
//! whether by a user-supplied closure or by the auto-installed
//! default.

use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use deepstream_nvtracker::{
    NvTrackerBatchingOperator, NvTrackerError, TrackerBatchFormationCallback,
    TrackerOperatorOutput, TrackerOperatorResultCallback, TrackerOperatorTrackingOutput,
};
use parking_lot::Mutex;

use crate::addr::Addr;
use crate::envelope::Envelope;
use crate::envelopes::{BatchDelivery, PipelineMsg, SingleDelivery};
use crate::instrument::{enter_callback_span, push_stage_span, CallbackSpanGuard};
use crate::stage_metrics::PHASE_E2E_START;
use crate::router::Router;
use crate::stage_metrics::{monotonic_ns, stamp_phase, take_phase, PHASE_INFER_START};
use crate::supervisor::StageName;
use crate::{
    Actor, ActorBuilder, BuildCtx, Context, Dispatch, ErrorAction, Flow, Handler, HookCtx,
    ResetStreamPayload, ShutdownPayload, SourceEosPayload,
};

/// Shared reference to the actor's
/// [`NvTrackerBatchingOperator`] wrapped in an
/// [`Arc<parking_lot::Mutex<_>>`] for interior mutability.
///
/// The actor stores one clone; another clone is stashed in every
/// [`NvTrackerHookCtx`] so user code can call
/// [`NvTrackerBatchingOperator::reset_stream`] from inside any
/// user-supplied hook without a self-send through the router.
type SharedOperator = Arc<Mutex<NvTrackerBatchingOperator>>;

/// Tracker-specific hook context — the framework-level
/// [`HookCtx`] (own name, shared-store look-ups, registry
/// resolve) plus a lazily-bound handle to the actor's
/// [`NvTrackerBatchingOperator`] so user hooks can invoke
/// [`reset_stream`](Self::reset_stream) without wiring an extra
/// handle.
///
/// # Composition
///
/// `NvTrackerHookCtx` **contains** a [`HookCtx`] by value and
/// re-exports its accessors (`own_name`, `shared`, `shared_as`,
/// `resolve`).  Use [`as_hook_ctx`](Self::as_hook_ctx) when
/// calling a helper that expects `&HookCtx` directly.
///
/// Cloning is cheap (four `Arc::clone`s — one per field plus one
/// for the composed `HookCtx`).
#[derive(Clone)]
pub struct NvTrackerHookCtx {
    base: HookCtx,
    /// Lazily populated during `NvTrackerBuilder::build` — see the
    /// factory in that function.  Every clone of the hook ctx
    /// observes the same `OnceLock`, so callbacks captured by
    /// the result dispatcher (which must exist BEFORE the
    /// operator) can reach into the operator once it's live.
    operator: Arc<OnceLock<SharedOperator>>,
}

impl NvTrackerHookCtx {
    /// Borrow the composed framework [`HookCtx`].
    pub fn as_hook_ctx(&self) -> &HookCtx {
        &self.base
    }

    /// The tracker actor's registered [`StageName`].  Delegates
    /// to [`HookCtx::own_name`].
    pub fn own_name(&self) -> &StageName {
        self.base.own_name()
    }

    /// Singleton shared-state look-up.  Delegates to
    /// [`HookCtx::shared`].
    pub fn shared<T: Send + Sync + 'static>(&self) -> Option<Arc<T>> {
        self.base.shared::<T>()
    }

    /// Named shared-state look-up.  Delegates to
    /// [`HookCtx::shared_as`].
    pub fn shared_as<T: Send + Sync + 'static>(&self, key: &str) -> Option<Arc<T>> {
        self.base.shared_as::<T>(key)
    }

    /// Resolve the [`Addr<M>`] registered under `peer`.  Delegates
    /// to [`HookCtx::resolve`].
    pub fn resolve<M: Envelope>(&self, peer: &StageName) -> Result<Addr<M>> {
        self.base.resolve::<M>(peer)
    }

    /// Flip the shared cooperative-stop flag so the tracker actor
    /// exits through its normal
    /// [`Actor::stopping`] path.  Delegates to
    /// [`HookCtx::request_stop`].
    pub fn request_stop(&self) {
        self.base.request_stop();
    }

    /// Whether [`Self::request_stop`] has already been called on
    /// this ctx (or any other context sharing the same flag).
    /// Delegates to [`HookCtx::stop_requested`].
    pub fn stop_requested(&self) -> bool {
        self.base.stop_requested()
    }

    /// Request that the tracker drop its per-source state for
    /// `source_id`, propagating the call to
    /// [`NvTrackerBatchingOperator::reset_stream`].
    ///
    /// Returns `Err` if the ctx is not yet bound to a live
    /// operator (i.e. the actor's factory has not completed —
    /// shouldn't happen in normal hook invocations) or if the
    /// operator rejects the reset (e.g. because it has begun
    /// draining).  Typical use:
    ///
    /// ```ignore
    /// .on_source_eos(|source_id, router, ctx| {
    ///     if let Err(e) = ctx.reset_stream(source_id) {
    ///         log::warn!("reset_stream({source_id}) failed: {e}");
    ///     }
    ///     router.send(PipelineMsg::SourceEos { source_id: source_id.to_string() });
    /// })
    /// ```
    ///
    /// Serialised against actor-thread `add_frame` /
    /// `flush_idle` calls via the underlying
    /// [`parking_lot::Mutex`].  Contention is negligible in
    /// practice — reset is rare and the critical section is a
    /// handful of dereferences.
    pub fn reset_stream(&self, source_id: &str) -> Result<()> {
        let op = self
            .operator
            .get()
            .ok_or_else(|| anyhow!("tracker hook ctx not bound; factory has not completed"))?;
        op.lock()
            .reset_stream(source_id)
            .map_err(|e| anyhow!("reset_stream({source_id}) failed: {e}"))
    }
}

/// Default graceful-shutdown drain timeout.
pub const DEFAULT_DRAIN_TIMEOUT: Duration = Duration::from_secs(30);
/// Default inbox receive-poll cadence.
pub const DEFAULT_POLL_TIMEOUT: Duration = Duration::from_millis(100);

/// Hook fired for every
/// [`TrackerOperatorOutput::Tracking`]
/// payload.  Receives the
/// [`TrackerOperatorTrackingOutput`]
/// (owning the sealed deliveries), the stage's
/// [`Router<PipelineMsg>`], and an [`NvTrackerHookCtx`] for
/// shared-state / registry / reset access.  Typical use:
/// reconcile per-frame tracking updates with the detections via
/// [`TrackerOperatorFrameOutput::apply_to_frame`](deepstream_nvtracker::TrackerOperatorFrameOutput::apply_to_frame),
/// then send
/// [`PipelineMsg::Deliveries`]
/// downstream.
///
/// See [`NvTracker::default_on_tracking`] for the ready-made
/// "apply tracking + forward" default.
pub type OnTrackingHook = Arc<
    dyn Fn(&NvTrackerHookCtx, &Router<PipelineMsg>, TrackerOperatorTrackingOutput)
        + Send
        + Sync
        + 'static,
>;

/// Hook fired for every
/// [`TrackerOperatorOutput::Eos`]
/// payload.  Receives the source id, the router, and the
/// [`NvTrackerHookCtx`] — call
/// [`NvTrackerHookCtx::reset_stream`] from here to clear the
/// tracker's per-source state without wiring a separate handle.
///
/// Return semantics — unified with every stage's
/// `on_source_eos`:
///
/// * `Ok(Flow::Cont)` — keep the actor loop running (default).
/// * `Ok(Flow::Stop)` — ask the loop to wind down cooperatively
///   (stage requests a stop via
///   [`NvTrackerHookCtx::request_stop`]).
/// * `Err(_)` — log the error and request a cooperative stop;
///   equivalent to `Ok(Flow::Stop)` modulo the `error!` log line
///   emitted by the stage.
pub type OnSourceEosHook = Arc<
    dyn Fn(&NvTrackerHookCtx, &Router<PipelineMsg>, &str) -> Result<Flow> + Send + Sync + 'static,
>;

/// Hook fired for every
/// [`TrackerOperatorOutput::Error`]
/// payload.  Receives the operator error, the router, and the
/// [`NvTrackerHookCtx`].
///
/// Return semantics — unified with every stage's `on_error`
/// hook via [`ErrorAction`]:
///
/// * [`ErrorAction::LogAndContinue`] — keep processing subsequent
///   outputs.  Default.
/// * [`ErrorAction::Fatal`] — stage escalates: aborts the
///   default sink and requests a cooperative stop through
///   [`NvTrackerHookCtx::request_stop`].
/// * [`ErrorAction::Swallow`] — same as `LogAndContinue` for the
///   tracker actor; provided for vocabulary parity.
///
/// The stage always emits an `error!` log line (stage-prefixed)
/// *before* invoking the hook.
pub type OnErrorHook = Arc<
    dyn Fn(&NvTrackerHookCtx, &Router<PipelineMsg>, &NvTrackerError) -> ErrorAction
        + Send
        + Sync
        + 'static,
>;

/// User shutdown hook invoked from [`Actor::stopping`] *after* the
/// stage's built-in cleanup (guarded
/// [`NvTrackerBatchingOperator::graceful_shutdown`] and drained
/// output dispatch) has completed.
///
/// Runs on the actor thread with full access to the [`Context`].
/// Ideal for final metrics flushes, bespoke log lines, or custom
/// bookkeeping that must observe the drained operator state.  The
/// stage's load-bearing cleanup (drain + fan drained outputs
/// through the per-variant hooks) cannot be skipped — users that
/// need to replace it entirely should implement [`Actor`] directly
/// on a bespoke struct.
pub type OnStoppingHook = Box<dyn FnMut(&mut Context<NvTracker>) + Send + 'static>;

/// Factory closure — receives the framework-wrapped
/// [`TrackerBatchFormationCallback`] (whose invocation is timed
/// as the per-batch preproc latency, see
/// [`NvTrackerBuilder::batch_formation`]) plus the wired
/// [`TrackerOperatorResultCallback`], and returns a ready
/// [`NvTrackerBatchingOperator`].
pub type NvTrackerOperatorFactory = Box<
    dyn FnOnce(
            &BuildCtx,
            TrackerBatchFormationCallback,
            TrackerOperatorResultCallback,
        ) -> Result<NvTrackerBatchingOperator>
        + Send,
>;

/// `PipelineMsg` → `PipelineMsg` actor stage wrapping an
/// [`NvTrackerBatchingOperator`].
pub struct NvTracker {
    operator: SharedOperator,
    router: Router<PipelineMsg>,
    hook_ctx: NvTrackerHookCtx,
    on_tracking: OnTrackingHook,
    on_source_eos: OnSourceEosHook,
    on_error: OnErrorHook,
    on_stopping: OnStoppingHook,
    drain_timeout: Duration,
    poll_timeout: Duration,
    drain_done: bool,
}

impl NvTracker {
    /// Start a fluent builder for a tracker registered under `name`
    /// with inbox capacity `capacity`.
    pub fn builder(name: StageName, capacity: usize) -> NvTrackerBuilder {
        NvTrackerBuilder::new(name, capacity)
    }

    /// Default `on_tracking` forwarder.
    ///
    /// For every per-frame output in `tracking.frames()`, calls
    /// [`TrackerOperatorFrameOutput::apply_to_frame`](deepstream_nvtracker::TrackerOperatorFrameOutput::apply_to_frame)
    /// and logs unmatched updates at `warn` level.  Then forwards
    /// [`PipelineMsg::Deliveries`]
    /// downstream via `router.send(..)`.
    ///
    /// No tensor-attribute attachment is performed — tracker outputs
    /// are object/track updates rather than tensor layers, so the
    /// "raw tensors as attributes" convention from
    /// [`NvInfer::default_on_inference`](super::nvinfer::NvInfer::default_on_inference)
    /// does not apply here.
    pub fn default_on_tracking(
    ) -> impl Fn(&NvTrackerHookCtx, &Router<PipelineMsg>, TrackerOperatorTrackingOutput)
           + Send
           + Sync
           + 'static {
        |ctx, router, mut tracking| {
            let ns = ctx.own_name();
            for frame_output in tracking.frames() {
                match frame_output.apply_to_frame() {
                    Ok(unmatched) => {
                        for tu in &unmatched {
                            log::warn!("[{ns}] unmatched {tu}");
                        }
                    }
                    Err(err) => log::warn!("[{ns}] apply_to_frame failed: {err}"),
                }
            }
            if let Some(sealed) = tracking.take_deliveries() {
                drop(tracking);
                if !router.send(PipelineMsg::deliveries(sealed)) {
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
    /// [`TrackerOperatorOutput::Eos`]
    /// fires strictly after the last
    /// [`TrackerOperatorOutput::Tracking`]
    /// for the same `source_id`.
    pub fn default_on_source_eos(
    ) -> impl Fn(&NvTrackerHookCtx, &Router<PipelineMsg>, &str) -> Result<Flow> + Send + Sync + 'static
    {
        |_ctx, router, source_id| {
            log::info!("TrackerOperatorOutput::Eos for source_id={source_id}; propagating");
            if !router.send(PipelineMsg::source_eos(source_id)) {
                log::warn!("downstream closed; dropping SourceEos({source_id})");
            }
            Ok(Flow::Cont)
        }
    }

    /// Default `on_error` classifier.  Returns
    /// [`ErrorAction::LogAndContinue`] unconditionally — a
    /// misbehaving tracker should not tear the pipeline down.
    /// The stage emits a single `error!` line prefixed by the
    /// stage name *before* the hook fires, so this classifier
    /// has no I/O of its own.
    pub fn default_on_error(
    ) -> impl Fn(&NvTrackerHookCtx, &Router<PipelineMsg>, &NvTrackerError) -> ErrorAction
           + Send
           + Sync
           + 'static {
        |_ctx, _router, _err| ErrorAction::LogAndContinue
    }

    /// Default user shutdown hook — a no-op.  The stage's own
    /// [`Actor::stopping`] body always runs the guarded
    /// [`NvTrackerBatchingOperator::graceful_shutdown`] drain and
    /// fans the drained outputs through the per-variant hooks
    /// before this hook fires, so omitting `.stopping(...)` simply
    /// means "don't add any extra cleanup on top of the built-in
    /// drain".
    pub fn default_on_stopping() -> impl FnMut(&mut Context<NvTracker>) + Send + 'static {
        |_ctx| {}
    }
}

impl Actor for NvTracker {
    type Msg = PipelineMsg;

    fn handle(&mut self, msg: PipelineMsg, ctx: &mut Context<Self>) -> Result<Flow> {
        msg.dispatch(self, ctx)
    }

    fn poll_timeout(&self) -> Duration {
        self.poll_timeout
    }

    fn started(&mut self, ctx: &mut Context<Self>) -> Result<()> {
        log::info!("[{}] nvtracker started", ctx.own_name());
        Ok(())
    }

    fn on_tick(&mut self, ctx: &mut Context<Self>) -> Result<Flow> {
        if let Err(e) = self.operator.lock().flush_idle() {
            log::warn!("[{}] flush_idle failed: {e}", ctx.own_name());
        }
        Ok(Flow::Cont)
    }

    fn stopping(&mut self, ctx: &mut Context<Self>) {
        if self.drain_done {
            return;
        }
        let result = self.operator.lock().graceful_shutdown(self.drain_timeout);
        match result {
            Ok(drained) => {
                log::info!(
                    "[{}] drained {} remaining outputs",
                    ctx.own_name(),
                    drained.len()
                );
                for out in drained {
                    dispatch_output(
                        out,
                        &self.on_tracking,
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
        log::info!("[{}] nvtracker stopping", ctx.own_name());
        (self.on_stopping)(ctx);
    }
}

impl Handler<SingleDelivery> for NvTracker {
    fn handle(&mut self, msg: SingleDelivery, ctx: &mut Context<Self>) -> Result<Flow> {
        let e2e_start_ns = monotonic_ns();
        let pairs = PipelineMsg::delivery(msg.0).into_pairs();
        let stage = ctx.own_name().to_string();
        let pipeline_name = ctx.pipeline_name().to_string();
        // Stage span and e2e_start — see [`super::nvinfer::Handler`]
        // for the narrative.  Preproc latency is **not** measured
        // here — the actor body just stamps + enqueues; the real
        // preprocessing duration is the user's batch-formation
        // callback (registered via
        // [`NvTrackerBuilder::batch_formation`]) running on the
        // operator's worker thread, which the framework wraps
        // with timing internally.
        for (frame, _) in &pairs {
            stamp_phase(frame, &stage, PHASE_E2E_START, e2e_start_ns);
            push_stage_span(frame, "tracker.stage", &pipeline_name, &stage);
        }
        submit_pairs(&self.operator, pairs, ctx.own_name(), &stage)
    }
}

impl Handler<BatchDelivery> for NvTracker {
    fn handle(&mut self, msg: BatchDelivery, ctx: &mut Context<Self>) -> Result<Flow> {
        let e2e_start_ns = monotonic_ns();
        let pairs = msg.0.unseal();
        let stage = ctx.own_name().to_string();
        let pipeline_name = ctx.pipeline_name().to_string();
        for (frame, _) in &pairs {
            stamp_phase(frame, &stage, PHASE_E2E_START, e2e_start_ns);
            push_stage_span(frame, "tracker.stage", &pipeline_name, &stage);
        }
        submit_pairs(&self.operator, pairs, ctx.own_name(), &stage)
    }
}

impl Handler<SourceEosPayload> for NvTracker {
    fn handle(&mut self, msg: SourceEosPayload, ctx: &mut Context<Self>) -> Result<Flow> {
        log::info!(
            "[{}] SourceEos {}: initiating operator drain",
            ctx.own_name(),
            msg.source_id
        );
        if let Err(e) = self.operator.lock().send_eos(&msg.source_id) {
            log::warn!(
                "[{}] send_eos({}) failed: {e}",
                ctx.own_name(),
                msg.source_id
            );
        }
        Ok(Flow::Cont)
    }
}

impl Handler<ShutdownPayload> for NvTracker {}

impl Handler<ResetStreamPayload> for NvTracker {
    /// Forward the reset request to
    /// [`NvTrackerBatchingOperator::reset_stream`].  Errors are
    /// logged and swallowed — a failed reset (e.g. because the
    /// operator is already draining) must not tear the actor down.
    fn handle(&mut self, msg: ResetStreamPayload, ctx: &mut Context<Self>) -> Result<Flow> {
        if let Err(e) = self.operator.lock().reset_stream(&msg.source_id) {
            log::warn!(
                "[{}] reset_stream({}) failed: {e}",
                ctx.own_name(),
                msg.source_id
            );
        } else {
            log::info!(
                "[{}] reset_stream({}) dispatched",
                ctx.own_name(),
                msg.source_id
            );
        }
        Ok(Flow::Cont)
    }
}

fn submit_pairs(
    operator: &SharedOperator,
    pairs: Vec<(
        savant_core::primitives::frame::VideoFrame,
        deepstream_buffers::SharedBuffer,
    )>,
    stage: &StageName,
    stage_str: &str,
) -> Result<Flow> {
    let op = operator.lock();
    for (frame, buffer) in pairs {
        // Mark the moment the frame leaves the actor thread for the
        // operator queue.  The result callback reads this stamp
        // back via `take_phase` to compute the per-frame tracker
        // turnaround duration cross-thread.
        stamp_phase(&frame, stage_str, PHASE_INFER_START, monotonic_ns());
        if let Err(e) = op.add_frame(frame, buffer) {
            log::error!("[{stage}] add_frame failed: {e}");
            return Err(anyhow!("track add_frame: {e}"));
        }
    }
    Ok(Flow::Cont)
}

/// Classify and dispatch a single [`TrackerOperatorOutput`]
/// through the stage's three per-variant hooks.
///
/// Per the unified hook semantics:
///
/// * `Tracking` → fire the forwarder; no return value inspected.
/// * `Eos` → fire `on_source_eos` and interpret its
///   [`Result<Flow>`]: `Ok(Flow::Stop)` or `Err(_)` request a
///   cooperative shutdown via
///   [`NvTrackerHookCtx::request_stop`]; `Err(_)` is additionally
///   logged at `error!`.
/// * `Error` → the stage emits a stage-prefixed `error!` line,
///   then dispatches to `on_error` and interprets its
///   [`ErrorAction`]: `Fatal` aborts the default downstream sink
///   (if any) and requests a cooperative shutdown; `LogAndContinue`
///   and `Swallow` are equivalent for the tracker actor.
fn dispatch_output(
    out: TrackerOperatorOutput,
    on_tracking: &OnTrackingHook,
    on_source_eos: &OnSourceEosHook,
    on_error: &OnErrorHook,
    router: &Router<PipelineMsg>,
    ctx: &NvTrackerHookCtx,
) {
    match out {
        TrackerOperatorOutput::Tracking(tracking) => {
            // Per-frame work — same shape as nvinfer's Inference
            // arm.  Push an `on_tracking` callback span onto each
            // frame's span stack before invoking the user hook;
            // pop after the hook returns.  See
            // [`super::nvinfer::dispatch_output`] for the narrative.
            let now_ns = monotonic_ns();
            let stage_str = ctx.own_name().to_string();
            let pipeline_name = ctx.as_hook_ctx().pipeline_name().to_string();
            // Cheap Arc clones so we can read `e2e_start` after
            // `tracking` is consumed by the user hook.
            let frames_for_e2e: Vec<savant_core::primitives::frame::VideoFrame> =
                tracking.frames().iter().map(|fo| fo.frame.clone()).collect();
            // Drop order (reverse declaration): callback guards
            // pop `on_tracking` first, stage guards then pop the
            // `tracker.stage` pushed in `Handler::handle`.
            let _stage_pops: Vec<CallbackSpanGuard> = frames_for_e2e
                .iter()
                .map(|f| CallbackSpanGuard::pop_at_drop(f.clone()))
                .collect();
            let _callback_spans: Vec<CallbackSpanGuard> = tracking
                .frames()
                .iter()
                .map(|fo| {
                    if let Some(start_ns) =
                        take_phase(&fo.frame, &stage_str, PHASE_INFER_START)
                    {
                        let delta = (now_ns - start_ns).max(0) as u64;
                        ctx.as_hook_ctx()
                            .stage_metrics()
                            .record_inference_latency(Duration::from_nanos(delta));
                    }
                    enter_callback_span(&fo.frame, "on_tracking", &pipeline_name, &stage_str)
                })
                .collect();
            let postproc_start = Instant::now();
            on_tracking(ctx, router, tracking);
            ctx.as_hook_ctx()
                .stage_metrics()
                .record_postproc_latency(postproc_start.elapsed());
            // Per-frame e2e: handle entry → here.
            let e2e_now_ns = monotonic_ns();
            for frame in &frames_for_e2e {
                if let Some(start_ns) = take_phase(frame, &stage_str, PHASE_E2E_START) {
                    let delta = (e2e_now_ns - start_ns).max(0) as u64;
                    ctx.as_hook_ctx()
                        .stage_metrics()
                        .record_e2e_latency(Duration::from_nanos(delta));
                }
            }
        }
        TrackerOperatorOutput::Eos { source_id } => {
            // No frame on EOS — no span.
            match on_source_eos(ctx, router, &source_id) {
                Ok(Flow::Cont) => {}
                Ok(Flow::Stop) => {
                    log::info!(
                        "[{}] on_source_eos({source_id}) requested stop",
                        ctx.own_name()
                    );
                    ctx.request_stop();
                }
                Err(e) => {
                    log::error!(
                        "[{}] on_source_eos({source_id}) returned error: {e}; requesting stop",
                        ctx.own_name()
                    );
                    ctx.request_stop();
                }
            }
        }
        TrackerOperatorOutput::Error(err) => {
            log::error!("[{}] operator error: {err}", ctx.own_name());
            // No frame on an operator error — no span.
            match on_error(ctx, router, &err) {
                ErrorAction::LogAndContinue | ErrorAction::Swallow => {}
                ErrorAction::Fatal => {
                    log::error!(
                        "[{}] on_error classified fatal; aborting default sink and requesting stop",
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

/// Per-variant [`TrackerOperatorOutput`] hook bundle — one branch
/// per [`TrackerOperatorOutput`] variant.
///
/// Built through [`NvTrackerResults::builder`] and handed to
/// [`NvTrackerBuilder::results`].  Omitted branches auto-install
/// the matching `NvTracker::default_on_*` at build time, so the
/// runtime invariant "every `TrackerOperatorOutput` variant is
/// always dispatched to a user-supplied or auto-installed hook"
/// holds regardless of how much the user overrides.
pub struct NvTrackerResults {
    on_tracking: OnTrackingHook,
    on_source_eos: OnSourceEosHook,
    on_error: OnErrorHook,
}

impl NvTrackerResults {
    /// Start a builder that auto-installs every default on
    /// [`NvTrackerResultsBuilder::build`].
    pub fn builder() -> NvTrackerResultsBuilder {
        NvTrackerResultsBuilder::new()
    }
}

impl Default for NvTrackerResults {
    /// Every branch wired to its matching `NvTracker::default_on_*`.
    fn default() -> Self {
        NvTrackerResultsBuilder::new().build()
    }
}

/// Fluent builder for [`NvTrackerResults`] — one setter per
/// [`TrackerOperatorOutput`] variant.
pub struct NvTrackerResultsBuilder {
    on_tracking: Option<OnTrackingHook>,
    on_source_eos: Option<OnSourceEosHook>,
    on_error: Option<OnErrorHook>,
}

impl NvTrackerResultsBuilder {
    /// Empty bundle — every hook defaults to its
    /// `NvTracker::default_on_*` equivalent at
    /// [`build`](Self::build) time.
    pub fn new() -> Self {
        Self {
            on_tracking: None,
            on_source_eos: None,
            on_error: None,
        }
    }

    /// Override the `on_tracking` hook.  Omitting this setter is
    /// equivalent to calling
    /// `.on_tracking(NvTracker::default_on_tracking())` — reconcile
    /// per-frame updates via
    /// [`TrackerOperatorFrameOutput::apply_to_frame`](deepstream_nvtracker::TrackerOperatorFrameOutput::apply_to_frame)
    /// then forward
    /// [`PipelineMsg::Deliveries`]
    /// via `router.send(..)`.
    pub fn on_tracking<F>(mut self, f: F) -> Self
    where
        F: Fn(&NvTrackerHookCtx, &Router<PipelineMsg>, TrackerOperatorTrackingOutput)
            + Send
            + Sync
            + 'static,
    {
        self.on_tracking = Some(Arc::new(f));
        self
    }

    /// Override the `on_source_eos` hook.  Omitting this setter is
    /// equivalent to calling
    /// `.on_source_eos(NvTracker::default_on_source_eos())` —
    /// forward [`PipelineMsg::SourceEos`]
    /// downstream and return `Ok(Flow::Cont)`.
    ///
    /// Return `Ok(Flow::Stop)` to request cooperative shutdown of
    /// the stage once this EOS is processed; `Err(_)` logs the
    /// error and also requests shutdown.  Call
    /// [`NvTrackerHookCtx::reset_stream`] from inside the closure
    /// to clear per-source state without a separate handle.
    pub fn on_source_eos<F>(mut self, f: F) -> Self
    where
        F: Fn(&NvTrackerHookCtx, &Router<PipelineMsg>, &str) -> Result<Flow>
            + Send
            + Sync
            + 'static,
    {
        self.on_source_eos = Some(Arc::new(f));
        self
    }

    /// Override the `on_error` hook.  Omitting this setter is
    /// equivalent to calling
    /// `.on_error(NvTracker::default_on_error())` — classify every
    /// error as [`ErrorAction::LogAndContinue`].
    ///
    /// The stage always emits an `error!` log line
    /// (stage-prefixed) *before* invoking this hook; the hook only
    /// classifies.  Return [`ErrorAction::Fatal`] to abort the
    /// default sink and request cooperative shutdown.
    pub fn on_error<F>(mut self, f: F) -> Self
    where
        F: Fn(&NvTrackerHookCtx, &Router<PipelineMsg>, &NvTrackerError) -> ErrorAction
            + Send
            + Sync
            + 'static,
    {
        self.on_error = Some(Arc::new(f));
        self
    }

    /// Finalise the bundle.  Every omitted branch collapses into
    /// the matching `NvTracker::default_on_*` so the resulting
    /// [`NvTrackerResults`] is always complete.
    pub fn build(self) -> NvTrackerResults {
        let NvTrackerResultsBuilder {
            on_tracking,
            on_source_eos,
            on_error,
        } = self;
        NvTrackerResults {
            on_tracking: on_tracking.unwrap_or_else(|| Arc::new(NvTracker::default_on_tracking())),
            on_source_eos: on_source_eos
                .unwrap_or_else(|| Arc::new(NvTracker::default_on_source_eos())),
            on_error: on_error.unwrap_or_else(|| Arc::new(NvTracker::default_on_error())),
        }
    }
}

impl Default for NvTrackerResultsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Loop-level common knobs + user shutdown hook.  Built through
/// [`NvTrackerCommon::builder`] and handed to
/// [`NvTrackerBuilder::common`].
pub struct NvTrackerCommon {
    drain_timeout: Duration,
    poll_timeout: Duration,
    on_stopping: OnStoppingHook,
}

impl NvTrackerCommon {
    /// Start a builder seeded with
    /// [`DEFAULT_DRAIN_TIMEOUT`] / [`DEFAULT_POLL_TIMEOUT`] / no-op
    /// stopping hook.
    pub fn builder() -> NvTrackerCommonBuilder {
        NvTrackerCommonBuilder::new()
    }
}

impl Default for NvTrackerCommon {
    /// `DEFAULT_DRAIN_TIMEOUT`, `DEFAULT_POLL_TIMEOUT`, and a no-op
    /// `stopping` hook.
    fn default() -> Self {
        NvTrackerCommonBuilder::new().build()
    }
}

/// Fluent builder for [`NvTrackerCommon`].
pub struct NvTrackerCommonBuilder {
    drain_timeout: Option<Duration>,
    poll_timeout: Option<Duration>,
    on_stopping: Option<OnStoppingHook>,
}

impl NvTrackerCommonBuilder {
    /// Empty bundle — build-time defaults are
    /// [`DEFAULT_DRAIN_TIMEOUT`] / [`DEFAULT_POLL_TIMEOUT`] /
    /// no-op stopping hook.
    pub fn new() -> Self {
        Self {
            drain_timeout: None,
            poll_timeout: None,
            on_stopping: None,
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
    /// [`NvTrackerBatchingOperator::graceful_shutdown`] + drained
    /// output dispatch) has completed.
    ///
    /// Runs on the actor thread with full access to the
    /// [`Context`].  Use for final metrics flushes, bespoke log
    /// lines, or custom bookkeeping that must observe the drained
    /// operator state.  The built-in cleanup is **load-bearing**
    /// and always runs first; it cannot be skipped through this
    /// hook.
    pub fn on_stopping<F>(mut self, f: F) -> Self
    where
        F: FnMut(&mut Context<NvTracker>) + Send + 'static,
    {
        self.on_stopping = Some(Box::new(f));
        self
    }

    /// Finalise the bundle, substituting defaults for every
    /// omitted setter.
    pub fn build(self) -> NvTrackerCommon {
        let NvTrackerCommonBuilder {
            drain_timeout,
            poll_timeout,
            on_stopping,
        } = self;
        NvTrackerCommon {
            drain_timeout: drain_timeout.unwrap_or(DEFAULT_DRAIN_TIMEOUT),
            poll_timeout: poll_timeout.unwrap_or(DEFAULT_POLL_TIMEOUT),
            on_stopping: on_stopping.unwrap_or_else(|| Box::new(NvTracker::default_on_stopping())),
        }
    }
}

impl Default for NvTrackerCommonBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Fluent builder for [`NvTracker`].
///
/// The builder only exposes wiring-level configuration at the top
/// level (`downstream`, `operator_factory`).  Per-variant
/// operator-output hooks live on [`NvTrackerResults`]; loop-level
/// knobs and the user shutdown hook live on [`NvTrackerCommon`].
/// Install them via [`NvTrackerBuilder::results`] and
/// [`NvTrackerBuilder::common`].
pub struct NvTrackerBuilder {
    name: StageName,
    capacity: usize,
    downstream: Option<StageName>,
    batch_formation: Option<TrackerBatchFormationCallback>,
    operator_factory: Option<NvTrackerOperatorFactory>,
    results: Option<NvTrackerResults>,
    common: Option<NvTrackerCommon>,
}

impl NvTrackerBuilder {
    /// Start a builder for a tracker actor registered under `name`
    /// with inbox capacity `capacity`.
    ///
    /// Every per-variant hook defaults to its
    /// `NvTracker::default_on_*` equivalent; loop-level knobs
    /// default to [`DEFAULT_DRAIN_TIMEOUT`] /
    /// [`DEFAULT_POLL_TIMEOUT`].  The user only needs to supply
    /// `batch_formation` and `operator_factory`; `results` /
    /// `common` are optional and composable.
    pub fn new(name: StageName, capacity: usize) -> Self {
        Self {
            name,
            capacity,
            downstream: None,
            batch_formation: None,
            operator_factory: None,
            results: None,
            common: None,
        }
    }

    /// Optional default peer for `router.send(msg)` inside the
    /// per-variant hooks (typically picasso).  When omitted, the
    /// hooks must use `router.send_to(&peer, msg)` for explicit
    /// name-based routing; a default-less `router.send(msg)` silently
    /// drops the message and logs once.
    pub fn downstream(mut self, peer: StageName) -> Self {
        self.downstream = Some(peer);
        self
    }

    /// Required: per-batch formation callback handed to the
    /// underlying [`NvTrackerBatchingOperator`].  The framework
    /// wraps the supplied callback with a per-batch
    /// [`StageMetrics::record_preproc_latency`](crate::stage_metrics::StageMetrics::record_preproc_latency)
    /// timing shim before passing it through to the
    /// [`operator_factory`](Self::operator_factory) — same
    /// semantics as
    /// [`NvInferBuilder::batch_formation`](super::nvinfer::NvInferBuilder::batch_formation).
    pub fn batch_formation(mut self, cb: TrackerBatchFormationCallback) -> Self {
        self.batch_formation = Some(cb);
        self
    }

    /// Required: factory that builds the
    /// [`NvTrackerBatchingOperator`] given the framework-wrapped
    /// batch-formation callback (see
    /// [`batch_formation`](Self::batch_formation)) and the wired
    /// result callback.
    pub fn operator_factory<F>(mut self, f: F) -> Self
    where
        F: FnOnce(
                &BuildCtx,
                TrackerBatchFormationCallback,
                TrackerOperatorResultCallback,
            ) -> Result<NvTrackerBatchingOperator>
            + Send
            + 'static,
    {
        self.operator_factory = Some(Box::new(f));
        self
    }

    /// Install a [`NvTrackerResults`] bundle — one branch per
    /// [`TrackerOperatorOutput`] variant (`tracking`, `source_eos`,
    /// `error`).  Omitting this call is equivalent to
    /// `.results(NvTrackerResults::default())`, which wires every
    /// branch to its matching `NvTracker::default_on_*`.
    pub fn results(mut self, r: NvTrackerResults) -> Self {
        self.results = Some(r);
        self
    }

    /// Install a [`NvTrackerCommon`] bundle — loop-level knobs
    /// (`drain_timeout`, `poll_timeout`) plus the user `stopping`
    /// hook.  Omitting this call is equivalent to
    /// `.common(NvTrackerCommon::default())`.
    pub fn common(mut self, c: NvTrackerCommon) -> Self {
        self.common = Some(c);
        self
    }

    /// Finalise the stage and obtain the Layer-A
    /// [`ActorBuilder<NvTracker>`].
    ///
    /// # Errors
    ///
    /// Returns `Err` if `batch_formation` or `operator_factory`
    /// is missing.  All three per-variant hooks and all loop
    /// knobs have defaults auto-installed when the matching
    /// `results` / `common` bundle is omitted.  `downstream` is
    /// optional — callers that route via [`Router::send_to`]
    /// exclusively may omit it.
    pub fn build(self) -> Result<ActorBuilder<NvTracker>> {
        let NvTrackerBuilder {
            name,
            capacity,
            downstream,
            batch_formation,
            operator_factory,
            results,
            common,
        } = self;
        let batch_formation =
            batch_formation.ok_or_else(|| anyhow!("NvTracker: missing batch_formation"))?;
        let operator_factory =
            operator_factory.ok_or_else(|| anyhow!("NvTracker: missing operator_factory"))?;
        let NvTrackerResults {
            on_tracking,
            on_source_eos,
            on_error,
        } = results.unwrap_or_default();
        let NvTrackerCommon {
            drain_timeout,
            poll_timeout,
            on_stopping,
        } = common.unwrap_or_default();
        Ok(ActorBuilder::new(name.clone(), capacity)
            .poll_timeout(poll_timeout)
            .factory(move |bx| {
                let router: Router<PipelineMsg> = bx.router(downstream.as_ref())?;
                // The hook ctx holds a lazily-populated reference to
                // the operator via `Arc<OnceLock<SharedOperator>>`.
                // We build and clone it BEFORE the operator exists
                // (so it can be captured into the result callback),
                // and populate the `OnceLock` right after
                // `operator_factory` returns.  Every clone of the
                // ctx — the one threaded into the result callback,
                // the one stored on the actor for drain-dispatch,
                // and any future clones taken inside
                // `Actor::stopping` — observes the same binding.
                let operator_slot: Arc<OnceLock<SharedOperator>> = Arc::new(OnceLock::new());
                let hook_ctx = NvTrackerHookCtx {
                    base: bx.hook_ctx(),
                    operator: Arc::clone(&operator_slot),
                };
                let router_for_cb = router.clone();
                let on_tracking_cb = on_tracking.clone();
                let on_source_eos_cb = on_source_eos.clone();
                let on_error_cb = on_error.clone();
                let hook_ctx_cb = hook_ctx.clone();
                let result_cb: TrackerOperatorResultCallback = Box::new(move |out| {
                    dispatch_output(
                        out,
                        &on_tracking_cb,
                        &on_source_eos_cb,
                        &on_error_cb,
                        &router_for_cb,
                        &hook_ctx_cb,
                    );
                });
                // Wrap the user's batch-formation callback with
                // preproc-latency timing.  Same shape as the
                // wrapper in `NvInferBuilder::build`.
                let preproc_metrics = hook_ctx.as_hook_ctx().stage_metrics().clone();
                let user_batch_formation = batch_formation;
                let wrapped_batch_formation: TrackerBatchFormationCallback =
                    Arc::new(move |frames| {
                        let start = Instant::now();
                        let result = (user_batch_formation)(frames);
                        preproc_metrics.record_preproc_latency(start.elapsed());
                        result
                    });
                let operator =
                    (operator_factory)(bx, wrapped_batch_formation, result_cb)?;
                let operator: SharedOperator = Arc::new(Mutex::new(operator));
                let _ = operator_slot.set(Arc::clone(&operator));
                Ok(NvTracker {
                    operator,
                    router,
                    hook_ctx,
                    on_tracking,
                    on_source_eos,
                    on_error,
                    on_stopping,
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

    fn noop_batch_formation() -> TrackerBatchFormationCallback {
        Arc::new(|_| deepstream_nvtracker::TrackerBatchFormationResult {
            ids: Vec::new(),
            rois: Vec::new(),
        })
    }

    fn noop_on_tracking(
        _: &NvTrackerHookCtx,
        _: &Router<PipelineMsg>,
        _: TrackerOperatorTrackingOutput,
    ) {
    }
    fn noop_on_source_eos(_: &NvTrackerHookCtx, _: &Router<PipelineMsg>, _: &str) -> Result<Flow> {
        Ok(Flow::Cont)
    }
    fn noop_on_error(
        _: &NvTrackerHookCtx,
        _: &Router<PipelineMsg>,
        _: &NvTrackerError,
    ) -> ErrorAction {
        ErrorAction::LogAndContinue
    }

    /// Build an unbound [`NvTrackerHookCtx`] for the standalone
    /// hook-behaviour tests.  `reset_stream` on this ctx always
    /// returns `Err` since no operator has been wired.
    fn test_hook_ctx() -> NvTrackerHookCtx {
        NvTrackerHookCtx {
            base: HookCtx::new(
                StageName::unnamed(StageKind::Tracker),
                Arc::from("test"),
                Arc::new(Registry::new()),
                Arc::new(crate::shared::SharedStore::new()),
                Arc::new(std::sync::atomic::AtomicBool::new(false)),
                crate::stage_metrics::StageMetrics::new("tracker-test"),
            ),
            operator: Arc::new(OnceLock::new()),
        }
    }

    /// Build a `Router<PipelineMsg>` whose default peer is a freshly
    /// registered `PipelineMsg` inbox.  Returns the router and the
    /// receiver so tests can observe forwarded messages.
    fn router_with_default_peer() -> (Router<PipelineMsg>, Receiver<PipelineMsg>) {
        let peer = StageName::unnamed(StageKind::Render);
        let (tx, rx) = bounded::<PipelineMsg>(4);
        let mut reg = Registry::new();
        reg.insert::<PipelineMsg>(peer.clone(), Addr::new(peer.clone(), tx));
        let addr: Addr<PipelineMsg> = reg.get::<PipelineMsg>(&peer).unwrap();
        let owner = StageName::unnamed(StageKind::Tracker);
        let default = OperatorSink::new(owner.clone(), addr);
        let router = Router::new(
            owner,
            Arc::new(reg),
            Some(default),
            crate::stage_metrics::StageMetrics::new("tracker-test"),
        );
        (router, rx)
    }

    #[test]
    fn builder_requires_batch_formation() {
        let name = StageName::unnamed(StageKind::Tracker);
        let err = NvTracker::builder(name.clone(), 4).build().err().unwrap();
        assert!(err.to_string().contains("missing batch_formation"));
    }

    #[test]
    fn builder_requires_operator_factory() {
        let name = StageName::unnamed(StageKind::Tracker);
        let err = NvTracker::builder(name.clone(), 4)
            .batch_formation(noop_batch_formation())
            .build()
            .err()
            .unwrap();
        assert!(err.to_string().contains("missing operator_factory"));
    }

    /// Downstream is optional — actors that route exclusively via
    /// `router.send_to(&peer, msg)` should still build successfully.
    #[test]
    fn builder_without_downstream_is_accepted() {
        let name = StageName::unnamed(StageKind::Tracker);
        let _ = NvTracker::builder(name, 4)
            .batch_formation(noop_batch_formation())
            .operator_factory(|_bx, _batch, _cb| unreachable!("not invoked in this test"))
            .results(
                NvTrackerResults::builder()
                    .on_tracking(noop_on_tracking)
                    .on_source_eos(noop_on_source_eos)
                    .on_error(noop_on_error)
                    .build(),
            )
            .build()
            .unwrap();
    }

    #[test]
    fn nvtracker_implements_reset_stream_handler() {
        fn assert_impl<H: Handler<ResetStreamPayload>>() {}
        assert_impl::<NvTracker>();
        let payload = ResetStreamPayload {
            source_id: "cam-0".to_string(),
        };
        assert_eq!(payload.source_id, "cam-0");
    }

    #[test]
    fn builder_accepts_full_config() {
        let name = StageName::unnamed(StageKind::Tracker);
        let _ = NvTracker::builder(name, 4)
            .downstream(StageName::unnamed(StageKind::Render))
            .batch_formation(noop_batch_formation())
            .operator_factory(|_bx, _batch, _cb| unreachable!("not invoked in this test"))
            .results(
                NvTrackerResults::builder()
                    .on_tracking(noop_on_tracking)
                    .on_source_eos(noop_on_source_eos)
                    .on_error(noop_on_error)
                    .build(),
            )
            .common(
                NvTrackerCommon::builder()
                    .drain_timeout(Duration::from_secs(10))
                    .poll_timeout(Duration::from_millis(50))
                    .build(),
            )
            .build()
            .unwrap();
    }

    /// All three per-variant hooks have defaults that
    /// [`NvTrackerBuilder::build`] auto-installs — a builder with
    /// only `.operator_factory(...)` set still produces a valid
    /// `ActorBuilder`.
    #[test]
    fn builder_accepts_default_forwarders() {
        let name = StageName::unnamed(StageKind::Tracker);
        let _ = NvTracker::builder(name, 4)
            .downstream(StageName::unnamed(StageKind::Render))
            .batch_formation(noop_batch_formation())
            .operator_factory(|_bx, _batch, _cb| unreachable!("not invoked in this test"))
            .build()
            .unwrap();
    }

    /// The three `default_on_*` associated functions slot into the
    /// [`NvTrackerResults`] builder's generic hook bounds as-is.
    #[test]
    fn builder_accepts_explicit_defaults() {
        let name = StageName::unnamed(StageKind::Tracker);
        let _ = NvTracker::builder(name.clone(), 4)
            .downstream(StageName::unnamed(StageKind::Render))
            .batch_formation(noop_batch_formation())
            .operator_factory(|_bx, _batch, _cb| unreachable!("not invoked in this test"))
            .results(
                NvTrackerResults::builder()
                    .on_tracking(NvTracker::default_on_tracking())
                    .on_source_eos(NvTracker::default_on_source_eos())
                    .on_error(NvTracker::default_on_error())
                    .build(),
            )
            .common(
                NvTrackerCommon::builder()
                    .on_stopping(NvTracker::default_on_stopping())
                    .build(),
            )
            .build()
            .unwrap();
    }

    /// The [`NvTrackerCommon`] bundle accepts a user-supplied
    /// `.stopping(F)` closure.  Compile-only verification of the
    /// hook bound; ordering is covered by end-to-end smoke runs.
    #[test]
    fn builder_accepts_user_stopping() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let flag = Arc::new(AtomicBool::new(false));
        let flag_hook = flag.clone();
        let name = StageName::unnamed(StageKind::Tracker);
        let _ = NvTracker::builder(name, 4)
            .downstream(StageName::unnamed(StageKind::Render))
            .batch_formation(noop_batch_formation())
            .operator_factory(|_bx, _batch, _cb| unreachable!("not invoked in this test"))
            .common(
                NvTrackerCommon::builder()
                    .on_stopping(move |_ctx| {
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
    /// returns `Ok(Flow::Cont)`.
    #[test]
    fn default_on_source_eos_forwards_via_router() {
        let hook = NvTracker::default_on_source_eos();
        let (router, rx) = router_with_default_peer();
        let ctx = test_hook_ctx();
        let flow = hook(&ctx, &router, "cam-1").expect("default hook is infallible");
        assert!(matches!(flow, Flow::Cont));
        match rx.try_recv().expect("SourceEos should be forwarded") {
            PipelineMsg::SourceEos { source_id, .. } => assert_eq!(source_id, "cam-1"),
            other => panic!("expected SourceEos, got {other:?}"),
        }
        assert!(rx.try_recv().is_err(), "exactly one message forwarded");
    }

    /// `default_on_error` classifies every error as
    /// [`ErrorAction::LogAndContinue`] and must not route anything
    /// downstream.
    #[test]
    fn default_on_error_does_not_route() {
        let hook = NvTracker::default_on_error();
        let (router, rx) = router_with_default_peer();
        let ctx = test_hook_ctx();
        let err = NvTrackerError::PipelineError("synthetic".to_string());
        let action = hook(&ctx, &router, &err);
        assert!(matches!(action, ErrorAction::LogAndContinue));
        assert!(rx.try_recv().is_err(), "errors must not be routed");
    }

    /// An unbound [`NvTrackerHookCtx`] (factory hasn't populated
    /// the operator slot) returns a descriptive `Err` from
    /// `reset_stream` rather than panicking or silently succeeding.
    #[test]
    fn hook_ctx_unbound_reset_stream_returns_err() {
        let ctx = test_hook_ctx();
        let err = ctx
            .reset_stream("cam-0")
            .expect_err("unbound ctx must error");
        assert!(
            err.to_string().contains("not bound"),
            "unexpected error: {err}"
        );
    }

    /// Clones of the same [`NvTrackerHookCtx`] share the
    /// [`OnceLock`] for the operator slot — once the factory
    /// populates the slot on one clone, every clone observes the
    /// binding.
    #[test]
    fn hook_ctx_clones_share_operator_slot() {
        let a = test_hook_ctx();
        let b = a.clone();
        assert!(Arc::ptr_eq(&a.operator, &b.operator));
        let c = test_hook_ctx();
        assert!(!Arc::ptr_eq(&a.operator, &c.operator));
    }

    /// The user-facing accessors on [`NvTrackerHookCtx`] — own
    /// name, singleton-shared, named-shared, registry resolve —
    /// round-trip the values supplied at construction time.
    #[test]
    fn hook_ctx_accessors_expose_construction_state() {
        use crate::addr::Addr;
        use crate::shared::SharedStore;

        let mut reg = Registry::new();
        let picasso = StageName::unnamed(StageKind::Render);
        let (tx, _rx) = bounded::<PipelineMsg>(1);
        reg.insert::<PipelineMsg>(picasso.clone(), Addr::new(picasso.clone(), tx));
        let mut shared = SharedStore::new();
        shared.insert::<u32>(42);
        shared.insert_as::<&'static str>("greeting", "hi");
        let ctx = NvTrackerHookCtx {
            base: HookCtx::new(
                StageName::unnamed(StageKind::Tracker),
                Arc::from("test"),
                Arc::new(reg),
                Arc::new(shared),
                Arc::new(std::sync::atomic::AtomicBool::new(false)),
                crate::stage_metrics::StageMetrics::new("tracker-test"),
            ),
            operator: Arc::new(OnceLock::new()),
        };
        assert_eq!(ctx.own_name().kind, StageKind::Tracker);
        assert_eq!(ctx.shared::<u32>().map(|a| *a), Some(42));
        assert_eq!(
            ctx.shared_as::<&'static str>("greeting").map(|a| *a),
            Some("hi")
        );
        assert!(ctx.resolve::<PipelineMsg>(&picasso).is_ok());
    }

    /// A builder with only the required factory still compiles
    /// and produces a valid [`ActorBuilder`] — the reset handle is
    /// now auto-created inside `build`, so no user-facing setter
    /// is required.
    #[test]
    fn builder_needs_no_reset_handle_setter() {
        let name = StageName::unnamed(StageKind::Tracker);
        let _ = NvTracker::builder(name, 4)
            .downstream(StageName::unnamed(StageKind::Render))
            .batch_formation(noop_batch_formation())
            .operator_factory(|_bx, _batch, _cb| unreachable!("not invoked in this test"))
            .results(
                NvTrackerResults::builder()
                    .on_tracking(noop_on_tracking)
                    .on_source_eos(noop_on_source_eos)
                    .on_error(noop_on_error)
                    .build(),
            )
            .build()
            .unwrap();
    }

    /// `dispatch_output` flips the shared stop flag when the
    /// `on_error` hook returns [`ErrorAction::Fatal`].
    #[test]
    fn dispatch_output_fatal_error_requests_stop() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let flag = Arc::new(AtomicBool::new(false));
        let ctx = NvTrackerHookCtx {
            base: HookCtx::new(
                StageName::unnamed(StageKind::Tracker),
                Arc::from("test"),
                Arc::new(Registry::new()),
                Arc::new(crate::shared::SharedStore::new()),
                flag.clone(),
                crate::stage_metrics::StageMetrics::new("tracker-test"),
            ),
            operator: Arc::new(OnceLock::new()),
        };
        let (router, _rx) = router_with_default_peer();
        let on_tracking: OnTrackingHook = Arc::new(noop_on_tracking);
        let on_source_eos: OnSourceEosHook = Arc::new(noop_on_source_eos);
        let on_error: OnErrorHook = Arc::new(
            |_c: &NvTrackerHookCtx, _r: &Router<PipelineMsg>, _err: &NvTrackerError| {
                ErrorAction::Fatal
            },
        );
        dispatch_output(
            TrackerOperatorOutput::Error(NvTrackerError::PipelineError("boom".into())),
            &on_tracking,
            &on_source_eos,
            &on_error,
            &router,
            &ctx,
        );
        assert!(flag.load(Ordering::SeqCst));
        assert!(ctx.stop_requested());
    }

    /// `dispatch_output` flips the shared stop flag when the
    /// `on_source_eos` hook returns `Ok(Flow::Stop)`.
    #[test]
    fn dispatch_output_source_eos_stop_requests_stop() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let flag = Arc::new(AtomicBool::new(false));
        let ctx = NvTrackerHookCtx {
            base: HookCtx::new(
                StageName::unnamed(StageKind::Tracker),
                Arc::from("test"),
                Arc::new(Registry::new()),
                Arc::new(crate::shared::SharedStore::new()),
                flag.clone(),
                crate::stage_metrics::StageMetrics::new("tracker-test"),
            ),
            operator: Arc::new(OnceLock::new()),
        };
        let (router, _rx) = router_with_default_peer();
        let on_tracking: OnTrackingHook = Arc::new(noop_on_tracking);
        let on_source_eos: OnSourceEosHook =
            Arc::new(|_c: &NvTrackerHookCtx, _r: &Router<PipelineMsg>, _sid: &str| Ok(Flow::Stop));
        let on_error: OnErrorHook = Arc::new(noop_on_error);
        dispatch_output(
            TrackerOperatorOutput::Eos {
                source_id: "cam-0".into(),
            },
            &on_tracking,
            &on_source_eos,
            &on_error,
            &router,
            &ctx,
        );
        assert!(flag.load(Ordering::SeqCst));
    }
}
