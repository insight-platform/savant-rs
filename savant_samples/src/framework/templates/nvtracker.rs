//! [`NvTracker`] — multiplexed `PipelineMsg` → `PipelineMsg`
//! stage wrapping an
//! [`NvTrackerBatchingOperator`](deepstream_nvtracker::NvTrackerBatchingOperator).
//!
//! Mirror of the [`NvInfer`](super::nvinfer::NvInfer)
//! template — structurally identical, only the operator and output
//! types differ.  Direct replacement for the handwritten tracker
//! thread in `cars_tracking/pipeline/tracker.rs`.
//!
//! # Default forwarders
//!
//! Every per-variant hook has an out-of-the-box default — if the
//! matching builder setter is not called,
//! [`build`](NvTrackerBuilder::build) auto-installs it:
//!
//! * [`NvTracker::default_on_tracking`] — for every frame
//!   output, calls
//!   [`TrackerOperatorFrameOutput::apply_to_frame`](deepstream_nvtracker::TrackerOperatorFrameOutput::apply_to_frame)
//!   to reconcile tracker-side updates with the frame's objects, logs
//!   unmatched updates at `warn` level, then forwards
//!   [`PipelineMsg::Deliveries`](crate::framework::envelopes::PipelineMsg::Deliveries)
//!   via `router.send(..)`.  Tracker outputs are object/track updates
//!   rather than tensor layers, so no attribute attachment is done.
//! * [`NvTracker::default_on_source_eos`] — forwards
//!   [`PipelineMsg::SourceEos`](crate::framework::envelopes::PipelineMsg::SourceEos).
//! * [`NvTracker::default_on_error`] — logs at `error` level,
//!   prefixed by the stage name.
//!
//! # Runtime invariant
//!
//! [`NvTracker`] stores its three hooks as non-`Option`
//! `Arc<dyn Fn>`.  The builder's `Option<...>` fields exist purely
//! as an internal "was the setter called?" marker;
//! [`NvTrackerBuilder::build`] always substitutes the matching
//! `NvTracker::default_on_*` before the actor value is
//! constructed.  There is therefore no runtime code path where a
//! [`TrackerOperatorOutput`] variant is dispatched to `None` —
//! every branch (`Tracking`, `Eos`, `Error`) is always handled,
//! whether by a user-supplied closure or by the auto-installed
//! default.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use deepstream_nvtracker::{
    NvTrackerBatchingOperator, NvTrackerError, TrackerOperatorOutput,
    TrackerOperatorResultCallback, TrackerOperatorTrackingOutput,
};

use crate::framework::envelopes::{BatchDelivery, PipelineMsg, SingleDelivery};
use crate::framework::router::Router;
use crate::framework::supervisor::StageName;
use crate::framework::{
    Actor, ActorBuilder, BuildCtx, Context, Dispatch, Flow, Handler, ResetStreamPayload,
    ShutdownPayload, SourceEosPayload,
};

/// Default graceful-shutdown drain timeout — matches the legacy
/// sample's `TRACKER_DRAIN_TIMEOUT`.
pub const DEFAULT_DRAIN_TIMEOUT: Duration = Duration::from_secs(30);
/// Default inbox receive-poll cadence.
pub const DEFAULT_POLL_TIMEOUT: Duration = Duration::from_millis(100);

/// Hook fired for every
/// [`TrackerOperatorOutput::Tracking`](deepstream_nvtracker::TrackerOperatorOutput::Tracking)
/// payload.  Receives the
/// [`TrackerOperatorTrackingOutput`](deepstream_nvtracker::TrackerOperatorTrackingOutput)
/// (owning the sealed deliveries) and the stage's
/// [`Router<PipelineMsg>`].  Typical use: reconcile per-frame
/// tracking updates with the detections via
/// [`TrackerOperatorFrameOutput::apply_to_frame`](deepstream_nvtracker::TrackerOperatorFrameOutput::apply_to_frame),
/// then send
/// [`PipelineMsg::Deliveries`](crate::framework::envelopes::PipelineMsg::Deliveries)
/// downstream.
///
/// See [`NvTracker::default_on_tracking`] for the ready-made
/// "apply tracking + forward" default.
pub type OnTrackingHook =
    Arc<dyn Fn(TrackerOperatorTrackingOutput, &Router<PipelineMsg>) + Send + Sync + 'static>;

/// Hook fired for every
/// [`TrackerOperatorOutput::Eos`](deepstream_nvtracker::TrackerOperatorOutput::Eos)
/// payload.  Receives the source id and the router.
pub type OnSourceEosHook = Arc<dyn Fn(&str, &Router<PipelineMsg>) + Send + Sync + 'static>;

/// Hook fired for every
/// [`TrackerOperatorOutput::Error`](deepstream_nvtracker::TrackerOperatorOutput::Error)
/// payload.  Receives the operator error and the router.
pub type OnErrorHook = Arc<dyn Fn(&NvTrackerError, &Router<PipelineMsg>) + Send + Sync + 'static>;

/// Factory closure — receives the wired
/// [`TrackerOperatorResultCallback`] and returns a ready
/// [`NvTrackerBatchingOperator`].
pub type NvTrackerOperatorFactory = Box<
    dyn FnOnce(&BuildCtx, TrackerOperatorResultCallback) -> Result<NvTrackerBatchingOperator>
        + Send,
>;

/// `PipelineMsg` → `PipelineMsg` actor template wrapping an
/// [`NvTrackerBatchingOperator`].
pub struct NvTracker {
    operator: NvTrackerBatchingOperator,
    router: Router<PipelineMsg>,
    on_tracking: OnTrackingHook,
    on_source_eos: OnSourceEosHook,
    on_error: OnErrorHook,
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
    /// [`PipelineMsg::Deliveries`](crate::framework::envelopes::PipelineMsg::Deliveries)
    /// downstream via `router.send(..)`.
    ///
    /// No tensor-attribute attachment is performed — tracker outputs
    /// are object/track updates rather than tensor layers, so the
    /// "raw tensors as attributes" convention from
    /// [`NvInfer::default_on_inference`](super::nvinfer::NvInfer::default_on_inference)
    /// does not apply here.
    pub fn default_on_tracking(
        namespace: StageName,
    ) -> impl Fn(TrackerOperatorTrackingOutput, &Router<PipelineMsg>) + Send + Sync + 'static {
        move |mut tracking, router| {
            let ns = namespace.to_string();
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
                if !router.send(PipelineMsg::Deliveries(Box::new(sealed))) {
                    log::warn!("[{ns}] downstream closed; dropping sealed batch");
                }
            }
        }
    }

    /// Default `on_source_eos` forwarder.  Sends
    /// [`PipelineMsg::SourceEos { source_id }`](crate::framework::envelopes::PipelineMsg::SourceEos)
    /// via `router.send(..)`.  The emit site preserves the stream-
    /// aligned contract: the operator guarantees
    /// [`TrackerOperatorOutput::Eos`](deepstream_nvtracker::TrackerOperatorOutput::Eos)
    /// fires strictly after the last
    /// [`TrackerOperatorOutput::Tracking`](deepstream_nvtracker::TrackerOperatorOutput::Tracking)
    /// for the same `source_id`.
    pub fn default_on_source_eos() -> impl Fn(&str, &Router<PipelineMsg>) + Send + Sync + 'static {
        |source_id, router| {
            log::info!("TrackerOperatorOutput::Eos for source_id={source_id}; propagating");
            if !router.send(PipelineMsg::SourceEos {
                source_id: source_id.to_string(),
            }) {
                log::warn!("downstream closed; dropping SourceEos({source_id})");
            }
        }
    }

    /// Default `on_error` logger.  Emits a single `error!` line
    /// prefixed by the stage name.
    pub fn default_on_error(
        namespace: StageName,
    ) -> impl Fn(&NvTrackerError, &Router<PipelineMsg>) + Send + Sync + 'static {
        move |err, _router| {
            log::error!("[{namespace}] operator error: {err}");
        }
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
                        &self.on_tracking,
                        &self.on_source_eos,
                        &self.on_error,
                        &self.router,
                    );
                }
            }
            Err(e) => {
                log::error!("[{}] graceful_shutdown failed: {e}", ctx.own_name());
            }
        }
        self.drain_done = true;
        log::info!("[{}] nvtracker stopping", ctx.own_name());
    }
}

impl Handler<SingleDelivery> for NvTracker {
    fn handle(&mut self, msg: SingleDelivery, ctx: &mut Context<Self>) -> Result<Flow> {
        let pairs = PipelineMsg::Delivery(msg.0).into_pairs();
        submit_pairs(&mut self.operator, pairs, ctx.own_name())
    }
}

impl Handler<BatchDelivery> for NvTracker {
    fn handle(&mut self, msg: BatchDelivery, ctx: &mut Context<Self>) -> Result<Flow> {
        let pairs = msg.0.unseal();
        submit_pairs(&mut self.operator, pairs, ctx.own_name())
    }
}

impl Handler<SourceEosPayload> for NvTracker {
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

impl Handler<ShutdownPayload> for NvTracker {}

impl Handler<ResetStreamPayload> for NvTracker {
    /// Forward the reset request to
    /// [`NvTrackerBatchingOperator::reset_stream`].  Errors are
    /// logged and swallowed — a failed reset (e.g. because the
    /// operator is already draining) must not tear the actor down.
    fn handle(&mut self, msg: ResetStreamPayload, ctx: &mut Context<Self>) -> Result<Flow> {
        if let Err(e) = self.operator.reset_stream(&msg.source_id) {
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
    operator: &mut NvTrackerBatchingOperator,
    pairs: Vec<(
        savant_core::primitives::frame::VideoFrameProxy,
        deepstream_buffers::SharedBuffer,
    )>,
    stage: &StageName,
) -> Result<Flow> {
    for (frame, buffer) in pairs {
        if let Err(e) = operator.add_frame(frame, buffer) {
            log::error!("[{stage}] add_frame failed: {e}");
            return Err(anyhow!("track add_frame: {e}"));
        }
    }
    Ok(Flow::Cont)
}

fn dispatch_output(
    out: TrackerOperatorOutput,
    on_tracking: &OnTrackingHook,
    on_source_eos: &OnSourceEosHook,
    on_error: &OnErrorHook,
    router: &Router<PipelineMsg>,
) {
    match out {
        TrackerOperatorOutput::Tracking(tracking) => on_tracking(tracking, router),
        TrackerOperatorOutput::Eos { source_id } => on_source_eos(&source_id, router),
        TrackerOperatorOutput::Error(err) => on_error(&err, router),
    }
}

/// Fluent builder for [`NvTracker`].
pub struct NvTrackerBuilder {
    name: StageName,
    capacity: usize,
    downstream: Option<StageName>,
    drain_timeout: Duration,
    poll_timeout: Duration,
    operator_factory: Option<NvTrackerOperatorFactory>,
    on_tracking: Option<OnTrackingHook>,
    on_source_eos: Option<OnSourceEosHook>,
    on_error: Option<OnErrorHook>,
}

impl NvTrackerBuilder {
    /// Start a builder with sample-style drain / poll defaults.
    pub fn new(name: StageName, capacity: usize) -> Self {
        Self {
            name,
            capacity,
            downstream: None,
            drain_timeout: DEFAULT_DRAIN_TIMEOUT,
            poll_timeout: DEFAULT_POLL_TIMEOUT,
            operator_factory: None,
            on_tracking: None,
            on_source_eos: None,
            on_error: None,
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

    /// Required: factory that builds the
    /// [`NvTrackerBatchingOperator`] given the already-wired result
    /// callback.
    pub fn operator_factory<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&BuildCtx, TrackerOperatorResultCallback) -> Result<NvTrackerBatchingOperator>
            + Send
            + 'static,
    {
        self.operator_factory = Some(Box::new(f));
        self
    }

    /// Override the `on_tracking` hook.  If not called, the builder
    /// auto-installs [`NvTracker::default_on_tracking`] at
    /// [`build`](Self::build) time.
    ///
    /// Omitting this setter is equivalent to calling
    /// `.on_tracking(NvTracker::default_on_tracking())`.
    pub fn on_tracking<F>(mut self, f: F) -> Self
    where
        F: Fn(TrackerOperatorTrackingOutput, &Router<PipelineMsg>) + Send + Sync + 'static,
    {
        self.on_tracking = Some(Arc::new(f));
        self
    }

    /// Override the `on_source_eos` hook.  If not called, the builder
    /// auto-installs [`NvTracker::default_on_source_eos`].
    ///
    /// Omitting this setter is equivalent to calling
    /// `.on_source_eos(NvTracker::default_on_source_eos())`.
    pub fn on_source_eos<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &Router<PipelineMsg>) + Send + Sync + 'static,
    {
        self.on_source_eos = Some(Arc::new(f));
        self
    }

    /// Override the `on_error` hook.  If not called, the builder
    /// auto-installs [`NvTracker::default_on_error`] with the
    /// actor's [`StageName`] as the log prefix.
    ///
    /// Omitting this setter is equivalent to calling
    /// `.on_error(NvTracker::default_on_error(name.clone()))`.
    pub fn on_error<F>(mut self, f: F) -> Self
    where
        F: Fn(&NvTrackerError, &Router<PipelineMsg>) + Send + Sync + 'static,
    {
        self.on_error = Some(Arc::new(f));
        self
    }

    /// Override the graceful-shutdown drain timeout.
    pub fn drain_timeout(mut self, d: Duration) -> Self {
        self.drain_timeout = d;
        self
    }

    /// Inbox receive-poll cadence.
    pub fn poll_timeout(mut self, d: Duration) -> Self {
        self.poll_timeout = d;
        self
    }

    /// Finalise the template.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `operator_factory` is missing.  All three
    /// per-variant hooks have defaults that are auto-installed when
    /// their setters have not been called.  `downstream` is
    /// optional — callers that route via [`Router::send_to`]
    /// exclusively may omit it.
    pub fn build(self) -> Result<ActorBuilder<NvTracker>> {
        let NvTrackerBuilder {
            name,
            capacity,
            downstream,
            drain_timeout,
            poll_timeout,
            operator_factory,
            on_tracking,
            on_source_eos,
            on_error,
        } = self;
        let operator_factory =
            operator_factory.ok_or_else(|| anyhow!("NvTracker: missing operator_factory"))?;
        let on_tracking: OnTrackingHook =
            on_tracking.unwrap_or_else(|| Arc::new(NvTracker::default_on_tracking(name.clone())));
        let on_source_eos: OnSourceEosHook =
            on_source_eos.unwrap_or_else(|| Arc::new(NvTracker::default_on_source_eos()));
        let on_error: OnErrorHook =
            on_error.unwrap_or_else(|| Arc::new(NvTracker::default_on_error(name.clone())));
        Ok(ActorBuilder::new(name, capacity)
            .poll_timeout(poll_timeout)
            .factory(move |bx| {
                let router: Router<PipelineMsg> = bx.router(downstream.as_ref())?;
                let router_for_cb = router.clone();
                let on_tracking_cb = on_tracking.clone();
                let on_source_eos_cb = on_source_eos.clone();
                let on_error_cb = on_error.clone();
                let result_cb: TrackerOperatorResultCallback = Box::new(move |out| {
                    dispatch_output(
                        out,
                        &on_tracking_cb,
                        &on_source_eos_cb,
                        &on_error_cb,
                        &router_for_cb,
                    );
                });
                let operator = (operator_factory)(bx, result_cb)?;
                Ok(NvTracker {
                    operator,
                    router,
                    on_tracking,
                    on_source_eos,
                    on_error,
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
    use crate::framework::addr::Addr;
    use crate::framework::operator_sink::OperatorSink;
    use crate::framework::registry::Registry;
    use crate::framework::supervisor::{StageKind, StageName};
    use crossbeam::channel::{bounded, Receiver};

    fn noop_on_tracking(_: TrackerOperatorTrackingOutput, _: &Router<PipelineMsg>) {}
    fn noop_on_source_eos(_: &str, _: &Router<PipelineMsg>) {}
    fn noop_on_error(_: &NvTrackerError, _: &Router<PipelineMsg>) {}

    /// Build a `Router<PipelineMsg>` whose default peer is a freshly
    /// registered `PipelineMsg` inbox.  Returns the router and the
    /// receiver so tests can observe forwarded messages.
    fn router_with_default_peer() -> (Router<PipelineMsg>, Receiver<PipelineMsg>) {
        let peer = StageName::unnamed(StageKind::Picasso);
        let (tx, rx) = bounded::<PipelineMsg>(4);
        let mut reg = Registry::new();
        reg.insert::<PipelineMsg>(peer.clone(), Addr::new(peer.clone(), tx));
        let addr: Addr<PipelineMsg> = reg.get::<PipelineMsg>(&peer).unwrap();
        let owner = StageName::unnamed(StageKind::Tracker);
        let default = OperatorSink::new(owner.clone(), addr);
        let router = Router::new(owner, Arc::new(reg), Some(default));
        (router, rx)
    }

    #[test]
    fn builder_requires_operator_factory() {
        let name = StageName::unnamed(StageKind::Tracker);
        let err = NvTracker::builder(name.clone(), 4).build().err().unwrap();
        assert!(err.to_string().contains("missing operator_factory"));
    }

    /// Downstream is optional — actors that route exclusively via
    /// `router.send_to(&peer, msg)` should still build successfully.
    #[test]
    fn builder_without_downstream_is_accepted() {
        let name = StageName::unnamed(StageKind::Tracker);
        let _ = NvTracker::builder(name, 4)
            .operator_factory(|_bx, _cb| unreachable!("not invoked in this test"))
            .on_tracking(noop_on_tracking)
            .on_source_eos(noop_on_source_eos)
            .on_error(noop_on_error)
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
            .downstream(StageName::unnamed(StageKind::Picasso))
            .operator_factory(|_bx, _cb| unreachable!("not invoked in this test"))
            .on_tracking(noop_on_tracking)
            .on_source_eos(noop_on_source_eos)
            .on_error(noop_on_error)
            .drain_timeout(Duration::from_secs(10))
            .poll_timeout(Duration::from_millis(50))
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
            .downstream(StageName::unnamed(StageKind::Picasso))
            .operator_factory(|_bx, _cb| unreachable!("not invoked in this test"))
            .build()
            .unwrap();
    }

    /// The three `default_on_*` associated functions slot into the
    /// builder's generic hook bounds as-is.
    #[test]
    fn builder_accepts_explicit_defaults() {
        let name = StageName::unnamed(StageKind::Tracker);
        let _ = NvTracker::builder(name.clone(), 4)
            .downstream(StageName::unnamed(StageKind::Picasso))
            .operator_factory(|_bx, _cb| unreachable!("not invoked in this test"))
            .on_tracking(NvTracker::default_on_tracking(name.clone()))
            .on_source_eos(NvTracker::default_on_source_eos())
            .on_error(NvTracker::default_on_error(name))
            .build()
            .unwrap();
    }

    /// `default_on_source_eos` forwards a `PipelineMsg::SourceEos`
    /// carrying the supplied `source_id` via `router.send(..)`.
    #[test]
    fn default_on_source_eos_forwards_via_router() {
        let hook = NvTracker::default_on_source_eos();
        let (router, rx) = router_with_default_peer();
        hook("cam-1", &router);
        match rx.try_recv().expect("SourceEos should be forwarded") {
            PipelineMsg::SourceEos { source_id } => assert_eq!(source_id, "cam-1"),
            other => panic!("expected SourceEos, got {other:?}"),
        }
        assert!(rx.try_recv().is_err(), "exactly one message forwarded");
    }

    /// `default_on_error` is log-only — it must not route anything
    /// downstream.
    #[test]
    fn default_on_error_does_not_route() {
        let hook = NvTracker::default_on_error(StageName::unnamed(StageKind::Tracker));
        let (router, rx) = router_with_default_peer();
        let err = NvTrackerError::PipelineError("synthetic".to_string());
        hook(&err, &router);
        assert!(rx.try_recv().is_err(), "errors must not be routed");
    }
}
