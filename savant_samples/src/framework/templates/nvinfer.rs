//! [`NvInfer`] — multiplexed `PipelineMsg` → `PipelineMsg` stage
//! wrapping a user-supplied
//! [`NvInferBatchingOperator`](deepstream_nvinfer::prelude::NvInferBatchingOperator).
//!
//! Direct replacement for the handwritten infer thread in
//! `cars_tracking/pipeline/infer.rs`.  The template:
//!
//! * Captures a [`Router<PipelineMsg>`] at build time — optionally
//!   bound to a default peer via [`NvInferBuilder::downstream`] — and
//!   hands it to three user-supplied per-variant hooks
//!   ([`on_inference`](NvInferBuilder::on_inference),
//!   [`on_source_eos`](NvInferBuilder::on_source_eos),
//!   [`on_error`](NvInferBuilder::on_error)).  The template dispatches
//!   each [`OperatorOutput`] variant to its matching hook both from
//!   the operator's result callback and from the final
//!   `graceful_shutdown` drain, preserving the stream-aligned
//!   `SourceEos` contract of the legacy sample's
//!   `process_infer_output`.  Inside each hook use `router.send(msg)`
//!   to route to the configured default peer, or
//!   `router.send_to(&peer, msg)` to route to any registered stage by
//!   name.
//! * Accepts the operator by **factory closure** — the factory
//!   receives the ready-to-install
//!   [`OperatorResultCallback`](deepstream_nvinfer::OperatorResultCallback)
//!   so the user can plug it into whatever their nvinfer
//!   construction needs (warmup, cached TRT engines, custom batch
//!   formation, …) without the template having an opinion.
//! * Routes
//!   [`PipelineMsg::Delivery`](crate::framework::envelopes::PipelineMsg::Delivery)
//!   and
//!   [`PipelineMsg::Deliveries`](crate::framework::envelopes::PipelineMsg::Deliveries)
//!   through
//!   [`NvInferBatchingOperator::add_frame`](deepstream_nvinfer::prelude::NvInferBatchingOperator::add_frame)
//!   on a flattened `Vec<(frame, buffer)>` basis — matching the
//!   "one code path per stage" contract of [`PipelineMsg::into_pairs`](crate::framework::envelopes::PipelineMsg::into_pairs).
//! * Routes
//!   [`PipelineMsg::SourceEos`](crate::framework::envelopes::PipelineMsg::SourceEos)
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
//! # Default forwarders
//!
//! Every hook has an out-of-the-box default — if the matching
//! builder setter is not called, [`build`](NvInferBuilder::build)
//! auto-installs it:
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
//!   [`PipelineMsg::Deliveries`](crate::framework::envelopes::PipelineMsg::Deliveries)
//!   downstream via `router.send(..)`.  Tensors with host data
//!   unavailable (e.g. `host_copy_enabled=false`) are logged and
//!   skipped.
//! * [`NvInfer::default_on_source_eos`] — forwards
//!   [`PipelineMsg::SourceEos`](crate::framework::envelopes::PipelineMsg::SourceEos)
//!   via `router.send(..)`.
//! * [`NvInfer::default_on_error`] — logs the operator error at
//!   `error` level, prefixed by the stage name.
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

use crate::framework::envelopes::{BatchDelivery, PipelineMsg, SingleDelivery};
use crate::framework::router::Router;
use crate::framework::supervisor::StageName;
use crate::framework::{
    Actor, ActorBuilder, BuildCtx, Context, Dispatch, Flow, Handler, ShutdownPayload,
    SourceEosPayload,
};

/// Default graceful-shutdown drain timeout — matches the legacy
/// sample's `INFER_DRAIN_TIMEOUT`.
pub const DEFAULT_DRAIN_TIMEOUT: Duration = Duration::from_secs(30);
/// Default inbox receive-poll cadence.
pub const DEFAULT_POLL_TIMEOUT: Duration = Duration::from_millis(100);

/// Hook fired for every
/// [`OperatorOutput::Inference`](deepstream_nvinfer::OperatorOutput::Inference)
/// payload.  Receives the
/// [`OperatorInferenceOutput`](deepstream_nvinfer::OperatorInferenceOutput)
/// (which owns the output buffer + tensor pointers) and the stage's
/// [`Router<PipelineMsg>`].  Typical use:
///
/// * Attach per-tensor attributes / decoded detections to
///   `inf.frames()[i].frame`.
/// * Call [`OperatorInferenceOutput::take_deliveries`](deepstream_nvinfer::OperatorInferenceOutput::take_deliveries),
///   drop `inf`, then send
///   [`PipelineMsg::Deliveries`](crate::framework::envelopes::PipelineMsg::Deliveries)
///   downstream via `router.send(..)` or `router.send_to(..)`.
///
/// See [`NvInfer::default_on_inference`] for the ready-made
/// "attach raw tensors + forward" default.
pub type OnInferenceHook =
    Arc<dyn Fn(OperatorInferenceOutput, &Router<PipelineMsg>) + Send + Sync + 'static>;

/// Hook fired for every
/// [`OperatorOutput::Eos`](deepstream_nvinfer::OperatorOutput::Eos)
/// payload.  Receives the source identifier and the router.  Typical
/// use: `router.send(PipelineMsg::SourceEos { source_id: source_id.into() })`.
pub type OnSourceEosHook = Arc<dyn Fn(&str, &Router<PipelineMsg>) + Send + Sync + 'static>;

/// Hook fired for every
/// [`OperatorOutput::Error`](deepstream_nvinfer::OperatorOutput::Error)
/// payload.  Receives the operator error and the router.  Typical
/// use: log (see [`NvInfer::default_on_error`]).  The router is
/// handed over for callers that want to translate the error into a
/// downstream control message.
pub type OnErrorHook = Arc<dyn Fn(&NvInferError, &Router<PipelineMsg>) + Send + Sync + 'static>;

/// Factory closure: builds the [`NvInferBatchingOperator`] once the
/// template has wired the result callback.  The factory is called
/// in phase 2 of `System::build`; it receives the fully-built
/// [`BuildCtx`] so it can resolve sibling stages and shared state
/// (e.g. warmup state, cached converter) without capturing them
/// from the builder's definition site.
pub type NvInferOperatorFactory =
    Box<dyn FnOnce(&BuildCtx, OperatorResultCallback) -> Result<NvInferBatchingOperator> + Send>;

/// `PipelineMsg` → `PipelineMsg` actor template wrapping an
/// [`NvInferBatchingOperator`].
pub struct NvInfer {
    operator: NvInferBatchingOperator,
    router: Router<PipelineMsg>,
    on_inference: OnInferenceHook,
    on_source_eos: OnSourceEosHook,
    on_error: OnErrorHook,
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
    /// [`PipelineMsg::Deliveries`](crate::framework::envelopes::PipelineMsg::Deliveries)
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
    /// [`OperatorInferenceOutput`](deepstream_nvinfer::OperatorInferenceOutput)
    /// with backing `GstBuffer` memory, which the public API does
    /// not expose.  Coverage is therefore exercised end-to-end via
    /// the `cars-demo` smoke run, which drives real tensors through
    /// the template.
    pub fn default_on_inference(
        namespace: StageName,
    ) -> impl Fn(OperatorInferenceOutput, &Router<PipelineMsg>) + Send + Sync + 'static {
        move |mut inf, router| {
            let ns = namespace.to_string();
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
    /// [`OperatorOutput::Eos`](deepstream_nvinfer::OperatorOutput::Eos)
    /// fires strictly after the last
    /// [`OperatorOutput::Inference`](deepstream_nvinfer::OperatorOutput::Inference)
    /// for the same `source_id`.
    pub fn default_on_source_eos() -> impl Fn(&str, &Router<PipelineMsg>) + Send + Sync + 'static {
        |source_id, router| {
            log::info!("OperatorOutput::Eos for source_id={source_id}; propagating");
            if !router.send(PipelineMsg::SourceEos {
                source_id: source_id.to_string(),
            }) {
                log::warn!("downstream closed; dropping SourceEos({source_id})");
            }
        }
    }

    /// Default `on_error` logger.  Emits a single `error!` line
    /// prefixed by the stage name; the error is otherwise swallowed
    /// so a misbehaving model does not tear the pipeline down.
    pub fn default_on_error(
        namespace: StageName,
    ) -> impl Fn(&NvInferError, &Router<PipelineMsg>) + Send + Sync + 'static {
        move |err, _router| {
            log::error!("[{namespace}] operator error: {err}");
        }
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
                    );
                }
            }
            Err(e) => {
                log::error!("[{}] graceful_shutdown failed: {e}", ctx.own_name());
            }
        }
        self.drain_done = true;
        log::info!("[{}] nvinfer stopping", ctx.own_name());
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
        savant_core::primitives::frame::VideoFrameProxy,
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
fn dispatch_output(
    out: OperatorOutput,
    on_inference: &OnInferenceHook,
    on_source_eos: &OnSourceEosHook,
    on_error: &OnErrorHook,
    router: &Router<PipelineMsg>,
) {
    match out {
        OperatorOutput::Inference(inf) => on_inference(inf, router),
        OperatorOutput::Eos { source_id } => on_source_eos(&source_id, router),
        OperatorOutput::Error(err) => on_error(&err, router),
    }
}

/// Fluent builder for [`NvInfer`].
pub struct NvInferBuilder {
    name: StageName,
    capacity: usize,
    downstream: Option<StageName>,
    drain_timeout: Duration,
    poll_timeout: Duration,
    operator_factory: Option<NvInferOperatorFactory>,
    on_inference: Option<OnInferenceHook>,
    on_source_eos: Option<OnSourceEosHook>,
    on_error: Option<OnErrorHook>,
}

impl NvInferBuilder {
    /// Start a builder with sample-style drain / poll defaults.
    pub fn new(name: StageName, capacity: usize) -> Self {
        Self {
            name,
            capacity,
            downstream: None,
            drain_timeout: DEFAULT_DRAIN_TIMEOUT,
            poll_timeout: DEFAULT_POLL_TIMEOUT,
            operator_factory: None,
            on_inference: None,
            on_source_eos: None,
            on_error: None,
        }
    }

    /// Optional default peer for `router.send(msg)` inside the
    /// per-variant hooks (typically the tracker).  When omitted, the
    /// hooks must use `router.send_to(&peer, msg)` for explicit
    /// name-based routing; a default-less `router.send(msg)` silently
    /// drops the message and logs once.
    pub fn downstream(mut self, peer: StageName) -> Self {
        self.downstream = Some(peer);
        self
    }

    /// Required: factory that builds the
    /// [`NvInferBatchingOperator`] given the already-wired result
    /// callback.  The factory runs in phase 2 of `System::build`.
    pub fn operator_factory<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&BuildCtx, OperatorResultCallback) -> Result<NvInferBatchingOperator>
            + Send
            + 'static,
    {
        self.operator_factory = Some(Box::new(f));
        self
    }

    /// Override the `on_inference` hook.  If not called, the builder
    /// auto-installs [`NvInfer::default_on_inference`] at
    /// [`build`](Self::build) time, capturing the actor's
    /// [`StageName`] as the attribute namespace.
    ///
    /// Omitting this setter is equivalent to calling
    /// `.on_inference(NvInfer::default_on_inference(name.clone()))`.
    pub fn on_inference<F>(mut self, f: F) -> Self
    where
        F: Fn(OperatorInferenceOutput, &Router<PipelineMsg>) + Send + Sync + 'static,
    {
        self.on_inference = Some(Arc::new(f));
        self
    }

    /// Override the `on_source_eos` hook.  If not called, the
    /// builder auto-installs
    /// [`NvInfer::default_on_source_eos`].
    ///
    /// Omitting this setter is equivalent to calling
    /// `.on_source_eos(NvInfer::default_on_source_eos())`.
    pub fn on_source_eos<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &Router<PipelineMsg>) + Send + Sync + 'static,
    {
        self.on_source_eos = Some(Arc::new(f));
        self
    }

    /// Override the `on_error` hook.  If not called, the builder
    /// auto-installs [`NvInfer::default_on_error`] with the
    /// actor's [`StageName`] as the log prefix.
    ///
    /// Omitting this setter is equivalent to calling
    /// `.on_error(NvInfer::default_on_error(name.clone()))`.
    pub fn on_error<F>(mut self, f: F) -> Self
    where
        F: Fn(&NvInferError, &Router<PipelineMsg>) + Send + Sync + 'static,
    {
        self.on_error = Some(Arc::new(f));
        self
    }

    /// Override the graceful-shutdown drain timeout (default
    /// [`DEFAULT_DRAIN_TIMEOUT`]).
    pub fn drain_timeout(mut self, d: Duration) -> Self {
        self.drain_timeout = d;
        self
    }

    /// Inbox receive-poll cadence (default [`DEFAULT_POLL_TIMEOUT`]).
    pub fn poll_timeout(mut self, d: Duration) -> Self {
        self.poll_timeout = d;
        self
    }

    /// Finalise the template and obtain the Layer-A
    /// [`ActorBuilder<NvInfer>`].
    ///
    /// # Errors
    ///
    /// Returns `Err` if `operator_factory` is missing.  All three
    /// per-variant hooks have defaults that are auto-installed when
    /// their setters have not been called.  `downstream` is
    /// optional — callers that route via [`Router::send_to`]
    /// exclusively may omit it.
    pub fn build(self) -> Result<ActorBuilder<NvInfer>> {
        let NvInferBuilder {
            name,
            capacity,
            downstream,
            drain_timeout,
            poll_timeout,
            operator_factory,
            on_inference,
            on_source_eos,
            on_error,
        } = self;
        let operator_factory =
            operator_factory.ok_or_else(|| anyhow!("NvInfer: missing operator_factory"))?;
        let on_inference: OnInferenceHook =
            on_inference.unwrap_or_else(|| Arc::new(NvInfer::default_on_inference(name.clone())));
        let on_source_eos: OnSourceEosHook =
            on_source_eos.unwrap_or_else(|| Arc::new(NvInfer::default_on_source_eos()));
        let on_error: OnErrorHook =
            on_error.unwrap_or_else(|| Arc::new(NvInfer::default_on_error(name.clone())));
        Ok(ActorBuilder::new(name, capacity)
            .poll_timeout(poll_timeout)
            .factory(move |bx| {
                let router: Router<PipelineMsg> = bx.router(downstream.as_ref())?;
                let router_for_cb = router.clone();
                let on_inference_cb = on_inference.clone();
                let on_source_eos_cb = on_source_eos.clone();
                let on_error_cb = on_error.clone();
                let result_cb: OperatorResultCallback = Box::new(move |out| {
                    dispatch_output(
                        out,
                        &on_inference_cb,
                        &on_source_eos_cb,
                        &on_error_cb,
                        &router_for_cb,
                    );
                });
                let operator = (operator_factory)(bx, result_cb)?;
                Ok(NvInfer {
                    operator,
                    router,
                    on_inference,
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

    fn noop_on_inference(_: OperatorInferenceOutput, _: &Router<PipelineMsg>) {}
    fn noop_on_source_eos(_: &str, _: &Router<PipelineMsg>) {}
    fn noop_on_error(_: &NvInferError, _: &Router<PipelineMsg>) {}

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
            .on_inference(noop_on_inference)
            .on_source_eos(noop_on_source_eos)
            .on_error(noop_on_error)
            .drain_timeout(Duration::from_secs(10))
            .poll_timeout(Duration::from_millis(50))
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
            .on_inference(noop_on_inference)
            .on_source_eos(noop_on_source_eos)
            .on_error(noop_on_error)
            .build()
            .unwrap();
    }

    /// All three per-variant hooks have defaults that
    /// [`NvInferBuilder::build`] auto-installs — a builder with only
    /// `.operator_factory(...)` set still produces a valid
    /// `ActorBuilder`.
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
    /// builder's generic hook bounds as-is.
    #[test]
    fn builder_accepts_explicit_defaults() {
        let name = StageName::unnamed(StageKind::Infer);
        let _ = NvInfer::builder(name.clone(), 4)
            .downstream(StageName::unnamed(StageKind::Tracker))
            .operator_factory(|_bx, _cb| unreachable!("not invoked in this test"))
            .on_inference(NvInfer::default_on_inference(name.clone()))
            .on_source_eos(NvInfer::default_on_source_eos())
            .on_error(NvInfer::default_on_error(name))
            .build()
            .unwrap();
    }

    /// `default_on_source_eos` forwards a `PipelineMsg::SourceEos`
    /// carrying the supplied `source_id` via `router.send(..)`.  This
    /// is the only place in the stage where `SourceEos` leaves, so
    /// the contract must stay covered even though the enum-dispatch
    /// test moved out of the sample.
    #[test]
    fn default_on_source_eos_forwards_via_router() {
        let hook = NvInfer::default_on_source_eos();
        let (router, rx) = router_with_default_peer();
        hook("cam-1", &router);
        match rx.try_recv().expect("SourceEos should be forwarded") {
            PipelineMsg::SourceEos { source_id } => assert_eq!(source_id, "cam-1"),
            other => panic!("expected SourceEos, got {other:?}"),
        }
        assert!(rx.try_recv().is_err(), "exactly one message forwarded");
    }

    /// `default_on_error` is log-only — it must not route anything
    /// downstream.  Pair with `default_on_source_eos_forwards_via_router`
    /// to pin the error variant's routing behaviour that used to be
    /// asserted in the sample-side enum-dispatch test.
    #[test]
    fn default_on_error_does_not_route() {
        let hook = NvInfer::default_on_error(StageName::unnamed(StageKind::Infer));
        let (router, rx) = router_with_default_peer();
        let err = NvInferError::PipelineError("synthetic".to_string());
        hook(&err, &router);
        assert!(rx.try_recv().is_err(), "errors must not be routed");
    }
}
