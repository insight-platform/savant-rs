//! [`BitstreamFunction`] — generic user-function actor for the
//! encoded-bitstream side of the pipeline.
//!
//! Structurally parallel to [`DeepStreamFunction`](super::deepstream_function::DeepStreamFunction)
//! but consumes [`EncodedMsg`] (the envelope used by
//! [`Mp4Muxer`](super::mp4_muxer::Mp4Muxer) and
//! [`Decoder`](super::decoder::Decoder)) instead of
//! [`PipelineMsg`](crate::envelopes::PipelineMsg).
//!
//! The stage exposes one hook per data variant
//! ([`on_stream_info`](BitstreamFunctionInboxBuilder::on_stream_info),
//! [`on_packet`](BitstreamFunctionInboxBuilder::on_packet),
//! [`on_frame`](BitstreamFunctionInboxBuilder::on_frame)) plus the
//! usual
//! [`on_source_eos`](BitstreamFunctionInboxBuilder::on_source_eos)
//! and lifecycle hooks, and carries a
//! [`Router<EncodedMsg>`](Router) built at factory time with an
//! optional default peer installed via
//! [`BitstreamFunctionBuilder::downstream`].  Hooks receive the
//! router by shared reference so user code is the single send site
//! — the stage itself never calls `router.send`.
//!
//! # Grouped builder API
//!
//! The stage exposes two hook bundles matching the cross-stage
//! grouped-builder pattern:
//!
//! * [`BitstreamFunctionInbox`] — inbox hooks dispatched on the
//!   actor thread (`on_stream_info`, `on_packet`, `on_frame`,
//!   `on_source_eos`).
//! * [`BitstreamFunctionCommon`] — lifecycle + loop-level knobs
//!   (`on_started`, `stopping`, `poll_timeout`).
//!
//! Build a bitstream function with zero user code:
//!
//! ```ignore
//! use savant_perception::supervisor::{StageKind, StageName};
//! use savant_perception::stages::BitstreamFunction;
//!
//! let builder = BitstreamFunction::builder(
//!     StageName::unnamed(StageKind::BitstreamFunction),
//!     16,
//! ).build();
//! ```
//!
//! The resulting actor drops every `EncodedMsg` data variant and
//! logs `SourceEos` sentinels without terminating (pure-terminus
//! default).  It cooperates with shutdown via the standard
//! [`EncodedMsg::Shutdown`]
//! → [`ShutdownHint::Graceful`](super::super::ShutdownHint::Graceful)
//! path handled by the loop driver — no custom code is required.
//!
//! # Hooks
//!
//! Override per-hook for domain behaviour via the bundles:
//!
//! * [`BitstreamFunctionInboxBuilder::on_stream_info`] — called
//!   with every
//!   [`EncodedMsg::StreamInfo`].
//!   Default: drop.
//! * [`BitstreamFunctionInboxBuilder::on_packet`] — called with
//!   every
//!   [`EncodedMsg::Packet`].
//!   Default: drop.
//! * [`BitstreamFunctionInboxBuilder::on_frame`] — called with
//!   every
//!   [`EncodedMsg::Frame`].
//!   Default: drop.
//! * [`BitstreamFunctionInboxBuilder::on_source_eos`] — called on
//!   every
//!   [`EncodedMsg::SourceEos`].
//!   Returns [`Flow`] so the function can self-terminate on the
//!   first EOS (e.g. single-source pipeline) or stay alive
//!   (default).
//! * [`BitstreamFunctionCommonBuilder::on_started`] /
//!   [`BitstreamFunctionCommonBuilder::stopping`] — optional
//!   lifecycle logs.  Both always installed; default is a single
//!   `info!` log line.
//!
//! All hooks run on the actor's OS thread; they may capture
//! `Send`-safe state by value and call `router.send(...)` to
//! forward to the configured default peer, or
//! `router.send_to(&peer, msg)` for source-id-based routing.
//!
//! # Runtime invariant
//!
//! Every hook slot on the runtime [`BitstreamFunction`] struct is
//! a non-`Option` boxed closure.  The bundle builders'
//! `Option<...>` fields exist purely as "was the setter called?"
//! markers;
//! [`BitstreamFunctionInboxBuilder::build`] and
//! [`BitstreamFunctionCommonBuilder::build`] always substitute the
//! matching default before the [`BitstreamFunctionBuilder`] is
//! finalised.

use std::time::Duration;

use anyhow::Result;

use crate::envelopes::{EncodedMsg, FramePayload, PacketPayload, StreamInfoPayload};
use crate::instrument::enter_callback_span;
use crate::router::Router;
use crate::supervisor::StageName;
use crate::{
    Actor, ActorBuilder, Context, Dispatch, Flow, Handler, ShutdownPayload, SourceEosPayload,
};

/// Closure type for `on_stream_info`.  Observes every
/// [`StreamInfoPayload`] by shared reference.  Returning `Err`
/// aborts the loop.
pub type OnStreamInfoHook = Box<
    dyn FnMut(
            &mut Context<BitstreamFunction>,
            &Router<EncodedMsg>,
            &StreamInfoPayload,
        ) -> Result<()>
        + Send
        + 'static,
>;

/// Closure type for `on_packet`.  Observes every
/// [`PacketPayload`] by shared reference.  Returning `Err` aborts
/// the loop.
pub type OnPacketHook = Box<
    dyn FnMut(&mut Context<BitstreamFunction>, &Router<EncodedMsg>, &PacketPayload) -> Result<()>
        + Send
        + 'static,
>;

/// Closure type for `on_frame`.  Observes every [`FramePayload`]
/// by shared reference.  Returning `Err` aborts the loop.
pub type OnFrameHook = Box<
    dyn FnMut(&mut Context<BitstreamFunction>, &Router<EncodedMsg>, &FramePayload) -> Result<()>
        + Send
        + 'static,
>;

/// Closure type for `on_source_eos`.  Returning [`Flow::Stop`]
/// exits the receive loop on that EOS; [`Flow::Cont`] keeps the
/// function alive.
pub type OnSourceEosHook = Box<
    dyn FnMut(&mut Context<BitstreamFunction>, &Router<EncodedMsg>, &str) -> Result<Flow>
        + Send
        + 'static,
>;

/// Closure type for `on_started` — runs once when the actor's
/// receive loop starts.  Always populated at runtime (the default
/// is a single `info!` log line).
pub type OnStartedHook = Box<dyn FnMut(&mut Context<BitstreamFunction>) + Send + 'static>;

/// User shutdown hook invoked from [`Actor::stopping`] — the
/// stage has no load-bearing built-in cleanup, so the hook body
/// is the only work done on stop.  Runs on the actor thread with
/// full access to the [`Context`].  Always populated at runtime;
/// default is a single `info!` log line.
pub type OnStoppingHook = Box<dyn FnMut(&mut Context<BitstreamFunction>) + Send + 'static>;

/// Generic user-function actor for the encoded-bitstream side of
/// the pipeline.
///
/// Construct with [`BitstreamFunction::builder`].  The actor
/// carries its hooks as boxed closures.  Every slot is non-`Option`
/// by construction — the [`BitstreamFunctionInbox`] /
/// [`BitstreamFunctionCommon`] bundles auto-install the matching
/// default when the user omits a setter.
pub struct BitstreamFunction {
    router: Router<EncodedMsg>,
    stream_info: OnStreamInfoHook,
    packet: OnPacketHook,
    frame: OnFrameHook,
    source_eos: OnSourceEosHook,
    on_started: OnStartedHook,
    on_stopping: OnStoppingHook,
    poll_timeout: Duration,
}

impl BitstreamFunction {
    /// Start a fluent builder for a bitstream function registered
    /// under `name` with inbox capacity `capacity`.
    pub fn builder(name: StageName, capacity: usize) -> BitstreamFunctionBuilder {
        BitstreamFunctionBuilder::new(name, capacity)
    }

    /// Default `on_stream_info` hook — drops every stream-info
    /// sentinel.
    pub fn default_on_stream_info() -> impl FnMut(
        &mut Context<BitstreamFunction>,
        &Router<EncodedMsg>,
        &StreamInfoPayload,
    ) -> Result<()>
           + Send
           + 'static {
        |_ctx, _router, _payload| Ok(())
    }

    /// Default `on_packet` hook — drops every packet.
    pub fn default_on_packet(
    ) -> impl FnMut(&mut Context<BitstreamFunction>, &Router<EncodedMsg>, &PacketPayload) -> Result<()>
           + Send
           + 'static {
        |_ctx, _router, _payload| Ok(())
    }

    /// Default `on_frame` hook — drops every pre-built frame.
    pub fn default_on_frame(
    ) -> impl FnMut(&mut Context<BitstreamFunction>, &Router<EncodedMsg>, &FramePayload) -> Result<()>
           + Send
           + 'static {
        |_ctx, _router, _payload| Ok(())
    }

    /// Default `on_source_eos` hook — logs at `info` level and
    /// returns `Ok(Flow::Cont)` so the function stays alive through
    /// an arbitrary number of per-source drainages.
    pub fn default_on_source_eos(
    ) -> impl FnMut(&mut Context<BitstreamFunction>, &Router<EncodedMsg>, &str) -> Result<Flow>
           + Send
           + 'static {
        |ctx, _router, source_id| {
            log::info!("[{}] SourceEos {source_id}: continuing", ctx.own_name());
            Ok(Flow::Cont)
        }
    }

    /// Default `on_started` hook — logs
    /// `[{stage}] bitstream function started` at `info` level.
    pub fn default_on_started() -> impl FnMut(&mut Context<BitstreamFunction>) + Send + 'static {
        |ctx| {
            log::info!("[{}] bitstream function started", ctx.own_name());
        }
    }

    /// Default user shutdown hook — emits one info-level log line.
    /// The stage has no load-bearing built-in cleanup, so this
    /// hook body is the only work done on stop.
    pub fn default_on_stopping() -> impl FnMut(&mut Context<BitstreamFunction>) + Send + 'static {
        |ctx| {
            log::info!("[{}] bitstream function stopping", ctx.own_name());
        }
    }
}

impl Actor for BitstreamFunction {
    type Msg = EncodedMsg;

    fn handle(&mut self, msg: EncodedMsg, ctx: &mut Context<Self>) -> Result<Flow> {
        msg.dispatch(self, ctx)
    }

    fn poll_timeout(&self) -> Duration {
        self.poll_timeout
    }

    fn started(&mut self, ctx: &mut Context<Self>) -> Result<()> {
        (self.on_started)(ctx);
        Ok(())
    }

    fn stopping(&mut self, ctx: &mut Context<Self>) {
        (self.on_stopping)(ctx);
    }
}

impl Handler<StreamInfoPayload> for BitstreamFunction {
    fn handle(&mut self, msg: StreamInfoPayload, ctx: &mut Context<Self>) -> Result<Flow> {
        // StreamInfo carries no frame — no parent context, no span.
        (self.stream_info)(ctx, &self.router, &msg)?;
        Ok(Flow::Cont)
    }
}

impl Handler<PacketPayload> for BitstreamFunction {
    fn handle(&mut self, msg: PacketPayload, ctx: &mut Context<Self>) -> Result<Flow> {
        // Packets are pre-decode bitstream — no frame, no span.
        (self.packet)(ctx, &self.router, &msg)?;
        Ok(Flow::Cont)
    }
}

impl Handler<FramePayload> for BitstreamFunction {
    fn handle(&mut self, msg: FramePayload, ctx: &mut Context<Self>) -> Result<Flow> {
        let _g = enter_callback_span(
            &msg.frame,
            "on_frame",
            ctx.pipeline_name(),
            &ctx.own_name().to_string(),
        );
        (self.frame)(ctx, &self.router, &msg)?;
        Ok(Flow::Cont)
    }
}

impl Handler<SourceEosPayload> for BitstreamFunction {
    fn handle(&mut self, msg: SourceEosPayload, ctx: &mut Context<Self>) -> Result<Flow> {
        // No frame on a SourceEos — no span.
        (self.source_eos)(ctx, &self.router, &msg.source_id)
    }
}

/// Default no-op handler: the framework's loop driver already
/// consumed the shutdown hint from
/// [`EncodedMsg::Shutdown`]
/// via [`Envelope::as_shutdown`](super::super::Envelope::as_shutdown),
/// so the in-band sentinel has no remaining work to do here.
impl Handler<ShutdownPayload> for BitstreamFunction {}

/// Inbox hook bundle for [`BitstreamFunction`] — one branch per
/// inbox message kind beyond the cooperative-shutdown sentinel.
/// Built through [`BitstreamFunctionInbox::builder`] and handed to
/// [`BitstreamFunctionBuilder::inbox`].  Omitted branches
/// auto-install the matching `BitstreamFunction::default_on_*` at
/// build time.
pub struct BitstreamFunctionInbox {
    stream_info: OnStreamInfoHook,
    packet: OnPacketHook,
    frame: OnFrameHook,
    source_eos: OnSourceEosHook,
}

impl BitstreamFunctionInbox {
    /// Start a builder that auto-installs every default on
    /// [`BitstreamFunctionInboxBuilder::build`].
    pub fn builder() -> BitstreamFunctionInboxBuilder {
        BitstreamFunctionInboxBuilder::new()
    }
}

impl Default for BitstreamFunctionInbox {
    fn default() -> Self {
        BitstreamFunctionInboxBuilder::new().build()
    }
}

/// Fluent builder for [`BitstreamFunctionInbox`].
pub struct BitstreamFunctionInboxBuilder {
    stream_info: Option<OnStreamInfoHook>,
    packet: Option<OnPacketHook>,
    frame: Option<OnFrameHook>,
    source_eos: Option<OnSourceEosHook>,
}

impl BitstreamFunctionInboxBuilder {
    /// Empty bundle — every hook defaults to the matching
    /// `BitstreamFunction::default_on_*` at [`build`](Self::build)
    /// time.
    pub fn new() -> Self {
        Self {
            stream_info: None,
            packet: None,
            frame: None,
            source_eos: None,
        }
    }

    /// Install a custom `on_stream_info` hook.  Fires on every
    /// [`EncodedMsg::StreamInfo`].
    /// The hook receives the payload by shared reference along
    /// with the stage's [`Router<EncodedMsg>`] so it can forward
    /// via `router.send(...)` if desired.  Returning `Err` aborts
    /// the loop and signals the supervisor via the
    /// [`StageExitGuard`](super::super::StageExitGuard).
    pub fn on_stream_info<F>(mut self, f: F) -> Self
    where
        F: FnMut(
                &mut Context<BitstreamFunction>,
                &Router<EncodedMsg>,
                &StreamInfoPayload,
            ) -> Result<()>
            + Send
            + 'static,
    {
        self.stream_info = Some(Box::new(f));
        self
    }

    /// Install a custom `on_packet` hook.  Fires on every
    /// [`EncodedMsg::Packet`].
    pub fn on_packet<F>(mut self, f: F) -> Self
    where
        F: FnMut(
                &mut Context<BitstreamFunction>,
                &Router<EncodedMsg>,
                &PacketPayload,
            ) -> Result<()>
            + Send
            + 'static,
    {
        self.packet = Some(Box::new(f));
        self
    }

    /// Install a custom `on_frame` hook.  Fires on every
    /// [`EncodedMsg::Frame`].
    pub fn on_frame<F>(mut self, f: F) -> Self
    where
        F: FnMut(&mut Context<BitstreamFunction>, &Router<EncodedMsg>, &FramePayload) -> Result<()>
            + Send
            + 'static,
    {
        self.frame = Some(Box::new(f));
        self
    }

    /// Install a custom `on_source_eos` hook.  Return
    /// [`Flow::Stop`] to self-terminate on the first EOS observed
    /// (single-source pipelines); [`Flow::Cont`] keeps the function
    /// alive through an arbitrary number of per-source drainages
    /// (multiplexed pipelines).
    pub fn on_source_eos<F>(mut self, f: F) -> Self
    where
        F: FnMut(&mut Context<BitstreamFunction>, &Router<EncodedMsg>, &str) -> Result<Flow>
            + Send
            + 'static,
    {
        self.source_eos = Some(Box::new(f));
        self
    }

    /// Finalise the bundle, substituting defaults for every omitted
    /// setter.
    pub fn build(self) -> BitstreamFunctionInbox {
        let BitstreamFunctionInboxBuilder {
            stream_info,
            packet,
            frame,
            source_eos,
        } = self;
        BitstreamFunctionInbox {
            stream_info: stream_info
                .unwrap_or_else(|| Box::new(BitstreamFunction::default_on_stream_info())),
            packet: packet.unwrap_or_else(|| Box::new(BitstreamFunction::default_on_packet())),
            frame: frame.unwrap_or_else(|| Box::new(BitstreamFunction::default_on_frame())),
            source_eos: source_eos
                .unwrap_or_else(|| Box::new(BitstreamFunction::default_on_source_eos())),
        }
    }
}

impl Default for BitstreamFunctionInboxBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Lifecycle + loop-level knob bundle for [`BitstreamFunction`].
/// Built through [`BitstreamFunctionCommon::builder`] and handed to
/// [`BitstreamFunctionBuilder::common`].
pub struct BitstreamFunctionCommon {
    on_started: OnStartedHook,
    on_stopping: OnStoppingHook,
    poll_timeout: Duration,
}

impl BitstreamFunctionCommon {
    /// Start a builder seeded with defaults.
    pub fn builder() -> BitstreamFunctionCommonBuilder {
        BitstreamFunctionCommonBuilder::new()
    }
}

impl Default for BitstreamFunctionCommon {
    fn default() -> Self {
        BitstreamFunctionCommonBuilder::new().build()
    }
}

/// Fluent builder for [`BitstreamFunctionCommon`].
pub struct BitstreamFunctionCommonBuilder {
    on_started: Option<OnStartedHook>,
    on_stopping: Option<OnStoppingHook>,
    poll_timeout: Option<Duration>,
}

impl BitstreamFunctionCommonBuilder {
    /// Default receive-poll cadence — matches
    /// [`Actor::poll_timeout`]'s
    /// 200 ms default.
    pub const DEFAULT_POLL: Duration = Duration::from_millis(200);

    /// Empty bundle — both hooks default to the matching
    /// `BitstreamFunction::default_*` and `poll_timeout` defaults
    /// to [`Self::DEFAULT_POLL`].
    pub fn new() -> Self {
        Self {
            on_started: None,
            on_stopping: None,
            poll_timeout: None,
        }
    }

    /// Install a custom `started` hook; the default logs
    /// `[{stage}] bitstream function started` at `info` level.
    pub fn on_started<F>(mut self, f: F) -> Self
    where
        F: FnMut(&mut Context<BitstreamFunction>) + Send + 'static,
    {
        self.on_started = Some(Box::new(f));
        self
    }

    /// Install a custom `stopping` hook; the default
    /// ([`BitstreamFunction::default_on_stopping`]) logs
    /// `[{stage}] bitstream function stopping`.  The stage has
    /// no load-bearing built-in cleanup on [`Actor::stopping`], so
    /// this hook's body is the **only** work done on stop.
    pub fn on_stopping<F>(mut self, f: F) -> Self
    where
        F: FnMut(&mut Context<BitstreamFunction>) + Send + 'static,
    {
        self.on_stopping = Some(Box::new(f));
        self
    }

    /// Override the inbox receive-poll cadence.
    pub fn poll_timeout(mut self, d: Duration) -> Self {
        self.poll_timeout = Some(d);
        self
    }

    /// Finalise the bundle, substituting defaults for every omitted
    /// setter.
    pub fn build(self) -> BitstreamFunctionCommon {
        let BitstreamFunctionCommonBuilder {
            on_started,
            on_stopping,
            poll_timeout,
        } = self;
        BitstreamFunctionCommon {
            on_started: on_started.unwrap_or_else(|| Box::new(BitstreamFunction::default_on_started())),
            on_stopping: on_stopping.unwrap_or_else(|| Box::new(BitstreamFunction::default_on_stopping())),
            poll_timeout: poll_timeout.unwrap_or(Self::DEFAULT_POLL),
        }
    }
}

impl Default for BitstreamFunctionCommonBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Fluent builder for [`BitstreamFunction`].
///
/// Every hook defaults to the "pure terminus" behaviour described
/// in [`BitstreamFunction`]'s module docs.  Install the
/// [`BitstreamFunctionInbox`] and [`BitstreamFunctionCommon`]
/// bundles via [`BitstreamFunctionBuilder::inbox`] /
/// [`BitstreamFunctionBuilder::common`] to override.  Terminate
/// with [`BitstreamFunctionBuilder::build`] to obtain an
/// [`ActorBuilder<BitstreamFunction>`] suitable for
/// [`System::register_actor`](super::super::System::register_actor).
pub struct BitstreamFunctionBuilder {
    name: StageName,
    capacity: usize,
    downstream: Option<StageName>,
    inbox: Option<BitstreamFunctionInbox>,
    common: Option<BitstreamFunctionCommon>,
}

impl BitstreamFunctionBuilder {
    /// Build a bitstream function under `name` with inbox capacity
    /// `capacity` and the default hooks — drop every inbound
    /// variant, log+continue on every EOS.
    pub fn new(name: StageName, capacity: usize) -> Self {
        Self {
            name,
            capacity,
            downstream: None,
            inbox: None,
            common: None,
        }
    }

    /// Optional default peer installed on the
    /// [`Router<EncodedMsg>`] handed to every variant hook.  When
    /// set, user hooks can use `router.send(msg)` to forward; when
    /// unset, `router.send(...)` returns `false` and
    /// `router.send_to(&peer, ...)` must be used instead for
    /// explicit per-call routing.
    pub fn downstream(mut self, peer: StageName) -> Self {
        self.downstream = Some(peer);
        self
    }

    /// Install a [`BitstreamFunctionInbox`] bundle — inbox hook
    /// slots (`on_stream_info`, `on_packet`, `on_frame`,
    /// `on_source_eos`).  Omitting this call is equivalent to
    /// `.inbox(BitstreamFunctionInbox::default())`, which wires
    /// every slot to the matching
    /// `BitstreamFunction::default_on_*`.
    pub fn inbox(mut self, i: BitstreamFunctionInbox) -> Self {
        self.inbox = Some(i);
        self
    }

    /// Install a [`BitstreamFunctionCommon`] bundle — lifecycle
    /// and loop-level knobs (`on_started`, `stopping`,
    /// `poll_timeout`).  Omitting this call is equivalent to
    /// `.common(BitstreamFunctionCommon::default())`.
    pub fn common(mut self, c: BitstreamFunctionCommon) -> Self {
        self.common = Some(c);
        self
    }

    /// Finalise the stage configuration and obtain the
    /// Layer-A [`ActorBuilder<BitstreamFunction>`] ready for
    /// [`System::register_actor`](super::super::System::register_actor).
    pub fn build(self) -> ActorBuilder<BitstreamFunction> {
        let Self {
            name,
            capacity,
            downstream,
            inbox,
            common,
        } = self;
        let BitstreamFunctionInbox {
            stream_info,
            packet,
            frame,
            source_eos,
        } = inbox.unwrap_or_default();
        let BitstreamFunctionCommon {
            on_started,
            on_stopping,
            poll_timeout,
        } = common.unwrap_or_default();
        // Hooks are owned by the actor and must be moved into the
        // factory closure.  We wrap the config into an Option so
        // the `FnOnce`-style take-once move is obvious to the
        // reader and to the borrow checker (the factory closure
        // here is called once per registration).
        let mut slots = Some((stream_info, packet, frame, source_eos, on_started, on_stopping));
        ActorBuilder::new(name, capacity)
            .poll_timeout(poll_timeout)
            .factory(move |bx| {
                let router: Router<EncodedMsg> = bx.router(downstream.as_ref())?;
                let (stream_info, packet, frame, source_eos, on_started, on_stopping) = slots
                    .take()
                    .expect("BitstreamFunction factory invoked more than once");
                Ok(BitstreamFunction {
                    router,
                    stream_info,
                    packet,
                    frame,
                    source_eos,
                    on_started,
                    on_stopping,
                    poll_timeout,
                })
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::supervisor::{StageKind, StageName};
    use crate::{ShutdownAction, ShutdownCause, ShutdownCtx, System};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// The default bitstream function stops when the supervisor
    /// signals shutdown (via `sys.run`'s in-band broadcast
    /// triggered by the absence of any other stage returning an
    /// exit trigger).
    #[test]
    fn default_bitstream_function_runs_and_stops_on_broadcast() {
        let mut sys = System::new()
            .install_ctrlc_handler(false)
            .quiescence(Duration::from_millis(0))
            .on_shutdown(|_cause: ShutdownCause, _ctx: &mut ShutdownCtx<'_>| {
                Ok(ShutdownAction::Broadcast {
                    grace: None,
                    reason: std::borrow::Cow::Borrowed("test"),
                })
            });
        let addr = sys
            .register_actor(
                BitstreamFunction::builder(StageName::unnamed(StageKind::BitstreamFunction), 4).build(),
            )
            .unwrap();

        addr.send(EncodedMsg::source_eos("s1")).unwrap();

        addr.send(EncodedMsg::Shutdown {
            grace: None,
            reason: std::borrow::Cow::Borrowed("done"),
        })
        .unwrap();

        let report = sys.run().unwrap();
        assert!(report.stage_results[0].1.is_ok());
    }

    /// Custom `on_source_eos` can self-terminate on the first EOS.
    #[test]
    fn custom_source_eos_stops_function() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();
        let mut sys = System::new()
            .install_ctrlc_handler(false)
            .quiescence(Duration::from_millis(0));
        let addr = sys
            .register_actor(
                BitstreamFunction::builder(StageName::unnamed(StageKind::BitstreamFunction), 4)
                    .inbox(
                        BitstreamFunctionInbox::builder()
                            .on_source_eos(move |_ctx, _router, sid| {
                                counter_clone.fetch_add(1, Ordering::Relaxed);
                                let _ = sid;
                                Ok(Flow::Stop)
                            })
                            .build(),
                    )
                    .build(),
            )
            .unwrap();

        addr.send(EncodedMsg::source_eos("s1")).unwrap();

        let report = sys.run().unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 1);
        assert!(report.stage_results[0].1.is_ok());
    }

    /// Compile-only verification: a full-hook builder (inbox +
    /// common) is accepted and every default forwarder slots into
    /// the matching generic bound.
    #[test]
    fn custom_hooks_are_installed() {
        let _ = BitstreamFunction::builder(StageName::unnamed(StageKind::BitstreamFunction), 2)
            .downstream(StageName::unnamed(StageKind::BitstreamSink))
            .inbox(
                BitstreamFunctionInbox::builder()
                    .on_stream_info(|_ctx, _router, _pl| Ok(()))
                    .on_packet(|_ctx, _router, _pl| Ok(()))
                    .on_frame(|_ctx, _router, _pl| Ok(()))
                    .on_source_eos(|_ctx, _router, _sid| Ok(Flow::Cont))
                    .build(),
            )
            .common(
                BitstreamFunctionCommon::builder()
                    .on_started(|ctx| log::debug!("started {}", ctx.own_name()))
                    .on_stopping(|ctx| log::debug!("stopping {}", ctx.own_name()))
                    .poll_timeout(Duration::from_millis(50))
                    .build(),
            )
            .build();
    }

    /// Confirms that each `default_on_*` associated function slots
    /// into the bundle builders' generic hook bounds as-is.
    #[test]
    fn default_forwarders_compile() {
        let _ = BitstreamFunction::builder(StageName::unnamed(StageKind::BitstreamFunction), 2)
            .inbox(
                BitstreamFunctionInbox::builder()
                    .on_stream_info(BitstreamFunction::default_on_stream_info())
                    .on_packet(BitstreamFunction::default_on_packet())
                    .on_frame(BitstreamFunction::default_on_frame())
                    .on_source_eos(BitstreamFunction::default_on_source_eos())
                    .build(),
            )
            .common(
                BitstreamFunctionCommon::builder()
                    .on_started(BitstreamFunction::default_on_started())
                    .on_stopping(BitstreamFunction::default_on_stopping())
                    .build(),
            )
            .build();
    }

    /// Every slot on the runtime [`BitstreamFunction`] struct is
    /// non-`Option` — the bundle builders always substitute a
    /// default.  Compile-only check via a destructuring pattern
    /// that refuses any `Option<_>` wrapper.
    #[test]
    fn runtime_invariant_all_hooks_populated() {
        use crate::context::BuildCtx;
        use crate::registry::Registry;
        use crate::shared::SharedStore;
        let sb = BitstreamFunction::builder(StageName::unnamed(StageKind::BitstreamFunction), 2).build();
        let parts = sb.into_parts();
        let reg = Arc::new(Registry::new());
        let shared = Arc::new(SharedStore::new());
        let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let pn: Arc<str> = Arc::from("test");
        let sm = crate::stage_metrics::StageMetrics::new(parts.name.to_string());
        let bx = BuildCtx::new(&parts.name, &pn, &reg, &shared, &stop_flag, &sm);
        let func = (parts.factory)(&bx).expect("factory resolves");
        let BitstreamFunction {
            router: _,
            stream_info: _,
            packet: _,
            frame: _,
            source_eos: _,
            on_started: _,
            on_stopping: _,
            poll_timeout: _,
        } = func;
    }
}
