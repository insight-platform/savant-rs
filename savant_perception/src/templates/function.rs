//! [`Function`] — generic user-function actor (default: pipeline
//! terminus).
//!
//! A [`Function`] is the simplest actor shape in the sample: it
//! consumes every
//! [`PipelineMsg`](crate::envelopes::PipelineMsg)
//! variant the upstream stage emits and — by default — drops the
//! payload, releasing any resources (e.g. GPU slots held by
//! [`SharedBuffer`](deepstream_buffers::SharedBuffer)) back to the
//! pipeline.  The name **`Function`** (previously `Sink`) reflects
//! the actor's real purpose: it hosts an arbitrary user-supplied
//! callback (`on_delivery`) that receives every
//! `(VideoFrameProxy, SharedBuffer)` batch by **shared reference**
//! and is therefore well-suited for terminals *and* for inline
//! analytics / custom processing stages.
//!
//! # Grouped builder API
//!
//! The template exposes two hook bundles matching the cross-template
//! grouped-builder pattern:
//!
//! * [`FunctionInbox`] — inbox hooks dispatched on the actor thread
//!   (`on_delivery`, `on_source_eos`).
//! * [`FunctionCommon`] — lifecycle + loop-level knobs
//!   (`on_started`, `stopping`, `poll_timeout`).
//!
//! Build a function with zero user code:
//!
//! ```ignore
//! use savant_perception::supervisor::{StageKind, StageName};
//! use savant_perception::templates::Function;
//!
//! let builder = Function::builder(StageName::unnamed(StageKind::Function), 16).build();
//! ```
//!
//! The resulting actor drops every delivery and logs `SourceEos`
//! sentinels without terminating (pure-terminus default).  It
//! cooperates with shutdown via the standard
//! [`PipelineMsg::Shutdown`](crate::envelopes::PipelineMsg::Shutdown)
//! → [`ShutdownHint::Graceful`](super::super::ShutdownHint::Graceful)
//! path handled by the loop driver — no custom code is required.
//!
//! # Hooks
//!
//! Override per-hook for domain behaviour via the bundles:
//!
//! * [`FunctionInboxBuilder::on_delivery`] — called with every
//!   unsealed `(VideoFrameProxy, SharedBuffer)` batch (both the
//!   single-delivery and batched-delivery variants funnel through
//!   here after unsealing).  The batch is passed by **shared
//!   slice reference** so the hook body can route it into nested
//!   handlers (e.g. inference / tracking helpers) without
//!   consuming it.  The template drops the owned `Vec` as soon as
//!   the hook returns; clone the frames / buffers inside the hook
//!   if they need to outlive the call.  Default: drop (no-op).
//! * [`FunctionInboxBuilder::on_source_eos`] — called on every
//!   [`PipelineMsg::SourceEos`](crate::envelopes::PipelineMsg::SourceEos).
//!   Returns [`Flow`] so the function can self-terminate on the
//!   first EOS (e.g. single-source pipeline) or stay alive
//!   (default).
//! * [`FunctionCommonBuilder::on_started`] /
//!   [`FunctionCommonBuilder::stopping`] — optional lifecycle logs.
//!   Both always installed; default is a single `info!` log line.
//!
//! All hooks run on the actor's OS thread; they may capture
//! `Send`-safe state by value.
//!
//! # Runtime invariant
//!
//! Every hook slot on the runtime [`Function`] struct is a
//! non-`Option` boxed closure.  The bundle builders' `Option<...>`
//! fields exist purely as "was the setter called?" markers;
//! [`FunctionInboxBuilder::build`] and
//! [`FunctionCommonBuilder::build`] always substitute the matching
//! default before the [`FunctionBuilder`] is finalised.

use std::time::Duration;

use anyhow::Result;

use crate::envelopes::{BatchDelivery, PipelineMsg, SingleDelivery};
use crate::supervisor::StageName;
use crate::{
    Actor, ActorBuilder, Context, Dispatch, Flow, Handler, ShutdownPayload, SourceEosPayload,
};
use deepstream_buffers::SharedBuffer;
use savant_core::primitives::frame::VideoFrameProxy;

/// Closure type for `on_delivery`: observes a batch of
/// `(frame, buffer)` pairs by **shared reference**.  The owned
/// `Vec` is held on the actor thread and dropped automatically as
/// soon as the hook returns; hooks that need to retain individual
/// frames or buffers must clone them explicitly.  Dropping the
/// argument (i.e. doing nothing) is a no-op terminus;
/// clone-and-forward is a tee pattern; dispatching to nested
/// helpers (e.g. inference / tracking / encoding) without
/// consuming is now the natural shape.
pub type DeliveryHook = Box<
    dyn FnMut(&[(VideoFrameProxy, SharedBuffer)], &mut Context<Function>) -> Result<()>
        + Send
        + 'static,
>;

/// Closure type for `on_source_eos`.  Returning [`Flow::Stop`] exits
/// the receive loop on that EOS; [`Flow::Cont`] keeps the function
/// alive.
pub type EosHook = Box<dyn FnMut(&str, &mut Context<Function>) -> Result<Flow> + Send + 'static>;

/// Closure type for `on_started` — runs once when the actor's
/// receive loop starts.  Always populated at runtime (the default
/// is a single `info!` log line matching the legacy "no hook set"
/// behaviour).
pub type OnStartedHook = Box<dyn FnMut(&mut Context<Function>) + Send + 'static>;

/// User shutdown hook invoked from [`Actor::stopping`] — the
/// template has no load-bearing built-in cleanup, so the hook body
/// is the only work done on stop.  Runs on the actor thread with
/// full access to the [`Context`].  Always populated at runtime;
/// default is a single `info!` log line.
pub type OnStoppingHook = Box<dyn FnMut(&mut Context<Function>) + Send + 'static>;

/// Generic user-function actor (default: pipeline terminus).
///
/// Construct with [`Function::builder`].  The actor carries its
/// hooks as boxed closures.  Every slot is non-`Option` by
/// construction — the [`FunctionInbox`] / [`FunctionCommon`]
/// bundles auto-install the matching default when the user omits
/// a setter.
pub struct Function {
    delivery: DeliveryHook,
    source_eos: EosHook,
    started: OnStartedHook,
    stopping: OnStoppingHook,
    poll_timeout: Duration,
}

impl Function {
    /// Start a fluent builder for a function registered under
    /// `name` with inbox capacity `capacity`.
    pub fn builder(name: StageName, capacity: usize) -> FunctionBuilder {
        FunctionBuilder::new(name, capacity)
    }

    /// Default `on_delivery` hook — drops every unsealed batch.
    pub fn default_on_delivery(
    ) -> impl FnMut(&[(VideoFrameProxy, SharedBuffer)], &mut Context<Function>) -> Result<()>
           + Send
           + 'static {
        |_pairs, _ctx| Ok(())
    }

    /// Default `on_source_eos` hook — logs at `info` level and
    /// returns `Ok(Flow::Cont)` so the function stays alive through
    /// an arbitrary number of per-source drainages.
    pub fn default_on_source_eos(
    ) -> impl FnMut(&str, &mut Context<Function>) -> Result<Flow> + Send + 'static {
        |source_id, ctx| {
            log::info!("[{}] SourceEos {source_id}: continuing", ctx.own_name());
            Ok(Flow::Cont)
        }
    }

    /// Default `on_started` hook — logs `[{stage}] function started`
    /// at `info` level.  Matches the legacy "no hook set" branch.
    pub fn default_on_started() -> impl FnMut(&mut Context<Function>) + Send + 'static {
        |ctx| {
            log::info!("[{}] function started", ctx.own_name());
        }
    }

    /// Default user shutdown hook — reproduces the "one info-level
    /// log line" behaviour so omitting the bundle setter remains
    /// equivalent to the legacy unset branch.  The template has no
    /// load-bearing built-in cleanup, so this hook body is the
    /// only work done on stop.
    pub fn default_stopping() -> impl FnMut(&mut Context<Function>) + Send + 'static {
        |ctx| {
            log::info!("[{}] function stopping", ctx.own_name());
        }
    }
}

impl Actor for Function {
    type Msg = PipelineMsg;

    fn handle(&mut self, msg: PipelineMsg, ctx: &mut Context<Self>) -> Result<Flow> {
        msg.dispatch(self, ctx)
    }

    fn poll_timeout(&self) -> Duration {
        self.poll_timeout
    }

    fn started(&mut self, ctx: &mut Context<Self>) -> Result<()> {
        (self.started)(ctx);
        Ok(())
    }

    fn stopping(&mut self, ctx: &mut Context<Self>) {
        (self.stopping)(ctx);
    }
}

impl Handler<SingleDelivery> for Function {
    fn handle(&mut self, msg: SingleDelivery, ctx: &mut Context<Self>) -> Result<Flow> {
        let pairs = PipelineMsg::Delivery(msg.0).into_pairs();
        (self.delivery)(&pairs, ctx)?;
        Ok(Flow::Cont)
    }
}

impl Handler<BatchDelivery> for Function {
    fn handle(&mut self, msg: BatchDelivery, ctx: &mut Context<Self>) -> Result<Flow> {
        let pairs = msg.0.unseal();
        (self.delivery)(&pairs, ctx)?;
        Ok(Flow::Cont)
    }
}

impl Handler<SourceEosPayload> for Function {
    fn handle(&mut self, msg: SourceEosPayload, ctx: &mut Context<Self>) -> Result<Flow> {
        (self.source_eos)(&msg.source_id, ctx)
    }
}

/// Default no-op handler: the framework's loop driver already
/// consumed the shutdown hint from [`PipelineMsg::Shutdown`] via
/// [`Envelope::as_shutdown`](super::super::Envelope::as_shutdown),
/// so the in-band sentinel has no remaining work to do here.
impl Handler<ShutdownPayload> for Function {}

/// Inbox hook bundle for [`Function`] — one branch per inbox
/// message kind.  Built through [`FunctionInbox::builder`] and
/// handed to [`FunctionBuilder::inbox`].  Omitted branches
/// auto-install the matching `Function::default_on_*` at build
/// time.
pub struct FunctionInbox {
    delivery: DeliveryHook,
    source_eos: EosHook,
}

impl FunctionInbox {
    /// Start a builder that auto-installs every default on
    /// [`FunctionInboxBuilder::build`].
    pub fn builder() -> FunctionInboxBuilder {
        FunctionInboxBuilder::new()
    }
}

impl Default for FunctionInbox {
    fn default() -> Self {
        FunctionInboxBuilder::new().build()
    }
}

/// Fluent builder for [`FunctionInbox`].
pub struct FunctionInboxBuilder {
    delivery: Option<DeliveryHook>,
    source_eos: Option<EosHook>,
}

impl FunctionInboxBuilder {
    /// Empty bundle — both hooks default to the matching
    /// `Function::default_on_*` at [`build`](Self::build) time.
    pub fn new() -> Self {
        Self {
            delivery: None,
            source_eos: None,
        }
    }

    /// Install a custom delivery hook.  The hook receives the
    /// unsealed batch of `(VideoFrameProxy, SharedBuffer)` pairs by
    /// **shared slice reference**, which lets the body forward the
    /// batch to nested helpers (inference / tracking / encoding)
    /// without consuming it.  Returning `Err` aborts the loop and
    /// signals the supervisor via the
    /// [`StageExitGuard`](super::super::StageExitGuard).
    /// Doing nothing with the argument is pure-terminus behaviour
    /// (the default); cloning individual frames or buffers out is
    /// a tee pattern; the template drops the owned `Vec`
    /// automatically as soon as this hook returns.
    pub fn on_delivery<F>(mut self, f: F) -> Self
    where
        F: FnMut(&[(VideoFrameProxy, SharedBuffer)], &mut Context<Function>) -> Result<()>
            + Send
            + 'static,
    {
        self.delivery = Some(Box::new(f));
        self
    }

    /// Install a custom source-EOS hook.  Return [`Flow::Stop`] to
    /// self-terminate on the first EOS observed (single-source
    /// pipelines); [`Flow::Cont`] keeps the function alive through
    /// an arbitrary number of per-source drainages (multiplexed
    /// pipelines).
    pub fn on_source_eos<F>(mut self, f: F) -> Self
    where
        F: FnMut(&str, &mut Context<Function>) -> Result<Flow> + Send + 'static,
    {
        self.source_eos = Some(Box::new(f));
        self
    }

    /// Finalise the bundle, substituting defaults for every omitted
    /// setter.
    pub fn build(self) -> FunctionInbox {
        let FunctionInboxBuilder {
            delivery,
            source_eos,
        } = self;
        FunctionInbox {
            delivery: delivery.unwrap_or_else(|| Box::new(Function::default_on_delivery())),
            source_eos: source_eos.unwrap_or_else(|| Box::new(Function::default_on_source_eos())),
        }
    }
}

impl Default for FunctionInboxBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Lifecycle + loop-level knob bundle for [`Function`].  Built
/// through [`FunctionCommon::builder`] and handed to
/// [`FunctionBuilder::common`].
pub struct FunctionCommon {
    started: OnStartedHook,
    stopping: OnStoppingHook,
    poll_timeout: Duration,
}

impl FunctionCommon {
    /// Start a builder seeded with defaults.
    pub fn builder() -> FunctionCommonBuilder {
        FunctionCommonBuilder::new()
    }
}

impl Default for FunctionCommon {
    fn default() -> Self {
        FunctionCommonBuilder::new().build()
    }
}

/// Fluent builder for [`FunctionCommon`].
pub struct FunctionCommonBuilder {
    started: Option<OnStartedHook>,
    stopping: Option<OnStoppingHook>,
    poll_timeout: Option<Duration>,
}

impl FunctionCommonBuilder {
    /// Default receive-poll cadence — matches
    /// [`Actor::poll_timeout`](crate::Actor::poll_timeout)'s
    /// 200 ms default.
    pub const DEFAULT_POLL: Duration = Duration::from_millis(200);

    /// Empty bundle — both hooks default to the matching
    /// `Function::default_*` and `poll_timeout` defaults to
    /// [`Self::DEFAULT_POLL`].
    pub fn new() -> Self {
        Self {
            started: None,
            stopping: None,
            poll_timeout: None,
        }
    }

    /// Install a custom `started` hook; the default logs
    /// `[{stage}] function started` at `info` level.
    pub fn on_started<F>(mut self, f: F) -> Self
    where
        F: FnMut(&mut Context<Function>) + Send + 'static,
    {
        self.started = Some(Box::new(f));
        self
    }

    /// Install a custom `stopping` hook; the default
    /// ([`Function::default_stopping`]) logs
    /// `[{stage}] function stopping`.  The template has no
    /// load-bearing built-in cleanup on [`Actor::stopping`], so
    /// this hook's body is the **only** work done on stop.
    pub fn stopping<F>(mut self, f: F) -> Self
    where
        F: FnMut(&mut Context<Function>) + Send + 'static,
    {
        self.stopping = Some(Box::new(f));
        self
    }

    /// Override the inbox receive-poll cadence.
    pub fn poll_timeout(mut self, d: Duration) -> Self {
        self.poll_timeout = Some(d);
        self
    }

    /// Finalise the bundle, substituting defaults for every omitted
    /// setter.
    pub fn build(self) -> FunctionCommon {
        let FunctionCommonBuilder {
            started,
            stopping,
            poll_timeout,
        } = self;
        FunctionCommon {
            started: started.unwrap_or_else(|| Box::new(Function::default_on_started())),
            stopping: stopping.unwrap_or_else(|| Box::new(Function::default_stopping())),
            poll_timeout: poll_timeout.unwrap_or(Self::DEFAULT_POLL),
        }
    }
}

impl Default for FunctionCommonBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Fluent builder for [`Function`].
///
/// Every hook defaults to the "pure terminus" behaviour described
/// in [`Function`]'s module docs.  Install the [`FunctionInbox`]
/// and [`FunctionCommon`] bundles via [`FunctionBuilder::inbox`] /
/// [`FunctionBuilder::common`] to override.  Terminate with
/// [`FunctionBuilder::build`] to obtain an
/// [`ActorBuilder<Function>`] suitable for
/// [`System::register_actor`](super::super::System::register_actor).
pub struct FunctionBuilder {
    name: StageName,
    capacity: usize,
    inbox: Option<FunctionInbox>,
    common: Option<FunctionCommon>,
}

impl FunctionBuilder {
    /// Build a function under `name` with inbox capacity `capacity`
    /// and the default hooks — drop every delivery, log+continue
    /// on every EOS.
    pub fn new(name: StageName, capacity: usize) -> Self {
        Self {
            name,
            capacity,
            inbox: None,
            common: None,
        }
    }

    /// Install a [`FunctionInbox`] bundle — inbox hook slots
    /// (`on_delivery`, `on_source_eos`).  Omitting this call is
    /// equivalent to `.inbox(FunctionInbox::default())`, which
    /// wires every slot to the matching `Function::default_on_*`.
    pub fn inbox(mut self, i: FunctionInbox) -> Self {
        self.inbox = Some(i);
        self
    }

    /// Install a [`FunctionCommon`] bundle — lifecycle and
    /// loop-level knobs (`on_started`, `stopping`, `poll_timeout`).
    /// Omitting this call is equivalent to
    /// `.common(FunctionCommon::default())`.
    pub fn common(mut self, c: FunctionCommon) -> Self {
        self.common = Some(c);
        self
    }

    /// Finalise the template configuration and obtain the
    /// Layer-A [`ActorBuilder<Function>`] ready for
    /// [`System::register_actor`](super::super::System::register_actor).
    pub fn build(self) -> ActorBuilder<Function> {
        let Self {
            name,
            capacity,
            inbox,
            common,
        } = self;
        let FunctionInbox {
            delivery,
            source_eos,
        } = inbox.unwrap_or_default();
        let FunctionCommon {
            started,
            stopping,
            poll_timeout,
        } = common.unwrap_or_default();
        ActorBuilder::new(name, capacity)
            .poll_timeout(poll_timeout)
            .factory(move |_bx| {
                Ok(Function {
                    delivery,
                    source_eos,
                    started,
                    stopping,
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

    /// The default function stops when the supervisor signals
    /// shutdown (via `sys.run`'s in-band broadcast triggered by the
    /// absence of any other stage returning an exit trigger).
    #[test]
    fn default_function_runs_and_stops_on_broadcast() {
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
            .register_actor(Function::builder(StageName::unnamed(StageKind::Function), 4).build())
            .unwrap();

        addr.send(PipelineMsg::SourceEos {
            source_id: "s1".into(),
        })
        .unwrap();

        addr.send(PipelineMsg::Shutdown {
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
                Function::builder(StageName::unnamed(StageKind::Function), 4)
                    .inbox(
                        FunctionInbox::builder()
                            .on_source_eos(move |sid, _ctx| {
                                counter_clone.fetch_add(1, Ordering::Relaxed);
                                let _ = sid;
                                Ok(Flow::Stop)
                            })
                            .build(),
                    )
                    .build(),
            )
            .unwrap();

        addr.send(PipelineMsg::SourceEos {
            source_id: "s1".into(),
        })
        .unwrap();

        let report = sys.run().unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 1);
        assert!(report.stage_results[0].1.is_ok());
    }

    /// Custom `on_delivery` can observe unsealed payloads by
    /// shared reference.  We cannot easily synthesise a real
    /// `SealedDelivery` inside a unit test, so this checks that
    /// the configured hook survives the builder path (compile +
    /// acceptance by the framework).
    #[test]
    fn custom_on_delivery_is_installed() {
        let _ = Function::builder(StageName::unnamed(StageKind::Function), 2)
            .inbox(
                FunctionInbox::builder()
                    .on_delivery(|pairs, _ctx| {
                        let _n = pairs.len();
                        Ok(())
                    })
                    .build(),
            )
            .common(
                FunctionCommon::builder()
                    .on_started(|ctx| log::debug!("started {}", ctx.own_name()))
                    .stopping(|ctx| log::debug!("stopping {}", ctx.own_name()))
                    .poll_timeout(Duration::from_millis(50))
                    .build(),
            )
            .build();
    }

    /// Every slot on the runtime [`Function`] struct is non-`Option`
    /// — the bundle builders always substitute a default.  Compile-
    /// only check via a destructuring pattern that refuses any
    /// `Option<_>` wrapper.
    #[test]
    fn runtime_invariant_all_hooks_populated() {
        let sb = Function::builder(StageName::unnamed(StageKind::Function), 2).build();
        // Extract the factory and run it to materialise the
        // runtime struct with all its defaults.
        use crate::context::BuildCtx;
        use crate::registry::Registry;
        use crate::shared::SharedStore;
        let parts = sb.into_parts();
        let reg = Arc::new(Registry::new());
        let shared = Arc::new(SharedStore::new());
        let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let bx = BuildCtx::new(&parts.name, &reg, &shared, &stop_flag);
        let func = (parts.factory)(&bx).expect("factory resolves");
        let Function {
            delivery: _,
            source_eos: _,
            started: _,
            stopping: _,
            poll_timeout: _,
        } = func;
    }
}
