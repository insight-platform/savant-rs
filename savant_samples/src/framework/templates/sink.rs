//! [`Sink`] — pipeline-terminus template.
//!
//! A sink is the simplest actor shape in the sample: it consumes
//! every [`PipelineMsg`](crate::framework::envelopes::PipelineMsg)
//! variant the upstream stage emits and drops the payload, releasing
//! any resources (e.g. GPU slots held by
//! [`SharedBuffer`](deepstream_buffers::SharedBuffer)) back to the
//! pipeline.  The only per-actor logic is typically metrics
//! accounting and per-source first-frame logging.
//!
//! # Defaults
//!
//! Build a sink with zero user code:
//!
//! ```ignore
//! use savant_samples::framework::supervisor::{StageKind, StageName};
//! use savant_samples::framework::templates::Sink;
//!
//! let builder = Sink::builder(StageName::unnamed(StageKind::Blackhole), 16).build();
//! ```
//!
//! The resulting actor drops every delivery and logs `SourceEos`
//! sentinels without terminating (pure-sink semantics).  It cooperates
//! with shutdown via the standard
//! [`PipelineMsg::Shutdown`](crate::framework::envelopes::PipelineMsg::Shutdown)
//! → [`ShutdownHint::Graceful`](super::super::ShutdownHint::Graceful)
//! path handled by the loop driver — no custom code is required.
//!
//! # Hooks
//!
//! Override per-hook for domain behaviour:
//!
//! * [`on_delivery`](SinkBuilder::on_delivery) — called with every
//!   unsealed `(VideoFrameProxy, SharedBuffer)` batch (both the
//!   single-delivery and batched-delivery variants funnel through
//!   here after unsealing).  Default: drop.
//! * [`on_source_eos`](SinkBuilder::on_source_eos) — called on
//!   every [`PipelineMsg::SourceEos`](crate::framework::envelopes::PipelineMsg::SourceEos).
//!   Returns [`Flow`] so the sink can self-terminate on the first
//!   EOS (e.g. single-source pipeline) or stay alive (default).
//! * [`on_started`](SinkBuilder::on_started) / [`on_stopping`](SinkBuilder::on_stopping)
//!   — optional lifecycle logs.
//!
//! All hooks run on the actor's OS thread; they may capture
//! `Send`-safe state by value.

use std::time::Duration;

use anyhow::Result;

use crate::framework::envelopes::{BatchDelivery, PipelineMsg, SingleDelivery};
use crate::framework::supervisor::StageName;
use crate::framework::{
    Actor, ActorBuilder, Context, Dispatch, Flow, Handler, ShutdownPayload, SourceEosPayload,
};
use deepstream_buffers::SharedBuffer;
use savant_core::primitives::frame::VideoFrameProxy;

/// Closure type for `on_delivery`: consumes a batch of
/// `(frame, buffer)` pairs.  Dropping the argument is a no-op sink;
/// clone-and-forward is a tee pattern.
type DeliveryHook =
    Box<dyn FnMut(Vec<(VideoFrameProxy, SharedBuffer)>, &mut Context<Sink>) -> Result<()> + Send>;

/// Closure type for `on_source_eos`.  Returning [`Flow::Stop`] exits
/// the receive loop on that EOS; [`Flow::Cont`] keeps the sink alive.
type EosHook = Box<dyn FnMut(&str, &mut Context<Sink>) -> Result<Flow> + Send>;

/// Closure type for `on_started` and `on_stopping` (no return).
type LifecycleHook = Box<dyn FnMut(&mut Context<Sink>) + Send>;

/// Pipeline-terminus actor.
///
/// Construct with [`Sink::builder`].  The actor carries its
/// hooks as boxed closures; the defaults fulfil the
/// "[`PipelineMsg`] blackhole" contract described in the module
/// docs.
pub struct Sink {
    delivery: DeliveryHook,
    source_eos: EosHook,
    started: Option<LifecycleHook>,
    stopping: Option<LifecycleHook>,
    poll_timeout: Duration,
}

impl Sink {
    /// Start a fluent builder for a sink registered under `name`
    /// with inbox capacity `capacity`.
    pub fn builder(name: StageName, capacity: usize) -> SinkBuilder {
        SinkBuilder::new(name, capacity)
    }
}

impl Actor for Sink {
    type Msg = PipelineMsg;

    fn handle(&mut self, msg: PipelineMsg, ctx: &mut Context<Self>) -> Result<Flow> {
        msg.dispatch(self, ctx)
    }

    fn poll_timeout(&self) -> Duration {
        self.poll_timeout
    }

    fn started(&mut self, ctx: &mut Context<Self>) -> Result<()> {
        if let Some(hook) = self.started.as_mut() {
            hook(ctx);
        } else {
            log::info!("[{}] sink started", ctx.own_name());
        }
        Ok(())
    }

    fn stopping(&mut self, ctx: &mut Context<Self>) {
        if let Some(hook) = self.stopping.as_mut() {
            hook(ctx);
        } else {
            log::info!("[{}] sink stopping", ctx.own_name());
        }
    }
}

impl Handler<SingleDelivery> for Sink {
    fn handle(&mut self, msg: SingleDelivery, ctx: &mut Context<Self>) -> Result<Flow> {
        let pairs = PipelineMsg::Delivery(msg.0).into_pairs();
        (self.delivery)(pairs, ctx)?;
        Ok(Flow::Cont)
    }
}

impl Handler<BatchDelivery> for Sink {
    fn handle(&mut self, msg: BatchDelivery, ctx: &mut Context<Self>) -> Result<Flow> {
        let pairs = msg.0.unseal();
        (self.delivery)(pairs, ctx)?;
        Ok(Flow::Cont)
    }
}

impl Handler<SourceEosPayload> for Sink {
    fn handle(&mut self, msg: SourceEosPayload, ctx: &mut Context<Self>) -> Result<Flow> {
        (self.source_eos)(&msg.source_id, ctx)
    }
}

/// Default no-op handler: the framework's loop driver already
/// consumed the shutdown hint from [`PipelineMsg::Shutdown`] via
/// [`Envelope::as_shutdown`](super::super::Envelope::as_shutdown),
/// so the in-band sentinel has no remaining work to do here.
impl Handler<ShutdownPayload> for Sink {}

/// Fluent builder for [`Sink`].
///
/// Every hook defaults to the "pure blackhole" behaviour described
/// in [`Sink`]'s module docs.  Call any of the `.on_*` methods
/// to override.  Terminate with [`SinkBuilder::build`] to obtain an
/// [`ActorBuilder<Sink>`] suitable for
/// [`System::register_actor`](super::super::System::register_actor).
pub struct SinkBuilder {
    name: StageName,
    capacity: usize,
    poll_timeout: Duration,
    delivery: DeliveryHook,
    source_eos: EosHook,
    started: Option<LifecycleHook>,
    stopping: Option<LifecycleHook>,
}

impl SinkBuilder {
    /// Default receive-poll cadence — matches
    /// [`Actor::poll_timeout`](crate::framework::Actor::poll_timeout)'s
    /// 200 ms default.
    const DEFAULT_POLL: Duration = Duration::from_millis(200);

    /// Build a sink under `name` with inbox capacity `capacity`
    /// and the default hooks — drop every delivery, log+continue
    /// on every EOS.
    pub fn new(name: StageName, capacity: usize) -> Self {
        Self {
            name,
            capacity,
            poll_timeout: Self::DEFAULT_POLL,
            delivery: Box::new(default_drop_delivery),
            source_eos: Box::new(default_log_eos_continue),
            started: None,
            stopping: None,
        }
    }

    /// Override the inbox receive-poll cadence.
    pub fn poll_timeout(mut self, d: Duration) -> Self {
        self.poll_timeout = d;
        self
    }

    /// Install a custom delivery hook.
    ///
    /// The hook receives each unsealed batch and must return `Ok(())`
    /// on success.  Returning `Err` aborts the loop and signals the
    /// supervisor via the [`StageExitGuard`](super::super::StageExitGuard).
    ///
    /// Dropping the argument is a pure-blackhole behaviour (the
    /// default); cloning or forwarding `pairs` is a tee.
    pub fn on_delivery<F>(mut self, f: F) -> Self
    where
        F: FnMut(Vec<(VideoFrameProxy, SharedBuffer)>, &mut Context<Sink>) -> Result<()>
            + Send
            + 'static,
    {
        self.delivery = Box::new(f);
        self
    }

    /// Install a custom source-EOS hook.
    ///
    /// Return [`Flow::Stop`] to self-terminate on the first EOS
    /// observed (single-source pipelines); [`Flow::Cont`] keeps the
    /// sink alive through an arbitrary number of per-source
    /// drainages (multiplexed pipelines).
    pub fn on_source_eos<F>(mut self, f: F) -> Self
    where
        F: FnMut(&str, &mut Context<Sink>) -> Result<Flow> + Send + 'static,
    {
        self.source_eos = Box::new(f);
        self
    }

    /// Install a custom `started` hook; the default logs
    /// `[{stage}] sink started`.
    pub fn on_started<F>(mut self, f: F) -> Self
    where
        F: FnMut(&mut Context<Sink>) + Send + 'static,
    {
        self.started = Some(Box::new(f));
        self
    }

    /// Install a custom `stopping` hook; the default logs
    /// `[{stage}] sink stopping`.
    pub fn on_stopping<F>(mut self, f: F) -> Self
    where
        F: FnMut(&mut Context<Sink>) + Send + 'static,
    {
        self.stopping = Some(Box::new(f));
        self
    }

    /// Finalise the template configuration and obtain the
    /// Layer-A [`ActorBuilder<Sink>`] ready for
    /// [`System::register_actor`](super::super::System::register_actor).
    pub fn build(self) -> ActorBuilder<Sink> {
        let Self {
            name,
            capacity,
            poll_timeout,
            delivery,
            source_eos,
            started,
            stopping,
        } = self;
        ActorBuilder::new(name, capacity)
            .poll_timeout(poll_timeout)
            .factory(move |_bx| {
                Ok(Sink {
                    delivery,
                    source_eos,
                    started,
                    stopping,
                    poll_timeout,
                })
            })
    }
}

fn default_drop_delivery(
    pairs: Vec<(VideoFrameProxy, SharedBuffer)>,
    _ctx: &mut Context<Sink>,
) -> Result<()> {
    drop(pairs);
    Ok(())
}

fn default_log_eos_continue(source_id: &str, ctx: &mut Context<Sink>) -> Result<Flow> {
    log::info!("[{}] SourceEos {source_id}: continuing", ctx.own_name());
    Ok(Flow::Cont)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::supervisor::{StageKind, StageName};
    use crate::framework::{ShutdownAction, ShutdownCause, ShutdownCtx, System};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// The default sink stops when the supervisor signals shutdown
    /// (via `sys.run`'s in-band broadcast triggered by the absence
    /// of any other stage returning an exit trigger).
    #[test]
    fn default_sink_runs_and_stops_on_broadcast() {
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
            .register_actor(Sink::builder(StageName::unnamed(StageKind::Blackhole), 4).build())
            .unwrap();

        // Send a SourceEos; default hook logs + continues.
        addr.send(PipelineMsg::SourceEos {
            source_id: "s1".into(),
        })
        .unwrap();

        // Send a Shutdown; envelope hint arms the break-now flag.
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
    fn custom_source_eos_stops_sink() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();
        let mut sys = System::new()
            .install_ctrlc_handler(false)
            .quiescence(Duration::from_millis(0));
        let addr = sys
            .register_actor(
                Sink::builder(StageName::unnamed(StageKind::Blackhole), 4)
                    .on_source_eos(move |sid, _ctx| {
                        counter_clone.fetch_add(1, Ordering::Relaxed);
                        let _ = sid;
                        Ok(Flow::Stop)
                    })
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

    /// Custom `on_delivery` can observe unsealed payloads.  We
    /// cannot easily synthesise a real `SealedDelivery` inside a
    /// unit test, so this checks that the configured hook survives
    /// the builder path (compile + acceptance by the framework).
    #[test]
    fn custom_on_delivery_is_installed() {
        let _ = Sink::builder(StageName::unnamed(StageKind::Blackhole), 2)
            .on_delivery(|pairs, _ctx| {
                drop(pairs);
                Ok(())
            })
            .on_started(|ctx| log::debug!("started {}", ctx.own_name()))
            .on_stopping(|ctx| log::debug!("stopping {}", ctx.own_name()))
            .poll_timeout(Duration::from_millis(50))
            .build();
    }
}
