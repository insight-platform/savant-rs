//! Generic receive-loop driver shared by every framework actor.
//!
//! The driver owns the actor by value for the lifetime of the
//! worker thread and calls:
//!
//! ```text
//! actor.started(ctx)
//! loop {
//!     match inbox.recv_timeout(poll) {
//!         Ok(msg)          => actor.handle(msg, ctx),
//!         Err(Timeout)     => actor.on_tick(ctx),
//!         Err(Disconnected) => break,
//!     }
//! }
//! actor.stopping(ctx)
//! ```
//!
//! plus cooperative-stop bookkeeping (observe
//! [`ShutdownHint`]s on incoming
//! envelopes, honour
//! [`Context::should_quit`](super::context::Context::should_quit)
//! between iterations, cap `recv_timeout` by the earliest of
//! `poll_timeout`, `ctx.deadline`, `ctx.tick_deadline`).

use std::time::{Duration, Instant};

use anyhow::Result;
use crossbeam::channel::{Receiver, RecvTimeoutError};
use super::actor::Actor;
use super::context::Context;
use super::envelope::{Envelope, ShutdownHint};
use super::handler::Flow;
use super::stage_metrics::monotonic_ns;

/// Drive `actor` on the current thread to completion.
///
/// * `inbox` is the receiving half of the actor's bounded
///   channel (the companion of the registered
///   [`Addr`](super::addr::Addr) on the sending side).
/// * `ctx` is the per-thread [`Context`] built by
///   [`System::run`](super::system::System::run) before the
///   actor thread is spawned.
///
/// Returns `Ok(())` on a natural exit (all senders dropped,
/// `Flow::Stop`, or [`Context::should_quit`] tripped) and the
/// first `Err` produced by a lifecycle method otherwise.
/// [`Actor::stopping`] always runs after the loop, regardless of
/// outcome, so cleanup is guaranteed.
pub fn run_actor<A>(mut actor: A, inbox: Receiver<A::Msg>, mut ctx: Context<A>) -> Result<()>
where
    A: Actor,
{
    run_actor_with(&mut actor, inbox, &mut ctx, |a, msg, c| a.handle(msg, c))
}

fn run_actor_with<A, F>(
    actor: &mut A,
    inbox: Receiver<A::Msg>,
    ctx: &mut Context<A>,
    mut handle: F,
) -> Result<()>
where
    A: Actor,
    F: FnMut(&mut A, A::Msg, &mut Context<A>) -> Result<Flow>,
{
    // Lifecycle callbacks (`started` / `on_tick` / `stopping`) are
    // intentionally not span-wrapped: they have no frame, so there
    // is no natural parent context and the framework would have to
    // create a fresh root span per call — high-volume noise that
    // very few users actually want.  Anything user-installed (e.g.
    // `on_started` / `on_stopping` hooks) runs without a current
    // OTel span; user code that needs lifecycle traces can open its
    // own root span explicitly.
    if let Err(e) = actor.started(ctx) {
        actor.stopping(ctx);
        return Err(e);
    }

    // Stage name is constant for this actor's lifetime — render it
    // once outside the receive loop so the per-frame ingress stamp
    // doesn't allocate a fresh `String` for every message.  The
    // stamp helper takes `&str`.
    let stage_name = ctx.own_name().to_string();

    let mut loop_result: Result<()> = Ok(());
    'outer: loop {
        if ctx.should_quit() {
            break 'outer;
        }
        let timeout = next_timeout(actor.poll_timeout(), ctx.deadline(), ctx.tick_deadline());
        match inbox.recv_timeout(timeout) {
            Ok(msg) => {
                if let Some(hint) = msg.as_shutdown() {
                    match hint {
                        ShutdownHint::Graceful { grace, reason } => {
                            ctx.arm_graceful_stop(grace, reason);
                        }
                        ShutdownHint::Abort { reason } => {
                            ctx.arm_abort(reason);
                        }
                    }
                }
                // Per-stage metrics — the framework reads object
                // counts off any [`VideoFrame`] the envelope
                // carries (via [`Envelope::frame_object_counts`])
                // and records them through the OpenTelemetry
                // instruments held by [`StageMetrics`], plus
                // sidecar atomics that drive the framework's
                // periodic 📊 console reporter.  No-op for sentinel
                // envelopes that carry no frames (`SourceEos`,
                // `Shutdown`, `MessageEx`, `StreamInfo`, `Packet`).
                //
                // Per-frame ingress timestamps are stamped onto
                // each frame as a temporary attribute under the
                // `telemetry.tracing` namespace; the stage's
                // [`Router`](crate::Router) reads + clears them
                // on send to compute the real `t_out − t_in`
                // frame latency.  Lifetime is bounded by the
                // frame's [`Drop`] — when the frame's last clone
                // dies, the attribute dies too, so no side-table
                // can bloat under user code that drops or retains
                // frames without forwarding.
                let object_counts = msg.frame_object_counts();
                if !object_counts.is_empty() {
                    let total_frames = object_counts.len();
                    let total_objects: usize = object_counts.iter().sum();
                    ctx.stage_metrics
                        .record_message(total_frames, total_objects, inbox.len());
                    msg.record_stage_ingress(&stage_name, monotonic_ns());
                }

                // No per-message `<stage>.recv` span is opened
                // here: per-frame spans (`on_delivery`, `on_frame`,
                // etc.) inside the handlers carry the trace signal
                // and are parented explicitly off each frame's
                // current context, sidestepping any reliance on the
                // thread-local `OtelContext::current()` that
                // interleaved-execution paths can't honour.  The
                // wall-clock duration of the handler call is still
                // captured via the stage metrics histogram.
                let handle_t0 = Instant::now();
                let result = handle(actor, msg, ctx);
                let handle_elapsed = handle_t0.elapsed();
                ctx.stage_metrics.record_handle_latency(handle_elapsed);
                match result {
                    Ok(Flow::Cont) => {}
                    Ok(Flow::Stop) => break 'outer,
                    Err(e) => {
                        loop_result = Err(e);
                        break 'outer;
                    }
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                if ctx
                    .tick_deadline()
                    .map(|d| Instant::now() >= d)
                    .unwrap_or(true)
                {
                    let r = actor.on_tick(ctx);
                    match r {
                        Ok(Flow::Cont) => {}
                        Ok(Flow::Stop) => break 'outer,
                        Err(e) => {
                            loop_result = Err(e);
                            break 'outer;
                        }
                    }
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                break 'outer;
            }
        }
    }

    actor.stopping(ctx);
    loop_result
}

fn next_timeout(
    poll: Duration,
    deadline: Option<Instant>,
    tick_deadline: Option<Instant>,
) -> Duration {
    let now = Instant::now();
    let mut out = poll;
    if let Some(d) = deadline {
        let rem = d.saturating_duration_since(now);
        if rem < out {
            out = rem;
        }
    }
    if let Some(t) = tick_deadline {
        let rem = t.saturating_duration_since(now);
        if rem < out {
            out = rem;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envelope::ShutdownHint;
    use crate::registry::Registry;
    use crate::shared::SharedStore;
    use crate::supervisor::StageKind;
    use crate::supervisor::StageName;
    use crossbeam::channel::bounded;
    use std::borrow::Cow;
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    enum Msg {
        Ping,
        Stop,
        Shutdown {
            grace: Option<Duration>,
            reason: Cow<'static, str>,
        },
    }

    impl Envelope for Msg {
        fn as_shutdown(&self) -> Option<ShutdownHint<'_>> {
            match self {
                Msg::Shutdown { grace, reason } => Some(ShutdownHint::Graceful {
                    grace: *grace,
                    reason,
                }),
                _ => None,
            }
        }
    }

    struct Counter {
        received: usize,
        ticks: usize,
        started: bool,
        stopped: bool,
    }

    impl Actor for Counter {
        type Msg = Msg;
        fn started(&mut self, _ctx: &mut Context<Self>) -> Result<()> {
            self.started = true;
            Ok(())
        }
        fn handle(&mut self, _msg: Self::Msg, _ctx: &mut Context<Self>) -> Result<Flow> {
            // Exercised only by the free-form path below; the
            // `run_actor_with` call in the tests substitutes a
            // closure via its `handle` parameter.
            unreachable!("tests drive `run_actor_with` directly")
        }
        fn on_tick(&mut self, _ctx: &mut Context<Self>) -> Result<Flow> {
            self.ticks += 1;
            Ok(Flow::Cont)
        }
        fn stopping(&mut self, _ctx: &mut Context<Self>) {
            self.stopped = true;
        }
        fn poll_timeout(&self) -> Duration {
            Duration::from_millis(5)
        }
    }

    fn ctx() -> Context<Counter> {
        Context::new(
            StageName::unnamed(StageKind::DeepStreamFunction),
            Arc::from("test"),
            Arc::new(Registry::new()),
            Arc::new(SharedStore::new()),
            Arc::new(AtomicBool::new(false)),
            crate::stage_metrics::StageMetrics::new("test"),
        )
    }

    fn drive<F>(tx_msgs: Vec<Msg>, ctx: Context<Counter>, handle: F) -> (Counter, Result<()>)
    where
        F: FnMut(&mut Counter, Msg, &mut Context<Counter>) -> Result<Flow>,
    {
        let (tx, rx) = bounded::<Msg>(tx_msgs.len().max(1));
        for m in tx_msgs {
            tx.send(m).unwrap();
        }
        drop(tx);
        let mut actor = Counter {
            received: 0,
            ticks: 0,
            started: false,
            stopped: false,
        };
        let mut ctx = ctx;
        let res = run_actor_with(&mut actor, rx, &mut ctx, handle);
        (actor, res)
    }

    #[test]
    fn free_form_handle_runs_lifecycle() {
        let mut seen: Vec<&'static str> = Vec::new();
        let (actor, result) = drive(
            vec![Msg::Ping, Msg::Stop],
            ctx(),
            |a: &mut Counter, m, _c| {
                a.received += 1;
                seen.push(match m {
                    Msg::Ping => "ping",
                    Msg::Stop => "stop",
                    Msg::Shutdown { .. } => "shutdown",
                });
                if matches!(seen.last(), Some(&"stop")) {
                    return Ok(Flow::Stop);
                }
                Ok(Flow::Cont)
            },
        );
        assert!(result.is_ok());
        assert!(actor.started, "started must have run");
        assert!(actor.stopped, "stopping must have run");
        assert_eq!(seen, vec!["ping", "stop"]);
    }

    #[test]
    fn shutdown_hint_arms_break_now() {
        let mut pings = 0usize;
        let (_, result) = drive(
            vec![
                Msg::Shutdown {
                    grace: None,
                    reason: Cow::Borrowed("test"),
                },
                Msg::Ping,
            ],
            ctx(),
            |_a, m, _c| {
                if matches!(m, Msg::Ping) {
                    pings += 1;
                }
                Ok(Flow::Cont)
            },
        );
        assert!(result.is_ok());
        assert_eq!(pings, 0, "messages after shutdown must be skipped");
    }

    #[test]
    fn disconnect_exits_cleanly() {
        let (_tx, rx) = bounded::<Msg>(1);
        let mut actor = Counter {
            received: 0,
            ticks: 0,
            started: false,
            stopped: false,
        };
        drop(_tx);
        let mut c = ctx();
        let result = run_actor_with(&mut actor, rx, &mut c, |_, _, _| Ok(Flow::Cont));
        assert!(result.is_ok());
        assert!(actor.stopped);
    }

    #[test]
    fn tick_runs_when_inbox_idle_and_deadline_breaks_loop() {
        let (_tx, rx) = bounded::<Msg>(1);
        let mut c = ctx();
        c.arm_graceful_stop(Some(Duration::from_millis(20)), "test-window");
        let mut actor = Counter {
            received: 0,
            ticks: 0,
            started: false,
            stopped: false,
        };
        let result = run_actor_with(&mut actor, rx, &mut c, |_, _, _| Ok(Flow::Cont));
        assert!(result.is_ok());
        assert!(actor.ticks >= 1, "at least one tick must have fired");
        assert!(actor.stopped);
    }

    #[test]
    fn next_timeout_picks_smallest() {
        let poll = Duration::from_millis(200);
        let now = Instant::now();
        let d = now + Duration::from_millis(50);
        let t = now + Duration::from_millis(30);
        let chosen = next_timeout(poll, Some(d), Some(t));
        assert!(chosen <= Duration::from_millis(30));
    }
}
