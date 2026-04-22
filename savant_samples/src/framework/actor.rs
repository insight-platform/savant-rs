//! [`Actor`] + [`Source`] traits — the behavioural contracts every
//! user-facing actor / source type implements.
//!
//! An [`Actor`] owns one OS thread, a bounded inbox of its own
//! [`Envelope`](super::envelope::Envelope) type, and a
//! [`Context`](super::context::Context) that brokers access to the
//! registry and shared state.  The
//! [`loop_driver::run_actor`](super::loop_driver::run_actor)
//! function drives the lifecycle:
//!
//! ```text
//! started -> (handle | on_tick)* -> stopping
//! ```
//!
//! A [`Source`] is a no-inbox producer — it owns its thread but
//! has no receive-loop; instead it pushes out messages on
//! [`Addr<M>`](super::addr::Addr) handles resolved at build time
//! and polls [`SourceContext::is_stopping`] to learn when the
//! pipeline is winding down.  The MP4 demuxer is the canonical
//! example.

use std::time::Duration;

use super::context::{Context, SourceContext};
use super::envelope::Envelope;
use super::handler::Flow;

/// Behavioural trait implemented by every framework-backed actor.
///
/// The receive-loop driver owns the actor by value for the
/// lifetime of its thread, so the actor is free to hold `&mut
/// self` state (operator handle, per-source `seen_eos` map,
/// counters, …) without reaching for any interior-mutability
/// machinery.
///
/// # Lifecycle
///
/// 1. [`started`](Actor::started) runs once *inside* the thread
///    body, after the thread has entered its
///    [`StageExitGuard`](super::supervisor::StageExitGuard) scope
///    but before any message is polled.  Use it to allocate
///    per-thread resources (e.g. GStreamer pipeline construction
///    that must happen on the worker thread) and to emit a
///    "hello" log record.
/// 2. [`handle`](Actor::handle) is called once per inbox message.
///    The default body delegates to
///    [`Dispatch::dispatch`](super::envelope::Dispatch::dispatch)
///    so actors with variant-specific behaviour only need a
///    zero-body impl *and* per-variant
///    [`Handler<V>`](super::handler::Handler) impls.  Free-form
///    actors (e.g. the blackhole sink) override `handle` and
///    pattern-match the envelope directly.
/// 3. [`on_tick`](Actor::on_tick) is called whenever the inbox
///    `recv_timeout` fires without a message — typically every
///    [`poll_timeout`](Actor::poll_timeout).  Override for actors
///    that need periodic work (e.g. surfacing a sealed batch from
///    an operator) while the inbox is idle.
/// 4. [`stopping`](Actor::stopping) runs once after the receive
///    loop exits (for *any* reason: `Flow::Stop`, `Err`,
///    disconnect, or panic unwind via the
///    [`StageExitGuard`](super::supervisor::StageExitGuard)).  Use
///    it to drain operator tails and emit a "goodbye" log record.
pub trait Actor: Sized + Send + 'static {
    /// Envelope type carried on this actor's inbox.
    type Msg: Envelope;

    /// One-time initialisation, called inside the thread body
    /// before the first `recv`.  Default: no-op.
    fn started(&mut self, _ctx: &mut Context<Self>) -> anyhow::Result<()> {
        Ok(())
    }

    /// Handle one inbox envelope.
    ///
    /// Every actor implementation chooses its dispatch strategy
    /// explicitly:
    ///
    /// * Templates that rely on per-variant
    ///   [`Handler<V>`](super::handler::Handler) impls forward to
    ///   [`Dispatch::dispatch`](super::envelope::Dispatch::dispatch)
    ///   in one line:
    ///   ```ignore
    ///   fn handle(&mut self, msg: Self::Msg, ctx: &mut Context<Self>)
    ///       -> anyhow::Result<Flow>
    ///   {
    ///       msg.dispatch(self, ctx)
    ///   }
    ///   ```
    /// * Free-form actors (e.g. the blackhole sink) pattern-match
    ///   the envelope directly and decide flow per variant.
    ///
    /// We deliberately do **not** provide a default body here: the
    /// explicit one-line delegation keeps the dispatch strategy
    /// visible at the actor impl site, which matters when
    /// reviewing pipelines that mix template and free-form
    /// actors.
    fn handle(&mut self, msg: Self::Msg, ctx: &mut Context<Self>) -> anyhow::Result<Flow>;

    /// Periodic tick, called whenever the inbox has been idle for
    /// [`poll_timeout`](Actor::poll_timeout).  Default: no-op.
    fn on_tick(&mut self, _ctx: &mut Context<Self>) -> anyhow::Result<Flow> {
        Ok(Flow::Cont)
    }

    /// Cleanup hook, called once after the receive loop exits for
    /// any reason (including panic unwind — the
    /// [`StageExitGuard`](super::supervisor::StageExitGuard) still
    /// fires).  Default: no-op.
    fn stopping(&mut self, _ctx: &mut Context<Self>) {}

    /// Maximum wall-clock time the receive loop will wait on an
    /// empty inbox before firing [`on_tick`](Actor::on_tick).
    /// Defaults to 200 ms — same cadence as the sample's existing
    /// stages.
    fn poll_timeout(&self) -> Duration {
        Duration::from_millis(200)
    }
}

/// Trait implemented by **no-inbox** producers — stages that own a
/// thread but push rather than poll.
///
/// The MP4 demuxer is the canonical example: it drives a
/// [`gstreamer`] pipeline whose bus callback emits `EncodedMsg`
/// items directly onto a bound [`Addr`](super::addr::Addr).  There
/// is no receive-loop to run; instead the framework calls
/// [`Source::run`] on the worker thread and the source returns
/// when it is done.
///
/// A source cooperates with shutdown by polling
/// [`SourceContext::is_stopping`] between chunks of work and
/// cleanly exiting (emitting its terminal
/// [`SourceEos`](crate::framework::envelopes::EncodedMsg::SourceEos)
/// before returning) once the flag is set.
pub trait Source: Sized + Send + 'static {
    /// Run the source to completion on its worker thread.
    ///
    /// `ctx` exposes the source's own
    /// [`StageName`](super::supervisor::StageName), shared-state
    /// lookups, and the [`SourceContext::is_stopping`] flag that
    /// the orchestrator flips to request shutdown.
    fn run(self, ctx: SourceContext) -> anyhow::Result<()>;
}
