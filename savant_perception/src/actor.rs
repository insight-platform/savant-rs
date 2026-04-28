//! [`Actor`] + [`Source`] traits — the behavioural contracts every
//! user-facing actor / source type implements.
//!
//! An [`Actor`] owns one OS thread, a bounded inbox of its own
//! [`Envelope`] type, and a
//! [`Context`] that brokers access to the
//! registry and shared state.  The
//! [`loop_driver::run_actor`](super::loop_driver::run_actor)
//! function drives the lifecycle:
//!
//! ```text
//! started -> (handle | on_tick)* -> stopping
//! ```
//!
//! A [`Source`] is a no-inbox producer — it owns its thread but
//! has no receive loop.  Instead it pushes messages onto
//! [`Addr<M>`](super::addr::Addr) handles resolved through
//! [`BuildCtx`](super::context::BuildCtx) and polls
//! [`SourceContext::is_stopping`] to learn when the pipeline is
//! winding down so it can return cleanly from
//! [`Source::run`].

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
///    but before any message is polled.  Use it for any
///    initialisation that must happen on the worker thread
///    itself (e.g. allocating thread-local resources).
/// 2. [`handle`](Actor::handle) is called once per inbox message.
///    Implementors choose between two dispatch styles:
///
///    * Forward to
///      [`Dispatch::dispatch`](super::envelope::Dispatch::dispatch)
///      so per-variant
///      [`Handler<V>`](super::handler::Handler) impls do the
///      work — appropriate for actors whose envelope has a
///      `Dispatch` impl that splits variants into typed
///      handlers.
///    * Pattern-match the envelope directly — appropriate for
///      free-form actors that handle each variant inline.
///
///    There is intentionally no default body: keeping the
///    dispatch strategy explicit at the impl site is easier to
///    review when a pipeline mixes stages with custom
///    actors.
/// 3. [`on_tick`](Actor::on_tick) is called whenever the inbox
///    `recv_timeout` fires without a message — at most once per
///    [`poll_timeout`](Actor::poll_timeout).  Override for
///    actors that need periodic work (deadline checks,
///    flushing, batched egress) while the inbox is idle.
/// 4. [`stopping`](Actor::stopping) runs once after the receive
///    loop exits, regardless of *why* it exited:
///    [`Flow::Stop`], a hook returning `Err`, the inbox closing,
///    or a panic unwinding through the
///    [`StageExitGuard`](super::supervisor::StageExitGuard).  Use
///    it for end-of-life cleanup and final logging.
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
    /// explicitly.  Two common shapes:
    ///
    /// * Forward to
    ///   [`Dispatch::dispatch`](super::envelope::Dispatch::dispatch)
    ///   so each envelope variant is routed to a typed
    ///   [`Handler<V>`](super::handler::Handler) impl:
    ///   ```ignore
    ///   fn handle(&mut self, msg: Self::Msg, ctx: &mut Context<Self>)
    ///       -> anyhow::Result<Flow>
    ///   {
    ///       msg.dispatch(self, ctx)
    ///   }
    ///   ```
    /// * Pattern-match the envelope directly — suitable for
    ///   actors that handle every variant inline.
    ///
    /// There is intentionally no default body: keeping the
    /// dispatch strategy visible at the impl site keeps
    /// pipelines that mix stage-backed and custom actors
    /// easy to review.
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
    /// Defaults to 200 ms.
    fn poll_timeout(&self) -> Duration {
        Duration::from_millis(200)
    }
}

/// Trait implemented by **no-inbox** producers — stages that own
/// a thread but push messages rather than poll an inbox.
///
/// The framework calls [`Source::run`] once on the source's
/// worker thread; the source pushes messages onto peer
/// [`Addr<M>`](super::addr::Addr) handles resolved through
/// [`BuildCtx`](super::context::BuildCtx) and returns when it is
/// done.
///
/// A source cooperates with shutdown by polling
/// [`SourceContext::is_stopping`] between chunks of work.  When
/// the flag is set the source should drain any in-flight work,
/// emit its terminal sentinel(s) (e.g.
/// [`EncodedMsg::SourceEos`](crate::envelopes::EncodedMsg::SourceEos)
/// for each source it represents), and return `Ok(())` from
/// `run`.
pub trait Source: Sized + Send + 'static {
    /// Run the source to completion on its worker thread.
    ///
    /// `ctx` exposes the source's own
    /// [`StageName`](super::supervisor::StageName), shared-state
    /// lookups, and the
    /// [`SourceContext::is_stopping`](SourceContext::is_stopping)
    /// flag that the supervisor flips when the system is winding
    /// down.
    fn run(self, ctx: SourceContext) -> anyhow::Result<()>;
}
