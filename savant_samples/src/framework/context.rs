//! [`Context<A>`], [`BuildCtx`], and [`SourceContext`] — the
//! framework's handles into the registry, shared state, and
//! shutdown machinery for, respectively, a running actor, an
//! actor being constructed, and a running source.
//!
//! All three types share the same two inputs (an
//! `Arc<Registry>` for address look-ups and an `Arc<SharedStore>`
//! for shared state), differing only in **what mutable state they
//! own** and **when they are available**:
//!
//! | Type              | Owned mutable state         | Lives during…
//! |-------------------|-----------------------------|------------------
//! | [`BuildCtx`]      | *(none)*                    | `System::build`
//! | [`Context`]       | `deadline`, `break_now`, `tick_deadline` | actor's receive loop
//! | [`SourceContext`] | *(none — stop flag is shared)* | source's `run()`
//!
//! `BuildCtx` and `Context` are handed to user code in
//! different phases: factories see a `BuildCtx` (address
//! look-ups only); `started`/`handle`/`on_tick`/`stopping` see a
//! `&mut Context` (same look-ups *plus* cooperative-stop knobs).

use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;

use super::actor::Actor;
use super::addr::Addr;
use super::envelope::Envelope;
use super::operator_sink::OperatorSink;
use super::registry::Registry;
use super::router::Router;
use super::shared::SharedStore;
use super::supervisor::{StageKind, StageName};

/// Construction-time context handed to every actor factory.
///
/// Factories run in phase 2 of
/// [`System::build`](super::actor::Actor), after phase 1 has
/// published every actor's [`Addr`] in the registry.  That ordering
/// lets factories resolve downstream peers eagerly — no more
/// cloning `Sender`s through closures — and produce a ready-to-run
/// actor value by return.
///
/// `BuildCtx` is entirely immutable: factories observe but don't
/// mutate the framework's construction state.  Any mutation
/// (e.g. populating a local `HashMap<source_id, State>`) happens
/// inside the actor value itself.
pub struct BuildCtx<'a> {
    own_name: &'a StageName,
    registry: &'a Arc<Registry>,
    shared: &'a SharedStore,
}

impl<'a> BuildCtx<'a> {
    /// Internal constructor used by
    /// [`System::build`](super::actor::Actor) to hand a borrowed
    /// context to each factory.
    #[allow(dead_code, reason = "consumed by System in Component 2")]
    pub(crate) fn new(
        own_name: &'a StageName,
        registry: &'a Arc<Registry>,
        shared: &'a SharedStore,
    ) -> Self {
        Self {
            own_name,
            registry,
            shared,
        }
    }

    /// This actor's own registered name, for logs and
    /// [`OperatorSink`] owner attribution.
    pub fn own_name(&self) -> &StageName {
        self.own_name
    }

    /// Resolve the [`Addr<M>`] registered under `peer`.  Returns
    /// `Err` on an unknown name or a type mismatch.
    pub fn addr<M: Envelope>(&self, peer: &StageName) -> Result<Addr<M>> {
        self.registry.get::<M>(peer)
    }

    /// Resolve an [`OperatorSink<M>`] pointing at `peer`, owned by
    /// `self` (the owner's name is used for abort-path log records
    /// so a "dropped frame" warning pinpoints the guilty stage).
    pub fn sink<M: Envelope>(&self, peer: &StageName) -> Result<OperatorSink<M>> {
        Ok(OperatorSink::new(
            self.own_name.clone(),
            self.addr::<M>(peer)?,
        ))
    }

    /// Build a [`Router<M>`] owned by `self` with an optional
    /// default peer installed from `default`.
    ///
    /// The returned router shares an `Arc<Registry>` clone, so the
    /// runtime `send_to(&peer, msg)` path can resolve any stage the
    /// system knows about — not only the one passed here.
    pub fn router<M: Envelope>(&self, default: Option<&StageName>) -> Result<Router<M>> {
        let default_sink = default.map(|peer| self.sink::<M>(peer)).transpose()?;
        Ok(Router::new(
            self.own_name.clone(),
            self.registry.clone(),
            default_sink,
        ))
    }

    /// Singleton shared-state look-up.  Returns `None` if no
    /// singleton of `T` was published.
    pub fn shared<T: Send + Sync + 'static>(&self) -> Option<Arc<T>> {
        self.shared.get::<T>()
    }

    /// Named shared-state look-up.  Returns `None` if no value of
    /// `T` was published under `key`.
    pub fn shared_as<T: Send + Sync + 'static>(&self, key: &str) -> Option<Arc<T>> {
        self.shared.get_as::<T>(key)
    }

    /// Every registered peer — intended for introspection /
    /// dynamic fan-out (e.g. "send a stream-info ping to every
    /// decoder in the pipeline").
    pub fn stages(&self) -> Vec<StageName> {
        self.registry.stages()
    }

    /// Every registered peer whose kind matches `kind`.
    pub fn stages_of(&self, kind: StageKind) -> Vec<StageName> {
        self.registry.stages_of(kind)
    }
}

/// Runtime context handed to [`Actor`] lifecycle methods.
///
/// Adds the cooperative-stop knobs (`deadline`, `break_now`) that
/// the receive-loop driver mutates as shutdown sentinels arrive:
///
/// * [`arm_graceful_stop`](Context::arm_graceful_stop) — "observed
///   a `ShutdownHint::Graceful`, arm the deadline".
/// * [`arm_abort`](Context::arm_abort) — "observed a
///   `ShutdownHint::Abort`, break immediately".
/// * [`should_quit`](Context::should_quit) — "loop, have I hit the
///   break condition?".
///
/// The driver calls these around each `handle` / `on_tick`; user
/// code rarely needs to touch them directly.
pub struct Context<A: Actor> {
    own_name: StageName,
    registry: Arc<Registry>,
    shared: Arc<SharedStore>,
    stop_flag: Arc<AtomicBool>,
    deadline: Option<Instant>,
    break_now: bool,
    tick_deadline: Option<Instant>,
    _marker: PhantomData<fn() -> A>,
}

impl<A: Actor> Context<A> {
    /// Internal constructor — the framework threads the shared
    /// registry, shared store, and stop flag into every per-actor
    /// context from
    /// [`Running`](super::loop_driver::run_actor).
    #[allow(dead_code, reason = "consumed by System in Component 2")]
    pub(crate) fn new(
        own_name: StageName,
        registry: Arc<Registry>,
        shared: Arc<SharedStore>,
        stop_flag: Arc<AtomicBool>,
    ) -> Self {
        Self {
            own_name,
            registry,
            shared,
            stop_flag,
            deadline: None,
            break_now: false,
            tick_deadline: None,
            _marker: PhantomData,
        }
    }

    /// This actor's registered [`StageName`].
    pub fn own_name(&self) -> &StageName {
        &self.own_name
    }

    /// Resolve the [`Addr<M>`] registered under `peer` at runtime.
    ///
    /// Useful for actors that pick their downstream targets
    /// dynamically (e.g. "route this detection to the
    /// `infer[attr]` stage iff the class is Person").
    pub fn resolve<M: Envelope>(&self, peer: &StageName) -> Result<Addr<M>> {
        self.registry.get::<M>(peer)
    }

    /// Resolve an [`OperatorSink<M>`] pointing at `peer`, owned by
    /// this actor.
    pub fn sink<M: Envelope>(&self, peer: &StageName) -> Result<OperatorSink<M>> {
        Ok(OperatorSink::new(
            self.own_name.clone(),
            self.resolve::<M>(peer)?,
        ))
    }

    /// Build a [`Router<M>`] owned by this actor with an optional
    /// default peer.  Rare at runtime (actors usually receive a
    /// router bound at construction time), but handy for free-form
    /// actors that wire their downstreams dynamically.
    pub fn router<M: Envelope>(&self, default: Option<&StageName>) -> Result<Router<M>> {
        let default_sink = default.map(|peer| self.sink::<M>(peer)).transpose()?;
        Ok(Router::new(
            self.own_name.clone(),
            self.registry.clone(),
            default_sink,
        ))
    }

    /// Singleton shared-state look-up.
    pub fn shared<T: Send + Sync + 'static>(&self) -> Option<Arc<T>> {
        self.shared.get::<T>()
    }

    /// Named shared-state look-up.
    pub fn shared_as<T: Send + Sync + 'static>(&self, key: &str) -> Option<Arc<T>> {
        self.shared.get_as::<T>(key)
    }

    /// Every registered peer (see [`BuildCtx::stages`]).
    pub fn stages(&self) -> Vec<StageName> {
        self.registry.stages()
    }

    /// Every registered peer whose kind matches `kind`.
    pub fn stages_of(&self, kind: StageKind) -> Vec<StageName> {
        self.registry.stages_of(kind)
    }

    /// Arm a graceful stop — threads shutdown semantics into the
    /// actor's receive loop without touching the mailbox.
    ///
    /// * `grace = None` — break after the current `handle` /
    ///   `on_tick` returns.
    /// * `grace = Some(d)` — arm a deadline of `now + d`; the
    ///   loop exits the next time it observes the deadline has
    ///   elapsed.  Earliest deadline wins — a later, longer
    ///   grace never pushes an already-recorded deadline into
    ///   the future.
    pub fn arm_graceful_stop(&mut self, grace: Option<Duration>, reason: &str) {
        match grace {
            None => {
                log::info!(
                    "[{}] Shutdown (reason={reason}, grace=none); exiting after current message",
                    self.own_name
                );
                self.break_now = true;
            }
            Some(d) => {
                let new_deadline = Instant::now() + d;
                self.deadline = Some(match self.deadline {
                    Some(existing) if existing < new_deadline => existing,
                    _ => new_deadline,
                });
                log::info!(
                    "[{}] Shutdown (reason={reason}, grace={d:?}); deadline set, continuing",
                    self.own_name
                );
            }
        }
    }

    /// Arm an immediate abort — set `break_now`.
    pub fn arm_abort(&mut self, reason: &str) {
        log::info!(
            "[{}] Shutdown (reason={reason}, abort); breaking immediately",
            self.own_name
        );
        self.break_now = true;
    }

    /// Whether the loop driver should break out of the receive
    /// loop at the next opportunity.
    pub fn should_quit(&self) -> bool {
        if self.break_now {
            return true;
        }
        if let Some(d) = self.deadline {
            if Instant::now() >= d {
                return true;
            }
        }
        if self.stop_flag.load(Ordering::Relaxed) {
            return true;
        }
        false
    }

    /// Current deadline, if armed.  Used by the loop driver to
    /// cap `recv_timeout` durations and by tests.
    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    /// Whether `break_now` has been armed (a `None` grace or an
    /// abort).
    pub fn break_now(&self) -> bool {
        self.break_now
    }

    /// Schedule the next `on_tick` call no later than `at` — used
    /// by templates that need a per-source deadline (e.g. a
    /// decoder draining the operator tail after an EOS).
    pub fn schedule_tick_at(&mut self, at: Instant) {
        self.tick_deadline = Some(match self.tick_deadline {
            Some(existing) if existing < at => existing,
            _ => at,
        });
    }

    /// Scheduled tick deadline, if any.  Returned to the loop
    /// driver so it can cap its `recv_timeout`.
    pub fn tick_deadline(&self) -> Option<Instant> {
        self.tick_deadline
    }

    /// Clear the scheduled tick deadline — typically called from
    /// `on_tick` once the work it scheduled has completed.
    pub fn clear_tick(&mut self) {
        self.tick_deadline = None;
    }

    /// Access to the shared stop flag — tests use this to flip the
    /// flag externally and observe the loop exit.
    #[allow(dead_code, reason = "consumed by System in Component 2")]
    pub(crate) fn stop_flag(&self) -> &Arc<AtomicBool> {
        &self.stop_flag
    }
}

/// Runtime context handed to [`Source::run`](super::actor::Source::run).
///
/// Sources don't have receive loops, so the stop signal is surfaced
/// as a pollable [`AtomicBool`] that the source checks between
/// chunks of work (in GStreamer callback threads, every frame on
/// the hot path, etc.).
pub struct SourceContext {
    own_name: StageName,
    registry: Arc<Registry>,
    shared: Arc<SharedStore>,
    stop_flag: Arc<AtomicBool>,
}

impl SourceContext {
    /// Internal constructor.
    #[allow(dead_code, reason = "consumed by System in Component 2")]
    pub(crate) fn new(
        own_name: StageName,
        registry: Arc<Registry>,
        shared: Arc<SharedStore>,
        stop_flag: Arc<AtomicBool>,
    ) -> Self {
        Self {
            own_name,
            registry,
            shared,
            stop_flag,
        }
    }

    /// This source's registered [`StageName`].
    pub fn own_name(&self) -> &StageName {
        &self.own_name
    }

    /// Resolve the [`Addr<M>`] registered under `peer`.
    pub fn resolve<M: Envelope>(&self, peer: &StageName) -> Result<Addr<M>> {
        self.registry.get::<M>(peer)
    }

    /// Resolve an [`OperatorSink<M>`] pointing at `peer`, owned by
    /// this source.
    pub fn sink<M: Envelope>(&self, peer: &StageName) -> Result<OperatorSink<M>> {
        Ok(OperatorSink::new(
            self.own_name.clone(),
            self.resolve::<M>(peer)?,
        ))
    }

    /// Build a [`Router<M>`] owned by this source with an optional
    /// default peer.  Sources typically wire their router from the
    /// builder's `.downstream(name)` call; pass `default = None`
    /// here when the source only uses on-demand routed sends.
    pub fn router<M: Envelope>(&self, default: Option<&StageName>) -> Result<Router<M>> {
        let default_sink = default.map(|peer| self.sink::<M>(peer)).transpose()?;
        Ok(Router::new(
            self.own_name.clone(),
            self.registry.clone(),
            default_sink,
        ))
    }

    /// Singleton shared-state look-up.
    pub fn shared<T: Send + Sync + 'static>(&self) -> Option<Arc<T>> {
        self.shared.get::<T>()
    }

    /// Named shared-state look-up.
    pub fn shared_as<T: Send + Sync + 'static>(&self, key: &str) -> Option<Arc<T>> {
        self.shared.get_as::<T>(key)
    }

    /// Whether the orchestrator has asked the source to wind down.
    /// Sources poll this between chunks of work and exit cleanly
    /// once it flips to `true`.
    pub fn is_stopping(&self) -> bool {
        self.stop_flag.load(Ordering::Relaxed)
    }

    /// Cloneable handle to the shared stop flag — useful for
    /// GStreamer callbacks that outlive the `run` call.
    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        self.stop_flag.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::envelope::ShutdownHint;
    use crate::framework::handler::Flow;
    use crate::framework::supervisor::StageKind;
    use crossbeam::channel::bounded;

    struct DummyEnv;
    impl Envelope for DummyEnv {
        fn as_shutdown(&self) -> Option<ShutdownHint<'_>> {
            None
        }
    }

    struct DummyActor;
    impl Actor for DummyActor {
        type Msg = DummyEnv;
        fn handle(&mut self, _msg: Self::Msg, _ctx: &mut Context<Self>) -> Result<Flow> {
            Ok(Flow::Cont)
        }
    }

    fn ctx() -> Context<DummyActor> {
        Context::new(
            StageName::unnamed(StageKind::Infer),
            Arc::new(Registry::new()),
            Arc::new(SharedStore::new()),
            Arc::new(AtomicBool::new(false)),
        )
    }

    #[test]
    fn arm_graceful_earliest_wins() {
        let mut c = ctx();
        c.arm_graceful_stop(Some(Duration::from_secs(1)), "first");
        let first = c.deadline().unwrap();
        c.arm_graceful_stop(Some(Duration::from_secs(10)), "second");
        assert_eq!(c.deadline().unwrap(), first, "earlier deadline must win");
    }

    #[test]
    fn arm_graceful_none_sets_break_now() {
        let mut c = ctx();
        c.arm_graceful_stop(None, "bye");
        assert!(c.break_now());
        assert!(c.should_quit());
    }

    #[test]
    fn abort_sets_break_now() {
        let mut c = ctx();
        c.arm_abort("fatal");
        assert!(c.break_now());
        assert!(c.should_quit());
    }

    #[test]
    fn stop_flag_drives_should_quit() {
        let c = ctx();
        assert!(!c.should_quit());
        c.stop_flag().store(true, Ordering::Relaxed);
        assert!(c.should_quit());
    }

    #[test]
    fn deadline_triggers_should_quit_after_elapsed() {
        let mut c = ctx();
        c.arm_graceful_stop(Some(Duration::from_millis(1)), "short");
        std::thread::sleep(Duration::from_millis(5));
        assert!(c.should_quit());
    }

    #[test]
    fn build_ctx_resolves_registered_address() {
        let mut reg = Registry::new();
        let name = StageName::unnamed(StageKind::Infer);
        let (tx, _rx) = bounded::<DummyEnv>(1);
        reg.insert::<DummyEnv>(name.clone(), Addr::new(name.clone(), tx));
        let reg = Arc::new(reg);
        let shared = SharedStore::new();
        let self_name = StageName::unnamed(StageKind::Tracker);
        let bx = BuildCtx::new(&self_name, &reg, &shared);
        let a: Addr<DummyEnv> = bx.addr(&name).unwrap();
        assert_eq!(a.name(), &name);
        assert!(bx
            .addr::<DummyEnv>(&StageName::unnamed(StageKind::Picasso))
            .is_err());
    }

    #[test]
    fn build_ctx_router_resolves_default_and_send_to() {
        let mut reg = Registry::new();
        let default_name = StageName::unnamed(StageKind::Tracker);
        let explicit_name = StageName::unnamed(StageKind::Picasso);
        let (tx_default, rx_default) = bounded::<DummyEnv>(2);
        let (tx_explicit, rx_explicit) = bounded::<DummyEnv>(2);
        reg.insert::<DummyEnv>(
            default_name.clone(),
            Addr::new(default_name.clone(), tx_default),
        );
        reg.insert::<DummyEnv>(
            explicit_name.clone(),
            Addr::new(explicit_name.clone(), tx_explicit),
        );
        let reg = Arc::new(reg);
        let shared = SharedStore::new();
        let self_name = StageName::unnamed(StageKind::Infer);
        let bx = BuildCtx::new(&self_name, &reg, &shared);

        let router = bx.router::<DummyEnv>(Some(&default_name)).unwrap();
        assert_eq!(router.default_peer(), Some(&default_name));
        assert!(router.send(DummyEnv));
        assert!(rx_default.try_recv().is_ok());
        assert!(router.send_to(&explicit_name, DummyEnv).unwrap());
        assert!(rx_explicit.try_recv().is_ok());

        let no_default = bx.router::<DummyEnv>(None).unwrap();
        assert!(!no_default.has_default());
        assert!(!no_default.send(DummyEnv));
    }

    #[test]
    fn source_context_stop_flag_round_trip() {
        let sc = SourceContext::new(
            StageName::unnamed(StageKind::Mp4Demux),
            Arc::new(Registry::new()),
            Arc::new(SharedStore::new()),
            Arc::new(AtomicBool::new(false)),
        );
        assert!(!sc.is_stopping());
        sc.stop_flag().store(true, Ordering::Relaxed);
        assert!(sc.is_stopping());
    }
}
