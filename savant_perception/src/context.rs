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
//! | [`BuildCtx`]      | *(none)*                    | each factory call inside `System::run`
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
/// Factories run inside
/// [`System::run`](super::system::System::run), after every
/// actor's [`Addr`] has already been published in the registry.
/// That ordering lets factories resolve downstream peers eagerly
/// from the registry and produce a ready-to-run actor value by
/// return — no `Sender` cloning through closures.
///
/// `BuildCtx` is entirely immutable: factories observe but don't
/// mutate the framework's construction state.  Any mutation
/// (e.g. populating a local `HashMap<source_id, State>`) happens
/// inside the actor value itself.
pub struct BuildCtx<'a> {
    own_name: &'a StageName,
    registry: &'a Arc<Registry>,
    shared: &'a Arc<SharedStore>,
    stop_flag: &'a Arc<AtomicBool>,
}

impl<'a> BuildCtx<'a> {
    /// Internal constructor used by
    /// [`System::run`](super::system::System::run) to hand a
    /// borrowed context to each factory.
    ///
    /// `stop_flag` is the same `Arc<AtomicBool>` the actor loop
    /// driver (or the source runtime) reads for cooperative
    /// shutdown; threading it through [`BuildCtx::hook_ctx`]
    /// lets egress hooks flip it via [`HookCtx::request_stop`].
    #[allow(dead_code, reason = "called from System::run before each factory")]
    pub(crate) fn new(
        own_name: &'a StageName,
        registry: &'a Arc<Registry>,
        shared: &'a Arc<SharedStore>,
        stop_flag: &'a Arc<AtomicBool>,
    ) -> Self {
        Self {
            own_name,
            registry,
            shared,
            stop_flag,
        }
    }

    /// Clone out the `Arc<SharedStore>` for callers that need to
    /// hold shared-state references past the factory's lifetime
    /// (e.g. to stash in a thread-local hook context fired from an
    /// egress worker thread).
    pub fn shared_arc(&self) -> Arc<SharedStore> {
        Arc::clone(self.shared)
    }

    /// Clone out the `Arc<Registry>` — same rationale as
    /// [`shared_arc`](Self::shared_arc).
    pub fn registry_arc(&self) -> Arc<Registry> {
        Arc::clone(self.registry)
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

    /// Owned snapshot of this `BuildCtx`'s construction state,
    /// suitable for capturing into off-loop egress hooks.
    ///
    /// See [`HookCtx`] for the full rationale.  Typical wiring
    /// inside a stage's actor factory:
    ///
    /// ```ignore
    /// .factory(move |bx| {
    ///     let hook_ctx = bx.hook_ctx();
    ///     let router = bx.router::<PipelineMsg>(downstream.as_ref())?;
    ///     let on_frame = on_frame.clone();
    ///     let operator = (operator_factory)(bx, Box::new(move |out| {
    ///         on_frame(out, &router, &hook_ctx);
    ///     }))?;
    ///     ...
    /// })
    /// ```
    pub fn hook_ctx(&self) -> HookCtx {
        HookCtx {
            own_name: self.own_name.clone(),
            registry: Arc::clone(self.registry),
            shared: Arc::clone(self.shared),
            stop_flag: Arc::clone(self.stop_flag),
        }
    }
}

/// Read-only snapshot of the framework's construction context —
/// owned clones of the actor's [`StageName`], [`Arc<Registry>`],
/// and [`Arc<SharedStore>`] — designed to be captured into
/// **off-loop egress hooks**.
///
/// # When to use this vs. [`Context<A>`]
///
/// | Hook fires…                          | Use               |
/// |--------------------------------------|-------------------|
/// | Inside the actor receive loop        | [`Context<A>`]    |
/// | From a worker thread / source thread | [`HookCtx`]       |
///
/// Egress hooks on operators that run their own worker threads
/// (nvinfer, nvtracker, decoder) or on [`Source`](super::actor::Source)-style
/// stages (mp4 demuxer, picasso engine) cannot share
/// [`Context<A>`] — the driver owns it, and it carries mutable
/// shutdown state that would race with the loop.  `HookCtx` is
/// the safe, read-only substitute: every accessor is infallible
/// and cheap (clones of existing `Arc`s).
///
/// # What it exposes
///
/// * [`own_name`](Self::own_name) — the actor's registered name,
///   for log-line attribution that stays aligned with the rest of
///   the system.
/// * [`shared`](Self::shared) / [`shared_as`](Self::shared_as) —
///   [`SharedStore`] look-ups so hooks can reach any `Arc<T>`
///   that the application published through
///   [`System::insert_shared`](super::system::System::insert_shared)
///   (or the named-key variants) before `System::run`.
/// * [`resolve`](Self::resolve) — registry look-up returning a
///   typed [`Addr<M>`] for "route this update to the stage named
///   `foo`"-style side channels (rare — prefer router peers when
///   possible).
///
/// Build one from a factory via [`BuildCtx::hook_ctx`], capture
/// clones into each per-variant hook, and dispatch calls receive
/// the same snapshot by `&HookCtx`.
#[derive(Clone)]
pub struct HookCtx {
    own_name: StageName,
    registry: Arc<Registry>,
    shared: Arc<SharedStore>,
    stop_flag: Arc<AtomicBool>,
}

impl HookCtx {
    /// Direct constructor for stages that assemble a hook ctx
    /// out of hand-picked pieces (e.g. tests).  Most call sites
    /// should prefer [`BuildCtx::hook_ctx`] or
    /// [`SourceContext::hook_ctx`], both of which thread the
    /// loop driver's `stop_flag` for free.
    ///
    /// The `stop_flag` parameter is the shared cooperative-stop
    /// sentinel the actor or source runtime already observes;
    /// hooks can flip it via [`Self::request_stop`] to request a
    /// graceful exit from worker threads.
    pub fn new(
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

    /// The actor's registered [`StageName`].
    pub fn own_name(&self) -> &StageName {
        &self.own_name
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

    /// Resolve the [`Addr<M>`] registered under `peer`, returning
    /// `Err` on an unknown name or type mismatch.  Equivalent to
    /// [`BuildCtx::addr`] but usable after the factory returns.
    pub fn resolve<M: Envelope>(&self, peer: &StageName) -> Result<Addr<M>> {
        self.registry.get::<M>(peer)
    }

    /// Clone-out the full [`Arc<SharedStore>`] — provided as an
    /// escape hatch for callers that need to hand the store off
    /// (e.g. spawn a helper thread that stashes its own state).
    pub fn shared_arc(&self) -> Arc<SharedStore> {
        Arc::clone(&self.shared)
    }

    /// Clone-out the full [`Arc<Registry>`] — escape hatch
    /// analogous to [`shared_arc`](Self::shared_arc).
    pub fn registry_arc(&self) -> Arc<Registry> {
        Arc::clone(&self.registry)
    }

    /// Request cooperative shutdown by flipping the shared stop
    /// flag.
    ///
    /// The stop flag is the same `Arc<AtomicBool>` the loop driver
    /// observes through [`Context::should_quit`] (for actors) and
    /// [`SourceContext::is_stopping`] (for sources).  The flag is
    /// also flipped by [`System`](super::system::System) when its
    /// shutdown handler returns
    /// [`ShutdownAction::Broadcast`](super::shutdown::ShutdownAction::Broadcast)
    /// or [`Ordered`](super::shutdown::ShutdownAction::Ordered),
    /// so calling this from a worker thread reaches the same flag
    /// the supervisor uses.
    ///
    /// The call is non-blocking and only sets the flag; the actual
    /// exit happens at the next polling point of the owning stage:
    ///
    /// * an actor finishes its current `handle` / `on_tick`, then
    ///   breaks out of the receive loop and runs
    ///   [`Actor::stopping`];
    /// * a source observes the flag the next time it calls
    ///   [`SourceContext::is_stopping`] and returns `Ok(())` from
    ///   [`Source::run`](super::actor::Source::run).
    pub fn request_stop(&self) {
        self.stop_flag.store(true, Ordering::Relaxed);
    }

    /// Whether [`Self::request_stop`] has already been invoked
    /// (by this hook, another hook on the same stage, or the
    /// supervisor itself).
    pub fn stop_requested(&self) -> bool {
        self.stop_flag.load(Ordering::Relaxed)
    }

    /// Clone-out the shared stop flag — escape hatch for
    /// unusual wiring (e.g. a GStreamer probe captured before
    /// `run` starts that wants to observe the same flag).
    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.stop_flag)
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
    /// context before handing it to
    /// [`run_actor`](super::loop_driver::run_actor).
    #[allow(
        dead_code,
        reason = "called from System::run before spawning each actor"
    )]
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
    /// by stages that need a per-source deadline (e.g. a
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
    #[allow(dead_code, reason = "used by System::run and unit tests")]
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
    /// Internal constructor — built by
    /// [`System::run`](super::system::System::run) before
    /// spawning the source's worker thread.
    #[allow(
        dead_code,
        reason = "called from System::run before spawning each source"
    )]
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

    /// Whether the supervisor has asked the source to wind down.
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

    /// Owned [`HookCtx`] snapshot, suitable for capturing into
    /// per-variant source hooks that fire from GStreamer callback
    /// threads (i.e. outside the `run` stack frame).
    ///
    /// Analogous to [`BuildCtx::hook_ctx`] but for
    /// [`Source`](super::actor::Source)-shaped stages.
    pub fn hook_ctx(&self) -> HookCtx {
        HookCtx {
            own_name: self.own_name.clone(),
            registry: Arc::clone(&self.registry),
            shared: Arc::clone(&self.shared),
            stop_flag: Arc::clone(&self.stop_flag),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envelope::ShutdownHint;
    use crate::handler::Flow;
    use crate::supervisor::StageKind;
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
        let shared = Arc::new(SharedStore::new());
        let stop_flag = Arc::new(AtomicBool::new(false));
        let self_name = StageName::unnamed(StageKind::Tracker);
        let bx = BuildCtx::new(&self_name, &reg, &shared, &stop_flag);
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
        let shared = Arc::new(SharedStore::new());
        let stop_flag = Arc::new(AtomicBool::new(false));
        let self_name = StageName::unnamed(StageKind::Infer);
        let bx = BuildCtx::new(&self_name, &reg, &shared, &stop_flag);

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

    #[test]
    fn hook_ctx_request_stop_flips_shared_flag() {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let hc = HookCtx::new(
            StageName::unnamed(StageKind::Infer),
            Arc::new(Registry::new()),
            Arc::new(SharedStore::new()),
            Arc::clone(&stop_flag),
        );
        assert!(!hc.stop_requested());
        assert!(!stop_flag.load(Ordering::Relaxed));
        hc.request_stop();
        assert!(hc.stop_requested());
        assert!(stop_flag.load(Ordering::Relaxed));
    }

    #[test]
    fn hook_ctx_from_build_ctx_shares_stop_flag_with_actor_context() {
        let reg = Arc::new(Registry::new());
        let shared = Arc::new(SharedStore::new());
        let stop_flag = Arc::new(AtomicBool::new(false));
        let name = StageName::unnamed(StageKind::Infer);
        let bx = BuildCtx::new(&name, &reg, &shared, &stop_flag);
        let hc = bx.hook_ctx();
        // Simulate the actor loop's runtime Context consuming the
        // same stop flag.
        let actor_ctx: Context<DummyActor> = Context::new(
            name.clone(),
            Arc::clone(&reg),
            Arc::clone(&shared),
            Arc::clone(&stop_flag),
        );
        assert!(!actor_ctx.should_quit());
        // Off-thread hook flips the flag; the actor loop
        // observes it through `should_quit` without any extra
        // plumbing.
        hc.request_stop();
        assert!(actor_ctx.should_quit());
    }

    #[test]
    fn hook_ctx_from_source_context_shares_stop_flag() {
        let sc = SourceContext::new(
            StageName::unnamed(StageKind::Mp4Demux),
            Arc::new(Registry::new()),
            Arc::new(SharedStore::new()),
            Arc::new(AtomicBool::new(false)),
        );
        let hc = sc.hook_ctx();
        assert!(!sc.is_stopping());
        hc.request_stop();
        assert!(sc.is_stopping());
    }
}
