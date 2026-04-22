//! [`System`] — the top-level builder / runner that stitches every
//! actor + source + shared value into a live pipeline.
//!
//! Usage pattern:
//!
//! ```ignore
//! let mut sys = System::new();
//! sys.insert_shared(my_config);
//! sys.register_source(demux_builder);
//! sys.register_actor(decoder_builder);
//! sys.register_actor(nvinfer_builder);
//! // …
//! sys.on_shutdown(|cause, ctx| {
//!     // custom policy; see [`crate::framework::shutdown`].
//!     Ok(ShutdownAction::Broadcast { grace: None, reason: Cow::Borrowed("custom") })
//! });
//! sys.run()?;
//! ```
//!
//! `System::run` drives the two-phase build described in the
//! module-level design docs:
//!
//! 1. **Phase 1** (inside [`register_actor`](System::register_actor)
//!    / [`register_source`](System::register_source)) — allocate
//!    each actor's inbox and publish its `Addr<M>` under its
//!    `StageName` in the registry.  Done eagerly at registration
//!    so the order of registration calls doesn't matter.
//! 2. **Phase 2** — run each factory against a `BuildCtx` with the
//!    fully-populated registry + shared store.  The factory
//!    returns the actor value.
//! 3. **Phase 3** — spawn one OS thread per actor / source, each
//!    with its own [`StageExitGuard`], running the appropriate
//!    receive-loop driver.
//!
//! After phase 3 the supervisor blocks on the exit channel,
//! invokes the installed (or default) [`ShutdownHandler`] on
//! each trigger, and joins every thread in LIFO registration
//! order so the pipeline tail logs first on a clean run.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::{anyhow, Context as _, Result};
use crossbeam::channel::bounded;

use super::actor::{Actor, Source};
use super::addr::Addr;
use super::builder::{ActorBuilder, SourceBuilder};
use super::context::{BuildCtx, Context, SourceContext};
use super::envelope::Envelope;
use super::loop_driver::run_actor;
use super::registry::Registry;
use super::shared::SharedStore;
use super::shutdown::{
    DefaultShutdownHandler, ShutdownAction, ShutdownCause, ShutdownCtx, ShutdownHandler,
};
use super::supervisor::{
    exit_channel, ExitReceiver, ExitSender, StageExit, StageExitGuard, StageKind, StageName,
};

/// Type-erased "send a shutdown envelope to this actor" closure,
/// built at registration time from the actor's envelope type.
type ShutdownSender = Box<dyn Fn(Option<Duration>, Cow<'static, str>) -> bool + Send + Sync>;

/// Pending factory + thread-spawn closure captured in phase 1.
/// Runs in phase 2 (factory call) + phase 3 (thread spawn).
type ThreadLauncher = Box<dyn FnOnce(BuildArgs) -> Result<JoinHandle<Result<()>>> + Send>;

/// Arguments threaded into each [`ThreadLauncher`] during phase 2/3.
struct BuildArgs {
    registry: Arc<Registry>,
    shared: Arc<SharedStore>,
    stop_flag: Arc<AtomicBool>,
    exit_tx: ExitSender,
}

/// Default quiescence grace period applied by the
/// [`DefaultShutdownHandler`] when no custom handler is
/// installed.
///
/// Matches the pre-existing cars-tracking pipeline constant — the
/// window between observing the first stage exit and broadcasting
/// `Shutdown` on every inbox.  Long enough to let an in-flight
/// in-band `SourceEos` drain through NVDEC + downstream channels.
pub const DEFAULT_QUIESCENCE_GRACE: Duration = Duration::from_secs(1);

/// Top-level pipeline builder / runner.  See the module docs for
/// the full usage pattern.
pub struct System {
    shared: SharedStore,
    registry: Registry,
    launchers: Vec<PendingLaunch>,
    shutdown_senders: HashMap<StageName, ShutdownSender>,
    shutdown_handler: Option<Box<dyn ShutdownHandler>>,
    quiescence: Duration,
    install_ctrlc: bool,
}

struct PendingLaunch {
    name: StageName,
    launcher: ThreadLauncher,
}

impl Default for System {
    fn default() -> Self {
        Self::new()
    }
}

impl System {
    /// Empty system with defaults: no shutdown handler (i.e.
    /// [`DefaultShutdownHandler`] will be used), Ctrl+C handler
    /// installation *enabled*, and the default quiescence grace
    /// period.
    pub fn new() -> Self {
        Self {
            shared: SharedStore::new(),
            registry: Registry::new(),
            launchers: Vec::new(),
            shutdown_senders: HashMap::new(),
            shutdown_handler: None,
            quiescence: DEFAULT_QUIESCENCE_GRACE,
            install_ctrlc: true,
        }
    }

    /// Override the quiescence grace period applied by the
    /// built-in [`DefaultShutdownHandler`].
    pub fn quiescence(mut self, d: Duration) -> Self {
        self.quiescence = d;
        self
    }

    /// Whether to install a process-wide Ctrl+C handler that
    /// routes `SIGINT` into the supervisor's back-channel as a
    /// [`ShutdownCause::CtrlC`].  On by default.  Set to `false`
    /// for unit tests or when the host application owns the
    /// Ctrl+C handler.
    pub fn install_ctrlc_handler(mut self, enabled: bool) -> Self {
        self.install_ctrlc = enabled;
        self
    }

    /// Install a custom [`ShutdownHandler`] policy.  When omitted,
    /// the [`DefaultShutdownHandler`] with the configured
    /// quiescence is used.
    pub fn on_shutdown<H: ShutdownHandler + 'static>(mut self, handler: H) -> Self {
        self.shutdown_handler = Some(Box::new(handler));
        self
    }

    /// Publish a singleton shared value.  See
    /// [`SharedStore::insert`].
    pub fn insert_shared<T: Send + Sync + 'static>(&mut self, value: T) -> &mut Self {
        self.shared.insert(value);
        self
    }

    /// Publish a named shared value.  See
    /// [`SharedStore::insert_as`].
    pub fn insert_shared_as<T: Send + Sync + 'static>(
        &mut self,
        key: impl Into<String>,
        value: T,
    ) -> &mut Self {
        self.shared.insert_as(key, value);
        self
    }

    /// Publish a singleton shared `Arc<T>`.
    pub fn insert_shared_arc<T: Send + Sync + 'static>(&mut self, value: Arc<T>) -> &mut Self {
        self.shared.insert_arc(value);
        self
    }

    /// Register an actor.  Allocates the actor's inbox, publishes
    /// its [`Addr<M>`] under the builder's name, and stores the
    /// factory for phase 2.  Returns the registered
    /// [`Addr<M>`](super::addr::Addr) so the orchestrator can
    /// retain its own clone if it wants to inject synthetic
    /// messages (not typically needed — peers resolve through
    /// the registry).
    pub fn register_actor<A>(&mut self, builder: ActorBuilder<A>) -> Result<Addr<A::Msg>>
    where
        A: Actor,
    {
        let parts = builder.into_parts();
        let name = parts.name.clone();
        if self.registry.contains(&name) {
            return Err(anyhow!("register_actor: duplicate stage name {name}"));
        }
        let (tx, rx) = bounded::<A::Msg>(parts.inbox_capacity);
        let addr = Addr::new(name.clone(), tx.clone());
        self.registry.insert::<A::Msg>(name.clone(), addr.clone());

        // Capture the sender into a type-erased shutdown closure.
        let sender_for_shutdown = tx.clone();
        let shutdown_sender: ShutdownSender = Box::new(move |grace, reason| {
            let Some(msg) = <A::Msg as Envelope>::build_shutdown(grace, reason) else {
                return false;
            };
            sender_for_shutdown.try_send(msg).is_ok()
        });
        self.shutdown_senders.insert(name.clone(), shutdown_sender);

        let factory = parts.factory;
        let poll_override = parts.poll_timeout;
        let launcher_name = name.clone();
        let launcher: ThreadLauncher =
            Box::new(move |args: BuildArgs| -> Result<JoinHandle<Result<()>>> {
                let BuildArgs {
                    registry,
                    shared,
                    stop_flag,
                    exit_tx,
                } = args;
                let bx = BuildCtx::new(&launcher_name, &registry, shared.as_ref());
                let actor: A = factory(&bx)
                    .with_context(|| format!("actor factory for [{launcher_name}] failed"))?;
                let _ = poll_override; // reserved for future per-instance overrides
                let ctx: Context<A> =
                    Context::new(launcher_name.clone(), registry, shared, stop_flag);
                let thread_name = format!("actor-{launcher_name}");
                let stage_for_guard = launcher_name.clone();
                let handle = thread::Builder::new()
                    .name(thread_name)
                    .spawn(move || -> Result<()> {
                        let _guard = StageExitGuard::new(stage_for_guard, exit_tx);
                        run_actor(actor, rx, ctx)
                    })
                    .map_err(|e| anyhow!("failed to spawn actor thread [{launcher_name}]: {e}"))?;
                Ok(handle)
            });
        self.launchers.push(PendingLaunch { name, launcher });
        Ok(addr)
    }

    /// Register a source (no-inbox producer).  The source's
    /// factory runs in phase 2 once every peer's address is
    /// published.  Sources cooperate with shutdown via the
    /// shared stop flag ([`SourceContext::is_stopping`]).
    pub fn register_source<S>(&mut self, builder: SourceBuilder<S>) -> Result<()>
    where
        S: Source,
    {
        let parts = builder.into_parts();
        let name = parts.name.clone();
        if self.registry.contains(&name) {
            return Err(anyhow!("register_source: duplicate stage name {name}"));
        }
        // Reserve the registry slot to prevent another actor from
        // registering the same name; a source has no inbox so we
        // don't insert an `Addr<_>`.
        // We use a "presence" marker: the registry's `contains`
        // will already return true for this name if we insert a
        // dummy `Addr<()>`.  To keep the `Envelope` bound honest
        // we skip the registry insert and track presence via a
        // separate set of source names.
        let factory = parts.factory;
        let launcher_name = name.clone();
        let launcher: ThreadLauncher =
            Box::new(move |args: BuildArgs| -> Result<JoinHandle<Result<()>>> {
                let BuildArgs {
                    registry,
                    shared,
                    stop_flag,
                    exit_tx,
                } = args;
                let bx = BuildCtx::new(&launcher_name, &registry, shared.as_ref());
                let src: S = factory(&bx)
                    .with_context(|| format!("source factory for [{launcher_name}] failed"))?;
                let sc = SourceContext::new(launcher_name.clone(), registry, shared, stop_flag);
                let thread_name = format!("source-{launcher_name}");
                let stage_for_guard = launcher_name.clone();
                let handle = thread::Builder::new()
                    .name(thread_name)
                    .spawn(move || -> Result<()> {
                        let _guard = StageExitGuard::new(stage_for_guard, exit_tx);
                        src.run(sc)
                    })
                    .map_err(|e| anyhow!("failed to spawn source thread [{launcher_name}]: {e}"))?;
                Ok(handle)
            });
        self.launchers.push(PendingLaunch { name, launcher });
        Ok(())
    }

    /// Every stage registered so far (actors + sources), in
    /// registration order.
    pub fn stages(&self) -> Vec<StageName> {
        self.launchers.iter().map(|p| p.name.clone()).collect()
    }

    /// Stages of a given [`StageKind`].
    pub fn stages_of(&self, kind: StageKind) -> Vec<StageName> {
        self.launchers
            .iter()
            .filter(|p| p.name.kind == kind)
            .map(|p| p.name.clone())
            .collect()
    }

    /// Finalize the system: run phase 2 (factory calls) + phase 3
    /// (thread spawn), install the Ctrl+C handler, block on the
    /// exit channel, execute the shutdown handler's actions, join
    /// every thread, and return an aggregated result per stage.
    pub fn run(self) -> Result<SystemReport> {
        let Self {
            shared,
            registry,
            launchers,
            shutdown_senders,
            shutdown_handler,
            quiescence,
            install_ctrlc,
        } = self;

        if launchers.is_empty() {
            log::warn!("[system] run() called with no registered actors or sources; nothing to do");
            return Ok(SystemReport {
                stage_results: Vec::new(),
                trigger: None,
                history: Vec::new(),
            });
        }

        let registry = Arc::new(registry);
        let shared = Arc::new(shared);
        let stop_flag = Arc::new(AtomicBool::new(false));
        let (exit_tx, exit_rx) = exit_channel();

        // Ctrl+C handler must use a clone of the exit sender; the
        // handler lives for the process lifetime.
        if install_ctrlc {
            let exit_tx_ctrl = exit_tx.clone();
            if let Err(e) = ctrlc::try_set_handler(move || {
                log::warn!("[ctrl-c] received; notifying supervisor");
                let _ = exit_tx_ctrl.send(StageExit {
                    stage: StageName::unnamed(StageKind::CtrlC),
                });
            }) {
                log::warn!("[system] could not install Ctrl+C handler: {e}");
            }
        }

        // Phase 2 + 3: run factories and spawn threads.
        let mut handles: Vec<(StageName, JoinHandle<Result<()>>)> =
            Vec::with_capacity(launchers.len());
        let mut pending_factory_errors: Vec<(StageName, anyhow::Error)> = Vec::new();
        for launch in launchers {
            let PendingLaunch { name, launcher } = launch;
            let args = BuildArgs {
                registry: registry.clone(),
                shared: shared.clone(),
                stop_flag: stop_flag.clone(),
                exit_tx: exit_tx.clone(),
            };
            match launcher(args) {
                Ok(handle) => handles.push((name, handle)),
                Err(e) => {
                    log::error!("[system] failed to spawn [{name}]: {e:#}");
                    pending_factory_errors.push((name, e));
                }
            }
        }

        // If any factory failed, trigger shutdown immediately.
        if !pending_factory_errors.is_empty() {
            log::error!(
                "[system] {} factory/spawn failure(s); aborting pipeline",
                pending_factory_errors.len()
            );
            let _ = exit_tx.send(StageExit {
                stage: StageName::unnamed(StageKind::CtrlC), // synthetic
            });
        }

        // Drop our local exit_tx clone; remaining senders are
        // per-thread guards (+ Ctrl+C clone, kept alive by ctrlc
        // crate internals).
        drop(exit_tx);

        let all_stage_names: Vec<StageName> = handles.iter().map(|(n, _)| n.clone()).collect();

        // Supervisor: pump triggers until we observe a terminating
        // action, then broadcast + join.
        let (trigger, history) = run_supervisor(
            &exit_rx,
            shutdown_handler.unwrap_or_else(|| Box::new(DefaultShutdownHandler::new(quiescence))),
            &all_stage_names,
            &shutdown_senders,
            stop_flag.clone(),
        );

        // Drain any remaining exit signals for diagnostic
        // visibility.
        while let Ok(extra) = exit_rx.try_recv() {
            log::debug!("[system] subsequent exit from [{}]", extra.stage);
        }

        // Join every thread in reverse registration order so the
        // pipeline tail logs first on a clean exit.
        let mut stage_results: Vec<(StageName, Result<()>)> = Vec::with_capacity(handles.len());
        while let Some((name, h)) = handles.pop() {
            let r = h
                .join()
                .unwrap_or_else(|_| Err(anyhow!("[{}] thread panicked", name)));
            stage_results.push((name, r));
        }

        // Propagate any factory/spawn errors as stage results.
        for (name, err) in pending_factory_errors {
            stage_results.push((name, Err(err)));
        }

        Ok(SystemReport {
            stage_results,
            trigger,
            history,
        })
    }
}

/// Outcome summary returned from [`System::run`].
#[must_use = "check per-stage results to observe stage errors"]
#[derive(Debug)]
pub struct SystemReport {
    /// Per-stage join results, in reverse registration order —
    /// i.e. downstream → upstream.  A `Result` per entry mirrors
    /// the thread's return value (and a panic is surfaced as
    /// `Err`).
    pub stage_results: Vec<(StageName, Result<()>)>,
    /// The first cause the supervisor acted on (or `None` if the
    /// system had no registered stages and [`System::run`]
    /// returned immediately).
    pub trigger: Option<ShutdownCause>,
    /// Full history of causes the supervisor processed (including
    /// ignored ones).
    pub history: Vec<ShutdownCause>,
}

impl SystemReport {
    /// Convert the per-stage results into a single aggregated
    /// `Result<()>` — logs each error, returns `Err(<combined>)`
    /// iff any stage returned `Err`, else `Ok(())`.
    pub fn into_result(self) -> Result<()> {
        let mut first_err: Option<anyhow::Error> = None;
        for (name, r) in self.stage_results {
            if let Err(e) = r {
                log::error!("[{name}] stage error: {e:#}");
                if first_err.is_none() {
                    first_err = Some(e.context(format!("stage [{name}] failed")));
                }
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

fn run_supervisor(
    exit_rx: &ExitReceiver,
    mut handler: Box<dyn ShutdownHandler>,
    stages: &[StageName],
    shutdown_senders: &HashMap<StageName, ShutdownSender>,
    stop_flag: Arc<AtomicBool>,
) -> (Option<ShutdownCause>, Vec<ShutdownCause>) {
    let mut history: Vec<ShutdownCause> = Vec::new();
    let mut trigger: Option<ShutdownCause> = None;
    loop {
        let Ok(exit) = exit_rx.recv() else {
            log::error!("[system] exit channel disconnected before a shutdown trigger; broadcasting defensively");
            broadcast_shutdown(
                stages,
                shutdown_senders,
                None,
                Cow::Borrowed("back-channel closed"),
            );
            stop_flag.store(true, Ordering::Relaxed);
            break;
        };
        let cause = classify_cause(exit);
        history.push(cause.clone());
        let mut ctx = ShutdownCtx::fake(stages, &history[..history.len().saturating_sub(1)]);
        let action = match handler.on_shutdown(cause.clone(), &mut ctx) {
            Ok(a) => a,
            Err(e) => {
                log::error!("[system] shutdown handler error: {e:#}; broadcasting defensively");
                ShutdownAction::Broadcast {
                    grace: None,
                    reason: Cow::Borrowed("shutdown handler error"),
                }
            }
        };
        match action {
            ShutdownAction::Ignore => {
                log::info!("[system] handler chose Ignore for {cause:?}; continuing");
                continue;
            }
            ShutdownAction::Broadcast { grace, reason } => {
                trigger = Some(cause);
                broadcast_shutdown(stages, shutdown_senders, grace, reason);
                stop_flag.store(true, Ordering::Relaxed);
                break;
            }
            ShutdownAction::Ordered { waves } => {
                trigger = Some(cause);
                for (i, wave) in waves.into_iter().enumerate() {
                    log::info!(
                        "[system] shutdown wave #{i}: {} target(s), grace={:?}, pause={:?}",
                        if wave.targets.is_empty() {
                            format!("{} (all)", stages.len())
                        } else {
                            wave.targets.len().to_string()
                        },
                        wave.grace,
                        wave.pause_after
                    );
                    let targets: Vec<StageName> = if wave.targets.is_empty() {
                        stages.to_vec()
                    } else {
                        wave.targets
                    };
                    broadcast_shutdown_targets(&targets, shutdown_senders, wave.grace, wave.reason);
                    if !wave.pause_after.is_zero() {
                        thread::sleep(wave.pause_after);
                    }
                }
                stop_flag.store(true, Ordering::Relaxed);
                break;
            }
        }
    }
    (trigger, history)
}

fn classify_cause(exit: StageExit) -> ShutdownCause {
    if exit.stage.kind == StageKind::CtrlC {
        ShutdownCause::CtrlC
    } else {
        ShutdownCause::StageExit { stage: exit.stage }
    }
}

fn broadcast_shutdown(
    stages: &[StageName],
    senders: &HashMap<StageName, ShutdownSender>,
    grace: Option<Duration>,
    reason: Cow<'static, str>,
) {
    broadcast_shutdown_targets(stages, senders, grace, reason);
}

fn broadcast_shutdown_targets(
    targets: &[StageName],
    senders: &HashMap<StageName, ShutdownSender>,
    grace: Option<Duration>,
    reason: Cow<'static, str>,
) {
    log::info!(
        "[system] broadcasting Shutdown to {} stage(s) (grace={:?}, reason={reason})",
        targets.len(),
        grace
    );
    for name in targets {
        let Some(sender) = senders.get(name) else {
            log::debug!("[system] no shutdown sender for [{name}] (source or unregistered stage)");
            continue;
        };
        let sent = sender(grace, reason.clone());
        if !sent {
            log::debug!("[system] shutdown send to [{name}] failed (inbox closed or envelope lacks shutdown variant)");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::context::Context;
    use crate::framework::envelope::{Envelope, ShutdownHint};
    use crate::framework::handler::Flow;

    #[derive(Debug)]
    enum TestMsg {
        Ping,
        Shutdown {
            grace: Option<Duration>,
            reason: Cow<'static, str>,
        },
    }

    impl Envelope for TestMsg {
        fn as_shutdown(&self) -> Option<ShutdownHint<'_>> {
            match self {
                TestMsg::Shutdown { grace, reason } => Some(ShutdownHint::Graceful {
                    grace: *grace,
                    reason,
                }),
                _ => None,
            }
        }
        fn build_shutdown(grace: Option<Duration>, reason: Cow<'static, str>) -> Option<Self> {
            Some(TestMsg::Shutdown { grace, reason })
        }
    }

    use std::sync::atomic::AtomicUsize;

    /// Test actor that stops itself after receiving its first
    /// `TestMsg::Ping`.  Returning [`Flow::Stop`] makes the actor
    /// thread exit naturally; its [`StageExitGuard`] then signals
    /// the supervisor, which is exactly what pre-existing cars
    /// tracking stages do (e.g. `blackhole` upon `SourceEos`).
    ///
    /// We cannot rely on inbox disconnection here: the registry
    /// retains a clone of every actor's `Addr<M>` for peer lookup
    /// and is itself held by every actor's `Context`, so senders
    /// never drop while the pipeline is live.  Natural exits must
    /// therefore be driven by the actor itself.
    struct PingActor {
        received: Arc<AtomicUsize>,
        stop_after: usize,
    }

    impl Actor for PingActor {
        type Msg = TestMsg;
        fn handle(&mut self, msg: TestMsg, _ctx: &mut Context<Self>) -> Result<Flow> {
            match msg {
                TestMsg::Ping => {
                    let prev = self.received.fetch_add(1, Ordering::Relaxed);
                    if prev + 1 >= self.stop_after {
                        Ok(Flow::Stop)
                    } else {
                        Ok(Flow::Cont)
                    }
                }
                TestMsg::Shutdown { .. } => Ok(Flow::Stop),
            }
        }
        fn poll_timeout(&self) -> Duration {
            Duration::from_millis(5)
        }
    }

    /// End-to-end: register one actor, send a Ping to it via its
    /// address; the actor stops after the ping, fires its exit
    /// guard, supervisor broadcasts, join returns Ok.
    #[test]
    fn register_run_broadcast_and_join() {
        let received = Arc::new(AtomicUsize::new(0));
        let received_clone = received.clone();
        let mut sys = System::new()
            .quiescence(Duration::from_millis(0))
            .install_ctrlc_handler(false);
        let builder: ActorBuilder<PingActor> =
            ActorBuilder::new(StageName::unnamed(StageKind::Infer), 4).factory(move |_bx| {
                Ok(PingActor {
                    received: received_clone.clone(),
                    stop_after: 1,
                })
            });
        let addr = sys.register_actor(builder).unwrap();
        addr.send(TestMsg::Ping).unwrap();

        let report = sys.run().unwrap();
        assert_eq!(
            received.load(Ordering::Relaxed),
            1,
            "the ping must have been observed exactly once"
        );
        assert_eq!(report.stage_results.len(), 1);
        assert_eq!(report.stage_results[0].0.kind, StageKind::Infer);
        assert!(report.stage_results[0].1.is_ok());
        match report
            .trigger
            .as_ref()
            .expect("supervisor must have a trigger")
        {
            ShutdownCause::StageExit { stage } => assert_eq!(stage.kind, StageKind::Infer),
            other => panic!("unexpected trigger: {other:?}"),
        }
    }

    /// Custom shutdown handler (closure flavour) is honoured.
    #[test]
    fn custom_closure_handler_runs() {
        let seen: Arc<std::sync::Mutex<Vec<ShutdownCause>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen_clone = seen.clone();
        let mut sys = System::new().install_ctrlc_handler(false).on_shutdown(
            move |cause: ShutdownCause, _ctx: &mut ShutdownCtx<'_>| {
                seen_clone.lock().unwrap().push(cause);
                Ok(ShutdownAction::Broadcast {
                    grace: None,
                    reason: Cow::Borrowed("custom"),
                })
            },
        );
        let builder: ActorBuilder<PingActor> =
            ActorBuilder::new(StageName::unnamed(StageKind::Infer), 4).factory(|_bx| {
                Ok(PingActor {
                    received: Arc::new(AtomicUsize::new(0)),
                    stop_after: 1,
                })
            });
        let addr = sys.register_actor(builder).unwrap();
        addr.send(TestMsg::Ping).unwrap();
        let report = sys.run().unwrap();
        assert!(report.stage_results[0].1.is_ok());
        let seen = seen.lock().unwrap();
        assert!(
            !seen.is_empty(),
            "custom handler must have fired at least once"
        );
    }

    /// Duplicate stage names are rejected at registration time.
    #[test]
    fn duplicate_stage_name_is_error() {
        let mut sys = System::new().install_ctrlc_handler(false);
        let name = StageName::unnamed(StageKind::Infer);
        let b1: ActorBuilder<PingActor> = ActorBuilder::new(name.clone(), 1).factory(|_| {
            Ok(PingActor {
                received: Arc::new(AtomicUsize::new(0)),
                stop_after: 1,
            })
        });
        let b2: ActorBuilder<PingActor> = ActorBuilder::new(name.clone(), 1).factory(|_| {
            Ok(PingActor {
                received: Arc::new(AtomicUsize::new(0)),
                stop_after: 1,
            })
        });
        sys.register_actor(b1).unwrap();
        let err = sys.register_actor(b2).unwrap_err();
        assert!(err.to_string().contains("duplicate"));
    }

    /// Empty system is a no-op.
    #[test]
    fn empty_system_runs_without_error() {
        let sys = System::new().install_ctrlc_handler(false);
        let report = sys.run().unwrap();
        assert!(report.stage_results.is_empty());
        assert!(report.trigger.is_none());
    }

    /// `SystemReport::into_result` folds per-stage errors into an
    /// aggregated `Result<()>`.
    #[test]
    fn system_report_into_result_flags_errors() {
        let report = SystemReport {
            stage_results: vec![
                (StageName::unnamed(StageKind::Infer), Ok(())),
                (StageName::unnamed(StageKind::Tracker), Err(anyhow!("boom"))),
            ],
            trigger: None,
            history: Vec::new(),
        };
        let err = report.into_result().unwrap_err();
        assert!(err.to_string().contains("tracker"));
    }
}
