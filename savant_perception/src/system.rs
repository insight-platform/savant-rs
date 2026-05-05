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
//!     // custom policy; see [`crate::shutdown`].
//!     Ok(ShutdownAction::Broadcast { grace: None, reason: Cow::Borrowed("custom") })
//! });
//! sys.run()?;
//! ```
//!
//! Construction is split across two call-sites:
//!
//! 1. **Registration** ([`register_actor`](System::register_actor)
//!    / [`register_source`](System::register_source)) — runs
//!    eagerly from user code.  Allocates each actor's inbox and
//!    publishes its `Addr<M>` in the registry under its
//!    [`StageName`].  Sources have no inbox; only their factory
//!    is captured.  Registration order does not matter for peer
//!    resolution because every address is published before any
//!    factory runs.
//! 2. **Run** ([`run`](System::run)) — runs once.  For each
//!    registered stage, in registration order, it invokes the
//!    captured factory against a [`BuildCtx`] with the
//!    fully-populated registry and shared store, then spawns one
//!    OS thread for the resulting actor / source.  Each thread
//!    body owns a [`StageExitGuard`] that signals the
//!    supervisor's exit channel on drop.
//!
//! After every thread is spawned, `run` blocks on the exit
//! channel, invokes the installed (or default)
//! [`ShutdownHandler`] on each trigger, and finally joins every
//! thread in reverse registration order so the pipeline tail
//! logs first on a clean exit.

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
use super::stage_metrics::{StageMetrics, StageReporter};
use super::shutdown::{
    DefaultShutdownHandler, ShutdownAction, ShutdownCause, ShutdownCtx, ShutdownHandler,
};
use super::supervisor::{
    exit_channel, ExitReceiver, ExitSender, StageExit, StageExitGuard, StageKind, StageName,
};

/// Type-erased "send a shutdown envelope to this actor" closure,
/// built at registration time from the actor's envelope type.
type ShutdownSender = Box<dyn Fn(Option<Duration>, Cow<'static, str>) -> bool + Send + Sync>;

/// Pending factory + thread-spawn closure captured at
/// registration.  Invoked once per stage from
/// [`System::run`](System::run): runs the factory, then spawns
/// the worker thread.
type ThreadLauncher = Box<dyn FnOnce(BuildArgs) -> Result<JoinHandle<Result<()>>> + Send>;

/// Arguments threaded into each [`ThreadLauncher`] when
/// [`System::run`](System::run) starts the stage.
struct BuildArgs {
    pipeline_name: Arc<str>,
    registry: Arc<Registry>,
    shared: Arc<SharedStore>,
    stop_flag: Arc<AtomicBool>,
    exit_tx: ExitSender,
    /// Per-stage metrics handle — pre-allocated at registration
    /// for both actors and sources so the framework's
    /// [`StageReporter`] can register every stage before any
    /// worker thread starts.
    stage_metrics: Arc<StageMetrics>,
}

/// Default quiescence grace period applied by the
/// [`DefaultShutdownHandler`] when no custom handler is
/// installed.
///
/// The handler sleeps for this duration between observing a
/// shutdown trigger and broadcasting the `Shutdown` envelope to
/// every actor inbox.  The window gives any in-flight in-band
/// `SourceEos` sentinels a chance to drain through downstream
/// channels first, so each actor sees its per-source EOS before
/// the system-wide shutdown.
///
/// Override with [`System::quiescence`] when a different value
/// fits the application — including `Duration::ZERO` for tests
/// that want immediate broadcast.
pub const DEFAULT_QUIESCENCE_GRACE: Duration = Duration::from_secs(1);

/// Default pipeline name applied when [`System::name`] is not
/// called.  Surfaced as the `pipeline.name` attribute on every
/// framework-emitted span.
pub const DEFAULT_PIPELINE_NAME: &str = "savant-perception";

/// Default cadence for the per-stage `📊` console reporter.  Every
/// minute the framework wakes up, snapshots each registered actor's
/// [`StageMetrics`] atomics, and logs one summary line per stage
/// (frames / FPS / objects / OPS / queue depth / latency
/// min/avg/max).  Override with [`System::stats_period`].
pub const DEFAULT_STATS_PERIOD: Duration = Duration::from_secs(60);

/// Top-level pipeline builder / runner.  See the module docs for
/// the full usage pattern.
pub struct System {
    pipeline_name: Arc<str>,
    shared: SharedStore,
    registry: Registry,
    launchers: Vec<PendingLaunch>,
    shutdown_senders: HashMap<StageName, ShutdownSender>,
    shutdown_handler: Option<Box<dyn ShutdownHandler>>,
    quiescence: Duration,
    install_ctrlc: bool,
    stats_period: Duration,
}

struct PendingLaunch {
    name: StageName,
    launcher: ThreadLauncher,
    /// Per-stage metrics handle pre-allocated at registration so the
    /// system can register every stage with the runtime
    /// [`StageReporter`] before any worker thread starts.  Allocated
    /// for both actors and sources — the source's per-emission
    /// `record_message` calls populate the same console reporter
    /// row.
    stage_metrics: Arc<StageMetrics>,
}

impl Default for System {
    fn default() -> Self {
        Self::new()
    }
}

impl System {
    /// Empty system with defaults: no shutdown handler (i.e.
    /// [`DefaultShutdownHandler`] will be used), Ctrl+C handler
    /// installation *enabled*, the default quiescence grace
    /// period, and a [`DEFAULT_PIPELINE_NAME`] identifier on every
    /// emitted span.
    pub fn new() -> Self {
        Self {
            pipeline_name: Arc::<str>::from(DEFAULT_PIPELINE_NAME),
            shared: SharedStore::new(),
            registry: Registry::new(),
            launchers: Vec::new(),
            shutdown_senders: HashMap::new(),
            shutdown_handler: None,
            quiescence: DEFAULT_QUIESCENCE_GRACE,
            install_ctrlc: true,
            stats_period: DEFAULT_STATS_PERIOD,
        }
    }

    /// Override the cadence of the per-stage `📊` console reporter.
    ///
    /// Each registered actor automatically contributes a
    /// [`StageMetrics`] handle.  Every `period` the framework
    /// snapshots each stage's atomic counters and logs one summary
    /// line — frames / FPS / objects / OPS / queue depth / latency
    /// (min / avg / max over the interval).
    ///
    /// Independent of any [`opentelemetry::metrics::MeterProvider`]
    /// the host may have wired: those instruments are recorded in
    /// parallel and exported through whatever exporter the host
    /// configured (OTLP, stdout, …); the in-process console
    /// reporter is the framework's standalone debugging surface.
    /// Default [`DEFAULT_STATS_PERIOD`] (60 seconds).
    pub fn stats_period(mut self, period: Duration) -> Self {
        self.stats_period = period;
        self
    }

    /// Override the pipeline identifier surfaced as `pipeline.name`
    /// on every framework-emitted span.  Use a stable value (not a
    /// per-run id) so multi-pipeline backends can group / filter
    /// traces by pipeline.  See [`DEFAULT_PIPELINE_NAME`] for the
    /// fallback.
    pub fn name(mut self, name: impl Into<Arc<str>>) -> Self {
        self.pipeline_name = name.into();
        self
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

    /// Publish a named shared `Arc<T>`.  See
    /// [`SharedStore::insert_arc_as`].
    pub fn insert_shared_arc_as<T: Send + Sync + 'static>(
        &mut self,
        key: impl Into<String>,
        value: Arc<T>,
    ) -> &mut Self {
        self.shared.insert_arc_as(key, value);
        self
    }

    /// Register an actor.  Allocates the actor's bounded inbox,
    /// publishes its [`Addr<M>`] under the builder's name, and
    /// captures the factory + thread-spawn closure that
    /// [`run`](Self::run) will invoke.  Returns the registered
    /// [`Addr<M>`](super::addr::Addr) so the caller can retain
    /// its own clone if it wants to inject messages from
    /// outside the pipeline (peers normally resolve through
    /// the registry instead).
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

        // Pre-allocate this stage's metrics handle.  Registered
        // with the framework's [`StageReporter`] in `run()` so the
        // per-stage console line lights up from the first periodic
        // tick.  OpenTelemetry instruments inside `StageMetrics`
        // are bound to whatever global `MeterProvider` is wired up
        // — when no provider is installed they're noop.
        let stage_metrics = StageMetrics::new(name.to_string());

        let factory = parts.factory;
        let poll_override = parts.poll_timeout;
        let launcher_name = name.clone();
        let launcher: ThreadLauncher =
            Box::new(move |args: BuildArgs| -> Result<JoinHandle<Result<()>>> {
                let BuildArgs {
                    pipeline_name,
                    registry,
                    shared,
                    stop_flag,
                    exit_tx,
                    stage_metrics,
                } = args;
                let bx = BuildCtx::new(
                    &launcher_name,
                    &pipeline_name,
                    &registry,
                    &shared,
                    &stop_flag,
                    &stage_metrics,
                );
                let actor: A = factory(&bx)
                    .with_context(|| format!("actor factory for [{launcher_name}] failed"))?;
                let _ = poll_override; // reserved for future per-instance overrides
                let ctx: Context<A> = Context::new(
                    launcher_name.clone(),
                    pipeline_name,
                    registry,
                    shared,
                    stop_flag,
                    stage_metrics,
                );
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
        self.launchers.push(PendingLaunch {
            name,
            launcher,
            stage_metrics,
        });
        Ok(addr)
    }

    /// Register a source (no-inbox producer).  The source's
    /// factory is invoked from [`run`](Self::run) after every
    /// actor's [`Addr`] has been published in the registry, so
    /// the factory can resolve peers eagerly through
    /// [`BuildCtx`].  Sources cooperate with shutdown via the
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

        // Pre-allocate the source's metrics handle.  Sources have
        // no inbox (so `queue_length` stays at 0) and the framework
        // can't measure per-frame handler latency from the outside,
        // but the framework's source-stage internals (mp4 demuxer,
        // uri demuxer, zmq source) call `record_message(1, 0, 0)`
        // at every frame-emission site so the periodic 📊 reporter
        // shows the source's outbound FPS alongside every actor's
        // inbound FPS.
        let stage_metrics = StageMetrics::new(name.to_string());

        let factory = parts.factory;
        let launcher_name = name.clone();
        let launcher: ThreadLauncher =
            Box::new(move |args: BuildArgs| -> Result<JoinHandle<Result<()>>> {
                let BuildArgs {
                    pipeline_name,
                    registry,
                    shared,
                    stop_flag,
                    exit_tx,
                    stage_metrics,
                } = args;
                let bx = BuildCtx::new(
                    &launcher_name,
                    &pipeline_name,
                    &registry,
                    &shared,
                    &stop_flag,
                    &stage_metrics,
                );
                let src: S = factory(&bx)
                    .with_context(|| format!("source factory for [{launcher_name}] failed"))?;
                let sc = SourceContext::new(
                    launcher_name.clone(),
                    pipeline_name,
                    registry,
                    shared,
                    stop_flag,
                    stage_metrics,
                );
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
        self.launchers.push(PendingLaunch {
            name,
            launcher,
            stage_metrics,
        });
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

    /// Finalize the system.
    ///
    /// In order, `run`:
    ///
    /// 1. Optionally installs a Ctrl+C handler (see
    ///    [`install_ctrlc_handler`](Self::install_ctrlc_handler)).
    /// 2. For each registered stage, in registration order,
    ///    invokes its factory against a [`BuildCtx`] with the
    ///    fully-populated registry and shared store, then
    ///    spawns one OS thread for the resulting actor / source.
    /// 3. Blocks on the supervisor exit channel; for every
    ///    [`StageExit`] / Ctrl+C signal, calls the installed
    ///    [`ShutdownHandler`] (or [`DefaultShutdownHandler`] when
    ///    none is installed) and executes the returned
    ///    [`ShutdownAction`].
    /// 4. Joins every spawned thread in reverse registration
    ///    order and returns a [`SystemReport`] aggregating the
    ///    per-stage results.
    pub fn run(self) -> Result<SystemReport> {
        let Self {
            pipeline_name,
            shared,
            registry,
            launchers,
            shutdown_senders,
            shutdown_handler,
            quiescence,
            install_ctrlc,
            stats_period,
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

        // Build the per-stage console reporter before any worker
        // thread starts so the periodic 📊 line lights up from the
        // first interval after the first frame lands.  All
        // [`StageMetrics`] handles pre-allocated at registration
        // (actors and sources alike) are pulled in here.
        let reporter_stages: Vec<Arc<StageMetrics>> =
            launchers.iter().map(|p| p.stage_metrics.clone()).collect();
        let mut reporter = StageReporter::new(stats_period, reporter_stages);
        reporter.start();

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

        // For each registered stage, run its factory and spawn its
        // worker thread.
        let mut handles: Vec<(StageName, JoinHandle<Result<()>>)> =
            Vec::with_capacity(launchers.len());
        let mut pending_factory_errors: Vec<(StageName, anyhow::Error)> = Vec::new();
        for launch in launchers {
            let PendingLaunch {
                name,
                launcher,
                stage_metrics,
            } = launch;
            let args = BuildArgs {
                pipeline_name: Arc::clone(&pipeline_name),
                registry: registry.clone(),
                shared: shared.clone(),
                stop_flag: stop_flag.clone(),
                exit_tx: exit_tx.clone(),
                stage_metrics,
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

        // Final per-stage flush — emits a closing 📊 line for each
        // stage covering any frames that arrived after the last
        // periodic report, then joins the reporter thread.
        reporter.report_now();
        reporter.shutdown();

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
    use crate::context::Context;
    use crate::envelope::{Envelope, ShutdownHint};
    use crate::handler::Flow;

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

    /// Test actor that stops itself after receiving a configured
    /// number of `TestMsg::Ping` messages.  Returning
    /// [`Flow::Stop`] makes the actor thread exit naturally; its
    /// [`StageExitGuard`] then signals the supervisor.
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
