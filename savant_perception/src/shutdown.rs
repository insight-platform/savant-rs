//! Custom shutdown handling — the user-installable policy that
//! decides *what happens* when the supervisor observes a shutdown
//! trigger.
//!
//! The [`System`](super::system::System) owns at most one
//! [`ShutdownHandler`]. When it observes a trigger
//! ([`ShutdownCause::StageExit`] or [`ShutdownCause::CtrlC`]) it
//! calls [`ShutdownHandler::on_shutdown`] with the cause and an
//! introspection [`ShutdownCtx`]; the handler returns a
//! [`ShutdownAction`] that the supervisor then executes —
//! broadcasting shutdown envelopes, walking through ordered
//! waves, or ignoring the trigger and waiting for the next one.
//!
//! # Default policy
//!
//! When no handler is installed, the supervisor uses a
//! [`DefaultShutdownHandler`] that sleeps for a configurable
//! *quiescence grace period* — letting any in-band
//! [`EncodedMsg::SourceEos`](super::envelopes::EncodedMsg::SourceEos)
//! /
//! [`PipelineMsg::SourceEos`](super::envelopes::PipelineMsg::SourceEos)
//! sentinels already in flight drain through downstream
//! channels — and then broadcasts a `Shutdown { grace: None }`
//! to every registered actor.
//!
//! # Example: press-Ctrl+C-twice-to-quit
//!
//! ```no_run
//! use std::borrow::Cow;
//! use std::time::Duration;
//! use savant_perception::shutdown::*;
//!
//! struct PressTwice { ctrlc_count: u32 }
//!
//! impl ShutdownHandler for PressTwice {
//!     fn on_shutdown(
//!         &mut self,
//!         cause: ShutdownCause,
//!         _ctx: &mut ShutdownCtx<'_>,
//!     ) -> anyhow::Result<ShutdownAction> {
//!         match cause {
//!             ShutdownCause::CtrlC => {
//!                 self.ctrlc_count += 1;
//!                 if self.ctrlc_count == 1 {
//!                     log::warn!("Ctrl+C received; press again to force stop");
//!                     Ok(ShutdownAction::Ignore)
//!                 } else {
//!                     Ok(ShutdownAction::Broadcast {
//!                         grace: None,
//!                         reason: Cow::Borrowed("double ctrl-c"),
//!                     })
//!                 }
//!             }
//!             ShutdownCause::StageExit { stage: _ } => {
//!                 std::thread::sleep(Duration::from_secs(2));
//!                 Ok(ShutdownAction::Broadcast {
//!                     grace: None,
//!                     reason: Cow::Borrowed("stage exited"),
//!                 })
//!             }
//!         }
//!     }
//! }
//! ```

use std::borrow::Cow;
use std::time::Duration;

use super::supervisor::StageName;

/// What the supervisor observed that prompted it to invoke the
/// [`ShutdownHandler`].
#[derive(Debug, Clone)]
pub enum ShutdownCause {
    /// A stage thread fired its
    /// [`StageExitGuard`](super::supervisor::StageExitGuard) —
    /// either via natural exit, an `Err` propagation, or a panic
    /// unwind.
    StageExit {
        /// The stage that signalled exit.  Pipeline policies may
        /// react differently depending on which stage exited
        /// first (e.g. "demux finished naturally" vs. "tracker
        /// panicked").
        stage: StageName,
    },
    /// The process received `SIGINT` / Ctrl+C. Routed through the
    /// same back-channel as stage exits so the handler sees one
    /// uniform stream of shutdown triggers.
    CtrlC,
}

/// Action the supervisor should execute on return from the
/// [`ShutdownHandler`].
#[derive(Debug, Clone)]
pub enum ShutdownAction {
    /// Do nothing; continue waiting for the next trigger.
    /// Useful for stateful handlers (e.g. "ignore the first
    /// Ctrl+C, broadcast on the second").
    Ignore,
    /// Broadcast a `Shutdown { grace, reason }` envelope to every
    /// registered actor, then (unconditionally) flip the shared
    /// stop flag for sources.
    Broadcast {
        /// Grace period (see
        /// [`ShutdownHint::Graceful`](super::envelope::ShutdownHint::Graceful)).
        grace: Option<Duration>,
        /// Human-readable reason — appears in every stage's
        /// shutdown log record.
        reason: Cow<'static, str>,
    },
    /// Shut actors down in ordered waves — useful when sink-side
    /// stages must drain before upstream stages are stopped
    /// (e.g. flush the muxer's `moov` atom first).  Each wave
    /// sends shutdown envelopes to its `targets`, sleeps for
    /// `pause_after`, and then moves on.  After the last wave the
    /// supervisor also flips the shared stop flag.
    Ordered {
        /// Ordered list of shutdown waves.
        waves: Vec<ShutdownWave>,
    },
}

/// One wave of an ordered shutdown plan — see
/// [`ShutdownAction::Ordered`].
#[derive(Debug, Clone)]
pub struct ShutdownWave {
    /// Actors to shut down in this wave.  Empty means "every
    /// registered actor" — useful as the final wave of a
    /// whitelist-style plan.
    pub targets: Vec<StageName>,
    /// Grace period propagated into the shutdown envelope.
    pub grace: Option<Duration>,
    /// Log reason propagated into the shutdown envelope.
    pub reason: Cow<'static, str>,
    /// Delay after sending this wave's shutdowns before moving
    /// on to the next wave (or finishing, if this is the last
    /// wave).
    pub pause_after: Duration,
}

/// Introspection context handed to the
/// [`ShutdownHandler::on_shutdown`] call.
///
/// Exposes read-only references to the known set of registered
/// stages and the full history of prior [`ShutdownCause`]s the
/// supervisor has processed so far (useful for stateful
/// policies).
pub struct ShutdownCtx<'a> {
    pub(crate) stages: &'a [StageName],
    pub(crate) history: &'a [ShutdownCause],
}

impl<'a> ShutdownCtx<'a> {
    /// Every actor/source registered with the
    /// [`System`](super::system::System) at registration time.
    pub fn stages(&self) -> &[StageName] {
        self.stages
    }

    /// Ordered history of prior causes the supervisor has
    /// already handled (empty on the first invocation).
    pub fn history(&self) -> &[ShutdownCause] {
        self.history
    }

    /// Number of Ctrl+C signals seen so far (including the
    /// current one, if `cause == CtrlC`).
    pub fn ctrlc_count(&self) -> usize {
        self.history
            .iter()
            .filter(|c| matches!(c, ShutdownCause::CtrlC))
            .count()
    }

    /// Whether any stage has already exited prior to the current
    /// cause.
    pub fn any_stage_exited(&self) -> bool {
        self.history
            .iter()
            .any(|c| matches!(c, ShutdownCause::StageExit { .. }))
    }

    /// Create a synthetic context for unit-testing custom
    /// handlers.  Not intended for production use.
    pub fn fake(stages: &'a [StageName], history: &'a [ShutdownCause]) -> Self {
        Self { stages, history }
    }
}

/// User-installable shutdown policy.  See the module docs for an
/// example; see [`DefaultShutdownHandler`] for the policy applied
/// when no user handler is installed.
pub trait ShutdownHandler: Send {
    /// Decide what to do in response to a shutdown trigger.
    ///
    /// The supervisor calls this method on its own thread (the
    /// thread that called [`System::run`](super::system::System));
    /// handlers may therefore block freely — the only running
    /// work is the spawned stage threads, which continue
    /// processing their inbox queues.
    ///
    /// Returning `Err(_)` is treated as an abort: the supervisor
    /// logs the error, broadcasts an immediate
    /// `Shutdown { grace: None }` with reason
    /// `"shutdown handler error"`, and joins every thread.
    fn on_shutdown(
        &mut self,
        cause: ShutdownCause,
        ctx: &mut ShutdownCtx<'_>,
    ) -> anyhow::Result<ShutdownAction>;
}

/// Default shutdown policy — sleep for a configurable quiescence
/// grace period (so natural in-band `SourceEos` sentinels reach
/// every stage before the supervisor's out-of-band `Shutdown`
/// overtakes them) and then broadcast to every actor.
///
/// Install a custom [`ShutdownHandler`] via
/// [`System::on_shutdown`](super::system::System::on_shutdown)
/// to change this policy.
pub struct DefaultShutdownHandler {
    quiescence: Duration,
}

impl DefaultShutdownHandler {
    /// Construct the default handler with `quiescence` grace
    /// period.  Use [`Duration::ZERO`] for tests or pipelines
    /// that want immediate broadcast; use a few seconds for
    /// pipelines whose downstream stages need time to drain
    /// in-flight `SourceEos` before being torn down.
    pub fn new(quiescence: Duration) -> Self {
        Self { quiescence }
    }

    /// Grace period the handler sleeps for before returning
    /// [`ShutdownAction::Broadcast`].
    pub fn quiescence(&self) -> Duration {
        self.quiescence
    }
}

impl ShutdownHandler for DefaultShutdownHandler {
    fn on_shutdown(
        &mut self,
        cause: ShutdownCause,
        _ctx: &mut ShutdownCtx<'_>,
    ) -> anyhow::Result<ShutdownAction> {
        log::info!(
            "[supervisor] shutdown trigger: {cause:?}; sleeping {:?} before broadcasting Shutdown",
            self.quiescence
        );
        if !self.quiescence.is_zero() {
            std::thread::sleep(self.quiescence);
        }
        Ok(ShutdownAction::Broadcast {
            grace: None,
            reason: Cow::Borrowed("supervisor"),
        })
    }
}

/// Blanket impl so any `FnMut(ShutdownCause, &mut ShutdownCtx<'_>) -> Result<ShutdownAction>`
/// can be installed as a [`ShutdownHandler`] directly, without
/// having to author a struct.
impl<F> ShutdownHandler for F
where
    F: FnMut(ShutdownCause, &mut ShutdownCtx<'_>) -> anyhow::Result<ShutdownAction> + Send,
{
    fn on_shutdown(
        &mut self,
        cause: ShutdownCause,
        ctx: &mut ShutdownCtx<'_>,
    ) -> anyhow::Result<ShutdownAction> {
        (self)(cause, ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::supervisor::StageKind;

    #[test]
    fn default_handler_returns_broadcast() {
        let mut h = DefaultShutdownHandler::new(Duration::from_millis(0));
        let stages: [StageName; 0] = [];
        let history: [ShutdownCause; 0] = [];
        let mut ctx = ShutdownCtx::fake(&stages, &history);
        let action = h.on_shutdown(ShutdownCause::CtrlC, &mut ctx).unwrap();
        match action {
            ShutdownAction::Broadcast { grace, reason } => {
                assert_eq!(grace, None);
                assert_eq!(reason, "supervisor");
            }
            other => panic!("expected Broadcast, got {other:?}"),
        }
    }

    #[test]
    fn closure_is_a_shutdown_handler() {
        let mut seen = Vec::<ShutdownCause>::new();
        let mut h = |cause: ShutdownCause, _ctx: &mut ShutdownCtx<'_>| {
            seen.push(cause.clone());
            Ok(ShutdownAction::Ignore)
        };
        let stages: [StageName; 0] = [];
        let history: [ShutdownCause; 0] = [];
        let mut ctx = ShutdownCtx::fake(&stages, &history);
        h.on_shutdown(ShutdownCause::CtrlC, &mut ctx).unwrap();
        assert_eq!(seen.len(), 1);
    }

    #[test]
    fn ctx_ctrlc_count_reflects_history() {
        let stages: [StageName; 1] = [StageName::unnamed(StageKind::Infer)];
        let history = [
            ShutdownCause::StageExit {
                stage: StageName::unnamed(StageKind::BitstreamSource),
            },
            ShutdownCause::CtrlC,
            ShutdownCause::CtrlC,
        ];
        let ctx = ShutdownCtx::fake(&stages, &history);
        assert_eq!(ctx.stages().len(), 1);
        assert_eq!(ctx.history().len(), 3);
        assert_eq!(ctx.ctrlc_count(), 2);
        assert!(ctx.any_stage_exited());
    }

    #[test]
    fn stateful_handler_ignores_first_ctrlc_then_broadcasts() {
        struct H {
            seen: u32,
        }
        impl ShutdownHandler for H {
            fn on_shutdown(
                &mut self,
                cause: ShutdownCause,
                _ctx: &mut ShutdownCtx<'_>,
            ) -> anyhow::Result<ShutdownAction> {
                match cause {
                    ShutdownCause::CtrlC => {
                        self.seen += 1;
                        if self.seen == 1 {
                            Ok(ShutdownAction::Ignore)
                        } else {
                            Ok(ShutdownAction::Broadcast {
                                grace: None,
                                reason: Cow::Borrowed("twice"),
                            })
                        }
                    }
                    ShutdownCause::StageExit { .. } => Ok(ShutdownAction::Broadcast {
                        grace: None,
                        reason: Cow::Borrowed("exit"),
                    }),
                }
            }
        }
        let stages: [StageName; 0] = [];
        let history: [ShutdownCause; 0] = [];
        let mut ctx = ShutdownCtx::fake(&stages, &history);
        let mut h = H { seen: 0 };
        assert!(matches!(
            h.on_shutdown(ShutdownCause::CtrlC, &mut ctx).unwrap(),
            ShutdownAction::Ignore
        ));
        assert!(matches!(
            h.on_shutdown(ShutdownCause::CtrlC, &mut ctx).unwrap(),
            ShutdownAction::Broadcast { .. }
        ));
    }
}
