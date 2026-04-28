//! Framework-shared error-classification types.
//!
//! Every error hook in the [`stages`](super::stages) layer
//! returns a value of [`ErrorAction`], letting user code choose
//! between three reactions:
//!
//! * [`ErrorAction::Fatal`] — the stage should tear down. For
//!   actors, the stage aborts its default downstream sink and
//!   surfaces the error so the actor exits through its
//!   [`stopping`](super::actor::Actor::stopping) path; for
//!   sources, the stage returns `Err(_)` from
//!   [`Source::run`](super::actor::Source::run).
//! * [`ErrorAction::LogAndContinue`] — the stage logs the
//!   error at the call site and keeps processing subsequent
//!   outputs.
//! * [`ErrorAction::Swallow`] — the stage drops the error
//!   silently (no extra log, no latch). Use this for known-noisy
//!   warnings re-surfaced as `Error` that do not affect stage
//!   liveness.
//!
//! Templates use [`ErrorAction`] as a single shared classification
//! vocabulary so that an application can install one error hook
//! (or one closure shape) across every stage in its pipeline.

/// Classification returned from every stage error hook.
/// See the module docs for semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ErrorAction {
    /// Latch the first error, abort the stage's default sink,
    /// and surface the failure to the loop driver. Actors exit
    /// through
    /// [`Actor::stopping`](super::actor::Actor::stopping); sources
    /// return `Err(_)` from
    /// [`Source::run`](super::actor::Source::run).
    #[default]
    Fatal,
    /// Log the error at the call site and continue processing
    /// subsequent inputs/outputs. The default sink is **not**
    /// aborted.
    LogAndContinue,
    /// Drop the error silently — no extra log line, no abort.
    /// Use for known-noisy warnings that the framework
    /// re-surfaces as `Error` but that should not affect stage
    /// liveness.
    Swallow,
}

#[cfg(test)]
mod tests {
    use super::ErrorAction;

    #[test]
    fn default_is_fatal() {
        assert_eq!(ErrorAction::default(), ErrorAction::Fatal);
    }

    #[test]
    fn copy_and_eq_are_derived() {
        let a = ErrorAction::LogAndContinue;
        let b = a;
        assert_eq!(a, b);
        assert_ne!(a, ErrorAction::Swallow);
    }
}
