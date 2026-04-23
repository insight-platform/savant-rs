//! Framework-shared error-classification types.
//!
//! Every egress error hook across the template layer returns a
//! value of [`ErrorAction`], letting user code choose between
//! three coarse reactions:
//!
//! * [`ErrorAction::Fatal`] — the stage should tear down
//!   (default sink aborted, stage exits through its `stopping`
//!   path for actors, or returns `Err(_)` from
//!   [`Source::run`](super::actor::Source::run) for sources).
//! * [`ErrorAction::LogAndContinue`] — the error is logged at
//!   the call site (templates issue the log line before the hook
//!   runs) and processing resumes with subsequent outputs.
//! * [`ErrorAction::Swallow`] — the error is dropped on the
//!   floor with no additional log; reserved for the rare case
//!   of a known-noisy warning that the framework re-surfaces as
//!   `Error`.
//!
//! This type used to live inside the `mp4_demuxer` template.
//! Promoting it to framework scope makes every egress error
//! hook (decoder, nvinfer, nvtracker, mp4 demuxer) speak the
//! same classification vocabulary — see
//! [`grouped_hook_builders.plan.md`](../../grouped_hook_builders.plan.md).

/// Classification returned from every template egress error
/// hook.  See the module docs for semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ErrorAction {
    /// Latch the first error, abort the default sink, and
    /// surface the failure: actors exit through
    /// [`Actor::stopping`](super::actor::Actor::stopping) with
    /// the error latched in the host's supervisor state;
    /// sources return `Err(_)` from `Source::run`.
    #[default]
    Fatal,
    /// Log the error at the call site and keep processing.
    /// The latched-error slot is still updated so post-mortem
    /// observers can see that *something* went wrong, but the
    /// sink is *not* aborted and subsequent outputs continue
    /// to flow.
    LogAndContinue,
    /// Swallow the error entirely (no extra log, no latch).
    /// Intended for known-noisy warnings that the framework
    /// re-surfaces as `Error` even though they do not affect
    /// stage liveness.
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
