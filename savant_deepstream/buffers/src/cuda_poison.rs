//! Process-wide CUDA poison flag.
//!
//! The first time anyone in the deepstream-buffers crate observes a non-zero
//! CUDA driver/runtime error code we flip an atomic flag and log the offending
//! site at error level.  Subsequent non-zero notifications still get logged,
//! but with a `cuda_post_poison:` prefix at warn level so the original cause
//! visually stands out from the cascade of follow-on failures (every CUDA op
//! against a poisoned context typically fails with the same rc).
//!
//! This is a **diagnostic-only** facility: no public behaviour depends on the
//! flag.  It exists to disambiguate "where did the GPU first go bad?" from
//! "everywhere now reports the same code 700" in production logs.
//!
//! ## Source-location tracking
//!
//! [`note_cuda_rc`] and [`note_cu_rc`] are `#[track_caller]`, so the `at=…`
//! field in the emitted log line is the file/line of the **outermost** caller
//! that itself opted into `#[track_caller]`.  Wrappers like
//! [`crate::cuda_device::cuda_device_synchronize`],
//! [`crate::CudaStream::synchronize`], and
//! [`crate::CudaStream::synchronize_or_log`] are also annotated, so a stream
//! sync that surfaces a deferred fault reports the application call site
//! rather than the wrapper internals.

use std::panic::Location;
use std::sync::atomic::{AtomicBool, Ordering};

static POISONED: AtomicBool = AtomicBool::new(false);

/// Returns `true` if any CUDA operation in this process has reported a
/// non-zero return code through [`note_cuda_rc`] / [`note_cu_rc`] since
/// process start (or since [`reset_poison_for_tests`] was last called).
#[inline]
pub fn is_poisoned() -> bool {
    POISONED.load(Ordering::Acquire)
}

/// Record the result of a CUDA runtime API call (`int` rc).
///
/// `rc == 0` is a no-op.  Non-zero `rc` flips the poison flag (idempotent)
/// and emits exactly one error-level log on the very first transition,
/// describing the originating `site`, the free-form `ctx`, and the
/// caller's source location.  Every subsequent non-zero notification logs
/// at warn level with a `cuda_post_poison` prefix so the cascade is
/// visually distinct from the root cause.
///
/// The `at=file:line` field is captured via
/// [`std::panic::Location::caller`].  Annotate intermediate wrappers with
/// `#[track_caller]` so the reported location is the application call
/// site rather than the wrapper body.
///
/// Returns `rc` unchanged for ergonomic use at call sites
/// (`if note_cuda_rc(...) != 0 { return Err(...); }`).
#[track_caller]
pub fn note_cuda_rc(site: &str, ctx: impl std::fmt::Display, rc: i32) -> i32 {
    if rc == 0 {
        return 0;
    }
    let loc = Location::caller();
    log_poison(site, &ctx, rc as i64, loc);
    rc
}

/// Same as [`note_cuda_rc`] but for `u32`-typed CUDA driver-API codes.
#[track_caller]
pub fn note_cu_rc(site: &str, ctx: impl std::fmt::Display, rc: u32) -> u32 {
    if rc == 0 {
        return 0;
    }
    let loc = Location::caller();
    log_poison(site, &ctx, rc as i64, loc);
    rc
}

/// Common log-emission path shared by [`note_cuda_rc`] and [`note_cu_rc`].
///
/// Splits the formatting from the `#[track_caller]` entry points so a single
/// branch handles the `compare_exchange` + `log::error!`/`log::warn!` choice
/// without forcing the caller-location capture into both functions twice.
fn log_poison(site: &str, ctx: &dyn std::fmt::Display, rc: i64, loc: &Location<'_>) {
    let file = loc.file();
    let line = loc.line();
    if POISONED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        log::error!("cuda_poisoned: site={site}, rc={rc}, ctx={ctx}, at={file}:{line}");
    } else {
        log::warn!("cuda_post_poison: site={site}, rc={rc}, ctx={ctx}, at={file}:{line}");
    }
}

/// Reset the process-wide poison flag.
///
/// **Test-only utility** — present in non-test builds for cross-crate
/// integration tests in `deepstream-encoders` and friends.  Production
/// code must never call this; the flag is meant to latch the first
/// observed CUDA fault for the lifetime of the process.
pub fn reset_poison_for_tests() {
    POISONED.store(false, Ordering::Release);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_rc_is_a_noop() {
        reset_poison_for_tests();
        assert_eq!(note_cuda_rc("test", "ctx", 0), 0);
        assert!(!is_poisoned());
        assert_eq!(note_cu_rc("test", "ctx", 0), 0);
        assert!(!is_poisoned());
    }

    #[test]
    fn first_nonzero_flips_flag_and_logs_error() {
        reset_poison_for_tests();
        assert_eq!(note_cuda_rc("first", "ctx", 700), 700);
        assert!(is_poisoned());
        // Subsequent calls still return rc; flag stays set.
        assert_eq!(note_cuda_rc("second", "ctx", 700), 700);
        assert!(is_poisoned());
    }

    /// `#[track_caller]` propagates through wrapper functions also marked
    /// with the attribute.  The captured `Location` should belong to the
    /// outermost call, not to the wrapper body.
    #[test]
    fn track_caller_propagates_through_wrappers() {
        #[track_caller]
        fn wrapper(rc: i32) -> &'static Location<'static> {
            // Use the same idiom as `note_cuda_rc`: capture the caller
            // location inside the `#[track_caller]` function body.  The
            // `rc` parameter is ignored — we just want the location.
            let _ = rc;
            Location::caller()
        }
        let loc_outer_a = wrapper(0);
        let loc_outer_b = wrapper(0);
        // Two consecutive lines have different line numbers.
        assert_ne!(loc_outer_a.line(), loc_outer_b.line());
        // And neither matches the wrapper's own definition site
        // (i.e. the body of `wrapper`).
        assert!(loc_outer_a.file().ends_with("cuda_poison.rs"));
    }
}
