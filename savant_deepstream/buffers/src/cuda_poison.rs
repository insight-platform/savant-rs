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
/// describing the originating `site` plus a free-form `ctx`.  Every
/// subsequent non-zero notification logs at warn level with a
/// `cuda_post_poison` prefix so the cascade is visually distinct from
/// the root cause.
///
/// Returns `rc` unchanged for ergonomic use at call sites
/// (`if note_cuda_rc(...) != 0 { return Err(...); }`).
pub fn note_cuda_rc(site: &str, ctx: impl std::fmt::Display, rc: i32) -> i32 {
    if rc == 0 {
        return 0;
    }
    if POISONED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        log::error!("cuda_poisoned: site={site}, rc={rc}, ctx={ctx}");
    } else {
        log::warn!("cuda_post_poison: site={site}, rc={rc}, ctx={ctx}");
    }
    rc
}

/// Same as [`note_cuda_rc`] but for `u32`-typed CUDA driver-API codes.
pub fn note_cu_rc(site: &str, ctx: impl std::fmt::Display, rc: u32) -> u32 {
    if rc == 0 {
        return 0;
    }
    if POISONED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        log::error!("cuda_poisoned: site={site}, rc={rc}, ctx={ctx}");
    } else {
        log::warn!("cuda_post_poison: site={site}, rc={rc}, ctx={ctx}");
    }
    rc
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
}
