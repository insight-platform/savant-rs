//! Safe wrappers around device-wide CUDA runtime calls.
//!
//! Currently exposes [`cuda_device_synchronize`], which blocks until **all**
//! CUDA work on the current device has completed.  The return code is routed
//! through [`crate::cuda_poison::note_cuda_rc`] so any non-zero status
//! surfaces with a stable, caller-supplied site label rather than silently
//! poisoning the next caller's stream.

use crate::cuda_poison::note_cuda_rc;

extern "C" {
    fn cudaDeviceSynchronize() -> i32;
}

/// Block until **all** CUDA work on the current device has completed.
///
/// This is the only portable barrier when the offending stream is owned by
/// a third-party component (e.g. the V4L2 NVENC plugin's private CUDA
/// stream).  Prefer per-stream `synchronize` whenever the relevant stream
/// is reachable.
///
/// The return code is recorded via [`note_cuda_rc`] using `site` plus the
/// caller-supplied `ctx`.  The raw rc is returned unchanged for ergonomic
/// branching at call sites.
///
/// # Examples
///
/// ```rust,no_run
/// use deepstream_buffers::cuda_device::cuda_device_synchronize;
///
/// let rc = cuda_device_synchronize(
///     "MyComponent::shutdown::device_sync",
///     format_args!("name={}", "demo"),
/// );
/// assert_eq!(rc, 0);
/// ```
#[track_caller]
pub fn cuda_device_synchronize(site: &str, ctx: impl std::fmt::Display) -> i32 {
    let rc = unsafe { cudaDeviceSynchronize() };
    note_cuda_rc(site, ctx, rc)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Wrapper compiles, links, and (on a host with CUDA available) returns 0.
    /// Without CUDA we still link successfully; the call itself may return a
    /// non-zero rc, which is fine — the test only verifies the function is
    /// reachable and routes through `note_cuda_rc` (verified indirectly: any
    /// non-zero rc would set the poison flag, which we tolerate but don't
    /// assert here to keep the test environment-agnostic).
    #[test]
    fn cuda_device_synchronize_links_and_returns() {
        let _rc = cuda_device_synchronize("test", format_args!("smoke"));
    }
}
