//! Generic "payload sealed until producer releases" wrapper.
//!
//! A [`Sealed<T>`] pairs an opaque payload with an [`Arc<ReleaseSeal>`]
//! that gates consumption.  The producer (typically owning the parent
//! struct whose [`Drop`] calls [`ReleaseSeal::release`]) holds the other
//! end of the `Arc`.  Consumers can:
//!
//! * Poll [`is_released`](Sealed::is_released) (non-blocking).
//! * Call [`unseal`](Sealed::unseal) (blocking until released).
//! * Call [`unseal_timeout`](Sealed::unseal_timeout) (blocking with a
//!   deadline; returns `Err(self)` on timeout so the caller can retry).
//! * Call [`try_unseal`](Sealed::try_unseal) (non-blocking; returns
//!   `Err(self)` if not yet released).
//!
//! # Drop safety
//!
//! Dropping a `Sealed<T>` without calling `unseal()` is safe — the
//! payload is freed and the producer's `Condvar::notify_all` in its
//! [`Drop`] runs against zero waiters (a no-op).
//!
//! # Domain aliases
//!
//! This module ships two ready-made aliases used across the DeepStream
//! pipeline crates:
//!
//! * [`SealedDelivery`] — a single `(VideoFrameProxy, SharedBuffer)`
//!   pair (used by single-stream decoders like `FlexibleDecoder`).
//! * [`SealedDeliveries`] — a batch `Vec<(VideoFrameProxy, SharedBuffer)>`
//!   (used by batched operators such as `NvInferBatchingOperator` and
//!   `NvTrackerBatchingOperator`).
//!
//! Both aliases are the **single source of truth** for the
//! buffer-seal pattern across the workspace — the individual crates
//! re-export them so user-facing APIs remain unchanged.
//!
//! # Example
//!
//! ```
//! use std::sync::Arc;
//! use deepstream_buffers::sealed::Sealed;
//! use savant_core::utils::release_seal::ReleaseSeal;
//!
//! let seal = Arc::new(ReleaseSeal::new());
//! let sealed = Sealed::new(42u32, Arc::clone(&seal));
//! assert!(!sealed.is_released());
//! assert!(sealed.try_unseal().is_err());
//!
//! seal.release();
//! let sealed2 = Sealed::new("ready", seal);
//! assert_eq!(sealed2.unseal(), "ready");
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};

use savant_core::primitives::frame::VideoFrameProxy;
use savant_core::utils::release_seal::ReleaseSeal;

use crate::SharedBuffer;

/// Slice length used by [`Sealed::unseal`] when polling a not-yet-released
/// seal. Each missed slice emits a single `log::warn!` carrying the seal id
/// and payload type so that operators stuck on a never-released producer can
/// be diagnosed from logs alone.
const UNSEAL_WARN_INTERVAL: Duration = Duration::from_secs(5);

/// A payload `T` gated by an [`Arc<ReleaseSeal>`].
///
/// The seal is *not* owned by the `Sealed<T>` — it is shared with the
/// producer (whose `Drop` releases it).  Consumers wait on the seal
/// before seeing the payload.
///
/// # Thread safety
///
/// `Sealed<T>` does not implement [`Send`]/[`Sync`] manually — both are
/// derived automatically from the fields.  Because [`ReleaseSeal`] is
/// `Send + Sync`, `Sealed<T>: Send` iff `T: Send` and `Sealed<T>: Sync`
/// iff `T: Sync`.  Payloads that contain raw pointers (e.g. FFI handles)
/// must provide their own targeted `unsafe impl Send` / `unsafe impl
/// Sync` on the payload type; do **not** add a blanket
/// `unsafe impl<T> Send for Sealed<T>` here — that would override the
/// compiler's auto-trait analysis and make it possible to send
/// `Sealed<Rc<_>>` (and other `!Send` payloads) across threads, causing
/// undefined behaviour.
///
/// The following must **not** compile (regression guard):
///
/// ```compile_fail
/// use std::rc::Rc;
/// use std::sync::Arc;
/// use deepstream_buffers::sealed::Sealed;
/// use savant_core::utils::release_seal::ReleaseSeal;
///
/// fn assert_send<T: Send>() {}
/// assert_send::<Sealed<Rc<u32>>>();
/// ```
///
/// See the [module docs](self) for the full protocol.
pub struct Sealed<T> {
    payload: T,
    seal: Arc<ReleaseSeal>,
}

impl<T> Sealed<T> {
    /// Wrap `payload` with the provided seal.  Callers typically clone
    /// the `Arc` held by the producing struct.
    ///
    /// Emits a `log::trace!` carrying the seal id and the payload type so
    /// that subsequent `warn!` lines from [`Self::unseal`] can be correlated
    /// back to the producer that originally minted the seal.
    #[inline]
    pub fn new(payload: T, seal: Arc<ReleaseSeal>) -> Self {
        log::trace!(
            "Sealed[{}]<{}> created",
            seal.id(),
            std::any::type_name::<T>()
        );
        Self { payload, seal }
    }

    /// Non-blocking check whether the producer has released the seal.
    #[inline]
    pub fn is_released(&self) -> bool {
        self.seal.is_released()
    }

    /// Block until the producer releases the seal, then yield the
    /// payload.
    ///
    /// The wait is internally chunked into [`UNSEAL_WARN_INTERVAL`]
    /// (5-second) slices: every slice that elapses without the producer
    /// releasing the seal triggers a single `log::warn!` line tagged with
    /// the seal id and payload type. The method only returns once the
    /// producer has actually released the seal.
    #[inline]
    pub fn unseal(self) -> T {
        let start = Instant::now();
        let mut iterations: u64 = 0;
        loop {
            if self.seal.wait_timeout(UNSEAL_WARN_INTERVAL) {
                return self.payload;
            }
            iterations += 1;
            log::warn!(
                "Sealed[{}]<{}> unseal still blocked after {:.1}s ({} x {}s slices), \
                 producer has not released yet",
                self.seal.id(),
                std::any::type_name::<T>(),
                start.elapsed().as_secs_f64(),
                iterations,
                UNSEAL_WARN_INTERVAL.as_secs(),
            );
        }
    }

    /// Block until released or `timeout` expires.  On timeout, returns
    /// `Err(self)` so the caller may retry or drop.
    #[inline]
    pub fn unseal_timeout(self, timeout: Duration) -> Result<T, Self> {
        if self.seal.wait_timeout(timeout) {
            Ok(self.payload)
        } else {
            Err(self)
        }
    }

    /// Non-blocking unseal.  Returns `Err(self)` if the seal has not
    /// yet been released.
    #[inline]
    pub fn try_unseal(self) -> Result<T, Self> {
        if self.seal.is_released() {
            Ok(self.payload)
        } else {
            Err(self)
        }
    }
}

impl<T> std::fmt::Debug for Sealed<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sealed")
            .field("seal_id", &self.seal.id())
            .field("payload_type", &std::any::type_name::<T>())
            .field("released", &self.seal.is_released())
            .finish()
    }
}

/// A single `(VideoFrameProxy, SharedBuffer)` pair sealed until the
/// associated producer is dropped.
///
/// Used by single-stream decoders (e.g. `FlexibleDecoder`).
pub type SealedDelivery = Sealed<(VideoFrameProxy, SharedBuffer)>;

/// A batch of `(VideoFrameProxy, SharedBuffer)` pairs sealed until the
/// associated producer is dropped.
///
/// Used by batched operators (e.g. `NvInferBatchingOperator`,
/// `NvTrackerBatchingOperator`).
pub type SealedDeliveries = Sealed<Vec<(VideoFrameProxy, SharedBuffer)>>;

impl SealedDeliveries {
    /// Number of frames in the sealed batch.
    #[inline]
    pub fn len(&self) -> usize {
        self.payload.len()
    }

    /// Whether the batch is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.payload.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn unseal_blocks_until_release() {
        let seal = Arc::new(ReleaseSeal::new());
        let sealed = Sealed::new(42u32, Arc::clone(&seal));

        let t = thread::spawn(move || sealed.unseal());
        thread::sleep(Duration::from_millis(10));
        seal.release();
        assert_eq!(t.join().unwrap(), 42);
    }

    #[test]
    fn try_unseal_err_when_not_released() {
        let seal = Arc::new(ReleaseSeal::new());
        let sealed = Sealed::new("x", seal);
        assert!(sealed.try_unseal().is_err());
    }

    #[test]
    fn try_unseal_ok_after_release() {
        let seal = Arc::new(ReleaseSeal::new());
        seal.release();
        let sealed = Sealed::new(vec![1, 2, 3], seal);
        assert_eq!(sealed.try_unseal().ok().unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn unseal_timeout_returns_payload_when_released() {
        let seal = Arc::new(ReleaseSeal::new());
        let sealed = Sealed::new(7u8, Arc::clone(&seal));

        let t = thread::spawn({
            let seal = Arc::clone(&seal);
            move || {
                thread::sleep(Duration::from_millis(10));
                seal.release();
            }
        });
        let got = sealed.unseal_timeout(Duration::from_secs(5)).unwrap();
        assert_eq!(got, 7);
        t.join().unwrap();
    }

    #[test]
    fn unseal_timeout_returns_err_on_expiry() {
        let seal = Arc::new(ReleaseSeal::new());
        let sealed = Sealed::new(0u8, seal);
        let back = sealed
            .unseal_timeout(Duration::from_millis(10))
            .expect_err("must time out");
        assert!(!back.is_released());
    }

    #[test]
    fn is_released_reflects_seal_state() {
        let seal = Arc::new(ReleaseSeal::new());
        let sealed = Sealed::new((), Arc::clone(&seal));
        assert!(!sealed.is_released());
        seal.release();
        assert!(sealed.is_released());
    }

    #[test]
    fn unseal_warns_periodically_then_returns() {
        // Verify the iterative unseal loop both (a) keeps blocking past one
        // 5-second slice and (b) ultimately returns the payload once the
        // producer releases. We don't assert the warn line text here (that
        // would require an env_logger capture); the value of this test is
        // the regression guard that the loop exits cleanly after at least
        // one warn iteration has elapsed.
        let seal = Arc::new(ReleaseSeal::new());
        let sealed = Sealed::new(123u64, Arc::clone(&seal));

        let releaser = thread::spawn({
            let seal = Arc::clone(&seal);
            move || {
                thread::sleep(Duration::from_millis(5_500));
                seal.release();
            }
        });

        let started = std::time::Instant::now();
        let value = sealed.unseal();
        let elapsed = started.elapsed();

        assert_eq!(value, 123u64);
        assert!(
            elapsed >= Duration::from_secs(5),
            "unseal must have looped past at least one 5s slice (elapsed={:?})",
            elapsed
        );
        releaser.join().unwrap();
    }

    #[test]
    fn unseal_returns_immediately_when_already_released() {
        let seal = Arc::new(ReleaseSeal::new());
        seal.release();
        let sealed = Sealed::new("ready", Arc::clone(&seal));
        let started = std::time::Instant::now();
        let value = sealed.unseal();
        let elapsed = started.elapsed();
        assert_eq!(value, "ready");
        assert!(
            elapsed < Duration::from_secs(1),
            "unseal must short-circuit when the seal is already released (elapsed={:?})",
            elapsed
        );
    }

    #[test]
    fn debug_includes_seal_id() {
        let seal = Arc::new(ReleaseSeal::new());
        let id = seal.id();
        let sealed = Sealed::new(0u8, seal);
        let s = format!("{:?}", sealed);
        assert!(
            s.contains(&format!("seal_id: {}", id)),
            "Debug output must include seal_id field, got: {}",
            s
        );
        assert!(
            s.contains("payload_type"),
            "Debug output must include payload_type field, got: {}",
            s
        );
    }

    /// Compile-time positive check: `Sealed<T>` must be `Send`/`Sync`
    /// when `T` is, and the domain aliases must be `Send` too.
    #[test]
    fn send_sync_positive_bounds() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<Sealed<u32>>();
        assert_sync::<Sealed<u32>>();
        assert_send::<SealedDelivery>();
        assert_send::<SealedDeliveries>();
    }
}
