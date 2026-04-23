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
use std::time::Duration;

use savant_core::primitives::frame::VideoFrameProxy;
use savant_core::utils::release_seal::ReleaseSeal;

use crate::SharedBuffer;

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
    #[inline]
    pub fn new(payload: T, seal: Arc<ReleaseSeal>) -> Self {
        Self { payload, seal }
    }

    /// Non-blocking check whether the producer has released the seal.
    #[inline]
    pub fn is_released(&self) -> bool {
        self.seal.is_released()
    }

    /// Block until the producer releases the seal, then yield the
    /// payload.
    #[inline]
    pub fn unseal(self) -> T {
        self.seal.wait();
        self.payload
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
