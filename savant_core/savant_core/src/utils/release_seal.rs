//! One-shot release primitive backed by [`parking_lot::Condvar`].
//!
//! Typical use: share an [`std::sync::Arc`] between a producer that calls
//! [`ReleaseSeal::release`] (e.g. on drop) and consumers that block on
//! [`ReleaseSeal::wait`] or poll [`ReleaseSeal::is_released`].

use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::{Condvar, Mutex};

/// Process-local monotonic counter used to mint a unique [`ReleaseSeal::id`]
/// for every seal instance. `Relaxed` ordering is sufficient: the counter is
/// only used to obtain distinct IDs, and the resulting `id` field is published
/// to other threads via `Arc`'s own acquire/release semantics, not via this
/// counter.
static SEAL_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Condvar-gated flag flipped once by [`Self::release`].
///
/// Each instance carries a process-local unique [`Self::id`] used by upstream
/// log lines (e.g. `Sealed<T>::unseal` emits a periodic `warn!` keyed by this
/// id when the producer fails to release in a timely manner).
///
/// Thread-safe; safe to share via [`std::sync::Arc`].
pub struct ReleaseSeal {
    id: u64,
    released: Mutex<bool>,
    condvar: Condvar,
}

impl std::fmt::Debug for ReleaseSeal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReleaseSeal")
            .field("id", &self.id)
            .field("released", &*self.released.lock())
            .finish()
    }
}

impl ReleaseSeal {
    /// Creates a seal in the non-released state.
    ///
    /// A process-local unique [`Self::id`] is assigned and emitted on
    /// `log::trace!` so that downstream `warn!` lines (see
    /// `deepstream_buffers::sealed::Sealed::unseal`) can be correlated back
    /// to the originating producer.
    pub fn new() -> Self {
        let id = SEAL_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        log::trace!("ReleaseSeal[{}] created", id);
        Self {
            id,
            released: Mutex::new(false),
            condvar: Condvar::new(),
        }
    }

    /// Returns the process-local unique identifier of this seal.
    ///
    /// IDs are assigned monotonically by [`Self::new`] from a global atomic
    /// counter; they are stable for the lifetime of the seal and unaffected
    /// by [`Self::release`].
    #[inline]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Marks the seal as released and wakes all waiters.
    pub fn release(&self) {
        *self.released.lock() = true;
        self.condvar.notify_all();
    }

    /// Blocks until [`Self::release`] has been called.
    pub fn wait(&self) {
        let mut released = self.released.lock();
        while !*released {
            self.condvar.wait(&mut released);
        }
    }

    /// Blocks until [`Self::release`] has been called **or** `timeout` elapses.
    ///
    /// Returns `true` if the seal was released, `false` on timeout.
    pub fn wait_timeout(&self, timeout: std::time::Duration) -> bool {
        let mut released = self.released.lock();
        let deadline = std::time::Instant::now() + timeout;
        while !*released {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                return false;
            }
            let result = self.condvar.wait_for(&mut released, remaining);
            if result.timed_out() && !*released {
                return false;
            }
        }
        true
    }

    /// Returns whether [`Self::release`] has been called (non-blocking).
    pub fn is_released(&self) -> bool {
        *self.released.lock()
    }
}

impl Default for ReleaseSeal {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::ReleaseSeal;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn release_unblocks_wait() {
        let seal = Arc::new(ReleaseSeal::new());
        let t = {
            let seal = Arc::clone(&seal);
            thread::spawn(move || {
                thread::sleep(Duration::from_millis(20));
                seal.release();
            })
        };
        seal.wait();
        t.join().unwrap();
    }

    #[test]
    fn is_released_false_until_release() {
        let seal = ReleaseSeal::new();
        assert!(!seal.is_released());
        seal.release();
        assert!(seal.is_released());
    }

    #[test]
    fn wait_timeout_returns_true_when_released() {
        let seal = Arc::new(ReleaseSeal::new());
        let t = {
            let seal = Arc::clone(&seal);
            thread::spawn(move || {
                thread::sleep(Duration::from_millis(20));
                seal.release();
            })
        };
        assert!(seal.wait_timeout(Duration::from_secs(5)));
        t.join().unwrap();
    }

    #[test]
    fn wait_timeout_returns_false_on_expiry() {
        let seal = ReleaseSeal::new();
        assert!(!seal.wait_timeout(Duration::from_millis(30)));
        assert!(!seal.is_released());
    }

    #[test]
    fn id_is_unique_and_monotonic() {
        // Allocate a batch of seals from a single thread and assert that
        // every id is distinct and strictly increasing. This is the
        // single-thread guarantee provided by `fetch_add` on the same
        // atomic, regardless of `Ordering::Relaxed`.
        const N: usize = 64;
        let ids: Vec<u64> = (0..N).map(|_| ReleaseSeal::new().id()).collect();
        for pair in ids.windows(2) {
            assert!(
                pair[1] > pair[0],
                "ids must be strictly increasing: {:?}",
                pair
            );
        }
        let unique: std::collections::HashSet<u64> = ids.iter().copied().collect();
        assert_eq!(unique.len(), N, "ids must all be unique");
    }

    #[test]
    fn id_accessible_after_release() {
        let seal = ReleaseSeal::new();
        let id_before = seal.id();
        seal.release();
        assert_eq!(seal.id(), id_before);
        assert!(seal.is_released());
    }

    #[test]
    fn ids_unique_across_threads() {
        // Concurrent allocation from multiple threads still yields a total
        // ordering on the atomic counter, so every id is unique even though
        // we use `Ordering::Relaxed` (atomicity of the read-modify-write is
        // independent of the chosen memory ordering).
        const THREADS: usize = 8;
        const PER_THREAD: usize = 64;
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                thread::spawn(|| {
                    (0..PER_THREAD)
                        .map(|_| ReleaseSeal::new().id())
                        .collect::<Vec<u64>>()
                })
            })
            .collect();
        let mut all_ids = Vec::with_capacity(THREADS * PER_THREAD);
        for h in handles {
            all_ids.extend(h.join().unwrap());
        }
        let unique: std::collections::HashSet<u64> = all_ids.iter().copied().collect();
        assert_eq!(
            unique.len(),
            all_ids.len(),
            "ids must be globally unique across threads"
        );
    }
}
