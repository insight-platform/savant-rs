//! One-shot release primitive backed by [`parking_lot::Condvar`].
//!
//! Typical use: share an [`std::sync::Arc`] between a producer that calls
//! [`ReleaseSeal::release`] (e.g. on drop) and consumers that block on
//! [`ReleaseSeal::wait`] or poll [`ReleaseSeal::is_released`].

use parking_lot::{Condvar, Mutex};

/// Condvar-gated flag flipped once by [`Self::release`].
///
/// Thread-safe; safe to share via [`std::sync::Arc`].
pub struct ReleaseSeal {
    released: Mutex<bool>,
    condvar: Condvar,
}

impl ReleaseSeal {
    /// Creates a seal in the non-released state.
    pub fn new() -> Self {
        Self {
            released: Mutex::new(false),
            condvar: Condvar::new(),
        }
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
}
