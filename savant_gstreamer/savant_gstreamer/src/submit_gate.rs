//! Single-counter serialisation primitive for PTS-ordered pipeline submission.
//!
//! See [`SubmitGate`] for the rationale and usage.

use parking_lot::Mutex;

/// Serialises access to a monotonic `u64` counter that is paired with a
/// non-trivial critical section (build-buffer + send-on-channel, drain +
/// submit, etc.).
///
/// # Why this exists
///
/// The surrounding pipelines (`NvInfer`, `NvTracker`, their batching
/// operators) each maintain a monotonic counter — an internal PTS or a
/// batch-id — whose value is assigned inside a critical section that also
/// builds a GStreamer buffer and pushes it onto a channel.  The
/// GStreamer feeder enforces [`PtsPolicy::StrictPts`]
/// (monotonically-increasing buffer PTS) and fails the pipeline on the
/// first violation, so the counter assignment must be serialised against
/// the channel push — not merely atomic.
///
/// The naive "[`std::sync::atomic::AtomicU64`] + sibling `Mutex<()>`"
/// pattern works at runtime but forfeits Rust's main concurrency
/// invariant: the atomic is lock-free at the type level yet "in reality
/// always held under the lock", and the sibling `Mutex<()>` protects
/// only a block of code — not data — so the compiler cannot catch a
/// future contributor who advances the counter without the lock.
///
/// [`SubmitGate`] owns the counter and only exposes it through a
/// closure-based API; there is no way to observe or mutate it without
/// holding the gate.
///
/// # API
///
/// [`SubmitGate::submit_with`] acquires the lock, invokes the caller's
/// closure with `&mut u64`, and releases the lock when the closure
/// returns.  The closure is responsible for deciding whether to
/// advance the counter (the usual pattern is "read current, do work,
/// advance on commit, propagate errors").
///
/// # Example
///
/// ```
/// use savant_gstreamer::submit_gate::SubmitGate;
///
/// let gate = SubmitGate::new();
///
/// // First submission: read 0, commit, advance to 1.
/// let assigned = gate.submit_with(|ctr| {
///     let cur = *ctr;
///     *ctr += 1;
///     cur
/// });
/// assert_eq!(assigned, 0);
///
/// // Second submission: read 1, commit, advance to 2.
/// let assigned = gate.submit_with(|ctr| {
///     let cur = *ctr;
///     *ctr += 1;
///     cur
/// });
/// assert_eq!(assigned, 1);
/// ```
#[derive(Debug, Default)]
pub struct SubmitGate {
    counter: Mutex<u64>,
}

impl SubmitGate {
    /// Create a fresh gate with the counter at `0`.
    pub fn new() -> Self {
        Self {
            counter: Mutex::new(0),
        }
    }

    /// Create a gate with the counter initialised to `start`.
    pub fn with_start(start: u64) -> Self {
        Self {
            counter: Mutex::new(start),
        }
    }

    /// Run `f` under the gate's lock, passing in `&mut u64` pointing at
    /// the counter.  The closure can read, mutate, or leave the counter
    /// untouched — the gate makes no policy decision, only enforces
    /// serialisation.
    ///
    /// The lock is held for the entire duration of `f`; callers should
    /// keep the closure body focused on the critical section and avoid
    /// blocking or long-latency work that could be hoisted outside.
    #[inline]
    pub fn submit_with<R>(&self, f: impl FnOnce(&mut u64) -> R) -> R {
        let mut guard = self.counter.lock();
        f(&mut guard)
    }
}

#[cfg(test)]
mod tests {
    use super::SubmitGate;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn new_starts_at_zero() {
        let gate = SubmitGate::new();
        let observed = gate.submit_with(|c| *c);
        assert_eq!(observed, 0);
    }

    #[test]
    fn with_start_sets_initial_value() {
        let gate = SubmitGate::with_start(42);
        let observed = gate.submit_with(|c| *c);
        assert_eq!(observed, 42);
    }

    #[test]
    fn default_equals_new() {
        let gate: SubmitGate = Default::default();
        let observed = gate.submit_with(|c| *c);
        assert_eq!(observed, 0);
    }

    #[test]
    fn closure_can_mutate_counter() {
        let gate = SubmitGate::new();
        gate.submit_with(|c| *c = 10);
        let observed = gate.submit_with(|c| *c);
        assert_eq!(observed, 10);
    }

    #[test]
    fn closure_can_abort_without_advance() {
        let gate = SubmitGate::new();
        // First caller: commits.
        gate.submit_with(|c| *c += 1);
        // Second caller: reads and decides not to commit.
        let observed = gate.submit_with(|c| *c);
        assert_eq!(observed, 1);
        // Third caller: can still commit afterwards.
        gate.submit_with(|c| *c += 1);
        let observed = gate.submit_with(|c| *c);
        assert_eq!(observed, 2);
    }

    #[test]
    fn closure_return_value_is_propagated() {
        let gate = SubmitGate::new();
        let r: Result<u64, &'static str> = gate.submit_with(|c| {
            let cur = *c;
            *c += 1;
            Ok(cur)
        });
        assert_eq!(r, Ok(0));
    }

    /// Concurrent `submit_with` callers always observe monotonically-
    /// increasing counter values with no duplicates and no gaps.
    ///
    /// This is the core ordering guarantee the NvInfer / NvTracker
    /// pipelines rely on.
    #[test]
    fn concurrent_submitters_see_monotonic_counter() {
        const THREADS: usize = 16;
        const PER_THREAD: usize = 1000;

        let gate = Arc::new(SubmitGate::new());
        let mut handles = Vec::with_capacity(THREADS);

        for _ in 0..THREADS {
            let gate = Arc::clone(&gate);
            handles.push(thread::spawn(move || {
                let mut observed = Vec::with_capacity(PER_THREAD);
                for _ in 0..PER_THREAD {
                    observed.push(gate.submit_with(|c| {
                        let cur = *c;
                        *c += 1;
                        cur
                    }));
                }
                observed
            }));
        }

        let mut all: Vec<u64> = handles
            .into_iter()
            .flat_map(|h| h.join().expect("thread panicked"))
            .collect();
        all.sort_unstable();

        assert_eq!(all.len(), THREADS * PER_THREAD);
        for (i, v) in all.iter().enumerate() {
            assert_eq!(*v, i as u64, "counter should be dense and monotonic");
        }

        let final_value = gate.submit_with(|c| *c);
        assert_eq!(final_value, (THREADS * PER_THREAD) as u64);
    }
}
