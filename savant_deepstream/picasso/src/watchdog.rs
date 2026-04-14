use crate::worker::SourceWorker;
use log::{debug, info};
use parking_lot::{Condvar, Mutex};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// Shared signal used to wake the watchdog thread immediately on shutdown.
pub(crate) struct WatchdogSignal {
    mu: Mutex<bool>,
    cv: Condvar,
}

impl Default for WatchdogSignal {
    fn default() -> Self {
        Self::new()
    }
}

impl WatchdogSignal {
    pub(crate) fn new() -> Self {
        Self {
            mu: Mutex::new(false),
            cv: Condvar::new(),
        }
    }

    /// Signal the watchdog to wake up and exit.
    pub(crate) fn notify_shutdown(&self) {
        let mut guard = self.mu.lock();
        *guard = true;
        self.cv.notify_one();
    }

    /// Sleep for at most `duration`, returning `true` if shutdown was signaled.
    fn wait_or_shutdown(&self, duration: Duration) -> bool {
        let mut guard = self.mu.lock();
        if *guard {
            return true;
        }
        self.cv.wait_for(&mut guard, duration);
        *guard
    }
}

/// Periodically scans all sources for idle workers and invokes the eviction
/// callback. Runs in its own thread.
pub(crate) fn spawn_watchdog(
    workers: Arc<Mutex<HashMap<String, SourceWorker>>>,
    scan_interval: Duration,
    signal: Arc<WatchdogSignal>,
) -> std::thread::JoinHandle<()> {
    std::thread::Builder::new()
        .name("picasso-watchdog".to_string())
        .spawn(move || {
            info!("watchdog started (scan interval: {scan_interval:?})");
            loop {
                if signal.wait_or_shutdown(scan_interval) {
                    break;
                }

                let mut to_remove = Vec::new();
                {
                    let workers_guard = workers.lock();
                    for (source_id, worker) in workers_guard.iter() {
                        if !worker.is_alive() {
                            to_remove.push(source_id.clone());
                        }
                    }
                }

                if !to_remove.is_empty() {
                    let mut workers_guard = workers.lock();
                    for source_id in &to_remove {
                        // Re-check under the removal lock: a dead worker may have
                        // been removed and replaced with a new live worker for the
                        // same source_id between the scan and removal phases.
                        let still_dead = workers_guard
                            .get(source_id)
                            .is_some_and(|worker| !worker.is_alive());
                        if still_dead {
                            debug!("watchdog: removing dead worker for source={source_id}");
                            workers_guard.remove(source_id);
                        }
                    }
                }
            }
            info!("watchdog stopped");
        })
        .expect("failed to spawn watchdog thread")
}
