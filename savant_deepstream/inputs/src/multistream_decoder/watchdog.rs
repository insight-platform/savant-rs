//! Periodic cleanup of finished stream workers and stale detecting entries.

use super::stream_slot::{teardown_stream_entry, StreamEntry};
use crate::multistream_decoder::error::DecoderOutput;
use log::{debug, info};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Wake the watchdog early on shutdown.
pub(crate) struct WatchdogSignal {
    shutdown_requested: std::sync::Mutex<bool>,
    cv: std::sync::Condvar,
}

impl Default for WatchdogSignal {
    fn default() -> Self {
        Self::new()
    }
}

impl WatchdogSignal {
    pub(crate) fn new() -> Self {
        Self {
            shutdown_requested: std::sync::Mutex::new(false),
            cv: std::sync::Condvar::new(),
        }
    }

    pub(crate) fn notify_shutdown(&self) {
        let mut g = self
            .shutdown_requested
            .lock()
            .expect("watchdog shutdown mutex poisoned");
        *g = true;
        self.cv.notify_one();
    }

    fn wait_or_shutdown(&self, duration: Duration) -> bool {
        let mut g = self
            .shutdown_requested
            .lock()
            .expect("watchdog shutdown mutex poisoned");
        loop {
            if *g {
                return true;
            }
            let (g2, wait) = self.cv.wait_timeout(g, duration).expect("watchdog condvar");
            g = g2;
            if *g {
                return true;
            }
            if wait.timed_out() {
                return false;
            }
        }
    }
}

/// Spawn a thread that removes dead workers and expired detecting streams.
pub(crate) fn spawn_watchdog(
    streams: Arc<Mutex<HashMap<String, StreamEntry>>>,
    idle_timeout: Duration,
    scan_interval: Duration,
    signal: Arc<WatchdogSignal>,
    shutdown_flag: Arc<AtomicBool>,
    on_output: Arc<dyn Fn(DecoderOutput) + Send + Sync + 'static>,
) -> JoinHandle<()> {
    std::thread::Builder::new()
        .name("multistream-decoder-watchdog".to_string())
        .spawn(move || {
            info!("multistream-decoder watchdog started ({scan_interval:?})");
            loop {
                if shutdown_flag.load(Ordering::Acquire) {
                    break;
                }
                if signal.wait_or_shutdown(scan_interval) {
                    break;
                }
                if shutdown_flag.load(Ordering::Acquire) {
                    break;
                }

                let now = Instant::now();
                let mut to_remove: Vec<String> = Vec::new();
                {
                    let guard = streams.lock();
                    for (sid, entry) in guard.iter() {
                        match entry {
                            StreamEntry::Active(a) if !a.alive.load(Ordering::Acquire) => {
                                to_remove.push(sid.clone());
                            }
                            StreamEntry::Detecting(d) => {
                                if now.duration_since(d.last_seen) >= idle_timeout {
                                    to_remove.push(sid.clone());
                                }
                            }
                            _ => {}
                        }
                    }
                }
                if !to_remove.is_empty() {
                    let mut removed: Vec<(String, StreamEntry)> = Vec::new();
                    {
                        let mut guard = streams.lock();
                        for sid in to_remove {
                            // Re-check under the lock: a new session for the same
                            // source_id may have been inserted between the scan and
                            // here (e.g. the EOS callback already removed the stale
                            // Active entry and the submit path created a fresh one).
                            // Only remove if the entry still looks dead.
                            let should_remove = match guard.get(&sid) {
                                Some(StreamEntry::Active(a)) => {
                                    !a.alive.load(std::sync::atomic::Ordering::Acquire)
                                }
                                Some(StreamEntry::Detecting(d)) => {
                                    now.duration_since(d.last_seen) >= idle_timeout
                                }
                                _ => false,
                            };
                            if should_remove {
                                debug!("watchdog: removing entry source_id={sid}");
                                if let Some(entry) = guard.remove(&sid) {
                                    removed.push((sid, entry));
                                }
                            }
                        }
                    }
                    for (sid, entry) in removed {
                        match entry {
                            StreamEntry::Detecting(d) => {
                                for (f, data) in d.pending {
                                    (on_output)(DecoderOutput::Undecoded {
                                        frame: f,
                                        data: Some(data),
                                        reason: crate::multistream_decoder::error::UndecodedReason::StreamEvicted,
                                    });
                                }
                                (on_output)(DecoderOutput::StreamStopped {
                                    source_id: sid.clone(),
                                    reason: crate::multistream_decoder::error::StopReason::IdleEviction,
                                });
                            }
                            StreamEntry::Active(_) => {
                                teardown_stream_entry(entry);
                            }
                            StreamEntry::Failed { message: _ } => {}
                        }
                    }
                }
            }
            info!("multistream-decoder watchdog stopped");
        })
        .expect("spawn watchdog")
}
