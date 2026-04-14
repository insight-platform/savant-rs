use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam::channel::Sender;
use parking_lot::Mutex;

use crate::pipeline::error::PipelineError;
use crate::pipeline::runner::{send_or_shutdown_on, PipelineOutput};

pub fn spawn_watchdog(
    thread_name: String,
    operation_timeout: Duration,
    in_flight: Arc<Mutex<HashMap<u64, Instant>>>,
    shutdown: Arc<AtomicBool>,
    failed: Arc<AtomicBool>,
    output_tx: Sender<PipelineOutput>,
) -> Option<JoinHandle<()>> {
    const POLL_INTERVAL: Duration = Duration::from_millis(10);

    std::thread::Builder::new()
        .name(thread_name)
        .spawn(move || {
            let check_interval = operation_timeout / 2;
            loop {
                let deadline = Instant::now() + check_interval;
                while Instant::now() < deadline {
                    if shutdown.load(Ordering::Acquire) || failed.load(Ordering::Acquire) {
                        return;
                    }
                    std::thread::sleep(POLL_INTERVAL);
                }
                if shutdown.load(Ordering::Acquire) || failed.load(Ordering::Acquire) {
                    return;
                }

                let now = Instant::now();
                let expired: Vec<u64> = in_flight
                    .lock()
                    .iter()
                    .filter(|(_, submitted)| now.duration_since(**submitted) > operation_timeout)
                    .map(|(&pts, _)| pts)
                    .collect();

                if !expired.is_empty() {
                    {
                        let mut map = in_flight.lock();
                        for pts in &expired {
                            map.remove(pts);
                        }
                    }
                    failed.store(true, Ordering::Release);
                    let _ = send_or_shutdown_on(
                        &output_tx,
                        &shutdown,
                        PipelineOutput::Error(PipelineError::RuntimeError(format!(
                            "watchdog timeout: {} in-flight buffer(s) exceeded {:?}",
                            expired.len(),
                            operation_timeout
                        ))),
                    );
                    return;
                }
            }
        })
        .ok()
}
