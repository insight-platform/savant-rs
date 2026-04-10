use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam::channel::Sender;
use parking_lot::Mutex;

use crate::pipeline::error::PipelineError;
use crate::pipeline::runner::PipelineOutput;

pub fn spawn_watchdog(
    thread_name: String,
    operation_timeout: Duration,
    in_flight: Arc<Mutex<HashMap<u64, Instant>>>,
    shutdown: Arc<AtomicBool>,
    failed: Arc<AtomicBool>,
    output_tx: Sender<PipelineOutput>,
) -> Option<JoinHandle<()>> {
    std::thread::Builder::new()
        .name(thread_name)
        .spawn(move || {
            let tick = operation_timeout / 2;
            loop {
                std::thread::sleep(tick);
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
                    let _ = output_tx.send(PipelineOutput::Error(PipelineError::RuntimeError(
                        format!(
                            "watchdog timeout: {} in-flight buffer(s) exceeded {:?}",
                            expired.len(),
                            operation_timeout
                        ),
                    )));
                    return;
                }
            }
        })
        .ok()
}
