use crate::callbacks::Callbacks;
use crate::error::PicassoError;
use crate::message::WorkerMessage;
use crate::spec::{GeneralSpec, SourceSpec};
use crate::watchdog::{self, WatchdogSignal};
use crate::worker::SourceWorker;
use log::{debug, info};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// The main entry point for the Picasso frame-processing pipeline.
///
/// Manages per-source worker threads, a watchdog for idle-source eviction,
/// and dispatches frames to the appropriate worker.
pub struct PicassoEngine {
    workers: Arc<Mutex<HashMap<String, SourceWorker>>>,
    callbacks: Arc<Callbacks>,
    default_spec: GeneralSpec,
    shutdown_flag: Arc<AtomicBool>,
    watchdog_signal: Arc<WatchdogSignal>,
    watchdog: Option<std::thread::JoinHandle<()>>,
}

const DEFAULT_PICASSO_NAME: &str = "picasso";

impl PicassoEngine {
    /// Create a new engine with the given global defaults and callbacks.
    ///
    /// Spawns the watchdog thread immediately.
    pub fn new(general: GeneralSpec, callbacks: Callbacks) -> Self {
        let name_display = if general.name.is_empty() {
            DEFAULT_PICASSO_NAME.to_string()
        } else {
            general.name.clone()
        };
        info!("PicassoEngine initializing (name={})", name_display);

        let callbacks = Arc::new(callbacks);
        let workers: Arc<Mutex<HashMap<String, SourceWorker>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let watchdog_signal = Arc::new(WatchdogSignal::new());

        let scan_interval = Duration::from_secs(general.idle_timeout_secs.max(1) / 2);

        let watchdog =
            watchdog::spawn_watchdog(workers.clone(), scan_interval, watchdog_signal.clone());

        info!(
            "PicassoEngine initialized (name={}, idle_timeout={}s, queue_size={})",
            name_display, general.idle_timeout_secs, general.inflight_queue_size
        );

        Self {
            workers,
            callbacks,
            default_spec: general,
            shutdown_flag,
            watchdog_signal,
            watchdog: Some(watchdog),
        }
    }

    /// Set or replace the processing spec for a specific source.
    ///
    /// If the worker already exists, sends an `UpdateSpec` message.
    /// Otherwise the spec is stored and used when the first frame arrives.
    pub fn set_source_spec(&self, source_id: &str, spec: SourceSpec) -> Result<(), PicassoError> {
        if self.shutdown_flag.load(Ordering::Relaxed) {
            return Err(PicassoError::Shutdown);
        }

        let mut workers = self.workers.lock();
        if let Some(worker) = workers.get(source_id) {
            worker
                .send(WorkerMessage::UpdateSpec(Box::new(spec)))
                .map_err(|_| PicassoError::ChannelDisconnected(source_id.to_string()))?;
        } else {
            let idle_timeout = Duration::from_secs(
                spec.idle_timeout_secs
                    .unwrap_or(self.default_spec.idle_timeout_secs),
            );
            let worker = SourceWorker::spawn(
                source_id.to_string(),
                spec,
                self.callbacks.clone(),
                idle_timeout,
                self.default_spec.inflight_queue_size,
            );
            workers.insert(source_id.to_string(), worker);
        }

        debug!("set_source_spec: source={source_id}");
        Ok(())
    }

    /// Remove the spec for a source. The worker will be shut down.
    pub fn remove_source_spec(&self, source_id: &str) {
        let mut workers = self.workers.lock();
        if let Some(worker) = workers.remove(source_id) {
            let _ = worker.send(WorkerMessage::Shutdown);
            drop(worker);
        }
        debug!("remove_source_spec: source={source_id}");
    }

    /// Submit a video frame for processing.
    ///
    /// Auto-creates a worker with `Drop` spec if one doesn't exist.
    pub fn send_frame(
        &self,
        source_id: &str,
        frame: savant_core::primitives::frame::VideoFrameProxy,
        view: deepstream_nvbufsurface::SurfaceView,
        src_rect: Option<deepstream_nvbufsurface::Rect>,
    ) -> Result<(), PicassoError> {
        if self.shutdown_flag.load(Ordering::Relaxed) {
            return Err(PicassoError::Shutdown);
        }

        let mut workers = self.workers.lock();
        let worker = workers.entry(source_id.to_string()).or_insert_with(|| {
            let idle_timeout = Duration::from_secs(self.default_spec.idle_timeout_secs);
            SourceWorker::spawn(
                source_id.to_string(),
                SourceSpec::default(),
                self.callbacks.clone(),
                idle_timeout,
                self.default_spec.inflight_queue_size,
            )
        });

        worker
            .send(WorkerMessage::Frame(frame, view, src_rect))
            .map_err(|_| PicassoError::ChannelDisconnected(source_id.to_string()))?;

        Ok(())
    }

    /// Send an end-of-stream signal to a specific source.
    pub fn send_eos(&self, source_id: &str) -> Result<(), PicassoError> {
        if self.shutdown_flag.load(Ordering::Relaxed) {
            return Err(PicassoError::Shutdown);
        }

        let workers = self.workers.lock();
        if let Some(worker) = workers.get(source_id) {
            worker
                .send(WorkerMessage::Eos)
                .map_err(|_| PicassoError::ChannelDisconnected(source_id.to_string()))?;
        }
        Ok(())
    }

    /// Gracefully shut down all workers and the watchdog.
    pub fn shutdown(&mut self) {
        info!("PicassoEngine shutting down");
        self.shutdown_flag.store(true, Ordering::Relaxed);
        self.watchdog_signal.notify_shutdown();

        let mut workers = self.workers.lock();
        for (source_id, worker) in workers.drain() {
            let _ = worker.send(WorkerMessage::Shutdown);
            debug!("shutdown sent: source={source_id}");
            drop(worker);
        }
        drop(workers);

        if let Some(handle) = self.watchdog.take() {
            let _ = handle.join();
        }
        info!("PicassoEngine shut down complete");
    }
}

impl Drop for PicassoEngine {
    fn drop(&mut self) {
        if !self.shutdown_flag.load(Ordering::Relaxed) {
            self.shutdown();
        }
    }
}
