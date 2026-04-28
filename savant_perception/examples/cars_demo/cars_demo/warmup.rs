//! Engine preparation phase for the `cars_demo` sample.
//!
//! `NvInferBatchingOperator::new` is a blocking call that can take
//! anywhere from a few hundred milliseconds (cached TensorRT plan) up to
//! a couple of minutes (first-time ONNX -> TRT build on Jetson).  The
//! pipeline's [`savant_core::pipeline::stats::Stats`] timestamp-based
//! FPS reporter runs on an independent background thread and would
//! happily log "0 FPS" throughout that blocking window, which is
//! confusing for the user and indistinguishable from a stall.
//!
//! This module wraps the blocking nvinfer constructor in a heartbeat
//! helper so the user sees progress lines while TensorRT builds or
//! loads its plan:
//!
//! - [`prepare_nvinfer_operator`] logs whether a cached plan exists
//!   (so the user knows up-front whether a multi-minute rebuild is
//!   coming), spawns a lightweight heartbeat thread that prints
//!   `"… still working (Ns elapsed)"` every
//!   [`HEARTBEAT_PERIOD`] seconds, builds the operator, and on success
//!   promotes the freshly-built plan into the cache.
//!
//! The NvDCF tracker is constructed inline in the orchestrator
//! without a heartbeat — it reliably initializes in well under a
//! second and never triggers the "looks stalled" failure mode the
//! heartbeat is there to suppress.
//!
//! The orchestrator ([`crate::cars_demo::pipeline::run`]) calls
//! [`prepare_nvinfer_operator`] *before*
//! [`savant_core::pipeline::stats::Stats::kick_off`], so the periodic
//! FPS lines only start emitting once real frames are flowing.

use anyhow::{anyhow, Result};
use deepstream_nvinfer::prelude::NvInferBatchingOperator;
use deepstream_nvinfer::{
    BatchFormationCallback, NvInferBatchingOperatorConfig, OperatorResultCallback,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::cars_demo::pipeline::infer::model::{promote_yolo11n_engine, yolo11n_engine_cache_path};

/// Period at which [`with_heartbeat`] logs "still working" progress
/// lines while the wrapped closure is blocked.
pub const HEARTBEAT_PERIOD: Duration = Duration::from_secs(5);

/// Run `f` while a background thread logs a periodic "still working"
/// heartbeat at `period` intervals.  The heartbeat thread emits its
/// first line after the first `period` elapses (not immediately),
/// and stops as soon as `f` returns.
///
/// Emits `log::info!` lines of the form
/// `"{label}… still working (Ns elapsed)"`.
///
/// The heartbeat thread is best-effort: a panic in `f` propagates
/// normally; the heartbeat thread observes the stop flag set by the
/// drop guard and exits without holding any locks.
fn with_heartbeat<T>(label: &'static str, period: Duration, f: impl FnOnce() -> T) -> T {
    struct StopGuard {
        stop: Arc<AtomicBool>,
        handle: Option<JoinHandle<()>>,
    }
    impl Drop for StopGuard {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Release);
            if let Some(h) = self.handle.take() {
                h.thread().unpark();
                let _ = h.join();
            }
        }
    }

    let stop = Arc::new(AtomicBool::new(false));
    let start = Instant::now();
    let stop_for_thread = stop.clone();
    let handle = thread::Builder::new()
        .name(format!("hb-{label}"))
        .spawn(move || {
            while !stop_for_thread.load(Ordering::Acquire) {
                thread::park_timeout(period);
                if stop_for_thread.load(Ordering::Acquire) {
                    break;
                }
                log::info!(
                    "{label}... still working ({:.1}s elapsed)",
                    start.elapsed().as_secs_f64()
                );
            }
        })
        .ok();
    let _guard = StopGuard { stop, handle };
    f()
}

/// Prepare the nvinfer operator for the cars-tracking sample.
///
/// Responsibilities beyond the bare [`NvInferBatchingOperator::new`] call:
///
/// - Emits a single `log::info!` line describing whether a cached
///   TensorRT plan already exists (and the exact path).  When it is
///   missing, the message explicitly warns the user that an ONNX ->
///   TRT build is about to happen, typically 20–120 seconds.
/// - Spawns a heartbeat thread via [`with_heartbeat`] so the user sees
///   "still working" progress every [`HEARTBEAT_PERIOD`] seconds while
///   the operator constructor is blocked.
/// - On success, promotes the freshly-built engine into the platform-
///   tagged cache via [`promote_yolo11n_engine`] so the next run
///   loads instead of rebuilding.
///
/// Returns the constructed operator plus the wall-clock duration of
/// the build.
pub fn prepare_nvinfer_operator(
    gpu_id: u32,
    cfg: NvInferBatchingOperatorConfig,
    batch_cb: BatchFormationCallback,
    result_cb: OperatorResultCallback,
) -> Result<(NvInferBatchingOperator, Duration)> {
    let cache_path = yolo11n_engine_cache_path(gpu_id)?;
    if cache_path.exists() {
        log::info!(
            "[warmup/nvinfer] loading cached TensorRT engine from {} (expected <2s)",
            cache_path.display()
        );
    } else {
        log::info!(
            "[warmup/nvinfer] no cached engine at {} - DeepStream will build an ONNX -> TensorRT plan on GPU {gpu_id}; this typically takes 20-120s on first run. Subsequent runs will load the cached plan.",
            cache_path.display()
        );
    }

    let start = Instant::now();
    let operator = with_heartbeat("[warmup/nvinfer]", HEARTBEAT_PERIOD, || {
        NvInferBatchingOperator::new(cfg, batch_cb, result_cb)
    })
    .map_err(|e| anyhow!("build NvInferBatchingOperator: {e}"))?;
    let elapsed = start.elapsed();
    log::info!(
        "[warmup/nvinfer] engine ready ({:.2}s)",
        elapsed.as_secs_f64()
    );

    // Promote the freshly-built plan into the platform-tagged cache.
    // Safe to call unconditionally: it is a no-op when DeepStream
    // reused the cached plan without writing a new file.
    if let Err(e) = promote_yolo11n_engine(gpu_id) {
        log::warn!("[warmup/nvinfer] engine promotion failed (will rebuild next run): {e:#}");
    }

    Ok((operator, elapsed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// The heartbeat must not fire when the wrapped closure returns
    /// before the first period elapses — users running fast repeated
    /// loads (cached plan) should not see spurious heartbeat lines.
    #[test]
    fn heartbeat_no_fire_on_quick_completion() {
        let counter = Arc::new(AtomicUsize::new(0));
        let out = with_heartbeat("[test/quick]", Duration::from_secs(3600), || {
            counter.fetch_add(1, Ordering::SeqCst);
            42u32
        });
        assert_eq!(out, 42);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    /// The heartbeat helper must always propagate the wrapped value
    /// unchanged — including `Result::Err` — because the caller uses
    /// it as a transparent wrapper around the blocking constructor.
    #[test]
    fn heartbeat_propagates_err() {
        let out: Result<(), &'static str> =
            with_heartbeat("[test/err]", Duration::from_secs(3600), || Err("boom"));
        assert_eq!(out, Err("boom"));
    }
}
