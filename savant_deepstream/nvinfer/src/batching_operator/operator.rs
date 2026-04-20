use crate::batch_meta_builder::rbbox_to_rect_params;
use crate::error::{NvInferError, Result};
use crate::output::BatchInferenceOutput;
use crate::pipeline::{NvInfer, NvInferOutput};
use crate::roi::RoiKind;
use deepstream_buffers::{BatchState, SavantIdMetaKind, SharedBuffer};
use log::{error, warn};
use parking_lot::{Condvar, Mutex};
use savant_core::primitives::frame::VideoFrameProxy;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::output::{
    OperatorElement, OperatorFrameOutput, OperatorInferenceOutput, OperatorOutput,
};
use super::submit::SubmitContext;
use super::types::{BatchFormationCallback, OperatorResultCallback, PendingMap};

/// Higher-level batching layer that accepts individual frames, accumulates
/// them into batches, and delegates inference to [`NvInfer`].
///
/// Results are delivered via a [`OperatorResultCallback`] with per-frame outputs
/// mapped back to the original `(VideoFrameProxy, SharedBuffer)` pairs.
pub struct NvInferBatchingOperator {
    ctx: Arc<SubmitContext>,
    condvar: Arc<Condvar>,
    timer_thread: Option<std::thread::JoinHandle<()>>,
    drain_thread: Option<std::thread::JoinHandle<()>>,
    /// During [`Self::graceful_shutdown`], operator outputs are collected here instead of the callback.
    draining_buffer: Arc<Mutex<Option<Vec<OperatorOutput>>>>,
}

impl NvInferBatchingOperator {
    /// Create a new batching operator.
    ///
    /// # Arguments
    ///
    /// * `config` — batching policy and NvInfer engine configuration.
    /// * `batch_formation` — called when a batch is ready, before NvInfer
    ///   submission. Must return per-frame ROIs and Savant IDs.
    /// * `result_callback` — called when inference results or source EOS are available.
    pub fn new(
        config: super::NvInferBatchingOperatorConfig,
        batch_formation: BatchFormationCallback,
        result_callback: OperatorResultCallback,
    ) -> Result<Self> {
        let pending_batches: PendingMap = Arc::new(Mutex::new(HashMap::new()));

        let nvinfer = Arc::new(NvInfer::new(config.nvinfer.clone())?);

        let state = Arc::new(Mutex::new(BatchState::new()));
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let failed = Arc::new(AtomicBool::new(false));
        let draining = Arc::new(AtomicBool::new(false));
        let draining_buffer: Arc<Mutex<Option<Vec<OperatorOutput>>>> = Arc::new(Mutex::new(None));
        let condvar = Arc::new(Condvar::new());

        let ctx = Arc::new(SubmitContext {
            config,
            batch_formation,
            state: state.clone(),
            pending_batches: pending_batches.clone(),
            next_batch_id: Arc::new(AtomicU64::new(0)),
            nvinfer: nvinfer.clone(),
            shutdown_flag: shutdown_flag.clone(),
            failed: failed.clone(),
            draining: draining.clone(),
            submit_lock: Mutex::new(()),
        });

        let drain_nvinfer = nvinfer;
        let drain_pending = pending_batches;
        let drain_failed = failed.clone();
        let drain_shutdown = shutdown_flag.clone();
        let drain_dbuf = draining_buffer.clone();
        let drain_thread_name = if ctx.config.nvinfer.name.is_empty() {
            "nvinfer-batching-operator-drain".to_string()
        } else {
            format!("nvinfer-{}-drain", ctx.config.nvinfer.name)
        };
        let drain_thread = std::thread::Builder::new()
            .name(drain_thread_name)
            .spawn({
                let mut result_callback = result_callback;
                move || {
                    drain_loop(
                        drain_nvinfer,
                        drain_pending,
                        drain_failed,
                        drain_shutdown,
                        drain_dbuf,
                        &mut result_callback,
                    );
                }
            })
            .map_err(|e| {
                NvInferError::PipelineError(format!("failed to spawn drain thread: {e}"))
            })?;

        let timer_ctx = ctx.clone();
        let timer_condvar = condvar.clone();
        let thread_name = if ctx.config.nvinfer.name.is_empty() {
            "nvinfer-batching-operator-timer".to_string()
        } else {
            format!("nvinfer-{}-timer", ctx.config.nvinfer.name)
        };
        let timer_thread = std::thread::Builder::new()
            .name(thread_name)
            .spawn(move || {
                timer_loop(timer_ctx, timer_condvar);
            })
            .map_err(|e| {
                NvInferError::PipelineError(format!("failed to spawn timer thread: {e}"))
            })?;

        Ok(Self {
            ctx,
            condvar,
            timer_thread: Some(timer_thread),
            drain_thread: Some(drain_thread),
            draining_buffer,
        })
    }

    /// Add a single frame for batched inference.
    ///
    /// If adding this frame fills the batch to `max_batch_size`, the batch is
    /// submitted immediately.
    pub fn add_frame(&self, frame: VideoFrameProxy, buffer: SharedBuffer) -> Result<()> {
        if self.ctx.failed.load(Ordering::Acquire) {
            return Err(NvInferError::OperatorFailed);
        }
        if self.ctx.draining.load(Ordering::Acquire) {
            return Err(NvInferError::OperatorShutdown);
        }
        if self.ctx.shutdown_flag.load(Ordering::Acquire) {
            return Err(NvInferError::OperatorShutdown);
        }

        let should_submit = {
            let mut st = self.ctx.state.lock();

            let is_first = st.frames.is_empty();
            st.frames.push((frame, buffer));

            if is_first {
                st.deadline = Some(Instant::now() + self.ctx.config.max_batch_wait);
                self.condvar.notify_one();
            }

            st.frames.len() >= self.ctx.config.max_batch_size
        };

        if should_submit {
            self.ctx.submit_batch()?;
        }

        Ok(())
    }

    /// Submit the current partial batch immediately (if non-empty).
    pub fn flush(&self) -> Result<()> {
        if self.ctx.draining.load(Ordering::Acquire) {
            return Err(NvInferError::OperatorShutdown);
        }
        self.ctx.submit_batch()
    }

    /// Propagate logical per-source EOS to the result callback.
    pub fn send_eos(&self, source_id: &str) -> Result<()> {
        if self.ctx.draining.load(Ordering::Acquire) {
            return Err(NvInferError::OperatorShutdown);
        }
        self.ctx.nvinfer.send_eos(source_id)
    }

    /// Graceful shutdown: reject new frames, flush pending batch, collect in-flight
    /// operator outputs until `timeout`, then stop threads and [`NvInfer`].
    pub fn graceful_shutdown(&mut self, timeout: Duration) -> Result<Vec<OperatorOutput>> {
        self.ctx.draining.store(true, Ordering::Release);
        self.ctx.submit_batch_for_graceful_flush()?;
        *self.draining_buffer.lock() = Some(Vec::new());

        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if self.ctx.failed.load(Ordering::Acquire) {
                break;
            }
            if self.ctx.pending_batches.lock().is_empty() {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }

        self.ctx.shutdown_flag.store(true, Ordering::Release);
        self.condvar.notify_one();

        if let Some(handle) = self.timer_thread.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.drain_thread.take() {
            let _ = handle.join();
        }

        let mut out = self.draining_buffer.lock().take().unwrap_or_default();

        let pending_count = self.ctx.pending_batches.lock().len();
        self.ctx.pending_batches.lock().clear();
        for _ in 0..pending_count {
            out.push(OperatorOutput::Error(NvInferError::PipelineError(
                "graceful_shutdown: pending batch incomplete before timeout".into(),
            )));
        }

        self.ctx.nvinfer.shutdown()?;
        Ok(out)
    }

    /// Flush pending frames, stop the timer thread, send EOS, and shut down.
    pub fn shutdown(&mut self) -> Result<()> {
        self.ctx.draining.store(true, Ordering::Release);
        let _ = self.ctx.submit_batch_for_graceful_flush();
        self.ctx.shutdown_flag.store(true, Ordering::Release);
        self.condvar.notify_one();

        if let Some(handle) = self.timer_thread.take() {
            let _ = handle.join();
        }

        if let Some(handle) = self.drain_thread.take() {
            let _ = handle.join();
        }

        self.ctx.nvinfer.shutdown()
    }
}

impl Drop for NvInferBatchingOperator {
    fn drop(&mut self) {
        self.ctx.shutdown_flag.store(true, Ordering::Release);
        self.condvar.notify_one();
        if let Some(handle) = self.timer_thread.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.drain_thread.take() {
            let _ = handle.join();
        }
    }
}

/// Drain thread: reads from `nvinfer.recv()` and routes results to the callback
/// or to `draining_buffer` when it is `Some` (graceful shutdown).
fn drain_loop(
    nvinfer: Arc<NvInfer>,
    pending_batches: PendingMap,
    failed: Arc<AtomicBool>,
    shutdown: Arc<AtomicBool>,
    draining_buffer: Arc<Mutex<Option<Vec<OperatorOutput>>>>,
    result_callback: &mut OperatorResultCallback,
) {
    let mut emit = |out: OperatorOutput| {
        let mut g = draining_buffer.lock();
        if let Some(v) = g.as_mut() {
            v.push(out);
        } else {
            result_callback(out);
        }
    };

    loop {
        if shutdown.load(Ordering::Acquire) || failed.load(Ordering::Acquire) {
            return;
        }
        let output = match nvinfer.recv_timeout(Duration::from_millis(100)) {
            Ok(Some(output)) => output,
            Ok(None) => continue,
            Err(e) => {
                if !shutdown.load(Ordering::Acquire) {
                    error!("Operator drain: recv error: {e}");
                    failed.store(true, Ordering::Release);
                }
                return;
            }
        };

        match output {
            NvInferOutput::Inference(batch_output) => {
                if let Some(op) = process_inference_output(batch_output, &pending_batches) {
                    emit(op);
                }
            }
            NvInferOutput::Eos { source_id } => {
                emit(OperatorOutput::Eos { source_id });
            }
            NvInferOutput::Error(e) => {
                emit(OperatorOutput::Error(e));
            }
            NvInferOutput::Event(_) => {}
        }
    }
}

/// Process a single inference output into an [`OperatorOutput`], or `None` if correlation failed.
fn process_inference_output(
    output: BatchInferenceOutput,
    pending_batches: &PendingMap,
) -> Option<OperatorOutput> {
    let batch_id = find_batch_id(&output);
    let (buffer, elements, _clear_on_drop, host_copy_enabled) = output.into_parts();

    let batch_id = match batch_id {
        Some(id) => id,
        None => {
            error!("Operator drain: no Batch SavantIdMeta on output buffer");
            return None;
        }
    };

    let pending = pending_batches.lock().remove(&batch_id);
    let pending = match pending {
        Some(p) => p,
        None => {
            warn!("Operator drain: no pending batch for id={batch_id}");
            return None;
        }
    };

    let mut grouped: HashMap<i64, Vec<crate::output::ElementOutput>> = HashMap::new();
    for elem in elements {
        grouped.entry(elem.slot_number).or_default().push(elem);
    }

    let num_frames = pending.frames.len();
    let mut frame_outputs = Vec::with_capacity(num_frames);
    let mut deliveries = Vec::with_capacity(num_frames);

    for (slot_idx, ((frame, buf), roi_kind)) in pending
        .frames
        .into_iter()
        .zip(pending.rois.into_iter())
        .enumerate()
    {
        let raw_elems = grouped.remove(&(slot_idx as i64)).unwrap_or_default();

        let frame_w = frame.get_width() as f32;
        let frame_h = frame.get_height() as f32;

        let roi_map: Option<HashMap<i64, (f32, f32, f32, f32)>> = match &roi_kind {
            RoiKind::Rois(rois) if !rois.is_empty() => {
                let map = rois
                    .iter()
                    .map(|r| {
                        let rect = rbbox_to_rect_params(&r.bbox, frame_w, frame_h);
                        (r.id, rect)
                    })
                    .collect();
                Some(map)
            }
            _ => None,
        };

        let op_elems: Vec<OperatorElement> = raw_elems
            .into_iter()
            .map(|elem| {
                let (rl, rt, rw, rh) = match (&roi_map, elem.roi_id) {
                    (Some(map), Some(id)) => *map.get(&id).unwrap_or(&(0.0, 0.0, frame_w, frame_h)),
                    _ => (0.0, 0.0, frame_w, frame_h),
                };
                OperatorElement::new(
                    elem,
                    rl,
                    rt,
                    rw,
                    rh,
                    pending.model_width,
                    pending.model_height,
                    pending.scaling,
                )
            })
            .collect();

        let frame_clone = frame.clone();
        deliveries.push((frame, buf));
        frame_outputs.push(OperatorFrameOutput {
            frame: frame_clone,
            elements: op_elems,
        });
    }

    let operator_output =
        OperatorInferenceOutput::new(frame_outputs, deliveries, host_copy_enabled, buffer);

    Some(OperatorOutput::Inference(operator_output))
}

/// Scan the output buffer's [`SavantIdMeta`] for a `Batch(id)` entry.
fn find_batch_id(output: &BatchInferenceOutput) -> Option<u128> {
    let ids = output.buffer().savant_ids();
    ids.into_iter().find_map(|id| match id {
        SavantIdMetaKind::Batch(v) => Some(v),
        _ => None,
    })
}

fn timer_loop(ctx: Arc<SubmitContext>, condvar: Arc<Condvar>) {
    loop {
        if ctx.failed.load(Ordering::Acquire) {
            return;
        }

        let deadline = {
            let st = ctx.state.lock();
            if ctx.shutdown_flag.load(Ordering::Acquire) {
                return;
            }
            st.deadline
        };

        match deadline {
            Some(dl) => {
                let now = Instant::now();
                if now >= dl {
                    if let Err(e) = ctx.submit_batch() {
                        error!("Operator timer submit_batch error: {e}");
                    }
                } else {
                    let mut st = ctx.state.lock();
                    let _ = condvar.wait_for(&mut st, dl - now);
                }
            }
            None => {
                let mut st = ctx.state.lock();
                if st.deadline.is_none() && !ctx.shutdown_flag.load(Ordering::Acquire) {
                    condvar.wait(&mut st);
                }
            }
        }

        let timeout = ctx.config.pending_batch_timeout;
        let expired: Vec<u128> = {
            let pending = ctx.pending_batches.lock();
            pending
                .iter()
                .filter(|(_, b)| b.submitted_at.elapsed() > timeout)
                .map(|(&id, _)| id)
                .collect()
        };
        if !expired.is_empty() {
            for id in &expired {
                ctx.pending_batches.lock().remove(id);
                error!(
                    "NvInfer: pending batch {id} timed out after {timeout:?}, \
                     operator entering failed state"
                );
            }
            ctx.failed.store(true, Ordering::Release);
            return;
        }
    }
}
