use crate::batch_meta_builder::rbbox_to_rect_params;
use crate::error::{NvInferError, Result};
use crate::output::BatchInferenceOutput;
use crate::pipeline::NvInfer;
use crate::roi::RoiKind;
use deepstream_buffers::{BatchState, SavantIdMetaKind, SharedBuffer};
use log::{error, warn};
use parking_lot::{Condvar, Mutex};
use savant_core::primitives::frame::VideoFrameProxy;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use super::output::{OperatorElement, OperatorFrameOutput, OperatorInferenceOutput};
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
}

impl NvInferBatchingOperator {
    /// Create a new batching operator.
    ///
    /// # Arguments
    ///
    /// * `config` — batching policy and NvInfer engine configuration.
    /// * `batch_formation` — called when a batch is ready, before NvInfer
    ///   submission. Must return per-frame ROIs and Savant IDs.
    /// * `result_callback` — called when inference results are available.
    pub fn new(
        config: super::NvInferBatchingOperatorConfig,
        batch_formation: BatchFormationCallback,
        mut result_callback: OperatorResultCallback,
    ) -> Result<Self> {
        let pending_batches: PendingMap = Arc::new(Mutex::new(HashMap::new()));
        let pending_for_cb = pending_batches.clone();

        let nvinfer_callback: Box<dyn FnMut(BatchInferenceOutput) + Send> =
            Box::new(move |output: BatchInferenceOutput| {
                let batch_id = find_batch_id(&output);
                let (buffer, elements, _clear_on_drop, host_copy_enabled) = output.into_parts();

                let batch_id = match batch_id {
                    Some(id) => id,
                    None => {
                        error!("Operator callback: no Batch SavantIdMeta on output buffer");
                        return;
                    }
                };

                let pending = pending_for_cb.lock().remove(&batch_id);
                let pending = match pending {
                    Some(p) => p,
                    None => {
                        warn!("Operator callback: no pending batch for id={batch_id}");
                        return;
                    }
                };

                let mut grouped: HashMap<u32, Vec<crate::output::ElementOutput>> = HashMap::new();
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
                    let raw_elems = grouped.remove(&(slot_idx as u32)).unwrap_or_default();

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
                                (Some(map), Some(id)) => {
                                    *map.get(&id).unwrap_or(&(0.0, 0.0, frame_w, frame_h))
                                }
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

                let operator_output = OperatorInferenceOutput::new(
                    frame_outputs,
                    deliveries,
                    host_copy_enabled,
                    buffer,
                );

                result_callback(operator_output);
            });

        let nvinfer = NvInfer::new(config.nvinfer.clone(), nvinfer_callback)?;

        let state = Arc::new(Mutex::new(BatchState::new()));
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let condvar = Arc::new(Condvar::new());

        let ctx = Arc::new(SubmitContext {
            config,
            batch_formation,
            state: state.clone(),
            pending_batches,
            next_batch_id: Arc::new(AtomicU64::new(0)),
            nvinfer: Arc::new(Mutex::new(nvinfer)),
            shutdown_flag: shutdown_flag.clone(),
        });

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
        })
    }

    /// Add a single frame for batched inference.
    ///
    /// If adding this frame fills the batch to `max_batch_size`, the batch is
    /// submitted immediately.
    pub fn add_frame(&self, frame: VideoFrameProxy, buffer: SharedBuffer) -> Result<()> {
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
        self.ctx.submit_batch()
    }

    /// Flush pending frames, stop the timer thread, and shut down NvInfer.
    pub fn shutdown(&mut self) -> Result<()> {
        let _ = self.flush();
        self.ctx.shutdown_flag.store(true, Ordering::Release);
        self.condvar.notify_one();

        if let Some(handle) = self.timer_thread.take() {
            let _ = handle.join();
        }

        self.ctx.nvinfer.lock().shutdown()
    }
}

impl Drop for NvInferBatchingOperator {
    fn drop(&mut self) {
        self.ctx.shutdown_flag.store(true, Ordering::Release);
        self.condvar.notify_one();
        if let Some(handle) = self.timer_thread.take() {
            let _ = handle.join();
        }
    }
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
    }
}
