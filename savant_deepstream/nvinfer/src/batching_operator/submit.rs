use crate::error::{NvInferError, Result};
use crate::pipeline::NvInfer;
use crate::roi::{Roi, RoiKind};
use deepstream_buffers::{BatchState, NonUniformBatch, SavantIdMetaKind, SurfaceView};
use log::debug;
use parking_lot::Mutex;
use savant_core::primitives::frame::VideoFrameProxy;
use savant_gstreamer::submit_gate::SubmitGate;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use super::config::NvInferBatchingOperatorConfig;
use super::types::{
    BatchFormationCallback, BatchFormationResult, FramePair, PendingBatch, PendingMap,
};

/// All the `Arc`-cloned references needed by both `add_frame` and the timer
/// thread to perform batch submission.
pub(super) struct SubmitContext {
    pub(super) config: NvInferBatchingOperatorConfig,
    pub(super) batch_formation: BatchFormationCallback,
    pub(super) state: Arc<Mutex<BatchState<FramePair>>>,
    pub(super) pending_batches: PendingMap,
    pub(super) nvinfer: Arc<NvInfer>,
    pub(super) shutdown_flag: Arc<AtomicBool>,
    pub(super) failed: Arc<AtomicBool>,
    /// When true, [`Self::submit_batch`] rejects; [`Self::submit_batch_for_graceful_flush`] still runs.
    pub(super) draining: Arc<AtomicBool>,
    /// Serialises the entire `drain → build → submit` window so that the
    /// order in which frames leave [`BatchState`] is also the order in
    /// which they enter [`NvInfer::submit`] (and therefore the order in
    /// which the internal GStreamer pipeline timestamps them), and owns
    /// the monotonic `batch_id` counter it assigns inside that window.
    ///
    /// Without this gate, `add_frame` and the timer thread can each
    /// drain disjoint frames under the short-held `state` lock, release
    /// it, and then race inside `nvinfer.submit`.  The thread that wins
    /// the race gets the earlier internal PTS, which inverts the
    /// downstream order when `add_frame`'s frame is the newer one.  The
    /// symptom surfaces at the next pipeline stage as a PTS backward
    /// jump.
    ///
    /// Housing the `batch_id` counter inside [`SubmitGate`] — rather
    /// than keeping it in a sibling `Arc<AtomicU64>` — makes it a
    /// compile-time error to advance the batch id without holding the
    /// serialiser.
    pub(super) submit_gate: SubmitGate,
}

impl SubmitContext {
    /// Drain the batch state and submit to NvInfer.
    ///
    /// Returns `Ok(())` even when the batch is empty (no-op).
    pub(super) fn submit_batch(&self) -> Result<()> {
        self.submit_batch_impl(true)
    }

    /// Submit pending frames during [`super::NvInferBatchingOperator::graceful_shutdown`] (ignores draining).
    pub(super) fn submit_batch_for_graceful_flush(&self) -> Result<()> {
        self.submit_batch_impl(false)
    }

    fn submit_batch_impl(&self, respect_draining: bool) -> Result<()> {
        if self.failed.load(Ordering::Acquire) {
            return Err(NvInferError::OperatorFailed);
        }
        if self.shutdown_flag.load(Ordering::Acquire) {
            return Err(NvInferError::OperatorShutdown);
        }
        if respect_draining && self.draining.load(Ordering::Acquire) {
            return Err(NvInferError::OperatorShutdown);
        }

        // Hold `submit_gate` for the whole critical section.  This ensures
        // that `state.take()`, `batch_id` assignment, `pending_batches`
        // insertion, and `nvinfer.submit` happen atomically with respect
        // to other callers of `submit_batch_impl` — preserving the
        // video-PTS order of frames as they enter the downstream
        // pipeline.  `batch_id` is only advanced after a non-empty batch
        // commits, matching the old `fetch_add`-on-commit semantics.
        self.submit_gate.submit_with(|next_batch_id| {
            let frames = {
                let mut st = self.state.lock();
                if st.is_empty() {
                    return Ok(());
                }
                st.take()
            };

            let batch_id = *next_batch_id as u128;
            *next_batch_id += 1;

            let frame_refs: Vec<VideoFrameProxy> = frames.iter().map(|(f, _)| f.clone()).collect();
            let BatchFormationResult { mut ids, rois } = (self.batch_formation)(&frame_refs);

            let mut batch = NonUniformBatch::new(self.config.nvinfer.gpu_id);
            for (_, buffer) in &frames {
                let view = SurfaceView::from_buffer(buffer, 0).map_err(|e| {
                    NvInferError::BatchFormationFailed(format!("SurfaceView::from_buffer: {e}"))
                })?;
                batch.add(&view).map_err(|e| {
                    NvInferError::BatchFormationFailed(format!("NonUniformBatch::add: {e}"))
                })?;
            }

            ids.insert(0, SavantIdMetaKind::Batch(batch_id));

            let shared_buffer = batch.finalize(ids).map_err(|e| {
                NvInferError::BatchFormationFailed(format!("NonUniformBatch::finalize: {e}"))
            })?;

            let rois_map: HashMap<u32, Vec<Roi>> = rois
                .iter()
                .enumerate()
                .filter_map(|(i, kind)| match kind {
                    RoiKind::FullFrame => None,
                    RoiKind::Rois(r) if r.is_empty() => None,
                    RoiKind::Rois(r) => Some((i as u32, r.clone())),
                })
                .collect();
            let rois_arg = if rois_map.is_empty() {
                None
            } else {
                Some(rois_map)
            };

            let (mw, mh) = self.config.nvinfer.model_input_dimensions();
            let model_width = mw as f32;
            let model_height = mh as f32;
            let scaling = self.config.nvinfer.scaling;

            self.pending_batches.lock().insert(
                batch_id,
                PendingBatch {
                    frames,
                    rois,
                    model_width,
                    model_height,
                    scaling,
                    submitted_at: Instant::now(),
                },
            );

            debug!("Operator submitting batch_id={batch_id}");
            self.nvinfer.submit(shared_buffer, rois_arg.as_ref())?;
            Ok(())
        })
    }
}
