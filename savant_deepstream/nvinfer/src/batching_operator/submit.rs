use crate::error::{NvInferError, Result};
use crate::pipeline::NvInfer;
use crate::roi::{Roi, RoiKind};
use deepstream_buffers::{NonUniformBatch, SavantIdMetaKind, SurfaceView};
use log::debug;
use parking_lot::Mutex;
use savant_core::primitives::frame::VideoFrameProxy;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;

use super::config::NvInferBatchingOperatorConfig;
use super::state::BatchState;
use super::types::{BatchFormationCallback, BatchFormationResult, PendingBatch, PendingMap};

/// All the `Arc`-cloned references needed by both `add_frame` and the timer
/// thread to perform batch submission.
pub(super) struct SubmitContext {
    pub(super) config: NvInferBatchingOperatorConfig,
    pub(super) batch_formation: BatchFormationCallback,
    pub(super) state: Arc<Mutex<BatchState>>,
    pub(super) pending_batches: PendingMap,
    pub(super) next_batch_id: Arc<AtomicI64>,
    pub(super) nvinfer: Arc<Mutex<NvInfer>>,
    pub(super) shutdown_flag: Arc<AtomicBool>,
}

impl SubmitContext {
    /// Drain the batch state and submit to NvInfer.
    ///
    /// Returns `Ok(())` even when the batch is empty (no-op).
    pub(super) fn submit_batch(&self) -> Result<()> {
        if self.shutdown_flag.load(Ordering::Acquire) {
            return Err(NvInferError::OperatorShutdown);
        }

        let frames = {
            let mut st = self.state.lock();
            if st.is_empty() {
                return Ok(());
            }
            st.take()
        };

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

        let batch_id = self.next_batch_id.fetch_add(1, Ordering::Relaxed);
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
            },
        );

        debug!("Operator submitting batch_id={batch_id}");
        self.nvinfer
            .lock()
            .submit(shared_buffer, rois_arg.as_ref())?;
        Ok(())
    }
}
