use crate::error::{NvTrackerError, Result};
use crate::pipeline::{NvTracker, TrackedFrame};
use deepstream_buffers::{BatchState, SavantIdMetaKind};
use parking_lot::Mutex;
use savant_core::primitives::frame::VideoFrameProxy;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use super::config::NvTrackerBatchingOperatorConfig;
use super::types::{
    FramePair, PendingBatch, PendingMap, TrackerBatchFormationCallback, TrackerBatchFormationResult,
};

/// All `Arc`-cloned references needed by both `add_frame` and the timer
/// thread to perform batch submission.
pub(super) struct SubmitContext {
    pub(super) config: NvTrackerBatchingOperatorConfig,
    pub(super) batch_formation: TrackerBatchFormationCallback,
    pub(super) state: Arc<Mutex<BatchState<FramePair>>>,
    pub(super) pending_batches: PendingMap,
    pub(super) next_batch_id: Arc<AtomicU64>,
    pub(super) source_frame_counters: Arc<Mutex<HashMap<String, u32>>>,
    pub(super) nvtracker: Arc<NvTracker>,
    pub(super) shutdown_flag: Arc<AtomicBool>,
    pub(super) failed: Arc<AtomicBool>,
    pub(super) draining: Arc<AtomicBool>,
    /// Serialises the entire `drain → build → submit` window so frames
    /// enter [`NvTracker::submit`] in the same order they were drained
    /// from [`BatchState`].
    ///
    /// Without this lock, `add_frame` and the timer thread can each drain
    /// disjoint frames under the short-held `state` lock, release it, and
    /// then race inside `nvtracker.submit`.  The thread that wins the race
    /// assigns the earlier internal PTS, inverting the downstream order
    /// whenever `add_frame`'s frame is the newer one.  The symptom surfaces
    /// at the next stage as a PTS backward jump.
    pub(super) submit_lock: Mutex<()>,
}

impl SubmitContext {
    /// Drain the batch state and submit to NvTracker.
    ///
    /// Returns `Ok(())` when the batch is empty (no-op).
    pub(super) fn submit_batch(&self) -> Result<()> {
        self.submit_batch_impl(true)
    }

    pub(super) fn submit_batch_for_graceful_flush(&self) -> Result<()> {
        self.submit_batch_impl(false)
    }

    fn submit_batch_impl(&self, respect_draining: bool) -> Result<()> {
        if self.failed.load(Ordering::Acquire) {
            return Err(NvTrackerError::OperatorFailed);
        }
        if self.shutdown_flag.load(Ordering::Acquire) {
            return Err(NvTrackerError::OperatorShutdown);
        }
        if respect_draining && self.draining.load(Ordering::Acquire) {
            return Err(NvTrackerError::OperatorShutdown);
        }

        // Hold `submit_lock` for the whole critical section.  See the field
        // docs on [`SubmitContext::submit_lock`] for the rationale.
        let _submit_guard = self.submit_lock.lock();

        let frames = {
            let mut st = self.state.lock();
            if st.is_empty() {
                return Ok(());
            }
            st.take()
        };

        let frame_refs: Vec<VideoFrameProxy> = frames.iter().map(|(f, _)| f.clone()).collect();
        let TrackerBatchFormationResult { mut ids, rois } = (self.batch_formation)(&frame_refs);

        if rois.len() != frames.len() {
            return Err(NvTrackerError::BatchFormationFailed(format!(
                "rois.len() {} != frames.len() {}",
                rois.len(),
                frames.len()
            )));
        }

        let batch_id = self.next_batch_id.fetch_add(1, Ordering::Relaxed) as u128;
        ids.insert(0, SavantIdMetaKind::Batch(batch_id));

        let (tracked_frames, frame_nums, per_source_increments) = {
            let mut counters = self.source_frame_counters.lock();
            let mut tracked_frames = Vec::with_capacity(frames.len());
            let mut frame_nums = Vec::with_capacity(frames.len());
            let mut per_source_increments: HashMap<String, u32> = HashMap::new();
            for ((frame, buffer), frame_rois) in frames.iter().zip(rois.iter()) {
                let source = frame.get_source_id().to_string();
                let entry = counters.entry(source.clone()).or_insert(0);
                let current = *entry;
                *entry = entry.saturating_add(1);
                frame_nums.push(current);
                *per_source_increments.entry(source.clone()).or_insert(0) += 1;
                tracked_frames.push(TrackedFrame {
                    source,
                    buffer: buffer.clone(),
                    rois: frame_rois.clone(),
                });
            }
            (tracked_frames, frame_nums, per_source_increments)
        };

        self.pending_batches.lock().insert(
            batch_id,
            PendingBatch {
                frames: frames.clone(),
                frame_nums,
                submitted_at: Instant::now(),
            },
        );

        if let Err(e) = self.nvtracker.submit(&tracked_frames, ids) {
            self.pending_batches.lock().remove(&batch_id);
            let mut counters = self.source_frame_counters.lock();
            for (source, inc) in per_source_increments {
                if let Some(value) = counters.get_mut(&source) {
                    *value = value.saturating_sub(inc);
                }
            }
            return Err(e);
        }

        Ok(())
    }
}
