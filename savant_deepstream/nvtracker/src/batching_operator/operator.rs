use crate::error::{NvTrackerError, Result};
use crate::output::TrackerOutput;
use crate::pipeline::NvTracker;
use deepstream_buffers::{BatchState, SavantIdMetaKind, SharedBuffer};
use log::{error, warn};
use parking_lot::{Condvar, Mutex};
use savant_core::primitives::frame::VideoFrameProxy;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use super::output::{TrackerOperatorFrameOutput, TrackerOperatorOutput};
use super::submit::SubmitContext;
use super::types::{PendingMap, TrackerBatchFormationCallback, TrackerOperatorResultCallback};

/// Higher-level batching layer that accepts individual frames, accumulates
/// them into batches, and delegates tracking to [`NvTracker`].
pub struct NvTrackerBatchingOperator {
    ctx: Arc<SubmitContext>,
    condvar: Arc<Condvar>,
    timer_thread: Option<std::thread::JoinHandle<()>>,
}

impl NvTrackerBatchingOperator {
    /// Create a new batching operator.
    pub fn new(
        config: super::NvTrackerBatchingOperatorConfig,
        batch_formation: TrackerBatchFormationCallback,
        mut result_callback: TrackerOperatorResultCallback,
    ) -> Result<Self> {
        let pending_batches: PendingMap = Arc::new(Mutex::new(HashMap::new()));
        let pending_for_cb = pending_batches.clone();

        let nvtracker_callback: crate::pipeline::TrackerCallback = Box::new(move |output| {
            let TrackerOutput {
                buffer,
                current_tracks,
                shadow_tracks,
                terminated_tracks,
                past_frame_data,
            } = output;

            let batch_id = match find_batch_id(&buffer) {
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

            let mut grouped_tracks: HashMap<u32, Vec<crate::TrackedObject>> = HashMap::new();
            for track in current_tracks {
                grouped_tracks
                    .entry(track.slot_number)
                    .or_default()
                    .push(track);
            }

            let mut grouped_shadow: HashMap<String, Vec<crate::MiscTrackData>> = HashMap::new();
            for data in shadow_tracks {
                grouped_shadow
                    .entry(data.source_id.clone())
                    .or_default()
                    .push(data);
            }

            let mut grouped_terminated: HashMap<String, Vec<crate::MiscTrackData>> = HashMap::new();
            for data in terminated_tracks {
                grouped_terminated
                    .entry(data.source_id.clone())
                    .or_default()
                    .push(data);
            }

            let mut grouped_past: HashMap<String, Vec<crate::MiscTrackData>> = HashMap::new();
            for data in past_frame_data {
                grouped_past
                    .entry(data.source_id.clone())
                    .or_default()
                    .push(data);
            }

            let mut frame_outputs = Vec::with_capacity(pending.frames.len());
            let mut deliveries: Vec<(VideoFrameProxy, SharedBuffer)> =
                Vec::with_capacity(pending.frames.len());
            for (slot_idx, ((frame, buffer), frame_num)) in pending
                .frames
                .into_iter()
                .zip(pending.frame_nums.into_iter())
                .enumerate()
            {
                let source_id = frame.get_source_id().to_string();
                let frame_clone = frame.clone();
                let tracked_objects = grouped_tracks
                    .remove(&(slot_idx as u32))
                    .unwrap_or_default();
                let shadow_tracks = grouped_shadow
                    .get(&source_id)
                    .map(|v| filter_misc_for_frame_num(v, frame_num))
                    .unwrap_or_default();
                let terminated_tracks = grouped_terminated
                    .get(&source_id)
                    .map(|v| filter_misc_for_frame_num(v, frame_num))
                    .unwrap_or_default();
                let past_frame_data = grouped_past
                    .get(&source_id)
                    .map(|v| filter_misc_for_frame_num(v, frame_num))
                    .unwrap_or_default();

                deliveries.push((frame, buffer));
                frame_outputs.push(TrackerOperatorFrameOutput {
                    frame: frame_clone,
                    tracked_objects,
                    shadow_tracks,
                    terminated_tracks,
                    past_frame_data,
                });
            }

            let operator_output = TrackerOperatorOutput::new(frame_outputs, deliveries);
            result_callback(operator_output);
        });

        let nvtracker = NvTracker::new(config.nvtracker.clone(), nvtracker_callback)?;

        let state = Arc::new(Mutex::new(BatchState::new()));
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let failed = Arc::new(AtomicBool::new(false));
        let condvar = Arc::new(Condvar::new());

        let ctx = Arc::new(SubmitContext {
            config,
            batch_formation,
            state: state.clone(),
            pending_batches,
            next_batch_id: Arc::new(AtomicU64::new(0)),
            source_frame_counters: Arc::new(Mutex::new(HashMap::new())),
            nvtracker: Arc::new(Mutex::new(nvtracker)),
            shutdown_flag: shutdown_flag.clone(),
            failed: failed.clone(),
        });

        let timer_ctx = ctx.clone();
        let timer_condvar = condvar.clone();
        let thread_name = if ctx.config.nvtracker.name.is_empty() {
            "nvtracker-batching-operator-timer".to_string()
        } else {
            format!("nvtracker-{}-timer", ctx.config.nvtracker.name)
        };
        let timer_thread = std::thread::Builder::new()
            .name(thread_name)
            .spawn(move || {
                timer_loop(timer_ctx, timer_condvar);
            })
            .map_err(|e| {
                NvTrackerError::PipelineError(format!("failed to spawn timer thread: {e}"))
            })?;

        Ok(Self {
            ctx,
            condvar,
            timer_thread: Some(timer_thread),
        })
    }

    /// Add a single frame for batched tracking.
    ///
    /// If adding this frame fills the batch to `max_batch_size`, the batch is
    /// submitted immediately.
    pub fn add_frame(&self, frame: VideoFrameProxy, buffer: SharedBuffer) -> Result<()> {
        if self.ctx.failed.load(Ordering::Acquire) {
            return Err(NvTrackerError::OperatorFailed);
        }
        if self.ctx.shutdown_flag.load(Ordering::Acquire) {
            return Err(NvTrackerError::OperatorShutdown);
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

    /// Forward stream reset to the inner tracker.
    pub fn reset_stream(&self, source_id: &str) -> Result<()> {
        self.ctx.source_frame_counters.lock().remove(source_id);
        self.ctx.nvtracker.lock().reset_stream(source_id)
    }

    /// Flush pending frames, stop timer thread, and shut down NvTracker.
    pub fn shutdown(&mut self) -> Result<()> {
        let _ = self.flush();
        self.ctx.shutdown_flag.store(true, Ordering::Release);
        self.condvar.notify_one();

        if let Some(handle) = self.timer_thread.take() {
            let _ = handle.join();
        }

        self.ctx.nvtracker.lock().shutdown()
    }
}

impl Drop for NvTrackerBatchingOperator {
    fn drop(&mut self) {
        self.ctx.shutdown_flag.store(true, Ordering::Release);
        self.condvar.notify_one();
        if let Some(handle) = self.timer_thread.take() {
            let _ = handle.join();
        }
    }
}

/// Scan buffer Savant IDs for a `Batch(id)` entry.
fn find_batch_id(buffer: &SharedBuffer) -> Option<u128> {
    let ids = buffer.savant_ids();
    ids.into_iter().find_map(|id| match id {
        SavantIdMetaKind::Batch(v) => Some(v),
        _ => None,
    })
}

fn filter_misc_for_frame_num(
    items: &[crate::MiscTrackData],
    frame_num: u32,
) -> Vec<crate::MiscTrackData> {
    let mut out = Vec::new();
    for item in items {
        let filtered_frames: Vec<crate::MiscTrackFrame> = item
            .frames
            .iter()
            .filter(|f| f.frame_num == frame_num)
            .cloned()
            .collect();
        if !filtered_frames.is_empty() {
            let mut filtered_item = item.clone();
            filtered_item.frames = filtered_frames;
            out.push(filtered_item);
        }
    }
    // Fallback keeps source-level data visible when frame-number correlation
    // is unavailable (for example after an internal tracker stream reset).
    if out.is_empty() {
        items.to_vec()
    } else {
        out
    }
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

        // Check for expired pending batches.
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
                    "NvTracker: pending batch {id} timed out after {timeout:?}, \
                     operator entering failed state"
                );
            }
            ctx.failed.store(true, Ordering::Release);
            return;
        }
    }
}
