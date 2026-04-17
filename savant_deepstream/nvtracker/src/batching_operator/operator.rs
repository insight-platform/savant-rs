use crate::error::{NvTrackerError, Result};
use crate::output::TrackerOutput;
use crate::pipeline::{NvTracker, NvTrackerOutput};
use deepstream_buffers::{BatchState, SavantIdMetaKind, SharedBuffer};
use log::{error, warn};
use parking_lot::{Condvar, Mutex};
use savant_core::primitives::frame::VideoFrameProxy;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::output::{
    TrackerOperatorFrameOutput, TrackerOperatorOutput, TrackerOperatorTrackingOutput,
};
use super::submit::SubmitContext;
use super::types::{PendingMap, TrackerBatchFormationCallback, TrackerOperatorResultCallback};

/// Higher-level batching layer that accepts individual frames, accumulates
/// them into batches, and delegates tracking to [`NvTracker`].
pub struct NvTrackerBatchingOperator {
    ctx: Arc<SubmitContext>,
    condvar: Arc<Condvar>,
    timer_thread: Option<std::thread::JoinHandle<()>>,
    drain_thread: Option<std::thread::JoinHandle<()>>,
    draining_buffer: Arc<Mutex<Option<Vec<TrackerOperatorOutput>>>>,
}

impl NvTrackerBatchingOperator {
    /// Create a new batching operator.
    pub fn new(
        config: super::NvTrackerBatchingOperatorConfig,
        batch_formation: TrackerBatchFormationCallback,
        result_callback: TrackerOperatorResultCallback,
    ) -> Result<Self> {
        let pending_batches: PendingMap = Arc::new(Mutex::new(HashMap::new()));
        let pending_for_drain = pending_batches.clone();

        let nvtracker = Arc::new(NvTracker::new(config.nvtracker.clone())?);

        let state = Arc::new(Mutex::new(BatchState::new()));
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let failed = Arc::new(AtomicBool::new(false));
        let draining = Arc::new(AtomicBool::new(false));
        let draining_buffer: Arc<Mutex<Option<Vec<TrackerOperatorOutput>>>> =
            Arc::new(Mutex::new(None));
        let condvar = Arc::new(Condvar::new());

        let ctx = Arc::new(SubmitContext {
            config,
            batch_formation,
            state: state.clone(),
            pending_batches,
            next_batch_id: Arc::new(AtomicU64::new(0)),
            source_frame_counters: Arc::new(Mutex::new(HashMap::new())),
            nvtracker: nvtracker.clone(),
            shutdown_flag: shutdown_flag.clone(),
            failed: failed.clone(),
            draining: draining.clone(),
        });

        let drain_nvtracker = nvtracker;
        let drain_failed = failed.clone();
        let drain_shutdown = shutdown_flag.clone();
        let drain_dbuf = draining_buffer.clone();
        let drain_thread_name = if ctx.config.nvtracker.name.is_empty() {
            "nvtracker-batching-operator-drain".to_string()
        } else {
            format!("nvtracker-{}-drain", ctx.config.nvtracker.name)
        };
        let poll = ctx.config.nvtracker.drain_poll_interval;
        let drain_thread = std::thread::Builder::new()
            .name(drain_thread_name)
            .spawn({
                let mut result_callback = result_callback;
                move || {
                    drain_loop(
                        drain_nvtracker,
                        pending_for_drain,
                        drain_failed,
                        drain_shutdown,
                        drain_dbuf,
                        poll,
                        &mut result_callback,
                    );
                }
            })
            .map_err(|e| {
                NvTrackerError::PipelineError(format!("failed to spawn drain thread: {e}"))
            })?;

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
            drain_thread: Some(drain_thread),
            draining_buffer,
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
        if self.ctx.draining.load(Ordering::Acquire) {
            return Err(NvTrackerError::OperatorShutdown);
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
        if self.ctx.draining.load(Ordering::Acquire) {
            return Err(NvTrackerError::OperatorShutdown);
        }
        self.ctx.submit_batch()
    }

    /// Forward logical per-source EOS to the inner tracker.
    pub fn send_eos(&self, source_id: &str) -> Result<()> {
        if self.ctx.draining.load(Ordering::Acquire) {
            return Err(NvTrackerError::OperatorShutdown);
        }
        self.ctx.nvtracker.send_eos(source_id)
    }

    /// Forward stream reset to the inner tracker.
    pub fn reset_stream(&self, source_id: &str) -> Result<()> {
        if self.ctx.draining.load(Ordering::Acquire) {
            return Err(NvTrackerError::OperatorShutdown);
        }
        self.ctx.source_frame_counters.lock().remove(source_id);
        self.ctx.nvtracker.reset_stream(source_id)
    }

    pub fn graceful_shutdown(&mut self, timeout: Duration) -> Result<Vec<TrackerOperatorOutput>> {
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
            out.push(TrackerOperatorOutput::Error(NvTrackerError::PipelineError(
                "graceful_shutdown: pending batch incomplete before timeout".into(),
            )));
        }

        self.ctx.nvtracker.shutdown()?;
        Ok(out)
    }

    /// Flush pending frames, stop timer thread, join drain thread, shut down NvTracker.
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

        self.ctx.nvtracker.shutdown()
    }
}

impl Drop for NvTrackerBatchingOperator {
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

fn drain_loop(
    nvtracker: Arc<NvTracker>,
    pending_batches: PendingMap,
    failed: Arc<AtomicBool>,
    shutdown: Arc<AtomicBool>,
    draining_buffer: Arc<Mutex<Option<Vec<TrackerOperatorOutput>>>>,
    poll: Duration,
    result_callback: &mut TrackerOperatorResultCallback,
) {
    let mut emit = |out: TrackerOperatorOutput| {
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
        let output = match nvtracker.recv_timeout(poll.min(Duration::from_millis(500))) {
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
            NvTrackerOutput::Tracking(output) => {
                if let Some(op) = process_tracking_output(output, &pending_batches) {
                    emit(op);
                }
            }
            NvTrackerOutput::Eos { source_id } => {
                emit(TrackerOperatorOutput::Eos { source_id });
            }
            NvTrackerOutput::Error(e) => {
                emit(TrackerOperatorOutput::Error(e));
            }
            NvTrackerOutput::Event(_) => {}
        }
    }
}

fn process_tracking_output(
    output: TrackerOutput,
    pending_batches: &PendingMap,
) -> Option<TrackerOperatorOutput> {
    let (buffer, current_tracks, shadow_tracks, terminated_tracks, past_frame_data) =
        output.into_parts();

    let batch_id = match find_batch_id(&buffer) {
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

    let mut grouped_tracks: HashMap<i64, Vec<crate::TrackedObject>> = HashMap::new();
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
            .remove(&(slot_idx as i64))
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

    let operator_output = TrackerOperatorTrackingOutput::new(frame_outputs, deliveries);
    Some(TrackerOperatorOutput::Tracking(operator_output))
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
    // DS-side frame_num is u32; the savant-owned field is i64 (lossless widen).
    let frame_num = frame_num as i64;
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
