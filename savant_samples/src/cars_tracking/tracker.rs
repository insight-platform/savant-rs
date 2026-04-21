//! NvDCF tracker stage for the `cars_tracking` sample.
//!
//! The tracker runs one frame at a time (`max_batch_size = 1`,
//! `max_batch_wait = 0`) to match the file-driven frame-by-frame
//! throughput of the rest of the pipeline.  The result callback
//! delegates each per-frame output to
//! [`deepstream_nvtracker::TrackerOperatorFrameOutput::apply_to_frame`]
//! — a pure transform that maps each [`TrackedObject`] via
//! [`TrackedObject::to_track_update`] (keyed by
//! [`TrackedObject::input_roi_id`] — no IoU reconciliation needed) and
//! delegates to
//! [`VideoFrameProxy::apply_tracking_info`](savant_core::primitives::frame::VideoFrameProxy::apply_tracking_info).
//! Any [`TrackUpdate`](savant_core::primitives::misc_track::TrackUpdate)
//! whose `object_id` does not resolve is forwarded in the returned
//! `unmatched` vec; the sample logs each via the
//! [`std::fmt::Display`] impl and ticks the `unmatched_updates`
//! counter.  The batch's [`SealedDeliveries`] are then forwarded on a
//! bounded channel; a dedicated consumer thread in
//! [`crate::cars_tracking::pipeline`] unseals each handle and submits
//! the frame to Picasso.  Per-source EOS / errors are only logged.

use anyhow::{anyhow, Result};
use crossbeam::channel::{Receiver, Sender};
use deepstream_buffers::SavantIdMetaKind;
use deepstream_nvtracker::{
    default_ll_lib_path, NvTrackerBatchingOperatorConfig, NvTrackerConfig, Roi, SealedDeliveries,
    TrackedObject, TrackerBatchFormationCallback, TrackerBatchFormationResult,
    TrackerOperatorOutput, TrackerOperatorResultCallback,
};
use savant_core::primitives::frame::VideoFrameProxy;
use savant_core::primitives::object::ObjectOperations;
use savant_core::primitives::RBBox;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use crate::assets;
use crate::cars_tracking::model::DETECTION_NAMESPACE;

/// Tracker dimensions (NvDCF working canvas, independent of source resolution).
pub const TRACKER_WIDTH: u32 = 960;
/// Tracker dimensions (NvDCF working canvas, independent of source resolution).
pub const TRACKER_HEIGHT: u32 = 544;
/// Workspace-relative path to the NvDCF max-perf YAML asset.
const NVDCF_CONFIG_REL: &str =
    "savant_deepstream/nvtracker/assets/config_tracker_NvDCF_max_perf.yml";

const UNKNOWN_VEHICLE_CLASS_ID: i32 = 99;

/// Sender half of the nvtracker-result channel — forwards sealed
/// deliveries from the operator callback to the consumer thread.
pub type TrackerResultSender = Sender<SealedDeliveries>;
/// Receiver half of the nvtracker-result channel.
pub type TrackerResultReceiver = Receiver<SealedDeliveries>;

/// Locate the NvDCF max-perf YAML shipped by `deepstream_nvtracker`.
pub fn nvdcf_config_path() -> Result<PathBuf> {
    assets::upstream_asset_path(NVDCF_CONFIG_REL)
}

/// Build an `NvTrackerBatchingOperatorConfig` targeting NvDCF max_perf.
///
/// Runs one frame at a time: `max_batch_size = 1`, `max_batch_wait = 0`.
pub fn build_tracker_config(gpu_id: u32) -> Result<NvTrackerBatchingOperatorConfig> {
    let ll_lib = default_ll_lib_path();
    if !Path::new(&ll_lib).is_file() {
        return Err(anyhow!("tracker low-level library not found: {}", ll_lib));
    }
    let ll_config = nvdcf_config_path()?;

    let mut nvtracker = NvTrackerConfig::new(ll_lib, ll_config.to_string_lossy().into_owned());
    nvtracker.tracker_width = TRACKER_WIDTH;
    nvtracker.tracker_height = TRACKER_HEIGHT;
    nvtracker.max_batch_size = 1;
    nvtracker.gpu_id = gpu_id;

    Ok(NvTrackerBatchingOperatorConfig::builder(nvtracker)
        .max_batch_size(1)
        .max_batch_wait(Duration::from_millis(0))
        .build())
}

/// Build the batch-formation callback for the tracker.
pub fn build_batch_formation() -> TrackerBatchFormationCallback {
    Arc::new(|frames: &[VideoFrameProxy]| {
        let mut ids = Vec::with_capacity(frames.len());
        let mut rois = Vec::with_capacity(frames.len());

        for (slot_idx, frame) in frames.iter().enumerate() {
            ids.push(SavantIdMetaKind::Frame(slot_idx as u128));

            let mut per_class: HashMap<i32, Vec<Roi>> = HashMap::new();
            for object in frame
                .get_all_objects()
                .into_iter()
                .filter(|o| o.get_namespace() == DETECTION_NAMESPACE)
            {
                let class_id = vehicle_class_id(&object.get_label());
                let roi = detection_to_roi(object.get_id(), &object.get_detection_box());
                per_class.entry(class_id).or_default().push(roi);
            }

            rois.push(per_class);
        }

        TrackerBatchFormationResult { ids, rois }
    })
}

/// Aggregate counters tracked across the tracking stage.
///
/// Cheap to clone (atomics + a small mutex-guarded set) and cheap to tick.
#[derive(Default, Debug)]
pub struct TrackerStats {
    frames: std::sync::atomic::AtomicU64,
    tracks_emitted: std::sync::atomic::AtomicU64,
    unmatched_updates: std::sync::atomic::AtomicU64,
    unique_track_ids: parking_lot::Mutex<std::collections::HashSet<i64>>,
}

impl TrackerStats {
    /// Create fresh zeroed counters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a single tracked frame and each tracked object's id.
    pub fn record_frame<'a, I: IntoIterator<Item = &'a TrackedObject>>(&self, tracks: I) {
        use std::sync::atomic::Ordering;
        self.frames.fetch_add(1, Ordering::Relaxed);
        let mut guard = self.unique_track_ids.lock();
        for t in tracks {
            guard.insert(t.object_id as i64);
            self.tracks_emitted.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record `n` unmatched [`TrackUpdate`](savant_core::primitives::misc_track::TrackUpdate)s
    /// returned by
    /// [`VideoFrameProxy::apply_tracking_info`](savant_core::primitives::frame::VideoFrameProxy::apply_tracking_info)
    /// (via the nvtracker bridge).  Per the `misc_obj_info[0]`
    /// preservation contract this should be 0 in normal operation; a
    /// non-zero value flags a tracker-config drift worth investigating.
    pub fn record_unmatched_updates(&self, n: usize) {
        self.unmatched_updates
            .fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
    }

    /// Frames seen by the tracker stage.
    pub fn frames(&self) -> u64 {
        self.frames.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Total number of (per-frame) tracked-object emissions.
    pub fn tracks_emitted(&self) -> u64 {
        self.tracks_emitted
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Number of distinct track ids seen.
    pub fn unique_tracks(&self) -> usize {
        self.unique_track_ids.lock().len()
    }

    /// Total number of tracker-produced updates whose `object_id` did
    /// not resolve to an existing `VideoObject` on the frame (see
    /// [`Self::record_unmatched_updates`]).
    pub fn unmatched_updates(&self) -> u64 {
        self.unmatched_updates
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Process a single [`TrackerOperatorOutput`]: reconcile tracker results
/// with detections, forward the resulting [`SealedDeliveries`], and log
/// per-source EOS / operator errors.
///
/// Shared by the result callback and the orchestrator's drain step (which
/// receives a `Vec<TrackerOperatorOutput>` from `graceful_shutdown`).
pub fn process_tracker_output(
    output: TrackerOperatorOutput,
    forward: &TrackerResultSender,
    stats: Option<&TrackerStats>,
) {
    match output {
        TrackerOperatorOutput::Tracking(mut tracking_output) => {
            for frame_output in tracking_output.frames() {
                match frame_output.apply_to_frame() {
                    Ok(unmatched) => {
                        if !unmatched.is_empty() {
                            if let Some(s) = stats {
                                s.record_unmatched_updates(unmatched.len());
                            }
                            for tu in &unmatched {
                                log::warn!("tracker: unmatched {tu}");
                            }
                        }
                    }
                    Err(err) => log::warn!("apply_to_frame failed: {err}"),
                }
                log::debug!(
                    "[track] frame source={} tracks={}",
                    frame_output.frame.get_source_id(),
                    frame_output.tracked_objects.len()
                );
                if let Some(s) = stats {
                    s.record_frame(frame_output.tracked_objects.iter());
                }
            }
            if let Some(sealed) = tracking_output.take_deliveries() {
                drop(tracking_output);
                if forward.send(sealed).is_err() {
                    log::warn!("nvtracker result receiver closed; dropping sealed batch");
                }
            }
        }
        TrackerOperatorOutput::Eos { source_id } => {
            log::info!("nvtracker source EOS: {source_id}");
        }
        TrackerOperatorOutput::Error(err) => {
            log::error!("nvtracker operator error: {err}");
        }
    }
}

/// Build the result callback — thin wrapper over [`process_tracker_output`].
pub fn build_result_callback(
    forward: TrackerResultSender,
    stats: Option<Arc<TrackerStats>>,
) -> TrackerOperatorResultCallback {
    Box::new(move |output: TrackerOperatorOutput| {
        process_tracker_output(output, &forward, stats.as_deref())
    })
}

fn vehicle_class_id(label: &str) -> i32 {
    match label {
        "car" => 0,
        "motorbike" => 1,
        "bus" => 2,
        "truck" => 3,
        _ => UNKNOWN_VEHICLE_CLASS_ID,
    }
}

fn detection_to_roi(index: i64, det_box: &RBBox) -> Roi {
    Roi {
        id: index,
        bbox: det_box.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: our NvDCF config wiring composes correctly and exposes
    /// the frame-by-frame knobs we commit to.  Skipped on hosts without
    /// DeepStream installed so the test suite stays portable.
    #[test]
    fn build_tracker_config_frame_by_frame_invariants() {
        if !Path::new(&default_ll_lib_path()).is_file() {
            return;
        }
        let cfg = build_tracker_config(0).expect("tracker config should resolve");
        assert_eq!(cfg.max_batch_size, 1);
        assert_eq!(cfg.max_batch_wait, Duration::from_millis(0));
    }

    /// Per-source EOS must be logged only — pipeline shutdown is driven by
    /// the orchestrator (sender drop), not the operator callback.
    #[test]
    fn callback_does_not_forward_on_source_eos() {
        let (tx, rx) = crossbeam::channel::bounded::<SealedDeliveries>(1);
        let mut cb = build_result_callback(tx, None);
        cb(TrackerOperatorOutput::Eos {
            source_id: "cam-1".to_string(),
        });
        assert!(rx.try_recv().is_err());
    }
}
