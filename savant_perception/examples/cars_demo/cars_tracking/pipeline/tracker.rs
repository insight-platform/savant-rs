//! NvDCF tracker helpers for the `cars_tracking` sample.
//!
//! The tracker runs one frame at a time (`max_batch_size = 1`,
//! `max_batch_wait = 0`) to match the file-driven frame-by-frame
//! throughput of the rest of the pipeline.  [`process_tracker_output`]
//! delegates each per-frame output to
//! [`deepstream_nvtracker::TrackerOperatorFrameOutput::apply_to_frame`]
//! — a pure transform that maps each [`TrackedObject`] via
//! [`TrackedObject::to_track_update`] (keyed by
//! [`TrackedObject::input_roi_id`] — no IoU reconciliation needed) and
//! delegates to
//! [`VideoFrameProxy::apply_tracking_info`](savant_core::primitives::frame::VideoFrameProxy::apply_tracking_info).
//! Any [`TrackUpdate`](savant_core::primitives::misc_track::TrackUpdate)
//! whose `object_id` does not resolve is forwarded in the returned
//! `unmatched` vec; the sample logs each via the [`std::fmt::Display`]
//! impl and ticks the `unmatched_updates` counter.  The batch's
//! [`SealedDeliveries`](deepstream_nvtracker::SealedDeliveries) are
//! then forwarded on the downstream channel via the `forward` closure.
//!
//! Source-EOS + operator-error propagation is handled by the
//! framework's per-variant defaults on
//! [`NvTracker`](savant_perception::templates::NvTracker),
//! so [`process_tracker_output`] here is scoped to the `Tracking`
//! payload only.  Stream alignment still holds: the operator
//! guarantees that
//! [`TrackerOperatorOutput::Eos`](deepstream_nvtracker::TrackerOperatorOutput::Eos)
//! fires strictly after the last `Tracking` output for the same
//! source, and the template dispatches `Eos` to
//! [`NvTracker::default_on_source_eos`](savant_perception::templates::NvTracker::default_on_source_eos)
//! (or a user override) through the same shared router.

use anyhow::{anyhow, Result};
use deepstream_buffers::SavantIdMetaKind;
use deepstream_nvtracker::{
    default_ll_lib_path, NvTrackerBatchingOperatorConfig, NvTrackerConfig, Roi, SealedDeliveries,
    TrackedObject, TrackerBatchFormationCallback, TrackerBatchFormationResult,
    TrackerOperatorTrackingOutput,
};
use savant_core::primitives::frame::VideoFrameProxy;
use savant_core::primitives::object::ObjectOperations;
use savant_core::primitives::RBBox;
// The `HashMap<i32, Vec<Roi>>` inside `TrackerBatchFormationResult.rois`
// is a boundary type from `deepstream_nvtracker` and is `std::collections::HashMap`,
// so we need the std variant here.  Internal sets (e.g. `unique_track_ids`)
// still use `hashbrown`.
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use crate::assets;
use crate::cars_tracking::pipeline::infer::model::DETECTION_NAMESPACE;
use savant_perception::supervisor::StageName;

/// Tracker dimensions (NvDCF working canvas, independent of source resolution).
pub const TRACKER_WIDTH: u32 = 960;
/// Tracker dimensions (NvDCF working canvas, independent of source resolution).
pub const TRACKER_HEIGHT: u32 = 544;
/// Workspace-relative path to the NvDCF max-perf YAML asset.
const NVDCF_CONFIG_REL: &str =
    "savant_deepstream/nvtracker/assets/config_tracker_NvDCF_max_perf.yml";

const UNKNOWN_VEHICLE_CLASS_ID: i32 = 99;

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
/// Cheap to clone (one atomic + a small mutex-guarded set) and cheap
/// to tick.  Only the counters the orchestrator's end-of-run summary
/// actually reads are exposed:
///
/// * [`unique_tracks`](Self::unique_tracks) — cardinality of the set
///   of distinct `track_id`s observed across the run.
/// * [`unmatched_updates`](Self::unmatched_updates) — tracker-produced
///   updates whose `object_id` did not resolve to an existing
///   `VideoObject` on the frame (per the `misc_obj_info[0]`
///   preservation contract this should be 0 in normal operation).
#[derive(Default, Debug)]
pub struct TrackerStats {
    unmatched_updates: std::sync::atomic::AtomicU64,
    unique_track_ids: parking_lot::Mutex<hashbrown::HashSet<i64>>,
}

impl TrackerStats {
    /// Create fresh zeroed counters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record each tracked object's id on a single frame.
    pub fn record_frame<'a, I: IntoIterator<Item = &'a TrackedObject>>(&self, tracks: I) {
        let mut guard = self.unique_track_ids.lock();
        for t in tracks {
            guard.insert(t.object_id as i64);
        }
    }

    /// Record `n` unmatched
    /// [`TrackUpdate`](savant_core::primitives::misc_track::TrackUpdate)s
    /// returned by
    /// [`VideoFrameProxy::apply_tracking_info`](savant_core::primitives::frame::VideoFrameProxy::apply_tracking_info).
    /// Per the `misc_obj_info[0]` preservation contract this should
    /// be 0 in normal operation; a non-zero value flags a
    /// tracker-config drift worth investigating.
    pub fn record_unmatched_updates(&self, n: usize) {
        self.unmatched_updates
            .fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
    }

    /// Number of distinct track ids seen across the run.
    pub fn unique_tracks(&self) -> usize {
        self.unique_track_ids.lock().len()
    }

    /// Total number of tracker-produced updates whose `object_id` did
    /// not resolve to an existing `VideoObject` on the frame.
    pub fn unmatched_updates(&self) -> u64 {
        self.unmatched_updates
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// Process a single [`TrackerOperatorTrackingOutput`]: reconcile
/// tracker updates with detections on every frame in the batch and
/// update [`TrackerStats`] along the way, then return the sealed
/// batch ready for the caller to forward.
///
/// **Pure data transform** — no sending, no routing.  The caller
/// (the `on_tracking` hook body installed on
/// [`NvTracker`](savant_perception::templates::NvTracker)) owns the
/// `Router<PipelineMsg>` and is the single place that wraps the
/// returned [`SealedDeliveries`] in a [`PipelineMsg::Deliveries`]
/// envelope and emits it.  That split keeps metadata reconciliation
/// decoupled from the downstream routing policy (the sample routes
/// by name via `router.send_to(&tail_peer, ...)`).
///
/// Source-EOS and operator errors are handled by the framework's
/// per-variant defaults — see
/// [`NvTracker::default_on_source_eos`](savant_perception::templates::NvTracker::default_on_source_eos)
/// and [`NvTracker::default_on_error`](savant_perception::templates::NvTracker::default_on_error) —
/// so this sample-side processor only handles the tracking variant.
///
/// Returns `None` if the batch has no sealed deliveries to forward.
///
/// [`PipelineMsg::Deliveries`]: savant_perception::envelopes::PipelineMsg::Deliveries
pub fn process_tracker_output(
    mut tracking_output: TrackerOperatorTrackingOutput,
    stats: Option<&TrackerStats>,
    stage: &StageName,
) -> Option<SealedDeliveries> {
    for frame_output in tracking_output.frames() {
        match frame_output.apply_to_frame() {
            Ok(unmatched) => {
                if !unmatched.is_empty() {
                    if let Some(s) = stats {
                        s.record_unmatched_updates(unmatched.len());
                    }
                    for tu in &unmatched {
                        log::warn!("[{stage}] unmatched {tu}");
                    }
                }
            }
            Err(err) => log::warn!("[{stage}] apply_to_frame failed: {err}"),
        }
        log::debug!(
            "[{stage}] frame source={} tracks={}",
            frame_output.frame.get_source_id(),
            frame_output.tracked_objects.len()
        );
        if let Some(s) = stats {
            s.record_frame(frame_output.tracked_objects.iter());
        }
    }
    tracking_output.take_deliveries()
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
}
