use crate::error::NvTrackerError;
use crate::{MiscTrackData, TrackedObject};
use deepstream_buffers::{Sealed, SharedBuffer};
use savant_core::primitives::frame::VideoFrameProxy;
use savant_core::primitives::misc_track::TrackUpdate;
use savant_core::utils::release_seal::ReleaseSeal;
use std::sync::Arc;

pub use deepstream_buffers::SealedDeliveries;

/// Per-frame tracking result.
///
/// The per-frame [`SharedBuffer`] is held internally by the parent
/// [`TrackerOperatorTrackingOutput`] and is only accessible after calling
/// [`TrackerOperatorTrackingOutput::take_deliveries`] and then
/// [`SealedDeliveries::unseal`].
pub struct TrackerOperatorFrameOutput {
    /// The original [`VideoFrameProxy`] submitted for this frame.
    pub frame: VideoFrameProxy,
    /// Tracked objects for this frame.
    pub tracked_objects: Vec<TrackedObject>,
    /// Shadow tracks relevant to this frame source.
    pub shadow_tracks: Vec<MiscTrackData>,
    /// Terminated tracks relevant to this frame source.
    pub terminated_tracks: Vec<MiscTrackData>,
    /// Past-frame data relevant to this frame source.
    pub past_frame_data: Vec<MiscTrackData>,
}

impl TrackerOperatorFrameOutput {
    /// Apply this per-frame tracker output to its [`VideoFrameProxy`]
    /// in a single call, delegating to
    /// [`VideoFrameProxy::apply_tracking_info`].
    ///
    /// Each [`TrackedObject`] becomes a [`TrackUpdate`] via
    /// [`TrackedObject::to_track_update`] (keyed by
    /// [`TrackedObject::input_roi_id`] — the `misc_obj_info[0]`
    /// round-trip channel from the pre-tracker ROI).
    /// `shadow_tracks`, `terminated_tracks`, and `past_frame_data`
    /// are concatenated (each already carries its
    /// [`crate::MiscTrackCategory`]) and installed as the frame's
    /// misc-track list.
    ///
    /// The caller-visible policy (update existing, silently collect
    /// unresolved ids, replace misc tracks) is fully inherited from
    /// [`VideoFrameProxy::apply_tracking_info`].
    ///
    /// Returns the vec of unmatched [`TrackUpdate`]s forwarded
    /// verbatim from the core method (empty on the happy path).
    /// Each [`TrackUpdate`] has a [`std::fmt::Display`] impl for
    /// cheap logging at the caller.
    pub fn apply_to_frame(&self) -> anyhow::Result<Vec<TrackUpdate>> {
        let updates: Vec<TrackUpdate> = self
            .tracked_objects
            .iter()
            .map(TrackedObject::to_track_update)
            .collect();

        let mut misc = Vec::with_capacity(
            self.shadow_tracks.len() + self.terminated_tracks.len() + self.past_frame_data.len(),
        );
        misc.extend(self.shadow_tracks.iter().cloned());
        misc.extend(self.terminated_tracks.iter().cloned());
        misc.extend(self.past_frame_data.iter().cloned());

        self.frame.apply_tracking_info(updates, misc)
    }
}

/// Full batch tracking result with sealed buffer delivery.
pub struct TrackerOperatorTrackingOutput {
    frames: Vec<TrackerOperatorFrameOutput>,
    deliveries: Option<Vec<(VideoFrameProxy, SharedBuffer)>>,
    seal: Arc<ReleaseSeal>,
}

unsafe impl Send for TrackerOperatorTrackingOutput {}

impl TrackerOperatorTrackingOutput {
    /// Build a new tracking output from its constituent parts.
    pub(super) fn new(
        frames: Vec<TrackerOperatorFrameOutput>,
        deliveries: Vec<(VideoFrameProxy, SharedBuffer)>,
    ) -> Self {
        Self {
            frames,
            deliveries: Some(deliveries),
            seal: Arc::new(ReleaseSeal::new()),
        }
    }

    /// Per-frame outputs (tracking results only — no direct buffer access).
    pub fn frames(&self) -> &[TrackerOperatorFrameOutput] {
        &self.frames
    }

    /// Extract sealed deliveries while keeping the tracking output alive.
    ///
    /// Returns `Some(SealedDeliveries)` on the first call, otherwise `None`.
    pub fn take_deliveries(&mut self) -> Option<SealedDeliveries> {
        self.deliveries
            .take()
            .map(|d| Sealed::new(d, self.seal.clone()))
    }

    /// Apply every per-frame output via
    /// [`TrackerOperatorFrameOutput::apply_to_frame`].
    ///
    /// Returns a vec of unmatched [`TrackUpdate`] lists **aligned 1:1
    /// with [`Self::frames`]** (same length, same order) so callers
    /// can pair each per-frame result with its source frame without
    /// rebuilding an index.  Unresolved ids are silently collected
    /// inside [`VideoFrameProxy::apply_tracking_info`] so the batch
    /// never aborts for that reason; `Err` is still propagated for
    /// any other failure (e.g. lock poisoning) reported by the core
    /// method.
    pub fn apply_all_to_frames(&self) -> anyhow::Result<Vec<Vec<TrackUpdate>>> {
        let mut out = Vec::with_capacity(self.frames.len());
        for f in &self.frames {
            out.push(f.apply_to_frame()?);
        }
        Ok(out)
    }
}

impl Drop for TrackerOperatorTrackingOutput {
    fn drop(&mut self) {
        self.deliveries.take();
        self.seal.release();
    }
}

impl std::fmt::Debug for TrackerOperatorTrackingOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrackerOperatorTrackingOutput")
            .field("num_frames", &self.frames.len())
            .field("deliveries_taken", &self.deliveries.is_none())
            .finish()
    }
}

/// Callback payload from [`super::NvTrackerBatchingOperator`].
#[derive(Debug)]
pub enum TrackerOperatorOutput {
    /// Completed tracking for one submitted batch.
    Tracking(TrackerOperatorTrackingOutput),
    /// Logical per-source EOS from the underlying [`crate::pipeline::NvTracker`].
    Eos { source_id: String },
    /// Pipeline or operator runtime error.
    Error(NvTrackerError),
}

impl TrackerOperatorOutput {
    /// `true` if this is a [`TrackerOperatorOutput::Tracking`] variant.
    pub fn is_tracking(&self) -> bool {
        matches!(self, Self::Tracking(_))
    }

    /// `true` if this is [`TrackerOperatorOutput::Eos`].
    pub fn is_eos(&self) -> bool {
        matches!(self, Self::Eos { .. })
    }

    /// `true` if this is [`TrackerOperatorOutput::Error`].
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }

    /// Borrow the tracking payload when [`Self::is_tracking`].
    pub fn as_tracking(&self) -> Option<&TrackerOperatorTrackingOutput> {
        match self {
            Self::Tracking(t) => Some(t),
            _ => None,
        }
    }

    /// Mutably borrow the tracking payload when [`Self::is_tracking`].
    pub fn as_tracking_mut(&mut self) -> Option<&mut TrackerOperatorTrackingOutput> {
        match self {
            Self::Tracking(t) => Some(t),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MiscTrackCategory;
    use savant_core::primitives::object::ObjectOperations;
    use savant_core::test::{gen_empty_frame, gen_frame};

    fn tracked_obj(input_roi_id: i64, object_id: u64) -> TrackedObject {
        TrackedObject {
            object_id,
            class_id: 0,
            bbox_left: 10.0,
            bbox_top: 20.0,
            bbox_width: 30.0,
            bbox_height: 40.0,
            confidence: 0.9,
            tracker_confidence: 0.8,
            label: None,
            slot_number: 0,
            source_id: "test".to_string(),
            misc_obj_info: [input_roi_id, 0, 0, 0],
        }
    }

    fn misc_track(category: MiscTrackCategory) -> MiscTrackData {
        MiscTrackData {
            object_id: 99,
            class_id: 0,
            label: None,
            source_id: "test".to_string(),
            category,
            frames: Vec::new(),
        }
    }

    fn frame_output(
        frame: VideoFrameProxy,
        tracked_objects: Vec<TrackedObject>,
        shadow_tracks: Vec<MiscTrackData>,
        terminated_tracks: Vec<MiscTrackData>,
        past_frame_data: Vec<MiscTrackData>,
    ) -> TrackerOperatorFrameOutput {
        TrackerOperatorFrameOutput {
            frame,
            tracked_objects,
            shadow_tracks,
            terminated_tracks,
            past_frame_data,
        }
    }

    #[test]
    fn apply_to_frame_updates_existing_object_and_returns_empty_unmatched() {
        let frame = gen_frame();
        // `gen_frame` populates object id 0.
        let fo = frame_output(
            frame.clone(),
            vec![tracked_obj(0, 42)],
            vec![],
            vec![],
            vec![],
        );

        let unmatched = fo.apply_to_frame().expect("apply should succeed");
        assert!(unmatched.is_empty(), "got unmatched: {unmatched:?}");

        let obj = frame.get_object(0).expect("existing object");
        assert_eq!(obj.get_track_id(), Some(42));
        let tb = obj.get_track_box().expect("track_box");
        assert_eq!(tb.get_xc(), 25.0); // 10 + 30/2
        assert_eq!(tb.get_yc(), 40.0); // 20 + 40/2
        assert_eq!(tb.get_width(), 30.0);
        assert_eq!(tb.get_height(), 40.0);
    }

    #[test]
    fn apply_to_frame_returns_unmatched_when_input_roi_id_missing() {
        let frame = gen_empty_frame();
        let fo = frame_output(
            frame.clone(),
            vec![tracked_obj(999, 7)],
            vec![],
            vec![],
            vec![],
        );

        let unmatched = fo.apply_to_frame().expect("apply should succeed");
        assert_eq!(unmatched.len(), 1);
        assert_eq!(unmatched[0].object_id, 999);
        assert_eq!(unmatched[0].track_id, 7);

        let display = format!("{}", unmatched[0]);
        assert!(display.contains("object_id=999"), "got: {display}");

        // No phantom object synthesized.
        assert!(frame.get_all_objects().is_empty());
    }

    #[test]
    fn apply_to_frame_replaces_misc_tracks() {
        let frame = gen_frame();
        frame.add_misc_track(misc_track(MiscTrackCategory::PastFrame));
        assert_eq!(frame.get_misc_tracks().len(), 1);

        let fo = frame_output(
            frame.clone(),
            vec![],
            vec![misc_track(MiscTrackCategory::Shadow)],
            vec![misc_track(MiscTrackCategory::Terminated)],
            vec![],
        );

        let unmatched = fo.apply_to_frame().expect("apply should succeed");
        assert!(unmatched.is_empty());

        let tracks = frame.get_misc_tracks();
        assert_eq!(tracks.len(), 2);
        assert!(frame
            .get_misc_tracks_by_category(MiscTrackCategory::PastFrame)
            .is_empty());
    }

    #[test]
    fn apply_to_frame_aggregates_all_three_misc_buckets() {
        let frame = gen_empty_frame();
        let fo = frame_output(
            frame.clone(),
            vec![],
            vec![misc_track(MiscTrackCategory::Shadow)],
            vec![misc_track(MiscTrackCategory::Terminated)],
            vec![misc_track(MiscTrackCategory::PastFrame)],
        );

        let unmatched = fo.apply_to_frame().expect("apply should succeed");
        assert!(unmatched.is_empty());

        let all = frame.get_misc_tracks();
        assert_eq!(all.len(), 3);
        assert_eq!(
            frame
                .get_misc_tracks_by_category(MiscTrackCategory::Shadow)
                .len(),
            1
        );
        assert_eq!(
            frame
                .get_misc_tracks_by_category(MiscTrackCategory::Terminated)
                .len(),
            1
        );
        assert_eq!(
            frame
                .get_misc_tracks_by_category(MiscTrackCategory::PastFrame)
                .len(),
            1
        );
    }
}
