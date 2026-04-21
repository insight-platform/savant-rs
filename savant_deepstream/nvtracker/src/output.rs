//! Tracker output: current objects plus shadow / terminated / past-frame lists.

use crate::error::{NvTrackerError, Result};
use deepstream::{
    target_misc_batch_from_user_meta, BatchMeta, TargetMiscBatch,
    TargetMiscFrame as DsTargetMiscFrame, TrackState as DsTrackState,
};
use deepstream_buffers::SharedBuffer;
use deepstream_sys::{
    NvDsMetaType_NVDS_TRACKER_PAST_FRAME_META, NvDsMetaType_NVDS_TRACKER_SHADOW_LIST_META,
    NvDsMetaType_NVDS_TRACKER_TERMINATED_LIST_META,
};
use glib::translate::from_glib_none;
use gstreamer as gst;

pub use savant_core::primitives::misc_track::{
    MiscTrackCategory, MiscTrackData, MiscTrackFrame, TrackState,
};
use savant_core::primitives::{misc_track::TrackUpdate, RBBox};

/// One object in the current frame list (`NvDsObjectMeta` after tracking).
#[derive(Debug, Clone)]
pub struct TrackedObject {
    pub object_id: u64,
    pub class_id: i64,
    pub bbox_left: f32,
    pub bbox_top: f32,
    pub bbox_width: f32,
    pub bbox_height: f32,
    pub confidence: f32,
    pub tracker_confidence: f32,
    pub label: Option<String>,
    pub slot_number: i64,
    pub source_id: String,
    /// Raw `NvDsObjectMeta::misc_obj_info` payload as seen after the
    /// tracker.  Slot `[0]` is the input [`crate::Roi::id`] that the
    /// detection-meta stage stamped before tracking; the DeepStream
    /// `NvMultiObjectTracker` preserves it intact for current-frame
    /// tracked objects (verified by `tests/test_iou_tracker.rs ::
    /// test_misc_obj_info_zero_preserves_roi_id_iou`), which makes it
    /// the canonical channel for mapping tracker output back to the
    /// caller-side detection id — no IoU reconciliation required.
    /// Prefer [`TrackedObject::input_roi_id`] for this use case.
    ///
    /// Slots `[1..=3]` are reserved by DeepStream and may contain
    /// tracker-specific state; do not rely on any particular meaning.
    pub misc_obj_info: [i64; 4],
}

impl TrackedObject {
    /// Caller-defined [`crate::Roi::id`] that produced this tracked
    /// object, recovered from `misc_obj_info[0]`.
    ///
    /// Use this to pair a [`TrackedObject`] back to the
    /// `savant_core::primitives::VideoObject` (or any other caller-side
    /// datum) whose id was stamped into the `Roi` that fed the tracker,
    /// e.g. when building `savant_core::primitives::misc_track::TrackUpdate`s
    /// for `VideoFrameProxy::apply_tracking_info`.
    #[inline]
    pub fn input_roi_id(&self) -> i64 {
        self.misc_obj_info[0]
    }

    /// Convert this [`TrackedObject`] into a [`TrackUpdate`] ready to be
    /// passed to [`VideoFrameProxy::apply_tracking_info`](savant_core::primitives::frame::VideoFrameProxy::apply_tracking_info).
    ///
    /// This is the single source of truth for the
    /// [`TrackedObject`] → [`TrackUpdate`] mapping used by the
    /// `TrackerOperatorFrameOutput::apply_to_frame` bridge.
    ///
    /// Field mapping:
    /// * `object_id` ← [`Self::input_roi_id`] — the id of the
    ///   pre-tracker ROI (i.e. the `VideoObject::id` on the frame),
    ///   round-tripped through `misc_obj_info[0]` per the tracker's
    ///   preservation contract.
    /// * `track_id` ← the tracker-assigned stable object id
    ///   (`self.object_id` cast to `i64`).
    /// * `track_box` ← `RBBox::new(cx, cy, w, h, None)` where
    ///   `cx = bbox_left + bbox_width / 2` and
    ///   `cy = bbox_top  + bbox_height / 2`.
    ///   DeepStream nvtracker emits top-left + width/height in frame
    ///   pixels; [`RBBox`] / [`TrackUpdate::track_box`] is
    ///   center-x/center-y/width/height so we translate the origin
    ///   here, in one place.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Pseudo-code (TrackedObject construction is internal).
    /// let update = tracked_object.to_track_update();
    /// frame.apply_tracking_info(vec![update], vec![])?;
    /// ```
    pub fn to_track_update(&self) -> TrackUpdate {
        TrackUpdate::new(
            self.input_roi_id(),
            self.object_id as i64,
            RBBox::new(
                self.bbox_left + self.bbox_width / 2.0,
                self.bbox_top + self.bbox_height / 2.0,
                self.bbox_width,
                self.bbox_height,
                None,
            ),
        )
    }
}

/// Output of one successful [`crate::pipeline::NvTracker::submit`] → [`crate::pipeline::NvTracker::recv`] cycle.
#[derive(Debug)]
pub struct TrackerOutput {
    pub buffer: SharedBuffer,
    clear_on_drop: bool,
    pub current_tracks: Vec<TrackedObject>,
    pub shadow_tracks: Vec<MiscTrackData>,
    pub terminated_tracks: Vec<MiscTrackData>,
    pub past_frame_data: Vec<MiscTrackData>,
}

impl TrackerOutput {
    /// Consume this output and return its internal components.
    ///
    /// This disables the clear-on-drop behaviour: the caller becomes
    /// responsible for the underlying [`SharedBuffer`]. Useful when ownership
    /// needs to be transferred (e.g. into Python bindings).
    pub fn into_parts(
        mut self,
    ) -> (
        SharedBuffer,
        Vec<TrackedObject>,
        Vec<MiscTrackData>,
        Vec<MiscTrackData>,
        Vec<MiscTrackData>,
    ) {
        self.clear_on_drop = false;
        let buffer = self.buffer.clone();
        let current_tracks = std::mem::take(&mut self.current_tracks);
        let shadow_tracks = std::mem::take(&mut self.shadow_tracks);
        let terminated_tracks = std::mem::take(&mut self.terminated_tracks);
        let past_frame_data = std::mem::take(&mut self.past_frame_data);
        (
            buffer,
            current_tracks,
            shadow_tracks,
            terminated_tracks,
            past_frame_data,
        )
    }
}

fn ds_track_state(s: DsTrackState) -> TrackState {
    match s {
        DsTrackState::Empty => TrackState::Empty,
        DsTrackState::Active => TrackState::Active,
        DsTrackState::Inactive => TrackState::Inactive,
        DsTrackState::Tentative => TrackState::Tentative,
        DsTrackState::Projected => TrackState::Projected,
    }
}

fn misc_frame_from_ds(f: &DsTargetMiscFrame) -> MiscTrackFrame {
    // Widen DeepStream native widths (u32 / u32) to the savant-owned i64 domain.
    // Both widenings are lossless; no runtime check needed.
    MiscTrackFrame {
        frame_num: f.frame_num as i64,
        bbox_left: f.bbox_left,
        bbox_top: f.bbox_top,
        bbox_width: f.bbox_width,
        bbox_height: f.bbox_height,
        confidence: f.confidence,
        age: f.age as i64,
        state: ds_track_state(f.state),
        visibility: f.visibility,
    }
}

fn append_misc_batch(
    out: &mut Vec<MiscTrackData>,
    batch: TargetMiscBatch,
    category: MiscTrackCategory,
    resolve: &impl Fn(u32) -> String,
) {
    for stream in batch.streams {
        let sid = resolve(stream.stream_id);
        for obj in stream.objects {
            out.push(MiscTrackData {
                object_id: obj.unique_id,
                // Widen DS u16 class_id to savant i64 (lossless).
                class_id: obj.class_id as i64,
                label: obj.label.clone(),
                source_id: sid.clone(),
                category,
                frames: obj.frames.iter().map(misc_frame_from_ds).collect(),
            });
        }
    }
}

/// Build [`TrackerOutput`] from a completed GStreamer buffer.
pub fn extract_tracker_output(
    buffer: gst::Buffer,
    resolve_source_id: impl Fn(u32) -> String,
    clear_after: bool,
) -> Result<TrackerOutput> {
    let batch_meta = unsafe {
        BatchMeta::from_gst_buffer(buffer.as_ptr() as *mut deepstream_sys::GstBuffer).map_err(
            |e| {
                NvTrackerError::PipelineError(format!("BatchMeta::from_gst_buffer failed: {:?}", e))
            },
        )?
    };

    let mut current_tracks = Vec::new();
    for frame in batch_meta.frames() {
        let slot_number = frame.batch_id();
        let pad_index = frame.pad_index();
        let source_id = resolve_source_id(pad_index);
        for obj in frame.objects() {
            // Widen DS native widths (i32 class_id, u32 slot_number) to i64 (lossless).
            current_tracks.push(TrackedObject {
                object_id: obj.object_id(),
                class_id: obj.class_id() as i64,
                bbox_left: obj.rect_left(),
                bbox_top: obj.rect_top(),
                bbox_width: obj.rect_width(),
                bbox_height: obj.rect_height(),
                confidence: obj.confidence(),
                tracker_confidence: obj.tracker_confidence(),
                label: obj.label().unwrap_or(None),
                slot_number: slot_number as i64,
                source_id: source_id.clone(),
                misc_obj_info: obj.misc_obj_info(),
            });
        }
    }

    let mut shadow_tracks = Vec::new();
    let mut terminated_tracks = Vec::new();
    let mut past_frame_data = Vec::new();

    for um in batch_meta.batch_user_meta() {
        let mt = um.meta_type();
        let (dest, cat) = if mt == NvDsMetaType_NVDS_TRACKER_SHADOW_LIST_META {
            (Some(&mut shadow_tracks), MiscTrackCategory::Shadow)
        } else if mt == NvDsMetaType_NVDS_TRACKER_TERMINATED_LIST_META {
            (Some(&mut terminated_tracks), MiscTrackCategory::Terminated)
        } else if mt == NvDsMetaType_NVDS_TRACKER_PAST_FRAME_META {
            (Some(&mut past_frame_data), MiscTrackCategory::PastFrame)
        } else {
            (None, MiscTrackCategory::Shadow)
        };
        if let Some(d) = dest {
            if let Some(batch) = target_misc_batch_from_user_meta(&um, mt)? {
                append_misc_batch(d, batch, cat, &resolve_source_id);
            }
        }
    }

    let owned: gst::Buffer = unsafe { from_glib_none(buffer.as_ptr()) };
    let shared = SharedBuffer::from(owned);

    Ok(TrackerOutput {
        buffer: shared,
        clear_on_drop: clear_after,
        current_tracks,
        shadow_tracks,
        terminated_tracks,
        past_frame_data,
    })
}

impl Drop for TrackerOutput {
    fn drop(&mut self) {
        if !self.clear_on_drop {
            return;
        }
        let guard = self.buffer.lock();
        let ptr = guard.as_ref().as_ptr() as *mut deepstream_sys::GstBuffer;
        unsafe {
            deepstream::clear_all_frame_objects(ptr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tracked_object(input_roi_id: i64, object_id: u64) -> TrackedObject {
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
            source_id: String::from("s"),
            misc_obj_info: [input_roi_id, 0, 0, 0],
        }
    }

    #[test]
    fn tracked_object_to_track_update_roundtrip() {
        let t = sample_tracked_object(7, 42);
        let u = t.to_track_update();
        assert_eq!(u.object_id, 7);
        assert_eq!(u.track_id, 42);
        assert_eq!(u.track_box.get_xc(), 25.0); // 10 + 30/2
        assert_eq!(u.track_box.get_yc(), 40.0); // 20 + 40/2
        assert_eq!(u.track_box.get_width(), 30.0);
        assert_eq!(u.track_box.get_height(), 40.0);
    }

    #[test]
    fn tracked_object_input_roi_id_matches_slot_zero() {
        let t = sample_tracked_object(999, 1);
        assert_eq!(t.input_roi_id(), 999);
        assert_eq!(t.misc_obj_info[0], 999);
    }
}
