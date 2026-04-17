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

/// One object in the current frame list (`NvDsObjectMeta` after tracking).
#[derive(Debug, Clone)]
pub struct TrackedObject {
    pub object_id: u64,
    pub class_id: i32,
    pub bbox_left: f32,
    pub bbox_top: f32,
    pub bbox_width: f32,
    pub bbox_height: f32,
    pub confidence: f32,
    pub tracker_confidence: f32,
    pub label: Option<String>,
    pub slot_number: u32,
    pub source_id: String,
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
    MiscTrackFrame {
        frame_num: f.frame_num,
        bbox_left: f.bbox_left,
        bbox_top: f.bbox_top,
        bbox_width: f.bbox_width,
        bbox_height: f.bbox_height,
        confidence: f.confidence,
        age: f.age,
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
                class_id: obj.class_id,
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
            current_tracks.push(TrackedObject {
                object_id: obj.object_id(),
                class_id: obj.class_id(),
                bbox_left: obj.rect_left(),
                bbox_top: obj.rect_top(),
                bbox_width: obj.rect_width(),
                bbox_height: obj.rect_height(),
                confidence: obj.confidence(),
                tracker_confidence: obj.tracker_confidence(),
                label: obj.label().unwrap_or(None),
                slot_number,
                source_id: source_id.clone(),
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
