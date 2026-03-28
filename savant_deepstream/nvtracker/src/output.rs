//! Tracker output: current objects plus shadow / terminated / past-frame lists.

use crate::error::{NvTrackerError, Result};
use deepstream::{
    target_misc_batch_from_user_meta, BatchMeta, TargetMiscBatch, TargetMiscFrame as DsMiscFrame,
};
use deepstream_buffers::SharedBuffer;
use deepstream_sys::{
    NvDsMetaType_NVDS_TRACKER_PAST_FRAME_META, NvDsMetaType_NVDS_TRACKER_SHADOW_LIST_META,
    NvDsMetaType_NVDS_TRACKER_TERMINATED_LIST_META,
};
use glib::translate::from_glib_none;
use gstreamer as gst;

pub use deepstream::TrackState;

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

/// Historical / shadow frame row from tracker misc output.
#[derive(Debug, Clone)]
pub struct MiscTrackFrame {
    pub frame_num: u32,
    pub bbox_left: f32,
    pub bbox_top: f32,
    pub bbox_width: f32,
    pub bbox_height: f32,
    pub confidence: f32,
    pub age: u32,
    pub state: TrackState,
    pub visibility: f32,
}

/// One target in shadow / terminated / past-frame misc lists.
#[derive(Debug, Clone)]
pub struct MiscTrackData {
    pub object_id: u64,
    pub class_id: u16,
    pub label: Option<String>,
    pub source_id: String,
    pub frames: Vec<MiscTrackFrame>,
}

/// Output of one `track` / `track_sync` call.
#[derive(Debug)]
pub struct TrackerOutput {
    pub buffer: SharedBuffer,
    pub current_tracks: Vec<TrackedObject>,
    pub shadow_tracks: Vec<MiscTrackData>,
    pub terminated_tracks: Vec<MiscTrackData>,
    pub past_frame_data: Vec<MiscTrackData>,
}

fn misc_frame_from_ds(f: &DsMiscFrame) -> MiscTrackFrame {
    MiscTrackFrame {
        frame_num: f.frame_num,
        bbox_left: f.bbox_left,
        bbox_top: f.bbox_top,
        bbox_width: f.bbox_width,
        bbox_height: f.bbox_height,
        confidence: f.confidence,
        age: f.age,
        state: f.state,
        visibility: f.visibility,
    }
}

fn append_misc_batch(
    out: &mut Vec<MiscTrackData>,
    batch: TargetMiscBatch,
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
                frames: obj.frames.iter().map(misc_frame_from_ds).collect(),
            });
        }
    }
}

/// Build [`TrackerOutput`] from a completed GStreamer buffer.
pub fn extract_tracker_output(
    buffer: gst::Buffer,
    resolve_source_id: impl Fn(u32) -> String,
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
        let dest = if mt == NvDsMetaType_NVDS_TRACKER_SHADOW_LIST_META {
            Some(&mut shadow_tracks)
        } else if mt == NvDsMetaType_NVDS_TRACKER_TERMINATED_LIST_META {
            Some(&mut terminated_tracks)
        } else if mt == NvDsMetaType_NVDS_TRACKER_PAST_FRAME_META {
            Some(&mut past_frame_data)
        } else {
            None
        };
        if let Some(d) = dest {
            if let Some(batch) = target_misc_batch_from_user_meta(&um, mt)? {
                append_misc_batch(d, batch, &resolve_source_id);
            }
        }
    }

    let owned: gst::Buffer = unsafe { from_glib_none(buffer.as_ptr()) };
    let shared = SharedBuffer::from(owned);

    Ok(TrackerOutput {
        buffer: shared,
        current_tracks,
        shadow_tracks,
        terminated_tracks,
        past_frame_data,
    })
}
