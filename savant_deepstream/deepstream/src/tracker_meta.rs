//! Tracker miscellaneous batch metadata (`NvDsTargetMiscDataBatch`).
//!
//! The nvtracker plugin attaches user meta of types such as
//! [`deepstream_sys::NvDsMetaType_NVDS_TRACKER_SHADOW_LIST_META`] to the batch.

use crate::{DeepStreamError, Result, UserMeta};
use deepstream_sys::{
    NvDsMetaType_NVDS_TRACKER_PAST_FRAME_META, NvDsMetaType_NVDS_TRACKER_SHADOW_LIST_META,
    NvDsMetaType_NVDS_TRACKER_TERMINATED_LIST_META, NvDsTargetMiscDataBatch,
    NvDsTargetMiscDataFrame, NvDsTargetMiscDataObject, NvDsTargetMiscDataStream, TRACKER_STATE,
};

/// Logical tracker state from DeepStream's `TRACKER_STATE` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackState {
    Empty,
    Active,
    Inactive,
    Tentative,
    Projected,
}

impl From<TRACKER_STATE> for TrackState {
    fn from(v: TRACKER_STATE) -> Self {
        match v {
            deepstream_sys::TRACKER_STATE_EMPTY => TrackState::Empty,
            deepstream_sys::TRACKER_STATE_ACTIVE => TrackState::Active,
            deepstream_sys::TRACKER_STATE_INACTIVE => TrackState::Inactive,
            deepstream_sys::TRACKER_STATE_TENTATIVE => TrackState::Tentative,
            deepstream_sys::TRACKER_STATE_PROJECTED => TrackState::Projected,
            _ => TrackState::Empty,
        }
    }
}

/// One frame row inside tracker misc output for a target.
#[derive(Debug, Clone)]
pub struct TargetMiscFrame {
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

/// One tracked target in misc output (shadow / terminated / past-frame).
#[derive(Debug, Clone)]
pub struct TargetMiscObject {
    pub unique_id: u64,
    pub class_id: u16,
    pub label: Option<String>,
    pub frames: Vec<TargetMiscFrame>,
}

/// Per-stream slice inside a misc batch.
#[derive(Debug, Clone)]
pub struct TargetMiscStream {
    pub stream_id: u32,
    pub surface_stream_id: u64,
    pub objects: Vec<TargetMiscObject>,
}

/// Owned copy of `NvDsTargetMiscDataBatch` contents.
#[derive(Debug, Clone)]
pub struct TargetMiscBatch {
    pub streams: Vec<TargetMiscStream>,
}

fn c_label_to_string(buf: &[std::os::raw::c_char]) -> Option<String> {
    let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    if len == 0 {
        return None;
    }
    let bytes = unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), len) };
    std::str::from_utf8(bytes).ok().map(|s| s.to_owned())
}

/// Parse `NvDsTargetMiscDataBatch` from batch user meta if `meta_type` matches
/// one of the tracker misc list types.
pub fn target_misc_batch_from_user_meta(
    user_meta: &UserMeta,
    expected_meta_type: i32,
) -> Result<Option<TargetMiscBatch>> {
    if user_meta.meta_type() != expected_meta_type {
        return Ok(None);
    }
    let expected = [
        NvDsMetaType_NVDS_TRACKER_SHADOW_LIST_META,
        NvDsMetaType_NVDS_TRACKER_TERMINATED_LIST_META,
        NvDsMetaType_NVDS_TRACKER_PAST_FRAME_META,
    ];
    if !expected.contains(&expected_meta_type) {
        return Err(DeepStreamError::invalid_parameter(
            "meta_type is not a tracker misc list type",
        ));
    }

    let ptr = user_meta.user_meta_data() as *mut NvDsTargetMiscDataBatch;
    if ptr.is_null() {
        return Ok(None);
    }
    // SAFETY: DeepStream owns the structure; we only read during the buffer callback.
    let batch = unsafe { &*ptr };
    Ok(Some(parse_misc_batch(batch)?))
}

fn parse_misc_batch(batch: &NvDsTargetMiscDataBatch) -> Result<TargetMiscBatch> {
    let n = batch.numFilled as usize;
    if batch.list.is_null() || n == 0 {
        return Ok(TargetMiscBatch {
            streams: Vec::new(),
        });
    }
    let streams_slice = unsafe { std::slice::from_raw_parts(batch.list, n) };
    let mut streams = Vec::with_capacity(n);
    for stream in streams_slice {
        streams.push(parse_misc_stream(stream)?);
    }
    Ok(TargetMiscBatch { streams })
}

fn parse_misc_stream(stream: &NvDsTargetMiscDataStream) -> Result<TargetMiscStream> {
    let n = stream.numFilled as usize;
    let objects = if stream.list.is_null() || n == 0 {
        Vec::new()
    } else {
        let objs = unsafe { std::slice::from_raw_parts(stream.list, n) };
        objs.iter()
            .map(parse_misc_object)
            .collect::<Result<Vec<_>>>()?
    };
    Ok(TargetMiscStream {
        stream_id: stream.streamID,
        surface_stream_id: stream.surfaceStreamID,
        objects,
    })
}

fn parse_misc_object(obj: &NvDsTargetMiscDataObject) -> Result<TargetMiscObject> {
    let n = obj.numObj as usize;
    let frames = if obj.list.is_null() || n == 0 {
        Vec::new()
    } else {
        let fr = unsafe { std::slice::from_raw_parts(obj.list, n) };
        fr.iter().map(parse_misc_frame).collect()
    };
    let label = c_label_to_string(&obj.objLabel);
    Ok(TargetMiscObject {
        unique_id: obj.uniqueId,
        class_id: obj.classId,
        label,
        frames,
    })
}

fn parse_misc_frame(frame: &NvDsTargetMiscDataFrame) -> TargetMiscFrame {
    TargetMiscFrame {
        frame_num: frame.frameNum,
        bbox_left: frame.tBbox.left,
        bbox_top: frame.tBbox.top,
        bbox_width: frame.tBbox.width,
        bbox_height: frame.tBbox.height,
        confidence: frame.confidence,
        age: frame.age,
        state: TrackState::from(frame.trackerState),
        visibility: frame.visibility,
    }
}
