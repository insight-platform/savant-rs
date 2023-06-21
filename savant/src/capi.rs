/// # C API for Savant Rust Library
///
/// This API is used to interface with the Savant Rust library from C.
///
use crate::primitives::message::video::object::objects_view::VideoObjectsView;
use crate::primitives::{VideoFrameProxy, VideoObjectProxy};
use std::slice::from_raw_parts;

/// When BBox is not defined, its elements are set to this value.
pub const BBOX_ELEMENT_UNDEFINED: f64 = 1.797_693_134_862_315_7e308_f64;

#[derive(Clone, Debug)]
#[repr(C)]
pub struct InferenceObjectMeta {
    pub id: i64,
    pub creator_id: i64,
    pub label_id: i64,
    pub confidence: f64,
    pub parent_id: i64,
    pub box_xc: f64,
    pub box_yx: f64,
    pub box_width: f64,
    pub box_height: f64,
    pub box_angle: f64,
    pub track_id: i64,
    pub track_box_xc: f64,
    pub track_box_yx: f64,
    pub track_box_width: f64,
    pub track_box_height: f64,
    pub track_box_angle: f64,
}

impl From<&VideoObjectProxy> for InferenceObjectMeta {
    fn from(o: &VideoObjectProxy) -> Self {
        let o = o.inner.read_recursive();
        let track_info = o.track_info.as_ref();
        Self {
            id: o.id,
            creator_id: o.creator_id.unwrap_or(i64::MAX),
            label_id: o.label_id.unwrap_or(i64::MAX),
            confidence: o.confidence.unwrap_or(f64::MAX),
            parent_id: o.parent_id.unwrap_or(i64::MAX),
            box_xc: o.bbox.get_xc(),
            box_yx: o.bbox.get_yc(),
            box_width: o.bbox.get_width(),
            box_height: o.bbox.get_height(),
            box_angle: o.bbox.get_angle().unwrap_or(0.0),
            track_id: track_info.map(|ti| ti.id).unwrap_or(i64::MAX),
            track_box_xc: track_info
                .map(|ti| ti.bounding_box.get_xc())
                .unwrap_or(f64::MAX),
            track_box_yx: track_info
                .map(|ti| ti.bounding_box.get_yc())
                .unwrap_or(f64::MAX),
            track_box_width: track_info
                .map(|ti| ti.bounding_box.get_width())
                .unwrap_or(f64::MAX),
            track_box_height: track_info
                .map(|ti| ti.bounding_box.get_height())
                .unwrap_or(f64::MAX),
            track_box_angle: track_info
                .map(|ti| ti.bounding_box.get_angle().unwrap_or(0.0))
                .unwrap_or(0.0),
        }
    }
}

/// Returns the object vector length
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer
///
#[no_mangle]
pub unsafe extern "C" fn object_vector_len(handle: usize) -> usize {
    let this = unsafe { &*(handle as *const VideoObjectsView) };
    this.inner.len()
}

/// Returns the object data casted to InferenceObjectMeta by index
///
/// # Safety
///
/// This function is unsafe because it dereferences a raw pointer
///
#[no_mangle]
pub unsafe extern "C" fn get_inference_meta(handle: usize, pos: usize) -> InferenceObjectMeta {
    let this = unsafe { &*(handle as *const VideoObjectsView) };
    (&this.inner[pos]).into()
}

/// Updates frame meta from inference meta
///
/// # Safety
///
/// This function is unsafe because it transforms raw pointer to VideoFrame
///
#[no_mangle]
pub unsafe extern "C" fn update_frame_meta(
    frame_handle: usize,
    ffi_inference_meta: *const InferenceObjectMeta,
    count: usize,
) {
    let inference_meta = unsafe { from_raw_parts(ffi_inference_meta, count) };
    let frame = unsafe { &*(frame_handle as *const VideoFrameProxy) };
    for m in inference_meta {
        frame
            .update_from_inference_meta(m)
            .expect("Unable to update frame meta from inference meta.");
    }
}
