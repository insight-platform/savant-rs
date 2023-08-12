/// # C API for Savant Rust Library
///
/// This API is used to interface with the Savant Rust library from C.
///
use crate::primitives::message::video::object::objects_view::VideoObjectsView;
use crate::primitives::{
    IdCollisionResolutionPolicy, RBBox, VideoFrameProxy, VideoObjectBBoxType, VideoObjectProxy,
};
use crate::utils::symbol_mapper::{get_model_name, get_object_label};
use anyhow::bail;
use std::collections::HashMap;
use std::slice::from_raw_parts;

/// When BBox is not defined, its elements are set to this value.
pub const BBOX_ELEMENT_UNDEFINED: f64 = 1.797_693_134_862_315_7e308_f64;

#[derive(Clone, Debug)]
#[repr(C)]
pub struct VideoObjectInferenceMeta {
    pub id: i64,
    pub namespace_id: i64,
    pub label_id: i64,
    pub confidence: f32,
    pub track_id: i64,
    pub xc: f64,
    pub yc: f64,
    pub width: f64,
    pub height: f64,
    pub angle: f64,
}

pub fn from_object(
    o: &VideoObjectProxy,
    t: VideoObjectBBoxType,
) -> anyhow::Result<VideoObjectInferenceMeta> {
    let o = o.inner.read_recursive();
    let track_id = o.track_id;
    let track_box = o.track_box.as_ref();

    let bind = match t {
        VideoObjectBBoxType::Detection => Some(&o.detection_box),
        VideoObjectBBoxType::TrackingInfo => track_box,
    };

    if bind.is_none() {
        bail!("Requested BBox is not defined for object with id {}", o.id)
    }
    let bb = bind.unwrap();

    if bb.angle.unwrap_or(0.0) != 0.0 {
        bail!("Rotated bounding boxes cannot be passed to inference engine. You must orient them first.")
    }

    Ok(VideoObjectInferenceMeta {
        id: o.id,
        namespace_id: o.namespace_id.unwrap_or(i64::MAX),
        label_id: o.label_id.unwrap_or(i64::MAX),
        confidence: o.confidence.unwrap_or(0.0),
        track_id: track_id.unwrap_or(i64::MAX),
        xc: bb.xc,
        yc: bb.yc,
        width: bb.width,
        height: bb.height,
        angle: BBOX_ELEMENT_UNDEFINED,
    })
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
pub unsafe extern "C" fn get_inference_meta(
    handle: usize,
    pos: usize,
    t: VideoObjectBBoxType,
) -> VideoObjectInferenceMeta {
    let this = unsafe { &*(handle as *const VideoObjectsView) };
    from_object(&this.inner[pos], t).unwrap_or(VideoObjectInferenceMeta {
        id: i64::MAX,
        namespace_id: i64::MAX,
        label_id: i64::MAX,
        confidence: 0.0,
        track_id: i64::MAX,
        xc: BBOX_ELEMENT_UNDEFINED,
        yc: BBOX_ELEMENT_UNDEFINED,
        width: BBOX_ELEMENT_UNDEFINED,
        height: BBOX_ELEMENT_UNDEFINED,
        angle: BBOX_ELEMENT_UNDEFINED,
    })
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
    ffi_inference_meta: *const VideoObjectInferenceMeta,
    count: usize,
    t: VideoObjectBBoxType,
) {
    let inference_meta = unsafe { from_raw_parts(ffi_inference_meta, count) };
    let frame = unsafe { &*(frame_handle as *const VideoFrameProxy) };
    for m in inference_meta {
        let angle = if m.angle == BBOX_ELEMENT_UNDEFINED {
            None
        } else {
            Some(m.angle)
        };

        assert!(
            m.xc != BBOX_ELEMENT_UNDEFINED
                && m.yc != BBOX_ELEMENT_UNDEFINED
                && m.width != BBOX_ELEMENT_UNDEFINED
                && m.height != BBOX_ELEMENT_UNDEFINED,
            "Bounding box elements must be defined"
        );

        let bounding_box = RBBox::new(m.xc, m.yc, m.width, m.height, angle);

        match t {
            VideoObjectBBoxType::Detection => {
                let namespace = get_model_name(m.namespace_id);
                let label = get_object_label(m.namespace_id, m.label_id);

                if namespace.is_none() {
                    log::warn!(
                        "Model with id={} not found. Object {} will be ignored.",
                        m.namespace_id,
                        m.id
                    );
                    continue;
                }

                if label.is_none() {
                    log::warn!(
                        "Label with id={} not found. Object {} will be ignored.",
                        m.label_id,
                        m.id
                    );
                    continue;
                }

                frame
                    .add_object(
                        &VideoObjectProxy::new(
                            m.id,
                            namespace.unwrap(),
                            label.unwrap(),
                            bounding_box,
                            HashMap::default(),
                            Some(m.confidence),
                            None,
                            None,
                        ),
                        IdCollisionResolutionPolicy::GenerateNewId,
                    )
                    .unwrap_or_else(|e| {
                        panic!(
                            "Failed to add object with id={} to frame '{}'. Error is {}",
                            m.id,
                            frame.get_source_id(),
                            e
                        )
                    });
            }
            VideoObjectBBoxType::TrackingInfo => {
                // update currently existing objects
                assert!(
                    m.track_id != i64::MAX,
                    "When updating tracking information track id must be set"
                );

                let o = frame.get_object(m.id).unwrap_or_else(|| {
                    panic!(
                        "Object with Id={} not found on frame '{}'.",
                        m.id,
                        frame.get_source_id()
                    )
                });

                o.set_track_info(m.track_id, bounding_box);
            }
        }
    }
}
