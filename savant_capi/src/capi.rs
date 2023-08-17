// use anyhow::{bail, Result};
// use savant_core::consts::BBOX_ELEMENT_UNDEFINED;
// use savant_core::primitives::frame::VideoFrameProxy;
// use savant_core::primitives::object::{
//     IdCollisionResolutionPolicy, VideoObjectBBoxType, VideoObjectProxy,
// };
// use savant_core::primitives::RBBox;
// use savant_core::symbol_mapper::{get_model_name, get_object_label};
/// # C API for Savant Rust Library
///
/// This API is used to interface with the Savant Rust library from C.
///
///
// use std::collections::HashMap;
use std::ffi::{c_char, CStr};

/// # Safety
///
/// The function is intended for invocation from C/C++, so it is unsafe by design.
#[no_mangle]
pub unsafe extern "C" fn check_version(external_version: *const c_char) -> bool {
    let external_version = CStr::from_ptr(external_version);
    savant_core::version()
        == *external_version.to_str().expect(
            "Failed to convert external version to string. This is a bug. Please report it.",
        )
}

#[cfg(test)]
mod tests {
    use std::ffi::c_char;

    #[test]
    fn test_check_version() {
        unsafe {
            assert!(crate::capi::check_version(
                savant_core::version().as_ptr() as *const c_char
            ));
        }
    }
}

// #[derive(Clone, Debug)]
// #[repr(C)]
// pub struct VideoObjectInferenceMeta {
//     pub id: i64,
//     pub namespace_id: i64,
//     pub label_id: i64,
//     pub confidence: f32,
//     pub track_id: i64,
//     pub xc: f32,
//     pub yc: f32,
//     pub width: f32,
//     pub height: f32,
//     pub angle: f32,
// }
//
// pub fn from_object(
//     o: &VideoObjectProxy,
//     t: VideoObjectBBoxType,
// ) -> Result<VideoObjectInferenceMeta> {
//     // let o = o.inner.read_recursive();
//     // let track_id = o.track_id;
//     // let track_box = o.tra;
//
//     let bind = match t {
//         VideoObjectBBoxType::Detection => Some(o.get_detection_box()),
//         VideoObjectBBoxType::TrackingInfo => o.get_track_box(),
//     };
//
//     if bind.is_none() {
//         bail!(
//             "Requested BBox is not defined for object with id {}",
//             o.get_id()
//         )
//     }
//
//     let bb = bind.unwrap();
//
//     if bb.get_angle().unwrap_or(0.0) != 0.0 {
//         bail!("Rotated bounding boxes cannot be passed to inference engine. You must orient them first.")
//     }
//
//     Ok(VideoObjectInferenceMeta {
//         id: o.get_id(),
//         namespace_id: o.get_namespace_id().unwrap_or(i64::MAX),
//         label_id: o.get_label_id().unwrap_or(i64::MAX),
//         confidence: o.get_confidence().unwrap_or(0.0),
//         track_id: o.get_track_id().unwrap_or(i64::MAX),
//         xc: bb.get_xc(),
//         yc: bb.get_yc(),
//         width: bb.get_width(),
//         height: bb.get_height(),
//         angle: BBOX_ELEMENT_UNDEFINED,
//     })
// }
//
// // Returns the object vector length
// //
// // # Safety
// //
// // This function is unsafe because it dereferences a raw pointer
// //
// // #[no_mangle]
// // pub unsafe extern "C" fn object_vector_len(handle: usize) -> usize {
// //     let this = unsafe { &*(handle as *const VideoObjectsView) };
// //     this.inner.len()
// // }
//
// // /// Returns the object data casted to InferenceObjectMeta by index
// // ///
// // /// # Safety
// // ///
// // /// This function is unsafe because it dereferences a raw pointer
// // ///
// // #[no_mangle]
// // pub unsafe extern "C" fn get_inference_meta(
// //     handle: usize,
// //     pos: usize,
// //     t: VideoObjectBBoxType,
// // ) -> VideoObjectInferenceMeta {
// //     let this = unsafe { &*(handle as *const VideoObjectsView) };
// //     from_object(&this.inner[pos], t).unwrap_or(VideoObjectInferenceMeta {
// //         id: i64::MAX,
// //         namespace_id: i64::MAX,
// //         label_id: i64::MAX,
// //         confidence: 0.0,
// //         track_id: i64::MAX,
// //         xc: BBOX_ELEMENT_UNDEFINED,
// //         yc: BBOX_ELEMENT_UNDEFINED,
// //         width: BBOX_ELEMENT_UNDEFINED,
// //         height: BBOX_ELEMENT_UNDEFINED,
// //         angle: BBOX_ELEMENT_UNDEFINED,
// //     })
// // }
//
// /// Updates frame meta from inference meta
// ///
// /// # Safety
// ///
// /// This function is unsafe because it transforms raw pointer to VideoFrame
// ///
// #[no_mangle]
// pub unsafe extern "C" fn update_frame_meta(
//     frame_handle: usize,
//     ffi_inference_meta: *const VideoObjectInferenceMeta,
//     count: usize,
//     t: VideoObjectBBoxType,
// ) {
//     let inference_meta = unsafe { from_raw_parts(ffi_inference_meta, count) };
//     let frame = unsafe { &*(frame_handle as *const VideoFrameProxy) };
//     for m in inference_meta {
//         let angle = if m.angle == BBOX_ELEMENT_UNDEFINED {
//             None
//         } else {
//             Some(m.angle)
//         };
//
//         assert!(
//             m.xc != BBOX_ELEMENT_UNDEFINED
//                 && m.yc != BBOX_ELEMENT_UNDEFINED
//                 && m.width != BBOX_ELEMENT_UNDEFINED
//                 && m.height != BBOX_ELEMENT_UNDEFINED,
//             "Bounding box elements must be defined"
//         );
//
//         let bounding_box = RBBox::new(m.xc, m.yc, m.width, m.height, angle);
//
//         match t {
//             VideoObjectBBoxType::Detection => {
//                 let namespace = get_model_name(m.namespace_id);
//                 let label = get_object_label(m.namespace_id, m.label_id);
//
//                 if namespace.is_none() {
//                     log::warn!(
//                         "Model with id={} not found. Object {} will be ignored.",
//                         m.namespace_id,
//                         m.id
//                     );
//                     continue;
//                 }
//
//                 if label.is_none() {
//                     log::warn!(
//                         "Label with id={} not found. Object {} will be ignored.",
//                         m.label_id,
//                         m.id
//                     );
//                     continue;
//                 }
//
//                 frame
//                     .add_object(
//                         &VideoObjectProxy::new(
//                             m.id,
//                             namespace.unwrap(),
//                             label.unwrap(),
//                             bounding_box,
//                             HashMap::default(),
//                             Some(m.confidence),
//                             None,
//                             None,
//                         ),
//                         IdCollisionResolutionPolicy::GenerateNewId,
//                     )
//                     .unwrap_or_else(|e| {
//                         panic!(
//                             "Failed to add object with id={} to frame '{}'. Error is {}",
//                             m.id,
//                             frame.get_source_id(),
//                             e
//                         )
//                     });
//             }
//             VideoObjectBBoxType::TrackingInfo => {
//                 // update currently existing objects
//                 assert!(
//                     m.track_id != i64::MAX,
//                     "When updating tracking information track id must be set"
//                 );
//
//                 let o = frame.get_object(m.id).unwrap_or_else(|| {
//                     panic!(
//                         "Object with Id={} not found on frame '{}'.",
//                         m.id,
//                         frame.get_source_id()
//                     )
//                 });
//
//                 o.set_track_info(m.track_id, bounding_box);
//             }
//         }
//     }
// }
