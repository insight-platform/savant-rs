use crate::capi::object::CAPIBoundingBox;
use crate::primitives::frame::VideoFrame;
use crate::primitives::object::BorrowedVideoObject;
use crate::primitives::objects_view::VideoObjectsView;
use savant_core::primitives::object::ObjectOperations;
use std::ffi::{c_char, CStr};
use std::ptr::null_mut;

#[repr(C)]
pub struct CAPIObjectCreateSpecification {
    namespace: *const c_char,
    label: *const c_char,
    confidence: f32,
    confidence_defined: bool,
    parent_id: i64,
    parent_id_defined: bool,
    detection_box_box: CAPIBoundingBox,
    tracking_id: i64,
    tracking_box: CAPIBoundingBox,
    tracking_id_defined: bool,
    resulting_object_id: i64,
}

#[no_mangle]
pub unsafe extern "C" fn savant_frame_from_handle(handle: usize) -> *mut VideoFrame {
    let frame = &*(handle as *const VideoFrame);
    let frame = Box::new(frame.clone());
    Box::into_raw(frame)
}

#[no_mangle]
pub unsafe extern "C" fn savant_release_frame(frame: *mut VideoFrame) {
    let frame = Box::from_raw(frame);
    drop(frame);
}

#[no_mangle]
pub unsafe extern "C" fn savant_frame_get_all_objects(
    frame: *const VideoFrame,
) -> *mut VideoObjectsView {
    if frame.is_null() {
        return null_mut();
    }
    let frame = &*frame;
    let view = frame.get_all_objects();
    let view = Box::new(view);
    Box::into_raw(view)
}

#[no_mangle]
pub unsafe extern "C" fn savant_object_view_from_handle(handle: usize) -> *mut VideoObjectsView {
    let view = &*(handle as *const VideoObjectsView);
    let view = Box::new(view.clone());
    Box::into_raw(view)
}

#[no_mangle]
pub unsafe extern "C" fn savant_release_object_view(view: *mut VideoObjectsView) {
    if view.is_null() {
        return;
    }
    let view = Box::from_raw(view);
    drop(view);
}

#[no_mangle]
pub unsafe extern "C" fn savant_frame_get_object(
    frame: *const VideoFrame,
    object_id: i64,
) -> *mut BorrowedVideoObject {
    if frame.is_null() {
        return null_mut();
    }
    let frame = &*frame;
    let object = frame.get_object(object_id);
    match object {
        Some(object) => {
            let object = Box::new(object);
            Box::into_raw(object)
        }
        None => null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn savant_frame_delete_objects_with_ids(
    frame: *mut VideoFrame,
    object_ids: *const i64,
    len: usize,
) {
    if frame.is_null() {
        return;
    }
    let frame = &mut *frame;
    let object_ids = std::slice::from_raw_parts(object_ids, len);
    frame.0.delete_objects_with_ids(object_ids);
}

#[no_mangle]
pub unsafe extern "C" fn savant_object_view_get_object(
    view: *const VideoObjectsView,
    object_id: i64,
) -> *mut BorrowedVideoObject {
    let view = &*view;
    let object = view.0.iter().find(|o| o.get_id() == object_id).cloned();
    match object {
        Some(object) => {
            let object = Box::new(object);
            Box::into_raw(object)
        }
        None => null_mut(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn savant_release_object(object: *mut BorrowedVideoObject) {
    if object.is_null() {
        return;
    }
    let object = Box::from_raw(object);
    drop(object);
}

#[no_mangle]
pub unsafe extern "C" fn savant_create_objects(
    frame: *mut VideoFrame,
    objects: *mut CAPIObjectCreateSpecification,
    len: usize,
) {
    if frame.is_null() {
        return;
    }
    let frame = &mut *frame;
    let objects = std::slice::from_raw_parts_mut(objects, len);
    for object in objects {
        let namespace = CStr::from_ptr(object.namespace)
            .to_str()
            .expect("Invalid namespace. Unable to convert to string.");

        let label = CStr::from_ptr(object.label)
            .to_str()
            .expect("Invalid label. Unable to convert to string.");

        let obj = frame
            .0
            .create_object(
                namespace,
                label,
                if object.parent_id_defined {
                    Some(object.parent_id)
                } else {
                    None
                },
                (&object.detection_box_box).into(),
                if object.confidence_defined {
                    Some(object.confidence)
                } else {
                    None
                },
                if object.tracking_id_defined {
                    Some(object.tracking_id)
                } else {
                    None
                },
                if object.tracking_id_defined {
                    Some((&object.tracking_box).into())
                } else {
                    None
                },
                vec![],
            )
            .expect("Failed to create object.");
        object.resulting_object_id = obj.get_id();
    }
}

#[cfg(test)]
mod tests {
    use crate::capi::frame::{
        savant_frame_get_all_objects, savant_object_view_from_handle, savant_release_object_view,
    };
    use crate::primitives::frame::VideoFrame;
    use savant_core::test::gen_frame;

    #[test]
    fn test_get_object_view_from_frame_handle() {
        let frame = VideoFrame(gen_frame());
        unsafe {
            for _ in 0..1000_usize {
                let view = savant_frame_get_all_objects(&frame);
                savant_release_object_view(view);
            }
        }
    }

    #[test]
    fn test_get_object_view_from_object_view_handle() {
        let frame = VideoFrame(gen_frame());
        let view = frame.get_all_objects();
        let handle = view.memory_handle();

        unsafe {
            for _ in 0..1000_usize {
                let view = savant_object_view_from_handle(handle);
                savant_release_object_view(view);
            }
        }
    }
}
