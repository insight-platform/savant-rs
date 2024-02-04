use crate::capi::object::CAPI_BoundingBox;
use crate::primitives::frame::VideoFrame;
use crate::primitives::object::BorrowedVideoObject;
use crate::primitives::objects_view::VideoObjectsView;
use std::ffi::c_char;
use std::ptr::null_mut;

#[repr(C)]
pub struct CAPI_ObjectCreateSpecification {
    namespace: *const c_char,
    label: *const c_char,
    parent_id: i64,
    parent_id_defined: bool,
    bounding_box: CAPI_BoundingBox,
    resulting_object_id: i64,
}

#[no_mangle]
pub unsafe extern "C" fn savant_get_object_view_from_frame_handle(
    handle: usize,
) -> *mut VideoObjectsView {
    let frame = &*(handle as *const VideoFrame);
    let view = frame.get_all_objects();
    let view = Box::new(view);
    Box::into_raw(view)
}

#[no_mangle]
pub unsafe extern "C" fn savant_get_object_view_from_object_view_handle(
    handle: usize,
) -> *mut VideoObjectsView {
    let view = &*(handle as *const VideoObjectsView);
    let view = Box::new(view.clone());
    Box::into_raw(view)
}

#[no_mangle]
pub unsafe extern "C" fn savant_release_object_view(view: *mut VideoObjectsView) {
    let view = Box::from_raw(view);
    drop(view);
}

#[no_mangle]
pub unsafe extern "C" fn savant_get_view_object(
    view: *const VideoObjectsView,
    object_id: i64,
) -> *mut BorrowedVideoObject {
    let view = &*view;
    let object = view.inner.iter().find(|o| o.get_id() == object_id).cloned();
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
    _frame_handle: usize,
    _objects: *mut CAPI_ObjectCreateSpecification,
    _len: usize,
) {
}

#[cfg(test)]
mod tests {
    use crate::capi::frame::{
        savant_get_object_view_from_frame_handle, savant_get_object_view_from_object_view_handle,
        savant_release_object_view,
    };
    use crate::primitives::frame::VideoFrame;
    use savant_core::test::gen_frame;

    #[test]
    fn test_get_object_view_from_frame_handle() {
        let frame = VideoFrame(gen_frame());
        let handle = frame.memory_handle();
        unsafe {
            for _ in 0..1000_usize {
                let view = savant_get_object_view_from_frame_handle(handle);
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
                let view = savant_get_object_view_from_object_view_handle(handle);
                savant_release_object_view(view);
            }
        }
    }
}
