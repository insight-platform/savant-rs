use crate::primitives::bbox::RBBox;
use crate::primitives::object::BorrowedVideoObject;
use crate::primitives::objects_view::VideoObjectsView;
use savant_core::primitives::attribute_value::{AttributeValue, AttributeValueVariant};
use savant_core::primitives::WithAttributes;
use std::cmp::min;
use std::ffi::{c_char, CStr};

#[repr(C)]
pub struct BoundingBox {
    pub xc: f32,
    pub yc: f32,
    pub width: f32,
    pub height: f32,
    pub angle: f32,
    pub oriented: bool,
}

#[repr(C)]
pub struct ObjectIds {
    pub id: i64,
    pub namespace_id: i64,
    pub label_id: i64,
    pub tracking_id: i64,

    pub namespace_id_set: bool,
    pub label_id_set: bool,
    pub tracking_id_set: bool,
}

#[no_mangle]
pub unsafe extern "C" fn object_view_get_handles(
    handle: usize,
    caller_allocated_handles: *mut usize,
    caller_allocated_max_handles: *mut usize,
) {
    if handle == 0 || caller_allocated_handles.is_null() || caller_allocated_max_handles.is_null() {
        panic!("Null pointer passed to object_view_get_handles");
    }
    unsafe {
        let object_view = &*(handle as *const VideoObjectsView);
        let handles = object_view.object_memory_handles();
        let handles_len = handles.len();
        let max_handles = *caller_allocated_max_handles;
        let len = min(handles_len, max_handles);
        let buf = std::slice::from_raw_parts_mut(caller_allocated_handles, len);
        buf[..len].copy_from_slice(&handles[..len]);
        *caller_allocated_max_handles = handles_len;
    }
}

#[no_mangle]
pub unsafe extern "C" fn object_get_ids(handle: usize) -> ObjectIds {
    if handle == 0 {
        panic!("Null pointer passed to object_get_id");
    }
    let object = &*(handle as *const BorrowedVideoObject);
    let id = object.get_id();
    let namespace_id = object.get_namespace_id();
    let label_id = object.get_label_id();
    let tracking_id = object.get_track_id();

    ObjectIds {
        id,
        namespace_id: namespace_id.unwrap_or(0),
        namespace_id_set: namespace_id.is_some(),
        label_id: label_id.unwrap_or(0),
        label_id_set: label_id.is_some(),
        tracking_id: tracking_id.unwrap_or(0),
        tracking_id_set: tracking_id.is_some(),
    }
}

#[no_mangle]
pub unsafe extern "C" fn object_get_confidence(handle: usize, conf: *mut f32) -> bool {
    if handle == 0 || conf.is_null() {
        panic!("Null pointer passed to object_get_confidence");
    }
    let object = &*(handle as *const BorrowedVideoObject);
    if let Some(c) = object.get_confidence() {
        *conf = c;
        true
    } else {
        false
    }
}

// set confidence
#[no_mangle]
pub unsafe extern "C" fn object_set_confidence(handle: usize, conf: f32) {
    if handle == 0 {
        panic!("Null pointer passed to object_set_confidence");
    }
    let object = &mut *(handle as *mut BorrowedVideoObject);
    object.set_confidence(Some(conf));
}

#[no_mangle]
pub unsafe extern "C" fn object_clear_confidence(handle: usize) {
    if handle == 0 {
        panic!("Null pointer passed to object_clear_confidence");
    }
    let object = &mut *(handle as *mut BorrowedVideoObject);
    object.set_confidence(None);
}

#[no_mangle]
pub unsafe extern "C" fn object_get_namespace(
    handle: usize,
    caller_allocated_buf: *mut c_char,
    len: usize,
) -> usize {
    if handle == 0 || caller_allocated_buf.is_null() {
        panic!("Null pointer passed to object_get_namespace");
    }
    let object = &*(handle as *const BorrowedVideoObject);
    let ns = object.get_namespace();
    let ns = ns.as_bytes();
    // copy ns to allocated_buf
    let ns_len = ns.len();
    let len = std::cmp::min(ns_len, len);
    let buf = unsafe { std::slice::from_raw_parts_mut(caller_allocated_buf as *mut u8, len) };
    // fill with ns
    buf[..len].copy_from_slice(&ns[..len]);
    ns_len
}

#[no_mangle]
pub unsafe extern "C" fn object_get_label(
    handle: usize,
    caller_allocated_buf: *mut c_char,
    len: usize,
) -> usize {
    if handle == 0 || caller_allocated_buf.is_null() {
        panic!("Null pointer passed to object_get_label");
    }
    let object = &*(handle as *const BorrowedVideoObject);
    let label = object.get_label();
    let label = label.as_bytes();
    // copy label to allocated_buf
    let label_len = label.len();
    let len = std::cmp::min(label_len, len);
    let buf = unsafe { std::slice::from_raw_parts_mut(caller_allocated_buf as *mut u8, len) };
    // fill with label
    buf[..len].copy_from_slice(&label[..len]);
    label_len
}

#[no_mangle]
pub unsafe extern "C" fn object_get_draw_label(
    handle: usize,
    caller_allocated_buf: *mut c_char,
    len: usize,
) -> usize {
    if handle == 0 || caller_allocated_buf.is_null() {
        panic!("Null pointer passed to object_get_draw_label");
    }
    let object = &*(handle as *const BorrowedVideoObject);
    let label = object.get_draw_label();
    let label = label.as_bytes();
    // copy label to allocated_buf
    let label_len = label.len();
    let len = std::cmp::min(label_len, len);
    let buf = unsafe { std::slice::from_raw_parts_mut(caller_allocated_buf as *mut u8, len) };
    // fill with label
    buf[..len].copy_from_slice(&label[..len]);
    label_len
}

#[no_mangle]
pub unsafe extern "C" fn object_get_detection_box(
    handle: usize,
    caller_allocated_bb: *mut BoundingBox,
) {
    if handle == 0 || caller_allocated_bb.is_null() {
        panic!("Null pointer passed to object_get_detection_box");
    }
    let object = &*(handle as *const BorrowedVideoObject);
    let bb = object.get_detection_box();
    let (xc, yc, width, height) = bb.as_xcycwh();
    let oriented = bb.get_angle().is_some();
    let angle = bb.get_angle().unwrap_or(0.0);
    *caller_allocated_bb = BoundingBox {
        xc,
        yc,
        width,
        height,
        angle,
        oriented,
    };
}

#[no_mangle]
pub unsafe extern "C" fn object_set_detection_box(handle: usize, bb: *const BoundingBox) {
    if handle == 0 || bb.is_null() {
        panic!("Null pointer passed to object_set_detection_box");
    }
    let object = &mut *(handle as *mut BorrowedVideoObject);
    let bb = &*bb;
    let bb = RBBox::new(
        bb.xc,
        bb.yc,
        bb.width,
        bb.height,
        if bb.oriented { Some(bb.angle) } else { None },
    );
    object.set_detection_box(bb);
}

#[no_mangle]
pub unsafe extern "C" fn object_get_tracking_info(
    handle: usize,
    caller_allocated_bb: *mut BoundingBox,
    caller_allocated_tracking_id: *mut i64,
) -> bool {
    if handle == 0 || caller_allocated_bb.is_null() || caller_allocated_tracking_id.is_null() {
        panic!("Null pointer passed to object_get_tracking_info");
    }

    let object = &*(handle as *const BorrowedVideoObject);
    let track_id = object.get_track_id();
    if track_id.is_none() {
        return false;
    }
    let track_box = object.get_track_box();
    if track_box.is_none() {
        return false;
    }
    let track_id = track_id.unwrap();
    let track_box = track_box.unwrap();
    let (xc, yc, width, height) = track_box.as_xcycwh();
    *caller_allocated_bb = BoundingBox {
        xc,
        yc,
        width,
        height,
        angle: track_box.get_angle().unwrap_or(0.0),
        oriented: track_box.get_angle().is_some(),
    };
    *caller_allocated_tracking_id = track_id;
    true
}

#[no_mangle]
pub unsafe extern "C" fn object_set_tracking_info(
    handle: usize,
    bb: *const BoundingBox,
    tracking_id: i64,
) {
    if handle == 0 || bb.is_null() {
        panic!("Null pointer passed to object_set_tracking_info");
    }

    let object = &mut *(handle as *mut BorrowedVideoObject);
    let track_box = &*bb;
    let track_box = RBBox::new(
        track_box.xc,
        track_box.yc,
        track_box.width,
        track_box.height,
        if track_box.oriented {
            Some(track_box.angle)
        } else {
            None
        },
    );
    object.set_track_id(Some(tracking_id));
    object.set_track_box(track_box);
}
#[no_mangle]
pub unsafe extern "C" fn object_clear_tracking_info(handle: usize) {
    if handle == 0 {
        panic!("Null pointer passed to object_clear_tracking_info");
    }

    let object = &mut *(handle as *mut BorrowedVideoObject);
    object.clear_track_info();
}

#[no_mangle]
pub unsafe extern "C" fn object_get_float_vec_attribute_value(
    handle: usize,
    namespace: *const c_char,
    name: *const c_char,
    value_index: usize,
    caller_allocated_result: *mut f64,
    caller_allocated_result_len: *mut usize,
    caller_allocated_confidence: *mut f32,
    caller_allocated_confidence_set: *mut bool,
) -> bool {
    unsafe {
        if handle == 0
            || caller_allocated_result.is_null()
            || caller_allocated_result_len.is_null()
            || caller_allocated_confidence.is_null()
            || caller_allocated_confidence_set.is_null()
            || namespace.is_null()
            || name.is_null()
        {
            panic!("Null pointer passed to object_get_float_vec_attribute_value");
        }

        if *caller_allocated_result_len < 1 {
            return false;
        }

        let object = &*(handle as *const BorrowedVideoObject);
        let namespace = CStr::from_ptr(namespace);
        let name = CStr::from_ptr(name);
        let namespace = namespace.to_str().unwrap();
        let name = name.to_str().unwrap();
        let attribute = object.get_attribute(namespace, name);
        if attribute.is_none() {
            return false;
        }

        let attribute = attribute.as_ref().unwrap();
        let attribute_values = attribute.0.get_values();

        if attribute_values.len() <= value_index {
            return false;
        }
        let val = &attribute_values[value_index];

        if let Some(conf) = val.confidence {
            *caller_allocated_confidence = conf;
            *caller_allocated_confidence_set = true;
        } else {
            *caller_allocated_confidence_set = false;
        }

        match &val.value {
            AttributeValueVariant::Float(f) => {
                *caller_allocated_result = *f;
                *caller_allocated_result_len = 1;
                true
            }
            AttributeValueVariant::FloatVector(f) => {
                if f.len() > *caller_allocated_result_len {
                    return false;
                }

                let buf = std::slice::from_raw_parts_mut(
                    caller_allocated_result,
                    *caller_allocated_result_len,
                );

                *caller_allocated_result_len = min(f.len(), *caller_allocated_result_len);

                // copy f to caller_allocated_result
                buf[..*caller_allocated_result_len]
                    .copy_from_slice(&f[..*caller_allocated_result_len]);

                true
            }
            _ => false,
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn object_set_float_vec_attribute_value(
    handle: usize,
    namespace: *const c_char,
    name: *const c_char,
    hint: *const c_char,
    values: *const f64,
    values_len: usize,
    confidence: *const f32,
    persistent: bool,
    hidden: bool,
) {
    unsafe {
        if handle == 0
            || namespace.is_null()
            || name.is_null()
            || values.is_null()
            || values_len == 0
        {
            panic!("Null pointer passed to object_set_float_vec_attribute_value");
        }

        let object = &mut *(handle as *mut BorrowedVideoObject);
        let namespace = CStr::from_ptr(namespace);
        let name = CStr::from_ptr(name);
        let hint = if hint.is_null() {
            None
        } else {
            Some(CStr::from_ptr(hint).to_str().unwrap().to_string())
        };
        let confidence = if confidence.is_null() {
            None
        } else {
            Some(*confidence)
        };

        let namespace = namespace.to_str().unwrap();
        let name = name.to_str().unwrap();
        let values = std::slice::from_raw_parts(values, values_len);
        let values = values.to_vec();
        let values = vec![AttributeValue::new(
            AttributeValueVariant::FloatVector(values),
            confidence,
        )];

        if persistent {
            object
                .0
                .set_persistent_attribute(namespace, name, &hint.as_deref(), hidden, values);
        } else {
            object
                .0
                .set_temporary_attribute(namespace, name, &hint.as_deref(), hidden, values);
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn object_get_int_vec_attribute_value(
    handle: usize,
    namespace: *const c_char,
    name: *const c_char,
    value_index: usize,
    caller_allocated_result: *mut i64,
    caller_allocated_result_len: *mut usize,
    caller_allocated_confidence: *mut f32,
    caller_allocated_confidence_set: *mut bool,
) -> bool {
    unsafe {
        if handle == 0
            || caller_allocated_result.is_null()
            || caller_allocated_result_len.is_null()
            || caller_allocated_confidence.is_null()
            || caller_allocated_confidence_set.is_null()
            || namespace.is_null()
            || name.is_null()
        {
            panic!("Null pointer passed to object_get_int_vec_attribute_value");
        }

        if *caller_allocated_result_len < 1 {
            return false;
        }

        let object = &*(handle as *const BorrowedVideoObject);
        let namespace = CStr::from_ptr(namespace);
        let name = CStr::from_ptr(name);
        let namespace = namespace.to_str().unwrap();
        let name = name.to_str().unwrap();
        let attribute = object.get_attribute(namespace, name);
        if attribute.is_none() {
            return false;
        }

        let attribute = attribute.as_ref().unwrap();
        let attribute_values = attribute.0.get_values();

        if attribute_values.len() <= value_index {
            return false;
        }
        let val = &attribute_values[value_index];

        if let Some(conf) = val.confidence {
            *caller_allocated_confidence = conf;
            *caller_allocated_confidence_set = true;
        } else {
            *caller_allocated_confidence_set = false;
        }

        match &val.value {
            AttributeValueVariant::Integer(i) => {
                *caller_allocated_result = *i;
                *caller_allocated_result_len = 1;
                true
            }
            AttributeValueVariant::IntegerVector(i) => {
                if i.len() > *caller_allocated_result_len {
                    return false;
                }

                let buf = std::slice::from_raw_parts_mut(
                    caller_allocated_result,
                    *caller_allocated_result_len,
                );

                *caller_allocated_result_len = min(i.len(), *caller_allocated_result_len);

                // copy i to caller_allocated_result
                buf[..*caller_allocated_result_len]
                    .copy_from_slice(&i[..*caller_allocated_result_len]);

                true
            }
            _ => false,
        }
    }
}
#[no_mangle]
pub unsafe extern "C" fn object_set_int_vec_attribute_value(
    handle: usize,
    namespace: *const c_char,
    name: *const c_char,
    hint: *const c_char,
    values: *const i64,
    values_len: usize,
    confidence: *const f32,
    persistent: bool,
    hidden: bool,
) {
    unsafe {
        if handle == 0
            || namespace.is_null()
            || name.is_null()
            || values.is_null()
            || values_len == 0
        {
            panic!("Null pointer passed to object_set_int_vec_attribute_value");
        }

        let object = &mut *(handle as *mut BorrowedVideoObject);
        let namespace = CStr::from_ptr(namespace);
        let name = CStr::from_ptr(name);
        let hint = if hint.is_null() {
            None
        } else {
            Some(CStr::from_ptr(hint).to_str().unwrap().to_string())
        };
        let confidence = if confidence.is_null() {
            None
        } else {
            Some(*confidence)
        };

        let namespace = namespace.to_str().unwrap();
        let name = name.to_str().unwrap();
        let values = std::slice::from_raw_parts(values, values_len);
        let values = values.to_vec();
        let values = vec![AttributeValue::new(
            AttributeValueVariant::IntegerVector(values),
            confidence,
        )];

        if persistent {
            object
                .0
                .set_persistent_attribute(namespace, name, &hint.as_deref(), hidden, values);
        } else {
            object
                .0
                .set_temporary_attribute(namespace, name, &hint.as_deref(), hidden, values);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::capi::object::{
        object_clear_confidence, object_clear_tracking_info, object_get_confidence,
        object_get_detection_box, object_get_draw_label, object_get_float_vec_attribute_value,
        object_get_ids, object_get_int_vec_attribute_value, object_get_label, object_get_namespace,
        object_get_tracking_info, object_set_confidence, object_set_detection_box,
        object_set_tracking_info, object_view_get_handles, BoundingBox,
    };
    use crate::primitives::frame::VideoFrame;
    use crate::primitives::object::BorrowedVideoObject;
    use savant_core::primitives::WithAttributes;
    use savant_core::test::gen_frame;
    use std::ffi::c_char;

    #[test]
    fn test_object_ops() {
        let f = gen_frame();
        let o = BorrowedVideoObject(f.get_object(1).unwrap());
        let ids = unsafe { object_get_ids(o.memory_handle()) };
        assert_eq!(ids.id, 1);
    }

    #[test]
    fn test_object_confidence() {
        let f = gen_frame();
        let o = BorrowedVideoObject(f.get_object(1).unwrap());
        unsafe { object_set_confidence(o.memory_handle(), 0.5) };
        let mut conf = 0.0;
        let res = unsafe { object_get_confidence(o.memory_handle(), &mut conf) };
        assert!(res);
        assert_eq!(conf, 0.5);
        unsafe { object_clear_confidence(o.memory_handle()) };
        let res = unsafe { object_get_confidence(o.memory_handle(), &mut conf) };
        assert!(!res);
    }

    #[test]
    fn test_object_namespace() {
        let f = gen_frame();
        let o = BorrowedVideoObject(f.get_object(1).unwrap());
        let result_buf = vec![0u8; 100];
        let result_buf = result_buf.as_ptr() as *mut i8;
        let ns = unsafe { object_get_namespace(o.memory_handle(), result_buf, 100) };
        let ns = unsafe { std::slice::from_raw_parts(result_buf as *const u8, ns) };
        let ns = std::str::from_utf8(ns).unwrap();
        assert_eq!(ns, "test2");
    }

    #[test]
    fn test_object_label() {
        let f = gen_frame();
        let o = BorrowedVideoObject(f.get_object(1).unwrap());
        let result_buf = vec![0u8; 100];
        let result_buf = result_buf.as_ptr() as *mut i8;
        let label = unsafe { object_get_label(o.memory_handle(), result_buf, 100) };
        let label = unsafe { std::slice::from_raw_parts(result_buf as *const u8, label) };
        let label = std::str::from_utf8(label).unwrap();
        assert_eq!(label, "test");
    }

    #[test]
    fn test_object_draw_label() {
        let f = gen_frame();
        let o = BorrowedVideoObject(f.get_object(1).unwrap());
        {
            let result_buf = vec![0u8; 100];
            let result_buf = result_buf.as_ptr() as *mut i8;
            let label = unsafe { object_get_draw_label(o.memory_handle(), result_buf, 100) };
            let label = unsafe { std::slice::from_raw_parts(result_buf as *const u8, label) };
            let label = std::str::from_utf8(label).unwrap();
            assert_eq!(label, "test");
        }
    }

    #[test]
    fn test_object_detection_box() {
        let f = gen_frame();
        let o = BorrowedVideoObject(f.get_object(1).unwrap());
        let mut bb = BoundingBox {
            xc: 0.0,
            yc: 0.0,
            width: 0.0,
            height: 0.0,
            angle: 0.0,
            oriented: true,
        };
        unsafe { object_get_detection_box(o.memory_handle(), &mut bb) };
        assert!(!bb.oriented);
        assert_eq!(bb.xc, 0.0);
        assert_eq!(bb.yc, 0.0);
        assert_eq!(bb.width, 0.0);
        assert_eq!(bb.height, 0.0);
        assert_eq!(bb.angle, 0.0);

        let bb = BoundingBox {
            xc: 1.0,
            yc: 2.0,
            width: 3.0,
            height: 4.0,
            angle: 5.0,
            oriented: true,
        };
        unsafe { object_set_detection_box(o.memory_handle(), &bb) };
        let mut bb = BoundingBox {
            xc: 0.0,
            yc: 0.0,
            width: 0.0,
            height: 0.0,
            angle: 0.0,
            oriented: false,
        };
        unsafe { object_get_detection_box(o.memory_handle(), &mut bb) };
        assert!(bb.oriented);
        assert_eq!(bb.xc, 1.0);
        assert_eq!(bb.yc, 2.0);
        assert_eq!(bb.width, 3.0);
        assert_eq!(bb.height, 4.0);
        assert_eq!(bb.angle, 5.0);
    }

    #[test]
    fn test_tracking_info() {
        let f = gen_frame();
        let o = BorrowedVideoObject(f.get_object(1).unwrap());
        let mut bb = BoundingBox {
            xc: 0.0,
            yc: 0.0,
            width: 0.0,
            height: 0.0,
            angle: 0.0,
            oriented: false,
        };
        let mut track_id = 0;
        let res = unsafe { object_get_tracking_info(o.memory_handle(), &mut bb, &mut track_id) };
        assert!(!res);

        let bb = BoundingBox {
            xc: 1.0,
            yc: 2.0,
            width: 3.0,
            height: 4.0,
            angle: 5.0,
            oriented: true,
        };
        unsafe { object_set_tracking_info(o.memory_handle(), &bb, 1) };
        let mut bb = BoundingBox {
            xc: 0.0,
            yc: 0.0,
            width: 0.0,
            height: 0.0,
            angle: 0.0,
            oriented: false,
        };
        let mut track_id = 0;
        let res = unsafe { object_get_tracking_info(o.memory_handle(), &mut bb, &mut track_id) };
        assert!(res);
        assert_eq!(track_id, 1);
        assert!(bb.oriented);
        assert_eq!(bb.xc, 1.0);
        assert_eq!(bb.yc, 2.0);
        assert_eq!(bb.width, 3.0);
        assert_eq!(bb.height, 4.0);
        assert_eq!(bb.angle, 5.0);

        unsafe { object_clear_tracking_info(o.memory_handle()) };
        let mut bb = BoundingBox {
            xc: 0.0,
            yc: 0.0,
            width: 0.0,
            height: 0.0,
            angle: 0.0,
            oriented: false,
        };
        let mut track_id = 0;
        let res = unsafe { object_get_tracking_info(o.memory_handle(), &mut bb, &mut track_id) };
        assert!(!res);
    }

    #[test]
    fn test_float_vec_attribute_value() {
        let f = gen_frame();
        let mut o = BorrowedVideoObject(f.get_object(1).unwrap());
        o.0.set_persistent_attribute(
            "test",
            "test",
            &None,
            false,
            vec![
                savant_core::primitives::attribute_value::AttributeValue::float_vector(
                    vec![1.0, 2.0, 3.0],
                    None,
                ),
                savant_core::primitives::attribute_value::AttributeValue::float(1.0, Some(0.5)),
            ],
        );
        {
            // access scalar attribute (index = 1)
            let mut result = 0.0;
            let mut result_len = 1;
            let mut confidence = 0.0;
            let mut confidence_set = false;
            let res = unsafe {
                object_get_float_vec_attribute_value(
                    o.memory_handle(),
                    "test".as_ptr() as *const c_char,
                    "test".as_ptr() as *const c_char,
                    1,
                    &mut result,
                    &mut result_len,
                    &mut confidence,
                    &mut confidence_set,
                )
            };

            assert!(res);
            assert_eq!(result, 1.0);
            assert_eq!(result_len, 1);
            assert_eq!(confidence, 0.5);
            assert!(confidence_set);
        }
        {
            // access vector attribute (index = 0)
            let mut result = vec![0.0; 3];
            let mut result_len = 3;
            let mut confidence = 1.0;
            let mut confidence_set = true;
            let res = unsafe {
                object_get_float_vec_attribute_value(
                    o.memory_handle(),
                    "test".as_ptr() as *const c_char,
                    "test".as_ptr() as *const c_char,
                    0,
                    result.as_mut_ptr(),
                    &mut result_len,
                    &mut confidence,
                    &mut confidence_set,
                )
            };

            assert!(res);
            assert_eq!(result, vec![1.0, 2.0, 3.0]);
            assert_eq!(result_len, 3);
            assert!(!confidence_set);
        }
    }

    #[test]
    fn test_int_vec_attribute_value() {
        let f = gen_frame();
        let mut o = BorrowedVideoObject(f.get_object(1).unwrap());
        o.0.set_persistent_attribute(
            "test",
            "test",
            &None,
            false,
            vec![
                savant_core::primitives::attribute_value::AttributeValue::integer_vector(
                    vec![1, 2, 3],
                    None,
                ),
                savant_core::primitives::attribute_value::AttributeValue::integer(1, Some(0.5)),
            ],
        );
        {
            // access scalar attribute (index = 1)
            let mut result = 0;
            let mut result_len = 1;
            let mut confidence = 0.0;
            let mut confidence_set = false;
            let res = unsafe {
                object_get_int_vec_attribute_value(
                    o.memory_handle(),
                    "test".as_ptr() as *const c_char,
                    "test".as_ptr() as *const c_char,
                    1,
                    &mut result,
                    &mut result_len,
                    &mut confidence,
                    &mut confidence_set,
                )
            };

            assert!(res);
            assert_eq!(result, 1);
            assert_eq!(result_len, 1);
            assert_eq!(confidence, 0.5);
            assert!(confidence_set);
        }
        {
            // access vector attribute (index = 0)
            let mut result = vec![0; 3];
            let mut result_len = 3;
            let mut confidence = 1.0;
            let mut confidence_set = true;
            let res = unsafe {
                object_get_int_vec_attribute_value(
                    o.memory_handle(),
                    "test".as_ptr() as *const c_char,
                    "test".as_ptr() as *const c_char,
                    0,
                    result.as_mut_ptr(),
                    &mut result_len,
                    &mut confidence,
                    &mut confidence_set,
                )
            };

            assert!(res);
            assert_eq!(result, vec![1, 2, 3]);
            assert_eq!(result_len, 3);
            assert!(!confidence_set);
        }
    }

    #[test]
    fn test_memory_view_accessor() {
        let frame = gen_frame();
        let frame = VideoFrame(frame);
        let objects = frame.get_all_objects();
        let handlers = objects.object_memory_handles();

        unsafe {
            let mut c_handlers = vec![0usize; 10];
            let mut max_handlers = handlers.len();
            object_view_get_handles(
                objects.memory_handle(),
                c_handlers.as_mut_ptr(),
                &mut max_handlers,
            );
            assert_eq!(max_handlers, handlers.len());
            assert_eq!(c_handlers[..max_handlers], handlers[..max_handlers]);
        }
    }
}
