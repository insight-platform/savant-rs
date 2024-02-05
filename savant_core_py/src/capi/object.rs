use std::cmp::min;
use std::ffi::{c_char, CStr};

use savant_core::primitives::attribute_value::{AttributeValue, AttributeValueVariant};
use savant_core::primitives::object::ObjectOperations;
use savant_core::primitives::{RBBox, WithAttributes};

use crate::primitives::object::BorrowedVideoObject;

#[repr(C)]
pub struct CAPIBoundingBox {
    pub xc: f32,
    pub yc: f32,
    pub width: f32,
    pub height: f32,
    pub angle: f32,
    pub oriented: bool,
}

impl From<&CAPIBoundingBox> for RBBox {
    fn from(bb: &CAPIBoundingBox) -> Self {
        RBBox::new(
            bb.xc,
            bb.yc,
            bb.width,
            bb.height,
            if bb.oriented { Some(bb.angle) } else { None },
        )
    }
}

#[repr(C)]
pub struct CAPI_ObjectIds {
    pub id: i64,
    pub namespace_id: i64,
    pub label_id: i64,
    pub tracking_id: i64,

    pub namespace_id_set: bool,
    pub label_id_set: bool,
    pub tracking_id_set: bool,
}

/// # Safety
///
/// The function is intended for invocation from C/C++, so it is unsafe by design.
#[no_mangle]
pub unsafe extern "C" fn savant_get_borrowed_object_from_handle(
    handle: usize,
) -> *mut BorrowedVideoObject {
    let object = &*(handle as *const BorrowedVideoObject);
    let object = Box::new(object.clone());
    Box::into_raw(object)
}

/// # Safety
///
/// The function is intended for invocation from C/C++, so it is unsafe by design.
#[no_mangle]
pub unsafe extern "C" fn savant_release_borrowed_object(object: *mut BorrowedVideoObject) {
    if object.is_null() {
        return;
    }
    let o = Box::from_raw(object);
    drop(o);
}

/// # Safety
///
/// The function is intended for invocation from C/C++, so it is unsafe by design.
#[no_mangle]
pub unsafe extern "C" fn savant_object_get_ids(
    object: *const BorrowedVideoObject,
) -> CAPI_ObjectIds {
    if object.is_null() {
        panic!("Null pointer passed to object_get_id");
    }
    let object = &*object;
    let id = object.get_id();
    let namespace_id = object.get_namespace_id();
    let label_id = object.get_label_id();
    let tracking_id = object.get_track_id();

    CAPI_ObjectIds {
        id,
        namespace_id: namespace_id.unwrap_or(0),
        namespace_id_set: namespace_id.is_some(),
        label_id: label_id.unwrap_or(0),
        label_id_set: label_id.is_some(),
        tracking_id: tracking_id.unwrap_or(0),
        tracking_id_set: tracking_id.is_some(),
    }
}

/// # Safety
///
/// The function is intended for invocation from C/C++, so it is unsafe by design.
#[no_mangle]
pub unsafe extern "C" fn savant_object_get_confidence(
    object: *const BorrowedVideoObject,
    conf: *mut f32,
) -> bool {
    if object.is_null() || conf.is_null() {
        panic!("Null pointer passed to object_get_confidence");
    }
    let object = &*object;
    if let Some(c) = object.get_confidence() {
        *conf = c;
        true
    } else {
        false
    }
}

// set confidence
/// # Safety
///
/// The function is intended for invocation from C/C++, so it is unsafe by design.
#[no_mangle]
pub unsafe extern "C" fn savant_object_set_confidence(object: *mut BorrowedVideoObject, conf: f32) {
    if object.is_null() {
        panic!("Null pointer passed to object_set_confidence");
    }
    let object = &mut *object;
    object.set_confidence(Some(conf));
}

/// # Safety
///
/// The function is intended for invocation from C/C++, so it is unsafe by design.
#[no_mangle]
pub unsafe extern "C" fn savant_object_clear_confidence(object: *mut BorrowedVideoObject) {
    if object.is_null() {
        panic!("Null pointer passed to object_clear_confidence");
    }
    let object = &mut *object;
    object.set_confidence(None);
}

/// # Safety
///
/// The function is intended for invocation from C/C++, so it is unsafe by design.
#[no_mangle]
pub unsafe extern "C" fn savant_object_get_namespace(
    object: *const BorrowedVideoObject,
    caller_allocated_buf: *mut c_char,
    len: usize,
) -> usize {
    if object.is_null() || caller_allocated_buf.is_null() {
        panic!("Null pointer passed to object_get_namespace");
    }
    let object = &*object;
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

/// # Safety
///
/// The function is intended for invocation from C/C++, so it is unsafe by design.
#[no_mangle]
pub unsafe extern "C" fn savant_object_get_label(
    object: *const BorrowedVideoObject,
    caller_allocated_buf: *mut c_char,
    len: usize,
) -> usize {
    if object.is_null() || caller_allocated_buf.is_null() {
        panic!("Null pointer passed to object_get_label");
    }
    let object = &*object;
    let label = object.get_label();
    let label = label.as_bytes();
    // copy label to allocated_buf
    let label_len = label.len();
    let len = min(label_len, len);
    let buf = unsafe { std::slice::from_raw_parts_mut(caller_allocated_buf as *mut u8, len) };
    // fill with label
    buf[..len].copy_from_slice(&label[..len]);
    label_len
}

/// # Safety
///
/// The function is intended for invocation from C/C++, so it is unsafe by design.
#[no_mangle]
pub unsafe extern "C" fn savant_object_get_draw_label(
    object: *const BorrowedVideoObject,
    caller_allocated_buf: *mut c_char,
    len: usize,
) -> usize {
    if object.is_null() || caller_allocated_buf.is_null() {
        panic!("Null pointer passed to object_get_draw_label");
    }
    let object = &*object;
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

/// # Safety
///
/// The function is intended for invocation from C/C++, so it is unsafe by design.
#[no_mangle]
pub unsafe extern "C" fn savant_object_get_detection_box(
    object: *const BorrowedVideoObject,
    caller_allocated_bb: *mut CAPIBoundingBox,
) {
    if object.is_null() || caller_allocated_bb.is_null() {
        panic!("Null pointer passed to object_get_detection_box");
    }
    let object = &*object;
    let bb = object.get_detection_box();
    let (xc, yc, width, height) = bb.as_xcycwh();
    let oriented = bb.get_angle().is_some();
    let angle = bb.get_angle().unwrap_or(0.0);
    *caller_allocated_bb = CAPIBoundingBox {
        xc,
        yc,
        width,
        height,
        angle,
        oriented,
    };
}

/// # Safety
///
/// The function is intended for invocation from C/C++, so it is unsafe by design.
#[no_mangle]
pub unsafe extern "C" fn savant_object_set_detection_box(
    object: *mut BorrowedVideoObject,
    bb: *const CAPIBoundingBox,
) {
    if object.is_null() || bb.is_null() {
        panic!("Null pointer passed to object_set_detection_box");
    }
    let object = &mut *object;
    let bb = &*bb;
    let bb = RBBox::new(
        bb.xc,
        bb.yc,
        bb.width,
        bb.height,
        if bb.oriented { Some(bb.angle) } else { None },
    );
    object.0.set_detection_box(bb);
}

/// # Safety
///
/// The function is intended for invocation from C/C++, so it is unsafe by design.
#[no_mangle]
pub unsafe extern "C" fn savant_object_get_tracking_info(
    object: *const BorrowedVideoObject,
    caller_allocated_bb: *mut CAPIBoundingBox,
    caller_allocated_tracking_id: *mut i64,
) -> bool {
    if object.is_null() || caller_allocated_bb.is_null() || caller_allocated_tracking_id.is_null() {
        panic!("Null pointer passed to object_get_tracking_info");
    }

    let object = &*object;
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
    *caller_allocated_bb = CAPIBoundingBox {
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

/// # Safety
///
/// The function is intended for invocation from C/C++, so it is unsafe by design.
#[no_mangle]
pub unsafe extern "C" fn savant_object_set_tracking_info(
    object: *mut BorrowedVideoObject,
    bb: *const CAPIBoundingBox,
    tracking_id: i64,
) {
    if object.is_null() || bb.is_null() {
        panic!("Null pointer passed to object_set_tracking_info");
    }

    let object = &mut *object;
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
    object.0.set_track_id(Some(tracking_id));
    object.0.set_track_box(track_box);
}
/// # Safety
///
/// The function is intended for invocation from C/C++, so it is unsafe by design.
#[no_mangle]
pub unsafe extern "C" fn savant_object_clear_tracking_info(object: *mut BorrowedVideoObject) {
    if object.is_null() {
        panic!("Null pointer passed to object_clear_tracking_info");
    }

    let object = &mut *object;
    object.clear_track_info();
}

/// # Safety
///
/// The function is intended for invocation from C/C++, so it is unsafe by design.
#[no_mangle]
pub unsafe extern "C" fn savant_object_get_float_vec_attribute_value(
    object: *const BorrowedVideoObject,
    namespace: *const c_char,
    name: *const c_char,
    value_index: usize,
    caller_allocated_result: *mut f64,
    caller_allocated_result_len: *mut usize,
    caller_allocated_confidence: *mut f32,
    caller_allocated_confidence_set: *mut bool,
) -> bool {
    unsafe {
        if object.is_null()
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

        let object = &*object;
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

/// # Safety
///
/// The function is intended for invocation from C/C++, so it is unsafe by design.
#[no_mangle]
pub unsafe extern "C" fn savant_object_set_float_vec_attribute_value(
    object: *mut BorrowedVideoObject,
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
        if object.is_null()
            || namespace.is_null()
            || name.is_null()
            || values.is_null()
            || values_len == 0
        {
            panic!("Null pointer passed to object_set_float_vec_attribute_value");
        }

        let object = &mut *object;
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

/// # Safety
///
/// The function is intended for invocation from C/C++, so it is unsafe by design.
#[no_mangle]
pub unsafe extern "C" fn savant_object_get_int_vec_attribute_value(
    object: *const BorrowedVideoObject,
    namespace: *const c_char,
    name: *const c_char,
    value_index: usize,
    caller_allocated_result: *mut i64,
    caller_allocated_result_len: *mut usize,
    caller_allocated_confidence: *mut f32,
    caller_allocated_confidence_set: *mut bool,
) -> bool {
    unsafe {
        if object.is_null()
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

        let object = &*object;
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
/// # Safety
///
/// The function is intended for invocation from C/C++, so it is unsafe by design.
#[no_mangle]
pub unsafe extern "C" fn savant_object_set_int_vec_attribute_value(
    object: *mut BorrowedVideoObject,
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
        if object.is_null()
            || namespace.is_null()
            || name.is_null()
            || values.is_null()
            || values_len == 0
        {
            panic!("Null pointer passed to object_set_int_vec_attribute_value");
        }

        let object = &mut *object;
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
    use std::ffi::CString;

    use savant_core::primitives::WithAttributes;
    use savant_core::test::gen_frame;

    use crate::capi::object::{
        savant_object_clear_confidence, savant_object_clear_tracking_info,
        savant_object_get_confidence, savant_object_get_detection_box,
        savant_object_get_draw_label, savant_object_get_float_vec_attribute_value,
        savant_object_get_ids, savant_object_get_int_vec_attribute_value, savant_object_get_label,
        savant_object_get_namespace, savant_object_get_tracking_info, savant_object_set_confidence,
        savant_object_set_detection_box, savant_object_set_float_vec_attribute_value,
        savant_object_set_int_vec_attribute_value, savant_object_set_tracking_info,
        CAPIBoundingBox,
    };
    use crate::primitives::object::BorrowedVideoObject;

    #[test]
    fn test_object_ops() {
        let f = gen_frame();
        let o = BorrowedVideoObject(f.get_object(1).unwrap());
        let optr = &o as *const BorrowedVideoObject;
        let ids = unsafe { savant_object_get_ids(optr) };
        assert_eq!(ids.id, 1);
    }

    #[test]
    fn test_object_confidence() {
        let f = gen_frame();
        let mut o = BorrowedVideoObject(f.get_object(1).unwrap());
        let optr_mut = &mut o as *mut BorrowedVideoObject;
        unsafe { savant_object_set_confidence(optr_mut, 0.5) };
        let mut conf = 0.0;
        let res = unsafe { savant_object_get_confidence(optr_mut, &mut conf) };
        assert!(res);
        assert_eq!(conf, 0.5);
        unsafe { savant_object_clear_confidence(optr_mut) };
        let res = unsafe { savant_object_get_confidence(optr_mut, &mut conf) };
        assert!(!res);
    }

    #[test]
    fn test_object_namespace() {
        let f = gen_frame();
        let o = BorrowedVideoObject(f.get_object(1).unwrap());
        let optr = &o as *const BorrowedVideoObject;
        let result_buf = vec![0u8; 100];
        let result_buf = result_buf.as_ptr() as *mut i8;
        let ns = unsafe { savant_object_get_namespace(optr, result_buf, 100) };
        let ns = unsafe { std::slice::from_raw_parts(result_buf as *const u8, ns) };
        let ns = std::str::from_utf8(ns).unwrap();
        assert_eq!(ns, "test2");
    }

    #[test]
    fn test_object_label() {
        let f = gen_frame();
        let o = BorrowedVideoObject(f.get_object(1).unwrap());
        let optr = &o as *const BorrowedVideoObject;
        let result_buf = vec![0u8; 100];
        let result_buf = result_buf.as_ptr() as *mut i8;
        let label = unsafe { savant_object_get_label(optr, result_buf, 100) };
        let label = unsafe { std::slice::from_raw_parts(result_buf as *const u8, label) };
        let label = std::str::from_utf8(label).unwrap();
        assert_eq!(label, "test");
    }

    #[test]
    fn test_object_draw_label() {
        let f = gen_frame();
        let o = BorrowedVideoObject(f.get_object(1).unwrap());
        let optr = &o as *const BorrowedVideoObject;
        let result_buf = vec![0u8; 100];
        let result_buf = result_buf.as_ptr() as *mut i8;
        let label = unsafe { savant_object_get_draw_label(optr, result_buf, 100) };
        let label = unsafe { std::slice::from_raw_parts(result_buf as *const u8, label) };
        let label = std::str::from_utf8(label).unwrap();
        assert_eq!(label, "test");
    }

    #[test]
    fn test_object_detection_box() {
        let f = gen_frame();
        let mut o = BorrowedVideoObject(f.get_object(1).unwrap());
        let optr_mut = &mut o as *mut BorrowedVideoObject;
        let mut bb = CAPIBoundingBox {
            xc: 0.0,
            yc: 0.0,
            width: 0.0,
            height: 0.0,
            angle: 0.0,
            oriented: true,
        };
        unsafe { savant_object_get_detection_box(optr_mut, &mut bb) };
        assert!(!bb.oriented);
        assert_eq!(bb.xc, 0.0);
        assert_eq!(bb.yc, 0.0);
        assert_eq!(bb.width, 0.0);
        assert_eq!(bb.height, 0.0);
        assert_eq!(bb.angle, 0.0);

        let bb = CAPIBoundingBox {
            xc: 1.0,
            yc: 2.0,
            width: 3.0,
            height: 4.0,
            angle: 5.0,
            oriented: true,
        };
        unsafe { savant_object_set_detection_box(optr_mut, &bb) };
        let mut bb = CAPIBoundingBox {
            xc: 0.0,
            yc: 0.0,
            width: 0.0,
            height: 0.0,
            angle: 0.0,
            oriented: false,
        };
        unsafe { savant_object_get_detection_box(optr_mut, &mut bb) };
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
        let mut o = BorrowedVideoObject(f.get_object(1).unwrap());
        let optr_mut = &mut o as *mut BorrowedVideoObject;
        let mut bb = CAPIBoundingBox {
            xc: 0.0,
            yc: 0.0,
            width: 0.0,
            height: 0.0,
            angle: 0.0,
            oriented: false,
        };
        let mut track_id = 0;
        let res = unsafe { savant_object_get_tracking_info(optr_mut, &mut bb, &mut track_id) };
        assert!(!res);

        let bb = CAPIBoundingBox {
            xc: 1.0,
            yc: 2.0,
            width: 3.0,
            height: 4.0,
            angle: 5.0,
            oriented: true,
        };
        unsafe { savant_object_set_tracking_info(optr_mut, &bb, 1) };
        let mut bb = CAPIBoundingBox {
            xc: 0.0,
            yc: 0.0,
            width: 0.0,
            height: 0.0,
            angle: 0.0,
            oriented: false,
        };
        let mut track_id = 0;
        let res = unsafe { savant_object_get_tracking_info(optr_mut, &mut bb, &mut track_id) };
        assert!(res);
        assert_eq!(track_id, 1);
        assert!(bb.oriented);
        assert_eq!(bb.xc, 1.0);
        assert_eq!(bb.yc, 2.0);
        assert_eq!(bb.width, 3.0);
        assert_eq!(bb.height, 4.0);
        assert_eq!(bb.angle, 5.0);

        unsafe { savant_object_clear_tracking_info(optr_mut) };
        let mut bb = CAPIBoundingBox {
            xc: 0.0,
            yc: 0.0,
            width: 0.0,
            height: 0.0,
            angle: 0.0,
            oriented: false,
        };
        let mut track_id = 0;
        let res = unsafe { savant_object_get_tracking_info(optr_mut, &mut bb, &mut track_id) };
        assert!(!res);
    }

    #[test]
    fn test_get_float_vec_attribute_value() {
        let f = gen_frame();
        let mut o = BorrowedVideoObject(f.get_object(1).unwrap());
        let optr = &o as *const BorrowedVideoObject;

        let namespace = "test";
        let name = "test";

        let c_namespace_bind = CString::new(namespace).unwrap();
        let c_namespace = c_namespace_bind.as_ptr();

        let c_name_bind = CString::new(name).unwrap();
        let c_name = c_name_bind.as_ptr();

        o.0.set_persistent_attribute(
            namespace,
            name,
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
                savant_object_get_float_vec_attribute_value(
                    optr,
                    c_namespace,
                    c_name,
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
                savant_object_get_float_vec_attribute_value(
                    optr,
                    c_namespace,
                    c_name,
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
    fn test_get_int_vec_attribute_value() {
        let f = gen_frame();
        let mut o = BorrowedVideoObject(f.get_object(1).unwrap());
        let optr = &o as *const BorrowedVideoObject;

        let namespace = "test";
        let name = "test";

        let c_namespace = CString::new(namespace).unwrap();
        let c_name = CString::new(name).unwrap();
        o.0.set_persistent_attribute(
            namespace,
            name,
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
                savant_object_get_int_vec_attribute_value(
                    optr,
                    c_namespace.as_ptr(),
                    c_name.as_ptr(),
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
                savant_object_get_int_vec_attribute_value(
                    optr,
                    c_namespace.as_ptr(),
                    c_name.as_ptr(),
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
    fn test_set_float_vec_attribute_value() {
        let f = gen_frame();
        let mut o = BorrowedVideoObject(f.get_object(1).unwrap());
        let optr_mut = &mut o as *mut BorrowedVideoObject;
        let values = vec![1.0, 2.0, 3.0];

        let namespace = "newtest";
        let name = "newtest";

        let c_namespace = CString::new("newtest").unwrap();
        let c_name = CString::new("newtest").unwrap();

        unsafe {
            savant_object_set_float_vec_attribute_value(
                optr_mut,
                c_namespace.as_ptr(),
                c_name.as_ptr(),
                std::ptr::null(),
                values.as_ptr(),
                values.len(),
                std::ptr::null(),
                false,
                false,
            )
        };
        let attribute = o.get_attribute(namespace, name).unwrap();
        let values = attribute.0.get_values();
        assert_eq!(values.len(), 1);
        assert_eq!(
            values[0].value,
            savant_core::primitives::attribute_value::AttributeValueVariant::FloatVector(vec![
                1.0, 2.0, 3.0
            ])
        );
    }

    #[test]
    fn test_set_int_vec_attribute_value() {
        let f = gen_frame();
        let mut o = BorrowedVideoObject(f.get_object(1).unwrap());
        let optr_mut = &mut o as *mut BorrowedVideoObject;
        let values = vec![1, 2, 3];

        let namespace = "newtest";
        let name = "newtest";

        let c_namespace = CString::new("newtest").unwrap();
        let c_name = CString::new("newtest").unwrap();

        unsafe {
            savant_object_set_int_vec_attribute_value(
                optr_mut,
                c_namespace.as_ptr(),
                c_name.as_ptr(),
                std::ptr::null(),
                values.as_ptr(),
                values.len(),
                std::ptr::null(),
                false,
                false,
            )
        };
        let attribute = o.get_attribute(namespace, name).unwrap();
        let values = attribute.0.get_values();
        assert_eq!(values.len(), 1);
        assert_eq!(
            values[0].value,
            savant_core::primitives::attribute_value::AttributeValueVariant::IntegerVector(vec![
                1, 2, 3
            ])
        );
    }
}
