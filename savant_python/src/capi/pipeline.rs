use crate::logging::{log_message, LogLevel};
use savant_core::pipeline::Pipeline;
use std::ffi::{c_char, CStr};
use std::slice::from_raw_parts;

/// # Safety
///
/// The function is intended for invocation from C/C++, so it is unsafe by design.
#[no_mangle]
pub unsafe extern "C" fn pipeline2_move_as_is(
    handle: usize,
    dest_stage: *const c_char,
    ids: *const i64,
    len: usize,
) {
    let dest_stage = CStr::from_ptr(dest_stage)
        .to_str()
        .expect("Failed to convert dest_stage to string. This is a bug. Please report it.");
    let pipeline = &*(handle as *const Pipeline);
    let ids = from_raw_parts(ids, len);
    pipeline
        .move_as_is(dest_stage, ids.to_vec())
        .expect("Failed to move objects as is.");
}

/// # Safety
///
/// The function is intended for invocation from C/C++, so it is unsafe by design.
#[no_mangle]
pub unsafe extern "C" fn pipeline2_move_and_pack_frames(
    handle: usize,
    dest_stage: *const c_char,
    frame_ids: *const i64,
    len: usize,
) -> i64 {
    let dest_stage = CStr::from_ptr(dest_stage)
        .to_str()
        .expect("Failed to convert dest_stage to string. This is a bug. Please report it.");
    let pipeline = &*(handle as *const Pipeline);
    let ids = from_raw_parts(frame_ids, len);
    pipeline
        .move_and_pack_frames(dest_stage, ids.to_vec())
        .expect("Failed to move objects as is.")
}

/// # Safety
///
/// The function is intended for invocation from C/C++, so it is unsafe by design.
#[no_mangle]
pub unsafe extern "C" fn pipeline2_move_and_unpack_batch(
    handle: usize,
    dest_stage: *const c_char,
    batch_id: i64,
    resulting_ids: *mut i64,
    resulting_ids_len: usize,
) -> usize {
    let dest_stage = CStr::from_ptr(dest_stage)
        .to_str()
        .expect("Failed to convert dest_stage to string. This is a bug. Please report it.");
    let pipeline = &*(handle as *const Pipeline);
    let ids = pipeline
        .move_and_unpack_batch(dest_stage, batch_id)
        .expect("Failed to move objects as is.");
    if ids.len() > resulting_ids_len {
        panic!("Not enough space in resulting_ids");
    }
    for (i, id) in ids.iter().enumerate() {
        *resulting_ids.add(i) = *id;
    }
    ids.len()
}

/// # Safety
///
/// The function is intended for invocation from C/C++, so it is unsafe by design.
///
/// Arguments
/// ---------
/// handle: usize
///   The pipeline handle
/// id: i64
///   The batch or frame id to apply updates to
///
/// Returns
/// -------
/// bool
///   True if the updates were applied, false otherwise
///
#[no_mangle]
pub unsafe extern "C" fn pipeline2_apply_updates(handle: usize, id: i64) -> bool {
    let pipeline = &*(handle as *const Pipeline);
    let res = pipeline.apply_updates(id);
    if let Err(e) = res {
        log_message(
            LogLevel::Error,
            String::from("pipeline::capi::apply_updates"),
            format!("Failed to apply updates: {}", e),
            None,
        );
        false
    } else {
        true
    }
}

/// # Safety
///
/// The function is intended for invocation from C/C++, so it is unsafe by design.
///
/// Arguments
/// ---------
/// handle: usize
///   The pipeline handle
/// id: i64
///   The batch or frame id to clear updates from
///
/// Returns
/// -------
/// bool
///   True if the updates were cleared, false otherwise
///
#[no_mangle]
pub unsafe extern "C" fn pipeline2_clear_updates(handle: usize, id: i64) -> bool {
    let pipeline = &*(handle as *const Pipeline);
    let res = pipeline.clear_updates(id);
    if let Err(e) = res {
        log_message(
            LogLevel::Error,
            String::from("pipeline::capi::clear_updates"),
            format!("Failed to clear updates: {}", e),
            None,
        );
        false
    } else {
        true
    }
}

#[cfg(test)]
mod tests {

    use crate::capi::pipeline::{
        pipeline2_move_and_pack_frames, pipeline2_move_and_unpack_batch, pipeline2_move_as_is,
    };
    use savant_core::pipeline::Pipeline;
    use savant_core::pipeline::PipelineStagePayloadType;
    use savant_core::test::gen_frame;
    use std::ffi::CString;

    #[test]
    fn test_move_as_is() {
        let pipeline = Pipeline::new(vec![
            ("stage1".to_owned(), PipelineStagePayloadType::Frame),
            ("stage2".to_owned(), PipelineStagePayloadType::Frame),
        ])
        .unwrap();
        let id = pipeline.add_frame("stage1", gen_frame()).unwrap();
        let stage = CString::new("stage2").unwrap();
        unsafe {
            pipeline2_move_as_is(pipeline.memory_handle(), stage.as_ptr(), [id].as_ptr(), 1);
        }
    }

    #[test]
    fn test_move_and_pack() {
        let pipeline = Pipeline::new(vec![
            ("stage1".to_owned(), PipelineStagePayloadType::Frame),
            ("stage2".to_owned(), PipelineStagePayloadType::Batch),
        ])
        .unwrap();
        let id = pipeline.add_frame("stage1", gen_frame()).unwrap();
        let stage = CString::new("stage2").unwrap();
        let batch_id = unsafe {
            pipeline2_move_and_pack_frames(
                pipeline.memory_handle(),
                stage.as_ptr(),
                [id].as_ptr(),
                1,
            )
        };
        assert_eq!(batch_id, 2);
    }

    #[test]
    fn test_move_and_unpack() {
        let pipeline = Pipeline::new(vec![
            ("stage1".to_owned(), PipelineStagePayloadType::Frame),
            ("stage2".to_owned(), PipelineStagePayloadType::Batch),
            ("stage3".to_owned(), PipelineStagePayloadType::Frame),
        ])
        .unwrap();
        let id = pipeline.add_frame("stage1", gen_frame()).unwrap();
        let batch_id = pipeline.move_and_pack_frames("stage2", vec![id]).unwrap();
        let stage = CString::new("stage3").unwrap();
        const MAX_ELTS: usize = 16;
        let mut frame_ids = [0i64; MAX_ELTS];
        let count = unsafe {
            pipeline2_move_and_unpack_batch(
                pipeline.memory_handle(),
                stage.as_ptr(),
                batch_id,
                frame_ids.as_mut_ptr(),
                MAX_ELTS,
            )
        };
        assert_eq!(count, 1);
        assert_eq!(frame_ids[0], id);
    }
}
