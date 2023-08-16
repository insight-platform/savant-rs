use savant_core::pipeline::Pipeline;
use std::ffi::{c_char, CStr};
use std::slice::from_raw_parts;

#[no_mangle]
pub unsafe extern "C" fn pipeline_move_as_is(
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

#[no_mangle]
pub unsafe extern "C" fn pipeline_move_and_pack_frames(
    handle: usize,
    dest_stage: *const c_char,
    frame_ids: *const i64,
    len: usize,
    batch_id: *mut i64,
) {
    let dest_stage = CStr::from_ptr(dest_stage)
        .to_str()
        .expect("Failed to convert dest_stage to string. This is a bug. Please report it.");
    let pipeline = &*(handle as *const Pipeline);
    let ids = from_raw_parts(frame_ids, len);
    let res = pipeline
        .move_and_pack_frames(dest_stage, ids.to_vec())
        .expect("Failed to move objects as is.");
    *batch_id = res;
}

#[no_mangle]
pub unsafe extern "C" fn pipeline_move_and_unpack_batch(
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
    let ids = ids.values().cloned().collect::<Vec<_>>();
    if ids.len() > resulting_ids_len {
        panic!("Not enough space in resulting_ids");
    }
    for (i, id) in ids.iter().enumerate() {
        *resulting_ids.add(i) = *id;
    }
    ids.len()
}

#[cfg(test)]
mod tests {
    use crate::pipeline::{
        pipeline_move_and_pack_frames, pipeline_move_and_unpack_batch, pipeline_move_as_is,
    };
    use savant_core::pipeline::PipelineStagePayloadType;
    use savant_core::rust::Pipeline;
    use savant_core::test::gen_frame;
    use std::ffi::CString;

    #[test]
    fn test_move_as_is() {
        let pipeline = Pipeline::default();
        pipeline
            .add_stage("stage1", PipelineStagePayloadType::Frame)
            .unwrap();
        pipeline
            .add_stage("stage2", PipelineStagePayloadType::Frame)
            .unwrap();
        let id = pipeline.add_frame("stage1", gen_frame()).unwrap();
        let stage = CString::new("stage2").unwrap();
        unsafe {
            pipeline_move_as_is(pipeline.memory_handle(), stage.as_ptr(), [id].as_ptr(), 1);
        }
    }

    #[test]
    fn test_move_and_pack() {
        let pipeline = Pipeline::default();
        pipeline
            .add_stage("stage1", PipelineStagePayloadType::Frame)
            .unwrap();
        pipeline
            .add_stage("stage2", PipelineStagePayloadType::Batch)
            .unwrap();
        let id = pipeline.add_frame("stage1", gen_frame()).unwrap();
        let stage = CString::new("stage2").unwrap();
        let mut batch_id: i64 = 0;
        unsafe {
            pipeline_move_and_pack_frames(
                pipeline.memory_handle(),
                stage.as_ptr(),
                [id].as_ptr(),
                1,
                &mut batch_id as *mut i64,
            );
        }
        assert_eq!(batch_id, 2);
    }

    #[test]
    fn test_move_and_unpack() {
        let pipeline = Pipeline::default();
        pipeline
            .add_stage("stage1", PipelineStagePayloadType::Frame)
            .unwrap();
        pipeline
            .add_stage("stage2", PipelineStagePayloadType::Batch)
            .unwrap();
        pipeline
            .add_stage("stage3", PipelineStagePayloadType::Frame)
            .unwrap();

        let id = pipeline.add_frame("stage1", gen_frame()).unwrap();
        let batch_id = pipeline.move_and_pack_frames("stage2", vec![id]).unwrap();
        let stage = CString::new("stage3").unwrap();
        let mut frame_ids = [0i64; 16];
        let count = unsafe {
            pipeline_move_and_unpack_batch(
                pipeline.memory_handle(),
                stage.as_ptr(),
                batch_id,
                frame_ids.as_mut_ptr(),
                16,
            )
        };
        assert_eq!(count, 1);
        assert_eq!(frame_ids[0], id);
    }
}
