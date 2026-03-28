//! Verify `attach_detection_meta` populates object metas (no nvtracker element).

mod common;

use deepstream_buffers::{
    BufferGenerator, NvBufSurfaceMemType, SavantIdMetaKind, SharedBuffer, SurfaceView,
    TransformConfig, UniformBatchGenerator, VideoFormat,
};
use deepstream_sys::{gst_buffer_get_nvds_batch_meta, NvDsFrameMeta};
use gstreamer as gst;
use nvtracker::{attach_detection_meta, Roi};
use savant_core::primitives::RBBox;
use serial_test::serial;

fn rb(left: f32, top: f32, w: f32, h: f32) -> RBBox {
    RBBox::ltwh(left, top, w, h).expect("RBBox")
}

fn make_single_slot_shared(w: u32, h: u32) -> SharedBuffer {
    let src_gen = BufferGenerator::new(
        VideoFormat::RGBA,
        w,
        h,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .expect("BufferGenerator");
    let batched_gen = UniformBatchGenerator::new(
        VideoFormat::RGBA,
        w,
        h,
        1,
        2,
        0,
        NvBufSurfaceMemType::Default,
    )
    .expect("UniformBatchGenerator");
    let id_kinds = vec![SavantIdMetaKind::Frame(1)];
    let mut batch = batched_gen
        .acquire_batch(TransformConfig::default(), id_kinds)
        .unwrap();
    let src = src_gen.acquire(None).unwrap();
    let src_view = SurfaceView::from_buffer(&src, 0).unwrap();
    batch.transform_slot(0, &src_view, None).unwrap();
    batch.finalize().unwrap();
    batch.shared_buffer().clone()
}

unsafe fn count_objects_first_frame(buf: &mut gst::BufferRef) -> u32 {
    let ptr = buf.as_mut_ptr() as *mut deepstream_sys::GstBuffer;
    let bm = gst_buffer_get_nvds_batch_meta(ptr);
    if bm.is_null() {
        return 0;
    }
    let fl = (*bm).frame_meta_list;
    if fl.is_null() {
        return 0;
    }
    let frame = (*fl).data as *mut NvDsFrameMeta;
    if frame.is_null() {
        return 0;
    }
    (*frame).num_obj_meta
}

#[test]
#[serial]
fn attach_detection_meta_writes_two_objects() {
    common::init();
    let shared = make_single_slot_shared(320, 240);
    let pad = 42u32;
    let slots = [(
        pad,
        vec![
            (
                0i32,
                Roi {
                    id: 1,
                    bbox: rb(10.0, 10.0, 32.0, 32.0),
                },
            ),
            (
                0i32,
                Roi {
                    id: 2,
                    bbox: rb(100.0, 80.0, 40.0, 40.0),
                },
            ),
        ],
    )];
    {
        let mut guard = shared.lock();
        let buf_ref = guard.make_mut();
        attach_detection_meta(buf_ref, 1, 4, &slots, &[0]).expect("attach_detection_meta");
    }
    let n = {
        let mut guard = shared.lock();
        let buf_ref = guard.make_mut();
        unsafe { count_objects_first_frame(buf_ref) }
    };
    assert_eq!(n, 2, "expected two NvDsObjectMeta on frame 0");
}
