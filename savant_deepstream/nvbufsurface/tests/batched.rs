//! Integration tests for [`DsNvUniformSurfaceBufferGenerator`] and [`DsNvUniformSurfaceBuffer`].

mod common;

use deepstream_nvbufsurface::{
    extract_nvbufsurface, set_num_filled, DsNvSurfaceBufferGenerator,
    DsNvUniformSurfaceBufferGenerator, NvBufSurfaceMemType, Padding, Rect, SavantIdMeta,
    SavantIdMetaKind, TransformConfig, VideoFormat,
};

fn make_src_gen(format: VideoFormat, w: u32, h: u32) -> DsNvSurfaceBufferGenerator {
    DsNvSurfaceBufferGenerator::builder(format, w, h)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(4)
        .max_buffers(4)
        .build()
        .expect("failed to build source generator")
}

fn make_batched_gen(
    format: VideoFormat,
    w: u32,
    h: u32,
    batch: u32,
    pool: u32,
) -> DsNvUniformSurfaceBufferGenerator {
    DsNvUniformSurfaceBufferGenerator::new(
        format,
        w,
        h,
        batch,
        pool,
        0,
        NvBufSurfaceMemType::Default,
    )
    .expect("failed to build batched generator")
}

// ─── Unit: DsNvUniformSurfaceBufferGenerator construction ─────────────────────────

#[test]
fn test_create_batched_generator() {
    common::init();
    let _gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 4, 2);
}

#[test]
fn test_batched_generator_builder() {
    common::init();
    let _gen = DsNvUniformSurfaceBufferGenerator::builder(VideoFormat::RGBA, 640, 640, 4)
        .fps(30, 1)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .pool_size(2)
        .build()
        .expect("builder should succeed");
}

#[test]
fn test_batched_generator_batch_size_1() {
    common::init();
    let gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 1, 1);
    let config = TransformConfig::default();
    let batch = gen.acquire_batched_surface(config).unwrap();
    assert_eq!(batch.max_batch_size(), 1);
}

// ─── Unit: DsNvUniformSurfaceBuffer acquisition and slot access ────────────────────────

#[test]
fn test_acquire_batched_surface() {
    common::init();
    let gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 4, 2);
    let config = TransformConfig::default();
    let mut batch = gen.acquire_batched_surface(config).unwrap();

    assert_eq!(batch.max_batch_size(), 4);
    assert_eq!(batch.num_filled(), 0);

    batch.finalize().unwrap();
    let surf_ptr = unsafe {
        extract_nvbufsurface(batch.as_gst_buffer().unwrap().as_ref()).expect("extract should work")
    };
    let surface = unsafe { &*surf_ptr };
    assert_eq!(surface.batchSize, 4);
    assert_eq!(surface.numFilled, 0);
}

#[test]
fn test_slot_ptrs_are_distinct() {
    common::init();
    let gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 4, 2);
    let config = TransformConfig::default();
    let batch = gen.acquire_batched_surface(config).unwrap();

    let mut ptrs = Vec::new();
    for i in 0..4 {
        let (data_ptr, pitch) = batch.slot_ptr(i).unwrap();
        assert!(!data_ptr.is_null(), "slot {} dataPtr should not be null", i);
        assert!(pitch > 0, "slot {} pitch should be > 0", i);
        ptrs.push(data_ptr);
    }

    for i in 0..4 {
        for j in (i + 1)..4 {
            assert_ne!(
                ptrs[i as usize], ptrs[j as usize],
                "slot {} and {} should have distinct dataPtrs",
                i, j
            );
        }
    }
}

#[test]
fn test_slot_ptr_out_of_bounds() {
    common::init();
    let gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 4, 2);
    let config = TransformConfig::default();
    let batch = gen.acquire_batched_surface(config).unwrap();

    assert!(batch.slot_ptr(4).is_err());
    assert!(batch.slot_ptr(100).is_err());
}

// ─── E2E: fill_slot + finalize ───────────────────────────────────────────────

#[test]
fn test_fill_all_slots() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 4, 2);

    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    let ids = [10i64, 20, 30, 40];
    for &id in &ids {
        let src = src_gen.acquire_surface(None).unwrap();
        batch.fill_slot(&src, None, Some(id)).unwrap();
    }

    assert_eq!(batch.num_filled(), 4);
    batch.finalize().unwrap();
    let buffer = batch.as_gst_buffer().unwrap();

    let surf_ptr = unsafe { extract_nvbufsurface(buffer.as_ref()).unwrap() };
    assert_eq!(unsafe { (*surf_ptr).numFilled }, 4);

    let meta = buffer
        .meta::<SavantIdMeta>()
        .expect("should have SavantIdMeta");
    assert_eq!(
        meta.ids(),
        &[
            SavantIdMetaKind::Frame(10),
            SavantIdMetaKind::Frame(20),
            SavantIdMetaKind::Frame(30),
            SavantIdMetaKind::Frame(40),
        ]
    );
}

#[test]
fn test_fill_partial_batch() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 8, 2);

    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    for id in 0..3 {
        let src = src_gen.acquire_surface(None).unwrap();
        batch.fill_slot(&src, None, Some(id)).unwrap();
    }

    batch.finalize().unwrap();
    let buffer = batch.as_gst_buffer().unwrap();

    let surf_ptr = unsafe { extract_nvbufsurface(buffer.as_ref()).unwrap() };
    assert_eq!(unsafe { (*surf_ptr).numFilled }, 3);

    let meta = buffer.meta::<SavantIdMeta>().unwrap();
    assert_eq!(meta.ids().len(), 3);
}

#[test]
fn test_fill_single_slot() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 4, 2);

    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    let src = src_gen.acquire_surface(None).unwrap();
    batch.fill_slot(&src, None, Some(42)).unwrap();

    batch.finalize().unwrap();
    let buffer = batch.as_gst_buffer().unwrap();

    let surf_ptr = unsafe { extract_nvbufsurface(buffer.as_ref()).unwrap() };
    assert_eq!(unsafe { (*surf_ptr).numFilled }, 1);

    let meta = buffer.meta::<SavantIdMeta>().unwrap();
    assert_eq!(meta.ids(), &[SavantIdMetaKind::Frame(42)]);
}

#[test]
fn test_fill_with_src_roi() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 2, 2);

    let config = TransformConfig {
        padding: Padding::Symmetric,
        ..Default::default()
    };
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    let src = src_gen.acquire_surface(None).unwrap();
    let roi = Rect {
        top: 100,
        left: 200,
        width: 800,
        height: 600,
    };
    batch.fill_slot(&src, Some(&roi), Some(1)).unwrap();

    batch.finalize().unwrap();
    let buffer = batch.as_gst_buffer().unwrap();
    assert!(buffer.size() > 0);
}

#[test]
fn test_fill_with_no_roi() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 2, 2);

    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    let src = src_gen.acquire_surface(None).unwrap();
    batch.fill_slot(&src, None, Some(1)).unwrap();

    batch.finalize().unwrap();
    let buffer = batch.as_gst_buffer().unwrap();
    assert!(buffer.size() > 0);
}

#[test]
fn test_fill_different_source_resolutions() {
    common::init();
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 4, 2);

    let resolutions = [(320, 240), (1920, 1080), (1280, 720)];
    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    for (i, (w, h)) in resolutions.iter().enumerate() {
        let src_gen = make_src_gen(VideoFormat::RGBA, *w, *h);
        let src = src_gen.acquire_surface(None).unwrap();
        batch.fill_slot(&src, None, Some(i as i64)).unwrap();
    }

    batch.finalize().unwrap();
    let buffer = batch.as_gst_buffer().unwrap();
    let surf_ptr = unsafe { extract_nvbufsurface(buffer.as_ref()).unwrap() };
    assert_eq!(unsafe { (*surf_ptr).numFilled }, 3);
}

#[test]
fn test_fill_nv12_to_rgba() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::NV12, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 2, 2);

    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    let src = src_gen.acquire_surface(None).unwrap();
    batch.fill_slot(&src, None, Some(1)).unwrap();

    batch.finalize().unwrap();
    let buffer = batch.as_gst_buffer().unwrap();
    assert!(buffer.size() > 0);
}

// ─── Corner case / error tests ───────────────────────────────────────────────

#[test]
fn test_fill_exceeds_batch_size() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 320, 240);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 2, 2);

    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    let src1 = src_gen.acquire_surface(None).unwrap();
    let src2 = src_gen.acquire_surface(None).unwrap();
    let src3 = src_gen.acquire_surface(None).unwrap();

    batch.fill_slot(&src1, None, Some(1)).unwrap();
    batch.fill_slot(&src2, None, Some(2)).unwrap();
    let result = batch.fill_slot(&src3, None, Some(3));
    assert!(result.is_err(), "3rd fill_slot should fail on batch_size=2");
}

#[test]
fn test_finalize_empty_batch() {
    common::init();
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 4, 2);

    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    batch.finalize().unwrap();
    let buffer = batch.as_gst_buffer().unwrap();
    let surf_ptr = unsafe { extract_nvbufsurface(buffer.as_ref()).unwrap() };
    assert_eq!(unsafe { (*surf_ptr).numFilled }, 0);

    assert!(
        buffer.meta::<SavantIdMeta>().is_none(),
        "empty batch should have no SavantIdMeta"
    );
}

#[test]
fn test_fill_reuse_after_finalize() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 320, 240);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 2, 2);

    // First batch
    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();
    let src = src_gen.acquire_surface(None).unwrap();
    batch.fill_slot(&src, None, Some(1)).unwrap();
    batch.finalize().unwrap();
    let _buf1 = batch.as_gst_buffer().unwrap();

    // Second batch from the same generator
    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();
    let src = src_gen.acquire_surface(None).unwrap();
    batch.fill_slot(&src, None, Some(2)).unwrap();
    batch.finalize().unwrap();
    let buf2 = batch.as_gst_buffer().unwrap();

    let meta = buf2.meta::<SavantIdMeta>().unwrap();
    assert_eq!(meta.ids(), &[SavantIdMetaKind::Frame(2)]);
}

// ─── ID handling tests ───────────────────────────────────────────────────────

#[test]
fn test_explicit_ids_in_order() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 320, 240);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 4, 2);

    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    for id in [100i64, 200, 300] {
        let src = src_gen.acquire_surface(None).unwrap();
        batch.fill_slot(&src, None, Some(id)).unwrap();
    }

    batch.finalize().unwrap();
    let buffer = batch.as_gst_buffer().unwrap();
    let meta = buffer.meta::<SavantIdMeta>().unwrap();
    assert_eq!(
        meta.ids(),
        &[
            SavantIdMetaKind::Frame(100),
            SavantIdMetaKind::Frame(200),
            SavantIdMetaKind::Frame(300),
        ]
    );
}

#[test]
fn test_auto_propagate_id_from_source() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 320, 240);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 4, 2);

    let src = src_gen.acquire_surface(Some(42)).unwrap();

    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();
    batch.fill_slot(&src, None, None).unwrap();

    batch.finalize().unwrap();
    let buffer = batch.as_gst_buffer().unwrap();
    let meta = buffer.meta::<SavantIdMeta>().unwrap();
    assert_eq!(meta.ids(), &[SavantIdMetaKind::Frame(42)]);
}

#[test]
fn test_mixed_explicit_and_none_ids() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 320, 240);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 4, 2);

    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    // Slot 0: explicit id
    let src = src_gen.acquire_surface(None).unwrap();
    batch.fill_slot(&src, None, Some(1)).unwrap();

    // Slot 1: no id, source has no meta
    let src = src_gen.acquire_surface(None).unwrap();
    batch.fill_slot(&src, None, None).unwrap();

    // Slot 2: explicit id
    let src = src_gen.acquire_surface(None).unwrap();
    batch.fill_slot(&src, None, Some(3)).unwrap();

    batch.finalize().unwrap();
    let buffer = batch.as_gst_buffer().unwrap();
    let meta = buffer.meta::<SavantIdMeta>().unwrap();
    assert_eq!(
        meta.ids(),
        &[SavantIdMetaKind::Frame(1), SavantIdMetaKind::Frame(3)]
    );
}

#[test]
fn test_no_ids_at_all() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 320, 240);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 4, 2);

    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    for _ in 0..3 {
        let src = src_gen.acquire_surface(None).unwrap();
        batch.fill_slot(&src, None, None).unwrap();
    }

    batch.finalize().unwrap();
    let buffer = batch.as_gst_buffer().unwrap();
    assert!(
        buffer.meta::<SavantIdMeta>().is_none(),
        "no IDs provided → no SavantIdMeta"
    );
}

// ─── set_num_filled standalone function ──────────────────────────────────────

#[test]
fn test_set_num_filled_standalone() {
    common::init();
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 4, 2);

    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    batch.finalize().unwrap();
    // Manually set num_filled via the standalone function
    let mut buffer = batch.as_gst_buffer().unwrap();
    set_num_filled(buffer.make_mut(), 3).expect("set_num_filled should succeed");

    let surf_ptr = unsafe { extract_nvbufsurface(buffer.as_ref()).unwrap() };
    assert_eq!(unsafe { (*surf_ptr).numFilled }, 3);
}

#[test]
fn test_set_num_filled_overflow() {
    common::init();
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 4, 2);

    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    batch.finalize().unwrap();
    let mut buffer = batch.as_gst_buffer().unwrap();
    let result = set_num_filled(buffer.make_mut(), 5);
    assert!(
        result.is_err(),
        "set_num_filled(5) on batch_size=4 should fail"
    );
}
