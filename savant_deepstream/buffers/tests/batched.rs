//! Integration tests for [`UniformBatchGenerator`] and [`SurfaceBatch`].

mod common;

use deepstream_buffers::{
    extract_nvbufsurface, BufferGenerator, NvBufSurfaceMemType, Padding, Rect, SavantIdMeta,
    SavantIdMetaKind, SurfaceView, TransformConfig, UniformBatchGenerator, VideoFormat,
};

fn make_src_gen(format: VideoFormat, w: u32, h: u32) -> BufferGenerator {
    BufferGenerator::new(format, w, h, 30, 1, 0, NvBufSurfaceMemType::Default)
        .expect("failed to build source generator")
}

fn make_batched_gen(
    format: VideoFormat,
    w: u32,
    h: u32,
    batch: u32,
    pool: u32,
) -> UniformBatchGenerator {
    UniformBatchGenerator::new(format, w, h, batch, pool, 0, NvBufSurfaceMemType::Default)
        .expect("failed to build batched generator")
}

// ─── Unit: UniformBatchGenerator construction ─────────────────────────

#[test]
fn test_create_batched_generator() {
    common::init();
    let _gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 4, 2);
}

#[test]
fn test_batched_generator_builder() {
    common::init();
    let _gen = UniformBatchGenerator::builder(VideoFormat::RGBA, 640, 640, 4)
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
    let batch = gen.acquire_batch(config, vec![]).unwrap();
    assert_eq!(batch.max_batch_size(), 1);
}

// ─── Unit: SurfaceBatch acquisition and slot access ────────────────────────

#[test]
fn test_acquire_batch() {
    common::init();
    let gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 4, 2);
    let config = TransformConfig::default();
    let mut batch = gen.acquire_batch(config, vec![]).unwrap();

    assert_eq!(batch.max_batch_size(), 4);
    assert_eq!(batch.num_filled(), 0);

    batch.finalize().unwrap();
    let shared = batch.shared_buffer();
    let guard = shared.lock();
    let surf_ptr = unsafe { extract_nvbufsurface(guard.as_ref()).expect("extract should work") };
    let surface = unsafe { &*surf_ptr };
    assert_eq!(surface.batchSize, 4);
    assert_eq!(surface.numFilled, 0);
}

// ─── E2E: transform_slot + finalize ───────────────────────────────────────────────

#[test]
fn test_fill_reuse_after_finalize() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 320, 240);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 2, 2);

    let config = TransformConfig::default();
    let mut batch1 = batched_gen
        .acquire_batch(config, vec![SavantIdMetaKind::Frame(1)])
        .unwrap();
    let src = src_gen.acquire(None).unwrap();
    let src_view = SurfaceView::from_buffer(&src, 0).unwrap();
    batch1.transform_slot(0, &src_view, None).unwrap();
    batch1.finalize().unwrap();
    let _shared1 = batch1.shared_buffer();

    let config = TransformConfig::default();
    let mut batch2 = batched_gen
        .acquire_batch(config, vec![SavantIdMetaKind::Frame(2)])
        .unwrap();
    let src = src_gen.acquire(None).unwrap();
    let src_view = SurfaceView::from_buffer(&src, 0).unwrap();
    batch2.transform_slot(0, &src_view, None).unwrap();
    batch2.finalize().unwrap();
    let shared2 = batch2.shared_buffer();

    let guard = shared2.lock();
    let meta = guard.meta::<SavantIdMeta>().unwrap();
    assert_eq!(meta.ids(), &[SavantIdMetaKind::Frame(2)]);
}

// ─── ID handling tests ───────────────────────────────────────────────────────

#[test]
fn test_fill_all_slots() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 4, 2);

    let ids = [10i64, 20, 30, 40];
    let id_kinds: Vec<_> = ids
        .iter()
        .map(|&id| SavantIdMetaKind::Frame(id as u128))
        .collect();
    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batch(config, id_kinds).unwrap();

    for (i, &_id) in ids.iter().enumerate() {
        let src = src_gen.acquire(None).unwrap();
        let src_view = SurfaceView::from_buffer(&src, 0).unwrap();
        batch.transform_slot(i as u32, &src_view, None).unwrap();
    }

    assert_eq!(batch.num_filled(), 4);
    batch.finalize().unwrap();
    let shared = batch.shared_buffer();

    let guard = shared.lock();
    let surf_ptr = unsafe { extract_nvbufsurface(guard.as_ref()).unwrap() };
    assert_eq!(unsafe { (*surf_ptr).numFilled }, 4);

    let meta = guard
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

    let id_kinds: Vec<_> = (0..3).map(SavantIdMetaKind::Frame).collect();
    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batch(config, id_kinds).unwrap();

    for i in 0..3 {
        let src = src_gen.acquire(None).unwrap();
        let src_view = SurfaceView::from_buffer(&src, 0).unwrap();
        batch.transform_slot(i as u32, &src_view, None).unwrap();
    }

    batch.finalize().unwrap();
    let shared = batch.shared_buffer();

    let guard = shared.lock();
    let surf_ptr = unsafe { extract_nvbufsurface(guard.as_ref()).unwrap() };
    assert_eq!(unsafe { (*surf_ptr).numFilled }, 3);

    let meta = guard.meta::<SavantIdMeta>().unwrap();
    assert_eq!(meta.ids().len(), 3);
}

#[test]
fn test_fill_single_slot() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 4, 2);

    let config = TransformConfig::default();
    let mut batch = batched_gen
        .acquire_batch(config, vec![SavantIdMetaKind::Frame(42)])
        .unwrap();

    let src = src_gen.acquire(None).unwrap();
    let src_view = SurfaceView::from_buffer(&src, 0).unwrap();
    batch.transform_slot(0, &src_view, None).unwrap();

    batch.finalize().unwrap();
    let shared = batch.shared_buffer();

    let guard = shared.lock();
    let surf_ptr = unsafe { extract_nvbufsurface(guard.as_ref()).unwrap() };
    assert_eq!(unsafe { (*surf_ptr).numFilled }, 1);

    let meta = guard.meta::<SavantIdMeta>().unwrap();
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
    let mut batch = batched_gen
        .acquire_batch(config, vec![SavantIdMetaKind::Frame(1)])
        .unwrap();

    let src = src_gen.acquire(None).unwrap();
    let src_view = SurfaceView::from_buffer(&src, 0).unwrap();
    let roi = Rect {
        top: 100,
        left: 200,
        width: 800,
        height: 600,
    };
    batch.transform_slot(0, &src_view, Some(&roi)).unwrap();

    batch.finalize().unwrap();
    let shared = batch.shared_buffer();
    let guard = shared.lock();
    assert!(guard.size() > 0);
}

#[test]
fn test_fill_with_no_roi() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 2, 2);

    let config = TransformConfig::default();
    let mut batch = batched_gen
        .acquire_batch(config, vec![SavantIdMetaKind::Frame(1)])
        .unwrap();

    let src = src_gen.acquire(None).unwrap();
    let src_view = SurfaceView::from_buffer(&src, 0).unwrap();
    batch.transform_slot(0, &src_view, None).unwrap();

    batch.finalize().unwrap();
    let shared = batch.shared_buffer();
    let guard = shared.lock();
    assert!(guard.size() > 0);
}

#[test]
fn test_fill_different_source_resolutions() {
    common::init();
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 4, 2);

    let resolutions = [(320, 240), (1920, 1080), (1280, 720)];
    let id_kinds: Vec<_> = (0..resolutions.len())
        .map(|i| SavantIdMetaKind::Frame(i as u128))
        .collect();
    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batch(config, id_kinds).unwrap();

    for (i, (w, h)) in resolutions.iter().enumerate() {
        let src_gen = make_src_gen(VideoFormat::RGBA, *w, *h);
        let src = src_gen.acquire(None).unwrap();
        let src_view = SurfaceView::from_buffer(&src, 0).unwrap();
        batch.transform_slot(i as u32, &src_view, None).unwrap();
    }

    batch.finalize().unwrap();
    let shared = batch.shared_buffer();
    let guard = shared.lock();
    let surf_ptr = unsafe { extract_nvbufsurface(guard.as_ref()).unwrap() };
    assert_eq!(unsafe { (*surf_ptr).numFilled }, 3);
}

#[test]
fn test_fill_nv12_to_rgba() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::NV12, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 2, 2);

    let config = TransformConfig::default();
    let mut batch = batched_gen
        .acquire_batch(config, vec![SavantIdMetaKind::Frame(1)])
        .unwrap();

    let src = src_gen.acquire(None).unwrap();
    let src_view = SurfaceView::from_buffer(&src, 0).unwrap();
    batch.transform_slot(0, &src_view, None).unwrap();

    batch.finalize().unwrap();
    let shared = batch.shared_buffer();
    let guard = shared.lock();
    assert!(guard.size() > 0);
}

// ─── Corner case / error tests ───────────────────────────────────────────────

#[test]
fn test_fill_exceeds_batch_size() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 320, 240);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 2, 2);

    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batch(config, vec![]).unwrap();

    let src1 = src_gen.acquire(None).unwrap();
    let src2 = src_gen.acquire(None).unwrap();
    let src3 = src_gen.acquire(None).unwrap();
    let v1 = SurfaceView::from_buffer(&src1, 0).unwrap();
    let v2 = SurfaceView::from_buffer(&src2, 0).unwrap();
    let v3 = SurfaceView::from_buffer(&src3, 0).unwrap();

    batch.transform_slot(0, &v1, None).unwrap();
    batch.transform_slot(1, &v2, None).unwrap();
    let result = batch.transform_slot(2, &v3, None);
    assert!(result.is_err(), "slot 2 should fail on batch_size=2");
}

#[test]
fn test_finalize_empty_batch() {
    common::init();
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 4, 2);

    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batch(config, vec![]).unwrap();

    batch.finalize().unwrap();
    let shared = batch.shared_buffer();
    let guard = shared.lock();
    let surf_ptr = unsafe { extract_nvbufsurface(guard.as_ref()).unwrap() };
    assert_eq!(unsafe { (*surf_ptr).numFilled }, 0);

    assert!(
        guard.meta::<SavantIdMeta>().is_none(),
        "empty batch should have no SavantIdMeta"
    );
}

#[test]
fn test_explicit_ids_in_order() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 320, 240);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 4, 2);

    let ids = [100i64, 200, 300];
    let id_kinds: Vec<_> = ids
        .iter()
        .map(|&id| SavantIdMetaKind::Frame(id as u128))
        .collect();
    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batch(config, id_kinds).unwrap();

    for (i, &_id) in ids.iter().enumerate() {
        let src = src_gen.acquire(None).unwrap();
        let src_view = SurfaceView::from_buffer(&src, 0).unwrap();
        batch.transform_slot(i as u32, &src_view, None).unwrap();
    }

    batch.finalize().unwrap();
    let shared = batch.shared_buffer();
    let guard = shared.lock();
    let meta = guard.meta::<SavantIdMeta>().unwrap();
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
fn test_no_ids_at_all() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 320, 240);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 4, 2);

    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batch(config, vec![]).unwrap();

    for i in 0..3 {
        let src = src_gen.acquire(None).unwrap();
        let src_view = SurfaceView::from_buffer(&src, 0).unwrap();
        batch.transform_slot(i as u32, &src_view, None).unwrap();
    }

    batch.finalize().unwrap();
    let shared = batch.shared_buffer();
    let guard = shared.lock();
    assert!(
        guard.meta::<SavantIdMeta>().is_none(),
        "no IDs provided → no SavantIdMeta"
    );
}

// ─── acquire (simple acquisition) ─────────────────────────────────────

#[test]
fn test_acquire_with_id() {
    common::init();
    let gen = BufferGenerator::new(
        VideoFormat::RGBA,
        640,
        640,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .unwrap();
    let shared = gen.acquire(Some(42)).unwrap();
    let guard = shared.lock();
    let meta = guard
        .meta::<SavantIdMeta>()
        .expect("should have SavantIdMeta");
    assert_eq!(meta.ids(), &[SavantIdMetaKind::Frame(42)]);
}

#[test]
fn test_acquire_without_id() {
    common::init();
    let gen = BufferGenerator::new(
        VideoFormat::RGBA,
        640,
        640,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .unwrap();
    let shared = gen.acquire(None).unwrap();
    let guard = shared.lock();
    assert!(
        guard.meta::<SavantIdMeta>().is_none(),
        "no ID → no SavantIdMeta"
    );
}

#[test]
fn test_acquire_surface_view() {
    common::init();
    let gen = BufferGenerator::new(
        VideoFormat::RGBA,
        640,
        640,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .unwrap();
    let shared = gen.acquire(Some(1)).unwrap();
    let view = SurfaceView::from_buffer(&shared, 0).unwrap();
    assert_eq!(view.width(), 640);
    assert_eq!(view.height(), 640);
    assert_eq!(view.channels(), 4);
    assert!(!view.data_ptr().is_null());
}

// ─── view and shared_buffer ─────────────────────────────────────────────

#[test]
fn test_slot_view_after_finalize() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 2, 2);
    let config = TransformConfig::default();
    let mut batch = batched_gen
        .acquire_batch(config, vec![SavantIdMetaKind::Frame(10)])
        .unwrap();

    let src = src_gen.acquire(None).unwrap();
    let src_view = SurfaceView::from_buffer(&src, 0).unwrap();
    batch.transform_slot(0, &src_view, None).unwrap();
    batch.finalize().unwrap();

    let view = batch.view(0).unwrap();
    assert_eq!(view.width(), 640);
    assert_eq!(view.height(), 640);
    assert!(!view.data_ptr().is_null());
}

#[test]
fn test_shared_buffer_returns_valid_ref() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 320, 240);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 2, 2);
    let config = TransformConfig::default();
    let mut batch = batched_gen
        .acquire_batch(config, vec![SavantIdMetaKind::Frame(1)])
        .unwrap();

    let src = src_gen.acquire(None).unwrap();
    let src_view = SurfaceView::from_buffer(&src, 0).unwrap();
    batch.transform_slot(0, &src_view, None).unwrap();
    batch.finalize().unwrap();

    let shared = batch.shared_buffer();
    let guard = shared.lock();
    let surf_ptr = unsafe { extract_nvbufsurface(guard.as_ref()).unwrap() };
    let surf = unsafe { &*surf_ptr };
    assert_eq!(surf.numFilled, 1);
}

// ─── transform_into ──────────────────────────────────────────────────────────

#[test]
fn test_transform_into_basic() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let dst_gen = BufferGenerator::new(
        VideoFormat::RGBA,
        640,
        640,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .unwrap();

    let src_shared = src_gen.acquire(None).unwrap();
    let src_view = SurfaceView::from_buffer(&src_shared, 0).unwrap();

    let dst_shared = dst_gen.acquire(None).unwrap();
    let dst_view = SurfaceView::from_buffer(&dst_shared, 0).unwrap();

    let config = TransformConfig::default();
    src_view.transform_into(&dst_view, &config, None).unwrap();

    assert!(!dst_view.data_ptr().is_null());
    assert_eq!(dst_view.width(), 640);
    assert_eq!(dst_view.height(), 640);
}

#[test]
fn test_transform_into_with_roi() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let dst_gen = BufferGenerator::new(
        VideoFormat::RGBA,
        640,
        640,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .unwrap();

    let src_shared = src_gen.acquire(None).unwrap();
    let src_view = SurfaceView::from_buffer(&src_shared, 0).unwrap();

    let dst_shared = dst_gen.acquire(None).unwrap();
    let dst_view = SurfaceView::from_buffer(&dst_shared, 0).unwrap();

    let roi = Rect {
        top: 100,
        left: 200,
        width: 800,
        height: 600,
    };
    let config = TransformConfig {
        padding: Padding::Symmetric,
        ..Default::default()
    };
    src_view
        .transform_into(&dst_view, &config, Some(&roi))
        .unwrap();
}
