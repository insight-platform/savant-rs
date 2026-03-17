//! Integration tests for [`DsNvUniformSurfaceBufferGenerator`] and [`DsNvUniformSurfaceBuffer`].

mod common;

use deepstream_nvbufsurface::{
    extract_nvbufsurface, DsNvUniformSurfaceBufferGenerator, NvBufSurfaceMemType, Padding, Rect,
    SavantIdMeta, SavantIdMetaKind, SurfaceView, TransformConfig, VideoFormat,
};

fn make_src_gen(format: VideoFormat, w: u32, h: u32) -> DsNvUniformSurfaceBufferGenerator {
    DsNvUniformSurfaceBufferGenerator::new(format, w, h, 1, 4, 0, NvBufSurfaceMemType::Default)
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

    batch.finalize(0, vec![]).unwrap();
    let shared = batch.shared_buffer();
    let guard = shared.lock();
    let surf_ptr = unsafe { extract_nvbufsurface(guard.as_ref()).expect("extract should work") };
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
fn test_fill_reuse_after_finalize() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 320, 240);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 2, 2);

    let config = TransformConfig::default();
    let mut batch1 = batched_gen.acquire_batched_surface(config).unwrap();
    let src = src_gen.acquire_buffer(None).unwrap();
    batch1.fill_slot(&*src.lock(), None, Some(1)).unwrap();
    batch1
        .finalize(1, vec![SavantIdMetaKind::Frame(1)])
        .unwrap();
    let _shared1 = batch1.shared_buffer();

    let config = TransformConfig::default();
    let mut batch2 = batched_gen.acquire_batched_surface(config).unwrap();
    let src = src_gen.acquire_buffer(None).unwrap();
    batch2.fill_slot(&*src.lock(), None, Some(2)).unwrap();
    batch2
        .finalize(1, vec![SavantIdMetaKind::Frame(2)])
        .unwrap();
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

    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    let ids = [10i64, 20, 30, 40];
    for &id in &ids {
        let src = src_gen.acquire_buffer(None).unwrap();
        batch.fill_slot(&*src.lock(), None, Some(id)).unwrap();
    }

    assert_eq!(batch.num_filled(), 4);
    let id_kinds: Vec<_> = ids.iter().map(|&id| SavantIdMetaKind::Frame(id)).collect();
    batch.finalize(4, id_kinds).unwrap();
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

    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    let mut id_kinds = Vec::new();
    for id in 0..3i64 {
        let src = src_gen.acquire_buffer(None).unwrap();
        batch.fill_slot(&*src.lock(), None, Some(id)).unwrap();
        id_kinds.push(SavantIdMetaKind::Frame(id));
    }

    batch.finalize(3, id_kinds).unwrap();
    let shared = batch.shared_buffer();

    let guard = shared.lock();
    let surf_ptr = unsafe { extract_nvbufsurface(guard.as_ref()).unwrap() };
    assert_eq!(unsafe { (*surf_ptr).numFilled }, 3);

    let meta = guard.meta::<SavantIdMeta>().unwrap();
    assert_eq!(meta.ids().len(), 3);
}

#[test]
fn test_finalize_overflow() {
    common::init();
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 4, 2);

    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    let result = batch.finalize(5, vec![]);
    assert!(
        result.is_err(),
        "finalize(5, ..) on batch_size=4 should fail"
    );
}

#[test]
fn test_fill_single_slot() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 4, 2);

    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    let src = src_gen.acquire_buffer(None).unwrap();
    batch.fill_slot(&*src.lock(), None, Some(42)).unwrap();

    batch
        .finalize(1, vec![SavantIdMetaKind::Frame(42)])
        .unwrap();
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
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    let src = src_gen.acquire_buffer(None).unwrap();
    let roi = Rect {
        top: 100,
        left: 200,
        width: 800,
        height: 600,
    };
    batch.fill_slot(&*src.lock(), Some(&roi), Some(1)).unwrap();

    batch.finalize(1, vec![SavantIdMetaKind::Frame(1)]).unwrap();
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
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    let src = src_gen.acquire_buffer(None).unwrap();
    batch.fill_slot(&*src.lock(), None, Some(1)).unwrap();

    batch.finalize(1, vec![SavantIdMetaKind::Frame(1)]).unwrap();
    let shared = batch.shared_buffer();
    let guard = shared.lock();
    assert!(guard.size() > 0);
}

#[test]
fn test_fill_different_source_resolutions() {
    common::init();
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 4, 2);

    let resolutions = [(320, 240), (1920, 1080), (1280, 720)];
    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    let mut id_kinds = Vec::new();
    for (i, (w, h)) in resolutions.iter().enumerate() {
        let src_gen = make_src_gen(VideoFormat::RGBA, *w, *h);
        let src = src_gen.acquire_buffer(None).unwrap();
        batch.fill_slot(&*src.lock(), None, Some(i as i64)).unwrap();
        id_kinds.push(SavantIdMetaKind::Frame(i as i64));
    }

    batch.finalize(3, id_kinds).unwrap();
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
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    let src = src_gen.acquire_buffer(None).unwrap();
    batch.fill_slot(&*src.lock(), None, Some(1)).unwrap();

    batch.finalize(1, vec![SavantIdMetaKind::Frame(1)]).unwrap();
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
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    let src1 = src_gen.acquire_buffer(None).unwrap();
    let src2 = src_gen.acquire_buffer(None).unwrap();
    let src3 = src_gen.acquire_buffer(None).unwrap();

    batch.fill_slot(&*src1.lock(), None, Some(1)).unwrap();
    batch.fill_slot(&*src2.lock(), None, Some(2)).unwrap();
    let result = batch.fill_slot(&*src3.lock(), None, Some(3));
    assert!(result.is_err(), "3rd fill_slot should fail on batch_size=2");
}

#[test]
fn test_finalize_empty_batch() {
    common::init();
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 4, 2);

    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    batch.finalize(0, vec![]).unwrap();
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

    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    let ids = [100i64, 200, 300];
    for &id in &ids {
        let src = src_gen.acquire_buffer(None).unwrap();
        batch.fill_slot(&*src.lock(), None, Some(id)).unwrap();
    }

    let id_kinds: Vec<_> = ids.iter().map(|&id| SavantIdMetaKind::Frame(id)).collect();
    batch.finalize(3, id_kinds).unwrap();
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
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    for _ in 0..3 {
        let src = src_gen.acquire_buffer(None).unwrap();
        batch.fill_slot(&*src.lock(), None, None).unwrap();
    }

    batch.finalize(3, vec![]).unwrap();
    let shared = batch.shared_buffer();
    let guard = shared.lock();
    assert!(
        guard.meta::<SavantIdMeta>().is_none(),
        "no IDs provided → no SavantIdMeta"
    );
}

// ─── acquire_buffer (simple acquisition) ─────────────────────────────────────

#[test]
fn test_acquire_buffer_with_id() {
    common::init();
    let gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 1, 2);
    let shared = gen.acquire_buffer(Some(42)).unwrap();
    let guard = shared.lock();
    let meta = guard
        .meta::<SavantIdMeta>()
        .expect("should have SavantIdMeta");
    assert_eq!(meta.ids(), &[SavantIdMetaKind::Frame(42)]);
}

#[test]
fn test_acquire_buffer_without_id() {
    common::init();
    let gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 1, 2);
    let shared = gen.acquire_buffer(None).unwrap();
    let guard = shared.lock();
    assert!(
        guard.meta::<SavantIdMeta>().is_none(),
        "no ID → no SavantIdMeta"
    );
}

#[test]
fn test_acquire_buffer_surface_view() {
    common::init();
    let gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 1, 2);
    let shared = gen.acquire_buffer(Some(1)).unwrap();
    let view = SurfaceView::from_shared(&shared, 0).unwrap();
    assert_eq!(view.width(), 640);
    assert_eq!(view.height(), 640);
    assert_eq!(view.channels(), 4);
    assert!(!view.data_ptr().is_null());
}

// ─── slot_view and shared_buffer ─────────────────────────────────────────────

#[test]
fn test_slot_view_after_finalize() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 2, 2);
    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    let src = src_gen.acquire_buffer(None).unwrap();
    batch.fill_slot(&*src.lock(), None, Some(10)).unwrap();
    batch
        .finalize(1, vec![SavantIdMetaKind::Frame(10)])
        .unwrap();

    let view = batch.slot_view(0).unwrap();
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
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    let src = src_gen.acquire_buffer(None).unwrap();
    batch.fill_slot(&*src.lock(), None, Some(1)).unwrap();
    batch.finalize(1, vec![SavantIdMetaKind::Frame(1)]).unwrap();

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
    let dst_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 1, 2);

    let src_shared = src_gen.acquire_buffer(None).unwrap();
    let src_view = SurfaceView::from_shared(&src_shared, 0).unwrap();

    let dst_shared = dst_gen.acquire_buffer(None).unwrap();
    let dst_view = SurfaceView::from_shared(&dst_shared, 0).unwrap();

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
    let dst_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 1, 2);

    let src_shared = src_gen.acquire_buffer(None).unwrap();
    let src_view = SurfaceView::from_shared(&src_shared, 0).unwrap();

    let dst_shared = dst_gen.acquire_buffer(None).unwrap();
    let dst_view = SurfaceView::from_shared(&dst_shared, 0).unwrap();

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
