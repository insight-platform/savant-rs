//! Integration tests for [`DsNvNonUniformSurfaceBuffer`] (zero-copy, nvstreammux2-style).

mod common;

use deepstream_nvbufsurface::{
    extract_nvbufsurface, DsNvNonUniformSurfaceBuffer, DsNvUniformSurfaceBufferGenerator,
    NvBufSurfaceMemType, SavantIdMeta, SavantIdMetaKind, SharedMutableGstBuffer, SurfaceView,
    VideoFormat,
};

fn make_gen(format: VideoFormat, w: u32, h: u32) -> DsNvUniformSurfaceBufferGenerator {
    DsNvUniformSurfaceBufferGenerator::new(format, w, h, 1, 4, 0, NvBufSurfaceMemType::Default)
        .expect("failed to build generator")
}

fn acquire_view(
    gen: &DsNvUniformSurfaceBufferGenerator,
    id: Option<i64>,
) -> (SharedMutableGstBuffer, SurfaceView) {
    let shared = gen.acquire_buffer(id).unwrap();
    let view = SurfaceView::from_shared(&shared, 0).unwrap();
    (shared, view)
}

// ─── Unit: creation ──────────────────────────────────────────────────────────

#[test]
fn test_heterogeneous_batch_create() {
    common::init();
    let batch = DsNvNonUniformSurfaceBuffer::new(0);
    assert_eq!(batch.num_filled(), 0);
    assert_eq!(batch.gpu_id(), 0);
}

// ─── E2E: add + finalize ────────────────────────────────────────────────────

#[test]
fn test_heterogeneous_add_different_sizes() {
    common::init();
    let gen_small = make_gen(VideoFormat::RGBA, 320, 240);
    let gen_1080p = make_gen(VideoFormat::RGBA, 1920, 1080);
    let gen_720p = make_gen(VideoFormat::RGBA, 1280, 720);

    let (_s1, v1) = acquire_view(&gen_small, None);
    let (_s2, v2) = acquire_view(&gen_1080p, None);
    let (_s3, v3) = acquire_view(&gen_720p, None);

    let mut batch = DsNvNonUniformSurfaceBuffer::new(0);
    batch.add(&v1, Some(1)).unwrap();
    batch.add(&v2, Some(2)).unwrap();
    batch.add(&v3, Some(3)).unwrap();

    let ids = vec![
        SavantIdMetaKind::Frame(1),
        SavantIdMetaKind::Frame(2),
        SavantIdMetaKind::Frame(3),
    ];
    let shared = batch.finalize(ids).unwrap();

    let guard = shared.lock();
    let surf_ptr = unsafe { extract_nvbufsurface(guard.as_ref()).unwrap() };
    let surf = unsafe { &*surf_ptr };
    assert_eq!(surf.numFilled, 3);

    let slot0 = unsafe { &*surf.surfaceList.add(0) };
    assert_eq!(slot0.width, 320);
    assert_eq!(slot0.height, 240);

    let slot1 = unsafe { &*surf.surfaceList.add(1) };
    assert_eq!(slot1.width, 1920);
    assert_eq!(slot1.height, 1080);

    let slot2 = unsafe { &*surf.surfaceList.add(2) };
    assert_eq!(slot2.width, 1280);
    assert_eq!(slot2.height, 720);
}

#[test]
fn test_heterogeneous_add_different_formats() {
    common::init();
    let gen_rgba = make_gen(VideoFormat::RGBA, 640, 480);
    let gen_bgrx = make_gen(VideoFormat::BGRx, 640, 480);

    let (_s1, v1) = acquire_view(&gen_rgba, None);
    let (_s2, v2) = acquire_view(&gen_bgrx, None);

    let mut batch = DsNvNonUniformSurfaceBuffer::new(0);
    batch.add(&v1, Some(1)).unwrap();
    batch.add(&v2, Some(2)).unwrap();

    let ids = vec![SavantIdMetaKind::Frame(1), SavantIdMetaKind::Frame(2)];
    let shared = batch.finalize(ids).unwrap();
    let guard = shared.lock();
    let surf_ptr = unsafe { extract_nvbufsurface(guard.as_ref()).unwrap() };
    let surf = unsafe { &*surf_ptr };
    assert_eq!(surf.numFilled, 2);

    let slot0 = unsafe { &*surf.surfaceList.add(0) };
    let slot1 = unsafe { &*surf.surfaceList.add(1) };
    assert_ne!(
        slot0.colorFormat, slot1.colorFormat,
        "RGBA and BGRx should have different color formats"
    );
}

#[test]
fn test_heterogeneous_surface_view_returns_correct_dims() {
    common::init();
    let gen_small = make_gen(VideoFormat::RGBA, 320, 240);
    let gen_1080p = make_gen(VideoFormat::RGBA, 1920, 1080);

    let (_s1, v1) = acquire_view(&gen_small, None);
    let (_s2, v2) = acquire_view(&gen_1080p, None);

    let mut batch = DsNvNonUniformSurfaceBuffer::new(0);
    batch.add(&v1, Some(1)).unwrap();
    batch.add(&v2, Some(2)).unwrap();

    let ids = vec![SavantIdMetaKind::Frame(1), SavantIdMetaKind::Frame(2)];
    let shared = batch.finalize(ids).unwrap();

    let view0 = SurfaceView::from_shared(&shared, 0).unwrap();
    assert!(!view0.data_ptr().is_null());
    assert!(view0.pitch() > 0);
    assert_eq!(view0.width(), 320);
    assert_eq!(view0.height(), 240);

    let view1 = SurfaceView::from_shared(&shared, 1).unwrap();
    assert!(!view1.data_ptr().is_null());
    assert!(view1.pitch() > 0);
    assert_eq!(view1.width(), 1920);
    assert_eq!(view1.height(), 1080);

    assert_ne!(
        view0.data_ptr(),
        view1.data_ptr(),
        "slots should have distinct dataPtrs"
    );
}

#[test]
fn test_heterogeneous_ids_preserved() {
    common::init();
    let gen = make_gen(VideoFormat::RGBA, 640, 480);

    let mut batch = DsNvNonUniformSurfaceBuffer::new(0);
    let mut views_and_bufs = Vec::new();
    for id in [10i64, 20, 30] {
        let (s, v) = acquire_view(&gen, None);
        batch.add(&v, Some(id)).unwrap();
        views_and_bufs.push((s, v));
    }

    let ids = vec![
        SavantIdMetaKind::Frame(10),
        SavantIdMetaKind::Frame(20),
        SavantIdMetaKind::Frame(30),
    ];
    let shared = batch.finalize(ids).unwrap();
    let guard = shared.lock();
    let meta = guard
        .meta::<SavantIdMeta>()
        .expect("should have SavantIdMeta");
    assert_eq!(
        meta.ids(),
        &[
            SavantIdMetaKind::Frame(10),
            SavantIdMetaKind::Frame(20),
            SavantIdMetaKind::Frame(30),
        ]
    );
}

#[test]
fn test_heterogeneous_parent_buffer_meta() {
    common::init();
    let gen = make_gen(VideoFormat::RGBA, 640, 480);

    let (_s, v) = acquire_view(&gen, None);

    let mut batch = DsNvNonUniformSurfaceBuffer::new(0);
    batch.add(&v, Some(1)).unwrap();

    let shared = batch.finalize(vec![SavantIdMetaKind::Frame(1)]).unwrap();
    let guard = shared.lock();
    let surf_ptr = unsafe { extract_nvbufsurface(guard.as_ref()).unwrap() };
    let slot0 = unsafe { &*(*surf_ptr).surfaceList };
    assert!(
        !slot0.dataPtr.is_null(),
        "slot dataPtr should reference source GPU memory"
    );
}

// ─── Corner case / error tests ───────────────────────────────────────────────

#[test]
fn test_heterogeneous_finalize_empty() {
    common::init();
    let batch = DsNvNonUniformSurfaceBuffer::new(0);
    let result = batch.finalize(vec![]);
    assert!(
        result.is_err(),
        "finalize on empty non-uniform batch should fail"
    );
}

#[test]
fn test_heterogeneous_add_partial() {
    common::init();
    let gen = make_gen(VideoFormat::RGBA, 640, 480);

    let (_s1, v1) = acquire_view(&gen, None);
    let (_s2, v2) = acquire_view(&gen, None);

    let mut batch = DsNvNonUniformSurfaceBuffer::new(0);
    batch.add(&v1, Some(1)).unwrap();
    batch.add(&v2, Some(2)).unwrap();

    let ids = vec![SavantIdMetaKind::Frame(1), SavantIdMetaKind::Frame(2)];
    let shared = batch.finalize(ids).unwrap();
    let guard = shared.lock();
    let surf_ptr = unsafe { extract_nvbufsurface(guard.as_ref()).unwrap() };
    assert_eq!(unsafe { (*surf_ptr).numFilled }, 2);
}

#[test]
fn test_heterogeneous_source_buffer_not_leaked() {
    common::init();
    let gen = make_gen(VideoFormat::RGBA, 640, 480);

    let (src_shared, src_view) = acquire_view(&gen, None);

    let mut batch = DsNvNonUniformSurfaceBuffer::new(0);
    batch.add(&src_view, Some(1)).unwrap();
    drop(src_view);
    drop(src_shared);

    let shared = batch.finalize(vec![SavantIdMetaKind::Frame(1)]).unwrap();
    drop(shared);
}
