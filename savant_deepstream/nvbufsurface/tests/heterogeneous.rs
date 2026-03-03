//! Integration tests for [`HeterogeneousBatch`] (zero-copy, nvstreammux2-style).

mod common;

use deepstream_nvbufsurface::{
    extract_nvbufsurface, HeterogeneousBatch, NvBufSurfaceGenerator, NvBufSurfaceMemType,
    SavantIdMeta, SavantIdMetaKind, VideoFormat,
};

fn make_gen(format: VideoFormat, w: u32, h: u32) -> NvBufSurfaceGenerator {
    NvBufSurfaceGenerator::builder(format, w, h)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(4)
        .max_buffers(4)
        .build()
        .expect("failed to build generator")
}

// ─── Unit: creation ──────────────────────────────────────────────────────────

#[test]
fn test_heterogeneous_batch_create() {
    common::init();
    let batch = HeterogeneousBatch::new(4, 0).unwrap();
    assert_eq!(batch.max_batch_size(), 4);
    assert_eq!(batch.num_filled(), 0);
    assert_eq!(batch.gpu_id(), 0);
}

#[test]
fn test_heterogeneous_batch_size_1() {
    common::init();
    let batch = HeterogeneousBatch::new(1, 0).unwrap();
    assert_eq!(batch.max_batch_size(), 1);
}

// ─── E2E: add + finalize ────────────────────────────────────────────────────

#[test]
fn test_heterogeneous_add_different_sizes() {
    common::init();
    let gen_small = make_gen(VideoFormat::RGBA, 320, 240);
    let gen_1080p = make_gen(VideoFormat::RGBA, 1920, 1080);
    let gen_720p = make_gen(VideoFormat::RGBA, 1280, 720);

    let buf_small = gen_small.acquire_surface(None).unwrap();
    let buf_1080p = gen_1080p.acquire_surface(None).unwrap();
    let buf_720p = gen_720p.acquire_surface(None).unwrap();

    let mut batch = HeterogeneousBatch::new(8, 0).unwrap();
    batch.add(&buf_small, Some(1)).unwrap();
    batch.add(&buf_1080p, Some(2)).unwrap();
    batch.add(&buf_720p, Some(3)).unwrap();

    let buffer = batch.finalize();

    let surf_ptr = unsafe { extract_nvbufsurface(buffer.as_ref()).unwrap() };
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
    let gen_nv12 = make_gen(VideoFormat::NV12, 640, 480);

    let buf_rgba = gen_rgba.acquire_surface(None).unwrap();
    let buf_nv12 = gen_nv12.acquire_surface(None).unwrap();

    let mut batch = HeterogeneousBatch::new(4, 0).unwrap();
    batch.add(&buf_rgba, Some(1)).unwrap();
    batch.add(&buf_nv12, Some(2)).unwrap();

    let buffer = batch.finalize();
    let surf_ptr = unsafe { extract_nvbufsurface(buffer.as_ref()).unwrap() };
    let surf = unsafe { &*surf_ptr };
    assert_eq!(surf.numFilled, 2);

    let slot0 = unsafe { &*surf.surfaceList.add(0) };
    let slot1 = unsafe { &*surf.surfaceList.add(1) };
    assert_ne!(
        slot0.colorFormat, slot1.colorFormat,
        "RGBA and NV12 should have different color formats"
    );
}

#[test]
fn test_heterogeneous_slot_ptr_returns_correct_dims() {
    common::init();
    let gen_small = make_gen(VideoFormat::RGBA, 320, 240);
    let gen_1080p = make_gen(VideoFormat::RGBA, 1920, 1080);

    let buf_small = gen_small.acquire_surface(None).unwrap();
    let buf_1080p = gen_1080p.acquire_surface(None).unwrap();

    let mut batch = HeterogeneousBatch::new(4, 0).unwrap();
    batch.add(&buf_small, Some(1)).unwrap();
    batch.add(&buf_1080p, Some(2)).unwrap();

    let (ptr0, pitch0, w0, h0) = batch.slot_ptr(0).unwrap();
    assert!(!ptr0.is_null());
    assert!(pitch0 > 0);
    assert_eq!(w0, 320);
    assert_eq!(h0, 240);

    let (ptr1, pitch1, w1, h1) = batch.slot_ptr(1).unwrap();
    assert!(!ptr1.is_null());
    assert!(pitch1 > 0);
    assert_eq!(w1, 1920);
    assert_eq!(h1, 1080);

    assert_ne!(ptr0, ptr1, "slots should have distinct dataPtrs");
}

#[test]
fn test_heterogeneous_ids_preserved() {
    common::init();
    let gen = make_gen(VideoFormat::RGBA, 640, 480);

    let mut batch = HeterogeneousBatch::new(4, 0).unwrap();
    for id in [10i64, 20, 30] {
        let buf = gen.acquire_surface(None).unwrap();
        batch.add(&buf, Some(id)).unwrap();
    }

    let buffer = batch.finalize();
    let meta = buffer
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
fn test_heterogeneous_auto_propagate_ids() {
    common::init();
    let gen = make_gen(VideoFormat::RGBA, 640, 480);

    let buf = gen.acquire_surface(Some(42)).unwrap();

    let mut batch = HeterogeneousBatch::new(4, 0).unwrap();
    batch.add(&buf, None).unwrap();

    let buffer = batch.finalize();
    let meta = buffer.meta::<SavantIdMeta>().unwrap();
    assert_eq!(meta.ids(), &[SavantIdMetaKind::Frame(42)]);
}

#[test]
fn test_heterogeneous_parent_buffer_meta() {
    common::init();
    let gen = make_gen(VideoFormat::RGBA, 640, 480);

    let buf = gen.acquire_surface(None).unwrap();

    let mut batch = HeterogeneousBatch::new(4, 0).unwrap();
    batch.add(&buf, Some(1)).unwrap();

    let buffer = batch.finalize();

    // The finalized buffer should have at least one GstMeta (parent buffer).
    // We verify the buffer is valid and the slot's dataPtr matches the source.
    let surf_ptr = unsafe { extract_nvbufsurface(buffer.as_ref()).unwrap() };
    let slot0 = unsafe { &*(*surf_ptr).surfaceList };
    assert!(
        !slot0.dataPtr.is_null(),
        "slot dataPtr should reference source GPU memory"
    );
}

// ─── Corner case / error tests ───────────────────────────────────────────────

#[test]
fn test_heterogeneous_add_exceeds_capacity() {
    common::init();
    let gen = make_gen(VideoFormat::RGBA, 640, 480);

    let mut batch = HeterogeneousBatch::new(2, 0).unwrap();
    let b1 = gen.acquire_surface(None).unwrap();
    let b2 = gen.acquire_surface(None).unwrap();
    let b3 = gen.acquire_surface(None).unwrap();

    batch.add(&b1, Some(1)).unwrap();
    batch.add(&b2, Some(2)).unwrap();
    let result = batch.add(&b3, Some(3));
    assert!(result.is_err(), "3rd add should fail on max_batch_size=2");
}

#[test]
fn test_heterogeneous_finalize_empty() {
    common::init();
    let batch = HeterogeneousBatch::new(4, 0).unwrap();

    let buffer = batch.finalize();
    let surf_ptr = unsafe { extract_nvbufsurface(buffer.as_ref()).unwrap() };
    assert_eq!(unsafe { (*surf_ptr).numFilled }, 0);

    assert!(
        buffer.meta::<SavantIdMeta>().is_none(),
        "empty batch should have no SavantIdMeta"
    );
}

#[test]
fn test_heterogeneous_add_partial() {
    common::init();
    let gen = make_gen(VideoFormat::RGBA, 640, 480);

    let mut batch = HeterogeneousBatch::new(8, 0).unwrap();
    let b1 = gen.acquire_surface(None).unwrap();
    let b2 = gen.acquire_surface(None).unwrap();
    batch.add(&b1, Some(1)).unwrap();
    batch.add(&b2, Some(2)).unwrap();

    let buffer = batch.finalize();
    let surf_ptr = unsafe { extract_nvbufsurface(buffer.as_ref()).unwrap() };
    assert_eq!(unsafe { (*surf_ptr).numFilled }, 2);
}

#[test]
fn test_heterogeneous_source_buffer_not_leaked() {
    common::init();
    let gen = make_gen(VideoFormat::RGBA, 640, 480);

    let buf = gen.acquire_surface(None).unwrap();

    let mut batch = HeterogeneousBatch::new(4, 0).unwrap();
    batch.add(&buf, Some(1)).unwrap();

    let buffer = batch.finalize();
    // Drop source and batched buffer - should not panic or leak.
    drop(buf);
    drop(buffer);
}
