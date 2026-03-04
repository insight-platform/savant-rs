//! Integration tests for [`extract_slot_view`].

mod common;

use deepstream_nvbufsurface::{
    extract_nvbufsurface, extract_slot_view, BatchedNvBufSurfaceGenerator, HeterogeneousBatch,
    NvBufSurfaceGenerator, NvBufSurfaceMemType, SavantIdMeta, SavantIdMetaKind, TransformConfig,
    VideoFormat,
};
use gstreamer as gst;

fn make_src_gen(format: VideoFormat, w: u32, h: u32) -> NvBufSurfaceGenerator {
    NvBufSurfaceGenerator::builder(format, w, h)
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
) -> BatchedNvBufSurfaceGenerator {
    BatchedNvBufSurfaceGenerator::new(format, w, h, batch, pool, 0, NvBufSurfaceMemType::Default)
        .expect("failed to build batched generator")
}

fn build_uniform_batch(ids: &[i64]) -> gst::Buffer {
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, ids.len() as u32, 2);
    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();
    for &id in ids {
        let src = src_gen.acquire_surface(None).unwrap();
        batch.fill_slot(&src, None, Some(id)).unwrap();
    }
    let mut buf = batch.finalize();
    {
        let buf_ref = buf.make_mut();
        buf_ref.set_pts(gst::ClockTime::from_nseconds(1_000_000));
        buf_ref.set_dts(gst::ClockTime::from_nseconds(2_000_000));
        buf_ref.set_duration(gst::ClockTime::from_nseconds(33_333_333));
        buf_ref.set_offset(42);
        buf_ref.set_offset_end(43);
    }
    buf
}

fn build_heterogeneous_batch(resolutions: &[(u32, u32)], ids: &[i64]) -> gst::Buffer {
    let mut batch = HeterogeneousBatch::new(resolutions.len() as u32, 0).unwrap();
    for (i, &(w, h)) in resolutions.iter().enumerate() {
        let gen = make_src_gen(VideoFormat::RGBA, w, h);
        let buf = gen.acquire_surface(None).unwrap();
        batch.add(&buf, Some(ids[i])).unwrap();
    }
    let mut buf = batch.finalize();
    {
        let buf_ref = buf.make_mut();
        buf_ref.set_pts(gst::ClockTime::from_nseconds(5_000_000));
        buf_ref.set_dts(gst::ClockTime::from_nseconds(6_000_000));
        buf_ref.set_duration(gst::ClockTime::from_nseconds(16_666_667));
        buf_ref.set_offset(99);
        buf_ref.set_offset_end(100);
    }
    buf
}

// ─── Uniform batch: basic extraction ─────────────────────────────────────────

#[test]
fn test_uniform_extract_first_slot() {
    common::init();
    let batch = build_uniform_batch(&[10, 20, 30]);
    let view = extract_slot_view(&batch, 0).unwrap();

    let surf_ptr = unsafe { extract_nvbufsurface(view.as_ref()).unwrap() };
    let surf = unsafe { &*surf_ptr };
    assert_eq!(surf.batchSize, 1);
    assert_eq!(surf.numFilled, 1);

    let params = unsafe { &*surf.surfaceList };
    assert!(!params.dataPtr.is_null());
    assert!(params.pitch > 0);
    assert_eq!(params.width, 640);
    assert_eq!(params.height, 640);
}

#[test]
fn test_uniform_extract_last_slot() {
    common::init();
    let batch = build_uniform_batch(&[10, 20, 30]);
    let view = extract_slot_view(&batch, 2).unwrap();

    let surf_ptr = unsafe { extract_nvbufsurface(view.as_ref()).unwrap() };
    let surf = unsafe { &*surf_ptr };
    assert_eq!(surf.numFilled, 1);
}

#[test]
fn test_uniform_extract_all_slots_have_distinct_ptrs() {
    common::init();
    let batch = build_uniform_batch(&[10, 20, 30, 40]);

    let mut ptrs = Vec::new();
    for i in 0..4 {
        let view = extract_slot_view(&batch, i).unwrap();
        let surf_ptr = unsafe { extract_nvbufsurface(view.as_ref()).unwrap() };
        let params = unsafe { &*(*surf_ptr).surfaceList };
        ptrs.push(params.dataPtr);
    }

    for i in 0..4 {
        for j in (i + 1)..4 {
            assert_ne!(ptrs[i], ptrs[j], "slots {} and {} must differ", i, j);
        }
    }
}

// ─── Uniform batch: timestamp propagation ────────────────────────────────────

#[test]
fn test_uniform_timestamps_propagated() {
    common::init();
    let batch = build_uniform_batch(&[10, 20]);
    let view = extract_slot_view(&batch, 0).unwrap();

    assert_eq!(
        view.as_ref().pts(),
        Some(gst::ClockTime::from_nseconds(1_000_000))
    );
    assert_eq!(
        view.as_ref().dts(),
        Some(gst::ClockTime::from_nseconds(2_000_000))
    );
    assert_eq!(
        view.as_ref().duration(),
        Some(gst::ClockTime::from_nseconds(33_333_333))
    );
    assert_eq!(view.as_ref().offset(), 42);
    assert_eq!(view.as_ref().offset_end(), 43);
}

// ─── Uniform batch: SavantIdMeta propagation ─────────────────────────────────

#[test]
fn test_uniform_id_propagated_per_slot() {
    common::init();
    let batch = build_uniform_batch(&[100, 200, 300]);

    for (i, expected_id) in [100i64, 200, 300].iter().enumerate() {
        let view = extract_slot_view(&batch, i as u32).unwrap();
        let meta = view
            .meta::<SavantIdMeta>()
            .expect("should have SavantIdMeta");
        assert_eq!(meta.ids(), &[SavantIdMetaKind::Frame(*expected_id)]);
    }
}

// ─── Uniform batch: error cases ──────────────────────────────────────────────

#[test]
fn test_uniform_slot_out_of_bounds() {
    common::init();
    let batch = build_uniform_batch(&[10, 20]);

    assert!(extract_slot_view(&batch, 2).is_err());
    assert!(extract_slot_view(&batch, 100).is_err());
}

// ─── Heterogeneous batch: basic extraction ───────────────────────────────────

#[test]
fn test_heterogeneous_extract_preserves_dimensions() {
    common::init();
    let resolutions = [(320, 240), (1920, 1080), (1280, 720)];
    let batch = build_heterogeneous_batch(&resolutions, &[1, 2, 3]);

    for (i, &(w, h)) in resolutions.iter().enumerate() {
        let view = extract_slot_view(&batch, i as u32).unwrap();
        let surf_ptr = unsafe { extract_nvbufsurface(view.as_ref()).unwrap() };
        let surf = unsafe { &*surf_ptr };
        assert_eq!(surf.batchSize, 1);
        assert_eq!(surf.numFilled, 1);

        let params = unsafe { &*surf.surfaceList };
        assert_eq!(params.width, w, "slot {} width", i);
        assert_eq!(params.height, h, "slot {} height", i);
        assert!(!params.dataPtr.is_null());
    }
}

#[test]
fn test_heterogeneous_timestamps_propagated() {
    common::init();
    let batch = build_heterogeneous_batch(&[(640, 480), (1280, 720)], &[1, 2]);
    let view = extract_slot_view(&batch, 1).unwrap();

    assert_eq!(
        view.as_ref().pts(),
        Some(gst::ClockTime::from_nseconds(5_000_000))
    );
    assert_eq!(
        view.as_ref().dts(),
        Some(gst::ClockTime::from_nseconds(6_000_000))
    );
    assert_eq!(
        view.as_ref().duration(),
        Some(gst::ClockTime::from_nseconds(16_666_667))
    );
    assert_eq!(view.as_ref().offset(), 99);
    assert_eq!(view.as_ref().offset_end(), 100);
}

#[test]
fn test_heterogeneous_id_propagated_per_slot() {
    common::init();
    let batch = build_heterogeneous_batch(&[(640, 480), (1920, 1080)], &[42, 99]);

    let v0 = extract_slot_view(&batch, 0).unwrap();
    let meta0 = v0.meta::<SavantIdMeta>().expect("slot 0 should have id");
    assert_eq!(meta0.ids(), &[SavantIdMetaKind::Frame(42)]);

    let v1 = extract_slot_view(&batch, 1).unwrap();
    let meta1 = v1.meta::<SavantIdMeta>().expect("slot 1 should have id");
    assert_eq!(meta1.ids(), &[SavantIdMetaKind::Frame(99)]);
}

#[test]
fn test_heterogeneous_slot_out_of_bounds() {
    common::init();
    let batch = build_heterogeneous_batch(&[(640, 480)], &[1]);
    assert!(extract_slot_view(&batch, 1).is_err());
}

// ─── Lifetime: view keeps batch alive ────────────────────────────────────────

#[test]
fn test_view_survives_batch_drop() {
    common::init();
    let batch = build_uniform_batch(&[10, 20]);

    let view = extract_slot_view(&batch, 0).unwrap();
    drop(batch);

    let surf_ptr = unsafe { extract_nvbufsurface(view.as_ref()).unwrap() };
    let params = unsafe { &*(*surf_ptr).surfaceList };
    assert!(
        !params.dataPtr.is_null(),
        "GPU pointer must still be valid after batch drop"
    );
}

// ─── Edge case: batch with no SavantIdMeta ───────────────────────────────────

#[test]
fn test_no_id_meta_on_batch() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 320, 240);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 2, 2);
    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();
    for _ in 0..2 {
        let src = src_gen.acquire_surface(None).unwrap();
        batch.fill_slot(&src, None, None).unwrap();
    }
    let buf = batch.finalize();

    let view = extract_slot_view(&buf, 0).unwrap();
    assert!(
        view.meta::<SavantIdMeta>().is_none(),
        "no id meta on batch → no id meta on view"
    );
}
