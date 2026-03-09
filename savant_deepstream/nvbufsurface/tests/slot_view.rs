//! Integration tests for [`extract_slot_view`].

mod common;

use deepstream_nvbufsurface::{
    extract_nvbufsurface, extract_slot_view, DsNvNonUniformSurfaceBuffer,
    DsNvSurfaceBufferGenerator, DsNvUniformSurfaceBufferGenerator, NvBufSurfaceError,
    NvBufSurfaceMemType, SavantIdMeta, SavantIdMetaKind, TransformConfig, VideoFormat,
};
use gstreamer as gst;

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

fn build_uniform_batch(ids: &[i64]) -> gst::Buffer {
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, ids.len() as u32, 2);
    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();
    for &id in ids {
        let src = src_gen.acquire_surface(None).unwrap();
        batch.fill_slot(&src, None, Some(id)).unwrap();
    }
    batch.finalize().unwrap();
    let mut buf = batch.as_gst_buffer().unwrap();
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
    let mut batch = DsNvNonUniformSurfaceBuffer::new(resolutions.len() as u32, 0).unwrap();
    for (i, &(w, h)) in resolutions.iter().enumerate() {
        let gen = make_src_gen(VideoFormat::RGBA, w, h);
        let buf = gen.acquire_surface(None).unwrap();
        batch.add(&buf, Some(ids[i])).unwrap();
    }
    batch.finalize().unwrap();
    let mut buf = batch.as_gst_buffer().unwrap();
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
    batch.finalize().unwrap();
    let buf = batch.as_gst_buffer().unwrap();

    let view = extract_slot_view(&buf, 0).unwrap();
    assert!(
        view.meta::<SavantIdMeta>().is_none(),
        "no id meta on batch → no id meta on view"
    );
}

// ─── Uniform: extract_slot_view / as_gst_buffer / fill_slot guards ───────────

#[test]
fn test_uniform_extract_slot_before_finalize_fails() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 2, 2);
    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();
    let src = src_gen.acquire_surface(None).unwrap();
    batch.fill_slot(&src, None, Some(10)).unwrap();

    let err = batch.extract_slot_view(0).unwrap_err();
    assert!(matches!(err, NvBufSurfaceError::NotFinalized));
}

#[test]
fn test_uniform_extract_slot_after_finalize_works() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 2, 2);
    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();
    let src = src_gen.acquire_surface(None).unwrap();
    batch.fill_slot(&src, None, Some(10)).unwrap();
    batch.finalize().unwrap();

    let view = batch.extract_slot_view(0).unwrap();
    let meta = view.meta::<SavantIdMeta>().expect("should have meta");
    assert_eq!(meta.ids(), &[SavantIdMetaKind::Frame(10)]);
}

#[test]
fn test_uniform_fill_after_finalize_fails() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 2, 2);
    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();
    let src = src_gen.acquire_surface(None).unwrap();
    batch.fill_slot(&src, None, Some(10)).unwrap();
    batch.finalize().unwrap();

    let src2 = src_gen.acquire_surface(None).unwrap();
    let err = batch.fill_slot(&src2, None, Some(20)).unwrap_err();
    assert!(matches!(err, NvBufSurfaceError::AlreadyFinalized));
}

#[test]
fn test_uniform_as_gst_buffer_before_finalize_fails() {
    common::init();
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 2, 2);
    let config = TransformConfig::default();
    let batch = batched_gen.acquire_batched_surface(config).unwrap();

    let err = batch.as_gst_buffer().unwrap_err();
    assert!(matches!(err, NvBufSurfaceError::NotFinalized));
}

// ─── Heterogeneous: extract_slot_view / as_gst_buffer / add guards ───────────

#[test]
fn test_heterogeneous_extract_slot_before_finalize_fails() {
    common::init();
    let gen = make_src_gen(VideoFormat::RGBA, 640, 480);
    let mut batch = DsNvNonUniformSurfaceBuffer::new(2, 0).unwrap();
    batch
        .add(&gen.acquire_surface(None).unwrap(), Some(1))
        .unwrap();

    let err = batch.extract_slot_view(0).unwrap_err();
    assert!(matches!(err, NvBufSurfaceError::NotFinalized));
}

#[test]
fn test_heterogeneous_extract_slot_after_finalize_works() {
    common::init();
    let gen = make_src_gen(VideoFormat::RGBA, 640, 480);
    let mut batch = DsNvNonUniformSurfaceBuffer::new(2, 0).unwrap();
    batch
        .add(&gen.acquire_surface(None).unwrap(), Some(42))
        .unwrap();
    batch.finalize().unwrap();

    let view = batch.extract_slot_view(0).unwrap();
    let meta = view.meta::<SavantIdMeta>().expect("should have meta");
    assert_eq!(meta.ids(), &[SavantIdMetaKind::Frame(42)]);
}

#[test]
fn test_heterogeneous_add_after_finalize_fails() {
    common::init();
    let gen = make_src_gen(VideoFormat::RGBA, 640, 480);
    let mut batch = DsNvNonUniformSurfaceBuffer::new(2, 0).unwrap();
    batch
        .add(&gen.acquire_surface(None).unwrap(), Some(1))
        .unwrap();
    batch.finalize().unwrap();

    let err = batch
        .add(&gen.acquire_surface(None).unwrap(), Some(2))
        .unwrap_err();
    assert!(matches!(err, NvBufSurfaceError::AlreadyFinalized));
}

#[test]
fn test_heterogeneous_as_gst_buffer_before_finalize_fails() {
    common::init();
    let batch = DsNvNonUniformSurfaceBuffer::new(4, 0).unwrap();

    let err = batch.as_gst_buffer().unwrap_err();
    assert!(matches!(err, NvBufSurfaceError::NotFinalized));
}

// ─── as_gst_buffer safety: buffer must survive struct drop and COW ───────────

#[test]
fn test_uniform_buffer_valid_after_struct_drop() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 3, 2);
    let config = TransformConfig::default();

    let buf = {
        let mut batch = batched_gen.acquire_batched_surface(config).unwrap();
        for &id in &[10i64, 20, 30] {
            let src = src_gen.acquire_surface(None).unwrap();
            batch.fill_slot(&src, None, Some(id)).unwrap();
        }
        batch.finalize().unwrap();
        batch.as_gst_buffer().unwrap()
    };

    let surf_ptr = unsafe { extract_nvbufsurface(buf.as_ref()).unwrap() };
    let surf = unsafe { &*surf_ptr };
    assert_eq!(surf.numFilled, 3);
    assert_eq!(surf.batchSize, 3);
    for i in 0..3u32 {
        let params = unsafe { &*surf.surfaceList.add(i as usize) };
        assert!(
            !params.dataPtr.is_null(),
            "slot {} dataPtr must be valid after struct drop",
            i
        );
        assert_eq!(params.width, 640);
        assert_eq!(params.height, 640);
        assert!(params.pitch > 0);
    }
}

#[test]
fn test_uniform_buffer_valid_after_cow() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 2, 2);
    let config = TransformConfig::default();

    let buf = {
        let mut batch = batched_gen.acquire_batched_surface(config).unwrap();
        for &id in &[1i64, 2] {
            let src = src_gen.acquire_surface(None).unwrap();
            batch.fill_slot(&src, None, Some(id)).unwrap();
        }
        batch.finalize().unwrap();
        let mut b = batch.as_gst_buffer().unwrap();
        b.make_mut().set_pts(gst::ClockTime::from_nseconds(42_000));
        b
    };

    assert_eq!(
        buf.as_ref().pts(),
        Some(gst::ClockTime::from_nseconds(42_000))
    );
    let surf_ptr = unsafe { extract_nvbufsurface(buf.as_ref()).unwrap() };
    let surf = unsafe { &*surf_ptr };
    assert_eq!(surf.numFilled, 2);
    for i in 0..2u32 {
        let params = unsafe { &*surf.surfaceList.add(i as usize) };
        assert!(
            !params.dataPtr.is_null(),
            "slot {} dataPtr must survive COW + struct drop",
            i
        );
    }
}

#[test]
fn test_uniform_buffer_id_meta_survives_struct_drop() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 2, 2);
    let config = TransformConfig::default();

    let buf = {
        let mut batch = batched_gen.acquire_batched_surface(config).unwrap();
        for &id in &[100i64, 200] {
            let src = src_gen.acquire_surface(None).unwrap();
            batch.fill_slot(&src, None, Some(id)).unwrap();
        }
        batch.finalize().unwrap();
        batch.as_gst_buffer().unwrap()
    };

    let meta = buf
        .meta::<SavantIdMeta>()
        .expect("SavantIdMeta must survive struct drop");
    assert_eq!(
        meta.ids(),
        &[SavantIdMetaKind::Frame(100), SavantIdMetaKind::Frame(200)]
    );
}

#[test]
fn test_uniform_slot_view_from_detached_buffer() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 3, 2);
    let config = TransformConfig::default();

    let buf = {
        let mut batch = batched_gen.acquire_batched_surface(config).unwrap();
        for &id in &[10i64, 20, 30] {
            let src = src_gen.acquire_surface(None).unwrap();
            batch.fill_slot(&src, None, Some(id)).unwrap();
        }
        batch.finalize().unwrap();
        batch.as_gst_buffer().unwrap()
    };

    for i in 0..3u32 {
        let view = extract_slot_view(&buf, i).unwrap();
        let surf_ptr = unsafe { extract_nvbufsurface(view.as_ref()).unwrap() };
        let params = unsafe { &*(*surf_ptr).surfaceList };
        assert!(
            !params.dataPtr.is_null(),
            "slot_view({}) dataPtr must be valid on detached buffer",
            i
        );
        assert_eq!(params.width, 640);
    }
}

#[test]
fn test_heterogeneous_buffer_valid_after_struct_drop() {
    common::init();
    let resolutions = [(320, 240), (1920, 1080), (1280, 720)];

    let buf = {
        let mut batch = DsNvNonUniformSurfaceBuffer::new(resolutions.len() as u32, 0).unwrap();
        for (i, &(w, h)) in resolutions.iter().enumerate() {
            let gen = make_src_gen(VideoFormat::RGBA, w, h);
            let src = gen.acquire_surface(None).unwrap();
            batch.add(&src, Some(i as i64 + 1)).unwrap();
        }
        batch.finalize().unwrap();
        batch.as_gst_buffer().unwrap()
    };

    let surf_ptr = unsafe { extract_nvbufsurface(buf.as_ref()).unwrap() };
    let surf = unsafe { &*surf_ptr };
    assert_eq!(surf.numFilled, 3);
    for (i, &(w, h)) in resolutions.iter().enumerate() {
        let params = unsafe { &*surf.surfaceList.add(i) };
        assert!(
            !params.dataPtr.is_null(),
            "heterogeneous slot {} dataPtr must survive struct drop",
            i
        );
        assert_eq!(params.width, w);
        assert_eq!(params.height, h);
    }
}

#[test]
fn test_heterogeneous_buffer_valid_after_cow() {
    common::init();

    let buf = {
        let gen = make_src_gen(VideoFormat::RGBA, 640, 480);
        let mut batch = DsNvNonUniformSurfaceBuffer::new(2, 0).unwrap();
        batch
            .add(&gen.acquire_surface(None).unwrap(), Some(1))
            .unwrap();
        batch
            .add(&gen.acquire_surface(None).unwrap(), Some(2))
            .unwrap();
        batch.finalize().unwrap();
        let mut b = batch.as_gst_buffer().unwrap();
        b.make_mut().set_pts(gst::ClockTime::from_nseconds(99_000));
        b
    };

    assert_eq!(
        buf.as_ref().pts(),
        Some(gst::ClockTime::from_nseconds(99_000))
    );
    let surf_ptr = unsafe { extract_nvbufsurface(buf.as_ref()).unwrap() };
    let surf = unsafe { &*surf_ptr };
    assert_eq!(surf.numFilled, 2);
    for i in 0..2usize {
        let params = unsafe { &*surf.surfaceList.add(i) };
        assert!(
            !params.dataPtr.is_null(),
            "heterogeneous slot {} dataPtr must survive COW + struct drop",
            i
        );
    }
}

// ─── Leak smoke tests: tight pools that exhaust if buffers leak ─────────────
//
// Each test uses pool_size=2 and loops many more times than the pool can hold.
// If `as_gst_buffer()` or `extract_slot_view()` leaks a parent-buffer ref,
// the pool buffer never returns and `acquire_batched_surface` blocks forever
// (caught by the test runner's timeout) or panics.

const LEAK_ITERATIONS: usize = 50;

#[test]
fn test_uniform_as_gst_buffer_no_pool_leak() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 64, 64);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 64, 64, 1, 2);

    for _ in 0..LEAK_ITERATIONS {
        let mut batch = batched_gen
            .acquire_batched_surface(TransformConfig::default())
            .unwrap();
        let src = src_gen.acquire_surface(None).unwrap();
        batch.fill_slot(&src, None, Some(1)).unwrap();
        batch.finalize().unwrap();
        let _buf = batch.as_gst_buffer().unwrap();
    }
}

#[test]
fn test_uniform_as_gst_buffer_cow_no_pool_leak() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 64, 64);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 64, 64, 1, 2);

    for _ in 0..LEAK_ITERATIONS {
        let mut batch = batched_gen
            .acquire_batched_surface(TransformConfig::default())
            .unwrap();
        let src = src_gen.acquire_surface(None).unwrap();
        batch.fill_slot(&src, None, Some(1)).unwrap();
        batch.finalize().unwrap();
        let mut buf = batch.as_gst_buffer().unwrap();
        buf.make_mut().set_pts(gst::ClockTime::from_nseconds(1));
        drop(batch);
        drop(buf);
    }
}

#[test]
fn test_uniform_slot_view_no_pool_leak() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 64, 64);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 64, 64, 2, 2);

    for _ in 0..LEAK_ITERATIONS {
        let mut batch = batched_gen
            .acquire_batched_surface(TransformConfig::default())
            .unwrap();
        for id in 0..2i64 {
            let src = src_gen.acquire_surface(None).unwrap();
            batch.fill_slot(&src, None, Some(id)).unwrap();
        }
        batch.finalize().unwrap();
        let buf = batch.as_gst_buffer().unwrap();
        let _v0 = extract_slot_view(&buf, 0).unwrap();
        let _v1 = extract_slot_view(&buf, 1).unwrap();
    }
}

#[test]
fn test_heterogeneous_as_gst_buffer_no_leak() {
    common::init();

    for _ in 0..LEAK_ITERATIONS {
        let gen = make_src_gen(VideoFormat::RGBA, 64, 64);
        let mut batch = DsNvNonUniformSurfaceBuffer::new(2, 0).unwrap();
        batch
            .add(&gen.acquire_surface(None).unwrap(), Some(1))
            .unwrap();
        batch
            .add(&gen.acquire_surface(None).unwrap(), Some(2))
            .unwrap();
        batch.finalize().unwrap();
        let _buf = batch.as_gst_buffer().unwrap();
    }
}

#[test]
fn test_heterogeneous_slot_view_no_leak() {
    common::init();

    for _ in 0..LEAK_ITERATIONS {
        let gen = make_src_gen(VideoFormat::RGBA, 64, 64);
        let mut batch = DsNvNonUniformSurfaceBuffer::new(2, 0).unwrap();
        batch
            .add(&gen.acquire_surface(None).unwrap(), Some(1))
            .unwrap();
        batch
            .add(&gen.acquire_surface(None).unwrap(), Some(2))
            .unwrap();
        batch.finalize().unwrap();
        let buf = batch.as_gst_buffer().unwrap();
        let _v0 = extract_slot_view(&buf, 0).unwrap();
        let _v1 = extract_slot_view(&buf, 1).unwrap();
    }
}
