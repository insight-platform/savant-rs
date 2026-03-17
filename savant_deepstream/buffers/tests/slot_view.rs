//! Integration tests for [`SurfaceView::from_buffer`] and batched buffer access.
//!
//! These tests verify slot-level access to batched buffers via the new API:
//! - `SurfaceView::from_buffer(&shared, slot_index)` for zero-copy GPU views

mod common;

use deepstream_buffers::{
    extract_nvbufsurface, BufferGenerator, NonUniformBatch, NvBufSurfaceMemType, SavantIdMeta,
    SavantIdMetaKind, SharedBuffer, SurfaceView, TransformConfig, UniformBatchGenerator,
    VideoFormat,
};
use gstreamer as gst;

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

fn build_uniform_batch(ids: &[i64]) -> SharedBuffer {
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, ids.len() as u32, 2);
    let id_kinds: Vec<_> = ids.iter().map(|&id| SavantIdMetaKind::Frame(id)).collect();
    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batch(config, id_kinds).unwrap();

    for (i, &_id) in ids.iter().enumerate() {
        let src = src_gen.acquire(None).unwrap();
        let src_view = SurfaceView::from_buffer(&src, 0).unwrap();
        batch.transform_slot(i as u32, &src_view, None).unwrap();
    }
    batch.finalize().unwrap();

    let shared = batch.shared_buffer();
    {
        let mut guard = shared.lock();
        let buf_ref = guard.make_mut();
        buf_ref.set_pts(gst::ClockTime::from_nseconds(1_000_000));
        buf_ref.set_dts(gst::ClockTime::from_nseconds(2_000_000));
        buf_ref.set_duration(gst::ClockTime::from_nseconds(33_333_333));
        buf_ref.set_offset(42);
        buf_ref.set_offset_end(43);
    }
    shared
}

fn build_heterogeneous_batch(resolutions: &[(u32, u32)], ids: &[i64]) -> SharedBuffer {
    let mut batch = NonUniformBatch::new(0);
    let mut keepalive = Vec::new();
    let id_kinds: Vec<_> = ids.iter().map(|&id| SavantIdMetaKind::Frame(id)).collect();

    for &(w, h) in resolutions.iter() {
        let gen = make_src_gen(VideoFormat::RGBA, w, h);
        let shared = gen.acquire(None).unwrap();
        let view = SurfaceView::from_buffer(&shared, 0).unwrap();
        batch.add(&view).unwrap();
        keepalive.push((shared, view));
    }

    let shared = batch.finalize(id_kinds).unwrap();
    {
        let mut guard = shared.lock();
        let buf_ref = guard.make_mut();
        buf_ref.set_pts(gst::ClockTime::from_nseconds(5_000_000));
        buf_ref.set_dts(gst::ClockTime::from_nseconds(6_000_000));
        buf_ref.set_duration(gst::ClockTime::from_nseconds(16_666_667));
        buf_ref.set_offset(99);
        buf_ref.set_offset_end(100);
    }
    shared
}

// ─── Uniform batch: basic SurfaceView access ─────────────────────────────────

#[test]
fn test_uniform_view_first_slot() {
    common::init();
    let shared = build_uniform_batch(&[10, 20, 30]);
    let view = SurfaceView::from_buffer(&shared, 0).unwrap();

    assert!(!view.data_ptr().is_null());
    assert!(view.pitch() > 0);
    assert_eq!(view.width(), 640);
    assert_eq!(view.height(), 640);
    assert_eq!(view.slot_index(), 0);
}

#[test]
fn test_uniform_view_last_slot() {
    common::init();
    let shared = build_uniform_batch(&[10, 20, 30]);
    let view = SurfaceView::from_buffer(&shared, 2).unwrap();

    assert!(!view.data_ptr().is_null());
    assert_eq!(view.slot_index(), 2);
    assert_eq!(view.width(), 640);
}

#[test]
fn test_uniform_all_slots_have_distinct_ptrs() {
    common::init();
    let shared = build_uniform_batch(&[10, 20, 30, 40]);

    let mut ptrs = Vec::new();
    for i in 0..4 {
        let view = SurfaceView::from_buffer(&shared, i).unwrap();
        ptrs.push(view.data_ptr());
    }

    for i in 0..4 {
        for j in (i + 1)..4 {
            assert_ne!(
                ptrs[i as usize], ptrs[j as usize],
                "slots {} and {} must differ",
                i, j
            );
        }
    }
}

// ─── Uniform batch: SavantIdMeta on shared buffer ────────────────────────────

#[test]
fn test_uniform_ids_on_shared_buffer() {
    common::init();
    let shared = build_uniform_batch(&[100, 200, 300]);

    let guard = shared.lock();
    let meta = guard
        .meta::<SavantIdMeta>()
        .expect("should have SavantIdMeta");
    assert_eq!(
        meta.ids(),
        &[
            SavantIdMetaKind::Frame(100),
            SavantIdMetaKind::Frame(200),
            SavantIdMetaKind::Frame(300),
        ]
    );
}

// ─── Uniform batch: error cases ──────────────────────────────────────────────

#[test]
fn test_uniform_slot_out_of_bounds() {
    common::init();
    let shared = build_uniform_batch(&[10, 20]);

    assert!(SurfaceView::from_buffer(&shared, 2).is_err());
    assert!(SurfaceView::from_buffer(&shared, 100).is_err());
}

// ─── Heterogeneous batch: SurfaceView access ─────────────────────────────────

#[test]
fn test_heterogeneous_view_preserves_dimensions() {
    common::init();
    let resolutions = [(320, 240), (1920, 1080), (1280, 720)];
    let shared = build_heterogeneous_batch(&resolutions, &[1, 2, 3]);

    for (i, &(w, h)) in resolutions.iter().enumerate() {
        let view = SurfaceView::from_buffer(&shared, i as u32).unwrap();
        assert_eq!(view.width(), w, "slot {} width", i);
        assert_eq!(view.height(), h, "slot {} height", i);
        assert!(!view.data_ptr().is_null());
    }
}

#[test]
fn test_heterogeneous_ids_on_shared_buffer() {
    common::init();
    let shared = build_heterogeneous_batch(&[(640, 480), (1920, 1080)], &[42, 99]);

    let guard = shared.lock();
    let meta = guard
        .meta::<SavantIdMeta>()
        .expect("should have SavantIdMeta");
    assert_eq!(
        meta.ids(),
        &[SavantIdMetaKind::Frame(42), SavantIdMetaKind::Frame(99)]
    );
}

#[test]
fn test_heterogeneous_slot_out_of_bounds() {
    common::init();
    let shared = build_heterogeneous_batch(&[(640, 480)], &[1]);
    assert!(SurfaceView::from_buffer(&shared, 1).is_err());
}

// ─── Lifetime: view / proxy keeps batch alive ────────────────────────────────

#[test]
fn test_view_survives_shared_clone_drop() {
    common::init();
    let shared = build_uniform_batch(&[10, 20]);
    let view = SurfaceView::from_buffer(&shared, 0).unwrap();
    drop(shared);

    assert!(
        !view.data_ptr().is_null(),
        "GPU pointer must still be valid after shared drop"
    );
    assert_eq!(view.width(), 640);
}

// ─── Buffer survives struct drop / COW ───────────────────────────────────────

#[test]
fn test_uniform_shared_valid_after_struct_drop() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 3, 2);
    let config = TransformConfig::default();

    let shared = {
        let id_kinds: Vec<_> = [10i64, 20, 30]
            .iter()
            .map(|&id| SavantIdMetaKind::Frame(id))
            .collect();
        let mut batch = batched_gen.acquire_batch(config, id_kinds).unwrap();
        for (i, &_id) in [10i64, 20, 30].iter().enumerate() {
            let src = src_gen.acquire(None).unwrap();
            let src_view = SurfaceView::from_buffer(&src, 0).unwrap();
            batch.transform_slot(i as u32, &src_view, None).unwrap();
        }
        batch.finalize().unwrap();
        batch.shared_buffer()
    };

    let guard = shared.lock();
    let surf_ptr = unsafe { extract_nvbufsurface(guard.as_ref()).unwrap() };
    let surf = unsafe { &*surf_ptr };
    assert_eq!(surf.numFilled, 3);
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
fn test_uniform_shared_valid_after_cow() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 2, 2);
    let config = TransformConfig::default();

    let shared = {
        let id_kinds: Vec<_> = [1i64, 2]
            .iter()
            .map(|&id| SavantIdMetaKind::Frame(id))
            .collect();
        let mut batch = batched_gen.acquire_batch(config, id_kinds).unwrap();
        for (i, &_id) in [1i64, 2].iter().enumerate() {
            let src = src_gen.acquire(None).unwrap();
            let src_view = SurfaceView::from_buffer(&src, 0).unwrap();
            batch.transform_slot(i as u32, &src_view, None).unwrap();
        }
        batch.finalize().unwrap();
        let shared = batch.shared_buffer();
        {
            let mut guard = shared.lock();
            guard
                .make_mut()
                .set_pts(gst::ClockTime::from_nseconds(42_000));
        }
        shared
    };

    {
        let guard = shared.lock();
        assert_eq!(
            guard.as_ref().pts(),
            Some(gst::ClockTime::from_nseconds(42_000))
        );
        let surf_ptr = unsafe { extract_nvbufsurface(guard.as_ref()).unwrap() };
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
}

#[test]
fn test_uniform_id_meta_survives_struct_drop() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 2, 2);
    let config = TransformConfig::default();

    let shared = {
        let id_kinds: Vec<_> = [100i64, 200]
            .iter()
            .map(|&id| SavantIdMetaKind::Frame(id))
            .collect();
        let mut batch = batched_gen.acquire_batch(config, id_kinds).unwrap();
        for (i, &_id) in [100i64, 200].iter().enumerate() {
            let src = src_gen.acquire(None).unwrap();
            let src_view = SurfaceView::from_buffer(&src, 0).unwrap();
            batch.transform_slot(i as u32, &src_view, None).unwrap();
        }
        batch.finalize().unwrap();
        batch.shared_buffer()
    };

    let guard = shared.lock();
    let meta = guard
        .meta::<SavantIdMeta>()
        .expect("SavantIdMeta must survive struct drop");
    assert_eq!(
        meta.ids(),
        &[SavantIdMetaKind::Frame(100), SavantIdMetaKind::Frame(200)]
    );
}

#[test]
fn test_uniform_slot_view_from_detached_shared() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 1920, 1080);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 640, 640, 3, 2);
    let config = TransformConfig::default();

    let shared = {
        let id_kinds: Vec<_> = [10i64, 20, 30]
            .iter()
            .map(|&id| SavantIdMetaKind::Frame(id))
            .collect();
        let mut batch = batched_gen.acquire_batch(config, id_kinds).unwrap();
        for (i, &_id) in [10i64, 20, 30].iter().enumerate() {
            let src = src_gen.acquire(None).unwrap();
            let src_view = SurfaceView::from_buffer(&src, 0).unwrap();
            batch.transform_slot(i as u32, &src_view, None).unwrap();
        }
        batch.finalize().unwrap();
        batch.shared_buffer()
    };

    for i in 0..3u32 {
        let view = SurfaceView::from_buffer(&shared, i).unwrap();
        assert!(
            !view.data_ptr().is_null(),
            "view({}) dataPtr must be valid on detached shared",
            i
        );
        assert_eq!(view.width(), 640);
    }
}

#[test]
fn test_heterogeneous_shared_valid_after_finalize() {
    common::init();
    let resolutions = [(320, 240), (1920, 1080), (1280, 720)];

    let shared = {
        let mut batch = NonUniformBatch::new(0);
        let mut keepalive = Vec::new();
        let mut id_kinds = Vec::new();
        for &(w, h) in resolutions.iter() {
            let gen = make_src_gen(VideoFormat::RGBA, w, h);
            let s = gen.acquire(None).unwrap();
            let v = SurfaceView::from_buffer(&s, 0).unwrap();
            batch.add(&v).unwrap();
            id_kinds.push(SavantIdMetaKind::Frame(keepalive.len() as i64 + 1));
            keepalive.push((gen, s, v));
        }
        batch.finalize(id_kinds).unwrap()
    };

    let guard = shared.lock();
    let surf_ptr = unsafe { extract_nvbufsurface(guard.as_ref()).unwrap() };
    let surf = unsafe { &*surf_ptr };
    assert_eq!(surf.numFilled, 3);
    for (i, &(w, h)) in resolutions.iter().enumerate() {
        let params = unsafe { &*surf.surfaceList.add(i) };
        assert!(
            !params.dataPtr.is_null(),
            "heterogeneous slot {} dataPtr must survive finalize",
            i
        );
        assert_eq!(params.width, w);
        assert_eq!(params.height, h);
    }
}

// ─── Leak smoke tests: tight pools that exhaust if buffers leak ─────────────

const LEAK_ITERATIONS: usize = 50;

#[test]
fn test_uniform_shared_buffer_no_pool_leak() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 64, 64);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 64, 64, 1, 2);

    for _ in 0..LEAK_ITERATIONS {
        let mut batch = batched_gen
            .acquire_batch(TransformConfig::default(), vec![SavantIdMetaKind::Frame(1)])
            .unwrap();
        let src = src_gen.acquire(None).unwrap();
        let src_view = SurfaceView::from_buffer(&src, 0).unwrap();
        batch.transform_slot(0, &src_view, None).unwrap();
        batch.finalize().unwrap();
        let _shared = batch.shared_buffer();
    }
}

#[test]
fn test_uniform_shared_buffer_cow_no_pool_leak() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 64, 64);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 64, 64, 1, 2);

    for _ in 0..LEAK_ITERATIONS {
        let mut batch = batched_gen
            .acquire_batch(TransformConfig::default(), vec![SavantIdMetaKind::Frame(1)])
            .unwrap();
        let src = src_gen.acquire(None).unwrap();
        let src_view = SurfaceView::from_buffer(&src, 0).unwrap();
        batch.transform_slot(0, &src_view, None).unwrap();
        batch.finalize().unwrap();
        let shared = batch.shared_buffer();
        {
            let mut guard = shared.lock();
            guard.make_mut().set_pts(gst::ClockTime::from_nseconds(1));
        }
        drop(batch);
        drop(shared);
    }
}

#[test]
fn test_uniform_surface_view_no_pool_leak() {
    common::init();
    let src_gen = make_src_gen(VideoFormat::RGBA, 64, 64);
    let batched_gen = make_batched_gen(VideoFormat::RGBA, 64, 64, 2, 2);

    for _ in 0..LEAK_ITERATIONS {
        let id_kinds: Vec<_> = (0..2i64).map(SavantIdMetaKind::Frame).collect();
        let mut batch = batched_gen
            .acquire_batch(TransformConfig::default(), id_kinds)
            .unwrap();
        for i in 0..2 {
            let src = src_gen.acquire(None).unwrap();
            let src_view = SurfaceView::from_buffer(&src, 0).unwrap();
            batch.transform_slot(i as u32, &src_view, None).unwrap();
        }
        batch.finalize().unwrap();
        let shared = batch.shared_buffer();
        let _v0 = SurfaceView::from_buffer(&shared, 0).unwrap();
        let _v1 = SurfaceView::from_buffer(&shared, 1).unwrap();
    }
}

#[test]
fn test_heterogeneous_finalize_no_leak() {
    common::init();

    for _ in 0..LEAK_ITERATIONS {
        let gen = make_src_gen(VideoFormat::RGBA, 64, 64);
        let (s1, v1) = {
            let s = gen.acquire(None).unwrap();
            let v = SurfaceView::from_buffer(&s, 0).unwrap();
            (s, v)
        };
        let (s2, v2) = {
            let s = gen.acquire(None).unwrap();
            let v = SurfaceView::from_buffer(&s, 0).unwrap();
            (s, v)
        };

        let mut batch = NonUniformBatch::new(0);
        batch.add(&v1).unwrap();
        batch.add(&v2).unwrap();
        let ids = vec![SavantIdMetaKind::Frame(1), SavantIdMetaKind::Frame(2)];
        let _shared = batch.finalize(ids).unwrap();
        drop(v1);
        drop(s1);
        drop(v2);
        drop(s2);
    }
}

#[test]
fn test_heterogeneous_surface_view_no_leak() {
    common::init();

    for _ in 0..LEAK_ITERATIONS {
        let gen = make_src_gen(VideoFormat::RGBA, 64, 64);
        let (s1, v1) = {
            let s = gen.acquire(None).unwrap();
            let v = SurfaceView::from_buffer(&s, 0).unwrap();
            (s, v)
        };
        let (s2, v2) = {
            let s = gen.acquire(None).unwrap();
            let v = SurfaceView::from_buffer(&s, 0).unwrap();
            (s, v)
        };

        let mut batch = NonUniformBatch::new(0);
        batch.add(&v1).unwrap();
        batch.add(&v2).unwrap();
        let ids = vec![SavantIdMetaKind::Frame(1), SavantIdMetaKind::Frame(2)];
        let shared = batch.finalize(ids).unwrap();
        let _view0 = SurfaceView::from_buffer(&shared, 0).unwrap();
        let _view1 = SurfaceView::from_buffer(&shared, 1).unwrap();
        drop(v1);
        drop(s1);
        drop(v2);
        drop(s2);
    }
}
