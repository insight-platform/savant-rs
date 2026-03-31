//! Benchmarks for `SurfaceView` construction and EGL-CUDA registration.
//!
//! On Jetson (aarch64), `SurfaceView::from_buffer` triggers
//! `NvBufSurfaceMapEglImage` + `cuGraphicsEGLRegisterImage` + frame query
//! on first access.  On dGPU, `dataPtr` is already CUDA-addressable —
//! `from_buffer` is essentially a field read.
//!
//! With the `GST_META_FLAG_POOLED` fix, recycled pool buffers retain
//! their `EglCudaMeta`, so the second and subsequent views on the same
//! physical buffer hit the fast O(1) path (no re-registration).

use criterion::{criterion_group, criterion_main, Criterion};
use deepstream_buffers::{
    cuda_init, BufferGenerator, NvBufSurfaceMemType, SavantIdMetaKind, SurfaceView,
    TransformConfig, UniformBatchGenerator, VideoFormat,
};
use gstreamer as gst;
use std::sync::Once;

static INIT: Once = Once::new();

fn init() {
    INIT.call_once(|| {
        let _ = env_logger::try_init();
        gst::init().unwrap();
        cuda_init(0).expect("CUDA init failed");
    });
}

fn make_gen(pool_size: u32) -> BufferGenerator {
    BufferGenerator::builder(VideoFormat::RGBA, 1920, 1080)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(pool_size)
        .max_buffers(pool_size)
        .build()
        .expect("generator creation failed")
}

/// Measures the cost of `SurfaceView::from_buffer` on a fresh buffer
/// (includes EGL-CUDA registration on Jetson).
fn bench_registration_plus_first_view(c: &mut Criterion) {
    init();

    c.bench_function("surface_view_registration_plus_first_view", |b| {
        b.iter_with_setup(
            || {
                let gen = make_gen(2);
                let buf = gen.acquire(None).unwrap();
                (gen, buf)
            },
            |(_gen, buf)| {
                let view = SurfaceView::from_buffer(&buf, 0).unwrap();
                assert!(!view.data_ptr().is_null());
            },
        );
    });
}

/// Measures the fast path: pool of 1 buffer recycled repeatedly.
/// After the first iteration registers the meta, all subsequent views
/// find the cached `EglCudaMeta` (POOLED flag keeps it across recycles).
fn bench_cached_meta_view(c: &mut Criterion) {
    init();

    let gen = make_gen(1);

    // Warm-up: first acquire + from_buffer pays the registration cost.
    {
        let buf = gen.acquire(None).unwrap();
        let view = SurfaceView::from_buffer(&buf, 0).unwrap();
        assert!(!view.data_ptr().is_null());
    }

    c.bench_function("surface_view_cached_meta", |b| {
        b.iter(|| {
            let buf = gen.acquire(None).unwrap();
            let view = SurfaceView::from_buffer(&buf, 0).unwrap();
            assert!(!view.data_ptr().is_null());
        });
    });
}

/// Measures the cost of creating multi-slot views on a batched buffer.
/// Each slot goes through EGL-CUDA registration on first access (Jetson).
fn bench_batched_multi_slot_views(c: &mut Criterion) {
    init();

    let batch_size: u32 = 4;
    let src_gen = BufferGenerator::builder(VideoFormat::RGBA, 128, 96)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(batch_size)
        .max_buffers(batch_size)
        .build()
        .expect("source generator failed");

    let batched_gen = UniformBatchGenerator::new(
        VideoFormat::RGBA,
        128,
        96,
        batch_size,
        2,
        0,
        NvBufSurfaceMemType::Default,
    )
    .expect("batched generator failed");

    c.bench_function("surface_view_batched_4slot", |b| {
        b.iter_with_setup(
            || {
                let ids: Vec<_> = (0..batch_size)
                    .map(|i| SavantIdMetaKind::Frame(i as u128))
                    .collect();
                let config = TransformConfig::default();
                let mut batch = batched_gen.acquire_batch(config, ids).unwrap();
                for i in 0..batch_size {
                    let src = src_gen.acquire(None).unwrap();
                    let src_view = SurfaceView::from_buffer(&src, 0).unwrap();
                    batch.transform_slot(i, &src_view, None).unwrap();
                }
                batch.finalize().unwrap();
                batch.shared_buffer()
            },
            |shared| {
                for i in 0..batch_size {
                    let view = SurfaceView::from_buffer(&shared, i).unwrap();
                    assert!(!view.data_ptr().is_null());
                }
            },
        );
    });
}

criterion_group!(
    benches,
    bench_registration_plus_first_view,
    bench_cached_meta_view,
    bench_batched_multi_slot_views,
);
criterion_main!(benches);
