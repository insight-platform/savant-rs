//! Reproducer for the field-observed CUDA error 700 cascade in picasso when a
//! source's decoder is destroyed while frames derived from its
//! `BufferGenerator` are still in flight downstream:
//!
//! ```text
//! [ERROR picasso::worker] Transform error for source 'source1': Failed to
//!     copy buffer contents: CUDA error: 700
//! [ERROR picasso::worker] Transform error for source 'source1': CUDA driver
//!     API cuGraphicsEGLRegisterImage failed (code 700)
//! ```
//!
//! User hypothesis (paraphrased): *"the pool was removed while the buffer was
//! mapped"*. In our model:
//!
//! - The "pool" is the `BufferGenerator` owned by `NvDecoder` that produces
//!   the RGBA frames consumed downstream by picasso.
//! - "Mapped" is the EGL-CUDA mapping cached on each frame buffer by
//!   `EglCudaMeta` (Jetson) — `NvBufSurfaceMapEglImage` +
//!   `cuGraphicsEGLRegisterImage`.
//! - Picasso receives a `SurfaceView` over a `gst::Buffer` from this pool, and
//!   then later (on its worker thread) calls
//!   [`SurfaceView::transform_into`] to copy the frame into its own encoder
//!   pool.
//!
//! These tests exercise several orderings against a `BufferGenerator` (which
//! wraps `gst_nvds_buffer_pool_new`):
//!
//! 1. [`transform_after_src_pool_drop_held`]: drop the source generator while
//!    a `SurfaceView` is held → transform from view into a fresh dest must
//!    succeed (gst_buffer keeps the pool alive).
//! 2. [`transform_after_src_pool_drop_released`]: hold the buffer past
//!    src_gen drop, then attempt transform — same expectation.
//! 3. [`worker_thread_consumes_after_src_pool_drop`]: emulate picasso's
//!    cross-thread channel: enqueue views, drop the source generator, let a
//!    worker thread dequeue and transform. This is the closest model of the
//!    real failure scenario.
//! 4. [`many_pool_drop_cycles`]: stress test simulating repeated source
//!    re-creation cycles on the same downstream encoder pool.
//! 5. [`concurrent_src_pool_recreate_vs_consumer_transform`]: aggressive
//!    multi-threaded race — producer thread continually rebuilds a source
//!    `BufferGenerator` while the consumer thread continually transforms.
//! 6. [`concurrent_src_pool_recreate_vs_consumer_transform_nonblocking`]:
//!    same as (5) but with a per-consumer `CudaStream::new_non_blocking`,
//!    matching picasso's worker setup.
//!
//! All of them are expected to **succeed** if `gst_nvds_buffer_pool` honours
//! the standard GStreamer contract (gst::Buffer keeps the pool and its
//! NvBufSurface backing alive). If any of them produce CUDA error 700 we
//! have confirmed the user's hypothesis at the buffer-pool level — the field
//! error is then triggered by the same path with real V4L2-backed decoder
//! buffers.

mod common;

use deepstream_buffers::{
    BufferGenerator, CudaStream, NvBufSurfaceError, NvBufSurfaceMemType, SharedBuffer, SurfaceView,
    TransformConfig, VideoFormat,
};
use serial_test::serial;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

const W: u32 = 640;
const H: u32 = 480;

/// Build a `gst_nvds_buffer_pool`-backed RGBA generator with a fixed pool
/// size, mirroring how the V4L2-backed `NvDecoder` configures its own
/// output pool inside `nvmm_to_rgba`.
fn make_gen(pool_size: u32) -> BufferGenerator {
    BufferGenerator::builder(VideoFormat::RGBA, W, H)
        .fps(30, 1)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(pool_size)
        .max_buffers(pool_size)
        .build()
        .expect("failed to build BufferGenerator")
}

/// Drive each `SurfaceView` through `transform_into` against a freshly
/// acquired destination buffer from `dst_gen`. Mirrors picasso's per-frame
/// `transform_into(input.view, dst_view)` step.
fn transform_each_into(
    src_views: &[SurfaceView],
    dst_gen: &BufferGenerator,
) -> Result<usize, NvBufSurfaceError> {
    let config = TransformConfig::default();
    let mut ok = 0usize;
    for (i, src_view) in src_views.iter().enumerate() {
        let dst_shared = dst_gen
            .acquire(None)
            .map_err(|e| NvBufSurfaceError::BufferAcquisitionFailed(format!("dst {i}: {e}")))?;
        let dst_view = SurfaceView::from_buffer(&dst_shared, 0)?;
        src_view.transform_into(&dst_view, &config, None)?;
        ok += 1;
    }
    Ok(ok)
}

/// Case 1: `SurfaceView` keeps the source `gst::Buffer` alive across the
/// `BufferGenerator` drop.
///
/// If the user's hypothesis holds, calling `transform_into` after the source
/// generator drops should fail with CUDA error 700. If it succeeds, the
/// `gst::Buffer`'s strong ref to the pool successfully extends the
/// NvBufSurface backing past pool deactivation.
#[test]
#[serial]
fn transform_after_src_pool_drop_held() {
    common::init();

    let src_gen = make_gen(4);
    let dst_gen = make_gen(8);

    let mut src_views = Vec::new();
    for i in 0..3u128 {
        let shared = src_gen.acquire(Some(i)).unwrap();
        let view = SurfaceView::from_buffer(&shared, 0).unwrap();
        view.fill(&[0xAB, 0xCD, 0xEF, 0xFF]).unwrap();
        src_views.push(view);
    }

    drop(src_gen);

    let n = transform_each_into(&src_views, &dst_gen)
        .expect("transform must succeed after src pool drop with views held");
    assert_eq!(n, src_views.len(), "expected all transforms to succeed");
}

/// Case 2: drop the `SurfaceView` first (so only the `SharedBuffer` keeps
/// the underlying gst::Buffer alive), then drop the source generator, then
/// reconstruct fresh `SurfaceView`s and transform.
///
/// Probes whether the EGL-CUDA mapping cached as `EglCudaMeta` on the
/// gst::Buffer is still valid after pool deactivation.
#[test]
#[serial]
fn transform_after_src_pool_drop_released() {
    common::init();

    let src_gen = make_gen(4);
    let dst_gen = make_gen(8);

    let mut shared_buffers: Vec<SharedBuffer> = Vec::new();
    for i in 0..3u128 {
        let shared = src_gen.acquire(Some(i)).unwrap();
        {
            let view = SurfaceView::from_buffer(&shared, 0).unwrap();
            view.fill(&[0x10, 0x20, 0x30, 0xFF]).unwrap();
        }
        shared_buffers.push(shared);
    }

    drop(src_gen);

    let src_views: Vec<SurfaceView> = shared_buffers
        .iter()
        .map(|sb| SurfaceView::from_buffer(sb, 0).expect("rebuild view after src pool drop"))
        .collect();

    let n = transform_each_into(&src_views, &dst_gen)
        .expect("transform must succeed after src pool drop with shared bufs held");
    assert_eq!(n, src_views.len());
}

/// Case 3: closest model of the real failure scenario.
///
/// A producer thread populates a bounded channel with `SurfaceView`s from
/// the source pool, then drops the source `BufferGenerator`. A consumer
/// thread (picasso worker emulation) dequeues each view and transforms it
/// into a fresh destination buffer — *while the source pool is dead*.
///
/// Failure of this test (any transform returning CUDA 700) confirms the
/// user's hypothesis at the buffer-pool level.
#[test]
#[serial]
fn worker_thread_consumes_after_src_pool_drop() {
    common::init();

    const QUEUE_DEPTH: usize = 8;
    const N_FRAMES: u32 = 16;

    let src_gen = make_gen(QUEUE_DEPTH as u32);
    let dst_gen = Arc::new(make_gen(QUEUE_DEPTH as u32));
    let (tx, rx) = mpsc::sync_channel::<SurfaceView>(QUEUE_DEPTH);

    let dst_clone = Arc::clone(&dst_gen);
    let consumer = std::thread::Builder::new()
        .name("picasso-worker-emulator".into())
        .spawn(move || -> Result<usize, String> {
            let config = TransformConfig::default();
            let mut count = 0usize;
            while let Ok(view) = rx.recv() {
                let dst_shared = dst_clone
                    .acquire(None)
                    .map_err(|e| format!("dst acquire failed at frame {count}: {e}"))?;
                let dst_view = SurfaceView::from_buffer(&dst_shared, 0).map_err(|e| {
                    format!("dst SurfaceView::from_buffer failed at frame {count}: {e}")
                })?;
                view.transform_into(&dst_view, &config, None)
                    .map_err(|e| format!("transform_into failed at frame {count}: {e}"))?;
                count += 1;
            }
            Ok(count)
        })
        .unwrap();

    for i in 0..N_FRAMES {
        let shared = src_gen.acquire(Some(i as u128)).unwrap();
        let view = SurfaceView::from_buffer(&shared, 0).unwrap();
        view.fill(&[(i & 0xFF) as u8, 0x80, 0x40, 0xFF]).unwrap();
        tx.send(view).expect("send view to consumer");
    }

    drop(src_gen);
    drop(tx);

    let processed = consumer
        .join()
        .expect("consumer thread panicked")
        .expect("consumer thread reported transform failure");
    assert_eq!(
        processed, N_FRAMES as usize,
        "consumer must process all frames after src pool drop"
    );
}

/// Case 4: stress test — simulate repeated source re-creation cycles on the
/// same downstream encoder pool. Each cycle acquires N frames, drops the src
/// generator, transforms them in a worker, then advances to the next cycle.
///
/// Stresses both the pool-drop-with-inflight path and `EglCudaMeta`'s
/// per-cycle deregistration path.
#[test]
#[serial]
fn many_pool_drop_cycles() {
    common::init();

    const CYCLES: u32 = 8;
    const FRAMES_PER_CYCLE: u32 = 6;

    let dst_gen = make_gen(FRAMES_PER_CYCLE * 2);
    let config = TransformConfig::default();

    for cycle in 0..CYCLES {
        let src_gen = make_gen(FRAMES_PER_CYCLE);
        let mut views = Vec::with_capacity(FRAMES_PER_CYCLE as usize);
        for i in 0..FRAMES_PER_CYCLE {
            let shared = src_gen
                .acquire(Some((cycle * FRAMES_PER_CYCLE + i) as u128))
                .unwrap();
            let view = SurfaceView::from_buffer(&shared, 0).unwrap();
            view.fill(&[(cycle & 0xFF) as u8, (i & 0xFF) as u8, 0x33, 0xFF])
                .unwrap();
            views.push(view);
        }

        drop(src_gen);

        for (i, src_view) in views.iter().enumerate() {
            let dst_shared = dst_gen.acquire(None).unwrap();
            let dst_view = SurfaceView::from_buffer(&dst_shared, 0).unwrap();
            src_view
                .transform_into(&dst_view, &config, None)
                .unwrap_or_else(|e| panic!("cycle {cycle}, frame {i}: transform_into failed: {e}"));
        }
    }
}

/// Case 5: concurrent destruction race — a producer thread is constantly
/// rebuilding a source `BufferGenerator` and shipping fresh views down a
/// channel; the consumer thread is constantly transforming them, mirroring
/// picasso's behaviour during repeated source re-creation. This is the
/// closest in-process model of the field race.
#[test]
#[serial]
fn concurrent_src_pool_recreate_vs_consumer_transform() {
    common::init();

    const RUN_DURATION: Duration = Duration::from_secs(3);
    const SRC_POOL_SIZE: u32 = 4;
    const QUEUE_DEPTH: usize = 8;

    let dst_gen = Arc::new(make_gen(QUEUE_DEPTH as u32 + 4));
    let stop = Arc::new(AtomicBool::new(false));
    let processed = Arc::new(AtomicUsize::new(0));

    let (tx, rx) = mpsc::sync_channel::<SurfaceView>(QUEUE_DEPTH);

    let dst_for_consumer = Arc::clone(&dst_gen);
    let stop_for_consumer = Arc::clone(&stop);
    let processed_for_consumer = Arc::clone(&processed);
    let consumer = std::thread::Builder::new()
        .name("transform-worker".into())
        .spawn(move || -> Result<(), String> {
            let config = TransformConfig::default();
            while !stop_for_consumer.load(Ordering::Relaxed) {
                let view = match rx.recv_timeout(Duration::from_millis(50)) {
                    Ok(v) => v,
                    Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                };
                let dst_shared = dst_for_consumer.acquire(None).map_err(|e| {
                    format!(
                        "dst acquire failed at frame {}: {e}",
                        processed_for_consumer.load(Ordering::Relaxed)
                    )
                })?;
                let dst_view = SurfaceView::from_buffer(&dst_shared, 0)
                    .map_err(|e| format!("dst SurfaceView::from_buffer failed: {e}"))?;
                view.transform_into(&dst_view, &config, None).map_err(|e| {
                    format!(
                        "transform_into failed at frame {}: {e}",
                        processed_for_consumer.load(Ordering::Relaxed)
                    )
                })?;
                processed_for_consumer.fetch_add(1, Ordering::Relaxed);
            }
            while let Ok(view) = rx.try_recv() {
                let dst_shared = dst_for_consumer
                    .acquire(None)
                    .map_err(|e| format!("drain dst acquire failed: {e}"))?;
                let dst_view = SurfaceView::from_buffer(&dst_shared, 0)
                    .map_err(|e| format!("drain dst view failed: {e}"))?;
                view.transform_into(&dst_view, &config, None)
                    .map_err(|e| format!("drain transform failed: {e}"))?;
                processed_for_consumer.fetch_add(1, Ordering::Relaxed);
            }
            Ok(())
        })
        .unwrap();

    let stop_for_producer = Arc::clone(&stop);
    let producer = std::thread::Builder::new()
        .name("source-recreator".into())
        .spawn(move || -> Result<usize, String> {
            let mut cycle = 0u32;
            let started = Instant::now();
            while started.elapsed() < RUN_DURATION && !stop_for_producer.load(Ordering::Relaxed) {
                let src_gen = make_gen(SRC_POOL_SIZE);
                for i in 0..SRC_POOL_SIZE {
                    let shared = src_gen
                        .acquire(Some(((cycle as u128) << 16) | i as u128))
                        .map_err(|e| format!("cycle {cycle}: src acquire failed: {e}"))?;
                    let view = SurfaceView::from_buffer(&shared, 0)
                        .map_err(|e| format!("cycle {cycle}: src view failed: {e}"))?;
                    view.fill(&[(cycle & 0xFF) as u8, (i & 0xFF) as u8, 0x55, 0xFF])
                        .map_err(|e| format!("cycle {cycle}: src fill failed: {e}"))?;
                    if tx.send(view).is_err() {
                        return Ok(cycle as usize);
                    }
                }
                drop(src_gen);
                cycle += 1;
            }
            drop(tx);
            Ok(cycle as usize)
        })
        .unwrap();

    let cycles = producer
        .join()
        .expect("producer thread panicked")
        .expect("producer reported failure");
    stop.store(true, Ordering::Relaxed);
    consumer
        .join()
        .expect("consumer thread panicked")
        .expect("consumer reported transform failure");

    let n = processed.load(Ordering::Relaxed);
    assert!(
        cycles >= 2,
        "must run at least 2 cycles within {:?}, got {cycles}",
        RUN_DURATION
    );
    assert!(
        n >= cycles,
        "consumer processed {n} frames across {cycles} cycles — must be >= cycles"
    );
}

/// Case 6: same as case 5 but uses a per-consumer non-blocking CUDA stream,
/// exactly matching picasso's worker setup (`CudaStream::new_non_blocking`
/// on the worker, all `transform_into` calls run on that stream).
#[test]
#[serial]
fn concurrent_src_pool_recreate_vs_consumer_transform_nonblocking() {
    common::init();

    const RUN_DURATION: Duration = Duration::from_secs(3);
    const SRC_POOL_SIZE: u32 = 4;
    const QUEUE_DEPTH: usize = 8;

    let dst_gen = Arc::new(make_gen(QUEUE_DEPTH as u32 + 4));
    let stop = Arc::new(AtomicBool::new(false));
    let processed = Arc::new(AtomicUsize::new(0));

    let (tx, rx) = mpsc::sync_channel::<SurfaceView>(QUEUE_DEPTH);

    let dst_for_consumer = Arc::clone(&dst_gen);
    let stop_for_consumer = Arc::clone(&stop);
    let processed_for_consumer = Arc::clone(&processed);
    let consumer = std::thread::Builder::new()
        .name("transform-worker-nb".into())
        .spawn(move || -> Result<(), String> {
            let cuda_stream = CudaStream::new_non_blocking()
                .map_err(|e| format!("non-blocking stream creation failed: {e}"))?;
            let mut config = TransformConfig::default();
            config.cuda_stream = cuda_stream.clone();
            while !stop_for_consumer.load(Ordering::Relaxed) {
                let view = match rx.recv_timeout(Duration::from_millis(50)) {
                    Ok(v) => v,
                    Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                };
                let dst_shared = dst_for_consumer
                    .acquire(None)
                    .map_err(|e| format!("dst acquire failed: {e}"))?;
                let dst_view = SurfaceView::from_buffer(&dst_shared, 0)
                    .map_err(|e| format!("dst view failed: {e}"))?
                    .with_cuda_stream(cuda_stream.clone());
                view.transform_into(&dst_view, &config, None)
                    .map_err(|e| format!("transform_into failed: {e}"))?;
                processed_for_consumer.fetch_add(1, Ordering::Relaxed);
            }
            while let Ok(view) = rx.try_recv() {
                let dst_shared = dst_for_consumer
                    .acquire(None)
                    .map_err(|e| format!("drain dst acquire failed: {e}"))?;
                let dst_view = SurfaceView::from_buffer(&dst_shared, 0)
                    .map_err(|e| format!("drain dst view failed: {e}"))?
                    .with_cuda_stream(cuda_stream.clone());
                view.transform_into(&dst_view, &config, None)
                    .map_err(|e| format!("drain transform failed: {e}"))?;
                processed_for_consumer.fetch_add(1, Ordering::Relaxed);
            }
            Ok(())
        })
        .unwrap();

    let stop_for_producer = Arc::clone(&stop);
    let producer = std::thread::Builder::new()
        .name("source-recreator-nb".into())
        .spawn(move || -> Result<usize, String> {
            let mut cycle = 0u32;
            let started = Instant::now();
            while started.elapsed() < RUN_DURATION && !stop_for_producer.load(Ordering::Relaxed) {
                let src_gen = make_gen(SRC_POOL_SIZE);
                for i in 0..SRC_POOL_SIZE {
                    let shared = src_gen
                        .acquire(Some(((cycle as u128) << 16) | i as u128))
                        .map_err(|e| format!("cycle {cycle}: src acquire failed: {e}"))?;
                    let view = SurfaceView::from_buffer(&shared, 0)
                        .map_err(|e| format!("cycle {cycle}: src view failed: {e}"))?;
                    view.fill(&[(cycle & 0xFF) as u8, (i & 0xFF) as u8, 0xAA, 0xFF])
                        .map_err(|e| format!("cycle {cycle}: src fill failed: {e}"))?;
                    if tx.send(view).is_err() {
                        return Ok(cycle as usize);
                    }
                }
                drop(src_gen);
                cycle += 1;
            }
            drop(tx);
            Ok(cycle as usize)
        })
        .unwrap();

    let cycles = producer
        .join()
        .expect("producer thread panicked")
        .expect("producer reported failure");
    stop.store(true, Ordering::Relaxed);
    consumer
        .join()
        .expect("consumer thread panicked")
        .expect("consumer reported transform failure");

    let n = processed.load(Ordering::Relaxed);
    assert!(
        cycles >= 2,
        "must run at least 2 cycles within {:?}, got {cycles}",
        RUN_DURATION
    );
    assert!(
        n >= cycles,
        "consumer processed {n} frames across {cycles} cycles — must be >= cycles"
    );
}
