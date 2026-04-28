//! E2E test: Repeated `(frames + EOS)` cycles on a single source id.
//!
//! Mirrors the `cars-demo-zmq` field scenario where a long-running
//! pipeline keeps a fixed `source_id` while the upstream producer is
//! restarted multiple times, replaying the same media file:
//!
//! ```text
//! producer 1: frames... EOS         (encoder created, drained, dropped)
//! producer 2: frames... EOS         (encoder re-created, drained, dropped)
//! producer 3: frames... EOS         (encoder re-created, drained, dropped)
//! ...
//! ```
//!
//! Each cycle exercises the [`crate::worker`] encoder lifecycle:
//! `ensure_encoder` → frames → `handle_eos` (which calls
//! [`stop_encoder`] and resets `last_pts_ns`) → next cycle's first
//! frame again triggers `ensure_encoder`.
//!
//! The two tests below cover both available codec backends:
//!
//! * [`repeated_same_source_jpeg_recreate`] uses
//!   [`jpeg_encoder_config`] which works on every CPU/GPU mix and is
//!   the safety net for hosts without NVENC.
//! * [`repeated_same_source_h264_recreate`] uses
//!   [`h264_encoder_config`] (NVENC) at 1920x1080 and is gated on
//!   [`has_nvenc`].  It is the closest in-process reproduction of
//!   the cars-demo-zmq pipeline.
//!
//! Failure modes the tests guard against:
//!
//! 1. `send_frame` / `send_eos` returning `Err` mid-cycle.
//! 2. The encoded-frame counter failing to grow across cycles
//!    (i.e. a recreated encoder that silently produces nothing).
//! 3. Final per-cycle EOS sentinel count diverging from the expected
//!    cycle count.

mod common;

use common::*;
use deepstream_buffers::TransformConfig;
use deepstream_encoders::prelude::*;
use picasso::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

const FRAMES_PER_CYCLE: u64 = 30;
const CYCLES: u64 = 10;
const FRAME_DUR: u64 = 33_333_333;

/// Drain delay after each cycle's `send_eos` so the async encoder
/// drain thread has time to push its tail frames into the callback
/// before the next cycle starts.
const POST_EOS_DRAIN: Duration = Duration::from_millis(750);

/// Run `cycles × frames_per_cycle` `(frames, EOS)` cycles against a
/// single `source_id` and validate per-cycle progress.
#[allow(clippy::too_many_arguments)]
fn run_cycles(
    engine: &PicassoEngine,
    gen: &BufferGenerator,
    source_id: &str,
    enc_count: &Arc<AtomicUsize>,
    eos_count: &Arc<AtomicUsize>,
    cycles: u64,
    frames_per_cycle: u64,
    width: i64,
    height: i64,
) {
    let mut prev_enc = 0usize;
    for cycle in 0..cycles {
        for i in 0..frames_per_cycle {
            let mut frame = make_frame_sized(source_id, width, height);
            let pts = (i * FRAME_DUR) as i64;
            frame.set_pts(pts).unwrap();
            frame.set_duration(Some(FRAME_DUR as i64)).unwrap();
            let global_idx = cycle * frames_per_cycle + i;
            let buf = make_gpu_surface_view_uniform(gen, global_idx, FRAME_DUR);
            engine
                .send_frame(source_id, frame, buf, None)
                .unwrap_or_else(|e| panic!("cycle {cycle}: send_frame({i}) failed: {e}"));
        }
        engine
            .send_eos(source_id)
            .unwrap_or_else(|e| panic!("cycle {cycle}: send_eos failed: {e}"));
        std::thread::sleep(POST_EOS_DRAIN);

        let enc_now = enc_count.load(Ordering::SeqCst);
        let eos_now = eos_count.load(Ordering::SeqCst);
        assert!(
            enc_now > prev_enc,
            "cycle {cycle}: encoded count did not grow ({enc_now} <= {prev_enc}); \
             encoder likely failed to recreate"
        );
        assert_eq!(
            eos_now,
            (cycle + 1) as usize,
            "cycle {cycle}: expected {} EOS sentinels, got {eos_now}",
            cycle + 1
        );
        prev_enc = enc_now;
    }
}

#[test]
#[serial_test::serial]
fn repeated_same_source_jpeg_recreate() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).unwrap();

    const W: u32 = 640;
    const H: u32 = 480;

    let enc_count = Arc::new(AtomicUsize::new(0));
    let eos_count = Arc::new(AtomicUsize::new(0));

    let callbacks = Callbacks {
        on_encoded_frame: Some(Arc::new(CountingEncodedCb {
            count: enc_count.clone(),
            eos_count: eos_count.clone(),
        })),
        ..Default::default()
    };

    let engine = PicassoEngine::new(
        GeneralSpec {
            idle_timeout_secs: 300,
            ..Default::default()
        },
        callbacks,
    );

    let spec = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(jpeg_encoder_config(W, H)),
        },
        ..Default::default()
    };
    engine.set_source_spec("cars-demo-zmq", spec).unwrap();

    let gen = BufferGenerator::builder(VideoFormat::RGBA, W, H)
        .fps(30, 1)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(32)
        .max_buffers(32)
        .build()
        .unwrap();

    run_cycles(
        &engine,
        &gen,
        "cars-demo-zmq",
        &enc_count,
        &eos_count,
        CYCLES,
        FRAMES_PER_CYCLE,
        W as i64,
        H as i64,
    );

    engine.shutdown();

    assert_eq!(
        eos_count.load(Ordering::SeqCst),
        CYCLES as usize,
        "expected one EOS sentinel per cycle"
    );
    assert!(
        enc_count.load(Ordering::SeqCst) >= CYCLES as usize,
        "expected at least one encoded frame per cycle"
    );
}

#[test]
#[serial_test::serial]
fn repeated_same_source_h264_recreate() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).unwrap();

    if !has_nvenc() {
        eprintln!("skipping repeated_same_source_h264_recreate: NVENC not available");
        return;
    }

    // Match the cars-demo-zmq pipeline as closely as possible: 1080p
    // RGBA + H264 NVENC.
    const W: u32 = 1920;
    const H: u32 = 1080;

    let enc_count = Arc::new(AtomicUsize::new(0));
    let eos_count = Arc::new(AtomicUsize::new(0));

    let callbacks = Callbacks {
        on_encoded_frame: Some(Arc::new(CountingEncodedCb {
            count: enc_count.clone(),
            eos_count: eos_count.clone(),
        })),
        ..Default::default()
    };

    let engine = PicassoEngine::new(
        GeneralSpec {
            idle_timeout_secs: 300,
            ..Default::default()
        },
        callbacks,
    );

    let spec = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(h264_encoder_config(W, H)),
        },
        ..Default::default()
    };
    engine.set_source_spec("cars-demo-zmq", spec).unwrap();

    let gen = BufferGenerator::builder(VideoFormat::RGBA, W, H)
        .fps(30, 1)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(32)
        .max_buffers(32)
        .build()
        .unwrap();

    run_cycles(
        &engine,
        &gen,
        "cars-demo-zmq",
        &enc_count,
        &eos_count,
        CYCLES,
        FRAMES_PER_CYCLE,
        W as i64,
        H as i64,
    );

    engine.shutdown();

    assert_eq!(
        eos_count.load(Ordering::SeqCst),
        CYCLES as usize,
        "expected one EOS sentinel per cycle"
    );
    assert!(
        enc_count.load(Ordering::SeqCst) >= CYCLES as usize,
        "expected at least one encoded frame per cycle"
    );
}
