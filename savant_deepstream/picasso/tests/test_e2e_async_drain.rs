//! E2E tests for the asynchronous drain thread.
//!
//! The drain thread pulls encoded output from the hardware encoder
//! independently of frame submission. These tests validate that:
//!
//! 1. Encoded frames arrive without needing another `send_frame` to trigger
//!    draining.
//! 2. Draw-spec hot-swap (same codec) keeps the encoder and drain thread
//!    alive — no spurious restart or frame loss.
//! 3. Rapid sustained submission doesn't lose frames.
//! 4. EOS flushes the exact number of in-flight frames.

mod common;

use common::*;
use deepstream_buffers::{BufferGenerator, TransformConfig};
use deepstream_encoders::prelude::*;
use picasso::prelude::*;
use savant_core::primitives::frame::VideoFrameProxy;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

const W: u32 = 640;
const H: u32 = 480;
const DUR: u64 = 33_333_333;

fn make_numbered_frame(source_id: &str, idx: u64) -> VideoFrameProxy {
    let mut frame = make_frame_sized(source_id, W as i64, H as i64);
    frame.set_pts((idx * DUR) as i64).unwrap();
    frame.set_duration(Some(DUR as i64)).unwrap();
    frame
}

fn setup_engine(enc_count: &Arc<AtomicUsize>, eos_count: &Arc<AtomicUsize>) -> PicassoEngine {
    let callbacks = Callbacks {
        on_encoded_frame: Some(Arc::new(CountingEncodedCb {
            count: enc_count.clone(),
            eos_count: eos_count.clone(),
        })),
        ..Default::default()
    };
    PicassoEngine::new(
        GeneralSpec {
            idle_timeout_secs: 300,
            ..Default::default()
        },
        callbacks,
    )
}

fn jpeg_spec() -> SourceSpec {
    SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(jpeg_encoder_config(W, H)),
        },
        ..Default::default()
    }
}

fn make_generator() -> BufferGenerator {
    BufferGenerator::new(
        VideoFormat::RGBA,
        W,
        H,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .unwrap()
}

/// The drain thread delivers encoded output without any subsequent
/// `send_frame` call. Submit a batch, then only wait — encoded frames
/// must arrive purely from the drain thread polling the encoder.
#[test]
#[serial_test::serial]
fn e2e_async_drain_delivers_independently() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).unwrap();

    let enc_count = Arc::new(AtomicUsize::new(0));
    let eos_count = Arc::new(AtomicUsize::new(0));
    let mut engine = setup_engine(&enc_count, &eos_count);

    engine.set_source_spec("drain", jpeg_spec()).unwrap();
    let gen = make_generator();

    let n = 10u64;
    for i in 0..n {
        let frame = make_numbered_frame("drain", i);
        let buf = make_gpu_surface_view(&gen, i, DUR);
        engine.send_frame("drain", frame, buf, None).unwrap();
    }

    // Do NOT send more frames — only wait for the drain thread to deliver.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while enc_count.load(Ordering::SeqCst) < n as usize {
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting for async drain: got {} of {n}",
            enc_count.load(Ordering::SeqCst),
        );
        std::thread::sleep(Duration::from_millis(50));
    }

    assert_eq!(enc_count.load(Ordering::SeqCst), n as usize);
    engine.shutdown();
}

/// Changing draw spec (same codec) doesn't restart the encoder.
/// The drain thread continues operating and no frames are lost across
/// the spec change.
#[test]
#[serial_test::serial]
fn e2e_draw_spec_hot_swap_preserves_drain() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).unwrap();

    let enc_count = Arc::new(AtomicUsize::new(0));
    let eos_count = Arc::new(AtomicUsize::new(0));
    let mut engine = setup_engine(&enc_count, &eos_count);

    let draw_spec = {
        use savant_core::draw::*;
        let mut spec = ObjectDrawSpec::new();
        let border = ColorDraw::new(0, 255, 0, 255).unwrap();
        let bg = ColorDraw::new(0, 255, 0, 50).unwrap();
        let bb = BoundingBoxDraw::new(border, bg, 2, PaddingDraw::default_padding()).unwrap();
        spec.insert("det", "car", ObjectDraw::new(Some(bb), None, None, false));
        spec
    };

    let spec_with_draw = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(jpeg_encoder_config(W, H)),
        },
        draw: draw_spec,
        ..Default::default()
    };

    engine.set_source_spec("swap", spec_with_draw).unwrap();
    let gen = make_generator();

    // Phase 1: 10 frames with draw spec + objects.
    let phase1 = 10u64;
    for i in 0..phase1 {
        let frame = make_numbered_frame("swap", i);
        add_object(&frame, 100.0, 100.0, 50.0, 30.0);
        let buf = make_gpu_surface_view(&gen, i, DUR);
        engine.send_frame("swap", frame, buf, None).unwrap();
    }

    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while enc_count.load(Ordering::SeqCst) < phase1 as usize {
        assert!(
            std::time::Instant::now() < deadline,
            "phase 1 timed out: got {} of {phase1}",
            enc_count.load(Ordering::SeqCst),
        );
        std::thread::sleep(Duration::from_millis(50));
    }

    // Hot-swap to empty draw spec (same codec → no encoder restart).
    engine.set_source_spec("swap", jpeg_spec()).unwrap();

    // No EOS should have been fired by the hot-swap.
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(
        eos_count.load(Ordering::SeqCst),
        0,
        "draw-only hot-swap must not fire EOS"
    );

    // Phase 2: 10 more frames without draw spec.
    let phase2 = 10u64;
    let total = phase1 + phase2;
    for i in phase1..total {
        let frame = make_numbered_frame("swap", i);
        let buf = make_gpu_surface_view(&gen, i, DUR);
        engine.send_frame("swap", frame, buf, None).unwrap();
    }

    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while enc_count.load(Ordering::SeqCst) < total as usize {
        assert!(
            std::time::Instant::now() < deadline,
            "phase 2 timed out: got {} of {total}",
            enc_count.load(Ordering::SeqCst),
        );
        std::thread::sleep(Duration::from_millis(50));
    }

    assert_eq!(enc_count.load(Ordering::SeqCst), total as usize);
    engine.shutdown();
}

/// Submit many frames rapidly, then EOS. Every submitted frame must
/// produce exactly one encoded output — no drops from the async drain.
#[test]
#[serial_test::serial]
fn e2e_sustained_throughput_no_frame_loss() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).unwrap();

    let enc_count = Arc::new(AtomicUsize::new(0));
    let eos_count = Arc::new(AtomicUsize::new(0));
    let mut engine = setup_engine(&enc_count, &eos_count);

    engine.set_source_spec("burst", jpeg_spec()).unwrap();
    let gen = make_generator();

    let n = 100u64;
    for i in 0..n {
        let frame = make_numbered_frame("burst", i);
        let buf = make_gpu_surface_view(&gen, i, DUR);
        engine.send_frame("burst", frame, buf, None).unwrap();
    }
    engine.send_eos("burst").unwrap();

    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while eos_count.load(Ordering::SeqCst) < 1 {
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting for EOS"
        );
        std::thread::sleep(Duration::from_millis(100));
    }

    assert_eq!(
        enc_count.load(Ordering::SeqCst),
        n as usize,
        "every submitted frame must produce an encoded output"
    );
    assert_eq!(eos_count.load(Ordering::SeqCst), 1);
    engine.shutdown();
}

/// EOS must flush every in-flight frame. Send exactly N frames, EOS,
/// and verify the count matches precisely.
#[test]
#[serial_test::serial]
fn e2e_eos_flushes_all_in_flight() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).unwrap();

    let enc_count = Arc::new(AtomicUsize::new(0));
    let eos_count = Arc::new(AtomicUsize::new(0));
    let mut engine = setup_engine(&enc_count, &eos_count);

    engine.set_source_spec("flush", jpeg_spec()).unwrap();
    let gen = make_generator();

    let n = 30u64;
    for i in 0..n {
        let frame = make_numbered_frame("flush", i);
        let buf = make_gpu_surface_view(&gen, i, DUR);
        engine.send_frame("flush", frame, buf, None).unwrap();
    }

    // Immediately send EOS — some frames may still be in the encoder
    // pipeline. The stop_encoder → drain_remaining → finish sequence
    // must flush them all.
    engine.send_eos("flush").unwrap();

    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while eos_count.load(Ordering::SeqCst) < 1 {
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting for EOS sentinel"
        );
        std::thread::sleep(Duration::from_millis(100));
    }

    assert_eq!(
        enc_count.load(Ordering::SeqCst),
        n as usize,
        "EOS must flush all {n} in-flight frames"
    );
    assert_eq!(eos_count.load(Ordering::SeqCst), 1);
    engine.shutdown();
}
