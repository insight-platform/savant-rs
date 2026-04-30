//! E2E tests for PTS reset policy.
//!
//! Verifies that when PTS goes backwards the engine:
//!
//! - With [`PtsResetPolicy::EosOnDecreasingPts`]: fires a synthetic EOS,
//!   recreates the encoder, and re-encodes the offending frame.
//! - With [`PtsResetPolicy::RecreateOnDecreasingPts`]: silently recreates
//!   the encoder (no EOS sentinel) and re-encodes the offending frame.
//! - In both cases: fires the `on_stream_reset` callback with the correct
//!   reason.
//!
//! Also tests that frames not marked for encoding (encode_attribute gate)
//! are discarded.

mod common;

use common::*;
use deepstream_buffers::{BufferGenerator, TransformConfig};
use deepstream_encoders::prelude::*;
use parking_lot::Mutex;
use picasso::callbacks::{OnStreamReset, StreamResetReason};
use picasso::prelude::*;
use savant_core::primitives::WithAttributes;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

const W: u32 = 640;
const H: u32 = 480;
const DUR: u64 = 33_333_333;

fn init() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    let _ = cuda_init(0);
}

fn send_frame_with_pts(
    engine: &PicassoEngine,
    gen: &BufferGenerator,
    source_id: &str,
    idx: u64,
    pts_ns: u64,
) {
    let mut frame = make_frame_sized(source_id, W as i64, H as i64);
    frame.set_pts(pts_ns as i64).unwrap();
    frame.set_duration(Some(DUR as i64)).unwrap();
    let buf = make_gpu_surface_view(gen, idx, DUR);
    engine.send_frame(source_id, frame, buf, None).unwrap();
}

/// Collects `StreamResetReason` events.
struct ResetCollector(Arc<Mutex<Vec<(String, StreamResetReason)>>>);

impl OnStreamReset for ResetCollector {
    fn call(&self, source_id: &str, reason: StreamResetReason) {
        self.0.lock().push((source_id.to_string(), reason));
    }
}

// -----------------------------------------------------------------------
// Test: EosOnDecreasingPts — fires EOS + resets encoder
// -----------------------------------------------------------------------

#[test]
#[serial_test::serial]
fn pts_reset_eos_policy_fires_eos_and_reencodes() {
    init();

    let enc_count = Arc::new(AtomicUsize::new(0));
    let eos_count = Arc::new(AtomicUsize::new(0));
    let reset_events: Arc<Mutex<Vec<(String, StreamResetReason)>>> =
        Arc::new(Mutex::new(Vec::new()));

    let callbacks = Callbacks {
        on_encoded_frame: Some(Arc::new(CountingEncodedCb {
            count: enc_count.clone(),
            eos_count: eos_count.clone(),
        })),
        on_stream_reset: Some(Arc::new(ResetCollector(reset_events.clone()))),
        ..Default::default()
    };

    let engine = PicassoEngine::new(
        GeneralSpec {
            idle_timeout_secs: 300,
            pts_reset_policy: PtsResetPolicy::EosOnDecreasingPts,
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
    engine.set_source_spec("src", spec).unwrap();

    let gen = BufferGenerator::new(
        VideoFormat::RGBA,
        W,
        H,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .unwrap();

    // Session 1: PTS 0, 33ms, 66ms (3 frames)
    for i in 0..3u64 {
        send_frame_with_pts(&engine, &gen, "src", i, i * DUR);
    }
    std::thread::sleep(Duration::from_secs(2));

    let enc_before_reset = enc_count.load(Ordering::SeqCst);
    let eos_before_reset = eos_count.load(Ordering::SeqCst);
    assert!(
        enc_before_reset > 0,
        "should have encoded frames before reset"
    );
    assert_eq!(eos_before_reset, 0, "no EOS yet");

    // PTS goes backwards: send frame with PTS = 0 (reset)
    send_frame_with_pts(&engine, &gen, "src", 3, 0);
    // Continue with new session PTS: 33ms, 66ms
    send_frame_with_pts(&engine, &gen, "src", 4, DUR);
    send_frame_with_pts(&engine, &gen, "src", 5, 2 * DUR);

    std::thread::sleep(Duration::from_secs(2));

    // Verify a synthetic EOS was fired
    let eos_after_reset = eos_count.load(Ordering::SeqCst);
    assert_eq!(
        eos_after_reset, 1,
        "EOS policy should fire exactly 1 synthetic EOS on PTS reset, got {eos_after_reset}"
    );

    // Verify frames after the reset were encoded
    let enc_after_reset = enc_count.load(Ordering::SeqCst);
    assert!(
        enc_after_reset > enc_before_reset,
        "should encode frames after PTS reset (got {enc_after_reset} vs {enc_before_reset})"
    );

    // Verify on_stream_reset callback fired
    let resets = reset_events.lock();
    assert_eq!(resets.len(), 1, "on_stream_reset should fire once");
    assert_eq!(resets[0].0, "src");
    match &resets[0].1 {
        StreamResetReason::PtsDecreased {
            last_pts_ns,
            new_pts_ns,
        } => {
            assert_eq!(*last_pts_ns, 2 * DUR);
            assert_eq!(*new_pts_ns, 0);
        }
        other => panic!("expected PtsDecreased, got {other:?}"),
    }
    drop(resets);

    // Final EOS + shutdown
    engine.send_eos("src").unwrap();
    std::thread::sleep(Duration::from_secs(1));
    engine.shutdown();

    let final_eos = eos_count.load(Ordering::SeqCst);
    assert_eq!(
        final_eos, 2,
        "should have 2 EOS total (1 synthetic + 1 explicit), got {final_eos}"
    );
}

// -----------------------------------------------------------------------
// Test: RecreateOnDecreasingPts — no EOS, silent recreation
// -----------------------------------------------------------------------

#[test]
#[serial_test::serial]
fn pts_reset_recreate_policy_no_eos() {
    init();

    let enc_count = Arc::new(AtomicUsize::new(0));
    let eos_count = Arc::new(AtomicUsize::new(0));
    let reset_events: Arc<Mutex<Vec<(String, StreamResetReason)>>> =
        Arc::new(Mutex::new(Vec::new()));

    let callbacks = Callbacks {
        on_encoded_frame: Some(Arc::new(CountingEncodedCb {
            count: enc_count.clone(),
            eos_count: eos_count.clone(),
        })),
        on_stream_reset: Some(Arc::new(ResetCollector(reset_events.clone()))),
        ..Default::default()
    };

    let engine = PicassoEngine::new(
        GeneralSpec {
            idle_timeout_secs: 300,
            pts_reset_policy: PtsResetPolicy::RecreateOnDecreasingPts,
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
    engine.set_source_spec("src", spec).unwrap();

    let gen = BufferGenerator::new(
        VideoFormat::RGBA,
        W,
        H,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .unwrap();

    // Session 1: PTS 0, 33ms, 66ms
    for i in 0..3u64 {
        send_frame_with_pts(&engine, &gen, "src", i, i * DUR);
    }
    std::thread::sleep(Duration::from_secs(2));

    let enc_before_reset = enc_count.load(Ordering::SeqCst);
    assert!(enc_before_reset > 0);

    // PTS goes backwards: send frame with PTS = 0
    send_frame_with_pts(&engine, &gen, "src", 3, 0);
    send_frame_with_pts(&engine, &gen, "src", 4, DUR);
    send_frame_with_pts(&engine, &gen, "src", 5, 2 * DUR);

    std::thread::sleep(Duration::from_secs(2));

    // No EOS should have been fired by the recreate policy
    let eos_after_reset = eos_count.load(Ordering::SeqCst);
    assert_eq!(
        eos_after_reset, 0,
        "Recreate policy should NOT fire EOS, got {eos_after_reset}"
    );

    // Frames after reset should still be encoded
    let enc_after_reset = enc_count.load(Ordering::SeqCst);
    assert!(
        enc_after_reset > enc_before_reset,
        "should encode frames after PTS reset (got {enc_after_reset} vs {enc_before_reset})"
    );

    // on_stream_reset should have fired
    let resets = reset_events.lock();
    assert_eq!(resets.len(), 1, "on_stream_reset should fire once");
    assert_eq!(resets[0].0, "src");
    drop(resets);

    engine.send_eos("src").unwrap();
    std::thread::sleep(Duration::from_secs(1));
    engine.shutdown();

    let final_eos = eos_count.load(Ordering::SeqCst);
    assert_eq!(
        final_eos, 1,
        "only the explicit EOS should be present, got {final_eos}"
    );
}

// -----------------------------------------------------------------------
// Test: equal PTS triggers reset
// -----------------------------------------------------------------------

#[test]
#[serial_test::serial]
fn pts_reset_triggered_by_equal_pts() {
    init();

    let enc_count = Arc::new(AtomicUsize::new(0));
    let eos_count = Arc::new(AtomicUsize::new(0));
    let reset_events: Arc<Mutex<Vec<(String, StreamResetReason)>>> =
        Arc::new(Mutex::new(Vec::new()));

    let callbacks = Callbacks {
        on_encoded_frame: Some(Arc::new(CountingEncodedCb {
            count: enc_count.clone(),
            eos_count: eos_count.clone(),
        })),
        on_stream_reset: Some(Arc::new(ResetCollector(reset_events.clone()))),
        ..Default::default()
    };

    let engine = PicassoEngine::new(
        GeneralSpec {
            idle_timeout_secs: 300,
            pts_reset_policy: PtsResetPolicy::EosOnDecreasingPts,
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
    engine.set_source_spec("src", spec).unwrap();

    let gen = BufferGenerator::new(
        VideoFormat::RGBA,
        W,
        H,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .unwrap();

    // Send two frames with the same PTS
    send_frame_with_pts(&engine, &gen, "src", 0, 1000);
    send_frame_with_pts(&engine, &gen, "src", 1, 1000); // equal PTS

    std::thread::sleep(Duration::from_secs(2));

    let resets = reset_events.lock();
    assert_eq!(
        resets.len(),
        1,
        "equal PTS should trigger a reset, got {} resets",
        resets.len()
    );
    match &resets[0].1 {
        StreamResetReason::PtsDecreased {
            last_pts_ns,
            new_pts_ns,
        } => {
            assert_eq!(*last_pts_ns, 1000);
            assert_eq!(*new_pts_ns, 1000);
        }
        other => panic!("expected PtsDecreased, got {other:?}"),
    }
    drop(resets);

    engine.send_eos("src").unwrap();
    std::thread::sleep(Duration::from_secs(1));
    engine.shutdown();
}

// -----------------------------------------------------------------------
// Test: multiple consecutive PTS resets
// -----------------------------------------------------------------------

#[test]
#[serial_test::serial]
fn pts_reset_multiple_consecutive_resets() {
    init();

    let enc_count = Arc::new(AtomicUsize::new(0));
    let eos_count = Arc::new(AtomicUsize::new(0));
    let reset_events: Arc<Mutex<Vec<(String, StreamResetReason)>>> =
        Arc::new(Mutex::new(Vec::new()));

    let callbacks = Callbacks {
        on_encoded_frame: Some(Arc::new(CountingEncodedCb {
            count: enc_count.clone(),
            eos_count: eos_count.clone(),
        })),
        on_stream_reset: Some(Arc::new(ResetCollector(reset_events.clone()))),
        ..Default::default()
    };

    let engine = PicassoEngine::new(
        GeneralSpec {
            idle_timeout_secs: 300,
            pts_reset_policy: PtsResetPolicy::EosOnDecreasingPts,
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
    engine.set_source_spec("src", spec).unwrap();

    let gen = BufferGenerator::new(
        VideoFormat::RGBA,
        W,
        H,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .unwrap();

    // Session 1
    send_frame_with_pts(&engine, &gen, "src", 0, 0);
    send_frame_with_pts(&engine, &gen, "src", 1, DUR);
    std::thread::sleep(Duration::from_millis(500));

    // Reset 1: PTS goes back to 0
    send_frame_with_pts(&engine, &gen, "src", 2, 0);
    send_frame_with_pts(&engine, &gen, "src", 3, DUR);
    std::thread::sleep(Duration::from_millis(500));

    // Reset 2: PTS goes back to 0 again
    send_frame_with_pts(&engine, &gen, "src", 4, 0);
    send_frame_with_pts(&engine, &gen, "src", 5, DUR);

    std::thread::sleep(Duration::from_secs(2));

    let resets = reset_events.lock();
    assert_eq!(
        resets.len(),
        2,
        "should have 2 PTS resets, got {}",
        resets.len()
    );
    drop(resets);

    let eos_mid = eos_count.load(Ordering::SeqCst);
    assert_eq!(eos_mid, 2, "should have 2 synthetic EOS, got {eos_mid}");

    engine.send_eos("src").unwrap();
    std::thread::sleep(Duration::from_secs(1));
    engine.shutdown();

    let final_eos = eos_count.load(Ordering::SeqCst);
    assert_eq!(
        final_eos, 3,
        "should have 3 EOS total (2 synthetic + 1 explicit), got {final_eos}"
    );
}

// -----------------------------------------------------------------------
// Test: encode_attribute gate discards unmarked frames
// -----------------------------------------------------------------------

#[test]
#[serial_test::serial]
fn encode_attribute_gate_discards_unmarked_frames() {
    init();

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
        conditional: ConditionalSpec {
            encode_attribute: Some(("gate".into(), "record".into())),
            render_attribute: None,
        },
        ..Default::default()
    };
    engine.set_source_spec("src", spec).unwrap();

    let gen = BufferGenerator::new(
        VideoFormat::RGBA,
        W,
        H,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .unwrap();

    // Send 5 frames: only frames 2 and 4 carry the gate attribute.
    for i in 0..5u64 {
        let mut frame = make_frame_sized("src", W as i64, H as i64);
        frame.set_pts((i * DUR) as i64).unwrap();
        frame.set_duration(Some(DUR as i64)).unwrap();
        if i == 2 || i == 4 {
            frame.set_persistent_attribute("gate", "record", &None, false, vec![]);
        }
        let buf = make_gpu_surface_view(&gen, i, DUR);
        engine.send_frame("src", frame, buf, None).unwrap();
    }

    std::thread::sleep(Duration::from_secs(2));
    engine.send_eos("src").unwrap();
    std::thread::sleep(Duration::from_secs(1));
    engine.shutdown();

    let encoded = enc_count.load(Ordering::SeqCst);
    assert_eq!(
        encoded, 2,
        "only 2 frames with the gate attribute should be encoded, got {encoded}"
    );

    let eos = eos_count.load(Ordering::SeqCst);
    assert_eq!(eos, 1, "should have exactly 1 EOS, got {eos}");
}
