//! E2E test: Encoder re-creation after EOS.
//!
//! Simulates camera reconnection: source sends frames, EOS (disconnect),
//! then new frames (reconnect). Encoder is destroyed on EOS and re-created
//! on the next frame.

mod common;

use common::*;
use deepstream_encoders::prelude::*;
use deepstream_nvbufsurface::TransformConfig;
use picasso::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

const W: u32 = 640;
const H: u32 = 480;
const DUR: u64 = 33_333_333;

#[test]
#[serial_test::serial]
fn e2e_encode_eos_reencode() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).unwrap();

    let enc_count = Arc::new(AtomicUsize::new(0));
    let eos_count = Arc::new(AtomicUsize::new(0));

    let callbacks = Callbacks {
        on_encoded_frame: Some(Arc::new(CountingEncodedCb {
            count: enc_count.clone(),
            eos_count: eos_count.clone(),
        })),
        ..Default::default()
    };

    let mut engine = PicassoEngine::new(
        GeneralSpec {
            idle_timeout_secs: 300,
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
    engine.set_source_spec("reconnect", spec).unwrap();

    let gen = NvBufSurfaceGenerator::new(
        VideoFormat::RGBA,
        W,
        H,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .unwrap();

    // Session 1: 10 frames, EOS
    for i in 0..10u64 {
        let mut frame = make_frame_sized("reconnect", W as i64, H as i64);
        frame.set_pts((i * DUR) as i64).unwrap();
        frame.set_duration(Some(DUR as i64)).unwrap();
        let buf = make_gpu_surface_view(&gen, i, DUR);
        engine.send_frame("reconnect", frame, buf, None).unwrap();
    }
    engine.send_eos("reconnect").unwrap();
    std::thread::sleep(Duration::from_secs(2));

    let enc1 = enc_count.load(Ordering::SeqCst);
    let eos1 = eos_count.load(Ordering::SeqCst);
    assert!(enc1 > 0, "session 1: expected encoded frames");
    assert_eq!(eos1, 1, "session 1: expected 1 EOS");

    // Session 2: 10 more frames, EOS (encoder re-created)
    for i in 10..20u64 {
        let mut frame = make_frame_sized("reconnect", W as i64, H as i64);
        frame.set_pts((i * DUR) as i64).unwrap();
        frame.set_duration(Some(DUR as i64)).unwrap();
        let buf = make_gpu_surface_view(&gen, i, DUR);
        engine.send_frame("reconnect", frame, buf, None).unwrap();
    }
    engine.send_eos("reconnect").unwrap();
    std::thread::sleep(Duration::from_secs(2));
    engine.shutdown();

    let enc2 = enc_count.load(Ordering::SeqCst);
    let eos2 = eos_count.load(Ordering::SeqCst);
    assert!(enc2 > enc1, "session 2: expected more encoded frames");
    assert_eq!(eos2, 2, "expected 2 EOS total");
}
