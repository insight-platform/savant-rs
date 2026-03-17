//! E2E test: Mixed Drop/Bypass/Encode sources in one engine.
//!
//! Simulates a real multi-camera deployment where some sources are recorded
//! (Encode), some are proxied (Bypass), and some are disabled (Drop).

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

/// Bypass callback that counts frames and EOS, and verifies source_id.
struct BypassCbWithSource {
    count: Arc<AtomicUsize>,
    eos_count: Arc<AtomicUsize>,
    expected_source: String,
}

impl OnBypassFrame for BypassCbWithSource {
    fn call(&self, output: OutputMessage) {
        match output {
            OutputMessage::VideoFrame(frame) => {
                assert_eq!(frame.get_source_id(), self.expected_source);
                self.count.fetch_add(1, Ordering::SeqCst);
            }
            OutputMessage::EndOfStream(_) => {
                self.eos_count.fetch_add(1, Ordering::SeqCst);
            }
        }
    }
}

/// Encoded callback that counts VideoFrame and EOS, verifies source_id for VideoFrame.
struct EncodedCbWithSource {
    count: Arc<AtomicUsize>,
    eos_count: Arc<AtomicUsize>,
    expected_source: String,
}

impl OnEncodedFrame for EncodedCbWithSource {
    fn call(&self, output: OutputMessage) {
        match output {
            OutputMessage::EndOfStream(_) => {
                self.eos_count.fetch_add(1, Ordering::SeqCst);
            }
            OutputMessage::VideoFrame(frame) => {
                assert_eq!(
                    frame.get_source_id(),
                    self.expected_source,
                    "VideoFrame must be from cam-encode"
                );
                self.count.fetch_add(1, Ordering::SeqCst);
            }
        }
    }
}

#[test]
#[serial_test::serial]
fn e2e_mixed_codecs() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).unwrap();

    let bypass_count = Arc::new(AtomicUsize::new(0));
    let bypass_eos_count = Arc::new(AtomicUsize::new(0));
    let enc_count = Arc::new(AtomicUsize::new(0));
    let enc_eos_count = Arc::new(AtomicUsize::new(0));

    let callbacks = Callbacks {
        on_bypass_frame: Some(Arc::new(BypassCbWithSource {
            count: bypass_count.clone(),
            eos_count: bypass_eos_count.clone(),
            expected_source: "cam-bypass".to_string(),
        })),
        on_encoded_frame: Some(Arc::new(EncodedCbWithSource {
            count: enc_count.clone(),
            eos_count: enc_eos_count.clone(),
            expected_source: "cam-encode".to_string(),
        })),
        ..Default::default()
    };

    let mut engine = PicassoEngine::new(
        GeneralSpec {
            idle_timeout_secs: 300,
            ..Default::default()
        },
        callbacks,
    );

    // Drop source
    engine
        .set_source_spec(
            "cam-drop",
            SourceSpec {
                codec: CodecSpec::Drop,
                ..Default::default()
            },
        )
        .unwrap();

    // Bypass source
    engine
        .set_source_spec(
            "cam-bypass",
            SourceSpec {
                codec: CodecSpec::Bypass,
                ..Default::default()
            },
        )
        .unwrap();

    // Encode source (JPEG)
    let enc_spec = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(common::jpeg_encoder_config(W, H)),
        },
        ..Default::default()
    };
    engine.set_source_spec("cam-encode", enc_spec).unwrap();

    let gen = DsNvUniformSurfaceBufferGenerator::builder(VideoFormat::RGBA, W, H, 1)
        .fps(30, 1)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .pool_size(32)
        .build()
        .unwrap();

    for i in 0..5u64 {
        // Drop: stub buffer
        let frame_drop = make_frame("cam-drop");
        engine
            .send_frame("cam-drop", frame_drop, make_surface_view(), None)
            .unwrap();

        // Bypass: stub buffer
        let frame_bypass = make_frame("cam-bypass");
        engine
            .send_frame("cam-bypass", frame_bypass, make_surface_view(), None)
            .unwrap();

        // Encode: GPU buffer
        let mut frame_enc = make_frame_sized("cam-encode", W as i64, H as i64);
        frame_enc.set_pts((i * DUR) as i64).unwrap();
        frame_enc.set_duration(Some(DUR as i64)).unwrap();
        let buf = make_gpu_surface_view_uniform(&gen, i, DUR);
        engine
            .send_frame("cam-encode", frame_enc, buf, None)
            .unwrap();
    }

    engine.send_eos("cam-drop").unwrap();
    engine.send_eos("cam-bypass").unwrap();
    engine.send_eos("cam-encode").unwrap();

    std::thread::sleep(Duration::from_secs(2));
    engine.shutdown();

    assert_eq!(bypass_count.load(Ordering::SeqCst), 5);
    assert!(
        enc_count.load(Ordering::SeqCst) >= 3,
        "expected >= 3 encoded frames, got {}",
        enc_count.load(Ordering::SeqCst)
    );
    // cam-bypass EOS goes through on_bypass_frame
    assert_eq!(bypass_eos_count.load(Ordering::SeqCst), 1);
    // cam-drop + cam-encode EOS go through on_encoded_frame
    assert_eq!(enc_eos_count.load(Ordering::SeqCst), 2);
}
