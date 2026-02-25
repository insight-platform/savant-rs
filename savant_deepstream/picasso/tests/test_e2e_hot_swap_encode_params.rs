//! E2E test: Hot-swap between Encode specs (resolution change).
//!
//! Simulates adaptive bitrate: source starts at 1280x720, then resolution
//! is changed to 640x480 mid-stream via set_source_spec.

mod common;

use common::*;
use deepstream_encoders::prelude::*;
use deepstream_nvbufsurface::TransformConfig;
use picasso::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

const W1: u32 = 1280;
const H1: u32 = 720;
const W2: u32 = 640;
const H2: u32 = 480;
const DUR: u64 = 33_333_333;

#[test]
#[serial_test::serial]
fn e2e_hot_swap_encode_params() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).unwrap();

    let enc_count = Arc::new(AtomicUsize::new(0));
    let eos_count = Arc::new(AtomicUsize::new(0));
    let resolutions: Arc<parking_lot::Mutex<Vec<(u32, u32)>>> =
        Arc::new(parking_lot::Mutex::new(Vec::new()));

    struct CaptureResolutions {
        resolutions: Arc<parking_lot::Mutex<Vec<(u32, u32)>>>,
        count: Arc<AtomicUsize>,
        eos_count: Arc<AtomicUsize>,
    }
    impl OnEncodedFrame for CaptureResolutions {
        fn call(&self, output: picasso::message::EncodedOutput) {
            match output {
                picasso::message::EncodedOutput::EndOfStream(_) => {
                    self.eos_count.fetch_add(1, Ordering::SeqCst);
                }
                picasso::message::EncodedOutput::VideoFrame(frame) => {
                    self.count.fetch_add(1, Ordering::SeqCst);
                    let w = frame.get_width() as u32;
                    let h = frame.get_height() as u32;
                    self.resolutions.lock().push((w, h));
                }
            }
        }
    }

    let callbacks = Callbacks {
        on_encoded_frame: Some(Arc::new(CaptureResolutions {
            resolutions: resolutions.clone(),
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

    let spec1 = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(jpeg_encoder_config(W1, H1)),
        },
        ..Default::default()
    };
    engine.set_source_spec("swap", spec1).unwrap();

    let gen1 = NvBufSurfaceGenerator::new(
        VideoFormat::RGBA,
        W1,
        H1,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .unwrap();

    for i in 0..10u64 {
        let mut frame = make_frame_sized("swap", W1 as i64, H1 as i64);
        frame.set_pts((i * DUR) as i64).unwrap();
        frame.set_duration(Some(DUR as i64)).unwrap();
        let buf = make_gpu_buffer(&gen1, i, DUR);
        engine.send_frame("swap", frame, buf).unwrap();
    }
    std::thread::sleep(Duration::from_secs(2));

    let spec2 = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(jpeg_encoder_config(W2, H2)),
        },
        ..Default::default()
    };
    engine.set_source_spec("swap", spec2).unwrap();

    let gen2 = NvBufSurfaceGenerator::new(
        VideoFormat::RGBA,
        W2,
        H2,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .unwrap();

    for i in 0..10u64 {
        let mut frame = make_frame_sized("swap", W2 as i64, H2 as i64);
        frame.set_pts((i * DUR) as i64).unwrap();
        frame.set_duration(Some(DUR as i64)).unwrap();
        let buf = make_gpu_buffer(&gen2, i, DUR);
        engine.send_frame("swap", frame, buf).unwrap();
    }
    engine.send_eos("swap").unwrap();
    std::thread::sleep(Duration::from_secs(2));
    engine.shutdown();

    let res = resolutions.lock();
    let has_720 = res.iter().any(|(w, h)| *w == W1 && *h == H1);
    let has_480 = res.iter().any(|(w, h)| *w == W2 && *h == H2);
    assert!(has_720, "expected some 1280x720 encoded frames");
    assert!(has_480, "expected some 640x480 encoded frames");
    assert_eq!(eos_count.load(Ordering::SeqCst), 1);
}
