//! E2E test: CUDA pointer access via OnGpuMat callback.
//!
//! Simulates a post-transform GPU processing step that accesses the raw
//! CUDA buffer after GPU transform.

mod common;

use common::*;
use deepstream_encoders::prelude::*;
use deepstream_nvbufsurface::TransformConfig;
use picasso::prelude::*;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const W: u32 = 640;
const H: u32 = 480;
const DUR: u64 = 33_333_333;

#[derive(Debug, Clone)]
struct GpuMatCall {
    source_id: String,
    data_ptr: usize,
    pitch: u32,
    width: u32,
    height: u32,
}

struct GpuMatRecorder {
    calls: Arc<Mutex<Vec<GpuMatCall>>>,
}

impl OnGpuMat for GpuMatRecorder {
    fn call(
        &self,
        source_id: &str,
        _frame: &savant_core::primitives::frame::VideoFrameProxy,
        data_ptr: usize,
        pitch: u32,
        width: u32,
        height: u32,
    ) {
        self.calls.lock().unwrap().push(GpuMatCall {
            source_id: source_id.to_string(),
            data_ptr,
            pitch,
            width,
            height,
        });
    }
}

#[test]
#[serial_test::serial]
fn e2e_on_gpumat_fires_when_enabled() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).unwrap();

    let calls = Arc::new(Mutex::new(Vec::new()));
    let enc_count = Arc::new(AtomicUsize::new(0));

    let callbacks = Callbacks {
        on_gpumat: Some(Arc::new(GpuMatRecorder {
            calls: calls.clone(),
        })),
        on_encoded_frame: Some(Arc::new(CountingEncodedCb {
            count: enc_count.clone(),
            eos_count: Arc::new(AtomicUsize::new(0)),
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
        use_on_gpumat: true,
        ..Default::default()
    };
    engine.set_source_spec("gpumat", spec).unwrap();

    let gen = DsNvSurfaceBufferGenerator::new(
        VideoFormat::RGBA,
        W,
        H,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .unwrap();

    for i in 0..5u64 {
        let mut frame = make_frame_sized("gpumat", W as i64, H as i64);
        frame.set_pts((i * DUR) as i64).unwrap();
        frame.set_duration(Some(DUR as i64)).unwrap();
        let buf = make_gpu_surface_view(&gen, i, DUR);
        engine.send_frame("gpumat", frame, buf, None).unwrap();
    }
    engine.send_eos("gpumat").unwrap();

    std::thread::sleep(Duration::from_secs(2));
    engine.shutdown();

    let recorded = calls.lock().unwrap();
    assert!(
        recorded.len() >= 3,
        "expected >= 3 OnGpuMat calls, got {}",
        recorded.len()
    );
    for call in recorded.iter() {
        assert_eq!(call.source_id, "gpumat");
        assert!(call.data_ptr != 0, "data_ptr should be non-null");
        assert!(call.pitch > 0, "pitch should be positive");
        assert_eq!(call.width, W);
        assert_eq!(call.height, H);
    }
}

#[test]
#[serial_test::serial]
fn e2e_on_gpumat_does_not_fire_when_disabled() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).unwrap();

    let calls = Arc::new(Mutex::new(Vec::new()));
    let enc_count = Arc::new(AtomicUsize::new(0));

    let callbacks = Callbacks {
        on_gpumat: Some(Arc::new(GpuMatRecorder {
            calls: calls.clone(),
        })),
        on_encoded_frame: Some(Arc::new(CountingEncodedCb {
            count: enc_count.clone(),
            eos_count: Arc::new(AtomicUsize::new(0)),
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
        use_on_gpumat: false,
        ..Default::default()
    };
    engine.set_source_spec("gpumat-off", spec).unwrap();

    let gen = DsNvSurfaceBufferGenerator::new(
        VideoFormat::RGBA,
        W,
        H,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .unwrap();

    for i in 0..5u64 {
        let mut frame = make_frame_sized("gpumat-off", W as i64, H as i64);
        frame.set_pts((i * DUR) as i64).unwrap();
        frame.set_duration(Some(DUR as i64)).unwrap();
        let buf = make_gpu_surface_view(&gen, i, DUR);
        engine.send_frame("gpumat-off", frame, buf, None).unwrap();
    }
    engine.send_eos("gpumat-off").unwrap();

    std::thread::sleep(Duration::from_secs(2));
    engine.shutdown();

    let recorded = calls.lock().unwrap();
    assert_eq!(
        recorded.len(),
        0,
        "OnGpuMat should not fire when use_on_gpumat=false"
    );
}
