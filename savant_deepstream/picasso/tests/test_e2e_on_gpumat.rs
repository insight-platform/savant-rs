//! E2E test: CUDA pointer access via OnGpuMat callback.
//!
//! Simulates a post-transform GPU processing step that accesses the raw
//! CUDA buffer after GPU transform.  The callback performs a real
//! `cudaMemcpy2D` read from the pointer to verify it is CUDA-addressable
//! (not a VIC-managed handle on Jetson).

mod common;

use common::*;
use deepstream_buffers::ffi;
use deepstream_buffers::{SurfaceView, TransformConfig};
use deepstream_encoders::prelude::*;
use picasso::prelude::*;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const W: u32 = 640;
const H: u32 = 480;
const DUR: u64 = 33_333_333;
const BPP: u32 = 4; // RGBA

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct GpuMatCall {
    source_id: String,
    /// Recorded for equality checks only — must NOT be dereferenced after the
    /// callback scope because the buffer may be recycled / the EGL-CUDA
    /// registration may be freed.
    data_ptr: usize,
    pitch: u32,
    width: u32,
    height: u32,
    cuda_stream: usize,
    /// CUDA return code from a `cudaMemcpy2D` read of the first row.
    /// 0 = success; anything else means the pointer was not CUDA-accessible.
    cuda_read_rc: i32,
    /// True when at least one byte in the copied row is non-zero.
    has_nonzero_pixel: bool,
}

/// Callback that records metadata **and** performs a real CUDA memory read
/// to prove the pointer is CUDA-addressable.
struct GpuMatRecorder(Arc<Mutex<Vec<GpuMatCall>>>);

impl OnGpuMat for GpuMatRecorder {
    fn call(
        &self,
        source_id: &str,
        _frame: &savant_core::primitives::frame::VideoFrameProxy,
        view: &SurfaceView,
    ) {
        let data_ptr = view.data_ptr() as usize;
        let pitch = view.pitch();
        let width = view.width();
        let height = view.height();
        let cuda_stream = view.cuda_stream().as_raw() as usize;
        let width_bytes = (width * BPP) as usize;
        let mut host_row = vec![0u8; width_bytes];

        let cuda_read_rc = unsafe {
            ffi::cudaStreamSynchronize(cuda_stream as *mut std::ffi::c_void);
            ffi::cudaMemcpy2D(
                host_row.as_mut_ptr() as *mut std::ffi::c_void,
                width_bytes,
                data_ptr as *const std::ffi::c_void,
                pitch as usize,
                width_bytes,
                1, // single row
                ffi::CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };

        let has_nonzero_pixel = host_row.iter().any(|&b| b != 0);

        self.0.lock().unwrap().push(GpuMatCall {
            source_id: source_id.to_string(),
            data_ptr,
            pitch,
            width,
            height,
            cuda_stream,
            cuda_read_rc,
            has_nonzero_pixel,
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
        on_gpumat: Some(Arc::new(GpuMatRecorder(calls.clone()))),
        on_encoded_frame: Some(Arc::new(CountingEncodedCb {
            count: enc_count.clone(),
            eos_count: Arc::new(AtomicUsize::new(0)),
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

    let spec = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(jpeg_encoder_config(W, H)),
        },
        use_on_gpumat: true,
        ..Default::default()
    };
    engine.set_source_spec("gpumat", spec).unwrap();

    let gen = BufferGenerator::builder(VideoFormat::RGBA, W, H)
        .fps(30, 1)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(32)
        .max_buffers(32)
        .build()
        .unwrap();

    for i in 0..5u64 {
        let mut frame = make_frame_sized("gpumat", W as i64, H as i64);
        frame.set_pts((i * DUR) as i64).unwrap();
        frame.set_duration(Some(DUR as i64)).unwrap();
        let buf = make_gpu_surface_view_uniform(&gen, i, DUR);
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
    for (i, call) in recorded.iter().enumerate() {
        assert_eq!(call.source_id, "gpumat");
        assert!(call.data_ptr != 0, "data_ptr should be non-null");
        assert!(call.pitch > 0, "pitch should be positive");
        assert_eq!(call.width, W);
        assert_eq!(call.height, H);
        assert!(call.cuda_stream != 0, "cuda_stream should be non-null");
        assert_eq!(
            call.cuda_read_rc, 0,
            "frame {i}: cudaMemcpy2D from data_ptr failed (rc={}); \
             pointer is not CUDA-accessible",
            call.cuda_read_rc
        );
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
        on_gpumat: Some(Arc::new(GpuMatRecorder(calls.clone()))),
        on_encoded_frame: Some(Arc::new(CountingEncodedCb {
            count: enc_count.clone(),
            eos_count: Arc::new(AtomicUsize::new(0)),
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

    let spec = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(jpeg_encoder_config(W, H)),
        },
        use_on_gpumat: false,
        ..Default::default()
    };
    engine.set_source_spec("gpumat-off", spec).unwrap();

    let gen = BufferGenerator::builder(VideoFormat::RGBA, W, H)
        .fps(30, 1)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(32)
        .max_buffers(32)
        .build()
        .unwrap();

    for i in 0..5u64 {
        let mut frame = make_frame_sized("gpumat-off", W as i64, H as i64);
        frame.set_pts((i * DUR) as i64).unwrap();
        frame.set_duration(Some(DUR as i64)).unwrap();
        let buf = make_gpu_surface_view_uniform(&gen, i, DUR);
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

#[test]
#[serial_test::serial]
fn e2e_external_cuda_stream_rejected() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).unwrap();

    let callbacks = Callbacks::default();
    let mut engine = PicassoEngine::new(
        GeneralSpec {
            idle_timeout_secs: 300,
            ..Default::default()
        },
        callbacks,
    );

    // Create a TransformConfig with a non-default cuda_stream.
    let transform = TransformConfig {
        cuda_stream: unsafe {
            deepstream_buffers::CudaStream::from_raw(0xDEAD_BEEF as *mut std::ffi::c_void)
        },
        ..Default::default()
    };

    let spec = SourceSpec {
        codec: CodecSpec::Encode {
            transform,
            encoder: Box::new(jpeg_encoder_config(W, H)),
        },
        ..Default::default()
    };

    let result = engine.set_source_spec("bad-stream", spec);
    assert!(result.is_err(), "Expected ExternalCudaStream error");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("cuda_stream must be null"),
        "Error should mention cuda_stream, got: {err_msg}"
    );

    engine.shutdown();
}

#[test]
#[serial_test::serial]
fn e2e_callback_order_gpumat_skia() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).unwrap();

    let calls = Arc::new(Mutex::new(Vec::new()));
    let enc_count = Arc::new(AtomicUsize::new(0));

    let callbacks = Callbacks {
        on_gpumat: Some(Arc::new(GpuMatRecorder(calls.clone()))),
        on_encoded_frame: Some(Arc::new(CountingEncodedCb {
            count: enc_count.clone(),
            eos_count: Arc::new(AtomicUsize::new(0)),
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

    let spec = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(jpeg_encoder_config(W, H)),
        },
        use_on_gpumat: true,
        callback_order: CallbackInvocationOrder::GpuMatSkia,
        ..Default::default()
    };
    engine.set_source_spec("order-test", spec).unwrap();

    let gen = BufferGenerator::builder(VideoFormat::RGBA, W, H)
        .fps(30, 1)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(32)
        .max_buffers(32)
        .build()
        .unwrap();

    for i in 0..3u64 {
        let mut frame = make_frame_sized("order-test", W as i64, H as i64);
        frame.set_pts((i * DUR) as i64).unwrap();
        frame.set_duration(Some(DUR as i64)).unwrap();
        let buf = make_gpu_surface_view_uniform(&gen, i, DUR);
        engine.send_frame("order-test", frame, buf, None).unwrap();
    }
    engine.send_eos("order-test").unwrap();

    std::thread::sleep(Duration::from_secs(2));
    engine.shutdown();

    let recorded = calls.lock().unwrap();
    assert!(
        !recorded.is_empty(),
        "OnGpuMat should fire with GpuMatSkia order"
    );
    for (i, call) in recorded.iter().enumerate() {
        assert_eq!(call.source_id, "order-test");
        assert!(call.cuda_stream != 0, "cuda_stream should be non-null");
        assert_eq!(
            call.cuda_read_rc, 0,
            "frame {i}: cudaMemcpy2D from data_ptr failed (rc={})",
            call.cuda_read_rc
        );
    }
}

#[test]
#[serial_test::serial]
fn e2e_callback_order_gpumat_skia_gpumat() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).unwrap();

    let calls = Arc::new(Mutex::new(Vec::new()));
    let enc_count = Arc::new(AtomicUsize::new(0));

    let callbacks = Callbacks {
        on_gpumat: Some(Arc::new(GpuMatRecorder(calls.clone()))),
        on_encoded_frame: Some(Arc::new(CountingEncodedCb {
            count: enc_count.clone(),
            eos_count: Arc::new(AtomicUsize::new(0)),
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

    let spec = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(jpeg_encoder_config(W, H)),
        },
        use_on_gpumat: true,
        callback_order: CallbackInvocationOrder::GpuMatSkiaGpuMat,
        ..Default::default()
    };
    engine.set_source_spec("double-test", spec).unwrap();

    let gen = BufferGenerator::builder(VideoFormat::RGBA, W, H)
        .fps(30, 1)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(32)
        .max_buffers(32)
        .build()
        .unwrap();

    let num_frames = 3u64;
    for i in 0..num_frames {
        let mut frame = make_frame_sized("double-test", W as i64, H as i64);
        frame.set_pts((i * DUR) as i64).unwrap();
        frame.set_duration(Some(DUR as i64)).unwrap();
        let buf = make_gpu_surface_view_uniform(&gen, i, DUR);
        engine.send_frame("double-test", frame, buf, None).unwrap();
    }
    engine.send_eos("double-test").unwrap();

    std::thread::sleep(Duration::from_secs(2));
    engine.shutdown();

    let recorded = calls.lock().unwrap();
    // GpuMatSkiaGpuMat fires twice per frame (before and after Skia).
    // Even without Skia render opts, the on_gpumat should fire twice
    // per frame if both pre and post conditions are met.
    assert!(
        recorded.len() >= num_frames as usize * 2,
        "Expected >= {} OnGpuMat calls (2x per frame), got {}",
        num_frames * 2,
        recorded.len()
    );
    for (i, call) in recorded.iter().enumerate() {
        assert_eq!(
            call.cuda_read_rc, 0,
            "call {i}: cudaMemcpy2D from data_ptr failed (rc={})",
            call.cuda_read_rc
        );
    }
}
