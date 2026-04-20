//! Shared test utilities for nvinfer integration tests.

pub mod age_gender_test_utils;
pub mod yolo_test_utils;

use deepstream_buffers::cuda_init;
use deepstream_buffers::{ComputeMode, TransformConfig};
use gstreamer as gst;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Once;

static INIT: Once = Once::new();

/// One-time GStreamer + CUDA initialization for all integration tests.
pub fn init() {
    INIT.call_once(|| {
        let _ = env_logger::try_init();
        gst::init().unwrap();
        cuda_init(0).expect("Failed to initialize CUDA - is a GPU available?");
    });
}

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets")
}

/// Platform-specific engine cache directory: `assets/engines/<platform>/`.
///
/// Creates the directory tree on first call.
pub fn engines_dir() -> PathBuf {
    let tag = nvidia_gpu_utils::gpu_platform_tag(0).unwrap_or_else(|_| "unknown".to_string());
    let dir = assets_dir().join("engines").join(tag);
    std::fs::create_dir_all(&dir).expect("create engine cache directory");
    dir
}

/// After NvInfer creation, move the auto-generated engine file (written
/// by DeepStream next to the ONNX model) into the platform-specific cache
/// directory so that subsequent runs find it via `model-engine-file`.
///
/// Delegates to [`deepstream_nvinfer::engine_cache::promote_built_engine`].
pub fn promote_built_engine(onnx_stem: &str, batch_size: u32) {
    let engine_name = format!("{}_b{}_gpu0_fp16.engine", onnx_stem, batch_size);
    let auto_path = assets_dir().join(&engine_name);
    let cached_path = engines_dir().join(&engine_name);
    match deepstream_nvinfer::engine_cache::promote_built_engine(&auto_path, &cached_path) {
        Ok(true) => eprintln!("  [cache] promoted {engine_name} to platform cache"),
        Ok(false) => {}
        Err(e) => eprintln!("  [cache] promotion failed for {engine_name}: {e}"),
    }
}

/// Identity model nvinfer properties with absolute paths.
#[allow(dead_code)]
pub fn identity_properties() -> HashMap<String, String> {
    let dir = assets_dir();
    let mut m = HashMap::new();
    m.insert("gpu-id".into(), "0".into());
    m.insert("net-scale-factor".into(), "1.0".into());
    m.insert(
        "onnx-file".into(),
        dir.join("identity.onnx").to_string_lossy().into(),
    );
    m.insert(
        "model-engine-file".into(),
        engines_dir()
            .join("identity.onnx_b16_gpu0_fp16.engine")
            .to_string_lossy()
            .into(),
    );
    m.insert("batch-size".into(), "16".into());
    m.insert("network-mode".into(), "2".into());
    inject_jetson_scaling(&mut m);
    m
}

/// Age/gender model nvinfer properties with absolute paths.
#[allow(dead_code)]
pub fn age_gender_properties() -> HashMap<String, String> {
    let dir = assets_dir();
    let mut m = HashMap::new();
    m.insert("gpu-id".into(), "0".into());
    m.insert("net-scale-factor".into(), "0.007843137254902".into());
    m.insert("offsets".into(), "127.5;127.5;127.5".into());
    m.insert(
        "onnx-file".into(),
        dir.join("age_gender_mobilenet_v2_dynBatch.onnx")
            .to_string_lossy()
            .into(),
    );
    m.insert(
        "model-engine-file".into(),
        engines_dir()
            .join("age_gender_mobilenet_v2_dynBatch.onnx_b16_gpu0_fp16.engine")
            .to_string_lossy()
            .into(),
    );
    m.insert("batch-size".into(), "16".into());
    m.insert("network-mode".into(), "2".into());
    inject_jetson_scaling(&mut m);
    m
}

/// Age/gender model with `batch-size=1` (engine `..._b1_gpu0_fp16.engine`).
///
/// Used by oversized-batch tests where the submitted surface has more frames /
/// ROIs than the TensorRT engine max batch; `gstnvinfer` must split internally.
#[allow(dead_code)]
pub fn age_gender_properties_bs1() -> HashMap<String, String> {
    let dir = assets_dir();
    let mut m = HashMap::new();
    m.insert("gpu-id".into(), "0".into());
    m.insert("net-scale-factor".into(), "0.007843137254902".into());
    m.insert("offsets".into(), "127.5;127.5;127.5".into());
    m.insert(
        "onnx-file".into(),
        dir.join("age_gender_mobilenet_v2_dynBatch.onnx")
            .to_string_lossy()
            .into(),
    );
    m.insert(
        "model-engine-file".into(),
        engines_dir()
            .join("age_gender_mobilenet_v2_dynBatch.onnx_b1_gpu0_fp16.engine")
            .to_string_lossy()
            .into(),
    );
    m.insert("batch-size".into(), "1".into());
    m.insert("network-mode".into(), "2".into());
    inject_jetson_scaling(&mut m);
    m
}

/// Identity 3x112x112 model nvinfer properties with absolute paths.
///
/// Uses batch-size 32 to match the age_gender E2E tests. The model
/// simply returns its input, so comparing the output tensor against
/// known pixel data verifies the crop pipeline.
#[allow(dead_code)]
pub fn identity_112x112_properties() -> HashMap<String, String> {
    let dir = assets_dir();
    let mut m = HashMap::new();
    m.insert("gpu-id".into(), "0".into());
    m.insert("net-scale-factor".into(), "1.0".into());
    m.insert(
        "onnx-file".into(),
        dir.join("identity_3x112x112.onnx").to_string_lossy().into(),
    );
    m.insert(
        "model-engine-file".into(),
        engines_dir()
            .join("identity_3x112x112.onnx_b32_gpu0_fp16.engine")
            .to_string_lossy()
            .into(),
    );
    m.insert("batch-size".into(), "32".into());
    m.insert("network-mode".into(), "2".into());
    inject_jetson_scaling(&mut m);
    m
}

/// FullHD identity model (3x1080x1920) nvinfer properties.
///
/// Batch-size 2 (only one frame needed but nvinfer needs >= 1).
/// Processes the entire 1920x1080 frame so that the output tensor
/// is a pixel-level copy of the input — ideal for diagnosing surface
/// memory layout issues.
#[allow(dead_code)]
pub fn identity_fullhd_properties() -> HashMap<String, String> {
    let dir = assets_dir();
    let mut m = HashMap::new();
    m.insert("gpu-id".into(), "0".into());
    m.insert("net-scale-factor".into(), "1.0".into());
    m.insert(
        "onnx-file".into(),
        dir.join("identity_fullhd.onnx").to_string_lossy().into(),
    );
    m.insert(
        "model-engine-file".into(),
        engines_dir()
            .join("identity_fullhd.onnx_b2_gpu0_fp16.engine")
            .to_string_lossy()
            .into(),
    );
    m.insert("batch-size".into(), "2".into());
    m.insert("network-mode".into(), "2".into());
    inject_jetson_scaling(&mut m);
    m
}

/// On Jetson, force GPU compute for nvinfer's internal NvBufSurfTransform.
/// VIC requires surfaces >= 16x16; GPU compute has no such restriction.
fn inject_jetson_scaling(props: &mut HashMap<String, String>) {
    if cfg!(target_arch = "aarch64") {
        props.insert("scaling-compute-hw".into(), "1".into());
    }
}

/// Returns a [`TransformConfig`] suitable for the current platform.
///
/// On Jetson (aarch64), uses GPU compute to avoid VIC limitations with small
/// surfaces (VIC requires >= 16x16).  On dGPU the default backend is already
/// GPU-based.
#[allow(dead_code)]
pub fn platform_transform_config() -> TransformConfig {
    let mut config = TransformConfig::default();
    if cfg!(target_arch = "aarch64") {
        config.compute_mode = ComputeMode::Gpu;
    }
    config
}

/// Send one dummy frame through the engine to prime the TensorRT execution
/// context. The first inference after engine creation may behave differently
/// (CUDA context warm-up, memory pool allocation, cuDNN autotuning, etc.),
/// so this ensures subsequent runs produce stable, comparable results.
#[allow(dead_code)]
pub fn warmup_engine(engine: &deepstream_nvinfer::NvInfer, width: u32, height: u32) {
    use deepstream_buffers::{
        BufferGenerator, NvBufSurfaceMemType, SavantIdMetaKind, UniformBatchGenerator, VideoFormat,
    };

    let src_gen = BufferGenerator::builder(VideoFormat::RGBA, width, height)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(1)
        .max_buffers(1)
        .build()
        .expect("warmup src generator");

    let batched_gen = UniformBatchGenerator::new(
        VideoFormat::RGBA,
        width,
        height,
        1,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .expect("warmup batched generator");

    let shared = {
        let config = platform_transform_config();
        let ids = vec![SavantIdMetaKind::Frame(0)];
        let mut batch = batched_gen.acquire_batch(config, ids).unwrap();
        let src_shared = src_gen.acquire(Some(0)).unwrap();
        src_shared
            .with_view(0, |src_view| batch.transform_slot(0, src_view, None))
            .unwrap();
        batch.finalize().unwrap();
        batch.into_shared_buffer()
    };
    engine.submit(shared, None).expect("warmup submit");
    let _ = recv_inference(engine);
    eprintln!("  [warmup] engine primed with {width}x{height} dummy frame");
}

/// Receive the next [`NvInferOutput::Inference`] from the engine, skipping
/// any intermediate [`NvInferOutput::Event`] (e.g. `stream-start`, `caps`).
///
/// Panics if EOS is received or if too many iterations pass without a result.
#[allow(dead_code)]
pub fn recv_inference(
    engine: &deepstream_nvinfer::NvInfer,
) -> deepstream_nvinfer::output::BatchInferenceOutput {
    for _ in 0..64 {
        match engine.recv().expect("recv failed") {
            deepstream_nvinfer::NvInferOutput::Inference(output) => return output,
            deepstream_nvinfer::NvInferOutput::Event(_) => continue,
            deepstream_nvinfer::NvInferOutput::Eos { source_id } => {
                panic!("unexpected EOS while waiting for inference: source_id={source_id}")
            }
            deepstream_nvinfer::NvInferOutput::Error(e) => {
                panic!("pipeline error while waiting for inference: {e}")
            }
        }
    }
    panic!("did not receive Inference output after 64 attempts");
}
