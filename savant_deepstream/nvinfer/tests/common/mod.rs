//! Shared test utilities for nvinfer integration tests.

use deepstream_nvbufsurface::cuda_init;
use deepstream_nvbufsurface::{ComputeMode, TransformConfig};
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

/// Identity model nvinfer properties with absolute paths.
#[allow(dead_code)]
pub fn identity_properties() -> HashMap<String, String> {
    let dir = assets_dir();
    let mut m = HashMap::new();
    m.insert("gpu-id".into(), "0".into());
    m.insert("gie-unique-id".into(), "1".into());
    m.insert("net-scale-factor".into(), "1.0".into());
    m.insert(
        "onnx-file".into(),
        dir.join("identity.onnx").to_string_lossy().into(),
    );
    m.insert(
        "model-engine-file".into(),
        dir.join("identity.onnx_b16_gpu0_fp16.engine")
            .to_string_lossy()
            .into(),
    );
    m.insert("batch-size".into(), "16".into());
    m.insert("network-mode".into(), "2".into());
    m.insert("network-type".into(), "100".into());
    m.insert("infer-dims".into(), "3;12;12".into());
    m.insert("model-color-format".into(), "0".into());
    inject_jetson_scaling(&mut m);
    m
}

/// Age/gender model nvinfer properties with absolute paths.
#[allow(dead_code)]
pub fn age_gender_properties() -> HashMap<String, String> {
    let dir = assets_dir();
    let mut m = HashMap::new();
    m.insert("gpu-id".into(), "0".into());
    m.insert("gie-unique-id".into(), "2".into());
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
        dir.join("age_gender_mobilenet_v2_dynBatch.onnx_b16_gpu0_fp16.engine")
            .to_string_lossy()
            .into(),
    );
    m.insert("batch-size".into(), "16".into());
    m.insert("network-mode".into(), "2".into());
    m.insert("network-type".into(), "100".into());
    m.insert("infer-dims".into(), "3;112;112".into());
    m.insert("model-color-format".into(), "0".into());
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
