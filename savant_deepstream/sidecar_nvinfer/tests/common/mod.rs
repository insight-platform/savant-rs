//! Shared test utilities for sidecar_nvinfer integration tests.

use deepstream_nvbufsurface::cuda_init;
use gstreamer as gst;
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

/// Path to identity nvinfer config.
#[allow(dead_code)]
pub fn identity_config_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/identity_nvinfer.txt")
}
