//! Shared test utilities for deepstream_nvbufsurface integration tests.

use deepstream_nvbufsurface::cuda_init;
use gstreamer as gst;
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
