//! Shared test setup for nvtracker integration tests.

use deepstream_buffers::cuda_init;
use gstreamer as gst;
use std::sync::Once;

static INIT: Once = Once::new();

pub fn init() {
    INIT.call_once(|| {
        let _ = env_logger::try_init();
        gst::init().expect("gst init");
        cuda_init(0).expect("cuda_init — GPU required for nvtracker tests");
    });
}
