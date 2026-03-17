//! Shared test utilities for deepstream_nvbufsurface integration tests.

use deepstream_nvbufsurface::cuda_init;
use gstreamer as gst;
use std::cell::Cell;
use std::sync::Once;

static INIT: Once = Once::new();

thread_local! {
    static CUDA_CTX_BOUND: Cell<bool> = const { Cell::new(false) };
}

extern "C" {
    fn cudaFree(dev_ptr: *mut std::ffi::c_void) -> i32;
}

/// One-time GStreamer + CUDA initialization for all integration tests.
///
/// GStreamer and `cuInit` run once per process. The CUDA primary-context
/// bind (`cudaFree(null)`) runs once per thread, because the test harness
/// may schedule tests on different threads and the primary context is
/// thread-local.
pub fn init() {
    INIT.call_once(|| {
        let _ = env_logger::try_init();
        gst::init().unwrap();
        cuda_init(0).expect("Failed to initialize CUDA - is a GPU available?");
    });

    CUDA_CTX_BOUND.with(|bound| {
        if !bound.get() {
            unsafe {
                cudaFree(std::ptr::null_mut());
            }
            bound.set(true);
        }
    });
}
