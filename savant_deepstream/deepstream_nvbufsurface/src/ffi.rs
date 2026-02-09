//! FFI bindings for NvBufSurface types and DeepStream buffer pool functions.

#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(clippy::all)]

// Bindgen-generated NvBufSurface types.
// This gives us the NvBufSurface struct definition so we can compute
// its size via std::mem::size_of::<NvBufSurface>().
include!(concat!(env!("OUT_DIR"), "/nvbufsurface_bindings.rs"));

// Manually declared DeepStream buffer pool function.
// We declare this manually (instead of via bindgen) to use gstreamer-rs
// compatible types, avoiding type conflicts from pulling in GStreamer
// headers through bindgen.
extern "C" {
    /// Creates a new DeepStream NvDS buffer pool.
    ///
    /// Returns a `GstBufferPool*` with a floating reference, which is a
    /// subclass of `GstBufferPool` specialized for NvBufSurface allocation.
    pub fn gst_nvds_buffer_pool_new() -> *mut gstreamer::ffi::GstBufferPool;
}
