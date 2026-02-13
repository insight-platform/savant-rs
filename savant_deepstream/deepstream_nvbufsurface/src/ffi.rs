//! FFI bindings for NvBufSurface types, NvBufSurfTransform, and DeepStream
//! buffer pool functions.

#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(clippy::all)]

// Bindgen-generated NvBufSurface types.
// This gives us the NvBufSurface struct definition so we can compute
// its size via std::mem::size_of::<NvBufSurface>().
include!(concat!(env!("OUT_DIR"), "/nvbufsurface_bindings.rs"));

/// Bindgen-generated NvBufSurfTransform types and functions.
pub mod transform_ffi {
    #![allow(non_camel_case_types)]
    #![allow(non_upper_case_globals)]
    #![allow(non_snake_case)]
    #![allow(dead_code)]
    #![allow(clippy::all)]
    include!(concat!(env!("OUT_DIR"), "/nvbufsurftransform_bindings.rs"));
}

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

// Re-export CUDA functions used for padding (cudaMemset2DAsync, etc.)
// and CUDA stream management.
extern "C" {
    pub fn cudaMemset2DAsync(
        devPtr: *mut std::ffi::c_void,
        pitch: usize,
        value: i32,
        width: usize,
        height: usize,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    pub fn cudaStreamSynchronize(stream: *mut std::ffi::c_void) -> i32;

    /// Create a CUDA stream with the given flags.
    ///
    /// `flags`:
    /// - `0x00` = `cudaStreamDefault` (legacy default behavior)
    /// - `0x01` = `cudaStreamNonBlocking` (does NOT synchronize with stream 0)
    pub fn cudaStreamCreateWithFlags(pStream: *mut *mut std::ffi::c_void, flags: u32) -> i32;

    /// Destroy a CUDA stream.
    pub fn cudaStreamDestroy(stream: *mut std::ffi::c_void) -> i32;
}
