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

// CUDA driver API functions for direct GPU memory access.
extern "C" {
    /// Fill GPU memory with a constant byte value.
    pub fn cuMemsetD8_v2(dst: u64, value: u8, count: usize) -> u32;

    /// Fill GPU memory with a constant 32-bit value.
    ///
    /// Sets `count` consecutive 32-bit values starting at `dst` to `value`.
    /// `dst` must be 4-byte aligned.
    pub fn cuMemsetD32_v2(dst: u64, value: u32, count: usize) -> u32;

    /// Pitched 2-D copy from host to device (or any combination).
    ///
    /// Copies a `width × height` byte region from `src` (pitch `spitch`) to
    /// `dst` (pitch `dpitch`) in a single DMA transfer.
    pub fn cudaMemcpy2D(
        dst: *mut std::ffi::c_void,
        dpitch: usize,
        src: *const std::ffi::c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: i32,
    ) -> i32;
}

// GStreamer parent buffer meta.
extern "C" {
    /// Attach a `GstParentBufferMeta` to `buffer`, preventing `ref_` from
    /// being freed while `buffer` is alive.
    pub fn gst_buffer_add_parent_buffer_meta(
        buffer: *mut gstreamer::ffi::GstBuffer,
        ref_: *mut gstreamer::ffi::GstBuffer,
    ) -> *mut std::ffi::c_void;
}

/// `cudaMemcpyKind` values from `<cuda_runtime_api.h>`.
pub const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
pub const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
pub const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

// ─── CUDA EGL interop (Jetson zero-copy VIC → CUDA) ─────────────────────────

/// Opaque handle returned by `cuGraphicsEGLRegisterImage`.
pub type CUgraphicsResource = *mut std::ffi::c_void;

pub const CU_EGL_FRAME_TYPE_PITCH: u32 = 1;
pub const CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE: u32 = 0x00;

const MAX_PLANES: usize = 3;

/// Per-plane data: either `CUarray` handles or pitched device pointers.
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUeglFrameData {
    pub p_array: [*mut std::ffi::c_void; MAX_PLANES],
    pub p_pitch: [*mut std::ffi::c_void; MAX_PLANES],
}

/// Mapped EGL frame returned by `cuGraphicsResourceGetMappedEglFrame`.
#[repr(C)]
pub struct CUeglFrame {
    pub frame: CUeglFrameData,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub pitch: u32,
    pub plane_count: u32,
    pub num_channels: u32,
    pub frame_type: u32,
    pub egl_color_format: u32,
    pub cu_format: u32,
}

extern "C" {
    pub fn cuGraphicsEGLRegisterImage(
        pCudaResource: *mut CUgraphicsResource,
        image: *mut std::ffi::c_void,
        flags: u32,
    ) -> u32;

    pub fn cuGraphicsMapResources(
        count: u32,
        resources: *mut CUgraphicsResource,
        hStream: *mut std::ffi::c_void,
    ) -> u32;

    pub fn cuGraphicsUnmapResources(
        count: u32,
        resources: *mut CUgraphicsResource,
        hStream: *mut std::ffi::c_void,
    ) -> u32;

    pub fn cuGraphicsResourceGetMappedEglFrame(
        eglFrame: *mut CUeglFrame,
        resource: CUgraphicsResource,
        index: u32,
        mipLevel: u32,
    ) -> u32;

    pub fn cuGraphicsUnregisterResource(resource: CUgraphicsResource) -> u32;
}
