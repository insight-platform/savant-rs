//! Safe Rust API for NVIDIA DeepStream NvBufSurface buffer generation.
//!
//! This crate provides a Rust implementation of the NvBufSurfaceGenerator,
//! which creates GStreamer buffers with NvBufSurface memory allocated via
//! DeepStream's buffer pool mechanism.
//!
//! # Overview
//!
//! The [`NvBufSurfaceGenerator`] creates a DeepStream buffer pool configured
//! with the specified caps, GPU ID, and memory type. It can then allocate
//! NvBufSurface memory and attach it to GStreamer buffers.
//!
//! # Example (Rust)
//!
//! ```rust,no_run
//! use deepstream_nvbufsurface::{NvBufSurfaceGenerator, NvBufSurfaceMemType, VideoFormat};
//!
//! gstreamer::init().unwrap();
//!
//! let generator = NvBufSurfaceGenerator::new(
//!     VideoFormat::RGBA, 640, 480, 30, 1,
//!     0, NvBufSurfaceMemType::Default,
//! ).unwrap();
//!
//! let buffer = generator.acquire_surface(None).unwrap();
//! ```

pub mod ffi;
pub mod transform;

#[cfg(feature = "skia")]
pub mod egl_context;
#[cfg(feature = "skia")]
pub mod skia_renderer;
#[cfg(feature = "skia")]
pub use skia_renderer::SkiaRenderer;

pub use transform::{
    buffer_gpu_id, ComputeMode, Interpolation, Padding, Rect, TransformConfig, TransformError,
};

use glib::translate::from_glib_full;
use gstreamer as gst;
use gstreamer::prelude::*;
use log::debug;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
// Re-export so downstream crates (benches, examples) can use these directly.
pub use savant_gstreamer::id_meta::{SavantIdMeta, SavantIdMetaKind};
pub use savant_gstreamer::VideoFormat;

/// Error type for NvBufSurface operations.
#[derive(Debug, thiserror::Error)]
pub enum NvBufSurfaceError {
    #[error("Failed to create NvDS buffer pool")]
    PoolCreationFailed,

    #[error("Failed to get buffer pool configuration")]
    PoolConfigFailed,

    #[error("Failed to set buffer pool configuration: {0}")]
    PoolSetConfigFailed(String),

    #[error("Failed to activate buffer pool: {0}")]
    PoolActivationFailed(String),

    #[error("Failed to acquire buffer from pool: {0}")]
    BufferAcquisitionFailed(String),

    #[error("Failed to copy buffer contents: {0}")]
    BufferCopyFailed(String),

    #[error("Null pointer: {0}")]
    NullPointer(String),

    #[error("CUDA initialization failed with error code {0}")]
    CudaInitFailed(i32),
}

/// NvBufSurface memory types.
///
/// Specifies the type of memory to allocate for NvBufSurface buffers.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvBufSurfaceMemType {
    /// Default memory type (CUDA Device for dGPU, Surface Array for Jetson).
    Default = 0,
    /// CUDA Host (pinned) memory.
    CudaPinned = 1,
    /// CUDA Device memory.
    CudaDevice = 2,
    /// CUDA Unified memory.
    CudaUnified = 3,
    /// NVRM Surface Array (Jetson only).
    SurfaceArray = 4,
    /// NVRM Handle (Jetson only).
    Handle = 5,
    /// System memory (malloc).
    System = 6,
}

impl From<u32> for NvBufSurfaceMemType {
    fn from(value: u32) -> Self {
        match value {
            0 => Self::Default,
            1 => Self::CudaPinned,
            2 => Self::CudaDevice,
            3 => Self::CudaUnified,
            4 => Self::SurfaceArray,
            5 => Self::Handle,
            6 => Self::System,
            _ => Self::Default,
        }
    }
}

impl From<NvBufSurfaceMemType> for u32 {
    fn from(value: NvBufSurfaceMemType) -> Self {
        value as u32
    }
}

/// Initialize CUDA context for the given GPU device.
///
/// This must be called before creating an [`NvBufSurfaceGenerator`] when
/// not running inside a DeepStream pipeline (which handles CUDA initialization
/// automatically). This is particularly needed in standalone usage and tests.
///
/// Internally, this calls `cudaSetDevice` followed by `cudaFree(NULL)` to
/// trigger lazy CUDA context creation.
///
/// # Arguments
///
/// * `gpu_id` - GPU device ID to initialize (typically 0).
///
/// # Errors
///
/// Returns an error if CUDA initialization fails (e.g., no GPU available).
pub fn cuda_init(gpu_id: u32) -> Result<(), NvBufSurfaceError> {
    extern "C" {
        fn cudaSetDevice(device: i32) -> i32;
        fn cudaFree(dev_ptr: *mut std::ffi::c_void) -> i32;
    }

    unsafe {
        let err = cudaSetDevice(gpu_id as i32);
        if err != 0 {
            return Err(NvBufSurfaceError::CudaInitFailed(err));
        }
        // cudaFree(NULL) triggers lazy CUDA context creation
        let err = cudaFree(std::ptr::null_mut());
        if err != 0 {
            return Err(NvBufSurfaceError::CudaInitFailed(err));
        }
    }

    debug!("CUDA initialized for GPU {}", gpu_id);
    Ok(())
}

/// Create a non-blocking CUDA stream.
///
/// Returns an opaque `*mut c_void` stream handle suitable for passing to
/// [`TransformConfig::cuda_stream`]. The returned stream has the
/// `cudaStreamNonBlocking` flag, meaning it will **not** implicitly
/// synchronize with the CUDA legacy default stream (stream 0). This
/// eliminates the global GPU serialization barrier that occurs when
/// `nvvideoconvert` uses the default stream for `NvBufSurfTransform`.
///
/// The caller must eventually call [`destroy_cuda_stream()`] to free
/// the stream.
///
/// # Errors
///
/// Returns [`NvBufSurfaceError::CudaInitFailed`] if stream creation fails.
pub fn create_cuda_stream() -> Result<*mut std::ffi::c_void, NvBufSurfaceError> {
    let mut stream: *mut std::ffi::c_void = std::ptr::null_mut();
    // 0x01 = cudaStreamNonBlocking
    let ret = unsafe { ffi::cudaStreamCreateWithFlags(&mut stream, 0x01) };
    if ret != 0 {
        return Err(NvBufSurfaceError::CudaInitFailed(ret));
    }
    debug!("Created non-blocking CUDA stream {:?}", stream);
    Ok(stream)
}

/// Destroy a CUDA stream previously created by [`create_cuda_stream()`].
///
/// Passing a null pointer is a no-op.
///
/// # Errors
///
/// Returns [`NvBufSurfaceError::CudaInitFailed`] if destruction fails.
/// # Safety
///
/// `stream` must be a valid CUDA stream handle previously returned by
/// [`create_cuda_stream()`], or null (in which case this is a no-op).
pub unsafe fn destroy_cuda_stream(stream: *mut std::ffi::c_void) -> Result<(), NvBufSurfaceError> {
    if stream.is_null() {
        return Ok(());
    }
    let ret = unsafe { ffi::cudaStreamDestroy(stream) };
    if ret != 0 {
        return Err(NvBufSurfaceError::CudaInitFailed(ret));
    }
    debug!("Destroyed CUDA stream {:?}", stream);
    Ok(())
}

/// Generates GStreamer buffers with NvBufSurface memory allocated.
///
/// Creates a DeepStream buffer pool and provides methods to allocate
/// NvBufSurface memory and attach it to GStreamer buffers. This is the
/// Rust equivalent of the C++ `NvBufSurfaceGenerator` class.
///
/// The generator maintains an internal buffer pool that is configured
/// with the specified video caps, GPU ID, and memory type. Buffers are
/// acquired from the pool via [`acquire_surface`](Self::acquire_surface)
/// or pushed directly to an AppSrc via
/// [`push_to_appsrc`](Self::push_to_appsrc).
///
/// # Example
///
/// ```rust,no_run
/// use deepstream_nvbufsurface::{NvBufSurfaceGenerator, NvBufSurfaceMemType, VideoFormat};
///
/// gstreamer::init().unwrap();
///
/// let gen = NvBufSurfaceGenerator::new(
///     VideoFormat::RGBA, 640, 480, 30, 1,
///     0, NvBufSurfaceMemType::Default,
/// ).unwrap();
///
/// let buffer = gen.acquire_surface(None).unwrap();
/// ```
pub struct NvBufSurfaceGenerator {
    pool: gst::BufferPool,
    format: VideoFormat,
    width: u32,
    height: u32,
    fps_num: i32,
    fps_den: i32,
    gpu_id: u32,
}

/// Builder for [`NvBufSurfaceGenerator`] with advanced pool configuration.
///
/// Provides fine-grained control over the buffer pool. For simple use cases,
/// prefer [`NvBufSurfaceGenerator::new()`] instead.
///
/// # Example
///
/// ```rust,no_run
/// use deepstream_nvbufsurface::{NvBufSurfaceGenerator, NvBufSurfaceMemType, VideoFormat};
///
/// gstreamer::init().unwrap();
///
/// let gen = NvBufSurfaceGenerator::builder(VideoFormat::NV12, 640, 480)
///     .fps(30, 1)
///     .gpu_id(0)
///     .mem_type(NvBufSurfaceMemType::CudaDevice)
///     .min_buffers(4)
///     .max_buffers(8)
///     .build()
///     .unwrap();
/// ```
pub struct NvBufSurfaceGeneratorBuilder {
    format: VideoFormat,
    width: u32,
    height: u32,
    fps_num: i32,
    fps_den: i32,
    gpu_id: u32,
    mem_type: NvBufSurfaceMemType,
    min_buffers: u32,
    max_buffers: u32,
}

impl NvBufSurfaceGeneratorBuilder {
    /// Set the framerate (numerator / denominator).
    pub fn fps(mut self, num: i32, den: i32) -> Self {
        self.fps_num = num;
        self.fps_den = den;
        self
    }

    /// Set the GPU device ID (default: 0).
    pub fn gpu_id(mut self, gpu_id: u32) -> Self {
        self.gpu_id = gpu_id;
        self
    }

    /// Set the NvBufSurface memory type (default: [`NvBufSurfaceMemType::Default`]).
    pub fn mem_type(mut self, mem_type: NvBufSurfaceMemType) -> Self {
        self.mem_type = mem_type;
        self
    }

    /// Set the minimum number of pre-allocated buffers in the pool (default: 0 = dynamic).
    pub fn min_buffers(mut self, min: u32) -> Self {
        self.min_buffers = min;
        self
    }

    /// Set the maximum number of buffers in the pool (default: 0 = unlimited).
    pub fn max_buffers(mut self, max: u32) -> Self {
        self.max_buffers = max;
        self
    }

    /// Build the [`NvBufSurfaceGenerator`].
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer pool cannot be created, configured,
    /// or activated.
    pub fn build(self) -> Result<NvBufSurfaceGenerator, NvBufSurfaceError> {
        let format_str = self.format.gst_name();
        let caps = gst::Caps::builder("video/x-raw")
            .field("format", format_str)
            .field("width", self.width as i32)
            .field("height", self.height as i32)
            .field("framerate", gst::Fraction::new(self.fps_num, self.fps_den))
            .build();

        NvBufSurfaceGenerator::create_from_parts(
            &caps,
            self.format,
            self.width,
            self.height,
            self.fps_num,
            self.fps_den,
            self.gpu_id,
            self.mem_type,
            self.min_buffers,
            self.max_buffers,
        )
    }
}

impl NvBufSurfaceGenerator {
    /// Create a new NvBufSurfaceGenerator with simple parameters.
    ///
    /// This is the primary constructor. It builds the required GStreamer caps
    /// internally, so you only need to specify the video format, dimensions,
    /// and framerate.
    ///
    /// # Arguments
    ///
    /// * `format` - Video pixel format (e.g., [`VideoFormat::RGBA`]).
    /// * `width` - Frame width in pixels.
    /// * `height` - Frame height in pixels.
    /// * `fps_num` - Framerate numerator (e.g., 30).
    /// * `fps_den` - Framerate denominator (e.g., 1).
    /// * `gpu_id` - GPU device ID to allocate buffers on (typically 0).
    /// * `mem_type` - Memory type for the NvBufSurface allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer pool cannot be created, configured,
    /// or activated.
    pub fn new(
        format: VideoFormat,
        width: u32,
        height: u32,
        fps_num: i32,
        fps_den: i32,
        gpu_id: u32,
        mem_type: NvBufSurfaceMemType,
    ) -> Result<Self, NvBufSurfaceError> {
        let format_str = format.gst_name();
        let caps = gst::Caps::builder("video/x-raw")
            .field("format", format_str)
            .field("width", width as i32)
            .field("height", height as i32)
            .field("framerate", gst::Fraction::new(fps_num, fps_den))
            .build();

        Self::create_from_parts(
            &caps, format, width, height, fps_num, fps_den, gpu_id, mem_type, 0, 0,
        )
    }

    /// Create a new NvBufSurfaceGenerator from pre-built GStreamer caps.
    ///
    /// This is the advanced constructor for cases where you already have a
    /// `gst::Caps` instance. The caps must contain `format`, `width`, `height`,
    /// and `framerate` fields.
    ///
    /// # Arguments
    ///
    /// * `caps` - GStreamer caps describing the buffer format (e.g., video/x-raw
    ///   with format, width, height, framerate).
    /// * `gpu_id` - GPU device ID to allocate buffers on (typically 0).
    /// * `mem_type` - Memory type for the NvBufSurface allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer pool cannot be created, configured,
    /// or activated, or if the caps are missing required fields.
    pub fn from_caps(
        caps: &gst::Caps,
        gpu_id: u32,
        mem_type: NvBufSurfaceMemType,
    ) -> Result<Self, NvBufSurfaceError> {
        // Extract format/width/height/fps from caps
        let structure = caps.structure(0).ok_or_else(|| {
            NvBufSurfaceError::PoolSetConfigFailed("Caps has no structure".to_string())
        })?;

        let format_str = structure.get::<String>("format").map_err(|e| {
            NvBufSurfaceError::PoolSetConfigFailed(format!("Caps missing 'format' field: {}", e))
        })?;

        let format = VideoFormat::from_name(&format_str).ok_or_else(|| {
            NvBufSurfaceError::PoolSetConfigFailed(format!(
                "Unsupported video format: '{}'. Expected one of: RGBA, BGRx, NV12, NV21, I420, UYVY, GRAY8",
                format_str,
            ))
        })?;

        let width = structure.get::<i32>("width").map_err(|e| {
            NvBufSurfaceError::PoolSetConfigFailed(format!("Caps missing 'width' field: {}", e))
        })? as u32;

        let height = structure.get::<i32>("height").map_err(|e| {
            NvBufSurfaceError::PoolSetConfigFailed(format!("Caps missing 'height' field: {}", e))
        })? as u32;

        let fps = structure.get::<gst::Fraction>("framerate").map_err(|e| {
            NvBufSurfaceError::PoolSetConfigFailed(format!("Caps missing 'framerate' field: {}", e))
        })?;

        Self::create_from_parts(
            caps,
            format,
            width,
            height,
            fps.numer(),
            fps.denom(),
            gpu_id,
            mem_type,
            0,
            0,
        )
    }

    /// Return a builder for advanced pool configuration.
    ///
    /// The builder requires `format`, `width`, and `height`. All other
    /// parameters have sensible defaults.
    pub fn builder(format: VideoFormat, width: u32, height: u32) -> NvBufSurfaceGeneratorBuilder {
        NvBufSurfaceGeneratorBuilder {
            format,
            width,
            height,
            fps_num: 30,
            fps_den: 1,
            gpu_id: 0,
            mem_type: NvBufSurfaceMemType::Default,
            min_buffers: 0,
            max_buffers: 0,
        }
    }

    /// Internal constructor used by both [`new()`](Self::new),
    /// [`from_caps()`](Self::from_caps), and the builder.
    #[allow(clippy::too_many_arguments)]
    fn create_from_parts(
        caps: &gst::Caps,
        format: VideoFormat,
        width: u32,
        height: u32,
        fps_num: i32,
        fps_den: i32,
        gpu_id: u32,
        mem_type: NvBufSurfaceMemType,
        min_buffers: u32,
        max_buffers: u32,
    ) -> Result<Self, NvBufSurfaceError> {
        debug!("Creating NvBufSurfaceGenerator");

        // Create DeepStream buffer pool
        let raw_pool = unsafe { ffi::gst_nvds_buffer_pool_new() };
        if raw_pool.is_null() {
            return Err(NvBufSurfaceError::PoolCreationFailed);
        }

        let pool: gst::BufferPool = unsafe { from_glib_full(raw_pool) };
        debug!("Buffer pool created");

        // Configure the pool using unsafe GStreamer C API.
        // We use the C API directly because we need to set DeepStream-specific
        // configuration fields (memtype, gpu-id) on the config structure,
        // which is most reliably done via the C structure API.
        //
        // This matches the configuration pattern from DeepStream's own
        // nvvideotestsrc plugin (see gst-nvvideotestsrc/gstnvvideotestsrc.cpp).
        let mem_type_u32: u32 = mem_type.into();
        let nvbufsurface_size = std::mem::size_of::<ffi::NvBufSurface>();

        unsafe {
            let config = gst::ffi::gst_buffer_pool_get_config(pool.as_ptr());
            if config.is_null() {
                return Err(NvBufSurfaceError::PoolConfigFailed);
            }

            // Set standard buffer pool parameters:
            // caps, buffer size = sizeof(NvBufSurface), min/max buffers.
            // The NvDsBufferPool GstBuffer memory simply contains the
            // NvBufSurface descriptor which in turn points to the actual
            // NVMM GPU buffer allocation(s).
            gst::ffi::gst_buffer_pool_config_set_params(
                config,
                caps.as_ptr() as *mut gst::ffi::GstCaps,
                nvbufsurface_size as u32,
                min_buffers,
                max_buffers,
            );

            // Set DeepStream-specific configuration fields.
            // NOTE: batch-size is intentionally NOT set here; it is not
            // used by nvvideotestsrc and causes issues with some pipelines.
            set_structure_uint(config, "memtype", mem_type_u32);
            set_structure_uint(config, "gpu-id", gpu_id);

            // Apply configuration to pool
            let ret = gst::ffi::gst_buffer_pool_set_config(pool.as_ptr(), config);
            if ret == glib::ffi::GFALSE {
                return Err(NvBufSurfaceError::PoolSetConfigFailed(
                    "gst_buffer_pool_set_config returned FALSE".to_string(),
                ));
            }
        }
        debug!("Buffer pool configured");

        // Activate the pool
        pool.set_active(true)
            .map_err(|e| NvBufSurfaceError::PoolActivationFailed(e.to_string()))?;

        debug!("NvBufSurfaceGenerator created successfully");
        Ok(Self {
            pool,
            format,
            width,
            height,
            fps_num,
            fps_den,
            gpu_id,
        })
    }

    /// Return NVMM-featured caps matching this generator's format/dimensions.
    ///
    /// The returned caps have the `memory:NVMM` feature set and can be used
    /// to configure an `appsrc` element for zero-copy NVMM buffer pushing.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use deepstream_nvbufsurface::{NvBufSurfaceGenerator, NvBufSurfaceMemType, VideoFormat};
    /// # use gstreamer::prelude::*;
    /// # gstreamer::init().unwrap();
    /// let gen = NvBufSurfaceGenerator::new(
    ///     VideoFormat::NV12, 640, 480, 30, 1, 0, NvBufSurfaceMemType::Default,
    /// ).unwrap();
    ///
    /// let appsrc = gstreamer::ElementFactory::make("appsrc").build().unwrap();
    /// appsrc.set_property("caps", &gen.nvmm_caps());
    /// ```
    pub fn nvmm_caps(&self) -> gst::Caps {
        gst::Caps::builder("video/x-raw")
            .features(["memory:NVMM"])
            .field("format", self.format.gst_name())
            .field("width", self.width as i32)
            .field("height", self.height as i32)
            .field("framerate", gst::Fraction::new(self.fps_num, self.fps_den))
            .build()
    }

    /// Return raw (non-NVMM) caps matching this generator's format/dimensions.
    ///
    /// These caps describe the video format without the `memory:NVMM` feature.
    /// This is the format used internally for pool configuration.
    pub fn raw_caps(&self) -> gst::Caps {
        gst::Caps::builder("video/x-raw")
            .field("format", self.format.gst_name())
            .field("width", self.width as i32)
            .field("height", self.height as i32)
            .field("framerate", gst::Fraction::new(self.fps_num, self.fps_den))
            .build()
    }

    /// Acquire a buffer from the internal pool with NvBufSurface ready for use.
    ///
    /// Returns a GStreamer buffer whose memory contains a fully initialized
    /// [`NvBufSurface`] descriptor with `numFilled` set to 1. The underlying
    /// GPU memory (referenced by the surface's `dataPtr`) is allocated by the
    /// pool and remains valid while the buffer is alive.
    ///
    /// When the returned buffer is dropped (or unreffed by a pipeline), it is
    /// automatically returned to the pool for reuse. This is the recommended
    /// way to obtain NVMM buffers for pushing into a GStreamer pipeline via
    /// `appsrc`, matching the pattern used by DeepStream's own
    /// `nvvideotestsrc` plugin.
    ///
    /// # Arguments
    ///
    /// * `id` - Optional frame identifier. When `Some(id)`, a
    ///   [`SavantIdMeta`](savant_gstreamer::id_meta::SavantIdMeta) containing
    ///   `Frame(id)` is attached to the buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the pool cannot provide a buffer.
    pub fn acquire_surface(&self, id: Option<i64>) -> Result<gst::Buffer, NvBufSurfaceError> {
        debug!("Acquiring NvBufSurface from pool");

        let mut buffer = self
            .pool
            .acquire_buffer(None)
            .map_err(|e| NvBufSurfaceError::BufferAcquisitionFailed(format!("{:?}", e)))?;

        // Map the buffer and set numFilled = 1 on the NvBufSurface descriptor.
        // This is required by downstream DeepStream plugins (nvstreammux,
        // nvvideoconvert, nvv4l2h265enc, etc.) to recognize the buffer as
        // containing a valid NVMM surface.
        //
        // NvBufSurface layout (first 12 bytes):
        //   offset 0: gpuId      (u32)
        //   offset 4: batchSize  (u32)
        //   offset 8: numFilled  (u32)
        {
            let buf_ref = buffer.make_mut();
            let mut map = buf_ref.map_writable().map_err(|e| {
                NvBufSurfaceError::BufferAcquisitionFailed(format!("Failed to map buffer: {:?}", e))
            })?;
            let data = map.as_mut_slice();
            if data.len() >= 12 {
                // Set numFilled = 1 (native-endian u32 at offset 8)
                data[8..12].copy_from_slice(&1u32.to_ne_bytes());
            }
        }

        // Attach SavantIdMeta if an id was provided
        if let Some(frame_id) = id {
            let buf_ref = buffer.make_mut();
            SavantIdMeta::replace(buf_ref, vec![SavantIdMetaKind::Frame(frame_id)]);
        }

        debug!("NvBufSurface acquired with numFilled=1");
        Ok(buffer)
    }

    /// Acquire a buffer and extract the GPU data pointer and pitch in one step.
    ///
    /// This is an optimized variant of [`acquire_surface`](Self::acquire_surface)
    /// that returns `(buffer, dataPtr, pitch)` using a single writable map
    /// instead of separate writable + readable maps. This eliminates one
    /// GstBuffer map/unmap cycle per frame.
    ///
    /// The returned `dataPtr` points to the NvBufSurface's GPU-allocated
    /// memory and `pitch` is the row stride in bytes.
    ///
    /// # Arguments
    ///
    /// * `id` - Optional frame identifier. When `Some(id)`, a
    ///   [`SavantIdMeta`](savant_gstreamer::id_meta::SavantIdMeta) containing
    ///   `Frame(id)` is attached to the buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the pool cannot provide a buffer or the buffer
    /// is too small to contain a valid NvBufSurface descriptor.
    pub fn acquire_surface_with_ptr(
        &self,
        id: Option<i64>,
    ) -> Result<(gst::Buffer, *mut std::ffi::c_void, u32), NvBufSurfaceError> {
        debug!("Acquiring NvBufSurface from pool (with ptr)");

        let mut buffer = self
            .pool
            .acquire_buffer(None)
            .map_err(|e| NvBufSurfaceError::BufferAcquisitionFailed(format!("{:?}", e)))?;

        // Single writable map: set numFilled=1 and read dataPtr/pitch
        let (data_ptr, pitch) = {
            let buf_ref = buffer.make_mut();
            let mut map = buf_ref.map_writable().map_err(|e| {
                NvBufSurfaceError::BufferAcquisitionFailed(format!("Failed to map buffer: {:?}", e))
            })?;
            let data = map.as_mut_slice();

            if data.len() < std::mem::size_of::<ffi::NvBufSurface>() {
                return Err(NvBufSurfaceError::BufferAcquisitionFailed(
                    "Buffer too small for NvBufSurface".into(),
                ));
            }

            // Set numFilled = 1 (offset 8, u32)
            data[8..12].copy_from_slice(&1u32.to_ne_bytes());

            // Read dataPtr and pitch from the NvBufSurface descriptor
            let surface_ptr = data.as_ptr() as *const ffi::NvBufSurface;
            let surface = unsafe { &*surface_ptr };
            let params = unsafe { &*surface.surfaceList };
            (params.dataPtr, params.pitch)
        };

        // Attach SavantIdMeta if an id was provided
        if let Some(frame_id) = id {
            let buf_ref = buffer.make_mut();
            SavantIdMeta::replace(buf_ref, vec![SavantIdMetaKind::Frame(frame_id)]);
        }

        debug!(
            "NvBufSurface acquired: dataPtr={:?}, pitch={}",
            data_ptr, pitch
        );
        Ok((buffer, data_ptr, pitch))
    }

    /// Push a new NVMM buffer to an AppSrc element.
    ///
    /// This is a convenience method that combines
    /// [`acquire_surface`](Self::acquire_surface), setting PTS/duration, and
    /// pushing the buffer to the given `appsrc`. The buffer is acquired from
    /// the pool, configured with the specified timestamps, and pushed in a
    /// single call -- zero CPU-to-GPU memory transfer.
    ///
    /// # Arguments
    ///
    /// * `appsrc` - The GStreamer AppSrc element to push to.
    /// * `pts_ns` - Presentation timestamp in nanoseconds.
    /// * `duration_ns` - Buffer duration in nanoseconds.
    /// * `id` - Optional frame identifier. When `Some(id)`, a
    ///   [`SavantIdMeta`](savant_gstreamer::id_meta::SavantIdMeta) containing
    ///   `Frame(id)` is attached to the buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if buffer acquisition or the push fails.
    pub fn push_to_appsrc(
        &self,
        appsrc: &gstreamer_app::AppSrc,
        pts_ns: u64,
        duration_ns: u64,
        id: Option<i64>,
    ) -> Result<(), NvBufSurfaceError> {
        let mut buffer = self.acquire_surface(id)?;

        {
            let buf_ref = buffer.get_mut().unwrap();
            buf_ref.set_pts(gst::ClockTime::from_nseconds(pts_ns));
            buf_ref.set_duration(gst::ClockTime::from_nseconds(duration_ns));
        }

        appsrc.push_buffer(buffer).map_err(|e| {
            NvBufSurfaceError::BufferAcquisitionFailed(format!("AppSrc push failed: {:?}", e))
        })?;

        Ok(())
    }

    /// Push a new NVMM buffer to an AppSrc given as a raw pointer.
    ///
    /// This is the FFI-friendly variant of
    /// [`push_to_appsrc`](Self::push_to_appsrc), accepting the AppSrc as a
    /// raw `GstElement*` pointer (e.g., from `hash(appsrc)` in Python).
    ///
    /// # Safety
    ///
    /// The caller must ensure that `appsrc_ptr` is a valid, non-null pointer
    /// to a `GstAppSrc` element that remains valid for the duration of this call.
    pub unsafe fn push_to_appsrc_raw(
        &self,
        appsrc_ptr: usize,
        pts_ns: u64,
        duration_ns: u64,
        id: Option<i64>,
    ) -> Result<(), NvBufSurfaceError> {
        if appsrc_ptr == 0 {
            return Err(NvBufSurfaceError::NullPointer(
                "appsrc pointer is null".to_string(),
            ));
        }

        let mut buffer = self.acquire_surface(id)?;
        {
            let buf_ref = buffer.get_mut().unwrap();
            buf_ref.set_pts(gst::ClockTime::from_nseconds(pts_ns));
            buf_ref.set_duration(gst::ClockTime::from_nseconds(duration_ns));
        }

        // Use raw FFI to push the buffer.
        // gst_app_src_push_buffer takes ownership of the buffer reference,
        // so we must convert our gst::Buffer into a raw pointer without
        // dropping it (into_glib_ptr does exactly this).
        let buf_ptr = {
            use glib::translate::IntoGlibPtr;
            buffer.into_glib_ptr()
        };

        extern "C" {
            fn gst_app_src_push_buffer(
                appsrc: *mut gst::ffi::GstElement,
                buffer: *mut gst::ffi::GstBuffer,
            ) -> i32;
        }

        let ret = gst_app_src_push_buffer(appsrc_ptr as *mut gst::ffi::GstElement, buf_ptr);

        if ret != 0 {
            return Err(NvBufSurfaceError::BufferAcquisitionFailed(format!(
                "gst_app_src_push_buffer returned {}",
                ret
            )));
        }

        Ok(())
    }

    /// Transform (scale + letterbox) a source NvBufSurface buffer into a
    /// new destination buffer from this generator's pool.
    ///
    /// Handles aspect-ratio-preserving scaling with configurable padding
    /// (symmetric or right-bottom), using the same NvBufSurfTransform API
    /// that nvinfer uses internally for ROI preparation.
    ///
    /// # Arguments
    ///
    /// * `src` - Source GStreamer buffer containing an NvBufSurface.
    /// * `config` - Transform configuration (padding, interpolation, etc.).
    /// * `id` - Optional frame identifier for SavantIdMeta attachment.
    /// * `src_rect` - Optional source crop rectangle. `None` means full source.
    ///
    /// # Errors
    ///
    /// Returns an error if the source buffer is invalid, the pool cannot
    /// provide a buffer, or the transform call fails.
    pub fn transform(
        &self,
        src: &gst::Buffer,
        config: &TransformConfig,
        id: Option<i64>,
        src_rect: Option<&Rect>,
    ) -> Result<gst::Buffer, NvBufSurfaceError> {
        // Acquire destination buffer from our pool
        let dst_buf = self.acquire_surface(id)?;

        // Extract NvBufSurface pointers from both src and dst
        let src_surf = unsafe {
            transform::extract_nvbufsurface(src.as_ref())
                .map_err(|e| NvBufSurfaceError::BufferCopyFailed(e.to_string()))?
        };
        let dst_surf = unsafe {
            transform::extract_nvbufsurface(dst_buf.as_ref())
                .map_err(|e| NvBufSurfaceError::BufferCopyFailed(e.to_string()))?
        };

        // Perform the transform
        unsafe {
            transform::do_transform(src_surf, dst_surf, config, src_rect)
                .map_err(|e| NvBufSurfaceError::BufferCopyFailed(e.to_string()))?;
        }

        Ok(dst_buf)
    }

    /// Like [`transform()`](Self::transform) but also returns `(data_ptr, pitch)`
    /// for direct GPU memory access (e.g. for Skia overlay).
    ///
    /// # Arguments
    ///
    /// * `src` - Source GStreamer buffer containing an NvBufSurface.
    /// * `config` - Transform configuration.
    /// * `id` - Optional frame identifier for SavantIdMeta attachment.
    /// * `src_rect` - Optional source crop rectangle. `None` means full source.
    ///
    /// # Errors
    ///
    /// Returns an error if the transform fails or the buffer is invalid.
    pub fn transform_with_ptr(
        &self,
        src: &gst::Buffer,
        config: &TransformConfig,
        id: Option<i64>,
        src_rect: Option<&Rect>,
    ) -> Result<(gst::Buffer, *mut std::ffi::c_void, u32), NvBufSurfaceError> {
        // Acquire destination with ptr info
        let (dst_buf, data_ptr, pitch) = self.acquire_surface_with_ptr(id)?;

        // Extract NvBufSurface pointers
        let src_surf = unsafe {
            transform::extract_nvbufsurface(src.as_ref())
                .map_err(|e| NvBufSurfaceError::BufferCopyFailed(e.to_string()))?
        };
        let dst_surf = unsafe {
            transform::extract_nvbufsurface(dst_buf.as_ref())
                .map_err(|e| NvBufSurfaceError::BufferCopyFailed(e.to_string()))?
        };

        // Perform the transform
        unsafe {
            transform::do_transform(src_surf, dst_surf, config, src_rect)
                .map_err(|e| NvBufSurfaceError::BufferCopyFailed(e.to_string()))?;
        }

        Ok((dst_buf, data_ptr, pitch))
    }

    /// Get the width of buffers produced by this generator.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get the height of buffers produced by this generator.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get the video format of buffers produced by this generator.
    pub fn format(&self) -> VideoFormat {
        self.format
    }

    /// Get the GPU device ID this generator allocates buffers on.
    pub fn gpu_id(&self) -> u32 {
        self.gpu_id
    }

    /// Send an end-of-stream signal to an AppSrc element.
    ///
    /// # Arguments
    ///
    /// * `appsrc` - The GStreamer AppSrc element to signal.
    ///
    /// # Errors
    ///
    /// Returns an error if the EOS signal fails.
    pub fn send_eos(appsrc: &gstreamer_app::AppSrc) -> Result<(), NvBufSurfaceError> {
        appsrc.end_of_stream().map_err(|e| {
            NvBufSurfaceError::BufferAcquisitionFailed(format!("AppSrc EOS failed: {:?}", e))
        })?;
        Ok(())
    }

    /// Send an end-of-stream signal to an AppSrc given as a raw pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `appsrc_ptr` is a valid, non-null pointer
    /// to a `GstAppSrc` element.
    pub unsafe fn send_eos_raw(appsrc_ptr: usize) -> Result<(), NvBufSurfaceError> {
        if appsrc_ptr == 0 {
            return Err(NvBufSurfaceError::NullPointer(
                "appsrc pointer is null".to_string(),
            ));
        }

        extern "C" {
            fn gst_app_src_end_of_stream(appsrc: *mut gst::ffi::GstElement) -> i32;
        }

        let ret = gst_app_src_end_of_stream(appsrc_ptr as *mut gst::ffi::GstElement);
        if ret != 0 {
            return Err(NvBufSurfaceError::BufferAcquisitionFailed(format!(
                "gst_app_src_end_of_stream returned {}",
                ret
            )));
        }

        Ok(())
    }

    /// Create a new NvBufSurface and attach it to the given buffer.
    ///
    /// Acquires a buffer from the internal pool and copies all of its memory,
    /// metadata, and flags into the destination buffer via a **shared** memory
    /// reference (`gst_memory_share`). This means the destination buffer and
    /// the pool buffer share the same NvBufSurface descriptor memory.
    ///
    /// **Important**: Because the copy is shallow, the destination buffer must
    /// be fully consumed (processed and unreffed) before the pool buffer is
    /// reacquired for the next frame. For pipeline use with `appsrc`, prefer
    /// [`acquire_surface`](Self::acquire_surface) which returns the pool buffer
    /// directly, avoiding shared-memory lifetime issues.
    ///
    /// # Arguments
    ///
    /// * `dest` - Mutable reference to the destination GStreamer buffer.
    /// * `id` - Optional frame identifier. When `Some(id)`, a
    ///   [`SavantIdMeta`](savant_gstreamer::id_meta::SavantIdMeta) containing
    ///   `Frame(id)` is attached to the destination buffer after copying.
    ///
    /// # Errors
    ///
    /// Returns an error if buffer acquisition or copying fails.
    pub fn create_surface(
        &self,
        dest: &mut gst::BufferRef,
        id: Option<i64>,
    ) -> Result<(), NvBufSurfaceError> {
        debug!("Creating NvBufSurface");

        // Acquire a buffer from the pool
        let acquired = self
            .pool
            .acquire_buffer(None)
            .map_err(|e| NvBufSurfaceError::BufferAcquisitionFailed(format!("{:?}", e)))?;
        debug!("Buffer acquired from pool");

        // Copy the acquired buffer's content into the destination.
        // This copies memory, metadata, timestamps, and flags.
        unsafe {
            let ret = gst::ffi::gst_buffer_copy_into(
                dest.as_mut_ptr(),
                acquired.as_ptr() as *mut gst::ffi::GstBuffer,
                gst::ffi::GST_BUFFER_COPY_FLAGS
                    | gst::ffi::GST_BUFFER_COPY_TIMESTAMPS
                    | gst::ffi::GST_BUFFER_COPY_META
                    | gst::ffi::GST_BUFFER_COPY_MEMORY,
                0,
                usize::MAX, // -1 in C: copy everything
            );
            if ret == glib::ffi::GFALSE {
                return Err(NvBufSurfaceError::BufferCopyFailed(
                    "gst_buffer_copy_into returned FALSE".to_string(),
                ));
            }
        }
        debug!("Buffer content copied to destination");

        // Attach SavantIdMeta if an id was provided
        if let Some(frame_id) = id {
            SavantIdMeta::replace(dest, vec![SavantIdMetaKind::Frame(frame_id)]);
        }

        // The acquired buffer is automatically returned to the pool when dropped.
        Ok(())
    }

    /// Create a new NvBufSurface and attach it to the buffer at the given raw pointer.
    ///
    /// This is the raw pointer variant of [`create_surface`](Self::create_surface),
    /// intended for use from FFI/Python bindings where the buffer is passed as
    /// a raw pointer address.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `dest` is a valid, non-null pointer to a
    /// `GstBuffer` that remains valid for the duration of this call.
    pub unsafe fn create_surface_raw(
        &self,
        dest: *mut gst::ffi::GstBuffer,
        id: Option<i64>,
    ) -> Result<(), NvBufSurfaceError> {
        if dest.is_null() {
            return Err(NvBufSurfaceError::NullPointer(
                "dest buffer is null".to_string(),
            ));
        }
        let dest_ref = gst::BufferRef::from_mut_ptr(dest);
        self.create_surface(dest_ref, id)
    }
}

impl Drop for NvBufSurfaceGenerator {
    fn drop(&mut self) {
        debug!("Destroying NvBufSurfaceGenerator");
        if let Err(e) = self.pool.set_active(false) {
            log::warn!("Failed to deactivate buffer pool on drop: {}", e);
        }
        debug!("NvBufSurfaceGenerator destroyed");
    }
}

/// Helper: set a uint field on a GstStructure using glib's safe Value API.
///
/// This avoids calling the variadic `gst_structure_set()` which cannot be
/// called safely from Rust.
unsafe fn set_structure_uint(structure: *mut gst::ffi::GstStructure, field_name: &str, value: u32) {
    use glib::prelude::ToValue;
    use glib::translate::ToGlibPtr;
    let c_name = std::ffi::CString::new(field_name).unwrap();
    let gvalue = value.to_value();
    gst::ffi::gst_structure_set_value(structure, c_name.as_ptr(), gvalue.to_glib_none().0);
}

// ─── PTS-keyed meta bridge ───────────────────────────────────────────────────

/// Install pad probes on `element` to propagate [`SavantIdMeta`] across
/// elements that create new output buffers (e.g. hardware video encoders).
///
/// Hardware encoders like `nvv4l2h265enc` allocate fresh buffers for the
/// compressed bitstream and do **not** copy custom `GstMeta` from input to
/// output.  This function works around that limitation by using PTS-keyed
/// side-channel storage:
///
/// 1. A **sink-pad** probe intercepts each incoming buffer, reads any
///    `SavantIdMeta`, and stores the mapping `PTS → Vec<SavantIdMetaKind>`
///    in a shared `HashMap`.
/// 2. A **src-pad** probe intercepts each outgoing buffer, looks up the PTS
///    in the map, and re-attaches the `SavantIdMeta`.
///
/// PTS is guaranteed to be preserved by all GStreamer encoder elements.
/// B-frame reordering is handled naturally because lookups are by value,
/// not by order.
///
/// # Panics
///
/// Panics if `element` does not have both `sink` and `src` static pads.
///
/// # Example
///
/// ```rust,no_run
/// # use deepstream_nvbufsurface::bridge_savant_id_meta;
/// # use gstreamer as gst;
/// # use gstreamer::prelude::*;
/// # gstreamer::init().unwrap();
/// let enc = gst::ElementFactory::make("nvv4l2h265enc")
///     .build()
///     .unwrap();
/// bridge_savant_id_meta(&enc);
/// // From this point, SavantIdMeta on buffers entering the encoder's
/// // sink pad will automatically appear on the encoder's src pad output.
/// ```
pub fn bridge_savant_id_meta(element: &gst::Element) {
    let map: Arc<Mutex<HashMap<u64, Vec<SavantIdMetaKind>>>> = Arc::new(Mutex::new(HashMap::new()));

    // ── Sink pad probe: extract meta, store by PTS ──────────────────────
    let sink_map = map.clone();
    let sink_pad = element
        .static_pad("sink")
        .expect("bridge_savant_id_meta: element has no 'sink' pad");

    sink_pad.add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
        if let Some(buffer) = info.buffer() {
            if let Some(meta) = buffer.meta::<SavantIdMeta>() {
                if let Some(pts) = buffer.pts() {
                    let ids = meta.ids().to_vec();
                    sink_map.lock().unwrap().insert(pts.nseconds(), ids);
                }
            }
        }
        gst::PadProbeReturn::Ok
    });

    // ── Src pad probe: look up PTS, re-attach meta ──────────────────────
    let src_map = map;
    let src_pad = element
        .static_pad("src")
        .expect("bridge_savant_id_meta: element has no 'src' pad");

    src_pad.add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
        if let Some(buffer) = info.buffer_mut() {
            if let Some(pts) = buffer.pts() {
                if let Some(ids) = src_map.lock().unwrap().remove(&pts.nseconds()) {
                    let buf_ref = buffer.make_mut();
                    SavantIdMeta::replace(buf_ref, ids);
                }
            }
        }
        gst::PadProbeReturn::Ok
    });
}

// PyO3 Python bindings have been moved to savant_core_py::deepstream.
// Enable the `deepstream` feature on savant_core_py / savant_python to use them.
