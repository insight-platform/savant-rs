//! Single-frame NvBufSurface buffer generator.
//!
//! [`BufferGenerator`] is a convenience wrapper around
//! [`UniformBatchGenerator`](crate::UniformBatchGenerator) with
//! `max_batch_size = 1`.
//!
//! GStreamer AppSrc helpers (push, EOS) live in the [`gst_app`] submodule.

pub mod gst_app;

use crate::{
    NvBufSurfaceError, NvBufSurfaceMemType, Rect, TransformConfig, UniformBatchGenerator,
    UniformBatchGeneratorBuilder, VideoFormat,
};
use gstreamer as gst;

/// Generates GStreamer buffers with single-frame NvBufSurface memory.
///
/// Internally delegates to [`UniformBatchGenerator`] with
/// `max_batch_size = 1`.
///
/// # Example
///
/// ```rust,no_run
/// use deepstream_buffers::{BufferGenerator, NvBufSurfaceMemType, VideoFormat};
///
/// gstreamer::init().unwrap();
///
/// let gen = BufferGenerator::new(
///     VideoFormat::RGBA, 640, 480, 30, 1,
///     0, NvBufSurfaceMemType::Default,
/// ).unwrap();
///
/// let buffer = gen.acquire(None).unwrap();
/// ```
pub struct BufferGenerator(UniformBatchGenerator);

/// Builder for [`BufferGenerator`] with advanced pool configuration.
///
/// Provides fine-grained control over the buffer pool. For simple use cases,
/// prefer [`BufferGenerator::new()`] instead.
///
/// # Example
///
/// ```rust,no_run
/// use deepstream_buffers::{BufferGenerator, NvBufSurfaceMemType, VideoFormat};
///
/// gstreamer::init().unwrap();
///
/// let gen = BufferGenerator::builder(VideoFormat::NV12, 640, 480)
///     .fps(30, 1)
///     .gpu_id(0)
///     .mem_type(NvBufSurfaceMemType::CudaDevice)
///     .min_buffers(4)
///     .max_buffers(8)
///     .build()
///     .unwrap();
/// ```
pub struct BufferGeneratorBuilder(UniformBatchGeneratorBuilder);

impl BufferGeneratorBuilder {
    /// Set the framerate (numerator / denominator).
    pub fn fps(mut self, num: i32, den: i32) -> Self {
        self.0 = self.0.fps(num, den);
        self
    }

    /// Set the GPU device ID (default: 0).
    pub fn gpu_id(mut self, gpu_id: u32) -> Self {
        self.0 = self.0.gpu_id(gpu_id);
        self
    }

    /// Set the NvBufSurface memory type (default: [`NvBufSurfaceMemType::Default`]).
    pub fn mem_type(mut self, mem_type: NvBufSurfaceMemType) -> Self {
        self.0 = self.0.mem_type(mem_type);
        self
    }

    /// Set the minimum number of pre-allocated buffers in the pool (default: 0 = dynamic).
    pub fn min_buffers(mut self, min: u32) -> Self {
        self.0 = self.0.min_buffers(min);
        self
    }

    /// Set the maximum number of buffers in the pool (default: 0 = unlimited).
    pub fn max_buffers(mut self, max: u32) -> Self {
        self.0 = self.0.max_buffers(max);
        self
    }

    /// Build the [`BufferGenerator`].
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer pool cannot be created, configured,
    /// or activated.
    pub fn build(self) -> Result<BufferGenerator, NvBufSurfaceError> {
        let inner = self.0.build()?;
        Ok(BufferGenerator(inner))
    }
}

impl BufferGenerator {
    /// Create a new BufferGenerator with simple parameters.
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
        let inner = UniformBatchGenerator::create_pool(
            format, width, height, 1, 0, 0, fps_num, fps_den, gpu_id, mem_type,
        )?;
        Ok(Self(inner))
    }

    /// Return a builder for advanced pool configuration.
    ///
    /// The builder requires `format`, `width`, and `height`. All other
    /// parameters have sensible defaults.
    pub fn builder(format: VideoFormat, width: u32, height: u32) -> BufferGeneratorBuilder {
        BufferGeneratorBuilder(
            UniformBatchGenerator::builder(format, width, height, 1)
                .min_buffers(0)
                .max_buffers(0),
        )
    }

    /// Return NVMM-featured caps matching this generator's format/dimensions.
    ///
    /// The returned caps have the `memory:NVMM` feature set and can be used
    /// to configure an `appsrc` element for zero-copy NVMM buffer pushing.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use deepstream_buffers::{BufferGenerator, NvBufSurfaceMemType, VideoFormat};
    /// # use gstreamer::prelude::*;
    /// # gstreamer::init().unwrap();
    /// let gen = BufferGenerator::new(
    ///     VideoFormat::NV12, 640, 480, 30, 1, 0, NvBufSurfaceMemType::Default,
    /// ).unwrap();
    ///
    /// let appsrc = gstreamer::ElementFactory::make("appsrc").build().unwrap();
    /// // gen.nvmm_caps() returns a String representation of the caps
    /// ```
    pub fn nvmm_caps(&self) -> String {
        self.0.nvmm_caps()
    }

    /// Return NVMM caps as a `gst::Caps` object (for pipeline integration).
    pub(crate) fn nvmm_caps_gst(&self) -> gstreamer::Caps {
        self.0.nvmm_caps_gst()
    }

    /// Return raw (non-NVMM) caps matching this generator's format/dimensions.
    pub fn raw_caps(&self) -> String {
        self.0.raw_caps()
    }

    /// Return raw caps as a `gst::Caps` object (for pipeline integration).
    pub(crate) fn raw_caps_gst(&self) -> gstreamer::Caps {
        self.0.raw_caps_gst()
    }

    /// Acquire a buffer from the internal pool with NvBufSurface ready for use.
    ///
    /// Returns a [`SharedBuffer`](crate::SharedBuffer) whose memory contains a
    /// fully initialized `NvBufSurface` descriptor with `numFilled` set to 1.
    /// The underlying GPU memory is allocated by the pool and remains valid
    /// while the buffer is alive.
    ///
    /// When the returned buffer is dropped (or unreffed by a pipeline), it is
    /// automatically returned to the pool for reuse.
    ///
    /// **Blocks** until a buffer becomes available when the pool is exhausted.
    /// See [`try_acquire`](Self::try_acquire) for a non-blocking alternative.
    ///
    /// # Arguments
    ///
    /// * `id` - Optional frame identifier. When `Some(id)`, a
    ///   [`SavantIdMeta`](savant_gstreamer::id_meta::SavantIdMeta) containing
    ///   `Frame(id)` is attached to the buffer.
    pub fn acquire(&self, id: Option<u128>) -> Result<crate::SharedBuffer, NvBufSurfaceError> {
        self.0.acquire(id)
    }

    /// Non-blocking variant of [`acquire`](Self::acquire).
    ///
    /// Returns [`NvBufSurfaceError::PoolExhausted`] immediately when all pool
    /// buffers are currently in use, instead of blocking the caller.
    pub fn try_acquire(&self, id: Option<u128>) -> Result<crate::SharedBuffer, NvBufSurfaceError> {
        self.0.try_acquire(id)
    }

    /// Transform (scale + letterbox) a source surface into a new destination
    /// buffer from this generator's pool.
    ///
    /// Handles aspect-ratio-preserving scaling with configurable padding
    /// (symmetric or right-bottom), using the same NvBufSurfTransform API
    /// that nvinfer uses internally for ROI preparation.
    ///
    /// The internally-constructed destination [`SurfaceView`] inherits
    /// `config.cuda_stream`, so both the transform itself (driven by
    /// [`SurfaceView::transform_into`]) and the destination's
    /// `Drop::sync()` run on the caller-provided stream.  This avoids
    /// the legacy default stream serialization point that
    /// [`SurfaceView::from_buffer`] would otherwise introduce.
    ///
    /// # Arguments
    ///
    /// * `src` - Source [`SurfaceView`](crate::SurfaceView) identifying the
    ///   frame to transform.
    /// * `config` - Transform configuration (padding, interpolation, etc.).
    /// * `src_rect` - Optional source crop rectangle. `None` means full source.
    pub fn transform(
        &self,
        src: &crate::SurfaceView,
        config: &TransformConfig,
        src_rect: Option<&Rect>,
    ) -> Result<crate::SharedBuffer, NvBufSurfaceError> {
        let shared = self.0.acquire(None)?;
        let dst_view = crate::SurfaceView::from_buffer(&shared, 0)?
            .with_cuda_stream(config.cuda_stream.clone());
        src.transform_into(&dst_view, config, src_rect)?;
        drop(dst_view);
        Ok(shared)
    }

    /// Transform a source surface and immediately extract the resulting
    /// `gst::Buffer`.
    ///
    /// Equivalent to [`transform`](Self::transform) followed by
    /// [`SharedBuffer::into_buffer`](crate::SharedBuffer::into_buffer),
    /// but avoids the need to manage `SharedBuffer` lifetime manually.
    ///
    /// # Arguments
    ///
    /// * `src` - Source [`SurfaceView`](crate::SurfaceView) to transform.
    /// * `config` - Transform configuration (padding, interpolation, etc.).
    /// * `src_rect` - Optional source crop rectangle. `None` means full source.
    pub fn transform_to_buffer(
        &self,
        src: &crate::SurfaceView,
        config: &TransformConfig,
        src_rect: Option<&Rect>,
    ) -> Result<gst::Buffer, NvBufSurfaceError> {
        let shared = self.transform(src, config, src_rect)?;
        shared.into_buffer().map_err(|_| {
            NvBufSurfaceError::BufferAcquisitionFailed(
                "SharedBuffer has outstanding references after transform".into(),
            )
        })
    }

    /// Get the width of buffers produced by this generator.
    pub fn width(&self) -> u32 {
        self.0.width()
    }

    /// Get the height of buffers produced by this generator.
    pub fn height(&self) -> u32 {
        self.0.height()
    }

    /// Get the video format of buffers produced by this generator.
    pub fn format(&self) -> VideoFormat {
        self.0.format()
    }

    /// Get the GPU device ID this generator allocates buffers on.
    pub fn gpu_id(&self) -> u32 {
        self.0.gpu_id()
    }
}
