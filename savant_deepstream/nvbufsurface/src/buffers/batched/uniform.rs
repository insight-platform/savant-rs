//! Homogeneous batched NvBufSurface buffer generator.
//!
//! [`DsNvUniformSurfaceBufferGenerator`] produces buffers whose `surfaceList` is
//! an array of `max_batch_size` independently fillable GPU surfaces — all
//! with the same width, height and format.

use crate::{
    ffi, set_structure_uint, transform, NvBufSurfaceError, NvBufSurfaceMemType, Rect, SavantIdMeta,
    SavantIdMetaKind, TransformConfig, VideoFormat,
};
use glib::translate::from_glib_full;
use gstreamer as gst;
use gstreamer::prelude::*;
use log::debug;

/// Generates GStreamer buffers with **batched** NvBufSurface memory.
///
/// Produces buffers whose `surfaceList` is an array of `max_batch_size`
/// independently fillable GPU surfaces — all with the same width, height
/// and format.
///
/// Use [`acquire_buffer`](Self::acquire_buffer) for simple acquisition with
/// `numFilled = batchSize`, or [`acquire_batched_surface`](Self::acquire_batched_surface)
/// for slot-by-slot filling with transform support.
///
/// # Example
///
/// ```rust,no_run
/// use deepstream_nvbufsurface::{
///     DsNvUniformSurfaceBufferGenerator, NvBufSurfaceMemType, VideoFormat, SurfaceView,
/// };
///
/// gstreamer::init().unwrap();
///
/// let gen = DsNvUniformSurfaceBufferGenerator::new(
///     VideoFormat::RGBA, 640, 640, 4, 2, 0, NvBufSurfaceMemType::Default,
/// ).unwrap();
///
/// let shared = gen.acquire_buffer(Some(42)).unwrap();
/// let view0 = SurfaceView::from_shared(&shared, 0).unwrap();
/// // ... use view0 for GPU operations ...
/// ```
pub struct DsNvUniformSurfaceBufferGenerator {
    pool: gst::BufferPool,
    format: VideoFormat,
    width: u32,
    height: u32,
    gpu_id: u32,
    max_batch_size: u32,
    fps_num: i32,
    fps_den: i32,
}

/// Builder for [`DsNvUniformSurfaceBufferGenerator`] with advanced pool
/// configuration.
///
/// # Example
///
/// ```rust,no_run
/// use deepstream_nvbufsurface::{
///     DsNvUniformSurfaceBufferGenerator, NvBufSurfaceMemType, VideoFormat,
/// };
///
/// gstreamer::init().unwrap();
///
/// let gen = DsNvUniformSurfaceBufferGenerator::builder(VideoFormat::RGBA, 640, 640, 4)
///     .pool_size(2)
///     .fps(30, 1)
///     .gpu_id(0)
///     .mem_type(NvBufSurfaceMemType::Default)
///     .build()
///     .unwrap();
/// ```
pub struct DsNvUniformSurfaceBufferGeneratorBuilder {
    format: VideoFormat,
    width: u32,
    height: u32,
    max_batch_size: u32,
    fps_num: i32,
    fps_den: i32,
    gpu_id: u32,
    mem_type: NvBufSurfaceMemType,
    pool_size: u32,
}

impl DsNvUniformSurfaceBufferGeneratorBuilder {
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

    /// Set the number of pre-allocated batched buffers in the pool.
    ///
    /// This controls how many batched buffers can be in-flight simultaneously
    /// (e.g. one being filled while another is being consumed by nvinfer).
    /// Default is 2.
    pub fn pool_size(mut self, size: u32) -> Self {
        self.pool_size = size;
        self
    }

    /// Build the [`DsNvUniformSurfaceBufferGenerator`].
    pub fn build(self) -> Result<DsNvUniformSurfaceBufferGenerator, NvBufSurfaceError> {
        DsNvUniformSurfaceBufferGenerator::create_pool(
            self.format,
            self.width,
            self.height,
            self.max_batch_size,
            self.pool_size,
            self.fps_num,
            self.fps_den,
            self.gpu_id,
            self.mem_type,
        )
    }
}

impl DsNvUniformSurfaceBufferGenerator {
    /// Create a new batched generator with simple parameters.
    ///
    /// # Arguments
    ///
    /// * `format` - Video pixel format for every slot.
    /// * `width` - Slot width in pixels.
    /// * `height` - Slot height in pixels.
    /// * `max_batch_size` - Maximum number of frames per batch.
    /// * `pool_size` - Number of pre-allocated batched buffers in the pool.
    /// * `gpu_id` - GPU device ID (typically 0).
    /// * `mem_type` - Memory type for NvBufSurface allocation.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        format: VideoFormat,
        width: u32,
        height: u32,
        max_batch_size: u32,
        pool_size: u32,
        gpu_id: u32,
        mem_type: NvBufSurfaceMemType,
    ) -> Result<Self, NvBufSurfaceError> {
        Self::create_pool(
            format,
            width,
            height,
            max_batch_size,
            pool_size,
            30,
            1,
            gpu_id,
            mem_type,
        )
    }

    /// Return a builder for advanced configuration.
    pub fn builder(
        format: VideoFormat,
        width: u32,
        height: u32,
        max_batch_size: u32,
    ) -> DsNvUniformSurfaceBufferGeneratorBuilder {
        DsNvUniformSurfaceBufferGeneratorBuilder {
            format,
            width,
            height,
            max_batch_size,
            fps_num: 30,
            fps_den: 1,
            gpu_id: 0,
            mem_type: NvBufSurfaceMemType::Default,
            pool_size: 2,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn create_pool(
        format: VideoFormat,
        width: u32,
        height: u32,
        max_batch_size: u32,
        pool_size: u32,
        fps_num: i32,
        fps_den: i32,
        gpu_id: u32,
        mem_type: NvBufSurfaceMemType,
    ) -> Result<Self, NvBufSurfaceError> {
        debug!(
            "Creating DsNvUniformSurfaceBufferGenerator ({}x{}, batch={})",
            width, height, max_batch_size
        );

        let format_str = format.gst_name();
        let caps = gst::Caps::builder("video/x-raw")
            .field("format", format_str)
            .field("width", width as i32)
            .field("height", height as i32)
            .field("framerate", gst::Fraction::new(fps_num, fps_den))
            .build();

        let raw_pool = unsafe { ffi::gst_nvds_buffer_pool_new() };
        if raw_pool.is_null() {
            return Err(NvBufSurfaceError::PoolCreationFailed);
        }
        let pool: gst::BufferPool = unsafe { from_glib_full(raw_pool) };

        let nvbufsurface_size = std::mem::size_of::<ffi::NvBufSurface>();
        let mem_type_u32: u32 = mem_type.into();

        unsafe {
            let config = gst::ffi::gst_buffer_pool_get_config(pool.as_ptr());
            if config.is_null() {
                return Err(NvBufSurfaceError::PoolConfigFailed);
            }

            gst::ffi::gst_buffer_pool_config_set_params(
                config,
                caps.as_ptr() as *mut gst::ffi::GstCaps,
                nvbufsurface_size as u32,
                pool_size,
                pool_size,
            );

            set_structure_uint(config, "memtype", mem_type_u32);
            set_structure_uint(config, "gpu-id", gpu_id);
            set_structure_uint(config, "batch-size", max_batch_size);

            let ret = gst::ffi::gst_buffer_pool_set_config(pool.as_ptr(), config);
            if ret == glib::ffi::GFALSE {
                return Err(NvBufSurfaceError::PoolSetConfigFailed(
                    "gst_buffer_pool_set_config returned FALSE".to_string(),
                ));
            }
        }

        pool.set_active(true)
            .map_err(|e| NvBufSurfaceError::PoolActivationFailed(e.to_string()))?;

        debug!(
            "DsNvUniformSurfaceBufferGenerator created (batch={}, pool={})",
            max_batch_size, pool_size
        );
        Ok(Self {
            pool,
            format,
            width,
            height,
            gpu_id,
            max_batch_size,
            fps_num,
            fps_den,
        })
    }

    /// Acquire a [`DsNvUniformSurfaceBuffer`] from the pool, ready for slot filling.
    ///
    /// The returned wrapper owns the underlying GstBuffer and stores the
    /// given [`TransformConfig`] for all subsequent
    /// [`fill_slot`](DsNvUniformSurfaceBuffer::fill_slot) calls.
    ///
    /// The NvBufSurface in the returned buffer has `batchSize=max_batch_size`
    /// and `numFilled=0`.
    pub fn acquire_batched_surface(
        &self,
        config: TransformConfig,
    ) -> Result<DsNvUniformSurfaceBuffer, NvBufSurfaceError> {
        let mut buffer = self
            .pool
            .acquire_buffer(None)
            .map_err(|e| NvBufSurfaceError::BufferAcquisitionFailed(format!("{:?}", e)))?;

        {
            let buf_ref = buffer.make_mut();
            let mut map = buf_ref.map_writable().map_err(|e| {
                NvBufSurfaceError::BufferAcquisitionFailed(format!("Failed to map buffer: {:?}", e))
            })?;
            let data = map.as_mut_slice();
            let surf = unsafe { &mut *(data.as_mut_ptr() as *mut ffi::NvBufSurface) };
            surf.numFilled = 0;
        }

        Ok(DsNvUniformSurfaceBuffer {
            buffer: crate::SharedMutableGstBuffer::from(buffer),
            config,
            ids: Vec::with_capacity(self.max_batch_size as usize),
            max_batch_size: self.max_batch_size,
            num_filled: 0,
            finalized: false,
        })
    }

    /// Acquire a [`SharedMutableGstBuffer`](crate::SharedMutableGstBuffer) from the pool.
    ///
    /// The returned buffer's `NvBufSurface` has `numFilled = batchSize` — all
    /// GPU memory slots are already allocated and usable. Create
    /// [`SurfaceView`](crate::SurfaceView)s via `SurfaceView::from_shared` to
    /// access individual slots.
    ///
    /// If `id` is `Some`, a [`SavantIdMeta`] with `SavantIdMetaKind::Frame(id)`
    /// is attached.
    pub fn acquire_buffer(
        &self,
        id: Option<i64>,
    ) -> Result<crate::SharedMutableGstBuffer, NvBufSurfaceError> {
        let mut buffer = self
            .pool
            .acquire_buffer(None)
            .map_err(|e| NvBufSurfaceError::BufferAcquisitionFailed(format!("{:?}", e)))?;

        {
            let buf_ref = buffer.make_mut();
            let mut map = buf_ref.map_writable().map_err(|e| {
                NvBufSurfaceError::BufferAcquisitionFailed(format!("Failed to map buffer: {:?}", e))
            })?;
            let data = map.as_mut_slice();
            let surf = unsafe { &mut *(data.as_mut_ptr() as *mut ffi::NvBufSurface) };
            surf.numFilled = self.max_batch_size;
        }

        if let Some(frame_id) = id {
            let buf_ref = buffer.make_mut();
            SavantIdMeta::replace(buf_ref, vec![SavantIdMetaKind::Frame(frame_id)]);
        }

        Ok(crate::SharedMutableGstBuffer::from(buffer))
    }

    /// Get the maximum batch size.
    pub fn max_batch_size(&self) -> u32 {
        self.max_batch_size
    }

    /// Get the slot width.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get the slot height.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get the video format.
    pub fn format(&self) -> VideoFormat {
        self.format
    }

    /// Get the GPU device ID.
    pub fn gpu_id(&self) -> u32 {
        self.gpu_id
    }

    /// Return NVMM caps matching this generator's format and dimensions.
    ///
    /// These caps include the `memory:NVMM` feature, suitable for configuring
    /// `appsrc` elements that push into DeepStream-compatible pipelines.
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
    pub fn raw_caps(&self) -> gst::Caps {
        gst::Caps::builder("video/x-raw")
            .field("format", self.format.gst_name())
            .field("width", self.width as i32)
            .field("height", self.height as i32)
            .field("framerate", gst::Fraction::new(self.fps_num, self.fps_den))
            .build()
    }
}

impl Drop for DsNvUniformSurfaceBufferGenerator {
    fn drop(&mut self) {
        debug!("Destroying DsNvUniformSurfaceBufferGenerator");
        if let Err(e) = self.pool.set_active(false) {
            log::warn!("Failed to deactivate batched buffer pool on drop: {}", e);
        }
    }
}

// ─── DsNvUniformSurfaceBuffer wrapper ──────────────────────────────────────────────────

/// A pool-allocated batched NvBufSurface with fill tracking and ID accumulation.
///
/// Obtained from
/// [`DsNvUniformSurfaceBufferGenerator::acquire_batched_surface`]. Access slots
/// via [`slot_view`](Self::slot_view) or [`slot_ptr`](Self::slot_ptr), then
/// call [`finalize`](Self::finalize) to set `numFilled` and attach IDs.
pub struct DsNvUniformSurfaceBuffer {
    buffer: crate::SharedMutableGstBuffer,
    config: TransformConfig,
    ids: Vec<Option<SavantIdMetaKind>>,
    max_batch_size: u32,
    num_filled: u32,
    finalized: bool,
}

impl DsNvUniformSurfaceBuffer {
    /// Return the GPU data pointer and pitch for slot `index`.
    ///
    /// Useful for direct GPU writes (e.g. CUDA memcpy, GpuMat copy).
    pub fn slot_ptr(&self, index: u32) -> Result<(*mut std::ffi::c_void, u32), NvBufSurfaceError> {
        if index >= self.max_batch_size {
            return Err(NvBufSurfaceError::SlotOutOfBounds {
                index,
                max: self.max_batch_size,
            });
        }
        let guard = self.buffer.lock();
        let surf_ptr = unsafe {
            transform::extract_nvbufsurface(guard.as_ref())
                .map_err(|e| NvBufSurfaceError::BufferCopyFailed(e.to_string()))?
        };
        let params = unsafe { &*(*surf_ptr).surfaceList.add(index as usize) };
        Ok((params.dataPtr, params.pitch))
    }

    /// Transform a source buffer into the next available batch slot.
    ///
    /// Uses the [`TransformConfig`] stored at acquisition time. The slot
    /// index is auto-incremented.
    ///
    /// # Arguments
    ///
    /// * `src_buf` - Source GstBuffer containing a single-frame NvBufSurface.
    /// * `src_rect` - Optional crop rectangle on the source. `None` = full frame.
    /// * `id` - Optional frame ID. If `None` and the source buffer carries a
    ///   [`SavantIdMeta`], the first `Frame(id)` is auto-propagated.
    pub fn fill_slot(
        &mut self,
        src_buf: &gst::Buffer,
        src_rect: Option<&Rect>,
        id: Option<i64>,
    ) -> Result<(), NvBufSurfaceError> {
        if self.finalized {
            return Err(NvBufSurfaceError::AlreadyFinalized);
        }
        if self.num_filled >= self.max_batch_size {
            return Err(NvBufSurfaceError::BatchOverflow {
                max: self.max_batch_size,
            });
        }

        let src_surf = unsafe {
            transform::extract_nvbufsurface(src_buf.as_ref())
                .map_err(|e| NvBufSurfaceError::BufferCopyFailed(e.to_string()))?
        };
        let guard = self.buffer.lock();
        let dst_surf = unsafe {
            transform::extract_nvbufsurface(guard.as_ref())
                .map_err(|e| NvBufSurfaceError::BufferCopyFailed(e.to_string()))?
        };

        unsafe {
            transform::do_transform_to_slot(
                src_surf,
                dst_surf,
                self.num_filled,
                &self.config,
                src_rect,
            )
            .map_err(|e| NvBufSurfaceError::BufferCopyFailed(e.to_string()))?;
        }

        let meta_kind = if let Some(frame_id) = id {
            Some(SavantIdMetaKind::Frame(frame_id))
        } else {
            src_buf
                .meta::<SavantIdMeta>()
                .and_then(|meta| meta.ids().first().cloned())
        };
        self.ids.push(meta_kind);
        self.num_filled += 1;

        Ok(())
    }

    /// Number of slots filled so far.
    pub fn num_filled(&self) -> u32 {
        self.num_filled
    }

    /// Maximum batch capacity.
    pub fn max_batch_size(&self) -> u32 {
        self.max_batch_size
    }

    /// Create a [`SurfaceView`](crate::SurfaceView) for the given batch slot.
    ///
    /// The view borrows the underlying [`SharedMutableGstBuffer`](crate::SharedMutableGstBuffer)
    /// (clones the `Arc` internally).
    pub fn slot_view(&self, index: u32) -> Result<crate::SurfaceView, NvBufSurfaceError> {
        crate::SurfaceView::from_shared(&self.buffer, index)
    }

    /// Clone of the internal [`SharedMutableGstBuffer`](crate::SharedMutableGstBuffer).
    ///
    /// Use this to pass the buffer to downstream consumers (encoder, NvInfer)
    /// without consuming the `DsNvUniformSurfaceBuffer`.
    pub fn shared_buffer(&self) -> crate::SharedMutableGstBuffer {
        self.buffer.clone()
    }

    /// Finalize the batch: set `numFilled` in the NvBufSurface descriptor and
    /// attach [`SavantIdMeta`] with the provided IDs.
    ///
    /// # Arguments
    ///
    /// * `num_filled` — number of slots that are actually used.
    /// * `ids` — per-slot IDs to attach as `SavantIdMeta`.
    pub fn finalize(
        &mut self,
        num_filled: u32,
        ids: Vec<SavantIdMetaKind>,
    ) -> Result<(), NvBufSurfaceError> {
        if self.finalized {
            return Err(NvBufSurfaceError::AlreadyFinalized);
        }
        if num_filled > self.max_batch_size {
            return Err(NvBufSurfaceError::BatchOverflow {
                max: self.max_batch_size,
            });
        }
        {
            let mut guard = self.buffer.lock();
            let buf_ref = guard.make_mut();
            let mut map = buf_ref
                .map_writable()
                .map_err(|e| NvBufSurfaceError::BufferAcquisitionFailed(format!("{:?}", e)))?;
            let data = map.as_mut_slice();
            let surf = unsafe { &mut *(data.as_mut_ptr() as *mut ffi::NvBufSurface) };
            surf.numFilled = num_filled;
        }

        if !ids.is_empty() {
            let mut guard = self.buffer.lock();
            let buf_ref = guard.make_mut();
            SavantIdMeta::replace(buf_ref, ids);
        }

        self.num_filled = num_filled;
        self.finalized = true;
        Ok(())
    }

    /// Whether the batch has been finalized.
    pub fn is_finalized(&self) -> bool {
        self.finalized
    }
}
