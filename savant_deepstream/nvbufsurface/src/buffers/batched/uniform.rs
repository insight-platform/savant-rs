//! Homogeneous batched NvBufSurface buffer generator.
//!
//! [`BatchedNvBufSurfaceGenerator`] produces buffers whose `surfaceList` is
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
/// Unlike [`NvBufSurfaceGenerator`](crate::NvBufSurfaceGenerator) which creates single-frame
/// buffers (`batchSize=1`), this generator produces buffers whose
/// `surfaceList` is an array of `max_batch_size` independently fillable GPU
/// surfaces — all with the same width, height and format.
///
/// Use [`acquire_batched_surface`](Self::acquire_batched_surface) to obtain
/// a [`BatchedSurface`] wrapper, then fill slots via
/// [`fill_slot`](BatchedSurface::fill_slot) or
/// [`slot_ptr`](BatchedSurface::slot_ptr) and finalize.
///
/// # Example
///
/// ```rust,no_run
/// use deepstream_nvbufsurface::{
///     BatchedNvBufSurfaceGenerator, NvBufSurfaceGenerator,
///     NvBufSurfaceMemType, TransformConfig, VideoFormat,
/// };
///
/// gstreamer::init().unwrap();
///
/// let src_gen = NvBufSurfaceGenerator::new(
///     VideoFormat::RGBA, 1920, 1080, 30, 1, 0, NvBufSurfaceMemType::Default,
/// ).unwrap();
///
/// let batched_gen = BatchedNvBufSurfaceGenerator::new(
///     VideoFormat::RGBA, 640, 640, 4, 2, 0, NvBufSurfaceMemType::Default,
/// ).unwrap();
///
/// let config = TransformConfig::default();
/// let mut batch = batched_gen.acquire_batched_surface(config).unwrap();
///
/// let src = src_gen.acquire_surface(Some(42)).unwrap();
/// batch.fill_slot(&src, None, Some(42)).unwrap();
/// let buffer = batch.finalize();
/// ```
pub struct BatchedNvBufSurfaceGenerator {
    pool: gst::BufferPool,
    format: VideoFormat,
    width: u32,
    height: u32,
    gpu_id: u32,
    max_batch_size: u32,
}

/// Builder for [`BatchedNvBufSurfaceGenerator`] with advanced pool
/// configuration.
///
/// # Example
///
/// ```rust,no_run
/// use deepstream_nvbufsurface::{
///     BatchedNvBufSurfaceGenerator, NvBufSurfaceMemType, VideoFormat,
/// };
///
/// gstreamer::init().unwrap();
///
/// let gen = BatchedNvBufSurfaceGenerator::builder(VideoFormat::RGBA, 640, 640, 4)
///     .pool_size(2)
///     .fps(30, 1)
///     .gpu_id(0)
///     .mem_type(NvBufSurfaceMemType::Default)
///     .build()
///     .unwrap();
/// ```
pub struct BatchedNvBufSurfaceGeneratorBuilder {
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

impl BatchedNvBufSurfaceGeneratorBuilder {
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

    /// Build the [`BatchedNvBufSurfaceGenerator`].
    pub fn build(self) -> Result<BatchedNvBufSurfaceGenerator, NvBufSurfaceError> {
        BatchedNvBufSurfaceGenerator::create_pool(
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

impl BatchedNvBufSurfaceGenerator {
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
    ) -> BatchedNvBufSurfaceGeneratorBuilder {
        BatchedNvBufSurfaceGeneratorBuilder {
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
            "Creating BatchedNvBufSurfaceGenerator ({}x{}, batch={})",
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
            "BatchedNvBufSurfaceGenerator created (batch={}, pool={})",
            max_batch_size, pool_size
        );
        Ok(Self {
            pool,
            format,
            width,
            height,
            gpu_id,
            max_batch_size,
        })
    }

    /// Acquire a [`BatchedSurface`] from the pool, ready for slot filling.
    ///
    /// The returned wrapper owns the underlying GstBuffer and stores the
    /// given [`TransformConfig`] for all subsequent
    /// [`fill_slot`](BatchedSurface::fill_slot) calls.
    ///
    /// The NvBufSurface in the returned buffer has `batchSize=max_batch_size`
    /// and `numFilled=0`.
    pub fn acquire_batched_surface(
        &self,
        config: TransformConfig,
    ) -> Result<BatchedSurface, NvBufSurfaceError> {
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

        Ok(BatchedSurface {
            buffer,
            config,
            ids: Vec::with_capacity(self.max_batch_size as usize),
            max_batch_size: self.max_batch_size,
            num_filled: 0,
        })
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
}

impl Drop for BatchedNvBufSurfaceGenerator {
    fn drop(&mut self) {
        debug!("Destroying BatchedNvBufSurfaceGenerator");
        if let Err(e) = self.pool.set_active(false) {
            log::warn!("Failed to deactivate batched buffer pool on drop: {}", e);
        }
    }
}

// ─── BatchedSurface wrapper ──────────────────────────────────────────────────

/// A pool-allocated batched NvBufSurface with per-slot fill tracking and ID
/// accumulation.
///
/// Obtained from
/// [`BatchedNvBufSurfaceGenerator::acquire_batched_surface`]. Fill slots via
/// [`fill_slot`](Self::fill_slot) or directly via
/// [`slot_ptr`](Self::slot_ptr), then call [`finalize`](Self::finalize) to
/// produce the finished GstBuffer.
pub struct BatchedSurface {
    buffer: gst::Buffer,
    config: TransformConfig,
    ids: Vec<Option<SavantIdMetaKind>>,
    max_batch_size: u32,
    num_filled: u32,
}

impl BatchedSurface {
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
        let surf_ptr = unsafe {
            transform::extract_nvbufsurface(self.buffer.as_ref())
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
        if self.num_filled >= self.max_batch_size {
            return Err(NvBufSurfaceError::BatchOverflow {
                max: self.max_batch_size,
            });
        }

        let src_surf = unsafe {
            transform::extract_nvbufsurface(src_buf.as_ref())
                .map_err(|e| NvBufSurfaceError::BufferCopyFailed(e.to_string()))?
        };
        let dst_surf = unsafe {
            transform::extract_nvbufsurface(self.buffer.as_ref())
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

    /// Finalize the batch: set `numFilled` in the NvBufSurface descriptor and
    /// attach [`SavantIdMeta`] with collected IDs. Consumes `self` and returns
    /// the owned GstBuffer.
    pub fn finalize(mut self) -> gst::Buffer {
        {
            let buf_ref = self.buffer.make_mut();
            let mut map = buf_ref.map_writable().expect("finalize: map failed");
            let data = map.as_mut_slice();
            let surf = unsafe { &mut *(data.as_mut_ptr() as *mut ffi::NvBufSurface) };
            surf.numFilled = self.num_filled;
        }

        let ids: Vec<SavantIdMetaKind> = self.ids.into_iter().flatten().collect();
        if !ids.is_empty() {
            let buf_ref = self.buffer.make_mut();
            SavantIdMeta::replace(buf_ref, ids);
        }

        self.buffer
    }

    /// Access the underlying GstBuffer (read-only).
    pub fn buffer(&self) -> &gst::Buffer {
        &self.buffer
    }

    /// Create a zero-copy single-frame view of one filled slot.
    ///
    /// Delegates to [`extract_slot_view`](super::extract_slot_view).
    /// See that function's documentation for details on lifetime,
    /// timestamps, and ID propagation.
    pub fn extract_slot_view(&self, slot_index: u32) -> Result<gst::Buffer, NvBufSurfaceError> {
        super::extract_slot_view(&self.buffer, slot_index)
    }
}

/// Set `numFilled` on a batched NvBufSurface GstBuffer.
///
/// Low-level helper for callers who fill slots manually via
/// [`BatchedSurface::slot_ptr`] instead of [`BatchedSurface::fill_slot`].
pub fn set_num_filled(buffer: &mut gst::BufferRef, count: u32) -> Result<(), NvBufSurfaceError> {
    let mut map = buffer
        .map_writable()
        .map_err(|e| NvBufSurfaceError::BufferAcquisitionFailed(format!("map failed: {:?}", e)))?;
    let data = map.as_mut_slice();
    if data.len() < std::mem::size_of::<ffi::NvBufSurface>() {
        return Err(NvBufSurfaceError::BufferAcquisitionFailed(
            "Buffer too small for NvBufSurface".into(),
        ));
    }
    let surf = unsafe { &mut *(data.as_mut_ptr() as *mut ffi::NvBufSurface) };
    if count > surf.batchSize {
        return Err(NvBufSurfaceError::BatchOverflow {
            max: surf.batchSize,
        });
    }
    surf.numFilled = count;
    Ok(())
}
