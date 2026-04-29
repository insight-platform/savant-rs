//! Homogeneous batched NvBufSurface buffer generator.
//!
//! [`UniformBatchGenerator`] produces buffers whose `surfaceList` is
//! an array of `max_batch_size` independently fillable GPU surfaces — all
//! with the same width, height and format.

use crate::{
    ffi, set_structure_uint, transform, NvBufSurfaceError, NvBufSurfaceMemType, Rect,
    SavantIdMetaKind, TransformConfig, VideoFormat,
};
use glib::translate::from_glib_full;
use gstreamer as gst;
use gstreamer::prelude::*;
use log::{debug, info};
use std::time::Instant;

/// Params that make `gst_buffer_pool_acquire_buffer` return immediately
/// with `GST_FLOW_EOS` when no buffers are available instead of blocking.
fn dontwait_params() -> gst::BufferPoolAcquireParams {
    gst::BufferPoolAcquireParams::with_flags(gst::BufferPoolAcquireFlags::DONTWAIT)
}

/// Generates GStreamer buffers with **batched** NvBufSurface memory.
///
/// Produces buffers whose `surfaceList` is an array of `max_batch_size`
/// independently fillable GPU surfaces — all with the same width, height
/// and format.
///
/// Use [`acquire`](Self::acquire) for simple acquisition with
/// `numFilled = batchSize`, or [`acquire_batch`](Self::acquire_batch)
/// for slot-by-slot filling with transform support.
///
/// # Example
///
/// ```rust,no_run
/// use deepstream_buffers::{
///     UniformBatchGenerator, NvBufSurfaceMemType, VideoFormat,
///     SavantIdMetaKind, TransformConfig,
/// };
///
/// gstreamer::init().unwrap();
///
/// let gen = UniformBatchGenerator::new(
///     VideoFormat::RGBA, 640, 640, 4, 2, 0, NvBufSurfaceMemType::Default,
/// ).unwrap();
///
/// let ids = vec![SavantIdMetaKind::Frame(1)];
/// let mut batch = gen.acquire_batch(TransformConfig::default(), ids).unwrap();
/// // ... fill slots via batch.transform_slot() ...
/// batch.finalize().unwrap();
/// let shared = batch.shared_buffer();
/// ```
pub struct UniformBatchGenerator {
    pool: gst::BufferPool,
    format: VideoFormat,
    width: u32,
    height: u32,
    gpu_id: u32,
    max_batch_size: u32,
    fps_num: i32,
    fps_den: i32,
    min_buffers: u32,
    max_buffers: u32,
    /// Memory type the pool was configured with (carried alongside the
    /// other shape parameters so we can include it in the lifecycle log
    /// at [`Drop`] time).
    mem_type: NvBufSurfaceMemType,
    /// Wallclock instant the underlying buffer pool was activated.
    /// Used to log pool lifetime at drop, which is the metric we care
    /// about when chasing "pool re-created too quickly" / "pool that
    /// outlived its consumers" lifecycle bugs.
    created_at: Instant,
}

/// Builder for [`UniformBatchGenerator`] with advanced pool
/// configuration.
///
/// # Example
///
/// ```rust,no_run
/// use deepstream_buffers::{
///     UniformBatchGenerator, NvBufSurfaceMemType, VideoFormat,
/// };
///
/// gstreamer::init().unwrap();
///
/// let gen = UniformBatchGenerator::builder(VideoFormat::RGBA, 640, 640, 4)
///     .pool_size(2)
///     .fps(30, 1)
///     .gpu_id(0)
///     .mem_type(NvBufSurfaceMemType::Default)
///     .build()
///     .unwrap();
/// ```
pub struct UniformBatchGeneratorBuilder {
    format: VideoFormat,
    width: u32,
    height: u32,
    max_batch_size: u32,
    fps_num: i32,
    fps_den: i32,
    gpu_id: u32,
    mem_type: NvBufSurfaceMemType,
    min_buffers: u32,
    max_buffers: u32,
}

impl UniformBatchGeneratorBuilder {
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

    /// Set the number of pre-allocated batched buffers in the pool (sets both
    /// `min_buffers` and `max_buffers` to `size`).
    ///
    /// This controls how many batched buffers can be in-flight simultaneously
    /// (e.g. one being filled while another is being consumed by nvinfer).
    /// Default is 2.
    pub fn pool_size(mut self, size: u32) -> Self {
        self.min_buffers = size;
        self.max_buffers = size;
        self
    }

    /// Set the minimum number of pre-allocated buffers in the pool.
    pub fn min_buffers(mut self, n: u32) -> Self {
        self.min_buffers = n;
        self
    }

    /// Set the maximum number of buffers in the pool.
    pub fn max_buffers(mut self, n: u32) -> Self {
        self.max_buffers = n;
        self
    }

    /// Build the [`UniformBatchGenerator`].
    pub fn build(self) -> Result<UniformBatchGenerator, NvBufSurfaceError> {
        UniformBatchGenerator::create_pool(
            self.format,
            self.width,
            self.height,
            self.max_batch_size,
            self.min_buffers,
            self.max_buffers,
            self.fps_num,
            self.fps_den,
            self.gpu_id,
            self.mem_type,
        )
    }
}

impl UniformBatchGenerator {
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
    ) -> UniformBatchGeneratorBuilder {
        UniformBatchGeneratorBuilder {
            format,
            width,
            height,
            max_batch_size,
            fps_num: 30,
            fps_den: 1,
            gpu_id: 0,
            mem_type: NvBufSurfaceMemType::Default,
            min_buffers: 2,
            max_buffers: 2,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn create_pool(
        format: VideoFormat,
        width: u32,
        height: u32,
        max_batch_size: u32,
        min_buffers: u32,
        max_buffers: u32,
        fps_num: i32,
        fps_den: i32,
        gpu_id: u32,
        mem_type: NvBufSurfaceMemType,
    ) -> Result<Self, NvBufSurfaceError> {
        debug!(
            "Creating UniformBatchGenerator ({}x{}, batch={})",
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
                min_buffers,
                max_buffers,
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

        let pool_addr = pool.as_ptr() as usize;
        info!(
            "UniformBatchGenerator created (pool={:#x}, {}x{}, batch={}, min={}, max={}, \
             format={:?}, mem_type={:?}, gpu_id={})",
            pool_addr,
            width,
            height,
            max_batch_size,
            min_buffers,
            max_buffers,
            format,
            mem_type,
            gpu_id
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
            min_buffers,
            max_buffers,
            mem_type,
            created_at: Instant::now(),
        })
    }

    /// Acquire a raw GstBuffer from the pool.
    ///
    /// When `blocking` is `true` (used by [`acquire_batch`](Self::acquire_batch)
    /// and [`acquire`](Self::acquire)), the call waits until a buffer becomes
    /// available.  When `false` (used by the `try_*` variants), it returns
    /// [`NvBufSurfaceError::PoolExhausted`] immediately if no buffer is free.
    fn acquire_raw(&self, blocking: bool) -> Result<gst::Buffer, NvBufSurfaceError> {
        let params = dontwait_params();
        let acquire_params = if blocking { None } else { Some(&params) };

        self.pool.acquire_buffer(acquire_params).map_err(|e| {
            if !blocking && e == gst::FlowError::Eos {
                NvBufSurfaceError::PoolExhausted
            } else {
                NvBufSurfaceError::BufferAcquisitionFailed(format!("{:?}", e))
            }
        })
    }

    /// Wrap a raw buffer into a [`SurfaceBatch`], resetting `numFilled` to 0.
    fn wrap_batch(
        &self,
        mut buffer: gst::Buffer,
        config: TransformConfig,
        ids: Vec<SavantIdMetaKind>,
    ) -> Result<SurfaceBatch, NvBufSurfaceError> {
        {
            let buf_ref = buffer.make_mut();
            let mut map = buf_ref.map_writable().map_err(|e| {
                NvBufSurfaceError::BufferAcquisitionFailed(format!("Failed to map buffer: {:?}", e))
            })?;
            let data = map.as_mut_slice();
            let surf = unsafe { &mut *(data.as_mut_ptr() as *mut ffi::NvBufSurface) };
            surf.numFilled = 0;
        }

        let shared = crate::SharedBuffer::from(buffer);
        if !ids.is_empty() {
            shared.set_savant_ids(ids);
        }

        Ok(SurfaceBatch {
            buffer: shared,
            config,
            max_batch_size: self.max_batch_size,
            num_filled: 0,
            finalized: false,
        })
    }

    /// Wrap a raw buffer into a `SharedBuffer`, setting `numFilled = batchSize`.
    fn wrap_single(
        &self,
        mut buffer: gst::Buffer,
        id: Option<u128>,
    ) -> Result<crate::SharedBuffer, NvBufSurfaceError> {
        {
            let buf_ref = buffer.make_mut();
            let mut map = buf_ref.map_writable().map_err(|e| {
                NvBufSurfaceError::BufferAcquisitionFailed(format!("Failed to map buffer: {:?}", e))
            })?;
            let data = map.as_mut_slice();
            let surf = unsafe { &mut *(data.as_mut_ptr() as *mut ffi::NvBufSurface) };
            surf.numFilled = self.max_batch_size;
        }

        let shared = crate::SharedBuffer::from(buffer);
        if let Some(frame_id) = id {
            shared.set_savant_ids(vec![SavantIdMetaKind::Frame(frame_id)]);
        }

        Ok(shared)
    }

    /// Acquire a [`SurfaceBatch`] from the pool, ready for slot filling.
    ///
    /// The returned wrapper owns the underlying GstBuffer and stores the
    /// given [`TransformConfig`] and `ids` for use at
    /// [`finalize`](SurfaceBatch::finalize) time.
    ///
    /// The NvBufSurface in the returned buffer has `batchSize=max_batch_size`
    /// and `numFilled=0`.
    ///
    /// **Blocks** until a buffer becomes available when the pool is exhausted.
    /// See [`try_acquire_batch`](Self::try_acquire_batch) for a non-blocking
    /// alternative.
    pub fn acquire_batch(
        &self,
        config: TransformConfig,
        ids: Vec<SavantIdMetaKind>,
    ) -> Result<SurfaceBatch, NvBufSurfaceError> {
        let buffer = self.acquire_raw(true)?;
        self.wrap_batch(buffer, config, ids)
    }

    /// Non-blocking variant of [`acquire_batch`](Self::acquire_batch).
    ///
    /// Returns [`NvBufSurfaceError::PoolExhausted`] immediately when all pool
    /// buffers are currently in use, instead of blocking the caller.
    pub fn try_acquire_batch(
        &self,
        config: TransformConfig,
        ids: Vec<SavantIdMetaKind>,
    ) -> Result<SurfaceBatch, NvBufSurfaceError> {
        let buffer = self.acquire_raw(false)?;
        self.wrap_batch(buffer, config, ids)
    }

    /// Internal: acquire a buffer with `numFilled = batchSize`.
    ///
    /// Used by [`BufferGenerator`](crate::BufferGenerator) (which wraps this
    /// type with `max_batch_size = 1`). External callers should use
    /// [`acquire_batch`](Self::acquire_batch) instead.
    ///
    /// **Blocks** until a buffer becomes available.
    pub(crate) fn acquire(
        &self,
        id: Option<u128>,
    ) -> Result<crate::SharedBuffer, NvBufSurfaceError> {
        let buffer = self.acquire_raw(true)?;
        self.wrap_single(buffer, id)
    }

    /// Non-blocking variant of [`acquire`](Self::acquire).
    ///
    /// Returns [`NvBufSurfaceError::PoolExhausted`] immediately when all pool
    /// buffers are currently in use.
    pub(crate) fn try_acquire(
        &self,
        id: Option<u128>,
    ) -> Result<crate::SharedBuffer, NvBufSurfaceError> {
        let buffer = self.acquire_raw(false)?;
        self.wrap_single(buffer, id)
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
    pub fn nvmm_caps(&self) -> String {
        self.nvmm_caps_gst().to_string()
    }

    pub fn nvmm_caps_gst(&self) -> gst::Caps {
        gst::Caps::builder("video/x-raw")
            .features(["memory:NVMM"])
            .field("format", self.format.gst_name())
            .field("width", self.width as i32)
            .field("height", self.height as i32)
            .field("framerate", gst::Fraction::new(self.fps_num, self.fps_den))
            .build()
    }

    /// Return raw (non-NVMM) caps matching this generator's format/dimensions.
    pub fn raw_caps(&self) -> String {
        self.raw_caps_gst().to_string()
    }

    pub(crate) fn raw_caps_gst(&self) -> gst::Caps {
        gst::Caps::builder("video/x-raw")
            .field("format", self.format.gst_name())
            .field("width", self.width as i32)
            .field("height", self.height as i32)
            .field("framerate", gst::Fraction::new(self.fps_num, self.fps_den))
            .build()
    }
}

impl Drop for UniformBatchGenerator {
    fn drop(&mut self) {
        let pool_addr = self.pool.as_ptr() as usize;
        let lifetime_ms = self.created_at.elapsed().as_millis();

        // Drain any in-flight GPU work that still references buffers from
        // this pool (e.g. queued NvBufSurfTransform / nvinfer preproc /
        // nvtracker kernels) before we tear the pool down.  Without this
        // fence, `set_active(false)` can race with downstream CUDA ops on
        // the per-source teardown path during rapid source-restart and
        // poison the device with `CUDA error: 700`.  Cost: one device
        // sync per pool drop — only at source teardown, never on hot path.
        crate::cuda_device::cuda_device_synchronize(
            "UniformBatchGenerator::drop::pre_deactivate_sync",
            format_args!(
                "pool={:#x}, {}x{}, batch={}, gpu_id={}",
                pool_addr, self.width, self.height, self.max_batch_size, self.gpu_id
            ),
        );

        let deactivate_result = self.pool.set_active(false);
        match &deactivate_result {
            Ok(()) => {
                info!(
                    "UniformBatchGenerator dropped (pool={:#x}, {}x{}, batch={}, min={}, \
                     max={}, format={:?}, mem_type={:?}, gpu_id={}, lifetime_ms={}, \
                     deactivate=ok)",
                    pool_addr,
                    self.width,
                    self.height,
                    self.max_batch_size,
                    self.min_buffers,
                    self.max_buffers,
                    self.format,
                    self.mem_type,
                    self.gpu_id,
                    lifetime_ms
                );
            }
            Err(e) => {
                log::warn!(
                    "UniformBatchGenerator dropped (pool={:#x}, {}x{}, batch={}, min={}, \
                     max={}, format={:?}, mem_type={:?}, gpu_id={}, lifetime_ms={}, \
                     deactivate_err={})",
                    pool_addr,
                    self.width,
                    self.height,
                    self.max_batch_size,
                    self.min_buffers,
                    self.max_buffers,
                    self.format,
                    self.mem_type,
                    self.gpu_id,
                    lifetime_ms,
                    e,
                );
            }
        }
    }
}

// ─── SurfaceBatch wrapper ──────────────────────────────────────────────────

/// A pool-allocated batched NvBufSurface with per-slot transform support.
///
/// Obtained from [`UniformBatchGenerator::acquire_batch`].  Fill slots with
/// [`transform_slot`](Self::transform_slot), then call
/// [`finalize`](Self::finalize) to set `numFilled`.
///
/// `SavantIdMeta` is attached at acquisition time (in `acquire_batch`),
/// not deferred to finalize.
pub struct SurfaceBatch {
    buffer: crate::SharedBuffer,
    config: TransformConfig,
    max_batch_size: u32,
    /// One past the highest slot index written by `transform_slot`.
    num_filled: u32,
    finalized: bool,
}

impl SurfaceBatch {
    /// Transform a source surface into a specific batch slot.
    ///
    /// Uses the [`TransformConfig`] stored at acquisition time.
    ///
    /// # Arguments
    ///
    /// * `slot` - Destination slot index in the batch (`0..max_batch_size`).
    /// * `src` - Source [`SurfaceView`](crate::SurfaceView) identifying the
    ///   frame to transform.  Works with any color format including multi-plane
    ///   (NV12, I420, etc.) — the GPU transform handles conversion.
    /// * `src_rect` - Optional crop rectangle on the source. `None` = full frame.
    pub fn transform_slot(
        &mut self,
        slot: u32,
        src: &crate::SurfaceView,
        src_rect: Option<&Rect>,
    ) -> Result<(), NvBufSurfaceError> {
        if self.finalized {
            return Err(NvBufSurfaceError::AlreadyFinalized);
        }
        if slot >= self.max_batch_size {
            return Err(NvBufSurfaceError::SlotOutOfBounds {
                index: slot,
                max: self.max_batch_size,
            });
        }

        let src_guard = src.gst_buffer();
        let src_surf = unsafe {
            transform::extract_nvbufsurface(src_guard.as_ref())
                .map_err(|e| NvBufSurfaceError::BufferCopyFailed(e.to_string()))?
        };
        let guard = self.buffer.lock();
        let dst_surf = unsafe {
            transform::extract_nvbufsurface(guard.as_ref())
                .map_err(|e| NvBufSurfaceError::BufferCopyFailed(e.to_string()))?
        };

        unsafe {
            transform::do_transform_to_slot(src_surf, dst_surf, slot, &self.config, src_rect)
                .map_err(|e| NvBufSurfaceError::BufferCopyFailed(e.to_string()))?;
        }

        self.num_filled = self.num_filled.max(slot + 1);
        Ok(())
    }

    /// Number of filled slots (one past the highest slot written).
    pub fn num_filled(&self) -> u32 {
        self.num_filled
    }

    /// Maximum batch capacity.
    pub fn max_batch_size(&self) -> u32 {
        self.max_batch_size
    }

    /// Create a [`SurfaceView`](crate::SurfaceView) for the given batch slot.
    ///
    /// The view borrows the underlying [`SharedBuffer`](crate::SharedBuffer)
    /// (clones the `Arc` internally).
    pub fn view(&self, index: u32) -> Result<crate::SurfaceView, NvBufSurfaceError> {
        crate::SurfaceView::from_buffer(&self.buffer, index)
    }

    /// Clone of the internal [`SharedBuffer`](crate::SharedBuffer).
    ///
    /// Use this to pass the buffer to downstream consumers (encoder, NvInfer)
    /// without consuming the `SurfaceBatch`.
    ///
    /// Prefer [`into_shared_buffer`](Self::into_shared_buffer) when the batch
    /// is no longer needed — it avoids the extra `Arc` clone.
    pub fn shared_buffer(&self) -> crate::SharedBuffer {
        self.buffer.clone()
    }

    /// Consume the batch and return the inner [`SharedBuffer`](crate::SharedBuffer).
    ///
    /// Unlike [`shared_buffer`](Self::shared_buffer) this moves the buffer
    /// out by value, so the `Arc` refcount stays at 1 — no clone, no
    /// manual `drop(batch)` needed.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use deepstream_buffers::{
    /// #     UniformBatchGenerator, NvBufSurfaceMemType, VideoFormat,
    /// #     SavantIdMetaKind, TransformConfig,
    /// # };
    /// # gstreamer::init().unwrap();
    /// # let gen = UniformBatchGenerator::new(
    /// #     VideoFormat::RGBA, 640, 640, 4, 2, 0, NvBufSurfaceMemType::Default,
    /// # ).unwrap();
    /// let ids = vec![SavantIdMetaKind::Frame(1)];
    /// let mut batch = gen.acquire_batch(TransformConfig::default(), ids).unwrap();
    /// batch.finalize().unwrap();
    /// let shared = batch.into_shared_buffer();
    /// assert_eq!(shared.strong_count(), 1);
    /// ```
    pub fn into_shared_buffer(self) -> crate::SharedBuffer {
        self.buffer
    }

    /// Finalize the batch: set `numFilled` in the NvBufSurface descriptor
    /// (derived from the highest slot written).
    ///
    /// `SavantIdMeta` was already attached at acquisition time.
    pub fn finalize(&mut self) -> Result<(), NvBufSurfaceError> {
        if self.finalized {
            return Err(NvBufSurfaceError::AlreadyFinalized);
        }
        {
            let mut guard = self.buffer.lock();
            let buf_ref = guard.make_mut();
            let mut map = buf_ref
                .map_writable()
                .map_err(|e| NvBufSurfaceError::BufferAcquisitionFailed(format!("{:?}", e)))?;
            let data = map.as_mut_slice();
            let surf = unsafe { &mut *(data.as_mut_ptr() as *mut ffi::NvBufSurface) };
            surf.numFilled = self.num_filled;
        }

        self.finalized = true;
        Ok(())
    }

    /// Whether the batch has been finalized.
    pub fn is_finalized(&self) -> bool {
        self.finalized
    }
}
