//! Zero-copy view of a single GPU surface with cached parameters.
//!
//! [`SurfaceView`] wraps a [`SharedBuffer`] containing an
//! NvBufSurface descriptor and caches the surface parameters (`dataPtr`,
//! `pitch`, `width`, `height`, `gpuId`, etc.) for fast access.
//!
//! Two construction paths are supported:
//!
//! - [`from_buffer`](SurfaceView::from_buffer) / [`from_buffer`](SurfaceView::from_buffer)
//!   — extract a view from any NvBufSurface-backed buffer (single-frame or batched).
//! - [`from_cuda_ptr`](SurfaceView::from_cuda_ptr) — wrap arbitrary CUDA
//!   device memory with a synthetic NvBufSurface descriptor.

use crate::cuda_stream::CudaStream;
use crate::shared_buffer::SharedBuffer;
use crate::transform::{Rect, TransformConfig};
use crate::{ffi, transform, NvBufSurfaceError};
use gstreamer as gst;
use parking_lot::MutexGuard;
use std::any::Any;

/// Zero-copy view of a single GPU surface with CUDA-addressable pointer.
///
/// Holds a [`SharedBuffer`] (refcounted, lockable) and caches
/// the key surface parameters for direct access without repeated
/// NvBufSurface extraction.
///
/// # Platform behaviour
///
/// - **dGPU**: `data_ptr()` is the NvBufSurface `dataPtr`, which is always
///   CUDA-addressable. No additional setup needed.
/// - **Jetson (aarch64)**: VIC-managed memory is not directly CUDA-addressable.
///   Construction transparently attaches an [`EglCudaMeta`] to the buffer
///   (if not already present), which performs `NvBufSurfaceMapEglImage` +
///   `cuGraphicsEGLRegisterImage` to obtain a permanent CUDA device pointer.
///   The mapping persists for the buffer's lifetime and is cleaned up by the
///   meta's `free` callback (`cuGraphicsUnregisterResource` +
///   `NvBufSurfaceUnMapEglImage`).
///
/// [`EglCudaMeta`]: crate::egl_cuda_meta::EglCudaMeta
///
/// # Batched buffers
///
/// Multiple `SurfaceView`s can share the same underlying `gst::Buffer` by
/// using [`from_buffer`](Self::from_buffer) with different `slot_index`
/// values. On Jetson, all slot registrations are stored in a single
/// multi-slot [`EglCudaMeta`] on the batch buffer — no synthetic buffers
/// are created.
///
/// # Construction
///
/// ```rust,no_run
/// use deepstream_buffers::{
///     BufferGenerator, NvBufSurfaceMemType, SurfaceView, VideoFormat,
/// };
///
/// gstreamer::init().unwrap();
///
/// let gen = BufferGenerator::new(
///     VideoFormat::RGBA, 640, 480, 30, 1, 0, NvBufSurfaceMemType::Default,
/// ).unwrap();
/// let shared = gen.acquire(None).unwrap();
/// let view = SurfaceView::from_buffer(&shared, 0).unwrap();
///
/// assert_eq!(view.width(), 640);
/// assert_eq!(view.height(), 480);
/// assert_eq!(view.channels(), 4);
/// ```
pub struct SurfaceView {
    buffer: SharedBuffer,
    slot_index: u32,
    _keepalive: Option<Box<dyn Any + Send + Sync>>,
    data_ptr: *mut std::ffi::c_void,
    pitch: u32,
    width: u32,
    height: u32,
    gpu_id: u32,
    channels: u32,
    color_format: u32,
    cuda_stream: CudaStream,
}

// SAFETY: SurfaceView is transferred across threads (e.g. via crossbeam
// channel in Picasso) and shared via Arc in PyO3.  The raw `data_ptr`
// points to GPU memory owned by the `buffer` (refcounted GstBuffer) or
// kept alive via `_keepalive`.  SharedBuffer is Send + Sync.
unsafe impl Send for SurfaceView {}
unsafe impl Sync for SurfaceView {}

impl std::fmt::Debug for SurfaceView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SurfaceView")
            .field("slot_index", &self.slot_index)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("pitch", &self.pitch)
            .field("gpu_id", &self.gpu_id)
            .field("channels", &self.channels)
            .field("data_ptr", &self.data_ptr)
            .field("cuda_stream", &self.cuda_stream)
            .finish()
    }
}

/// Map an `NvBufSurfaceColorFormat` raw value to the number of interleaved
/// channels in the primary plane.  Returns `None` for multi-plane formats
/// (NV12, NV21, I420, YUV444, etc.) which are not representable as a
/// single contiguous array.
pub(crate) fn color_format_channels(color_format: u32) -> Option<u32> {
    match color_format {
        1 | 88 => Some(1),
        68 => Some(1),
        10 | 11 => Some(2),
        12 | 13 => Some(2),
        14 | 15 => Some(2),
        16 | 17 => Some(2),
        89..=91 => Some(2),
        27 => Some(3),
        28 => Some(3),
        42 => Some(3),
        43 => Some(3),
        19 => Some(4),
        20 => Some(4),
        21 => Some(4),
        22 => Some(4),
        23 => Some(4),
        24 => Some(4),
        25 => Some(4),
        26 => Some(4),
        _ => None,
    }
}

/// Resolve a CUDA-addressable device pointer for a specific slot.
///
/// On dGPU the `NvBufSurfaceParams::dataPtr` is already a CUDA pointer.
/// On Jetson (aarch64) the memory is VIC-managed, so we go through
/// EGL-CUDA interop via [`EglCudaMeta`] to obtain a usable device pointer.
///
/// Fast path: if the buffer already has an [`EglCudaMeta`] with this slot
/// registered (e.g. from a previous pool cycle with `GST_META_FLAG_POOLED`),
/// the cached pointers are returned in O(1) without calling `make_mut()`.
fn resolve_cuda_ptr(
    buf: &SharedBuffer,
    slot_index: u32,
    params: &ffi::NvBufSurfaceParams,
) -> Result<(*mut std::ffi::c_void, u32), NvBufSurfaceError> {
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (buf, slot_index);
        Ok((params.dataPtr, params.pitch))
    }

    #[cfg(target_arch = "aarch64")]
    {
        let _ = params;
        let mut guard = buf.lock();
        // Fast path: read existing meta without make_mut() (no COW risk).
        if let Some(mapping) = crate::egl_cuda_meta::read_meta(guard.as_ref(), slot_index) {
            return Ok((mapping.cuda_ptr(0), mapping.pitch(0)));
        }
        // Slow path: first access — register and attach meta.
        let buf_ref = guard.make_mut();
        let mapping = unsafe { crate::egl_cuda_meta::ensure_meta(buf_ref, slot_index)? };
        Ok((mapping.cuda_ptr(0), mapping.pitch(0)))
    }
}

impl SurfaceView {
    /// Wrap a plain GStreamer buffer without NvBufSurface validation.
    ///
    /// Surface parameters are zeroed. For stubs, tests, or pipelines without a
    /// real NvBufSurface; most operations on this view will fail until replaced
    /// with a buffer from a [`BufferGenerator`](crate::BufferGenerator).
    pub fn wrap(buf: gst::Buffer) -> Self {
        Self {
            buffer: SharedBuffer::from(buf),
            slot_index: 0,
            _keepalive: None,
            data_ptr: std::ptr::null_mut(),
            pitch: 0,
            width: 0,
            height: 0,
            gpu_id: 0,
            channels: 0,
            color_format: 0,
            cuda_stream: CudaStream::default(),
        }
    }

    /// Create a view from an NvBufSurface-backed GStreamer buffer.
    ///
    /// The buffer is wrapped in a [`SharedBuffer`] internally.
    /// On Jetson, [`EglCudaMeta`] is attached with `GST_META_FLAG_POOLED` so
    /// that it **survives pool recycles**.  Subsequent calls on the same
    /// physical buffer return the cached CUDA pointers in O(1).
    ///
    /// For batched buffers, use [`from_buffer`](Self::from_buffer) to create
    /// multiple views referencing different slots of the same buffer.
    ///
    /// [`EglCudaMeta`]: crate::egl_cuda_meta::EglCudaMeta
    ///
    /// # Arguments
    ///
    /// * `buf` — owned buffer containing an NvBufSurface.
    /// * `slot_index` — zero-based index of the surface to view.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer is not a valid NvBufSurface, if
    /// `slot_index >= numFilled`, or if the surface's color format is
    /// multi-plane (NV12, NV21, I420).
    pub fn from_gst_buffer(buf: gst::Buffer, slot_index: u32) -> Result<Self, NvBufSurfaceError> {
        Self::from_buffer(&SharedBuffer::from(buf), slot_index)
    }

    /// Create a view from a shared buffer with a specific slot index.
    ///
    /// This is the primary constructor for batched buffers: wrap the batch
    /// buffer in a [`SharedBuffer`], then create one `SurfaceView`
    /// per slot using `shared.clone()`.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use deepstream_buffers::{SharedBuffer, SurfaceView};
    /// # fn example(batch_buf: gstreamer::Buffer) {
    /// let shared = SharedBuffer::from(batch_buf);
    /// let view0 = SurfaceView::from_buffer(&shared, 0).unwrap();
    /// let view1 = SurfaceView::from_buffer(&shared, 1).unwrap();
    /// # }
    /// ```
    pub fn from_buffer(buf: &SharedBuffer, slot_index: u32) -> Result<Self, NvBufSurfaceError> {
        let buf = buf.clone();
        let (surf_ptr, gpu_id, num_filled, params_copy) = {
            let guard = buf.lock();
            let surf = unsafe {
                transform::extract_nvbufsurface(guard.as_ref())
                    .map_err(|e| NvBufSurfaceError::BufferCopyFailed(e.to_string()))?
            };
            let surf_ref = unsafe { &*surf };
            let num_filled = surf_ref.numFilled;
            if slot_index >= num_filled {
                return Err(NvBufSurfaceError::SlotOutOfBounds {
                    index: slot_index,
                    max: num_filled,
                });
            }
            let params = unsafe { &*surf_ref.surfaceList.add(slot_index as usize) };
            (surf, surf_ref.gpuId, num_filled, *params)
        };

        let _ = (surf_ptr, num_filled);

        let channels = color_format_channels(params_copy.colorFormat).unwrap_or(0);

        let (data_ptr, pitch) = resolve_cuda_ptr(&buf, slot_index, &params_copy)?;

        Ok(Self {
            data_ptr,
            pitch,
            width: params_copy.width,
            height: params_copy.height,
            gpu_id,
            channels,
            color_format: params_copy.colorFormat,
            buffer: buf,
            slot_index,
            _keepalive: None,
            cuda_stream: CudaStream::default(),
        })
    }

    /// Consume the view and extract the underlying GStreamer buffer.
    ///
    /// Succeeds only when this is the **sole** `SurfaceView` / clone
    /// referencing the shared buffer.  Returns `Err(self)` if other
    /// references exist — drop them first.
    ///
    /// On Jetson (aarch64), this synchronizes CUDA before releasing the
    /// buffer to ensure all writes through the EGL-CUDA mapped pointer
    /// are visible to VIC / NvBufSurfTransform.
    ///
    /// Use this to pass the buffer to APIs that need `gst::Buffer` by value
    /// (e.g. `NvInfer::submit`).
    pub fn into_gst_buffer(self) -> Result<gst::Buffer, Self> {
        self.sync();

        // Prevent `Drop` from running — we handle cleanup manually below.
        let me = std::mem::ManuallyDrop::new(self);

        // SAFETY: reading from `me` which will not be dropped.
        let keepalive = unsafe { std::ptr::read(&me._keepalive) };
        let buffer = unsafe { std::ptr::read(&me.buffer) };
        let data_ptr = me.data_ptr;
        let pitch = me.pitch;
        let width = me.width;
        let height = me.height;
        let gpu_id = me.gpu_id;
        let channels = me.channels;
        let color_format = me.color_format;
        let cuda_stream = me.cuda_stream.clone();
        let slot_index = me.slot_index;

        match buffer.into_buffer() {
            Ok(buf) => Ok(buf),
            Err(shared) => Err(Self {
                buffer: shared,
                slot_index,
                _keepalive: keepalive,
                data_ptr,
                pitch,
                width,
                height,
                gpu_id,
                channels,
                color_format,
                cuda_stream,
            }),
        }
    }

    /// Get a clone of the shared buffer handle.
    ///
    /// Use this to create sibling views for other slots, or to pass the
    /// buffer to an encoder without consuming the view.
    pub fn shared_buffer(&self) -> SharedBuffer {
        self.buffer.clone()
    }

    /// Create a view wrapping arbitrary CUDA device memory.
    ///
    /// A synthetic `gst::Buffer` with an `NvBufSurface` descriptor is created
    /// in system memory, pointing at the given `data_ptr`.  The `keepalive`
    /// object (if any) is stored to prevent the source memory from being
    /// freed while this view exists.
    #[allow(clippy::too_many_arguments)]
    pub fn from_cuda_ptr(
        data_ptr: *mut std::ffi::c_void,
        pitch: u32,
        width: u32,
        height: u32,
        gpu_id: u32,
        channels: u32,
        color_format: u32,
        keepalive: Option<Box<dyn Any + Send + Sync>>,
    ) -> Result<Self, NvBufSurfaceError> {
        if data_ptr.is_null() {
            return Err(NvBufSurfaceError::NullPointer("data_ptr is null".into()));
        }

        let surface_size = std::mem::size_of::<ffi::NvBufSurface>();
        let params_size = std::mem::size_of::<ffi::NvBufSurfaceParams>();
        let total_size = surface_size + params_size;

        let mut buffer = gst::Buffer::with_size(total_size).map_err(|_| {
            NvBufSurfaceError::BufferAcquisitionFailed(
                "failed to allocate system memory for SurfaceView".into(),
            )
        })?;

        {
            let buf_ref = buffer.make_mut();
            let mut map = buf_ref.map_writable().map_err(|e| {
                NvBufSurfaceError::BufferAcquisitionFailed(format!("map failed: {:?}", e))
            })?;
            let data = map.as_mut_slice();
            data.fill(0);

            let surf = unsafe { &mut *(data.as_mut_ptr() as *mut ffi::NvBufSurface) };
            surf.gpuId = gpu_id;
            surf.batchSize = 1;
            surf.numFilled = 1;
            surf.memType = 2; // NVBUF_MEM_CUDA_DEVICE
            surf.isContiguous = false;
            surf.surfaceList =
                unsafe { data.as_mut_ptr().add(surface_size) as *mut ffi::NvBufSurfaceParams };

            let params = unsafe {
                &mut *(data.as_mut_ptr().add(surface_size) as *mut ffi::NvBufSurfaceParams)
            };
            params.width = width;
            params.height = height;
            params.pitch = pitch;
            params.colorFormat = color_format;
            params.dataPtr = data_ptr;
            params.dataSize = pitch * height;
        }

        Ok(Self {
            buffer: SharedBuffer::from(buffer),
            slot_index: 0,
            _keepalive: keepalive,
            data_ptr,
            pitch,
            width,
            height,
            gpu_id,
            channels,
            color_format,
            cuda_stream: CudaStream::default(),
        })
    }

    /// Transform this surface into the destination surface via NvBufSurfTransform.
    ///
    /// Performs a GPU-to-GPU transform (scale/letterbox) from `self` (source) to `dest`
    /// (destination). Does not require [`make_mut`](SharedBuffer::make_mut) —
    /// the operation uses NvBufSurfTransform directly on the underlying NvBufSurface memory.
    ///
    /// # Arguments
    ///
    /// * `dest` — The destination [`SurfaceView`] to write into.
    /// * `config` — Transform configuration (padding, interpolation, etc.). The CUDA stream
    ///   is overridden with `dest.cuda_stream()` so that the transform runs on the destination's
    ///   stream.
    /// * `src_rect` — Optional source crop rectangle. When `None`, the full source surface
    ///   is used.
    ///
    /// # Slot handling
    ///
    /// Correctly handles any slot index — not hardcoded to 0. Both source and destination
    /// may be views into batched buffers at arbitrary slot indices.
    ///
    /// # Replaces
    ///
    /// This method replaces the old `transform_slot` and `transform` APIs.
    pub fn transform_into(
        &self,
        dest: &SurfaceView,
        config: &TransformConfig,
        src_rect: Option<&Rect>,
    ) -> Result<(), NvBufSurfaceError> {
        let src_surf_ptr = {
            let guard = self.buffer.lock();
            unsafe {
                transform::extract_nvbufsurface(guard.as_ref())
                    .map_err(|e| NvBufSurfaceError::BufferCopyFailed(e.to_string()))?
            }
        };
        let dest_surf_ptr = {
            let guard = dest.buffer.lock();
            unsafe {
                transform::extract_nvbufsurface(guard.as_ref())
                    .map_err(|e| NvBufSurfaceError::BufferCopyFailed(e.to_string()))?
            }
        };

        let mut eff_config = config.clone();
        eff_config.cuda_stream = dest.cuda_stream().clone();

        let (mut src_view, mut dst_view) = unsafe {
            let mut src = *src_surf_ptr;
            src.surfaceList = src.surfaceList.add(self.slot_index() as usize);
            src.batchSize = 1;
            src.numFilled = 1;
            let mut dst = *dest_surf_ptr;
            dst.surfaceList = dst.surfaceList.add(dest.slot_index() as usize);
            dst.batchSize = 1;
            dst.numFilled = 1;
            (src, dst)
        };

        unsafe {
            transform::do_transform(
                &mut src_view as *mut ffi::NvBufSurface,
                &mut dst_view as *mut ffi::NvBufSurface,
                &eff_config,
                src_rect,
            )
        }
        .map_err(|e| NvBufSurfaceError::BufferCopyFailed(e.to_string()))
    }

    /// Lock the underlying GStreamer buffer.
    ///
    /// The returned `MutexGuard` auto-derefs to `&gst::Buffer` (via `Deref`)
    /// and `&mut gst::Buffer` (via `DerefMut`).
    pub fn gst_buffer(&self) -> MutexGuard<'_, gst::Buffer> {
        self.buffer.lock()
    }

    /// GPU data pointer to the first pixel of the surface.
    pub fn data_ptr(&self) -> *mut std::ffi::c_void {
        self.data_ptr
    }

    /// Row stride in bytes.
    pub fn pitch(&self) -> u32 {
        self.pitch
    }

    /// Surface width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Surface height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// GPU device ID the surface memory resides on.
    pub fn gpu_id(&self) -> u32 {
        self.gpu_id
    }

    /// Number of interleaved channels per pixel (e.g. 4 for RGBA, 1 for GRAY8).
    ///
    /// Returns `0` for multi-plane formats (NV12, I420, etc.) where the
    /// concept of "channels per pixel" does not apply to a single plane.
    pub fn channels(&self) -> u32 {
        self.channels
    }

    /// Raw `NvBufSurfaceColorFormat` value.
    pub fn color_format(&self) -> u32 {
        self.color_format
    }

    /// The batch slot index this view refers to.
    pub fn slot_index(&self) -> u32 {
        self.slot_index
    }

    /// The CUDA stream this view synchronizes on release.
    ///
    /// Null (default) means the CUDA legacy default stream.
    pub fn cuda_stream(&self) -> &CudaStream {
        &self.cuda_stream
    }

    /// Override the CUDA stream used for synchronization on release.
    ///
    /// Chainable after any constructor:
    ///
    /// ```rust,no_run
    /// # use deepstream_buffers::SurfaceView;
    /// # fn example(view: SurfaceView, stream: deepstream_buffers::CudaStream) {
    /// let view = view.with_cuda_stream(stream);
    /// # }
    /// ```
    pub fn with_cuda_stream(mut self, stream: CudaStream) -> Self {
        self.cuda_stream = stream;
        self
    }

    /// Fill the surface with a constant byte `value`.
    ///
    /// All bytes up to `pitch * height` are set to `value`.
    pub fn memset(&self, value: u8) -> Result<(), NvBufSurfaceError> {
        let count = self.pitch as usize * self.height as usize;
        let ret = unsafe { ffi::cuMemsetD8_v2(self.data_ptr as u64, value, count) };
        if ret != 0 {
            return Err(NvBufSurfaceError::CudaDriverError {
                function: "cuMemsetD8_v2",
                code: ret,
            });
        }
        Ok(())
    }

    /// Fill the surface with a repeating pixel colour, entirely on-GPU.
    ///
    /// `color` must have exactly as many elements as the surface has channels
    /// (e.g. 4 bytes for RGBA, 1 for GRAY8).
    ///
    /// Supported channel counts:
    /// - **1** (GRAY8): uses `cuMemsetD8_v2`.
    /// - **4** (RGBA / BGRx): packs the 4 bytes into a `u32` and uses
    ///   `cuMemsetD32_v2`.
    pub fn fill(&self, color: &[u8]) -> Result<(), NvBufSurfaceError> {
        let bpp = self.channels as usize;
        if color.len() != bpp {
            return Err(NvBufSurfaceError::InvalidInput(format!(
                "color length {} does not match surface channel count {}",
                color.len(),
                bpp,
            )));
        }

        let total_bytes = self.pitch as usize * self.height as usize;

        match bpp {
            1 => {
                let ret =
                    unsafe { ffi::cuMemsetD8_v2(self.data_ptr as u64, color[0], total_bytes) };
                if ret != 0 {
                    return Err(NvBufSurfaceError::CudaDriverError {
                        function: "cuMemsetD8_v2",
                        code: ret,
                    });
                }
            }
            4 => {
                let value = u32::from_le_bytes([color[0], color[1], color[2], color[3]]);
                let count = total_bytes / 4;
                let ret = unsafe { ffi::cuMemsetD32_v2(self.data_ptr as u64, value, count) };
                if ret != 0 {
                    return Err(NvBufSurfaceError::CudaDriverError {
                        function: "cuMemsetD32_v2",
                        code: ret,
                    });
                }
            }
            other => {
                return Err(NvBufSurfaceError::InvalidInput(format!(
                    "fill not supported for {other} channels; only 1 (GRAY8) and 4 (RGBA) are supported",
                )));
            }
        }

        Ok(())
    }

    /// Upload CPU pixel data to this surface.
    ///
    /// `data` is a tightly-packed row-major pixel buffer of dimensions
    /// `width × height × channels` in the surface's color format.
    /// Row-by-row copies respect the destination's GPU pitch.
    pub fn upload(
        &self,
        data: &[u8],
        width: u32,
        height: u32,
        channels: u32,
    ) -> Result<(), NvBufSurfaceError> {
        if width > self.width || height > self.height {
            return Err(NvBufSurfaceError::InvalidInput(format!(
                "array dimensions {}x{} exceed surface dimensions {}x{}",
                width, height, self.width, self.height
            )));
        }

        let bpp = self.channels;
        if channels != bpp {
            return Err(NvBufSurfaceError::InvalidInput(format!(
                "channel count mismatch: array has {} channels but surface expects {}",
                channels, bpp
            )));
        }

        let src_stride = width as usize * bpp as usize;
        let required = src_stride * height as usize;
        if data.len() < required {
            return Err(NvBufSurfaceError::InvalidInput(format!(
                "data too small: need {} bytes ({}x{}x{}), got {}",
                required,
                width,
                height,
                bpp,
                data.len()
            )));
        }

        let ret = unsafe {
            ffi::cudaMemcpy2D(
                self.data_ptr,
                self.pitch as usize,
                data.as_ptr() as *const std::ffi::c_void,
                src_stride,
                src_stride,
                height as usize,
                ffi::CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        if ret != 0 {
            return Err(NvBufSurfaceError::CudaDriverError {
                function: "cudaMemcpy2D",
                code: ret as u32,
            });
        }
        Ok(())
    }

    /// Block until all async CUDA work on this view's stream completes.
    ///
    /// Ensures that transforms, Skia renders, and async copies enqueued on
    /// the view's stream are finished before the buffer is handed to the
    /// next consumer (encoder, another transform, etc.).
    ///
    /// Skipped only when `data_ptr` is null (no GPU backing).
    fn sync(&self) {
        if self.data_ptr.is_null() {
            return;
        }
        self.cuda_stream.synchronize_or_log();
    }
}

impl Drop for SurfaceView {
    fn drop(&mut self) {
        self.sync();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_gpu_ptr() -> *mut std::ffi::c_void {
        0xDEAD_BEEF as *mut std::ffi::c_void
    }

    #[test]
    fn test_from_cuda_ptr_rgba() {
        gst::init().unwrap();
        let view = SurfaceView::from_cuda_ptr(
            fake_gpu_ptr(),
            2560, // pitch
            640,
            480,
            0,
            4,
            19, // RGBA
            None,
        )
        .unwrap();

        assert_eq!(view.width(), 640);
        assert_eq!(view.height(), 480);
        assert_eq!(view.pitch(), 2560);
        assert_eq!(view.channels(), 4);
        assert_eq!(view.gpu_id(), 0);
        assert_eq!(view.color_format(), 19);
        assert_eq!(view.data_ptr(), fake_gpu_ptr());
        assert_eq!(view.slot_index(), 0);
    }

    #[test]
    fn test_from_cuda_ptr_gray8() {
        gst::init().unwrap();
        let view =
            SurfaceView::from_cuda_ptr(fake_gpu_ptr(), 640, 640, 480, 1, 1, 1, None).unwrap();
        assert_eq!(view.channels(), 1);
        assert_eq!(view.color_format(), 1);
    }

    #[test]
    fn test_from_cuda_ptr_null_rejected() {
        gst::init().unwrap();
        let result =
            SurfaceView::from_cuda_ptr(std::ptr::null_mut(), 640, 640, 480, 0, 4, 19, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_cuda_ptr_with_keepalive() {
        gst::init().unwrap();
        let keepalive: Box<dyn std::any::Any + Send + Sync> = Box::new(String::from("alive"));
        let view =
            SurfaceView::from_cuda_ptr(fake_gpu_ptr(), 2560, 640, 480, 0, 4, 19, Some(keepalive))
                .unwrap();
        assert_eq!(view.width(), 640);
    }

    #[test]
    fn test_wrap_plain_buffer() {
        gst::init().unwrap();
        let buf = gst::Buffer::new();
        let view = SurfaceView::wrap(buf);

        assert_eq!(view.width(), 0);
        assert_eq!(view.height(), 0);
        assert_eq!(view.pitch(), 0);
        assert_eq!(view.channels(), 0);
        assert_eq!(view.gpu_id(), 0);
        assert!(view.data_ptr().is_null());
        assert_eq!(view.slot_index(), 0);
    }

    #[test]
    fn test_buffer_accessible() {
        gst::init().unwrap();
        let view =
            SurfaceView::from_cuda_ptr(fake_gpu_ptr(), 2560, 640, 480, 0, 4, 19, None).unwrap();

        {
            let mut guard = view.gst_buffer();
            let buf_ref = guard.make_mut();
            buf_ref.set_pts(gst::ClockTime::from_nseconds(42_000));
        }
        {
            let guard = view.gst_buffer();
            assert_eq!(
                guard.as_ref().pts(),
                Some(gst::ClockTime::from_nseconds(42_000))
            );
        }
    }

    #[test]
    fn test_into_buffer_sole_owner() {
        gst::init().unwrap();
        let view =
            SurfaceView::from_cuda_ptr(fake_gpu_ptr(), 2560, 640, 480, 0, 4, 19, None).unwrap();
        let shared = view.shared_buffer();
        drop(view);
        let _buf = shared.into_buffer().expect("sole owner should succeed");
    }

    #[test]
    fn test_into_buffer_fails_with_sibling() {
        gst::init().unwrap();
        let view =
            SurfaceView::from_cuda_ptr(fake_gpu_ptr(), 2560, 640, 480, 0, 4, 19, None).unwrap();
        let shared = view.shared_buffer();
        let _sibling = shared.clone();
        drop(view);
        assert!(shared.into_buffer().is_err());
    }

    #[test]
    fn test_shared_buffer_clone() {
        gst::init().unwrap();
        let view =
            SurfaceView::from_cuda_ptr(fake_gpu_ptr(), 2560, 640, 480, 0, 4, 19, None).unwrap();
        let shared = view.shared_buffer();
        assert_eq!(shared.strong_count(), 2);
    }

    #[test]
    fn test_color_format_channels() {
        assert_eq!(color_format_channels(1), Some(1));
        assert_eq!(color_format_channels(88), Some(1));
        assert_eq!(color_format_channels(68), Some(1));
        assert_eq!(color_format_channels(19), Some(4));
        assert_eq!(color_format_channels(20), Some(4));
        assert_eq!(color_format_channels(24), Some(4));
        assert_eq!(color_format_channels(27), Some(3));
        assert_eq!(color_format_channels(28), Some(3));
        assert_eq!(color_format_channels(10), Some(2));
        assert_eq!(color_format_channels(14), Some(2));
        assert_eq!(color_format_channels(6), None);
        assert_eq!(color_format_channels(2), None);
        assert_eq!(color_format_channels(0), None);
    }

    #[test]
    fn test_surface_view_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<SurfaceView>();
    }

    #[test]
    fn test_cuda_stream_default() {
        gst::init().unwrap();
        let view =
            SurfaceView::from_cuda_ptr(fake_gpu_ptr(), 2560, 640, 480, 0, 4, 19, None).unwrap();
        assert!(view.cuda_stream().is_default());
    }

    #[test]
    fn test_with_cuda_stream() {
        gst::init().unwrap();
        let stream = CudaStream::new_non_blocking().expect("CUDA stream creation failed");
        let raw = stream.as_raw();
        let view =
            SurfaceView::from_cuda_ptr(fake_gpu_ptr(), 2560, 640, 480, 0, 4, 19, None).unwrap();
        let view = view.with_cuda_stream(stream);
        assert!(!view.cuda_stream().is_default());
        assert_eq!(view.cuda_stream().as_raw(), raw);
    }
}
