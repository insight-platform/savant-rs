//! Heterogeneous batched NvBufSurface (zero-copy, nvstreammux2-style).
//!
//! [`DsNvNonUniformSurfaceBuffer`] assembles individual NvBufSurface buffers of arbitrary
//! dimensions and formats into a single synthetic batched descriptor — no
//! GPU memory is allocated or copied.

use crate::{ffi, transform, NvBufSurfaceError, SavantIdMeta, SavantIdMetaKind};
use gstreamer as gst;

extern "C" {
    fn gst_buffer_add_parent_buffer_meta(
        buffer: *mut gst::ffi::GstBuffer,
        ref_: *mut gst::ffi::GstBuffer,
    ) -> *mut std::ffi::c_void;
}

/// A zero-copy heterogeneous batch that assembles individual NvBufSurface
/// buffers of arbitrary dimensions and formats into a single synthetic
/// batched NvBufSurface descriptor — the same approach nvstreammux2 uses.
///
/// No GPU memory is allocated or copied. Each source buffer's
/// `NvBufSurfaceParams` is referenced by pointer, and
/// [`GstParentBufferMeta`] keeps the source buffers alive automatically.
///
/// # Example
///
/// ```rust,no_run
/// use deepstream_nvbufsurface::{
///     DsNvNonUniformSurfaceBuffer, DsNvSurfaceBufferGenerator, NvBufSurfaceMemType, VideoFormat,
/// };
///
/// gstreamer::init().unwrap();
///
/// let gen_1080p = DsNvSurfaceBufferGenerator::new(
///     VideoFormat::RGBA, 1920, 1080, 30, 1, 0, NvBufSurfaceMemType::Default,
/// ).unwrap();
/// let gen_720p = DsNvSurfaceBufferGenerator::new(
///     VideoFormat::RGBA, 1280, 720, 30, 1, 0, NvBufSurfaceMemType::Default,
/// ).unwrap();
///
/// let buf_1080p = gen_1080p.acquire_surface(Some(1)).unwrap();
/// let buf_720p = gen_720p.acquire_surface(Some(2)).unwrap();
///
/// let mut batch = DsNvNonUniformSurfaceBuffer::new(8, 0).unwrap();
/// batch.add(&buf_1080p, Some(1)).unwrap();
/// batch.add(&buf_720p, Some(2)).unwrap();
/// batch.finalize().unwrap();
/// let buffer = batch.as_gst_buffer().unwrap();
/// ```
pub struct DsNvNonUniformSurfaceBuffer {
    buffer: gst::Buffer,
    ids: Vec<Option<SavantIdMetaKind>>,
    max_batch_size: u32,
    num_filled: u32,
    gpu_id: u32,
    finalized: bool,
}

impl DsNvNonUniformSurfaceBuffer {
    /// Create a new heterogeneous batch with capacity for `max_batch_size` slots.
    ///
    /// Allocates a single GstBuffer whose system memory contains an
    /// `NvBufSurface` header followed by `max_batch_size`
    /// `NvBufSurfaceParams` entries (contiguous layout).
    pub fn new(max_batch_size: u32, gpu_id: u32) -> Result<Self, NvBufSurfaceError> {
        let surface_size = std::mem::size_of::<ffi::NvBufSurface>();
        let params_size = std::mem::size_of::<ffi::NvBufSurfaceParams>();
        let total_size = surface_size + max_batch_size as usize * params_size;

        let mut buffer = gst::Buffer::with_size(total_size).map_err(|_| {
            NvBufSurfaceError::BufferAcquisitionFailed(
                "failed to allocate system memory buffer".into(),
            )
        })?;

        {
            let buf_ref = buffer.make_mut();
            let mut map = buf_ref.map_writable().map_err(|e| {
                NvBufSurfaceError::BufferAcquisitionFailed(format!("Failed to map buffer: {:?}", e))
            })?;
            let data = map.as_mut_slice();
            data.fill(0);

            let surf = unsafe { &mut *(data.as_mut_ptr() as *mut ffi::NvBufSurface) };
            surf.gpuId = gpu_id;
            surf.batchSize = max_batch_size;
            surf.numFilled = 0;
            surf.surfaceList =
                unsafe { data.as_mut_ptr().add(surface_size) as *mut ffi::NvBufSurfaceParams };
        }

        Ok(Self {
            buffer,
            ids: Vec::with_capacity(max_batch_size as usize),
            max_batch_size,
            num_filled: 0,
            gpu_id,
            finalized: false,
        })
    }

    /// Add a source buffer to the batch (zero-copy).
    ///
    /// Copies the source's `NvBufSurfaceParams` descriptor into the next
    /// available slot and attaches `GstParentBufferMeta` to keep the source
    /// buffer's GPU memory alive.
    ///
    /// # Arguments
    ///
    /// * `src_buf` - Source GstBuffer containing a single-frame NvBufSurface.
    /// * `id` - Optional frame ID. If `None` and the source carries a
    ///   [`SavantIdMeta`], the first `Frame(id)` is auto-propagated.
    pub fn add(&mut self, src_buf: &gst::Buffer, id: Option<i64>) -> Result<(), NvBufSurfaceError> {
        if self.finalized {
            return Err(NvBufSurfaceError::AlreadyFinalized);
        }
        if self.num_filled >= self.max_batch_size {
            return Err(NvBufSurfaceError::BatchOverflow {
                max: self.max_batch_size,
            });
        }

        let src_params = unsafe {
            let src_surf = transform::extract_nvbufsurface(src_buf.as_ref())
                .map_err(|e| NvBufSurfaceError::BufferCopyFailed(e.to_string()))?;
            (*src_surf).surfaceList.read()
        };

        let slot = self.num_filled as usize;
        let surface_size = std::mem::size_of::<ffi::NvBufSurface>();
        let params_size = std::mem::size_of::<ffi::NvBufSurfaceParams>();

        {
            let buf_ref = self.buffer.make_mut();
            let mut map = buf_ref.map_writable().map_err(|e| {
                NvBufSurfaceError::BufferAcquisitionFailed(format!("Failed to map buffer: {:?}", e))
            })?;
            let data = map.as_mut_slice();
            let params_offset = surface_size + slot * params_size;
            let dst_params = unsafe {
                &mut *(data.as_mut_ptr().add(params_offset) as *mut ffi::NvBufSurfaceParams)
            };
            *dst_params = src_params;
        }

        unsafe {
            gst_buffer_add_parent_buffer_meta(
                self.buffer.make_mut().as_mut_ptr(),
                src_buf.as_ptr() as *mut gst::ffi::GstBuffer,
            );
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

    /// Return `(dataPtr, pitch, width, height)` for the given slot.
    ///
    /// Width and height are included because they can differ per slot.
    ///
    /// Available only after [`finalize`](Self::finalize) has been called.
    pub fn slot_ptr(
        &self,
        index: u32,
    ) -> Result<(*mut std::ffi::c_void, u32, u32, u32), NvBufSurfaceError> {
        if !self.finalized {
            return Err(NvBufSurfaceError::NotFinalized);
        }
        if index >= self.num_filled {
            return Err(NvBufSurfaceError::SlotOutOfBounds {
                index,
                max: self.num_filled,
            });
        }

        let surf_ptr = unsafe {
            transform::extract_nvbufsurface(self.buffer.as_ref())
                .map_err(|e| NvBufSurfaceError::BufferCopyFailed(e.to_string()))?
        };
        let params = unsafe { &*(*surf_ptr).surfaceList.add(index as usize) };
        Ok((params.dataPtr, params.pitch, params.width, params.height))
    }

    /// Number of slots filled so far.
    pub fn num_filled(&self) -> u32 {
        self.num_filled
    }

    /// Maximum batch capacity.
    pub fn max_batch_size(&self) -> u32 {
        self.max_batch_size
    }

    /// GPU device ID.
    pub fn gpu_id(&self) -> u32 {
        self.gpu_id
    }

    /// Create a zero-copy single-frame view of one filled slot.
    ///
    /// Delegates to [`extract_slot_view`](super::extract_slot_view).
    /// See that function's documentation for details on lifetime,
    /// timestamps, and ID propagation.
    ///
    /// Available only after [`finalize`](Self::finalize) has been called.
    pub fn extract_slot_view(&self, slot_index: u32) -> Result<gst::Buffer, NvBufSurfaceError> {
        if !self.finalized {
            return Err(NvBufSurfaceError::NotFinalized);
        }
        super::extract_slot_view(&self.buffer, slot_index)
    }

    /// Finalize the batch: set `numFilled` in the NvBufSurface descriptor and
    /// attach [`SavantIdMeta`]. Non-consuming; call
    /// [`as_gst_buffer`](Self::as_gst_buffer) afterward to access the buffer.
    ///
    /// Source buffers are automatically kept alive via `GstParentBufferMeta`.
    pub fn finalize(&mut self) -> Result<(), NvBufSurfaceError> {
        if self.finalized {
            return Err(NvBufSurfaceError::AlreadyFinalized);
        }
        {
            let buf_ref = self.buffer.make_mut();
            let mut map = buf_ref
                .map_writable()
                .map_err(|e| NvBufSurfaceError::BufferAcquisitionFailed(format!("{:?}", e)))?;
            let data = map.as_mut_slice();
            let surf = unsafe { &mut *(data.as_mut_ptr() as *mut ffi::NvBufSurface) };
            surf.numFilled = self.num_filled;
        }

        let ids: Vec<SavantIdMetaKind> = self.ids.drain(..).flatten().collect();
        if !ids.is_empty() {
            let buf_ref = self.buffer.make_mut();
            SavantIdMeta::replace(buf_ref, ids);
        }

        self.finalized = true;
        Ok(())
    }

    /// Whether the batch has been finalized.
    pub fn is_finalized(&self) -> bool {
        self.finalized
    }

    /// Return a reference-counted copy of the underlying `gst::Buffer`.
    ///
    /// The returned buffer shares the same system memory (zero-copy) with
    /// the internal buffer and inherits all `GstParentBufferMeta` entries
    /// that keep source GPU buffers alive.
    ///
    /// Available only after [`finalize`](Self::finalize) has been called.
    pub fn as_gst_buffer(&self) -> Result<gst::Buffer, NvBufSurfaceError> {
        if !self.finalized {
            return Err(NvBufSurfaceError::NotFinalized);
        }
        Ok(self.buffer.clone())
    }
}
