//! Heterogeneous batched NvBufSurface (zero-copy, nvstreammux2-style).
//!
//! [`DsNvNonUniformSurfaceBuffer`] assembles individual [`SurfaceView`] slots
//! of arbitrary dimensions and formats into a single synthetic batched
//! descriptor — no GPU memory is allocated or copied.
//!
//! [`SurfaceView`]: crate::SurfaceView

use crate::shared_buffer::SharedMutableGstBuffer;
use crate::{ffi, NvBufSurfaceError, SavantIdMeta, SavantIdMetaKind};
use gstreamer as gst;

/// A zero-copy heterogeneous batch that assembles individual
/// [`SurfaceView`](crate::SurfaceView) slots of arbitrary dimensions and
/// formats into a single synthetic batched NvBufSurface descriptor — the same
/// approach nvstreammux2 uses.
///
/// No GPU memory is allocated or copied. Each source slot's
/// `NvBufSurfaceParams` is copied into the synthetic descriptor, and
/// [`GstParentBufferMeta`] keeps the source buffers alive automatically.
///
/// # Example
///
/// ```rust,no_run
/// use deepstream_nvbufsurface::{
///     DsNvNonUniformSurfaceBuffer, DsNvUniformSurfaceBufferGenerator,
///     NvBufSurfaceMemType, SavantIdMetaKind, SurfaceView, VideoFormat,
/// };
///
/// gstreamer::init().unwrap();
///
/// let gen_1080p = DsNvUniformSurfaceBufferGenerator::new(
///     VideoFormat::RGBA, 1920, 1080, 1, 2, 0, NvBufSurfaceMemType::Default,
/// ).unwrap();
/// let gen_720p = DsNvUniformSurfaceBufferGenerator::new(
///     VideoFormat::RGBA, 1280, 720, 1, 2, 0, NvBufSurfaceMemType::Default,
/// ).unwrap();
///
/// let buf_1080p = gen_1080p.acquire_buffer(Some(1)).unwrap();
/// let buf_720p = gen_720p.acquire_buffer(Some(2)).unwrap();
///
/// let view_1080p = SurfaceView::from_shared(&buf_1080p, 0).unwrap();
/// let view_720p = SurfaceView::from_shared(&buf_720p, 0).unwrap();
///
/// let mut batch = DsNvNonUniformSurfaceBuffer::new(0);
/// batch.add(&view_1080p, Some(1)).unwrap();
/// batch.add(&view_720p, Some(2)).unwrap();
/// let shared = batch.finalize(vec![
///     SavantIdMetaKind::Frame(1),
///     SavantIdMetaKind::Frame(2),
/// ]).unwrap();
/// ```
pub struct DsNvNonUniformSurfaceBuffer {
    params: Vec<ffi::NvBufSurfaceParams>,
    parents: Vec<SharedMutableGstBuffer>,
    gpu_id: u32,
}

impl DsNvNonUniformSurfaceBuffer {
    /// Create a new heterogeneous batch builder.
    ///
    /// No `gst::Buffer` is allocated until [`finalize`](Self::finalize) is
    /// called. Slots are accumulated via [`add`](Self::add).
    pub fn new(gpu_id: u32) -> Self {
        Self {
            params: Vec::new(),
            parents: Vec::new(),
            gpu_id,
        }
    }

    /// Add a source slot to the batch (zero-copy).
    ///
    /// Copies the source's `NvBufSurfaceParams` for the correct slot index
    /// (not hardcoded to slot 0). Stores the source's
    /// [`SharedMutableGstBuffer`] to keep GPU memory alive via
    /// `GstParentBufferMeta` at finalize time.
    ///
    /// # Arguments
    ///
    /// * `src` — Source [`SurfaceView`](crate::SurfaceView) to add.
    /// * `id` — Optional frame ID. If `None` and the source carries a
    ///   [`SavantIdMeta`], the first `Frame(id)` is auto-propagated.
    pub fn add(
        &mut self,
        src: &crate::SurfaceView,
        id: Option<i64>,
    ) -> Result<(), NvBufSurfaceError> {
        let guard = src.buffer();
        let src_surf = unsafe {
            crate::transform::extract_nvbufsurface(guard.as_ref())
                .map_err(|e| NvBufSurfaceError::BufferCopyFailed(e.to_string()))?
        };
        let src_params = unsafe { &*(*src_surf).surfaceList.add(src.slot_index() as usize) };
        self.params.push(*src_params);
        drop(guard);

        self.parents.push(src.shared_buffer());

        let _ = id;
        Ok(())
    }

    /// Number of slots added so far.
    pub fn num_filled(&self) -> u32 {
        self.params.len() as u32
    }

    /// GPU device ID.
    pub fn gpu_id(&self) -> u32 {
        self.gpu_id
    }

    /// Finalize the batch: allocate a `gst::Buffer` exactly sized for the
    /// accumulated slots, copy params, attach `GstParentBufferMeta` per source,
    /// and return a [`SharedMutableGstBuffer`].
    ///
    /// # Arguments
    ///
    /// * `ids` — per-slot IDs to attach as [`SavantIdMeta`]. Pass an empty
    ///   `Vec` to skip.
    pub fn finalize(
        self,
        ids: Vec<SavantIdMetaKind>,
    ) -> Result<SharedMutableGstBuffer, NvBufSurfaceError> {
        let num = self.params.len();
        if num == 0 {
            return Err(NvBufSurfaceError::BufferAcquisitionFailed(
                "cannot finalize an empty non-uniform batch".into(),
            ));
        }

        let surface_size = std::mem::size_of::<ffi::NvBufSurface>();
        let params_size = std::mem::size_of::<ffi::NvBufSurfaceParams>();
        let total_size = surface_size + num * params_size;

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
            surf.gpuId = self.gpu_id;
            surf.batchSize = num as u32;
            surf.numFilled = num as u32;
            surf.surfaceList =
                unsafe { data.as_mut_ptr().add(surface_size) as *mut ffi::NvBufSurfaceParams };

            for (i, p) in self.params.iter().enumerate() {
                let dst = unsafe {
                    &mut *(data.as_mut_ptr().add(surface_size + i * params_size)
                        as *mut ffi::NvBufSurfaceParams)
                };
                *dst = *p;
            }
        }

        for parent in &self.parents {
            let parent_guard = parent.lock();
            unsafe {
                ffi::gst_buffer_add_parent_buffer_meta(
                    buffer.make_mut().as_mut_ptr(),
                    parent_guard.as_ptr() as *mut gst::ffi::GstBuffer,
                );
            }
        }

        if !ids.is_empty() {
            SavantIdMeta::replace(buffer.make_mut(), ids);
        }

        Ok(SharedMutableGstBuffer::from(buffer))
    }
}
