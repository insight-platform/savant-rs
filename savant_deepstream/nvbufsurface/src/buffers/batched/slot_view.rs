//! Zero-copy extraction of a single-frame view from a batched NvBufSurface.
//!
//! [`extract_slot_view`] creates a lightweight `gst::Buffer` whose
//! `NvBufSurface` descriptor references the GPU memory of a specific slot
//! in a batched buffer.  No GPU copy is performed; the batch buffer is kept
//! alive via `GstParentBufferMeta`.

use crate::{ffi, transform, NvBufSurfaceError, SavantIdMeta};
use gstreamer as gst;

extern "C" {
    fn gst_buffer_add_parent_buffer_meta(
        buffer: *mut gst::ffi::GstBuffer,
        ref_: *mut gst::ffi::GstBuffer,
    ) -> *mut std::ffi::c_void;
}

/// Create a zero-copy single-frame view of one slot in a batched NvBufSurface.
///
/// The returned buffer contains its own `NvBufSurface` descriptor (in system
/// memory) with `batchSize=1, numFilled=1` whose `surfaceList[0]` points at
/// the GPU memory of `batch[slot_index]`.
///
/// PTS, DTS, duration, and offset are copied from the batch buffer.  If the
/// batch carries [`SavantIdMeta`], the ID for `slot_index` is propagated.
///
/// The batch buffer is kept alive via `GstParentBufferMeta` — the view is
/// valid as long as the returned buffer exists.
///
/// # Arguments
///
/// * `batch` - A GstBuffer containing a batched NvBufSurface (`numFilled >= 1`).
/// * `slot_index` - Zero-based slot to extract.
///
/// # Errors
///
/// Returns an error if `slot_index >= numFilled` or the buffer is invalid.
///
/// # Example
///
/// ```rust,no_run
/// use deepstream_nvbufsurface::{
///     extract_slot_view, BatchedNvBufSurfaceGenerator,
///     NvBufSurfaceGenerator, NvBufSurfaceMemType,
///     TransformConfig, VideoFormat,
/// };
///
/// gstreamer::init().unwrap();
///
/// let src_gen = NvBufSurfaceGenerator::new(
///     VideoFormat::RGBA, 1920, 1080, 30, 1, 0, NvBufSurfaceMemType::Default,
/// ).unwrap();
/// let batch_gen = BatchedNvBufSurfaceGenerator::new(
///     VideoFormat::RGBA, 640, 640, 4, 2, 0, NvBufSurfaceMemType::Default,
/// ).unwrap();
///
/// let config = TransformConfig::default();
/// let mut batch = batch_gen.acquire_batched_surface(config).unwrap();
/// let src = src_gen.acquire_surface(Some(42)).unwrap();
/// batch.fill_slot(&src, None, Some(42)).unwrap();
/// let batch_buf = batch.finalize();
///
/// let view = extract_slot_view(&batch_buf, 0).unwrap();
/// // `view` is a single-frame buffer referencing slot 0's GPU memory.
/// ```
pub fn extract_slot_view(
    batch: &gst::Buffer,
    slot_index: u32,
) -> Result<gst::Buffer, NvBufSurfaceError> {
    let batch_surf = unsafe {
        transform::extract_nvbufsurface(batch.as_ref())
            .map_err(|e| NvBufSurfaceError::BufferCopyFailed(e.to_string()))?
    };

    let num_filled = unsafe { (*batch_surf).numFilled };
    if slot_index >= num_filled {
        return Err(NvBufSurfaceError::SlotOutOfBounds {
            index: slot_index,
            max: num_filled,
        });
    }

    let surface_size = std::mem::size_of::<ffi::NvBufSurface>();
    let params_size = std::mem::size_of::<ffi::NvBufSurfaceParams>();
    let total_size = surface_size + params_size;

    let mut view_buf = gst::Buffer::with_size(total_size).map_err(|_| {
        NvBufSurfaceError::BufferAcquisitionFailed(
            "failed to allocate system memory for slot view".into(),
        )
    })?;

    {
        let buf_ref = view_buf.make_mut();
        let mut map = buf_ref.map_writable().map_err(|e| {
            NvBufSurfaceError::BufferAcquisitionFailed(format!("map failed: {:?}", e))
        })?;
        let data = map.as_mut_slice();
        data.fill(0);

        let surf = unsafe { &mut *(data.as_mut_ptr() as *mut ffi::NvBufSurface) };
        let batch_ref = unsafe { &*batch_surf };
        surf.gpuId = batch_ref.gpuId;
        surf.batchSize = 1;
        surf.numFilled = 1;
        surf.memType = batch_ref.memType;
        surf.isContiguous = false;
        surf.surfaceList =
            unsafe { data.as_mut_ptr().add(surface_size) as *mut ffi::NvBufSurfaceParams };

        let src_params = unsafe { &*batch_ref.surfaceList.add(slot_index as usize) };
        let dst_params =
            unsafe { &mut *(data.as_mut_ptr().add(surface_size) as *mut ffi::NvBufSurfaceParams) };
        *dst_params = *src_params;
    }

    // Keep batch buffer alive so the GPU memory referenced by surfaceList[0]
    // remains valid.
    unsafe {
        gst_buffer_add_parent_buffer_meta(
            view_buf.make_mut().as_mut_ptr(),
            batch.as_ptr() as *mut gst::ffi::GstBuffer,
        );
    }

    // Propagate timestamps and offsets from the batch buffer.
    {
        let batch_ref = batch.as_ref();
        let buf_ref = view_buf.make_mut();
        buf_ref.set_pts(batch_ref.pts());
        buf_ref.set_dts(batch_ref.dts());
        buf_ref.set_duration(batch_ref.duration());
        buf_ref.set_offset(batch_ref.offset());
        buf_ref.set_offset_end(batch_ref.offset_end());
    }

    // Propagate the SavantIdMeta for this specific slot.
    if let Some(meta) = batch.meta::<SavantIdMeta>() {
        let ids = meta.ids();
        if let Some(id) = ids.get(slot_index as usize) {
            let buf_ref = view_buf.make_mut();
            SavantIdMeta::replace(buf_ref, vec![id.clone()]);
        }
    }

    Ok(view_buf)
}
