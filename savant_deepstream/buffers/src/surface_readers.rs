use crate::ffi;
use crate::NvBufSurfaceError;
use gstreamer as gst;

/// Read `(num_filled, batch_size)` from the NvBufSurface header in a GStreamer buffer.
pub fn read_surface_header(buffer: &gst::Buffer) -> Result<(u32, u32), NvBufSurfaceError> {
    let map = buffer
        .map_readable()
        .map_err(|e| NvBufSurfaceError::InvalidInput(format!("map_readable failed: {e:?}")))?;
    let data = map.as_slice();

    let surface_size = std::mem::size_of::<ffi::NvBufSurface>();
    if data.len() < surface_size {
        return Err(NvBufSurfaceError::InvalidInput(format!(
            "buffer too small for NvBufSurface struct (need {} bytes, have {})",
            surface_size,
            data.len()
        )));
    }

    let surf = unsafe { &*(data.as_ptr() as *const ffi::NvBufSurface) };
    Ok((surf.numFilled, surf.batchSize))
}

/// Read per-slot `(width, height)` from `NvBufSurface.surfaceList`.
pub fn read_slot_dimensions(
    buffer: &gst::Buffer,
    num_filled: u32,
) -> Result<Vec<(u32, u32)>, NvBufSurfaceError> {
    let map = buffer
        .map_readable()
        .map_err(|e| NvBufSurfaceError::InvalidInput(format!("map_readable failed: {e:?}")))?;
    let data = map.as_slice();

    let surface_size = std::mem::size_of::<ffi::NvBufSurface>();
    if data.len() < surface_size {
        return Err(NvBufSurfaceError::InvalidInput(format!(
            "buffer too small for NvBufSurface struct (need {} bytes, have {})",
            surface_size,
            data.len()
        )));
    }

    let surf = unsafe { &*(data.as_ptr() as *const ffi::NvBufSurface) };
    if surf.surfaceList.is_null() {
        return Err(NvBufSurfaceError::NullPointer(
            "NvBufSurface.surfaceList".into(),
        ));
    }
    let mut dims = Vec::with_capacity(num_filled as usize);
    for i in 0..num_filled {
        let params = unsafe { &*surf.surfaceList.add(i as usize) };
        dims.push((params.width, params.height));
    }
    Ok(dims)
}
