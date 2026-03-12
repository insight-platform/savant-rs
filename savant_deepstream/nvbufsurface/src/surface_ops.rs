//! Platform-aware GPU surface memory operations.
//!
//! On dGPU (`NvBufSurfaceMemType::Default` = `CudaDevice`) the CUDA driver
//! API is used directly.  On Jetson (`Default` = `SurfaceArray`), surfaces
//! are mapped to CPU address space via `NvBufSurfaceMap`, modified, synced
//! back with `NvBufSurfaceSyncForDevice`, and unmapped.

use crate::surface_view::color_format_channels;
use crate::{ffi, transform, NvBufSurfaceError};
use gstreamer as gst;

/// Fill the first surface in `buf` with a constant byte `value`.
///
/// The buffer must contain a valid NvBufSurface descriptor.  All bytes
/// up to `pitch * height` are set to `value`.
///
/// # Safety
///
/// The caller must ensure `buf` is a live NvBufSurface-backed GStreamer
/// buffer with at least one filled surface.
pub unsafe fn memset_surface(buf: &gst::Buffer, value: u8) -> Result<(), NvBufSurfaceError> {
    let surf_ptr = extract_surf(buf)?;
    let params = &*(*surf_ptr).surfaceList;

    #[cfg(target_arch = "aarch64")]
    {
        memset_via_map(surf_ptr, params.height, value)
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let count = params.pitch as usize * params.height as usize;
        let ret = ffi::cuMemsetD8_v2(params.dataPtr as u64, value, count);
        if ret != 0 {
            return Err(NvBufSurfaceError::CudaDriverError {
                function: "cuMemsetD8_v2",
                code: ret,
            });
        }
        Ok(())
    }
}

/// Upload CPU pixel data to the first surface in `buf`.
///
/// `data` is a tightly-packed row-major pixel buffer of dimensions
/// `width × height × channels` in the surface's color format (e.g. 4
/// bytes/pixel for RGBA).  Row-by-row copies respect the destination's
/// GPU pitch.
///
/// `channels` is the number of interleaved colour channels per pixel in
/// `data`.  It **must** match the surface's colour format; a mismatch is
/// rejected with [`NvBufSurfaceError::InvalidInput`].
///
/// # Safety
///
/// The caller must ensure `buf` is a live NvBufSurface-backed GStreamer
/// buffer with at least one filled surface whose color format has a
/// known number of interleaved channels.
pub unsafe fn upload_to_surface(
    buf: &gst::Buffer,
    data: &[u8],
    width: u32,
    height: u32,
    channels: u32,
) -> Result<(), NvBufSurfaceError> {
    let surf_ptr = extract_surf(buf)?;
    let params = &*(*surf_ptr).surfaceList;

    let surf_w = params.width;
    let surf_h = params.height;
    if width > surf_w || height > surf_h {
        return Err(NvBufSurfaceError::InvalidInput(format!(
            "array dimensions {}x{} exceed surface dimensions {}x{}",
            width, height, surf_w, surf_h
        )));
    }

    let bpp = color_format_channels(params.colorFormat).ok_or_else(|| {
        NvBufSurfaceError::InvalidInput(format!(
            "unsupported color format {} (multi-plane not supported)",
            params.colorFormat
        ))
    })?;

    if channels != bpp {
        return Err(NvBufSurfaceError::InvalidInput(format!(
            "channel count mismatch: array has {} channels but surface \
             color format expects {}",
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

    #[cfg(target_arch = "aarch64")]
    {
        upload_via_map(surf_ptr, data, src_stride, height)
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let ret = ffi::cudaMemcpy2D(
            params.dataPtr,
            params.pitch as usize,
            data.as_ptr() as *const std::ffi::c_void,
            src_stride,
            src_stride,
            height as usize,
            ffi::CUDA_MEMCPY_HOST_TO_DEVICE,
        );
        if ret != 0 {
            return Err(NvBufSurfaceError::CudaDriverError {
                function: "cudaMemcpy2D",
                code: ret as u32,
            });
        }
        Ok(())
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

unsafe fn extract_surf(buf: &gst::Buffer) -> Result<*mut ffi::NvBufSurface, NvBufSurfaceError> {
    transform::extract_nvbufsurface(buf.as_ref())
        .map_err(|e| NvBufSurfaceError::InvalidInput(e.to_string()))
}

/// Jetson (aarch64): Map → write_bytes → SyncForDevice → UnMap.
#[cfg(target_arch = "aarch64")]
unsafe fn memset_via_map(
    surf_ptr: *mut ffi::NvBufSurface,
    height: u32,
    value: u8,
) -> Result<(), NvBufSurfaceError> {
    let ret = ffi::NvBufSurfaceMap(
        surf_ptr,
        0,
        -1,
        ffi::NvBufSurfaceMemMapFlags_NVBUF_MAP_READ_WRITE,
    );
    if ret != 0 {
        return Err(NvBufSurfaceError::SurfaceMapFailed(ret));
    }

    let params = &*(*surf_ptr).surfaceList;
    let mapped = params.mappedAddr.addr[0] as *mut u8;
    if mapped.is_null() {
        let _ = ffi::NvBufSurfaceUnMap(surf_ptr, 0, -1);
        return Err(NvBufSurfaceError::NullPointer(
            "mappedAddr.addr[0] is null after NvBufSurfaceMap".into(),
        ));
    }
    let mapped_pitch = params.planeParams.pitch[0] as usize;
    let total = mapped_pitch * height as usize;
    std::ptr::write_bytes(mapped, value, total);

    let ret = ffi::NvBufSurfaceSyncForDevice(surf_ptr, 0, -1);
    if ret != 0 {
        let _ = ffi::NvBufSurfaceUnMap(surf_ptr, 0, -1);
        return Err(NvBufSurfaceError::SurfaceSyncFailed(ret));
    }
    let ret = ffi::NvBufSurfaceUnMap(surf_ptr, 0, -1);
    if ret != 0 {
        return Err(NvBufSurfaceError::SurfaceUnmapFailed(ret));
    }
    Ok(())
}

/// Jetson (aarch64): Map → row-by-row copy → SyncForDevice → UnMap.
#[cfg(target_arch = "aarch64")]
unsafe fn upload_via_map(
    surf_ptr: *mut ffi::NvBufSurface,
    data: &[u8],
    src_stride: usize,
    height: u32,
) -> Result<(), NvBufSurfaceError> {
    let ret = ffi::NvBufSurfaceMap(
        surf_ptr,
        0,
        -1,
        ffi::NvBufSurfaceMemMapFlags_NVBUF_MAP_READ_WRITE,
    );
    if ret != 0 {
        return Err(NvBufSurfaceError::SurfaceMapFailed(ret));
    }

    let params = &*(*surf_ptr).surfaceList;
    let mapped = params.mappedAddr.addr[0] as *mut u8;
    if mapped.is_null() {
        let _ = ffi::NvBufSurfaceUnMap(surf_ptr, 0, -1);
        return Err(NvBufSurfaceError::NullPointer(
            "mappedAddr.addr[0] is null after NvBufSurfaceMap".into(),
        ));
    }
    let mapped_pitch = params.planeParams.pitch[0] as usize;

    for row in 0..height as usize {
        let dst = mapped.add(row * mapped_pitch);
        let src = &data[row * src_stride..(row + 1) * src_stride];
        std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src_stride);
    }

    let ret = ffi::NvBufSurfaceSyncForDevice(surf_ptr, 0, -1);
    if ret != 0 {
        let _ = ffi::NvBufSurfaceUnMap(surf_ptr, 0, -1);
        return Err(NvBufSurfaceError::SurfaceSyncFailed(ret));
    }
    let ret = ffi::NvBufSurfaceUnMap(surf_ptr, 0, -1);
    if ret != 0 {
        return Err(NvBufSurfaceError::SurfaceUnmapFailed(ret));
    }
    Ok(())
}
