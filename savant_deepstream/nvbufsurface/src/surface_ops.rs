//! Platform-aware GPU surface memory operations.
//!
//! Uses [`SurfaceView`] to resolve CUDA device pointers on both dGPU
//! (direct `dataPtr`) and Jetson (EGL-CUDA interop via [`EglCudaMeta`]).

use crate::{ffi, NvBufSurfaceError, SurfaceView};

/// Fill the surface in `view` with a constant byte `value`.
///
/// All bytes up to `pitch * height` are set to `value`.
pub fn memset_surface(view: &SurfaceView, value: u8) -> Result<(), NvBufSurfaceError> {
    let count = view.pitch() as usize * view.height() as usize;
    let ret = unsafe { ffi::cuMemsetD8_v2(view.data_ptr() as u64, value, count) };
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
///   `cuMemsetD32_v2` — no host-to-device transfer.
pub fn fill_surface(view: &SurfaceView, color: &[u8]) -> Result<(), NvBufSurfaceError> {
    let bpp = view.channels() as usize;
    if color.len() != bpp {
        return Err(NvBufSurfaceError::InvalidInput(format!(
            "color length {} does not match surface channel count {}",
            color.len(),
            bpp,
        )));
    }

    let total_bytes = view.pitch() as usize * view.height() as usize;

    match bpp {
        1 => {
            let ret = unsafe { ffi::cuMemsetD8_v2(view.data_ptr() as u64, color[0], total_bytes) };
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
            let ret = unsafe { ffi::cuMemsetD32_v2(view.data_ptr() as u64, value, count) };
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

/// Upload CPU pixel data to the surface in `view`.
///
/// `data` is a tightly-packed row-major pixel buffer of dimensions
/// `width × height × channels` in the surface's color format (e.g. 4
/// bytes/pixel for RGBA).  Row-by-row copies respect the destination's
/// GPU pitch.
///
/// `channels` is the number of interleaved colour channels per pixel in
/// `data`.  It **must** match the surface's colour format; a mismatch is
/// rejected with [`NvBufSurfaceError::InvalidInput`].
pub fn upload_to_surface(
    view: &SurfaceView,
    data: &[u8],
    width: u32,
    height: u32,
    channels: u32,
) -> Result<(), NvBufSurfaceError> {
    let surf_w = view.width();
    let surf_h = view.height();
    if width > surf_w || height > surf_h {
        return Err(NvBufSurfaceError::InvalidInput(format!(
            "array dimensions {}x{} exceed surface dimensions {}x{}",
            width, height, surf_w, surf_h
        )));
    }

    let bpp = view.channels();
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

    let ret = unsafe {
        ffi::cudaMemcpy2D(
            view.data_ptr(),
            view.pitch() as usize,
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
