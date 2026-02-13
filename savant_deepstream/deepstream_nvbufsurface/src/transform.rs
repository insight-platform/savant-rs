//! Safe Rust wrapper types and implementation for NvBufSurfTransform.
//!
//! Provides aspect-ratio-preserving scaling with configurable padding,
//! interpolation, and compute backend selection. This mirrors the approach
//! used internally by nvinfer for ROI preparation.

use crate::ffi;
use crate::ffi::transform_ffi;

/// Padding mode for letterboxing when source and destination have different
/// aspect ratios.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Padding {
    /// No padding -- scale to fill the entire destination, possibly
    /// distorting the aspect ratio.
    None,
    /// Padding added to the right and bottom edges only. The image is
    /// placed at the top-left corner of the destination.
    RightBottom,
    /// Symmetric padding on all sides. The image is centered in the
    /// destination.
    #[default]
    Symmetric,
}

/// Interpolation method used during scaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Interpolation {
    Nearest,
    #[default]
    Bilinear,
    /// GPU: Cubic, VIC: 5-tap
    Algo1,
    /// GPU: Super, VIC: 10-tap
    Algo2,
    /// GPU: Lanczos, VIC: Smart
    Algo3,
    /// GPU: Ignored, VIC: Nicest
    Algo4,
    /// Default (GPU: Nearest, VIC: Nearest)
    Default,
}

impl Interpolation {
    pub(crate) fn to_ffi(self) -> transform_ffi::NvBufSurfTransform_Inter {
        match self {
            Interpolation::Nearest => {
                transform_ffi::NvBufSurfTransform_Inter_NvBufSurfTransformInter_Nearest
            }
            Interpolation::Bilinear => {
                transform_ffi::NvBufSurfTransform_Inter_NvBufSurfTransformInter_Bilinear
            }
            Interpolation::Algo1 => {
                transform_ffi::NvBufSurfTransform_Inter_NvBufSurfTransformInter_Algo1
            }
            Interpolation::Algo2 => {
                transform_ffi::NvBufSurfTransform_Inter_NvBufSurfTransformInter_Algo2
            }
            Interpolation::Algo3 => {
                transform_ffi::NvBufSurfTransform_Inter_NvBufSurfTransformInter_Algo3
            }
            Interpolation::Algo4 => {
                transform_ffi::NvBufSurfTransform_Inter_NvBufSurfTransformInter_Algo4
            }
            Interpolation::Default => {
                transform_ffi::NvBufSurfTransform_Inter_NvBufSurfTransformInter_Default
            }
        }
    }

    /// Parse from a string name (e.g. CLI arguments).
    pub fn from_str_name(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "nearest" => Some(Interpolation::Nearest),
            "bilinear" => Some(Interpolation::Bilinear),
            "algo1" | "cubic" => Some(Interpolation::Algo1),
            "algo2" | "super" => Some(Interpolation::Algo2),
            "algo3" | "lanczos" => Some(Interpolation::Algo3),
            "algo4" | "nicest" => Some(Interpolation::Algo4),
            "default" => Some(Interpolation::Default),
            _ => None,
        }
    }
}

/// Compute backend for transform operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComputeMode {
    /// VIC on Jetson, dGPU on x86_64.
    #[default]
    Default,
    /// GPU compute.
    Gpu,
    /// VIC (Jetson only).
    Vic,
}

impl ComputeMode {
    pub(crate) fn to_ffi(self) -> transform_ffi::NvBufSurfTransform_Compute {
        match self {
            ComputeMode::Default => {
                transform_ffi::NvBufSurfTransform_Compute_NvBufSurfTransformCompute_Default
            }
            ComputeMode::Gpu => {
                transform_ffi::NvBufSurfTransform_Compute_NvBufSurfTransformCompute_GPU
            }
            ComputeMode::Vic => {
                transform_ffi::NvBufSurfTransform_Compute_NvBufSurfTransformCompute_VIC
            }
        }
    }
}

/// A rectangle in pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub top: u32,
    pub left: u32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    pub(crate) fn to_ffi(self) -> transform_ffi::NvBufSurfTransformRect {
        transform_ffi::NvBufSurfTransformRect {
            top: self.top,
            left: self.left,
            width: self.width,
            height: self.height,
        }
    }
}

/// Configuration for a transform (scale/letterbox) operation.
#[derive(Debug, Clone)]
pub struct TransformConfig {
    /// Padding mode for letterboxing.
    pub padding: Padding,
    /// Interpolation method.
    pub interpolation: Interpolation,
    /// Optional source crop rectangle. `None` means full source.
    pub src_rect: Option<Rect>,
    /// Compute backend.
    pub compute_mode: ComputeMode,
    /// Optional CUDA stream for the transform operation.
    ///
    /// When `null` (the default), the legacy default stream (stream 0) is used,
    /// which has implicit synchronization semantics with all other blocking
    /// streams. Set to a non-blocking stream created via
    /// [`create_cuda_stream()`](crate::create_cuda_stream) to avoid global GPU
    /// serialization.
    pub cuda_stream: *mut std::ffi::c_void,
}

// Safety: the cuda_stream pointer is only used within the same GPU context
// that created it, and is never dereferenced on the Rust side.
unsafe impl Send for TransformConfig {}
unsafe impl Sync for TransformConfig {}

impl Default for TransformConfig {
    fn default() -> Self {
        Self {
            padding: Padding::default(),
            interpolation: Interpolation::default(),
            src_rect: None,
            compute_mode: ComputeMode::default(),
            cuda_stream: std::ptr::null_mut(),
        }
    }
}

/// Extract a `*mut NvBufSurface` from a GstBuffer that was allocated by
/// the DeepStream NvDS buffer pool.
///
/// # Safety
/// The buffer must contain a valid NvBufSurface in its first memory block.
pub(crate) unsafe fn extract_nvbufsurface(
    buf: &gstreamer::BufferRef,
) -> Result<*mut ffi::NvBufSurface, TransformError> {
    let map = buf
        .map_readable()
        .map_err(|_| TransformError::InvalidBuffer("failed to map buffer for reading"))?;
    let data = map.as_slice();
    if data.len() < std::mem::size_of::<ffi::NvBufSurface>() {
        return Err(TransformError::InvalidBuffer(
            "buffer too small for NvBufSurface",
        ));
    }
    let surf_ptr = data.as_ptr() as *mut ffi::NvBufSurface;
    if surf_ptr.is_null() {
        return Err(TransformError::InvalidBuffer(
            "NvBufSurface pointer is null",
        ));
    }
    Ok(surf_ptr)
}

/// Errors from transform operations.
#[derive(Debug, thiserror::Error)]
pub enum TransformError {
    #[error("NvBufSurfTransform failed with error code {0}")]
    TransformFailed(i32),
    #[error("NvBufSurfTransformSetSessionParams failed with error code {0}")]
    SetSessionFailed(i32),
    #[error("Invalid buffer: {0}")]
    InvalidBuffer(&'static str),
    #[error("CUDA error: {0}")]
    CudaError(i32),
}

/// Compute the destination rectangle for aspect-ratio-preserving letterboxing.
///
/// Given source dimensions and destination dimensions, compute the largest
/// rectangle within the destination that preserves the source aspect ratio,
/// positioned according to `padding`.
pub(crate) fn compute_letterbox_rect(
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    padding: Padding,
) -> Rect {
    if padding == Padding::None {
        return Rect {
            top: 0,
            left: 0,
            width: dst_w,
            height: dst_h,
        };
    }

    let src_aspect = src_w as f64 / src_h as f64;
    let dst_aspect = dst_w as f64 / dst_h as f64;

    let (scale_w, scale_h) = if src_aspect > dst_aspect {
        // Source is wider -- fit width, letterbox top/bottom
        let w = dst_w;
        let h = (dst_w as f64 / src_aspect).round() as u32;
        (w, h)
    } else {
        // Source is taller -- fit height, pillarbox left/right
        let h = dst_h;
        let w = (dst_h as f64 * src_aspect).round() as u32;
        (w, h)
    };

    let (top, left) = match padding {
        Padding::RightBottom => (0, 0),
        Padding::Symmetric => {
            let top = (dst_h.saturating_sub(scale_h)) / 2;
            let left = (dst_w.saturating_sub(scale_w)) / 2;
            (top, left)
        }
        Padding::None => unreachable!(),
    };

    Rect {
        top,
        left,
        width: scale_w,
        height: scale_h,
    }
}

/// Perform the actual NvBufSurfTransform call with optional letterboxing.
///
/// # Safety
/// Both `src_surf` and `dst_surf` must point to valid, allocated NvBufSurface
/// structs with at least one surface in their batch.
pub(crate) unsafe fn do_transform(
    src_surf: *mut ffi::NvBufSurface,
    dst_surf: *mut ffi::NvBufSurface,
    config: &TransformConfig,
) -> Result<(), TransformError> {
    let src = &*src_surf;
    let dst = &*dst_surf;

    let src_w = src.surfaceList.as_ref().unwrap().width;
    let src_h = src.surfaceList.as_ref().unwrap().height;
    let dst_w = dst.surfaceList.as_ref().unwrap().width;
    let dst_h = dst.surfaceList.as_ref().unwrap().height;

    // Set session params (compute mode + GPU ID + optional CUDA stream)
    let gpu_id = src.gpuId as i32;
    let mut session_params = transform_ffi::NvBufSurfTransformConfigParams {
        compute_mode: config.compute_mode.to_ffi(),
        gpu_id,
        cuda_stream: config.cuda_stream as *mut transform_ffi::CUstream_st,
    };
    let ret = transform_ffi::NvBufSurfTransformSetSessionParams(&mut session_params);
    if ret != 0 {
        return Err(TransformError::SetSessionFailed(ret));
    }

    // Compute source rect (optional crop or full source)
    let mut src_rect_ffi = match &config.src_rect {
        Some(r) => r.to_ffi(),
        None => transform_ffi::NvBufSurfTransformRect {
            top: 0,
            left: 0,
            width: src_w,
            height: src_h,
        },
    };

    // Source dimensions for letterbox computation (after crop)
    let effective_src_w = src_rect_ffi.width;
    let effective_src_h = src_rect_ffi.height;

    // Compute destination rect with letterboxing
    let dst_letterbox = compute_letterbox_rect(
        effective_src_w,
        effective_src_h,
        dst_w,
        dst_h,
        config.padding,
    );

    // If using padding, clear the destination surface to black first
    if config.padding != Padding::None
        && (dst_letterbox.width != dst_w || dst_letterbox.height != dst_h)
    {
        let dst_surface = &*dst.surfaceList;
        let data_ptr = dst_surface.dataPtr;
        let pitch = dst_surface.pitch as usize;
        let bpp = dst_surface.width as usize * 4; // Assume RGBA for clearing, width * bytes per pixel

        // Clear the entire destination surface to black (0)
        // Use pitch-based memset for the full surface
        let ret = ffi::cudaMemset2DAsync(
            data_ptr,
            pitch,
            0, // black
            bpp.min(pitch),
            dst_h as usize,
            config.cuda_stream, // use the configured CUDA stream
        );
        if ret != 0 {
            return Err(TransformError::CudaError(ret));
        }
        let ret = ffi::cudaStreamSynchronize(config.cuda_stream);
        if ret != 0 {
            return Err(TransformError::CudaError(ret));
        }
    }

    let mut dst_rect_ffi = dst_letterbox.to_ffi();

    // Build transform flags
    let mut flags: u32 = transform_ffi::NvBufSurfTransform_Transform_Flag_NVBUFSURF_TRANSFORM_FILTER
        | transform_ffi::NvBufSurfTransform_Transform_Flag_NVBUFSURF_TRANSFORM_CROP_DST
        | transform_ffi::NvBufSurfTransform_Transform_Flag_NVBUFSURF_TRANSFORM_CROP_SRC;

    // If user specified a source crop, add the CROP_SRC flag (already included)
    if config.src_rect.is_some() {
        flags |= transform_ffi::NvBufSurfTransform_Transform_Flag_NVBUFSURF_TRANSFORM_CROP_SRC;
    }

    let mut transform_params = transform_ffi::_NvBufSurfaceTransformParams {
        transform_flag: flags,
        transform_flip: 0, // No flip
        transform_filter: config.interpolation.to_ffi(),
        src_rect: &mut src_rect_ffi,
        dst_rect: &mut dst_rect_ffi,
    };

    // Cast from our NvBufSurface to the transform_ffi NvBufSurface.
    // They are the same struct, just generated separately by bindgen.
    let src_as_transform = src_surf as *mut transform_ffi::NvBufSurface;
    let dst_as_transform = dst_surf as *mut transform_ffi::NvBufSurface;

    let ret = transform_ffi::NvBufSurfTransform(
        src_as_transform,
        dst_as_transform,
        &mut transform_params,
    );
    if ret != 0 {
        return Err(TransformError::TransformFailed(ret));
    }

    // Synchronize the CUDA stream to ensure the transform is fully complete
    // before the caller releases the source or destination buffers.
    //
    // This is essential when using a non-blocking CUDA stream: without it,
    // the source buffer could be returned to the pool and reused by the next
    // frame while the GPU is still reading from it, causing stale-data
    // artifacts (e.g. "trembling" bounding boxes).
    //
    // For the legacy default stream (null), this is effectively a no-op
    // because the default stream already has implicit sync semantics.
    let ret = ffi::cudaStreamSynchronize(config.cuda_stream);
    if ret != 0 {
        return Err(TransformError::CudaError(ret));
    }

    Ok(())
}

impl Padding {
    /// Parse from a string name (e.g. CLI arguments).
    pub fn from_str_name(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "none" => Some(Padding::None),
            "right_bottom" | "rightbottom" => Some(Padding::RightBottom),
            "symmetric" => Some(Padding::Symmetric),
            _ => None,
        }
    }
}
