//! Safe Rust wrapper types and implementation for NvBufSurfTransform.
//!
//! Provides aspect-ratio-preserving scaling with configurable padding,
//! interpolation, and compute backend selection. This mirrors the approach
//! used internally by nvinfer for ROI preparation.

use crate::cuda_stream::CudaStream;
use crate::ffi;
use crate::ffi::transform_ffi;
use crate::NvBufSurfaceError;

/// Minimum effective dimension (width or height) after applying
/// [`DstPadding`]. Prevents division-by-zero and degenerate transforms
/// when the padding nearly fills the destination surface.
pub const MIN_EFFECTIVE_DIM: u32 = 16;

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
///
/// Variants whose behaviour differs between GPU (dGPU / x86_64) and VIC
/// (Video Image Compositor / Jetson) carry compound names that reflect
/// both backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Interpolation {
    /// Nearest-neighbor (same on GPU and VIC).
    Nearest,
    /// Bilinear (same on GPU and VIC).
    #[default]
    Bilinear,
    /// GPU: Cubic interpolation. VIC: 5-tap filter.
    GpuCubicVic5Tap,
    /// GPU: Super-sampling. VIC: 10-tap filter.
    GpuSuperVic10Tap,
    /// GPU: Lanczos interpolation. VIC: Smart filter.
    GpuLanczosVicSmart,
    /// GPU: Ignored (no-op). VIC: Nicest quality.
    GpuIgnoredVicNicest,
    /// Platform default (GPU: Nearest, VIC: Nearest).
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
            Interpolation::GpuCubicVic5Tap => {
                transform_ffi::NvBufSurfTransform_Inter_NvBufSurfTransformInter_Algo1
            }
            Interpolation::GpuSuperVic10Tap => {
                transform_ffi::NvBufSurfTransform_Inter_NvBufSurfTransformInter_Algo2
            }
            Interpolation::GpuLanczosVicSmart => {
                transform_ffi::NvBufSurfTransform_Inter_NvBufSurfTransformInter_Algo3
            }
            Interpolation::GpuIgnoredVicNicest => {
                transform_ffi::NvBufSurfTransform_Inter_NvBufSurfTransformInter_Algo4
            }
            Interpolation::Default => {
                transform_ffi::NvBufSurfTransform_Inter_NvBufSurfTransformInter_Default
            }
        }
    }
}

impl std::str::FromStr for Interpolation {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().replace('_', "").as_str() {
            "nearest" => Ok(Interpolation::Nearest),
            "bilinear" => Ok(Interpolation::Bilinear),
            "gpucubicvic5tap" | "cubic" | "algo1" => Ok(Interpolation::GpuCubicVic5Tap),
            "gpusupervic10tap" | "super" | "algo2" => Ok(Interpolation::GpuSuperVic10Tap),
            "gpulanczosvicsmart" | "lanczos" | "algo3" => Ok(Interpolation::GpuLanczosVicSmart),
            "gpuignoredvicnicest" | "nicest" | "algo4" => Ok(Interpolation::GpuIgnoredVicNicest),
            "default" => Ok(Interpolation::Default),
            _ => Err(format!("unknown interpolation: '{s}'")),
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

/// Optional per-side destination padding for letterboxing.
///
/// When specified in [`TransformConfig::dst_padding`], reduces the effective
/// destination area before the letterbox rect is computed. The scaled image
/// is placed within the inset region; padding areas are filled with black.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DstPadding {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
}

impl DstPadding {
    /// Create destination padding with per-side values.
    ///
    /// # Examples
    ///
    /// ```
    /// use deepstream_buffers::DstPadding;
    /// let p = DstPadding::new(10, 20, 10, 20);
    /// assert_eq!(p.left, 10);
    /// assert_eq!(p.top, 20);
    /// assert_eq!(p.right, 10);
    /// assert_eq!(p.bottom, 20);
    /// ```
    pub fn new(left: u32, top: u32, right: u32, bottom: u32) -> Self {
        Self {
            left,
            top,
            right,
            bottom,
        }
    }

    /// Create destination padding with equal values on all sides.
    ///
    /// # Examples
    ///
    /// ```
    /// use deepstream_buffers::DstPadding;
    /// let p = DstPadding::uniform(15);
    /// assert_eq!(p, DstPadding::new(15, 15, 15, 15));
    /// ```
    pub fn uniform(value: u32) -> Self {
        Self {
            left: value,
            top: value,
            right: value,
            bottom: value,
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
    /// Create a new rectangle from pixel coordinates.
    ///
    /// # Examples
    ///
    /// ```
    /// use deepstream_buffers::Rect;
    /// let r = Rect::new(10, 20, 640, 480);
    /// assert_eq!(r.left, 10);
    /// assert_eq!(r.top, 20);
    /// assert_eq!(r.width, 640);
    /// assert_eq!(r.height, 480);
    /// ```
    pub fn new(left: u32, top: u32, width: u32, height: u32) -> Self {
        Self {
            top,
            left,
            width,
            height,
        }
    }

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
#[derive(Debug, Clone, Default)]
pub struct TransformConfig {
    /// Padding mode for letterboxing.
    pub padding: Padding,
    /// Optional per-side destination padding. When set, constrains the area
    /// available for the scaled image; padding regions are filled with black.
    pub dst_padding: Option<DstPadding>,
    /// Interpolation method.
    pub interpolation: Interpolation,
    /// Compute backend.
    pub compute_mode: ComputeMode,
    /// Optional CUDA stream for the transform operation.
    ///
    /// When default (null), the legacy default stream (stream 0) is used,
    /// which has implicit synchronization semantics with all other blocking
    /// streams. Set to a non-blocking stream created via
    /// [`CudaStream::new_non_blocking()`] to avoid global GPU serialization.
    pub cuda_stream: CudaStream,
}

impl TransformConfig {
    /// Create a new builder starting from default values.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use deepstream_buffers::{TransformConfig, Interpolation, ComputeMode};
    /// let config = TransformConfig::builder()
    ///     .interpolation(Interpolation::GpuCubicVic5Tap)
    ///     .compute_mode(ComputeMode::Gpu)
    ///     .build();
    /// ```
    pub fn builder() -> TransformConfigBuilder {
        TransformConfigBuilder(TransformConfig::default())
    }
}

/// Builder for [`TransformConfig`].
pub struct TransformConfigBuilder(TransformConfig);

impl TransformConfigBuilder {
    /// Set the padding mode for letterboxing.
    pub fn padding(mut self, padding: Padding) -> Self {
        self.0.padding = padding;
        self
    }

    /// Set optional per-side destination padding.
    pub fn dst_padding(mut self, dst_padding: DstPadding) -> Self {
        self.0.dst_padding = Some(dst_padding);
        self
    }

    /// Set the interpolation method.
    pub fn interpolation(mut self, interpolation: Interpolation) -> Self {
        self.0.interpolation = interpolation;
        self
    }

    /// Set the compute backend.
    pub fn compute_mode(mut self, compute_mode: ComputeMode) -> Self {
        self.0.compute_mode = compute_mode;
        self
    }

    /// Set the CUDA stream for the transform operation.
    pub fn cuda_stream(mut self, cuda_stream: CudaStream) -> Self {
        self.0.cuda_stream = cuda_stream;
        self
    }

    /// Finish building and return the [`TransformConfig`].
    pub fn build(self) -> TransformConfig {
        self.0
    }
}

/// Extract a `*mut NvBufSurface` from a GstBuffer that was allocated by
/// the DeepStream NvDS buffer pool.
///
/// # Safety
///
/// The buffer must contain a valid NvBufSurface in its first memory block.
/// The returned pointer is valid as long as the buffer is alive and not
/// reallocated.
pub unsafe fn extract_nvbufsurface(
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

/// Extract the GPU ID from an NvBufSurface-backed GStreamer buffer.
///
/// Returns the `gpuId` field of the underlying `NvBufSurface`, which
/// identifies the CUDA device the surface memory was allocated on.
pub fn buffer_gpu_id(buf: &gstreamer::BufferRef) -> Result<u32, TransformError> {
    let surf_ptr = unsafe { extract_nvbufsurface(buf)? };
    let gpu_id = unsafe { (*surf_ptr).gpuId };
    Ok(gpu_id)
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
    #[error("Invalid dst_padding: {0}")]
    InvalidDstPadding(&'static str),
    #[error("CUDA error: {0}")]
    CudaError(i32),
}

/// Compute the destination rectangle for aspect-ratio-preserving letterboxing.
///
/// Given source dimensions and destination dimensions, compute the largest
/// rectangle within the destination that preserves the source aspect ratio,
/// positioned according to `padding`. When `dst_padding` is set, the letterbox
/// is computed within the inset area `(effective_w, effective_h)` and the
/// result is offset by `(pad_left, pad_top)`.
pub(crate) fn compute_letterbox_rect(
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    padding: Padding,
    dst_padding: Option<DstPadding>,
) -> Rect {
    let (eff_w, eff_h, offset_left, offset_top) = match dst_padding {
        Some(p) => (
            dst_w.saturating_sub(p.left).saturating_sub(p.right),
            dst_h.saturating_sub(p.top).saturating_sub(p.bottom),
            p.left,
            p.top,
        ),
        None => (dst_w, dst_h, 0, 0),
    };

    if padding == Padding::None {
        return Rect {
            top: offset_top,
            left: offset_left,
            width: eff_w,
            height: eff_h,
        };
    }

    let src_aspect = src_w as f64 / src_h as f64;
    let dst_aspect = eff_w as f64 / eff_h as f64;

    let (scale_w, scale_h) = if src_aspect > dst_aspect {
        // Source is wider -- fit width, letterbox top/bottom
        let w = eff_w;
        let h = (eff_w as f64 / src_aspect).round() as u32;
        (w, h)
    } else {
        // Source is taller -- fit height, pillarbox left/right
        let h = eff_h;
        let w = (eff_h as f64 * src_aspect).round() as u32;
        (w, h)
    };

    let (top, left) = match padding {
        Padding::RightBottom => (0, 0),
        Padding::Symmetric => {
            let top = (eff_h.saturating_sub(scale_h)) / 2;
            let left = (eff_w.saturating_sub(scale_w)) / 2;
            (top, left)
        }
        Padding::None => unreachable!(),
    };

    Rect {
        top: offset_top + top,
        left: offset_left + left,
        width: scale_w,
        height: scale_h,
    }
}

/// Clear the first surface in `dst_surf` to black (all zeros).
///
/// On dGPU (`x86_64`), uses `cudaMemset2DAsync` on the device pointer.
/// On Jetson (`aarch64`), NVMM memory is not directly addressable by CUDA
/// runtime calls, so the surface is mapped to CPU, zeroed, synced back, and
/// unmapped.
///
/// # Safety
///
/// `dst_surf` must point to a valid NvBufSurface with at least one allocated
/// surface in its `surfaceList`.
unsafe fn clear_surface_black(
    dst_surf: *mut ffi::NvBufSurface,
    dst_h: u32,
    cuda_stream: *mut std::ffi::c_void,
) -> Result<(), TransformError> {
    #[cfg(target_arch = "aarch64")]
    {
        // On Jetson, do_transform's dst_surf is a raw pointer without a
        // GstBuffer, so we cannot attach EglCudaMeta.  Use CPU-staging
        // memset; this path is infrequent (only when letterboxing).
        let _ = (dst_h, cuda_stream);
        clear_surface_black_mapped(dst_surf)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let dst_surface = &*(*dst_surf).surfaceList;
        let data_ptr = dst_surface.dataPtr;
        let pitch = dst_surface.pitch as usize;
        let bpp = crate::surface_view::color_format_channels(dst_surface.colorFormat).unwrap_or(4)
            as usize;
        let row_bytes = dst_surface.width as usize * bpp;

        let ret = ffi::cudaMemset2DAsync(
            data_ptr,
            pitch,
            0,
            row_bytes.min(pitch),
            dst_h as usize,
            cuda_stream,
        );
        if ret != 0 {
            return Err(TransformError::CudaError(ret));
        }
        let ret = ffi::cudaStreamSynchronize(cuda_stream);
        if ret != 0 {
            return Err(TransformError::CudaError(ret));
        }
        Ok(())
    }
}

/// Jetson (aarch64): Map -> zero all planes -> SyncForDevice -> UnMap.
#[cfg(target_arch = "aarch64")]
unsafe fn clear_surface_black_mapped(
    dst_surf: *mut ffi::NvBufSurface,
) -> Result<(), TransformError> {
    let ret = ffi::NvBufSurfaceMap(
        dst_surf,
        0,
        -1,
        ffi::NvBufSurfaceMemMapFlags_NVBUF_MAP_READ_WRITE,
    );
    if ret != 0 {
        return Err(TransformError::CudaError(ret));
    }

    let params = &*(*dst_surf).surfaceList;
    let num_planes = params.planeParams.num_planes as usize;
    for plane in 0..num_planes {
        let mapped = params.mappedAddr.addr[plane] as *mut u8;
        if mapped.is_null() {
            let _ = ffi::NvBufSurfaceUnMap(dst_surf, 0, -1);
            return Err(TransformError::InvalidBuffer(
                "mapped address is null after NvBufSurfaceMap",
            ));
        }
        let pitch = params.planeParams.pitch[plane] as usize;
        let h = params.planeParams.height[plane] as usize;
        std::ptr::write_bytes(mapped, 0, pitch * h);
    }

    let ret = ffi::NvBufSurfaceSyncForDevice(dst_surf, 0, -1);
    if ret != 0 {
        let _ = ffi::NvBufSurfaceUnMap(dst_surf, 0, -1);
        return Err(TransformError::CudaError(ret));
    }
    let ret = ffi::NvBufSurfaceUnMap(dst_surf, 0, -1);
    if ret != 0 {
        return Err(TransformError::CudaError(ret));
    }
    Ok(())
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
    src_rect: Option<&Rect>,
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
        cuda_stream: config.cuda_stream.as_raw() as *mut transform_ffi::CUstream_st,
    };
    let ret = transform_ffi::NvBufSurfTransformSetSessionParams(&mut session_params);
    if ret != 0 {
        return Err(TransformError::SetSessionFailed(ret));
    }

    // Compute source rect (optional crop or full source)
    let mut src_rect_ffi = match src_rect {
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

    // Validate dst_padding if present
    if let Some(ref p) = config.dst_padding {
        let eff_w = dst_w.saturating_sub(p.left).saturating_sub(p.right);
        let eff_h = dst_h.saturating_sub(p.top).saturating_sub(p.bottom);
        if eff_w < MIN_EFFECTIVE_DIM {
            return Err(TransformError::InvalidDstPadding(
                "effective width after dst_padding must be >= 16 px",
            ));
        }
        if eff_h < MIN_EFFECTIVE_DIM {
            return Err(TransformError::InvalidDstPadding(
                "effective height after dst_padding must be >= 16 px",
            ));
        }
    }

    // Compute destination rect with letterboxing
    let dst_letterbox = compute_letterbox_rect(
        effective_src_w,
        effective_src_h,
        dst_w,
        dst_h,
        config.padding,
        config.dst_padding,
    );

    // If the letterbox doesn't fill the full destination, clear to black first
    let fills_dst = dst_letterbox.left == 0
        && dst_letterbox.top == 0
        && dst_letterbox.width == dst_w
        && dst_letterbox.height == dst_h;
    if !fills_dst {
        clear_surface_black(dst_surf, dst_h, config.cuda_stream.as_raw())?;
    }

    let mut dst_rect_ffi = dst_letterbox.to_ffi();

    // Build transform flags
    let mut flags: u32 = transform_ffi::NvBufSurfTransform_Transform_Flag_NVBUFSURF_TRANSFORM_FILTER
        | transform_ffi::NvBufSurfTransform_Transform_Flag_NVBUFSURF_TRANSFORM_CROP_DST;

    if src_rect.is_some() {
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
    config.cuda_stream.synchronize().map_err(|e| match e {
        NvBufSurfaceError::CudaInitFailed(code) => TransformError::CudaError(code),
        _ => unreachable!("CudaStream::synchronize only returns CudaInitFailed, got {e:?}"),
    })?;

    Ok(())
}

/// Perform a transform into a specific batch slot of a batched NvBufSurface.
///
/// Creates a stack-local NvBufSurface "view" whose `surfaceList` points to
/// `dst_surf.surfaceList[dst_slot]`, allowing the existing [`do_transform`]
/// logic to write into an arbitrary slot without modification.
///
/// # Safety
///
/// - `src_surf` must point to a valid NvBufSurface with at least one filled entry.
/// - `dst_surf` must point to a valid batched NvBufSurface with `batchSize > dst_slot`.
/// - The caller must ensure that `dst_slot < dst_surf.batchSize`.
pub(crate) unsafe fn do_transform_to_slot(
    src_surf: *mut ffi::NvBufSurface,
    dst_surf: *mut ffi::NvBufSurface,
    dst_slot: u32,
    config: &TransformConfig,
    src_rect: Option<&Rect>,
) -> Result<(), TransformError> {
    let dst = &*dst_surf;
    if dst_slot >= dst.batchSize {
        return Err(TransformError::InvalidBuffer("dst_slot exceeds batchSize"));
    }

    let mut view = *dst;
    view.surfaceList = dst.surfaceList.add(dst_slot as usize);
    view.batchSize = 1;
    view.numFilled = 1;

    do_transform(
        src_surf,
        &mut view as *mut ffi::NvBufSurface,
        config,
        src_rect,
    )
}

impl std::str::FromStr for Padding {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" => Ok(Padding::None),
            "right_bottom" | "rightbottom" => Ok(Padding::RightBottom),
            "symmetric" => Ok(Padding::Symmetric),
            _ => Err(format!("unknown padding: '{s}'")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_letterbox_rect_symmetric_no_dst_padding() {
        let r = compute_letterbox_rect(800, 600, 800, 800, Padding::Symmetric, None);
        assert_eq!(r.top, 100);
        assert_eq!(r.left, 0);
        assert_eq!(r.width, 800);
        assert_eq!(r.height, 600);
    }

    #[test]
    fn compute_letterbox_rect_symmetric_with_dst_padding() {
        let dst_pad = DstPadding {
            left: 10,
            top: 20,
            right: 10,
            bottom: 20,
        };
        let r = compute_letterbox_rect(800, 600, 840, 660, Padding::Symmetric, Some(dst_pad));
        // Effective area 820x620, source 800x600 (aspect 1.333) -> fit width, scale_h=615
        // Symmetric: top offset = (620-615)/2 = 2, so top = 20+2 = 22
        assert_eq!(r.top, 22);
        assert_eq!(r.left, 10);
        assert_eq!(r.width, 820);
        assert_eq!(r.height, 615);
    }

    #[test]
    fn compute_letterbox_rect_right_bottom_with_dst_padding() {
        let dst_pad = DstPadding {
            left: 5,
            top: 5,
            right: 5,
            bottom: 5,
        };
        let r = compute_letterbox_rect(1920, 1080, 650, 650, Padding::RightBottom, Some(dst_pad));
        // Effective 640x640, source wider -> fit width, letterbox top/bottom
        // scale_w=640, scale_h=360
        assert_eq!(r.left, 5);
        assert_eq!(r.top, 5);
        assert_eq!(r.width, 640);
        assert_eq!(r.height, 360);
    }

    #[test]
    fn compute_letterbox_rect_none_with_dst_padding() {
        let dst_pad = DstPadding {
            left: 10,
            top: 20,
            right: 10,
            bottom: 20,
        };
        let r = compute_letterbox_rect(800, 600, 820, 640, Padding::None, Some(dst_pad));
        assert_eq!(r.left, 10);
        assert_eq!(r.top, 20);
        assert_eq!(r.width, 800);
        assert_eq!(r.height, 600);
    }

    #[test]
    fn rect_new() {
        let r = Rect::new(10, 20, 640, 480);
        assert_eq!(r.left, 10);
        assert_eq!(r.top, 20);
        assert_eq!(r.width, 640);
        assert_eq!(r.height, 480);
    }

    #[test]
    fn dst_padding_new() {
        let p = DstPadding::new(1, 2, 3, 4);
        assert_eq!(p.left, 1);
        assert_eq!(p.top, 2);
        assert_eq!(p.right, 3);
        assert_eq!(p.bottom, 4);
    }

    #[test]
    fn dst_padding_uniform() {
        let p = DstPadding::uniform(15);
        assert_eq!(p, DstPadding::new(15, 15, 15, 15));
    }

    #[test]
    fn dst_padding_default_is_zero() {
        let p = DstPadding::default();
        assert_eq!(p, DstPadding::uniform(0));
    }

    #[test]
    fn transform_config_builder_default() {
        let config = TransformConfig::builder().build();
        let default = TransformConfig::default();
        assert_eq!(config.padding, default.padding);
        assert_eq!(config.interpolation, default.interpolation);
        assert_eq!(config.compute_mode, default.compute_mode);
        assert!(config.dst_padding.is_none());
    }

    #[test]
    fn transform_config_builder_overrides() {
        let config = TransformConfig::builder()
            .interpolation(Interpolation::GpuCubicVic5Tap)
            .padding(Padding::None)
            .build();
        assert_eq!(config.interpolation, Interpolation::GpuCubicVic5Tap);
        assert_eq!(config.padding, Padding::None);
    }
}
