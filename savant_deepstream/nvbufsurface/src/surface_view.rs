//! Zero-copy view of a single GPU surface with cached parameters.
//!
//! [`SurfaceView`] wraps a refcounted [`gst::Buffer`] containing an
//! NvBufSurface descriptor and caches the surface parameters (`dataPtr`,
//! `pitch`, `width`, `height`, `gpuId`, etc.) for fast access.
//!
//! Two construction paths are supported:
//!
//! - [`from_buffer`](SurfaceView::from_buffer) — extract a view from any
//!   NvBufSurface-backed buffer (single-frame or batched).
//! - [`from_cuda_ptr`](SurfaceView::from_cuda_ptr) — wrap arbitrary CUDA
//!   device memory with a synthetic NvBufSurface descriptor.

use crate::buffers::extract_slot_view;
use crate::{ffi, transform, NvBufSurfaceError};
use gstreamer as gst;
use std::any::Any;

/// Zero-copy view of a single GPU surface.
///
/// Holds an owned `gst::Buffer` (with NvBufSurface descriptor, `batchSize=1,
/// numFilled=1`) and caches the key surface parameters for direct access
/// without repeated NvBufSurface extraction.
///
/// # Construction
///
/// ```rust,no_run
/// use deepstream_nvbufsurface::{
///     NvBufSurfaceGenerator, NvBufSurfaceMemType, SurfaceView, VideoFormat,
/// };
///
/// gstreamer::init().unwrap();
///
/// let gen = NvBufSurfaceGenerator::new(
///     VideoFormat::RGBA, 640, 480, 30, 1, 0, NvBufSurfaceMemType::Default,
/// ).unwrap();
/// let buf = gen.acquire_surface(None).unwrap();
/// let view = SurfaceView::from_buffer(&buf, 0).unwrap();
///
/// assert_eq!(view.width(), 640);
/// assert_eq!(view.height(), 480);
/// assert_eq!(view.channels(), 4);
/// ```
pub struct SurfaceView {
    buffer: gst::Buffer,
    _keepalive: Option<Box<dyn Any + Send + Sync>>,
    data_ptr: *mut std::ffi::c_void,
    pitch: u32,
    width: u32,
    height: u32,
    gpu_id: u32,
    channels: u32,
    color_format: u32,
}

// SAFETY: SurfaceView is transferred across threads (e.g. via crossbeam
// channel in Picasso) and shared via Arc in PyO3.  The raw `data_ptr`
// points to GPU memory owned by the `buffer` (refcounted GstBuffer) or
// kept alive via `_keepalive`.  Both `gst::Buffer` and
// `Box<dyn Any + Send + Sync>` are Send + Sync.
unsafe impl Send for SurfaceView {}
unsafe impl Sync for SurfaceView {}

/// Map an `NvBufSurfaceColorFormat` raw value to the number of interleaved
/// channels in the primary plane.  Returns `None` for multi-plane formats
/// (NV12, NV21, I420, YUV444, etc.) which are not representable as a
/// single contiguous array.
fn color_format_channels(color_format: u32) -> Option<u32> {
    // NvBufSurfaceColorFormat values (from nvbufsurface.h / bindgen output):
    match color_format {
        // 1 channel — GRAY8, GRAY8_ER, A32
        1 | 88 => Some(1), // GRAY8, GRAY8_ER
        68 => Some(1),     // A32 (single-channel alpha)

        // 2 channels — packed YUV 4:2:2 variants
        10 | 11 => Some(2), // UYVY, UYVY_ER
        12 | 13 => Some(2), // VYUY, VYUY_ER
        14 | 15 => Some(2), // YUYV, YUYV_ER
        16 | 17 => Some(2), // YVYU, YVYU_ER
        89..=91 => Some(2), // UYVY_709, UYVY_709_ER, UYVY_2020

        // 3 channels — interleaved RGB/BGR
        27 => Some(3), // RGB
        28 => Some(3), // BGR
        42 => Some(3), // R8_G8_B8
        43 => Some(3), // B8_G8_R8

        // 4 channels — interleaved RGBA/BGRA/ARGB/ABGR/RGBx/BGRx/xRGB/xBGR
        19 => Some(4), // RGBA
        20 => Some(4), // BGRA
        21 => Some(4), // ARGB
        22 => Some(4), // ABGR
        23 => Some(4), // RGBx
        24 => Some(4), // BGRx
        25 => Some(4), // xRGB
        26 => Some(4), // xBGR

        _ => None, // multi-plane or unknown
    }
}

impl SurfaceView {
    /// Wrap a plain GStreamer buffer without NvBufSurface validation.
    ///
    /// Surface parameters are zeroed. This is intended for code paths that
    /// do not access GPU surface data (e.g. Drop/Bypass in Picasso) and
    /// for testing without a GPU.
    pub fn wrap(buf: gst::Buffer) -> Self {
        Self {
            buffer: buf,
            _keepalive: None,
            data_ptr: std::ptr::null_mut(),
            pitch: 0,
            width: 0,
            height: 0,
            gpu_id: 0,
            channels: 0,
            color_format: 0,
        }
    }

    /// Create a view from an NvBufSurface-backed GStreamer buffer.
    ///
    /// For batched buffers (`numFilled > 1`) a lightweight single-frame view
    /// is created via [`extract_slot_view`]; for single-frame buffers
    /// (`numFilled == 1, slot_index == 0`) the buffer is used directly.
    ///
    /// # Arguments
    ///
    /// * `buf` — source buffer containing an NvBufSurface.
    /// * `slot_index` — zero-based index of the surface to view.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer is not a valid NvBufSurface, if
    /// `slot_index >= numFilled`, or if the surface's color format is
    /// multi-plane (NV12, NV21, I420).
    pub fn from_buffer(buf: &gst::Buffer, slot_index: u32) -> Result<Self, NvBufSurfaceError> {
        let surf = unsafe {
            transform::extract_nvbufsurface(buf.as_ref())
                .map_err(|e| NvBufSurfaceError::BufferCopyFailed(e.to_string()))?
        };
        let num_filled = unsafe { (*surf).numFilled };

        let view_buf = if num_filled == 1 && slot_index == 0 {
            buf.clone()
        } else {
            extract_slot_view(buf, slot_index)?
        };

        let view_surf = unsafe {
            transform::extract_nvbufsurface(view_buf.as_ref())
                .map_err(|e| NvBufSurfaceError::BufferCopyFailed(e.to_string()))?
        };
        let view_ref = unsafe { &*view_surf };
        let params = unsafe { &*view_ref.surfaceList };

        let channels = color_format_channels(params.colorFormat).ok_or_else(|| {
            NvBufSurfaceError::BufferCopyFailed(format!(
                "unsupported color format {} for SurfaceView (multi-plane formats not supported)",
                params.colorFormat
            ))
        })?;

        Ok(Self {
            data_ptr: params.dataPtr,
            pitch: params.pitch,
            width: params.width,
            height: params.height,
            gpu_id: view_ref.gpuId,
            channels,
            color_format: params.colorFormat,
            buffer: view_buf,
            _keepalive: None,
        })
    }

    /// Create a view wrapping arbitrary CUDA device memory.
    ///
    /// A synthetic `gst::Buffer` with an `NvBufSurface` descriptor is created
    /// in system memory, pointing at the given `data_ptr`.  The `keepalive`
    /// object (if any) is stored to prevent the source memory from being
    /// freed while this view exists.
    ///
    /// # Arguments
    ///
    /// * `data_ptr` — GPU device pointer to the first pixel.
    /// * `pitch` — row stride in bytes.
    /// * `width` — surface width in pixels.
    /// * `height` — surface height in pixels.
    /// * `gpu_id` — CUDA device ID the memory resides on.
    /// * `channels` — number of interleaved channels (e.g. 4 for RGBA).
    /// * `color_format` — `NvBufSurfaceColorFormat` raw value.
    /// * `keepalive` — optional boxed object that must outlive this view
    ///   (e.g. a `Py<PyAny>` holding a reference to a Python array).
    ///
    /// # Errors
    ///
    /// Returns an error if `data_ptr` is null or buffer allocation fails.
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
            buffer,
            _keepalive: keepalive,
            data_ptr,
            pitch,
            width,
            height,
            gpu_id,
            channels,
            color_format,
        })
    }

    /// The underlying GStreamer buffer containing the NvBufSurface descriptor.
    ///
    /// Use this to pass the view to APIs that expect `&gst::Buffer` (e.g.
    /// `NvBufSurfaceGenerator::transform`).
    pub fn buffer(&self) -> &gst::Buffer {
        &self.buffer
    }

    /// Mutable reference to the underlying GStreamer buffer.
    ///
    /// Needed for operations that require `buffer.make_mut()`, such as
    /// setting timestamps or submitting to an encoder.
    pub fn buffer_mut(&mut self) -> &mut gst::Buffer {
        &mut self.buffer
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
    pub fn channels(&self) -> u32 {
        self.channels
    }

    /// Raw `NvBufSurfaceColorFormat` value.
    pub fn color_format(&self) -> u32 {
        self.color_format
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
    }

    #[test]
    fn test_from_cuda_ptr_gray8() {
        gst::init().unwrap();
        let view = SurfaceView::from_cuda_ptr(
            fake_gpu_ptr(),
            640,
            640,
            480,
            1,
            1,
            1, // GRAY8
            None,
        )
        .unwrap();

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
    }

    #[test]
    fn test_buffer_mut_accessible() {
        gst::init().unwrap();
        let mut view =
            SurfaceView::from_cuda_ptr(fake_gpu_ptr(), 2560, 640, 480, 0, 4, 19, None).unwrap();

        let buf = view.buffer_mut();
        let buf_ref = buf.make_mut();
        buf_ref.set_pts(gst::ClockTime::from_nseconds(42_000));
        assert_eq!(
            view.buffer().as_ref().pts(),
            Some(gst::ClockTime::from_nseconds(42_000))
        );
    }

    #[test]
    fn test_color_format_channels() {
        assert_eq!(color_format_channels(1), Some(1)); // GRAY8
        assert_eq!(color_format_channels(88), Some(1)); // GRAY8_ER
        assert_eq!(color_format_channels(68), Some(1)); // A32
        assert_eq!(color_format_channels(19), Some(4)); // RGBA
        assert_eq!(color_format_channels(20), Some(4)); // BGRA
        assert_eq!(color_format_channels(24), Some(4)); // BGRx
        assert_eq!(color_format_channels(27), Some(3)); // RGB
        assert_eq!(color_format_channels(28), Some(3)); // BGR
        assert_eq!(color_format_channels(10), Some(2)); // UYVY
        assert_eq!(color_format_channels(14), Some(2)); // YUYV
        assert_eq!(color_format_channels(6), None); // NV12
        assert_eq!(color_format_channels(2), None); // I420
        assert_eq!(color_format_channels(0), None); // INVALID
    }

    #[test]
    fn test_surface_view_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<SurfaceView>();
    }
}
