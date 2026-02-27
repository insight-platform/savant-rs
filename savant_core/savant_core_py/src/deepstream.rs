//! PyO3 bindings for the `deepstream_nvbufsurface` crate.
//!
//! These types are registered in the `savant_rs.deepstream` Python submodule
//! by `savant_python` when the `deepstream` feature is enabled.

use deepstream_nvbufsurface::transform::{self, Rect};
use deepstream_nvbufsurface::{
    bridge_savant_id_meta, cuda_init, ComputeMode, Interpolation, NvBufSurfaceGenerator,
    NvBufSurfaceMemType, Padding, TransformConfig,
};
use glib::translate::from_glib_none;
use gstreamer as gst;
use pyo3::prelude::*;
use savant_gstreamer::id_meta::{SavantIdMeta, SavantIdMetaKind};
use savant_gstreamer::VideoFormat;

// ─── Padding enum ────────────────────────────────────────────────────────

/// Padding mode for letterboxing.
///
/// - ``NONE`` -- scale to fill, may distort aspect ratio.
/// - ``RIGHT_BOTTOM`` -- image at top-left, padding on right/bottom.
/// - ``SYMMETRIC`` -- image centered, equal padding on all sides (default).
#[pyclass(
    from_py_object,
    name = "Padding",
    module = "savant_rs.deepstream",
    eq,
    eq_int
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyPadding {
    #[pyo3(name = "NONE")]
    None = 0,
    #[pyo3(name = "RIGHT_BOTTOM")]
    RightBottom = 1,
    #[pyo3(name = "SYMMETRIC")]
    Symmetric = 2,
}

impl From<PyPadding> for Padding {
    fn from(p: PyPadding) -> Self {
        match p {
            PyPadding::None => Padding::None,
            PyPadding::RightBottom => Padding::RightBottom,
            PyPadding::Symmetric => Padding::Symmetric,
        }
    }
}

// ─── Interpolation enum ──────────────────────────────────────────────────

/// Interpolation method for scaling.
///
/// - ``NEAREST``  -- nearest-neighbor.
/// - ``BILINEAR`` -- bilinear (default).
/// - ``ALGO1``    -- GPU: cubic, VIC: 5-tap.
/// - ``ALGO2``    -- GPU: super, VIC: 10-tap.
/// - ``ALGO3``    -- GPU: Lanczos, VIC: smart.
/// - ``ALGO4``    -- GPU: (ignored), VIC: nicest.
/// - ``DEFAULT``  -- GPU: nearest, VIC: nearest.
#[pyclass(
    from_py_object,
    name = "Interpolation",
    module = "savant_rs.deepstream",
    eq,
    eq_int
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyInterpolation {
    #[pyo3(name = "NEAREST")]
    Nearest = 0,
    #[pyo3(name = "BILINEAR")]
    Bilinear = 1,
    #[pyo3(name = "ALGO1")]
    Algo1 = 2,
    #[pyo3(name = "ALGO2")]
    Algo2 = 3,
    #[pyo3(name = "ALGO3")]
    Algo3 = 4,
    #[pyo3(name = "ALGO4")]
    Algo4 = 5,
    #[pyo3(name = "DEFAULT")]
    Default = 6,
}

impl From<PyInterpolation> for Interpolation {
    fn from(i: PyInterpolation) -> Self {
        match i {
            PyInterpolation::Nearest => Interpolation::Nearest,
            PyInterpolation::Bilinear => Interpolation::Bilinear,
            PyInterpolation::Algo1 => Interpolation::Algo1,
            PyInterpolation::Algo2 => Interpolation::Algo2,
            PyInterpolation::Algo3 => Interpolation::Algo3,
            PyInterpolation::Algo4 => Interpolation::Algo4,
            PyInterpolation::Default => Interpolation::Default,
        }
    }
}

// ─── ComputeMode enum ───────────────────────────────────────────────────

/// Compute backend for transform operations.
///
/// - ``DEFAULT`` -- VIC on Jetson, dGPU on x86_64 (default).
/// - ``GPU``     -- always use GPU compute.
/// - ``VIC``     -- VIC hardware (Jetson only, raises error on dGPU).
#[pyclass(
    from_py_object,
    name = "ComputeMode",
    module = "savant_rs.deepstream",
    eq,
    eq_int
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyComputeMode {
    #[pyo3(name = "DEFAULT")]
    Default = 0,
    #[pyo3(name = "GPU")]
    Gpu = 1,
    #[pyo3(name = "VIC")]
    Vic = 2,
}

impl From<PyComputeMode> for ComputeMode {
    fn from(c: PyComputeMode) -> Self {
        match c {
            PyComputeMode::Default => ComputeMode::Default,
            PyComputeMode::Gpu => ComputeMode::Gpu,
            PyComputeMode::Vic => ComputeMode::Vic,
        }
    }
}

// ─── VideoFormat enum ───────────────────────────────────────────────────

/// Video pixel format.
///
/// - ``RGBA``  — 8-bit RGBA (4 bytes/pixel).
/// - ``BGRx``  — 8-bit BGRx (4 bytes/pixel, alpha ignored).
/// - ``NV12``  — YUV 4:2:0 semi-planar (default encoder format).
/// - ``NV21``  — YUV 4:2:0 semi-planar (UV swapped).
/// - ``I420``  — YUV 4:2:0 planar (JPEG encoder format).
/// - ``UYVY``  — YUV 4:2:2 packed.
/// - ``GRAY8`` — single-channel grayscale.
#[pyclass(
    from_py_object,
    name = "VideoFormat",
    module = "savant_rs.deepstream",
    eq,
    eq_int
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyVideoFormat {
    #[pyo3(name = "RGBA")]
    Rgba = 0,
    #[pyo3(name = "BGRx")]
    Bgrx = 1,
    #[pyo3(name = "NV12")]
    Nv12 = 2,
    #[pyo3(name = "NV21")]
    Nv21 = 3,
    #[pyo3(name = "I420")]
    I420 = 4,
    #[pyo3(name = "UYVY")]
    Uyvy = 5,
    #[pyo3(name = "GRAY8")]
    Gray8 = 6,
}

#[pymethods]
impl PyVideoFormat {
    /// Parse a video format from a string name.
    #[staticmethod]
    fn from_name(name: &str) -> PyResult<Self> {
        match VideoFormat::from_name(name) {
            Some(f) => Ok(f.into()),
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown video format: '{}'. Expected one of: RGBA, BGRx, NV12, NV21, I420, UYVY, GRAY8",
                name
            ))),
        }
    }

    /// Return the canonical name of this format (e.g. ``"NV12"``).
    fn name(&self) -> &'static str {
        let f: VideoFormat = (*self).into();
        f.name()
    }

    fn __repr__(&self) -> String {
        format!(
            "VideoFormat.{}",
            match self {
                PyVideoFormat::Rgba => "RGBA",
                PyVideoFormat::Bgrx => "BGRx",
                PyVideoFormat::Nv12 => "NV12",
                PyVideoFormat::Nv21 => "NV21",
                PyVideoFormat::I420 => "I420",
                PyVideoFormat::Uyvy => "UYVY",
                PyVideoFormat::Gray8 => "GRAY8",
            }
        )
    }
}

impl From<PyVideoFormat> for VideoFormat {
    fn from(f: PyVideoFormat) -> Self {
        match f {
            PyVideoFormat::Rgba => VideoFormat::RGBA,
            PyVideoFormat::Bgrx => VideoFormat::BGRx,
            PyVideoFormat::Nv12 => VideoFormat::NV12,
            PyVideoFormat::Nv21 => VideoFormat::NV21,
            PyVideoFormat::I420 => VideoFormat::I420,
            PyVideoFormat::Uyvy => VideoFormat::UYVY,
            PyVideoFormat::Gray8 => VideoFormat::GRAY8,
        }
    }
}

impl From<VideoFormat> for PyVideoFormat {
    fn from(f: VideoFormat) -> Self {
        match f {
            VideoFormat::RGBA => PyVideoFormat::Rgba,
            VideoFormat::BGRx => PyVideoFormat::Bgrx,
            VideoFormat::NV12 => PyVideoFormat::Nv12,
            VideoFormat::NV21 => PyVideoFormat::Nv21,
            VideoFormat::I420 => PyVideoFormat::I420,
            VideoFormat::UYVY => PyVideoFormat::Uyvy,
            VideoFormat::GRAY8 => PyVideoFormat::Gray8,
        }
    }
}

fn extract_video_format(ob: &Bound<'_, PyAny>) -> PyResult<VideoFormat> {
    if let Ok(py_fmt) = ob.extract::<PyVideoFormat>() {
        return Ok(py_fmt.into());
    }
    if let Ok(s) = ob.extract::<String>() {
        return VideoFormat::from_name(&s).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown video format: '{s}'. Expected one of: RGBA, BGRx, NV12, NV21, I420, UYVY, GRAY8"
            ))
        });
    }
    if let Ok(name_val) = ob.call_method0("name") {
        if let Ok(s) = name_val.extract::<String>() {
            return VideoFormat::from_name(&s).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown video format: '{s}'. Expected one of: RGBA, BGRx, NV12, NV21, I420, UYVY, GRAY8"
                ))
            });
        }
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected a VideoFormat enum value or a format name string (RGBA, BGRx, NV12, NV21, I420, UYVY, GRAY8)",
    ))
}

// ─── MemType enum ───────────────────────────────────────────────────────

/// NvBufSurface memory type.
///
/// - ``DEFAULT``       — CUDA Device for dGPU, Surface Array for Jetson.
/// - ``CUDA_PINNED``   — CUDA Host (pinned) memory.
/// - ``CUDA_DEVICE``   — CUDA Device memory.
/// - ``CUDA_UNIFIED``  — CUDA Unified memory.
/// - ``SURFACE_ARRAY`` — NVRM Surface Array (Jetson only).
/// - ``HANDLE``        — NVRM Handle (Jetson only).
/// - ``SYSTEM``        — System memory (malloc).
#[pyclass(
    from_py_object,
    name = "MemType",
    module = "savant_rs.deepstream",
    eq,
    eq_int
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyMemType {
    #[pyo3(name = "DEFAULT")]
    Default = 0,
    #[pyo3(name = "CUDA_PINNED")]
    CudaPinned = 1,
    #[pyo3(name = "CUDA_DEVICE")]
    CudaDevice = 2,
    #[pyo3(name = "CUDA_UNIFIED")]
    CudaUnified = 3,
    #[pyo3(name = "SURFACE_ARRAY")]
    SurfaceArray = 4,
    #[pyo3(name = "HANDLE")]
    Handle = 5,
    #[pyo3(name = "SYSTEM")]
    System = 6,
}

#[pymethods]
impl PyMemType {
    /// Return the canonical name of this memory type.
    fn name(&self) -> &'static str {
        match self {
            PyMemType::Default => "default",
            PyMemType::CudaPinned => "cuda_pinned",
            PyMemType::CudaDevice => "cuda_device",
            PyMemType::CudaUnified => "cuda_unified",
            PyMemType::SurfaceArray => "surface_array",
            PyMemType::Handle => "handle",
            PyMemType::System => "system",
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "MemType.{}",
            match self {
                PyMemType::Default => "DEFAULT",
                PyMemType::CudaPinned => "CUDA_PINNED",
                PyMemType::CudaDevice => "CUDA_DEVICE",
                PyMemType::CudaUnified => "CUDA_UNIFIED",
                PyMemType::SurfaceArray => "SURFACE_ARRAY",
                PyMemType::Handle => "HANDLE",
                PyMemType::System => "SYSTEM",
            }
        )
    }
}

impl From<PyMemType> for NvBufSurfaceMemType {
    fn from(m: PyMemType) -> Self {
        match m {
            PyMemType::Default => NvBufSurfaceMemType::Default,
            PyMemType::CudaPinned => NvBufSurfaceMemType::CudaPinned,
            PyMemType::CudaDevice => NvBufSurfaceMemType::CudaDevice,
            PyMemType::CudaUnified => NvBufSurfaceMemType::CudaUnified,
            PyMemType::SurfaceArray => NvBufSurfaceMemType::SurfaceArray,
            PyMemType::Handle => NvBufSurfaceMemType::Handle,
            PyMemType::System => NvBufSurfaceMemType::System,
        }
    }
}

impl From<NvBufSurfaceMemType> for PyMemType {
    fn from(m: NvBufSurfaceMemType) -> Self {
        match m {
            NvBufSurfaceMemType::Default => PyMemType::Default,
            NvBufSurfaceMemType::CudaPinned => PyMemType::CudaPinned,
            NvBufSurfaceMemType::CudaDevice => PyMemType::CudaDevice,
            NvBufSurfaceMemType::CudaUnified => PyMemType::CudaUnified,
            NvBufSurfaceMemType::SurfaceArray => PyMemType::SurfaceArray,
            NvBufSurfaceMemType::Handle => PyMemType::Handle,
            NvBufSurfaceMemType::System => PyMemType::System,
        }
    }
}

fn extract_mem_type(ob: &Bound<'_, PyAny>) -> PyResult<NvBufSurfaceMemType> {
    if let Ok(py_mt) = ob.extract::<PyMemType>() {
        return Ok(py_mt.into());
    }
    if let Ok(v) = ob.extract::<u32>() {
        return Ok(NvBufSurfaceMemType::from(v));
    }
    if let Ok(name_val) = ob.call_method0("name") {
        if let Ok(s) = name_val.extract::<String>() {
            let mt = match s.as_str() {
                "default" => NvBufSurfaceMemType::Default,
                "cuda_pinned" => NvBufSurfaceMemType::CudaPinned,
                "cuda_device" => NvBufSurfaceMemType::CudaDevice,
                "cuda_unified" => NvBufSurfaceMemType::CudaUnified,
                "surface_array" => NvBufSurfaceMemType::SurfaceArray,
                "handle" => NvBufSurfaceMemType::Handle,
                "system" => NvBufSurfaceMemType::System,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Unknown memory type: '{s}'"
                    )));
                }
            };
            return Ok(mt);
        }
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected a MemType enum value or an int memory type (0-6)",
    ))
}

// ─── Rect ────────────────────────────────────────────────────────────────

/// A rectangle in pixel coordinates (top, left, width, height).
///
/// Used as an optional source crop region for transform and send_frame.
#[pyclass(name = "Rect", module = "savant_rs.deepstream", skip_from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct PyRect {
    #[pyo3(get, set)]
    pub top: u32,
    #[pyo3(get, set)]
    pub left: u32,
    #[pyo3(get, set)]
    pub width: u32,
    #[pyo3(get, set)]
    pub height: u32,
}

#[pymethods]
impl PyRect {
    #[new]
    #[pyo3(signature = (top, left, width, height))]
    fn new(top: u32, left: u32, width: u32, height: u32) -> Self {
        Self {
            top,
            left,
            width,
            height,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Rect(top={}, left={}, width={}, height={})",
            self.top, self.left, self.width, self.height
        )
    }
}

impl PyRect {
    pub(crate) fn into_rust(self) -> Rect {
        Rect {
            top: self.top,
            left: self.left,
            width: self.width,
            height: self.height,
        }
    }
}

// ─── TransformConfig ────────────────────────────────────────────────────

/// Configuration for a transform (scale / letterbox) operation.
///
/// All fields have sensible defaults (``Padding.SYMMETRIC``,
/// ``Interpolation.BILINEAR``, ``ComputeMode.DEFAULT``).
#[pyclass(
    from_py_object,
    name = "TransformConfig",
    module = "savant_rs.deepstream"
)]
#[derive(Debug, Clone)]
pub struct PyTransformConfig {
    #[pyo3(get, set)]
    pub padding: PyPadding,
    #[pyo3(get, set)]
    pub interpolation: PyInterpolation,
    #[pyo3(get, set)]
    pub compute_mode: PyComputeMode,
}

#[pymethods]
impl PyTransformConfig {
    #[new]
    #[pyo3(signature = (
        padding = PyPadding::Symmetric,
        interpolation = PyInterpolation::Bilinear,
        compute_mode = PyComputeMode::Default,
    ))]
    fn new(
        padding: PyPadding,
        interpolation: PyInterpolation,
        compute_mode: PyComputeMode,
    ) -> Self {
        Self {
            padding,
            interpolation,
            compute_mode,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TransformConfig(padding={:?}, interpolation={:?}, compute_mode={:?})",
            self.padding, self.interpolation, self.compute_mode,
        )
    }
}

impl PyTransformConfig {
    fn to_rust(&self) -> TransformConfig {
        TransformConfig {
            padding: self.padding.into(),
            interpolation: self.interpolation.into(),
            compute_mode: self.compute_mode.into(),
            cuda_stream: std::ptr::null_mut(),
        }
    }
}

// ─── NvBufSurfaceGenerator ──────────────────────────────────────────────

/// Python wrapper for NvBufSurfaceGenerator.
///
/// Args:
///     format (VideoFormat | str): Video format.
///     width (int): Frame width in pixels.
///     height (int): Frame height in pixels.
///     fps_num (int): Framerate numerator (default 30).
///     fps_den (int): Framerate denominator (default 1).
///     gpu_id (int): GPU device ID (default 0).
///     mem_type (MemType | int): Memory type (default ``MemType.DEFAULT``).
///     pool_size (int): Buffer pool size (default 4).
#[pyclass(name = "NvBufSurfaceGenerator", module = "savant_rs.deepstream")]
pub struct PyNvBufSurfaceGenerator {
    inner: NvBufSurfaceGenerator,
}

#[pymethods]
impl PyNvBufSurfaceGenerator {
    #[new]
    #[pyo3(signature = (format, width, height, fps_num=30, fps_den=1, gpu_id=0, mem_type=None, pool_size=4))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        format: &Bound<'_, PyAny>,
        width: u32,
        height: u32,
        fps_num: i32,
        fps_den: i32,
        gpu_id: u32,
        mem_type: Option<&Bound<'_, PyAny>>,
        pool_size: u32,
    ) -> PyResult<Self> {
        let _ = gst::init();

        let format = extract_video_format(format)?;
        let mem_type = match mem_type {
            Some(m) => extract_mem_type(m)?,
            None => NvBufSurfaceMemType::Default,
        };

        let inner = NvBufSurfaceGenerator::builder(format, width, height)
            .fps(fps_num, fps_den)
            .gpu_id(gpu_id)
            .mem_type(mem_type)
            .min_buffers(pool_size)
            .max_buffers(pool_size)
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self { inner })
    }

    /// Return the NVMM caps string for configuring an ``appsrc``.
    fn nvmm_caps_str(&self) -> String {
        self.inner.nvmm_caps().to_string()
    }

    #[getter]
    fn width(&self) -> u32 {
        self.inner.width()
    }

    #[getter]
    fn height(&self) -> u32 {
        self.inner.height()
    }

    #[getter]
    fn format(&self) -> PyVideoFormat {
        self.inner.format().into()
    }

    /// Acquire a new NvBufSurface buffer from the pool.
    ///
    /// Returns:
    ///     int: Raw pointer address of the GstBuffer.
    #[pyo3(signature = (id=None))]
    fn acquire_surface(&self, py: Python<'_>, id: Option<i64>) -> PyResult<usize> {
        py.detach(|| {
            let buffer = self
                .inner
                .acquire_surface(id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let raw = unsafe {
                use glib::translate::IntoGlibPtr;
                buffer.into_glib_ptr() as usize
            };
            Ok(raw)
        })
    }

    /// Acquire a buffer and stamp PTS and duration on it.
    ///
    /// Convenience wrapper around :meth:`acquire_surface` +
    /// :func:`set_buffer_pts` + :func:`set_buffer_duration`.
    ///
    /// Args:
    ///     pts_ns (int): Presentation timestamp in nanoseconds.
    ///     duration_ns (int): Frame duration in nanoseconds.
    ///     id (int or None): Optional buffer ID / frame index.
    ///
    /// Returns:
    ///     int: Raw pointer address of the GstBuffer.
    #[pyo3(signature = (pts_ns, duration_ns, id=None))]
    fn acquire_surface_with_params(
        &self,
        py: Python<'_>,
        pts_ns: u64,
        duration_ns: u64,
        id: Option<i64>,
    ) -> PyResult<usize> {
        py.detach(|| {
            let mut buffer = self
                .inner
                .acquire_surface(id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            {
                let buf_ref = buffer.make_mut();
                buf_ref.set_pts(gst::ClockTime::from_nseconds(pts_ns));
                buf_ref.set_duration(gst::ClockTime::from_nseconds(duration_ns));
            }
            let raw = unsafe {
                use glib::translate::IntoGlibPtr;
                buffer.into_glib_ptr() as usize
            };
            Ok(raw)
        })
    }

    /// Acquire a buffer and return ``(gst_buffer_ptr, data_ptr, pitch)``.
    #[pyo3(signature = (id=None))]
    fn acquire_surface_with_ptr(
        &self,
        py: Python<'_>,
        id: Option<i64>,
    ) -> PyResult<(usize, usize, u32)> {
        py.detach(|| {
            let (buffer, data_ptr, pitch) = self
                .inner
                .acquire_surface_with_ptr(id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let raw_buf = unsafe {
                use glib::translate::IntoGlibPtr;
                buffer.into_glib_ptr() as usize
            };
            Ok((raw_buf, data_ptr as usize, pitch))
        })
    }

    /// Transform (scale + letterbox) a source buffer into a new destination.
    #[pyo3(signature = (src_buf_ptr, config, id=None, src_rect=None))]
    fn transform(
        &self,
        py: Python<'_>,
        src_buf_ptr: usize,
        config: &PyTransformConfig,
        id: Option<i64>,
        src_rect: Option<&PyRect>,
    ) -> PyResult<usize> {
        let config = config.to_rust();
        let src_rect_rust = src_rect.map(|r| r.into_rust());
        py.detach(|| {
            if src_buf_ptr == 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "src_buf_ptr is null",
                ));
            }
            let src_buf =
                unsafe { gst::Buffer::from_glib_none(src_buf_ptr as *const gst::ffi::GstBuffer) };
            let dst_buf = self
                .inner
                .transform(&src_buf, &config, id, src_rect_rust.as_ref())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let raw = unsafe {
                use glib::translate::IntoGlibPtr;
                dst_buf.into_glib_ptr() as usize
            };
            Ok(raw)
        })
    }

    /// Like :meth:`transform` but also returns ``(buf_ptr, data_ptr, pitch)``.
    #[pyo3(signature = (src_buf_ptr, config, id=None, src_rect=None))]
    fn transform_with_ptr(
        &self,
        py: Python<'_>,
        src_buf_ptr: usize,
        config: &PyTransformConfig,
        id: Option<i64>,
        src_rect: Option<&PyRect>,
    ) -> PyResult<(usize, usize, u32)> {
        let config = config.to_rust();
        let src_rect_rust = src_rect.map(|r| r.into_rust());
        py.detach(|| {
            if src_buf_ptr == 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "src_buf_ptr is null",
                ));
            }
            let src_buf =
                unsafe { gst::Buffer::from_glib_none(src_buf_ptr as *const gst::ffi::GstBuffer) };
            let (dst_buf, data_ptr, pitch) = self
                .inner
                .transform_with_ptr(&src_buf, &config, id, src_rect_rust.as_ref())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let raw_buf = unsafe {
                use glib::translate::IntoGlibPtr;
                dst_buf.into_glib_ptr() as usize
            };
            Ok((raw_buf, data_ptr as usize, pitch))
        })
    }

    /// Push a new NVMM buffer to an AppSrc element.
    #[pyo3(signature = (appsrc_ptr, pts_ns, duration_ns, id=None))]
    fn push_to_appsrc(
        &self,
        py: Python<'_>,
        appsrc_ptr: usize,
        pts_ns: u64,
        duration_ns: u64,
        id: Option<i64>,
    ) -> PyResult<()> {
        py.detach(|| unsafe {
            self.inner
                .push_to_appsrc_raw(appsrc_ptr, pts_ns, duration_ns, id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Send an end-of-stream signal to an AppSrc element.
    #[staticmethod]
    fn send_eos(appsrc_ptr: usize) -> PyResult<()> {
        unsafe {
            NvBufSurfaceGenerator::send_eos_raw(appsrc_ptr)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        }
    }

    /// Create a new NvBufSurface and attach it to the given buffer.
    #[pyo3(signature = (gst_buffer_dest, id=None))]
    fn create_surface(
        &self,
        py: Python<'_>,
        gst_buffer_dest: usize,
        id: Option<i64>,
    ) -> PyResult<()> {
        py.detach(|| unsafe {
            self.inner
                .create_surface_raw(gst_buffer_dest as *mut gst::ffi::GstBuffer, id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }
}

// ─── Module-level functions ─────────────────────────────────────────────

/// Initialize CUDA context for the given GPU device.
///
/// Args:
///     gpu_id (int): GPU device ID (default 0).
#[pyfunction]
#[pyo3(name = "init_cuda", signature = (gpu_id=0))]
pub fn py_init_cuda(gpu_id: u32) -> PyResult<()> {
    cuda_init(gpu_id).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Install pad probes on an element to propagate ``SavantIdMeta``.
///
/// Args:
///     element_ptr (int): Raw pointer address of the GstElement.
#[pyfunction]
#[pyo3(name = "bridge_savant_id_meta")]
pub fn py_bridge_savant_id_meta(element_ptr: usize) -> PyResult<()> {
    if element_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "element_ptr is null",
        ));
    }
    let _ = gst::init();
    unsafe {
        let elem: gst::Element = from_glib_none(element_ptr as *mut gst::ffi::GstElement);
        bridge_savant_id_meta(&elem);
    }
    Ok(())
}

/// Read ``SavantIdMeta`` from a GStreamer buffer.
///
/// Returns:
///     list[tuple[str, int]]: Meta entries, e.g. ``[("frame", 42)]``.
#[pyfunction]
#[pyo3(name = "get_savant_id_meta")]
pub fn py_get_savant_id_meta(buffer_ptr: usize) -> PyResult<Vec<(String, i64)>> {
    if buffer_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "buffer_ptr is null",
        ));
    }
    let _ = gst::init();
    unsafe {
        let buf_ref = gst::BufferRef::from_ptr(buffer_ptr as *const gst::ffi::GstBuffer);
        match buf_ref.meta::<SavantIdMeta>() {
            Some(meta) => {
                let ids = meta
                    .ids()
                    .iter()
                    .map(|k| match k {
                        SavantIdMetaKind::Frame(id) => ("frame".to_string(), *id),
                        SavantIdMetaKind::Batch(id) => ("batch".to_string(), *id),
                    })
                    .collect();
                Ok(ids)
            }
            None => Ok(vec![]),
        }
    }
}

/// Extract NvBufSurface descriptor fields from an existing GstBuffer.
///
/// Returns:
///     tuple[int, int, int, int]: ``(data_ptr, pitch, width, height)``
#[pyfunction]
#[pyo3(name = "get_nvbufsurface_info")]
pub fn py_get_nvbufsurface_info(buffer_ptr: usize) -> PyResult<(usize, u32, u32, u32)> {
    if buffer_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "buffer_ptr is null",
        ));
    }
    let _ = gst::init();
    unsafe {
        let buf_ref = gst::BufferRef::from_ptr(buffer_ptr as *const gst::ffi::GstBuffer);
        let surf_ptr = transform::extract_nvbufsurface(buf_ref)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let surf = &*surf_ptr;
        let params = &*surf.surfaceList;
        Ok((
            params.dataPtr as usize,
            params.pitch as u32,
            params.width,
            params.height,
        ))
    }
}

// ─── SkiaContext ────────────────────────────────────────────────────────

/// GPU-accelerated Skia rendering context backed by CUDA-GL interop.
#[pyclass(name = "SkiaContext", module = "savant_rs.deepstream", unsendable)]
pub struct PySkiaContext {
    inner: deepstream_nvbufsurface::SkiaRenderer,
}

#[pymethods]
impl PySkiaContext {
    #[new]
    #[pyo3(signature = (width, height, gpu_id=0))]
    fn new(width: u32, height: u32, gpu_id: u32) -> PyResult<Self> {
        let inner = deepstream_nvbufsurface::SkiaRenderer::new(width, height, gpu_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (buf_ptr, gpu_id=0))]
    fn from_nvbuf(buf_ptr: usize, gpu_id: u32) -> PyResult<Self> {
        if buf_ptr == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("buf_ptr is null"));
        }
        let _ = gst::init();
        unsafe {
            let buf_ref = gst::BufferRef::from_ptr(buf_ptr as *const gst::ffi::GstBuffer);
            let surf_ptr = transform::extract_nvbufsurface(buf_ref)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let surf = &*surf_ptr;
            let params = &*surf.surfaceList;
            let inner = deepstream_nvbufsurface::SkiaRenderer::from_nvbuf(
                params.width,
                params.height,
                gpu_id,
                params.dataPtr,
                params.pitch,
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(Self { inner })
        }
    }

    #[getter]
    fn fbo_id(&self) -> u32 {
        self.inner.fbo_id()
    }

    #[getter]
    fn width(&self) -> u32 {
        self.inner.width()
    }

    #[getter]
    fn height(&self) -> u32 {
        self.inner.height()
    }

    #[pyo3(signature = (buf_ptr, config=None))]
    fn render_to_nvbuf(
        &mut self,
        buf_ptr: usize,
        config: Option<&PyTransformConfig>,
    ) -> PyResult<()> {
        if buf_ptr == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("buf_ptr is null"));
        }
        let _ = gst::init();
        let rust_config = config.map(|c| c.to_rust());
        unsafe {
            let buf_ref = gst::BufferRef::from_mut_ptr(buf_ptr as *mut gst::ffi::GstBuffer);
            self.inner
                .render_to_nvbuf(buf_ref, rust_config.as_ref())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        }
    }
}

// ─── GstBuffer timestamp helpers ────────────────────────────────────────

/// Set the PTS (presentation timestamp) on a raw ``GstBuffer`` pointer.
///
/// Args:
///     buf_ptr (int): Raw pointer to a ``GstBuffer``.
///     pts_ns (int): PTS in nanoseconds.
#[pyfunction]
#[pyo3(name = "set_buffer_pts")]
pub fn py_set_buffer_pts(buf_ptr: usize, pts_ns: u64) -> PyResult<()> {
    if buf_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("buf_ptr is null"));
    }
    let _ = gst::init();
    unsafe {
        let buf_ref = gst::BufferRef::from_mut_ptr(buf_ptr as *mut gst::ffi::GstBuffer);
        buf_ref.set_pts(gst::ClockTime::from_nseconds(pts_ns));
    }
    Ok(())
}

/// Set the duration on a raw ``GstBuffer`` pointer.
///
/// Args:
///     buf_ptr (int): Raw pointer to a ``GstBuffer``.
///     duration_ns (int): Duration in nanoseconds.
#[pyfunction]
#[pyo3(name = "set_buffer_duration")]
pub fn py_set_buffer_duration(buf_ptr: usize, duration_ns: u64) -> PyResult<()> {
    if buf_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("buf_ptr is null"));
    }
    let _ = gst::init();
    unsafe {
        let buf_ref = gst::BufferRef::from_mut_ptr(buf_ptr as *mut gst::ffi::GstBuffer);
        buf_ref.set_duration(gst::ClockTime::from_nseconds(duration_ns));
    }
    Ok(())
}

// ─── Registration ───────────────────────────────────────────────────────

/// Register the DeepStream Python classes on the given module.
pub fn register_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPadding>()?;
    m.add_class::<PyInterpolation>()?;
    m.add_class::<PyComputeMode>()?;
    m.add_class::<PyVideoFormat>()?;
    m.add_class::<PyMemType>()?;
    m.add_class::<PyRect>()?;
    m.add_class::<PyTransformConfig>()?;
    m.add_class::<PyNvBufSurfaceGenerator>()?;
    m.add_function(pyo3::wrap_pyfunction!(py_init_cuda, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_bridge_savant_id_meta, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_get_savant_id_meta, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_get_nvbufsurface_info, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_set_buffer_pts, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_set_buffer_duration, m)?)?;
    m.add_class::<PySkiaContext>()?;
    Ok(())
}
