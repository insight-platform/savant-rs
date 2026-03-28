//! Python enum wrappers and extraction helpers for DeepStream types.

use deepstream_buffers::{
    ComputeMode, Interpolation, NvBufSurfaceMemType, Padding, SavantIdMetaKind,
};
use pyo3::prelude::*;
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

// ─── SavantIdMetaKind enum ───────────────────────────────────────────────

/// Kind tag for ``SavantIdMeta`` entries.
///
/// Each NvBufSurface buffer can carry a list of ``(SavantIdMetaKind, int)``
/// pairs that identify the logical frame or batch it belongs to.
///
/// - ``FRAME`` — per-frame identifier.
/// - ``BATCH`` — per-batch identifier.
#[pyclass(
    from_py_object,
    name = "SavantIdMetaKind",
    module = "savant_rs.deepstream",
    frozen,
    eq,
    eq_int,
    hash
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PySavantIdMetaKind {
    #[pyo3(name = "FRAME")]
    Frame = 0,
    #[pyo3(name = "BATCH")]
    Batch = 1,
}

#[pymethods]
impl PySavantIdMetaKind {
    fn __repr__(&self) -> &'static str {
        match self {
            PySavantIdMetaKind::Frame => "SavantIdMetaKind.FRAME",
            PySavantIdMetaKind::Batch => "SavantIdMetaKind.BATCH",
        }
    }
}

/// Convert a Python `(SavantIdMetaKind, int)` pair into the Rust enum.
pub(crate) fn to_rust_id_kind(kind: PySavantIdMetaKind, id: i64) -> SavantIdMetaKind {
    match kind {
        PySavantIdMetaKind::Frame => SavantIdMetaKind::Frame(id),
        PySavantIdMetaKind::Batch => SavantIdMetaKind::Batch(id),
    }
}

/// Convert a Rust `SavantIdMetaKind` into a Python `(SavantIdMetaKind, int)` pair.
pub(crate) fn from_rust_id_kind(kind: &SavantIdMetaKind) -> (PySavantIdMetaKind, i64) {
    match kind {
        SavantIdMetaKind::Frame(id) => (PySavantIdMetaKind::Frame, *id),
        SavantIdMetaKind::Batch(id) => (PySavantIdMetaKind::Batch, *id),
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

pub(crate) fn extract_video_format(ob: &Bound<'_, PyAny>) -> PyResult<VideoFormat> {
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

pub(crate) fn extract_mem_type(ob: &Bound<'_, PyAny>) -> PyResult<NvBufSurfaceMemType> {
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
