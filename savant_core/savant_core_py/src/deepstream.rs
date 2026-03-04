//! PyO3 bindings for the `deepstream_nvbufsurface` crate.
//!
//! These types are registered in the `savant_rs.deepstream` Python submodule
//! by `savant_python` when the `deepstream` feature is enabled.

use deepstream_nvbufsurface::transform::{self, Rect};
use deepstream_nvbufsurface::{
    bridge_savant_id_meta, cuda_init, set_num_filled, ComputeMode, DsNvNonUniformSurfaceBuffer,
    DsNvSurfaceBufferGenerator, DsNvUniformSurfaceBuffer, DsNvUniformSurfaceBufferGenerator,
    Interpolation, NvBufSurfaceMemType, Padding, TransformConfig,
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

// ─── DsNvBufSurfaceGstBuffer guard ──────────────────────────────────────

/// RAII guard for an NvBufSurface-backed ``GstBuffer``.
///
/// Wraps a GStreamer buffer and automatically unrefs it when the Python
/// object is garbage-collected.  Use ``ptr`` to obtain the raw pointer
/// for interop with functions that accept raw addresses, and ``take``
/// to transfer ownership out of the guard.
#[pyclass(name = "DsNvBufSurfaceGstBuffer", module = "savant_rs.deepstream")]
pub struct PyDsNvBufSurfaceGstBuffer {
    inner: Option<gst::Buffer>,
}

impl PyDsNvBufSurfaceGstBuffer {
    pub fn new(buffer: gst::Buffer) -> Self {
        Self {
            inner: Some(buffer),
        }
    }

    /// Return the raw pointer as `usize`, or an error if consumed.
    pub fn ptr_usize(&self) -> PyResult<usize> {
        self.inner
            .as_ref()
            .map(|b| b.as_ptr() as usize)
            .ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err(
                    "DsNvBufSurfaceGstBuffer has already been consumed via take(); \
                     create a new one with as_gst_buffer() or from_ptr()",
                )
            })
    }
}

#[pymethods]
impl PyDsNvBufSurfaceGstBuffer {
    /// Wrap a raw ``GstBuffer*`` pointer in a guard.
    ///
    /// Args:
    ///     ptr (int): Raw ``GstBuffer*`` pointer address.
    ///     add_ref (bool): If ``True`` (default) an additional reference
    ///         is taken — use for borrowed pointers (pad probes,
    ///         callbacks).  If ``False`` the guard assumes ownership of
    ///         an existing reference — use for pointers obtained via the
    ///         legacy ``int``-returning API.
    ///
    /// Raises:
    ///     ValueError: If *ptr* is 0 (null).
    #[staticmethod]
    #[pyo3(signature = (ptr, add_ref=true))]
    fn from_ptr(ptr: usize, add_ref: bool) -> PyResult<Self> {
        if ptr == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("ptr is null"));
        }
        let _ = gst::init();
        let buffer = unsafe {
            if add_ref {
                gst::Buffer::from_glib_none(ptr as *const gst::ffi::GstBuffer)
            } else {
                gst::Buffer::from_glib_full(ptr as *mut gst::ffi::GstBuffer)
            }
        };
        Ok(Self::new(buffer))
    }

    /// Raw ``GstBuffer*`` pointer address.
    ///
    /// Raises:
    ///     RuntimeError: If the buffer has been consumed via ``take``.
    #[getter]
    fn ptr(&self) -> PyResult<usize> {
        self.ptr_usize()
    }

    /// Transfer ownership out of the guard and return the raw pointer.
    ///
    /// After this call the guard is empty — ``ptr`` will raise and the
    /// destructor becomes a no-op.
    ///
    /// Returns:
    ///     int: Raw ``GstBuffer*`` pointer (caller owns the reference).
    ///
    /// Raises:
    ///     RuntimeError: If already consumed.
    fn take(&mut self) -> PyResult<usize> {
        let buffer = self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "DsNvBufSurfaceGstBuffer has already been consumed via take(); \
                 create a new one with as_gst_buffer() or from_ptr()",
            )
        })?;
        let raw = unsafe {
            use glib::translate::IntoGlibPtr;
            buffer.into_glib_ptr() as usize
        };
        Ok(raw)
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            Some(b) => format!("DsNvBufSurfaceGstBuffer(ptr=0x{:x})", b.as_ptr() as usize),
            None => "DsNvBufSurfaceGstBuffer(<consumed>)".to_string(),
        }
    }

    fn __bool__(&self) -> bool {
        self.inner.is_some()
    }
}

/// Obtain a `&mut gst::BufferRef` from a Python buffer argument.
///
/// When `buf` is a [`PyDsNvBufSurfaceGstBuffer`] the inner `gst::Buffer` is
/// made writable via COW (`make_mut`), so the caller always gets a valid
/// mutable reference regardless of the current refcount. For raw integer
/// pointers writability is checked explicitly; a clear error is raised when
/// the buffer has refcount > 1.
fn with_mut_buffer_ref<F, R>(buf: &Bound<'_, PyAny>, f: F) -> PyResult<R>
where
    F: FnOnce(&mut gst::BufferRef) -> PyResult<R>,
{
    let _ = gst::init();

    if let Ok(mut guard) = buf.extract::<PyRefMut<'_, PyDsNvBufSurfaceGstBuffer>>() {
        let buffer = guard.inner.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "DsNvBufSurfaceGstBuffer has already been consumed via take(); \
                 create a new one with as_gst_buffer() or from_ptr()",
            )
        })?;
        let buf_ref = buffer.make_mut();
        return f(buf_ref);
    }

    let raw = buf.extract::<usize>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "expected DsNvBufSurfaceGstBuffer or int (raw GstBuffer* pointer)",
        )
    })?;
    if raw == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "GstBuffer pointer is null (0); pass a valid DsNvBufSurfaceGstBuffer \
             or a non-zero raw pointer",
        ));
    }
    unsafe {
        let writable = gst::ffi::gst_mini_object_is_writable(raw as *mut gst::ffi::GstMiniObject);
        if writable == glib::ffi::GFALSE {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "GstBuffer is not writable (refcount > 1). When passing a raw \
                 pointer, ensure the buffer is exclusively owned. Prefer passing \
                 a DsNvBufSurfaceGstBuffer object instead \u{2014} it handles \
                 copy-on-write automatically.",
            ));
        }
        let buf_ref = gst::BufferRef::from_mut_ptr(raw as *mut gst::ffi::GstBuffer);
        f(buf_ref)
    }
}

/// Extract a raw ``GstBuffer*`` pointer from either a ``DsNvBufSurfaceGstBuffer`` guard
/// or a plain ``int``.
pub(crate) fn extract_buf_ptr(ob: &Bound<'_, PyAny>) -> PyResult<usize> {
    if let Ok(guard) = ob.extract::<PyRef<'_, PyDsNvBufSurfaceGstBuffer>>() {
        return guard.ptr_usize();
    }
    let raw = ob.extract::<usize>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "expected DsNvBufSurfaceGstBuffer or int (raw pointer)",
        )
    })?;
    if raw == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("buf_ptr is null"));
    }
    Ok(raw)
}

/// Extract a `gst::Buffer` from a Python buffer argument with correct
/// refcount handling.
///
/// When `buf` is a [`PyDsNvBufSurfaceGstBuffer`] the inner buffer pointer
/// is borrowed (`from_glib_none` — refcount incremented), because the
/// Python object retains ownership. For raw `usize` pointers the buffer is
/// assumed to be transferred (`from_glib_full` — no extra refcount).
///
/// GStreamer must already be initialised before calling this function.
pub(crate) fn extract_gst_buffer(buf: &Bound<'_, PyAny>) -> PyResult<gst::Buffer> {
    let is_guard = buf
        .extract::<PyRef<'_, PyDsNvBufSurfaceGstBuffer>>()
        .is_ok();
    let buf_ptr = extract_buf_ptr(buf)?;
    let gst_buf = unsafe {
        if is_guard {
            from_glib_none(buf_ptr as *const gst::ffi::GstBuffer)
        } else {
            gst::Buffer::from_glib_full(buf_ptr as *mut gst::ffi::GstBuffer)
        }
    };
    Ok(gst_buf)
}

// ─── SurfaceView ────────────────────────────────────────────────────────

/// Zero-copy view of a single GPU surface.
///
/// Wraps an NvBufSurface-backed buffer or arbitrary CUDA memory with cached
/// surface parameters.  Implements ``__cuda_array_interface__`` for
/// single-plane formats (RGBA, BGRx, GRAY8) so the surface can be consumed
/// by CuPy, PyTorch, and other CUDA-aware libraries.
///
/// Construction:
///
/// - ``SurfaceView.from_buffer(buf, slot_index)`` — from a ``GstBuffer``.
/// - ``SurfaceView.from_cuda_array(obj)`` — from any object exposing
///   ``__cuda_array_interface__`` (CuPy array, PyTorch CUDA tensor, etc.).
#[pyclass(name = "SurfaceView", module = "savant_rs.deepstream")]
pub struct PySurfaceView {
    inner: Option<deepstream_nvbufsurface::SurfaceView>,
}

impl PySurfaceView {
    pub fn new(view: deepstream_nvbufsurface::SurfaceView) -> Self {
        Self { inner: Some(view) }
    }

    /// Consume the inner SurfaceView (e.g. for passing to Picasso).
    pub fn take(&mut self) -> PyResult<deepstream_nvbufsurface::SurfaceView> {
        self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("SurfaceView has been consumed")
        })
    }

    fn inner_ref(&self) -> PyResult<&deepstream_nvbufsurface::SurfaceView> {
        self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("SurfaceView has been consumed")
        })
    }

    /// Create a SurfaceView from a `__cuda_array_interface__` object.
    /// Callable from Rust code (e.g. Picasso `send_frame` dispatch).
    pub(crate) fn from_cuda_iface(
        py: Python<'_>,
        obj: Bound<'_, PyAny>,
        gpu_id: u32,
    ) -> PyResult<Self> {
        let iface = obj.getattr("__cuda_array_interface__").map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "object does not expose __cuda_array_interface__",
            )
        })?;

        let shape: Vec<u64> = iface.get_item("shape")?.extract()?;
        let typestr: String = iface.get_item("typestr")?.extract()?;
        let data_tuple: (usize, bool) = iface.get_item("data")?.extract()?;
        let data_ptr = data_tuple.0;

        if data_ptr == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "data pointer is null",
            ));
        }
        if typestr != "|u1" && typestr != "<u1" {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unsupported dtype '{}'; only uint8 ('|u1' / '<u1') is supported",
                typestr
            )));
        }

        let (height, width, channels) = match shape.len() {
            2 => (shape[0] as u32, shape[1] as u32, 1u32),
            3 => (shape[0] as u32, shape[1] as u32, shape[2] as u32),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unsupported shape {:?}; expected (H, W) or (H, W, C)",
                    shape
                )));
            }
        };

        let color_format: u32 = match channels {
            1 => 1,  // NVBUF_COLOR_FORMAT_GRAY8
            4 => 19, // NVBUF_COLOR_FORMAT_RGBA
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unsupported channel count {}; expected 1 (GRAY8) or 4 (RGBA)",
                    channels
                )));
            }
        };

        let pitch = if let Ok(strides_obj) = iface.get_item("strides") {
            if !strides_obj.is_none() {
                let strides: Vec<u64> = strides_obj.extract()?;
                if strides.is_empty() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "strides must be non-empty when present",
                    ));
                }
                strides[0] as u32
            } else {
                width * channels
            }
        } else {
            width * channels
        };

        let keepalive: pyo3::Py<PyAny> = obj.unbind();
        let boxed: Box<dyn std::any::Any + Send + Sync> = Box::new(keepalive);

        py.detach(|| {
            let _ = gst::init();
            let view = deepstream_nvbufsurface::SurfaceView::from_cuda_ptr(
                data_ptr as *mut std::ffi::c_void,
                pitch,
                width,
                height,
                gpu_id,
                channels,
                color_format,
                Some(boxed),
            )
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(PySurfaceView::new(view))
        })
    }
}

#[pymethods]
impl PySurfaceView {
    /// Create a view from an NvBufSurface-backed buffer.
    ///
    /// Args:
    ///     buf (GstBuffer | int): Source buffer.
    ///     slot_index (int): Zero-based slot index (default 0).
    ///
    /// Raises:
    ///     ValueError: If ``buf`` is null or ``slot_index`` is out of bounds.
    ///     RuntimeError: If the buffer is not a valid NvBufSurface or uses
    ///         a multi-plane format (NV12, I420, etc.).
    #[staticmethod]
    #[pyo3(signature = (buf, slot_index=0))]
    fn from_buffer(py: Python<'_>, buf: &Bound<'_, PyAny>, slot_index: u32) -> PyResult<Self> {
        let gst_buf = extract_gst_buffer(buf)?;
        py.detach(|| {
            let view = deepstream_nvbufsurface::SurfaceView::from_buffer(&gst_buf, slot_index)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(PySurfaceView::new(view))
        })
    }

    /// Create a view from any object exposing ``__cuda_array_interface__``.
    ///
    /// Supported shapes:
    ///
    /// - ``(H, W, C)`` — interleaved: C must be 1 (GRAY8) or 4 (RGBA).
    /// - ``(H, W)``    — grayscale (GRAY8).
    ///
    /// The source object is kept alive for the lifetime of this view.
    ///
    /// Args:
    ///     obj: A CuPy array, PyTorch CUDA tensor, or any object with
    ///         ``__cuda_array_interface__``.
    ///     gpu_id (int): CUDA device ID (default 0).
    ///
    /// Raises:
    ///     TypeError: If *obj* has no ``__cuda_array_interface__``.
    ///     ValueError: If shape, dtype, or strides are unsupported.
    #[staticmethod]
    #[pyo3(signature = (obj, gpu_id=0))]
    fn from_cuda_array(py: Python<'_>, obj: Bound<'_, PyAny>, gpu_id: u32) -> PyResult<Self> {
        Self::from_cuda_iface(py, obj, gpu_id)
    }

    /// CUDA data pointer to the first pixel.
    #[getter]
    fn data_ptr(&self) -> PyResult<usize> {
        Ok(self.inner_ref()?.data_ptr() as usize)
    }

    /// Row stride in bytes.
    #[getter]
    fn pitch(&self) -> PyResult<u32> {
        Ok(self.inner_ref()?.pitch())
    }

    /// Surface width in pixels.
    #[getter]
    fn width(&self) -> PyResult<u32> {
        Ok(self.inner_ref()?.width())
    }

    /// Surface height in pixels.
    #[getter]
    fn height(&self) -> PyResult<u32> {
        Ok(self.inner_ref()?.height())
    }

    /// GPU device ID.
    #[getter]
    fn gpu_id(&self) -> PyResult<u32> {
        Ok(self.inner_ref()?.gpu_id())
    }

    /// Number of interleaved channels per pixel.
    #[getter]
    fn channels(&self) -> PyResult<u32> {
        Ok(self.inner_ref()?.channels())
    }

    /// Raw ``NvBufSurfaceColorFormat`` value.
    #[getter]
    fn color_format(&self) -> PyResult<u32> {
        Ok(self.inner_ref()?.color_format())
    }

    /// The ``__cuda_array_interface__`` descriptor (v3).
    ///
    /// Exposes the GPU surface as a CUDA array so that CuPy, PyTorch, and
    /// other external Python consumers can access the data without copies
    /// (e.g. ``cupy.asarray(surface_view)``, ``torch.as_tensor(surface_view)``).
    ///
    /// Only available for single-plane formats (RGBA, BGRx, GRAY8).
    #[getter]
    #[pyo3(name = "__cuda_array_interface__")]
    fn __cuda_array_interface__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let view = self.inner_ref()?;
        let dict = pyo3::types::PyDict::new(py);

        let channels = view.channels();
        let height = view.height();
        let width = view.width();

        let shape = if channels == 1 {
            (height as u64, width as u64)
                .into_pyobject(py)?
                .into_any()
                .unbind()
        } else {
            (height as u64, width as u64, channels as u64)
                .into_pyobject(py)?
                .into_any()
                .unbind()
        };

        let strides = if channels == 1 {
            (view.pitch() as u64, 1u64)
                .into_pyobject(py)?
                .into_any()
                .unbind()
        } else {
            (view.pitch() as u64, channels as u64, 1u64)
                .into_pyobject(py)?
                .into_any()
                .unbind()
        };

        dict.set_item("shape", shape)?;
        dict.set_item("typestr", "|u1")?;
        dict.set_item("descr", vec![("", "|u1")])?;
        dict.set_item("data", (view.data_ptr() as usize, false))?;
        dict.set_item("strides", strides)?;
        dict.set_item("version", 3)?;

        Ok(dict)
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            Some(v) => format!(
                "SurfaceView({}x{}, ch={}, gpu={})",
                v.width(),
                v.height(),
                v.channels(),
                v.gpu_id()
            ),
            None => "SurfaceView(<consumed>)".to_string(),
        }
    }

    fn __bool__(&self) -> bool {
        self.inner.is_some()
    }
}

// ─── DsNvSurfaceBufferGenerator ─────────────────────────────────────────

/// Python wrapper for DsNvSurfaceBufferGenerator.
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
#[pyclass(name = "DsNvSurfaceBufferGenerator", module = "savant_rs.deepstream")]
pub struct PyDsNvSurfaceBufferGenerator {
    inner: DsNvSurfaceBufferGenerator,
}

#[pymethods]
impl PyDsNvSurfaceBufferGenerator {
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

        let inner = DsNvSurfaceBufferGenerator::builder(format, width, height)
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
    ///     GstBuffer: Guard owning the acquired buffer.
    #[pyo3(signature = (id=None))]
    fn acquire_surface(
        &self,
        py: Python<'_>,
        id: Option<i64>,
    ) -> PyResult<PyDsNvBufSurfaceGstBuffer> {
        let buffer = py.detach(|| {
            self.inner
                .acquire_surface(id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;
        Ok(PyDsNvBufSurfaceGstBuffer::new(buffer))
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
    ///     GstBuffer: Guard owning the acquired buffer.
    #[pyo3(signature = (pts_ns, duration_ns, id=None))]
    fn acquire_surface_with_params(
        &self,
        py: Python<'_>,
        pts_ns: u64,
        duration_ns: u64,
        id: Option<i64>,
    ) -> PyResult<PyDsNvBufSurfaceGstBuffer> {
        let buffer = py.detach(|| -> PyResult<gst::Buffer> {
            let mut buffer = self
                .inner
                .acquire_surface(id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            {
                let buf_ref = buffer.make_mut();
                buf_ref.set_pts(gst::ClockTime::from_nseconds(pts_ns));
                buf_ref.set_duration(gst::ClockTime::from_nseconds(duration_ns));
            }
            Ok(buffer)
        })?;
        Ok(PyDsNvBufSurfaceGstBuffer::new(buffer))
    }

    /// Acquire a buffer and return ``(GstBuffer, data_ptr, pitch)``.
    #[pyo3(signature = (id=None))]
    fn acquire_surface_with_ptr(
        &self,
        py: Python<'_>,
        id: Option<i64>,
    ) -> PyResult<(PyDsNvBufSurfaceGstBuffer, usize, u32)> {
        let (buffer, data_ptr, pitch) = py.detach(|| -> PyResult<(gst::Buffer, usize, u32)> {
            let (buf, ptr, p) = self
                .inner
                .acquire_surface_with_ptr(id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok((buf, ptr as usize, p))
        })?;
        Ok((PyDsNvBufSurfaceGstBuffer::new(buffer), data_ptr, pitch))
    }

    /// Transform (scale + letterbox) a source buffer into a new destination.
    #[pyo3(signature = (src_buf, config, id=None, src_rect=None))]
    fn transform(
        &self,
        py: Python<'_>,
        src_buf: &Bound<'_, PyAny>,
        config: &PyTransformConfig,
        id: Option<i64>,
        src_rect: Option<&PyRect>,
    ) -> PyResult<PyDsNvBufSurfaceGstBuffer> {
        let src_buf_ptr = extract_buf_ptr(src_buf)?;
        let config = config.to_rust();
        let src_rect_rust = src_rect.map(|r| r.into_rust());
        let dst_buf = py.detach(|| {
            let src_buf =
                unsafe { gst::Buffer::from_glib_none(src_buf_ptr as *const gst::ffi::GstBuffer) };
            self.inner
                .transform(&src_buf, &config, id, src_rect_rust.as_ref())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;
        Ok(PyDsNvBufSurfaceGstBuffer::new(dst_buf))
    }

    /// Like :meth:`transform` but also returns ``(GstBuffer, data_ptr, pitch)``.
    #[pyo3(signature = (src_buf, config, id=None, src_rect=None))]
    fn transform_with_ptr(
        &self,
        py: Python<'_>,
        src_buf: &Bound<'_, PyAny>,
        config: &PyTransformConfig,
        id: Option<i64>,
        src_rect: Option<&PyRect>,
    ) -> PyResult<(PyDsNvBufSurfaceGstBuffer, usize, u32)> {
        let src_buf_ptr = extract_buf_ptr(src_buf)?;
        let config = config.to_rust();
        let src_rect_rust = src_rect.map(|r| r.into_rust());
        let (dst_buf, data_ptr, pitch) = py.detach(|| -> PyResult<(gst::Buffer, usize, u32)> {
            let src_buf =
                unsafe { gst::Buffer::from_glib_none(src_buf_ptr as *const gst::ffi::GstBuffer) };
            let (buf, ptr, p) = self
                .inner
                .transform_with_ptr(&src_buf, &config, id, src_rect_rust.as_ref())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok((buf, ptr as usize, p))
        })?;
        Ok((PyDsNvBufSurfaceGstBuffer::new(dst_buf), data_ptr, pitch))
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
            DsNvSurfaceBufferGenerator::send_eos_raw(appsrc_ptr)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        }
    }

    /// Create a new NvBufSurface and attach it to the given buffer.
    #[pyo3(signature = (gst_buffer_dest, id=None))]
    fn create_surface(
        &self,
        py: Python<'_>,
        gst_buffer_dest: &Bound<'_, PyAny>,
        id: Option<i64>,
    ) -> PyResult<()> {
        let dest_ptr = extract_buf_ptr(gst_buffer_dest)?;
        py.detach(|| unsafe {
            self.inner
                .create_surface_raw(dest_ptr as *mut gst::ffi::GstBuffer, id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }
}

// ─── DsNvUniformSurfaceBufferGenerator ───────────────────────────────────

/// Homogeneous batched NvBufSurface buffer generator.
///
/// Produces buffers whose ``surfaceList`` is an array of independently
/// fillable GPU surfaces, all sharing the same pixel format and
/// dimensions.
///
/// Args:
///     format (VideoFormat | str): Pixel format (e.g. ``"RGBA"``).
///     width (int): Slot width in pixels.
///     height (int): Slot height in pixels.
///     max_batch_size (int): Maximum number of slots per batch.
///     pool_size (int): Number of pre-allocated batched buffers (default 2).
///     fps_num (int): Framerate numerator (default 30).
///     fps_den (int): Framerate denominator (default 1).
///     gpu_id (int): GPU device ID (default 0).
///     mem_type (MemType | None): Memory type (default ``MemType.DEFAULT``).
///
/// Raises:
///     RuntimeError: If pool creation fails.
#[pyclass(
    name = "DsNvUniformSurfaceBufferGenerator",
    module = "savant_rs.deepstream"
)]
pub struct PyDsNvUniformSurfaceBufferGenerator {
    inner: DsNvUniformSurfaceBufferGenerator,
}

#[pymethods]
impl PyDsNvUniformSurfaceBufferGenerator {
    #[new]
    #[pyo3(signature = (format, width, height, max_batch_size, pool_size=2, fps_num=30, fps_den=1, gpu_id=0, mem_type=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        format: &Bound<'_, PyAny>,
        width: u32,
        height: u32,
        max_batch_size: u32,
        pool_size: u32,
        fps_num: i32,
        fps_den: i32,
        gpu_id: u32,
        mem_type: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let _ = gst::init();

        let format = extract_video_format(format)?;
        let mem_type = match mem_type {
            Some(m) => extract_mem_type(m)?,
            None => NvBufSurfaceMemType::Default,
        };

        let inner =
            DsNvUniformSurfaceBufferGenerator::builder(format, width, height, max_batch_size)
                .fps(fps_num, fps_den)
                .gpu_id(gpu_id)
                .mem_type(mem_type)
                .pool_size(pool_size)
                .build()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(Self { inner })
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

    #[getter]
    fn gpu_id(&self) -> u32 {
        self.inner.gpu_id()
    }

    #[getter]
    fn max_batch_size(&self) -> u32 {
        self.inner.max_batch_size()
    }

    /// Acquire a ``DsNvUniformSurfaceBuffer`` from the pool, ready for slot filling.
    ///
    /// Args:
    ///     config (TransformConfig): Scaling / letterboxing configuration
    ///         applied to every ``fill_slot`` call on the returned surface.
    ///
    /// Returns:
    ///     DsNvUniformSurfaceBuffer: A fresh batched surface with ``num_filled == 0``.
    ///
    /// Raises:
    ///     RuntimeError: If the pool is exhausted.
    fn acquire_batched_surface(
        &self,
        py: Python<'_>,
        config: &PyTransformConfig,
    ) -> PyResult<PyDsNvUniformSurfaceBuffer> {
        py.detach(|| {
            let batch = self
                .inner
                .acquire_batched_surface(config.to_rust())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(PyDsNvUniformSurfaceBuffer { inner: batch })
        })
    }
}

// ─── DsNvUniformSurfaceBuffer ─────────────────────────────────────────────

/// Pool-allocated batched NvBufSurface with per-slot fill tracking.
///
/// Obtained from
/// ``DsNvUniformSurfaceBufferGenerator.acquire_batched_surface``.
/// Fill individual slots with ``fill_slot``, then call ``finalize``,
/// then ``as_gst_buffer`` to access the buffer.
#[pyclass(name = "DsNvUniformSurfaceBuffer", module = "savant_rs.deepstream")]
pub struct PyDsNvUniformSurfaceBuffer {
    inner: DsNvUniformSurfaceBuffer,
}

#[pymethods]
impl PyDsNvUniformSurfaceBuffer {
    #[getter]
    fn num_filled(&self) -> u32 {
        self.inner.num_filled()
    }

    #[getter]
    fn max_batch_size(&self) -> u32 {
        self.inner.max_batch_size()
    }

    #[getter]
    fn is_finalized(&self) -> bool {
        self.inner.is_finalized()
    }

    /// Return ``(data_ptr, pitch)`` for a slot by index.
    ///
    /// Args:
    ///     index (int): Zero-based slot index (``0 .. max_batch_size - 1``).
    ///
    /// Returns:
    ///     tuple[int, int]: ``(data_ptr, pitch)`` — CUDA device pointer and
    ///     row stride in bytes.
    ///
    /// Raises:
    ///     RuntimeError: If *index* is out of bounds.
    fn slot_ptr(&self, index: u32) -> PyResult<(usize, u32)> {
        let (data_ptr, pitch) = self
            .inner
            .slot_ptr(index)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok((data_ptr as usize, pitch))
    }

    /// Transform a source buffer into the next available batch slot.
    ///
    /// The source surface is scaled (with optional letterboxing) into the
    /// destination slot according to the ``TransformConfig`` that was passed
    /// to ``acquire_batched_surface``.  The same source buffer may be used
    /// for several slots with different ``src_rect`` regions.
    ///
    /// Args:
    ///     src_buf_ptr (int): Raw ``GstBuffer*`` pointer of the source NVMM
    ///         surface (as returned by ``DsNvSurfaceBufferGenerator.acquire_surface``).
    ///     src_rect (Rect | None): Optional crop rectangle applied to the
    ///         source before scaling.  When ``None`` the full source frame is
    ///         used.  Coordinates are ``(top, left, width, height)`` in pixels.
    ///     id (int | None): Optional frame identifier stored in
    ///         ``SavantIdMeta``.  When ``None``, the id is inherited from the
    ///         source buffer's existing ``SavantIdMeta`` (if any).
    ///
    /// Raises:
    ///     ValueError: If ``src_buf_ptr`` is 0 (null).
    ///     RuntimeError: If the batch is already finalized, the batch is full,
    ///         or the GPU transform fails.
    #[pyo3(signature = (src_buf, src_rect=None, id=None))]
    fn fill_slot(
        &mut self,
        py: Python<'_>,
        src_buf: &Bound<'_, PyAny>,
        src_rect: Option<&PyRect>,
        id: Option<i64>,
    ) -> PyResult<()> {
        let src_buf_ptr = extract_buf_ptr(src_buf)?;
        let src_rect_rust = src_rect.map(|r| r.into_rust());
        py.detach(|| {
            let ptr = src_buf_ptr as *mut gst::ffi::GstBuffer;
            let src_buf = unsafe { gst::Buffer::from_glib_none(ptr) };
            self.inner
                .fill_slot(&src_buf, src_rect_rust.as_ref(), id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Finalize the batch (non-consuming).
    ///
    /// Writes ``SavantIdMeta`` with the collected frame IDs and sets
    /// ``numFilled`` on the underlying ``NvBufSurface``.  Call
    /// ``as_gst_buffer`` afterward to access the buffer.
    ///
    /// Raises:
    ///     RuntimeError: If already finalized.
    fn finalize(&mut self) -> PyResult<()> {
        self.inner
            .finalize()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Return the underlying GstBuffer guard. Available only after ``finalize``.
    ///
    /// Returns:
    ///     DsNvBufSurfaceGstBuffer: Guard for the finalized batched buffer.
    ///
    /// Raises:
    ///     RuntimeError: If not yet finalized.
    fn as_gst_buffer(&self) -> PyResult<PyDsNvBufSurfaceGstBuffer> {
        let buffer = self
            .inner
            .as_gst_buffer()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyDsNvBufSurfaceGstBuffer::new(buffer))
    }

    /// Create a zero-copy single-frame view of one filled slot.
    ///
    /// Available only after ``finalize``.
    ///
    /// Raises:
    ///     RuntimeError: If not yet finalized or slot index out of bounds.
    fn extract_slot_view(&self, slot_index: u32) -> PyResult<PyDsNvBufSurfaceGstBuffer> {
        let view = self
            .inner
            .extract_slot_view(slot_index)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyDsNvBufSurfaceGstBuffer::new(view))
    }
}

// ─── DsNvNonUniformSurfaceBuffer ─────────────────────────────────────────

/// Zero-copy heterogeneous batch (nvstreammux2-style).
///
/// Assembles individual NvBufSurface buffers of arbitrary dimensions
/// and pixel formats into a single batched ``GstBuffer``.
///
/// Args:
///     max_batch_size (int): Maximum number of surfaces in the batch.
///     gpu_id (int): GPU device ID (default 0).
///
/// Raises:
///     RuntimeError: If batch creation fails.
#[pyclass(name = "DsNvNonUniformSurfaceBuffer", module = "savant_rs.deepstream")]
pub struct PyDsNvNonUniformSurfaceBuffer {
    inner: DsNvNonUniformSurfaceBuffer,
}

#[pymethods]
impl PyDsNvNonUniformSurfaceBuffer {
    #[new]
    #[pyo3(signature = (max_batch_size, gpu_id=0))]
    fn new(max_batch_size: u32, gpu_id: u32) -> PyResult<Self> {
        let _ = gst::init();
        let inner = DsNvNonUniformSurfaceBuffer::new(max_batch_size, gpu_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    fn num_filled(&self) -> u32 {
        self.inner.num_filled()
    }

    #[getter]
    fn max_batch_size(&self) -> u32 {
        self.inner.max_batch_size()
    }

    #[getter]
    fn gpu_id(&self) -> u32 {
        self.inner.gpu_id()
    }

    #[getter]
    fn is_finalized(&self) -> bool {
        self.inner.is_finalized()
    }

    /// Add a source buffer to the batch (zero-copy).
    ///
    /// The source buffer's ``NvBufSurface`` is appended to the batch
    /// without copying pixel data.
    ///
    /// Args:
    ///     src_buf_ptr (int): Raw ``GstBuffer*`` pointer of the source NVMM
    ///         surface.
    ///     id (int | None): Optional frame identifier stored in
    ///         ``SavantIdMeta``.  When ``None``, the id is inherited from
    ///         the source buffer's existing ``SavantIdMeta`` (if any).
    ///
    /// Raises:
    ///     ValueError: If ``src_buf_ptr`` is 0 (null).
    ///     RuntimeError: If the batch is already finalized or full.
    #[pyo3(signature = (src_buf, id=None))]
    fn add(&mut self, src_buf: &Bound<'_, PyAny>, id: Option<i64>) -> PyResult<()> {
        let src_buf_ptr = extract_buf_ptr(src_buf)?;
        let src_buf =
            unsafe { gst::Buffer::from_glib_none(src_buf_ptr as *const gst::ffi::GstBuffer) };
        self.inner
            .add(&src_buf, id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Return ``(data_ptr, pitch, width, height)`` for a slot by index.
    ///
    /// Args:
    ///     index (int): Zero-based slot index (``0 .. num_filled - 1``).
    ///
    /// Returns:
    ///     tuple[int, int, int, int]: ``(data_ptr, pitch, width, height)``
    ///     — CUDA device pointer, row stride, and the slot's native
    ///     dimensions in pixels.
    ///
    /// Raises:
    ///     RuntimeError: If not yet finalized or *index* is out of bounds.
    fn slot_ptr(&self, index: u32) -> PyResult<(usize, u32, u32, u32)> {
        let (data_ptr, pitch, width, height) = self
            .inner
            .slot_ptr(index)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok((data_ptr as usize, pitch, width, height))
    }

    /// Finalize the batch (non-consuming).
    ///
    /// Writes ``SavantIdMeta`` with the collected frame IDs and
    /// assembles the heterogeneous ``NvBufSurface``.  Call
    /// ``as_gst_buffer`` afterward to access the buffer.
    ///
    /// Raises:
    ///     RuntimeError: If already finalized.
    fn finalize(&mut self) -> PyResult<()> {
        self.inner
            .finalize()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Return the underlying GstBuffer guard. Available only after ``finalize``.
    ///
    /// Returns:
    ///     DsNvBufSurfaceGstBuffer: Guard for the finalized batch buffer.
    ///
    /// Raises:
    ///     RuntimeError: If not yet finalized.
    fn as_gst_buffer(&self) -> PyResult<PyDsNvBufSurfaceGstBuffer> {
        let buffer = self
            .inner
            .as_gst_buffer()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyDsNvBufSurfaceGstBuffer::new(buffer))
    }

    /// Create a zero-copy single-frame view of one filled slot.
    ///
    /// Available only after ``finalize``.
    ///
    /// Raises:
    ///     RuntimeError: If not yet finalized or slot index out of bounds.
    fn extract_slot_view(&self, slot_index: u32) -> PyResult<PyDsNvBufSurfaceGstBuffer> {
        let view = self
            .inner
            .extract_slot_view(slot_index)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyDsNvBufSurfaceGstBuffer::new(view))
    }
}

// ─── set_num_filled ──────────────────────────────────────────────────────

/// Set numFilled on a batched NvBufSurface GstBuffer.
///
/// Args:
///     buf (GstBuffer | int): Buffer containing a batched NvBufSurface.
///     count (int): Number of filled slots.
#[pyfunction]
#[pyo3(name = "set_num_filled")]
pub fn py_set_num_filled(buf: &Bound<'_, PyAny>, count: u32) -> PyResult<()> {
    with_mut_buffer_ref(buf, |buf_ref| {
        set_num_filled(buf_ref, count)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })
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

/// Returns GPU memory currently used, in MiB.
///
/// - dGPU (x86_64): Uses NVML to query device ``gpu_id``.
/// - Jetson (aarch64): Reads /proc/meminfo (unified memory).
///
/// Args:
///     gpu_id (int): GPU device ID (default 0).
///
/// Returns:
///     int: GPU memory used in MiB.
///
/// Raises:
///     RuntimeError: If NVML or /proc/meminfo is unavailable.
#[pyfunction]
#[pyo3(name = "gpu_mem_used_mib", signature = (gpu_id=0))]
pub fn py_gpu_mem_used_mib(gpu_id: u32) -> PyResult<u64> {
    nvidia_gpu_utils::gpu_mem_used_mib(gpu_id)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
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
pub fn py_get_savant_id_meta(buf: &Bound<'_, PyAny>) -> PyResult<Vec<(String, i64)>> {
    let buf_ptr = extract_buf_ptr(buf)?;
    let _ = gst::init();
    unsafe {
        let buf_ref = gst::BufferRef::from_ptr(buf_ptr as *const gst::ffi::GstBuffer);
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
pub fn py_get_nvbufsurface_info(buf: &Bound<'_, PyAny>) -> PyResult<(usize, u32, u32, u32)> {
    let buf_ptr = extract_buf_ptr(buf)?;
    let _ = gst::init();
    unsafe {
        let buf_ref = gst::BufferRef::from_ptr(buf_ptr as *const gst::ffi::GstBuffer);
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
    #[pyo3(signature = (buf, gpu_id=0))]
    fn from_nvbuf(buf: &Bound<'_, PyAny>, gpu_id: u32) -> PyResult<Self> {
        let buf_ptr = extract_buf_ptr(buf)?;
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

    #[pyo3(signature = (buf, config=None))]
    fn render_to_nvbuf(
        &mut self,
        buf: &Bound<'_, PyAny>,
        config: Option<&PyTransformConfig>,
    ) -> PyResult<()> {
        let rust_config = config.map(|c| c.to_rust());
        with_mut_buffer_ref(buf, |buf_ref| {
            self.inner
                .render_to_nvbuf(buf_ref, rust_config.as_ref())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }
}

// ─── GstBuffer timestamp helpers ────────────────────────────────────────

/// Set the PTS (presentation timestamp) on a ``GstBuffer``.
///
/// Args:
///     buf (GstBuffer | int): Buffer to modify.
///     pts_ns (int): PTS in nanoseconds.
#[pyfunction]
#[pyo3(name = "set_buffer_pts")]
pub fn py_set_buffer_pts(buf: &Bound<'_, PyAny>, pts_ns: u64) -> PyResult<()> {
    with_mut_buffer_ref(buf, |buf_ref| {
        buf_ref.set_pts(gst::ClockTime::from_nseconds(pts_ns));
        Ok(())
    })
}

/// Set the duration on a ``GstBuffer``.
///
/// Args:
///     buf (GstBuffer | int): Buffer to modify.
///     duration_ns (int): Duration in nanoseconds.
#[pyfunction]
#[pyo3(name = "set_buffer_duration")]
pub fn py_set_buffer_duration(buf: &Bound<'_, PyAny>, duration_ns: u64) -> PyResult<()> {
    with_mut_buffer_ref(buf, |buf_ref| {
        buf_ref.set_duration(gst::ClockTime::from_nseconds(duration_ns));
        Ok(())
    })
}

// ─── release_buffer ─────────────────────────────────────────────────────

/// Release (unref) a raw ``GstBuffer*`` pointer.
///
/// Call this to free a buffer obtained from ``acquire_surface``,
/// ``acquire_surface_with_params``, ``acquire_surface_with_ptr``,
/// ``transform``, ``transform_with_ptr``, or ``finalize`` when the
/// buffer is no longer needed and is not being passed into a GStreamer
/// pipeline.
///
/// Args:
///     buf_ptr (int): Raw ``GstBuffer*`` pointer to release.
///
/// Raises:
///     ValueError: If ``buf_ptr`` is 0 (null).
#[pyfunction]
#[pyo3(name = "release_buffer")]
pub fn py_release_buffer(buf_ptr: usize) -> PyResult<()> {
    if buf_ptr == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("buf_ptr is null"));
    }
    let _ = gst::init();
    unsafe {
        gst::ffi::gst_mini_object_unref(buf_ptr as *mut gst::ffi::GstMiniObject);
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
    m.add_class::<PyDsNvBufSurfaceGstBuffer>()?;
    m.add_class::<PySurfaceView>()?;
    m.add_class::<PyDsNvSurfaceBufferGenerator>()?;
    m.add_class::<PyDsNvUniformSurfaceBufferGenerator>()?;
    m.add_class::<PyDsNvUniformSurfaceBuffer>()?;
    m.add_class::<PyDsNvNonUniformSurfaceBuffer>()?;
    m.add_function(pyo3::wrap_pyfunction!(py_set_num_filled, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_init_cuda, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_gpu_mem_used_mib, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_bridge_savant_id_meta, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_get_savant_id_meta, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_get_nvbufsurface_info, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_set_buffer_pts, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_set_buffer_duration, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(py_release_buffer, m)?)?;
    m.add_class::<PySkiaContext>()?;
    Ok(())
}
