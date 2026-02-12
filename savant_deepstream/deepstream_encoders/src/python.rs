//! Python bindings for the GPU-accelerated encoder API.
//!
//! Provides a high-level Python interface that hides all GStreamer and
//! DeepStream internals. Users configure an encoder, submit NVMM buffers,
//! and read back encoded frames.
//!
//! # Example
//!
//! ```python
//! from deepstream_encoders import NvEncoder, Codec, EncoderConfig
//! from deepstream_nvbufsurface import init_cuda
//!
//! init_cuda()
//!
//! config = EncoderConfig(Codec.HEVC, 1920, 1080)
//! encoder = NvEncoder(config)
//!
//! gen = encoder.generator
//! for i in range(100):
//!     buf = gen.acquire_surface(id=i)
//!     encoder.submit_frame(buf, frame_id=i, pts_ns=i * 33_333_333, duration_ns=33_333_333)
//!
//! # Pull encoded frames
//! while True:
//!     frame = encoder.pull_encoded()
//!     if frame is None:
//!         break
//!     print(f"Encoded frame {frame.frame_id}: {len(frame.data)} bytes")
//!
//! remaining = encoder.finish()
//! ```

use crate::{
    EncodedFrame, EncoderConfig, EncoderError, NvBufSurfaceMemType, NvEncoder, VideoFormat,
};
use gstreamer as gst;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use savant_gstreamer::Codec;

// ---------------------------------------------------------------------------
// PyCodec — local Python enum, convertible to/from the shared Rust Codec
// ---------------------------------------------------------------------------

/// Python enum for video codecs.
///
/// - ``H264`` — H.264 / AVC.
/// - ``HEVC`` — H.265 / HEVC.
/// - ``JPEG`` — Motion JPEG.
/// - ``AV1``  — AV1.
#[pyclass(name = "Codec", eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyCodec {
    #[pyo3(name = "H264")]
    H264 = 0,
    #[pyo3(name = "HEVC")]
    Hevc = 1,
    #[pyo3(name = "JPEG")]
    Jpeg = 2,
    #[pyo3(name = "AV1")]
    Av1 = 3,
}

#[pymethods]
impl PyCodec {
    /// Parse a codec from a string name.
    ///
    /// Accepted names (case-insensitive): ``h264``, ``hevc``, ``h265``,
    /// ``jpeg``, ``av1``.
    #[staticmethod]
    fn from_name(name: &str) -> PyResult<Self> {
        match Codec::from_name(name) {
            Some(c) => Ok(c.into()),
            None => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown codec: '{}'. Expected one of: h264, hevc, h265, jpeg, av1",
                name
            ))),
        }
    }

    /// Return the canonical name of this codec.
    fn name(&self) -> &'static str {
        let c: Codec = (*self).into();
        c.name()
    }

    fn __repr__(&self) -> String {
        format!(
            "Codec.{}",
            match self {
                PyCodec::H264 => "H264",
                PyCodec::Hevc => "HEVC",
                PyCodec::Jpeg => "JPEG",
                PyCodec::Av1 => "AV1",
            }
        )
    }
}

impl From<PyCodec> for Codec {
    fn from(c: PyCodec) -> Self {
        match c {
            PyCodec::H264 => Codec::H264,
            PyCodec::Hevc => Codec::Hevc,
            PyCodec::Jpeg => Codec::Jpeg,
            PyCodec::Av1 => Codec::Av1,
        }
    }
}

impl From<Codec> for PyCodec {
    fn from(c: Codec) -> Self {
        match c {
            Codec::H264 => PyCodec::H264,
            Codec::Hevc => PyCodec::Hevc,
            Codec::Jpeg => PyCodec::Jpeg,
            Codec::Av1 => PyCodec::Av1,
        }
    }
}

// ---------------------------------------------------------------------------
// Codec extraction helper
// ---------------------------------------------------------------------------

/// Extract a [`Codec`] from a Python object.
///
/// Accepts:
/// 1. This module's `PyCodec` enum value.
/// 2. A `str` codec name (e.g. `"hevc"`).
/// 3. Any object with a `.name()` method returning a codec name string
///    (e.g. a `Codec` from another extension module).
fn extract_codec(ob: &Bound<'_, PyAny>) -> PyResult<Codec> {
    if let Ok(py_codec) = ob.extract::<PyCodec>() {
        return Ok(py_codec.into());
    }
    if let Ok(s) = ob.extract::<String>() {
        return Codec::from_name(&s).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown codec: '{s}'. Expected one of: h264, hevc, h265, jpeg, av1"
            ))
        });
    }
    if let Ok(name_val) = ob.call_method0("name") {
        if let Ok(s) = name_val.extract::<String>() {
            return Codec::from_name(&s).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown codec: '{s}'. Expected one of: h264, hevc, h265, jpeg, av1"
                ))
            });
        }
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected a Codec enum value or a codec name string (h264, hevc, h265, jpeg, av1)",
    ))
}

// ---------------------------------------------------------------------------
// PyVideoFormat — local Python enum for video pixel formats
// ---------------------------------------------------------------------------

/// Video pixel format.
///
/// - ``RGBA``  — 8-bit RGBA (4 bytes/pixel).
/// - ``BGRx``  — 8-bit BGRx (4 bytes/pixel, alpha ignored).
/// - ``NV12``  — YUV 4:2:0 semi-planar (default encoder format).
/// - ``NV21``  — YUV 4:2:0 semi-planar (UV swapped).
/// - ``I420``  — YUV 4:2:0 planar (JPEG encoder format).
/// - ``UYVY``  — YUV 4:2:2 packed.
/// - ``GRAY8`` — single-channel grayscale.
#[pyclass(name = "VideoFormat", eq, eq_int)]
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

// ---------------------------------------------------------------------------
// PyMemType — local Python enum for NvBufSurface memory types
// ---------------------------------------------------------------------------

/// NvBufSurface memory type.
///
/// - ``DEFAULT``       — CUDA Device for dGPU, Surface Array for Jetson.
/// - ``CUDA_PINNED``   — CUDA Host (pinned) memory.
/// - ``CUDA_DEVICE``   — CUDA Device memory.
/// - ``CUDA_UNIFIED``  — CUDA Unified memory.
/// - ``SURFACE_ARRAY`` — NVRM Surface Array (Jetson only).
/// - ``HANDLE``        — NVRM Handle (Jetson only).
/// - ``SYSTEM``        — System memory (malloc).
#[pyclass(name = "MemType", eq, eq_int)]
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

// ---------------------------------------------------------------------------
// PyEncoderConfig
// ---------------------------------------------------------------------------

/// Configuration for creating an :class:`NvEncoder`.
///
/// The internal buffer pools are always configured with exactly 1 buffer.
/// This is required because the NVENC hardware encoder may continue
/// DMA-reading from GPU memory after releasing the GStreamer buffer
/// reference. A pool of 1 forces serialization that prevents stale-data
/// artifacts.
///
/// Args:
///     codec (Codec | str): Video codec — a :class:`Codec` enum value or
///         a string name (``"h264"``, ``"hevc"`` / ``"h265"``,
///         ``"jpeg"``, ``"av1"``).
///     width (int): Frame width in pixels.
///     height (int): Frame height in pixels.
///     format (VideoFormat | str): Video format (default
///         ``VideoFormat.NV12``).
///     fps_num (int): Framerate numerator (default 30).
///     fps_den (int): Framerate denominator (default 1).
///     gpu_id (int): GPU device ID (default 0).
///     mem_type (MemType | int): NvBufSurface memory type (default
///         ``MemType.DEFAULT``).
///     encoder_properties (dict | None): Encoder-specific GStreamer
///         properties as string key/value pairs. B-frame properties are
///         rejected.
///
/// Example::
///
///     config = EncoderConfig(
///         Codec.HEVC, 1920, 1080,
///         format=VideoFormat.RGBA,
///         encoder_properties={"bitrate": "4000000"},
///     )
#[pyclass(name = "EncoderConfig")]
#[derive(Debug, Clone)]
pub struct PyEncoderConfig {
    inner: EncoderConfig,
}

#[pymethods]
impl PyEncoderConfig {
    #[new]
    #[pyo3(signature = (
        codec,
        width,
        height,
        format = None,
        fps_num = 30,
        fps_den = 1,
        gpu_id = 0,
        mem_type = None,
        encoder_properties = None,
    ))]
    fn new(
        codec: &Bound<'_, PyAny>,
        width: u32,
        height: u32,
        format: Option<&Bound<'_, PyAny>>,
        fps_num: i32,
        fps_den: i32,
        gpu_id: u32,
        mem_type: Option<&Bound<'_, PyAny>>,
        encoder_properties: Option<std::collections::HashMap<String, String>>,
    ) -> PyResult<Self> {
        let codec = extract_codec(codec)?;
        let format = match format {
            Some(f) => extract_video_format(f)?,
            None => VideoFormat::NV12,
        };
        let mem_type = match mem_type {
            Some(m) => extract_mem_type(m)?,
            None => NvBufSurfaceMemType::Default,
        };

        let mut config = EncoderConfig::new(codec, width, height)
            .format(format)
            .fps(fps_num, fps_den)
            .gpu_id(gpu_id)
            .mem_type(mem_type);

        if let Some(props) = encoder_properties {
            for (k, v) in props {
                config = config.encoder_property(&k, &v).map_err(to_py_err)?;
            }
        }

        Ok(Self { inner: config })
    }

    /// Video codec.
    #[getter]
    fn codec(&self) -> PyCodec {
        self.inner.codec.into()
    }

    /// Frame width in pixels.
    #[getter]
    fn width(&self) -> u32 {
        self.inner.width
    }

    /// Frame height in pixels.
    #[getter]
    fn height(&self) -> u32 {
        self.inner.height
    }

    /// Video format.
    #[getter]
    fn format(&self) -> PyVideoFormat {
        self.inner.format.into()
    }

    fn __repr__(&self) -> String {
        format!(
            "EncoderConfig(codec={:?}, {}x{}, format={}, fps={}/{})",
            self.inner.codec,
            self.inner.width,
            self.inner.height,
            self.inner.format,
            self.inner.fps_num,
            self.inner.fps_den,
        )
    }
}

/// A single encoded frame returned by :meth:`NvEncoder.pull_encoded`.
///
/// Attributes:
///     frame_id (int): User-defined frame identifier.
///     pts_ns (int): Presentation timestamp in nanoseconds.
///     dts_ns (int | None): Decode timestamp in nanoseconds.
///     duration_ns (int | None): Duration in nanoseconds.
///     data (bytes): Encoded bitstream data.
///     codec (Codec): Codec used to produce this frame.
#[pyclass(name = "EncodedFrame")]
pub struct PyEncodedFrame {
    inner: EncodedFrame,
}

#[pymethods]
impl PyEncodedFrame {
    /// User-defined frame identifier.
    #[getter]
    fn frame_id(&self) -> i64 {
        self.inner.frame_id
    }

    /// Presentation timestamp in nanoseconds.
    #[getter]
    fn pts_ns(&self) -> u64 {
        self.inner.pts_ns
    }

    /// Decode timestamp in nanoseconds (if set by the encoder).
    ///
    /// For streams without B-frames this is typically equal to PTS
    /// or ``None``.
    #[getter]
    fn dts_ns(&self) -> Option<u64> {
        self.inner.dts_ns
    }

    /// Duration in nanoseconds (if known).
    #[getter]
    fn duration_ns(&self) -> Option<u64> {
        self.inner.duration_ns
    }

    /// Encoded bitstream data as bytes.
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.data)
    }

    /// Codec used to produce this frame.
    #[getter]
    fn codec(&self) -> PyCodec {
        self.inner.codec.into()
    }

    /// Size of the encoded data in bytes.
    #[getter]
    fn size(&self) -> usize {
        self.inner.data.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "EncodedFrame(frame_id={}, pts_ns={}, size={} bytes, codec={:?})",
            self.inner.frame_id,
            self.inner.pts_ns,
            self.inner.data.len(),
            self.inner.codec,
        )
    }
}

/// GPU-accelerated video encoder.
///
/// Creates an internal GStreamer pipeline that encodes NVMM buffers
/// using hardware-accelerated NVENC / NVJPEG encoders.
///
/// The encoder:
///
/// - Rejects any property that would enable B-frames.
/// - Validates that PTS values are strictly monotonically increasing.
/// - Provides access to the internal :class:`~deepstream_nvbufsurface.NvBufSurfaceGenerator`
///   for acquiring GPU buffers.
///
/// Args:
///     config (EncoderConfig): Encoder configuration.
///
/// Example::
///
///     from deepstream_encoders import NvEncoder, EncoderConfig, Codec
///     from deepstream_nvbufsurface import init_cuda
///
///     init_cuda()
///     config = EncoderConfig(Codec.HEVC, 1920, 1080)
///     encoder = NvEncoder(config)
///
///     gen = encoder.generator
///     for i in range(100):
///         buf = gen.acquire_surface(id=i)
///         encoder.submit_frame(buf, frame_id=i,
///                              pts_ns=i * 33_333_333,
///                              duration_ns=33_333_333)
///
///     remaining = encoder.finish()
#[pyclass(name = "NvEncoder", unsendable)]
pub struct PyNvEncoder {
    inner: NvEncoder,
}

#[pymethods]
impl PyNvEncoder {
    #[new]
    fn new(config: &PyEncoderConfig) -> PyResult<Self> {
        // Ensure GStreamer is initialized from the Rust side.
        let _ = gst::init();
        let inner = NvEncoder::new(&config.inner).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// The codec used by this encoder.
    #[getter]
    fn codec(&self) -> PyCodec {
        self.inner.codec().into()
    }

    /// Return the NVMM caps string for the internal generator.
    ///
    /// Returns:
    ///     str: Caps string with ``memory:NVMM`` feature.
    fn nvmm_caps_str(&self) -> String {
        self.inner.generator().nvmm_caps().to_string()
    }

    /// Acquire a new NvBufSurface buffer from the internal pool.
    ///
    /// This is a convenience shortcut for
    /// ``encoder.generator.acquire_surface(id=...)``.
    ///
    /// Args:
    ///     id (int | None): Optional frame identifier for SavantIdMeta.
    ///
    /// Returns:
    ///     int: Raw GstBuffer pointer address.
    #[pyo3(signature = (id=None))]
    fn acquire_surface(&self, py: Python<'_>, id: Option<i64>) -> PyResult<usize> {
        py.detach(|| {
            let buffer = self
                .inner
                .generator()
                .acquire_surface(id)
                .map_err(|e| to_py_err(EncoderError::from(e)))?;
            let raw = unsafe {
                use glib::translate::IntoGlibPtr;
                buffer.into_glib_ptr() as usize
            };
            Ok(raw)
        })
    }

    /// Submit a filled NVMM buffer to the encoder.
    ///
    /// The buffer must have been acquired from :meth:`acquire_surface` or
    /// from the generator directly. PTS values must be strictly monotonically
    /// increasing.
    ///
    /// Args:
    ///     buffer_ptr (int): Raw GstBuffer pointer (from :meth:`acquire_surface`).
    ///     frame_id (int): User-defined frame identifier.
    ///     pts_ns (int): Presentation timestamp in nanoseconds.
    ///     duration_ns (int | None): Optional duration in nanoseconds.
    ///
    /// Raises:
    ///     RuntimeError: On PTS reordering, pipeline error, or if the
    ///         encoder has been finalized.
    #[pyo3(signature = (buffer_ptr, frame_id, pts_ns, duration_ns=None))]
    fn submit_frame(
        &mut self,
        py: Python<'_>,
        buffer_ptr: usize,
        frame_id: i64,
        pts_ns: u64,
        duration_ns: Option<u64>,
    ) -> PyResult<()> {
        if buffer_ptr == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "buffer_ptr is null",
            ));
        }

        // Take ownership of the buffer from the raw pointer.
        let buffer =
            unsafe { glib::translate::from_glib_full(buffer_ptr as *mut gst::ffi::GstBuffer) };

        py.detach(|| {
            self.inner
                .submit_frame(buffer, frame_id, pts_ns, duration_ns)
                .map_err(to_py_err)
        })
    }

    /// Pull one encoded frame (non-blocking).
    ///
    /// Returns:
    ///     EncodedFrame | None: The encoded frame, or ``None`` if no frame
    ///         is ready yet.
    fn pull_encoded(&self, py: Python<'_>) -> PyResult<Option<PyEncodedFrame>> {
        py.detach(|| {
            self.inner
                .pull_encoded()
                .map(|opt| opt.map(|f| PyEncodedFrame { inner: f }))
                .map_err(to_py_err)
        })
    }

    /// Pull one encoded frame with a timeout.
    ///
    /// Args:
    ///     timeout_ms (int): Maximum time to wait in milliseconds.
    ///
    /// Returns:
    ///     EncodedFrame | None: The encoded frame, or ``None`` on timeout.
    #[pyo3(signature = (timeout_ms=100))]
    fn pull_encoded_timeout(
        &self,
        py: Python<'_>,
        timeout_ms: u64,
    ) -> PyResult<Option<PyEncodedFrame>> {
        py.detach(|| {
            self.inner
                .pull_encoded_timeout(timeout_ms)
                .map(|opt| opt.map(|f| PyEncodedFrame { inner: f }))
                .map_err(to_py_err)
        })
    }

    /// Send EOS and drain all remaining encoded frames.
    ///
    /// After this call, no more frames can be submitted.
    ///
    /// Args:
    ///     drain_timeout_ms (int | None): Per-frame drain timeout in ms
    ///         (default 2000).
    ///
    /// Returns:
    ///     list[EncodedFrame]: Remaining encoded frames from the pipeline.
    #[pyo3(signature = (drain_timeout_ms=None))]
    fn finish(
        &mut self,
        py: Python<'_>,
        drain_timeout_ms: Option<u64>,
    ) -> PyResult<Vec<PyEncodedFrame>> {
        py.detach(|| {
            self.inner
                .finish(drain_timeout_ms)
                .map(|frames| {
                    frames
                        .into_iter()
                        .map(|f| PyEncodedFrame { inner: f })
                        .collect()
                })
                .map_err(to_py_err)
        })
    }

    /// Check the pipeline bus for errors (non-blocking).
    ///
    /// Raises:
    ///     RuntimeError: If a pipeline error is pending.
    fn check_error(&self) -> PyResult<()> {
        self.inner.check_error().map_err(to_py_err)
    }
}

/// Convert an [`EncoderError`] into a Python exception.
fn to_py_err(e: EncoderError) -> PyErr {
    match &e {
        EncoderError::BFramesNotAllowed(_) | EncoderError::InvalidProperty { .. } => {
            pyo3::exceptions::PyValueError::new_err(e.to_string())
        }
        EncoderError::PtsReordered { .. } => pyo3::exceptions::PyValueError::new_err(e.to_string()),
        EncoderError::AlreadyFinalized => pyo3::exceptions::PyRuntimeError::new_err(e.to_string()),
        _ => pyo3::exceptions::PyRuntimeError::new_err(e.to_string()),
    }
}

/// Register the Python module.
#[pymodule]
#[pyo3(name = "_native")]
pub fn deepstream_encoders(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCodec>()?;
    m.add_class::<PyVideoFormat>()?;
    m.add_class::<PyMemType>()?;
    m.add_class::<PyEncoderConfig>()?;
    m.add_class::<PyEncodedFrame>()?;
    m.add_class::<PyNvEncoder>()?;
    Ok(())
}
