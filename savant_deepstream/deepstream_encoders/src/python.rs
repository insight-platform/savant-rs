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

use crate::{Codec, EncodedFrame, EncoderConfig, EncoderError, NvEncoder};
use gstreamer as gst;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

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
    ///
    /// Args:
    ///     name (str): Codec name.
    ///
    /// Returns:
    ///     Codec: The parsed codec.
    ///
    /// Raises:
    ///     ValueError: If the name is not recognized.
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
        format!("Codec.{}", match self {
            PyCodec::H264 => "H264",
            PyCodec::Hevc => "HEVC",
            PyCodec::Jpeg => "JPEG",
            PyCodec::Av1 => "AV1",
        })
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

/// Configuration for creating an :class:`NvEncoder`.
///
/// The internal buffer pools are always configured with exactly 1 buffer.
/// This is required because the NVENC hardware encoder may continue
/// DMA-reading from GPU memory after releasing the GStreamer buffer
/// reference. A pool of 1 forces serialization that prevents stale-data
/// artifacts.
///
/// Args:
///     codec (Codec): Video codec.
///     width (int): Frame width in pixels.
///     height (int): Frame height in pixels.
///     format (str): Video format (default ``"NV12"``).
///     fps_num (int): Framerate numerator (default 30).
///     fps_den (int): Framerate denominator (default 1).
///     gpu_id (int): GPU device ID (default 0).
///     mem_type (int): NvBufSurface memory type (default 0).
///     encoder_properties (dict | None): Encoder-specific GStreamer
///         properties as string key/value pairs. B-frame properties are
///         rejected.
///
/// Example::
///
///     config = EncoderConfig(
///         Codec.HEVC, 1920, 1080,
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
        format = "NV12",
        fps_num = 30,
        fps_den = 1,
        gpu_id = 0,
        mem_type = 0,
        encoder_properties = None,
    ))]
    fn new(
        codec: PyCodec,
        width: u32,
        height: u32,
        format: &str,
        fps_num: i32,
        fps_den: i32,
        gpu_id: u32,
        mem_type: u32,
        encoder_properties: Option<std::collections::HashMap<String, String>>,
    ) -> PyResult<Self> {
        let mut config = EncoderConfig::new(codec.into(), width, height)
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

    /// Video format string.
    #[getter]
    fn format(&self) -> &str {
        &self.inner.format
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
        let buffer = unsafe {
            glib::translate::from_glib_full(buffer_ptr as *mut gst::ffi::GstBuffer)
        };

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
        EncoderError::PtsReordered { .. } => {
            pyo3::exceptions::PyValueError::new_err(e.to_string())
        }
        EncoderError::AlreadyFinalized => {
            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
        }
        _ => pyo3::exceptions::PyRuntimeError::new_err(e.to_string()),
    }
}

/// Register the Python module.
#[pymodule]
#[pyo3(name = "_native")]
pub fn deepstream_encoders(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCodec>()?;
    m.add_class::<PyEncoderConfig>()?;
    m.add_class::<PyEncodedFrame>()?;
    m.add_class::<PyNvEncoder>()?;
    Ok(())
}
