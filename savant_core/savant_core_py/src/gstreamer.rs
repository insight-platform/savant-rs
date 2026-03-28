//! PyO3 bindings for the `savant_gstreamer` crate.
//!
//! These types are registered in the `savant_rs.gstreamer` Python submodule
//! by `savant_python` when the `gst` feature is enabled.

use pyo3::prelude::*;
use savant_gstreamer::codec::Codec;
use savant_gstreamer::mp4_muxer::{Mp4Muxer, Mp4MuxerError};

fn to_py_err(e: Mp4MuxerError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

/// Python enum for video codecs.
///
/// - ``H264``     — H.264 / AVC.
/// - ``HEVC``     — H.265 / HEVC.
/// - ``JPEG``     — Motion JPEG.
/// - ``AV1``      — AV1.
/// - ``PNG``      — PNG (CPU-based, lossless).
/// - ``RAW_RGBA`` — Raw RGBA pixel data (no encoding).
/// - ``RAW_RGB``  — Raw RGB pixel data (no encoding).
#[pyclass(
    from_py_object,
    name = "Codec",
    module = "savant_rs.gstreamer",
    eq,
    eq_int
)]
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
    #[pyo3(name = "PNG")]
    Png = 4,
    #[pyo3(name = "RAW_RGBA")]
    RawRgba = 5,
    #[pyo3(name = "RAW_RGB")]
    RawRgb = 6,
}

#[pymethods]
impl PyCodec {
    /// Parse a codec from a string name.
    ///
    /// Accepted names (case-insensitive): ``h264``, ``hevc``, ``h265``,
    /// ``jpeg``, ``av1``, ``png``, ``raw_rgba``, ``raw_rgb``.
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
                "Unknown codec: '{}'. Expected one of: h264, hevc, h265, jpeg, av1, png, raw_rgba, raw_rgb",
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
                PyCodec::Png => "PNG",
                PyCodec::RawRgba => "RAW_RGBA",
                PyCodec::RawRgb => "RAW_RGB",
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
            PyCodec::Png => Codec::Png,
            PyCodec::RawRgba => Codec::RawRgba,
            PyCodec::RawRgb => Codec::RawRgb,
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
            Codec::Png => PyCodec::Png,
            Codec::RawRgba => PyCodec::RawRgba,
            Codec::RawRgb => PyCodec::RawRgb,
        }
    }
}

/// Extract a [`Codec`] from a Python object.
///
/// Accepts:
/// 1. A `PyCodec` enum value (from this module).
/// 2. A `str` codec name (e.g. `"hevc"`).
/// 3. Any object with a `.name()` method returning a codec name string
///    (e.g. a `Codec` instance from a *different* extension module).
pub fn extract_codec(ob: &Bound<'_, PyAny>) -> PyResult<Codec> {
    if let Ok(py_codec) = ob.extract::<PyCodec>() {
        return Ok(py_codec.into());
    }
    if let Ok(s) = ob.extract::<String>() {
        return Codec::from_name(&s).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown codec: '{s}'. Expected one of: h264, hevc, h265, jpeg, av1, png"
            ))
        });
    }
    if let Ok(name_val) = ob.call_method0("name") {
        if let Ok(s) = name_val.extract::<String>() {
            return Codec::from_name(&s).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown codec: '{s}'. Expected one of: h264, hevc, h265, jpeg, av1, png"
                ))
            });
        }
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected a Codec enum value or a codec name string (h264, hevc, h265, jpeg, av1, png)",
    ))
}

/// Minimal GStreamer pipeline: ``appsrc -> parser -> qtmux -> filesink``.
///
/// Accepts raw encoded frames (H.264, HEVC, JPEG, AV1) and writes them
/// into an MP4 (QuickTime) container.
///
/// Args:
///     codec (Codec | str): Video codec — a :class:`Codec` enum value or
///         a string name (``"h264"``, ``"hevc"`` / ``"h265"``,
///         ``"jpeg"``, ``"av1"``).
///     output_path (str): Filesystem path for the output ``.mp4`` file.
///     fps_num (int): Framerate numerator (default 30).
///     fps_den (int): Framerate denominator (default 1).
///
/// Example::
///
///     from savant_rs.gstreamer import Mp4Muxer, Codec
///
///     muxer = Mp4Muxer(Codec.HEVC, "/tmp/out.mp4", fps_num=30)
///     muxer.push(b"\\x00\\x00\\x00\\x01...", pts_ns=0,
///                dts_ns=0, duration_ns=33_333_333)
///     muxer.finish()
#[pyclass(name = "Mp4Muxer", module = "savant_rs.gstreamer")]
pub struct PyMp4Muxer {
    inner: Mp4Muxer,
}

#[pymethods]
impl PyMp4Muxer {
    #[new]
    #[pyo3(signature = (codec, output_path, fps_num = 30, fps_den = 1))]
    fn new(
        codec: &Bound<'_, PyAny>,
        output_path: &str,
        fps_num: i32,
        fps_den: i32,
    ) -> PyResult<Self> {
        let codec = extract_codec(codec)?;
        let inner = Mp4Muxer::new(codec, output_path, fps_num, fps_den).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Push an encoded frame into the muxer pipeline.
    ///
    /// Args:
    ///     data (bytes): Raw encoded bitstream for a single frame.
    ///     pts_ns (int): Presentation timestamp in nanoseconds.
    ///     dts_ns (int | None): Optional decode timestamp in nanoseconds.
    ///         Required for streams with B-frames where DTS != PTS.
    ///     duration_ns (int | None): Optional frame duration in nanoseconds.
    ///
    /// Raises:
    ///     RuntimeError: On push failure or if the muxer has been finalized.
    #[pyo3(signature = (data, pts_ns, dts_ns = None, duration_ns = None))]
    fn push(
        &mut self,
        py: Python<'_>,
        data: &[u8],
        pts_ns: u64,
        dts_ns: Option<u64>,
        duration_ns: Option<u64>,
    ) -> PyResult<()> {
        py.detach(|| {
            self.inner
                .push(data, pts_ns, dts_ns, duration_ns)
                .map_err(to_py_err)
        })
    }

    /// Send EOS and shut down the muxer pipeline.
    ///
    /// Safe to call multiple times. After this call, :meth:`push` will raise.
    fn finish(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.inner.finish().map_err(to_py_err))
    }

    /// Whether the muxer has been finalized.
    #[getter]
    fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }
}

/// Register the GStreamer Python classes on the given module.
pub fn register_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCodec>()?;
    m.add_class::<PyMp4Muxer>()?;
    Ok(())
}
