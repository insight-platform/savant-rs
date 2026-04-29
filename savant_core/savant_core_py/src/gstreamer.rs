//! PyO3 bindings for the `savant_gstreamer` crate.
//!
//! These types are registered in the `savant_rs.gstreamer` Python submodule
//! by `savant_python` when the `gst` feature is enabled.
//!
//! The :class:`Codec` Python class is re-exported from
//! :mod:`savant_rs.primitives` (see [`crate::primitives::frame::PyVideoCodec`])
//! so both modules expose the *same* object.

use std::sync::Arc;
use std::time::Duration;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use savant_core::primitives::video_codec::VideoCodec;
use savant_gstreamer::mp4_demuxer::{
    DemuxedPacket, Mp4Demuxer, Mp4DemuxerError, Mp4DemuxerOutput, VideoInfo,
};
use savant_gstreamer::mp4_muxer::{Mp4Muxer, Mp4MuxerError};
use savant_gstreamer::uri_demuxer::{
    PropertyValue, UriDemuxer, UriDemuxerConfig, UriDemuxerError, UriDemuxerOutput,
};

use crate::primitives::frame::PyVideoCodec;

fn muxer_err(e: Mp4MuxerError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

fn demuxer_err(e: Mp4DemuxerError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

fn uri_demuxer_err(e: UriDemuxerError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

/// Coerce a Python dict of `{str: scalar}` to a `Vec<(String, PropertyValue)>`.
///
/// Accepts `bool`, `int`, `float`, `str`, and `bytes` values. Any other type
/// raises :class:`TypeError` naming the offending key.
fn pydict_to_props(dict: &Bound<'_, PyDict>) -> PyResult<Vec<(String, PropertyValue)>> {
    let mut out = Vec::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let k: String = key.extract().map_err(|_| {
            let tname = key
                .get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "<?>".to_string());
            pyo3::exceptions::PyTypeError::new_err(format!("property key must be str, got {tname}"))
        })?;
        // Order matters: `bool` must be checked before `i64` because `True`/`False`
        // extract as both. Likewise `i64` before `u64`/`f64` to preserve signedness.
        let pv = if value.is_instance_of::<pyo3::types::PyBool>() {
            PropertyValue::Bool(value.extract::<bool>()?)
        } else if let Ok(i) = value.extract::<i64>() {
            PropertyValue::I64(i)
        } else if let Ok(u) = value.extract::<u64>() {
            PropertyValue::U64(u)
        } else if let Ok(f) = value.extract::<f64>() {
            PropertyValue::F64(f)
        } else if let Ok(s) = value.extract::<String>() {
            PropertyValue::String(s)
        } else if let Ok(bytes) = value.cast::<PyBytes>() {
            PropertyValue::Bytes(bytes.as_bytes().to_vec())
        } else {
            let tname = value
                .get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "<?>".to_string());
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "property '{k}' has unsupported value type '{tname}': expected bool, int, float, str, or bytes"
            )));
        };
        out.push((k, pv));
    }
    Ok(out)
}

/// Extract a [`VideoCodec`] from a Python object.
///
/// Accepts:
/// 1. A [`PyVideoCodec`] (the unified ``Codec`` enum exposed in both
///    ``savant_rs.primitives`` and ``savant_rs.gstreamer``).
/// 2. A `str` codec name (e.g. ``"hevc"``).
/// 3. Any object with a `.name()` method returning a codec name string
///    (e.g. a ``Codec`` instance from a *different* extension module).
pub fn extract_codec(ob: &Bound<'_, PyAny>) -> PyResult<VideoCodec> {
    if let Ok(py_codec) = ob.extract::<PyVideoCodec>() {
        return Ok(py_codec.into());
    }
    if let Ok(s) = ob.extract::<String>() {
        return VideoCodec::from_name(&s).ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown codec: '{s}'. Expected one of: h264, hevc, h265, jpeg, swjpeg, av1, png, vp8, vp9, raw_rgba, raw_rgb, raw_nv12"
            ))
        });
    }
    if let Ok(name_val) = ob.call_method0("name") {
        if let Ok(s) = name_val.extract::<String>() {
            return VideoCodec::from_name(&s).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown codec: '{s}'. Expected one of: h264, hevc, h265, jpeg, swjpeg, av1, png, vp8, vp9, raw_rgba, raw_rgb, raw_nv12"
                ))
            });
        }
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected a Codec enum value or a codec name string (h264, hevc, h265, jpeg, swjpeg, av1, png, vp8, vp9, raw_rgba, raw_rgb, raw_nv12)",
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
///     muxer = Mp4Muxer(Codec.Hevc, "/tmp/out.mp4", fps_num=30)
///     muxer.push(b"\\x00\\x00\\x00\\x01...", pts_ns=0,
///                dts_ns=0, duration_ns=33_333_333)
///     muxer.finish()
#[pyclass(name = "Mp4Muxer", module = "savant_rs.gstreamer")]
pub struct PyMp4Muxer(Mp4Muxer);

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
        let inner = Mp4Muxer::new(codec, output_path, fps_num, fps_den).map_err(muxer_err)?;
        Ok(Self(inner))
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
            self.0
                .push(data, pts_ns, dts_ns, duration_ns)
                .map_err(muxer_err)
        })
    }

    /// Send EOS and shut down the muxer pipeline.
    ///
    /// Safe to call multiple times. After this call, :meth:`push` will raise.
    fn finish(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.0.finish().map_err(muxer_err))
    }

    /// Whether the muxer has been finalized.
    #[getter]
    fn is_finished(&self) -> bool {
        self.0.is_finished()
    }
}

// ---------------------------------------------------------------------------
// DemuxedPacket
// ---------------------------------------------------------------------------

/// A single demuxed elementary stream packet.
///
/// Attributes:
///     data (bytes): Encoded bitstream payload.
///     pts_ns (int): Presentation timestamp in nanoseconds.
///     dts_ns (int | None): Decode timestamp in nanoseconds, if present.
///     duration_ns (int | None): Frame duration in nanoseconds, if present.
///     is_keyframe (bool): Whether this packet is a keyframe (sync point).
#[pyclass(name = "DemuxedPacket", module = "savant_rs.gstreamer")]
pub struct PyDemuxedPacket(DemuxedPacket);

#[pymethods]
impl PyDemuxedPacket {
    /// Encoded bitstream payload.
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        pyo3::types::PyBytes::new(py, &self.0.data)
    }

    /// Presentation timestamp in nanoseconds.
    #[getter]
    fn pts_ns(&self) -> u64 {
        self.0.pts_ns
    }

    /// Decode timestamp in nanoseconds, or ``None``.
    #[getter]
    fn dts_ns(&self) -> Option<u64> {
        self.0.dts_ns
    }

    /// Frame duration in nanoseconds, or ``None``.
    #[getter]
    fn duration_ns(&self) -> Option<u64> {
        self.0.duration_ns
    }

    /// Whether this packet is a keyframe.
    #[getter]
    fn is_keyframe(&self) -> bool {
        self.0.is_keyframe
    }

    fn __repr__(&self) -> String {
        format!(
            "DemuxedPacket(pts_ns={}, dts_ns={:?}, duration_ns={:?}, is_keyframe={}, len={})",
            self.0.pts_ns,
            self.0.dts_ns,
            self.0.duration_ns,
            self.0.is_keyframe,
            self.0.data.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// Mp4DemuxerOutput
// ---------------------------------------------------------------------------

/// Video-stream metadata extracted from the MP4 container at demux time.
///
/// Attributes:
///     codec (Codec): Detected video codec.
///     width (int): Encoded frame width in pixels.
///     height (int): Encoded frame height in pixels.
///     framerate_num (int): Framerate numerator (``0`` if unknown).
///     framerate_den (int): Framerate denominator (``1`` if unknown).
///
/// Dimensions are the **encoded** values - any QuickTime display-orientation
/// transform is NOT applied here.
#[pyclass(from_py_object, name = "VideoInfo", module = "savant_rs.gstreamer")]
#[derive(Debug, Clone, Copy)]
pub struct PyVideoInfo(VideoInfo);

#[pymethods]
impl PyVideoInfo {
    #[getter]
    fn codec(&self) -> PyVideoCodec {
        self.0.codec.into()
    }

    #[getter]
    fn width(&self) -> u32 {
        self.0.width
    }

    #[getter]
    fn height(&self) -> u32 {
        self.0.height
    }

    #[getter]
    fn framerate_num(&self) -> u32 {
        self.0.framerate_num
    }

    #[getter]
    fn framerate_den(&self) -> u32 {
        self.0.framerate_den
    }

    fn __repr__(&self) -> String {
        format!(
            "VideoInfo(codec={:?}, width={}, height={}, framerate={}/{})",
            self.0.codec,
            self.0.width,
            self.0.height,
            self.0.framerate_num,
            self.0.framerate_den,
        )
    }
}

enum DemuxerOutputVariant {
    StreamInfo(Py<PyVideoInfo>),
    Packet(Py<PyDemuxedPacket>),
    Eos,
    Error(String),
}

/// Callback payload from :class:`Mp4Demuxer`.
///
/// Use the ``is_*`` properties to determine the variant, then call the
/// corresponding ``as_*`` method to get a typed value.
///
/// Variants:
///     - **StreamInfo** - auto-detected :class:`VideoInfo`.
///     - **Packet** — a demuxed :class:`DemuxedPacket`.
///     - **Eos** — end of stream; all packets have been delivered.
///     - **Error** — a pipeline error message (string).
#[pyclass(name = "Mp4DemuxerOutput", module = "savant_rs.gstreamer")]
pub struct PyMp4DemuxerOutput(DemuxerOutputVariant);

impl PyMp4DemuxerOutput {
    fn from_rust(py: Python<'_>, output: Mp4DemuxerOutput) -> PyResult<Self> {
        let variant = match output {
            Mp4DemuxerOutput::StreamInfo(info) => {
                DemuxerOutputVariant::StreamInfo(Py::new(py, PyVideoInfo(info))?)
            }
            Mp4DemuxerOutput::Packet(pkt) => {
                DemuxerOutputVariant::Packet(Py::new(py, PyDemuxedPacket(pkt))?)
            }
            Mp4DemuxerOutput::Eos => DemuxerOutputVariant::Eos,
            Mp4DemuxerOutput::Error(e) => DemuxerOutputVariant::Error(e.to_string()),
        };
        Ok(Self(variant))
    }

    /// Build from a [`UriDemuxerOutput`]. The Python-side class is shared
    /// across both demuxers so user callbacks have a single signature.
    fn from_uri_rust(py: Python<'_>, output: UriDemuxerOutput) -> PyResult<Self> {
        let variant = match output {
            UriDemuxerOutput::StreamInfo(info) => {
                DemuxerOutputVariant::StreamInfo(Py::new(py, PyVideoInfo(info))?)
            }
            UriDemuxerOutput::Packet(pkt) => {
                DemuxerOutputVariant::Packet(Py::new(py, PyDemuxedPacket(pkt))?)
            }
            UriDemuxerOutput::Eos => DemuxerOutputVariant::Eos,
            UriDemuxerOutput::Error(e) => DemuxerOutputVariant::Error(e.to_string()),
        };
        Ok(Self(variant))
    }
}

#[pymethods]
impl PyMp4DemuxerOutput {
    /// ``True`` if this is a :class:`VideoInfo` variant.
    #[getter]
    fn is_stream_info(&self) -> bool {
        matches!(self.0, DemuxerOutputVariant::StreamInfo(_))
    }

    /// ``True`` if this is a :class:`DemuxedPacket` variant.
    #[getter]
    fn is_packet(&self) -> bool {
        matches!(self.0, DemuxerOutputVariant::Packet(_))
    }

    /// ``True`` if this is an end-of-stream marker.
    #[getter]
    fn is_eos(&self) -> bool {
        matches!(self.0, DemuxerOutputVariant::Eos)
    }

    /// ``True`` if this is an error variant.
    #[getter]
    fn is_error(&self) -> bool {
        matches!(self.0, DemuxerOutputVariant::Error(_))
    }

    /// Downcast to :class:`DemuxedPacket`, or ``None``.
    fn as_packet(&self, py: Python<'_>) -> Option<Py<PyDemuxedPacket>> {
        match &self.0 {
            DemuxerOutputVariant::Packet(p) => Some(p.clone_ref(py)),
            _ => None,
        }
    }

    /// Downcast to :class:`VideoInfo`, or ``None``.
    fn as_stream_info(&self, py: Python<'_>) -> Option<Py<PyVideoInfo>> {
        match &self.0 {
            DemuxerOutputVariant::StreamInfo(v) => Some(v.clone_ref(py)),
            _ => None,
        }
    }

    /// Return the error message string, or ``None``.
    fn as_error_message(&self) -> Option<&str> {
        match &self.0 {
            DemuxerOutputVariant::Error(msg) => Some(msg.as_str()),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            DemuxerOutputVariant::StreamInfo(_) => "Mp4DemuxerOutput(StreamInfo)".to_string(),
            DemuxerOutputVariant::Packet(_) => "Mp4DemuxerOutput(Packet)".to_string(),
            DemuxerOutputVariant::Eos => "Mp4DemuxerOutput(Eos)".to_string(),
            DemuxerOutputVariant::Error(msg) => format!("Mp4DemuxerOutput(Error({msg}))"),
        }
    }
}

// ---------------------------------------------------------------------------
// Mp4Demuxer
// ---------------------------------------------------------------------------

/// Callback-based GStreamer pipeline: ``filesrc -> qtdemux -> queue -> appsink``.
///
/// Reads encoded packets from an MP4 (QuickTime) container and delivers them
/// through the ``result_callback`` supplied at construction.
///
/// When ``parsed=True`` (the default), codec-specific parsers are inserted
/// so that H.264/HEVC output uses byte-stream (Annex-B) format instead of
/// container-native AVC/HEV1 length-prefixed NALUs.
///
/// The pipeline starts immediately on construction.  Use :meth:`wait` to
/// block until all packets have been delivered (EOS) or an error occurs.
///
/// **Threading**: the callback fires on GStreamer's internal streaming
/// thread.  Do **not** call :meth:`finish` from within the callback.
///
/// Args:
///     input_path (str): Filesystem path to the ``.mp4`` file.
///     result_callback: ``Callable[[Mp4DemuxerOutput], None]`` invoked for
///         every packet, EOS, or error.
///     parsed (bool): If ``True``, insert parsers for byte-stream output.
///         Defaults to ``True``.
///
/// Example::
///
///     from savant_rs.gstreamer import Mp4Demuxer
///
///     packets = []
///     def on_output(out):
///         if out.is_packet:
///             packets.append(out.as_packet())
///
///     demuxer = Mp4Demuxer("/data/clip.mp4", on_output)
///     demuxer.wait()
///     for pkt in packets:
///         print(pkt.pts_ns, len(pkt.data))
#[pyclass(name = "Mp4Demuxer", module = "savant_rs.gstreamer")]
pub struct PyMp4Demuxer(Option<Mp4Demuxer>);

#[pymethods]
impl PyMp4Demuxer {
    #[new]
    #[pyo3(signature = (input_path, result_callback, parsed = true))]
    fn new(
        py: Python<'_>,
        input_path: &str,
        result_callback: Py<PyAny>,
        parsed: bool,
    ) -> PyResult<Self> {
        let on_output: Arc<dyn Fn(Mp4DemuxerOutput) + Send + Sync> =
            Arc::new(move |output: Mp4DemuxerOutput| {
                Python::attach(|py| {
                    let py_output = match PyMp4DemuxerOutput::from_rust(py, output) {
                        Ok(o) => o,
                        Err(e) => {
                            log::error!("Mp4Demuxer: failed to wrap output: {e}");
                            return;
                        }
                    };
                    if let Err(e) = result_callback.call1(py, (py_output,)) {
                        log::error!("Mp4Demuxer result_callback error: {e}");
                    }
                });
            });

        let input_owned = input_path.to_string();
        let demuxer = py.detach(move || {
            if parsed {
                Mp4Demuxer::new_parsed(&input_owned, move |out| on_output(out))
            } else {
                Mp4Demuxer::new(&input_owned, move |out| on_output(out))
            }
            .map_err(demuxer_err)
        })?;

        Ok(Self(Some(demuxer)))
    }

    /// Block until the demuxer reaches EOS, encounters an error, or
    /// :meth:`finish` is called.
    ///
    /// The GIL is released while waiting so the callback can fire.
    fn wait(&self, py: Python<'_>) -> PyResult<()> {
        if let Some(ref inner) = self.0 {
            py.detach(|| inner.wait());
        }
        Ok(())
    }

    /// Block until the demuxer finishes or the timeout expires.
    ///
    /// Args:
    ///     timeout_ms (int): Timeout in milliseconds.
    ///
    /// Returns:
    ///     bool: ``True`` if finished, ``False`` on timeout.
    fn wait_timeout(&self, py: Python<'_>, timeout_ms: u64) -> PyResult<bool> {
        if let Some(ref inner) = self.0 {
            let timeout = Duration::from_millis(timeout_ms);
            Ok(py.detach(|| inner.wait_timeout(timeout)))
        } else {
            Ok(true)
        }
    }

    /// Auto-detected video codec from the container, or ``None``.
    #[getter]
    fn detected_codec(&self) -> Option<PyVideoCodec> {
        self.0
            .as_ref()
            .and_then(|d| d.detected_codec())
            .map(|c| c.into())
    }

    /// Auto-detected video-stream metadata, or ``None`` if caps have not been
    /// observed yet.
    #[getter]
    fn video_info(&self, py: Python<'_>) -> PyResult<Option<Py<PyVideoInfo>>> {
        self.0
            .as_ref()
            .and_then(|d| d.video_info())
            .map(|inner| Py::new(py, PyVideoInfo(inner)))
            .transpose()
    }

    /// Block until :class:`VideoInfo` is known, the pipeline terminates, or
    /// the timeout expires.
    ///
    /// Args:
    ///     timeout_ms (int): Timeout in milliseconds.
    ///
    /// Returns:
    ///     Optional[VideoInfo]: ``None`` on timeout or if the pipeline ended
    ///     before caps were observed.
    ///
    /// The GIL is released while waiting.
    fn wait_for_video_info(
        &self,
        py: Python<'_>,
        timeout_ms: u64,
    ) -> PyResult<Option<Py<PyVideoInfo>>> {
        let Some(ref inner) = self.0 else {
            return Ok(None);
        };
        let timeout = Duration::from_millis(timeout_ms);
        let info = py.detach(|| inner.wait_for_video_info(timeout));
        info.map(|inner| Py::new(py, PyVideoInfo(inner)))
            .transpose()
    }

    /// Shut down the demuxer pipeline.
    ///
    /// Safe to call multiple times.  After this call, no more callbacks
    /// will fire.
    ///
    /// Must **not** be called from within the ``result_callback``.
    fn finish(&mut self, py: Python<'_>) {
        if let Some(mut inner) = self.0.take() {
            py.detach(move || inner.finish());
        }
    }

    /// Whether the demuxer has been finalized.
    #[getter]
    fn is_finished(&self) -> bool {
        self.0.as_ref().map(|d| d.is_finished()).unwrap_or(true)
    }
}

impl Drop for PyMp4Demuxer {
    fn drop(&mut self) {
        if let Some(inner) = self.0.take() {
            // Move the demuxer to a detached thread so that the GC thread
            // (which holds the GIL) does not block on GStreamer streaming
            // threads that try to re-acquire the GIL via callbacks.
            std::thread::spawn(move || {
                drop(inner);
            });
        }
    }
}

// ---------------------------------------------------------------------------
// UriDemuxer
// ---------------------------------------------------------------------------

/// Callback-based GStreamer URI demuxer:
/// ``urisourcebin -> parsebin -> [byte-stream capsfilter] -> queue -> appsink``.
///
/// Reads encoded elementary video packets from any GStreamer-supported URI
/// (``file://``, ``http(s)://``, ``rtsp://``, ``hls://``, ``mpegts://``, ...)
/// without decoding, and delivers them through ``result_callback``.
///
/// The callback receives :class:`Mp4DemuxerOutput` instances (shared with
/// :class:`Mp4Demuxer`); the variants (StreamInfo / Packet / Eos / Error)
/// behave identically.
///
/// When ``parsed=True`` (the default), codec-specific parsers downstream of
/// ``parsebin`` emit byte-stream (Annex-B) H.264/HEVC; otherwise the
/// parsebin-native format passes through.
///
/// **Threading**: the callback fires on GStreamer's internal streaming
/// thread.  Do **not** call :meth:`finish` from within the callback.
///
/// Args:
///     uri (str): Source URI (non-empty).
///     result_callback: ``Callable[[Mp4DemuxerOutput], None]`` invoked for
///         every stream-info, packet, EOS, or error event.
///     parsed (bool): If ``True``, request byte-stream output for H.264/HEVC.
///         Defaults to ``True``.
///     bin_properties (dict | None): Properties applied to the
///         ``urisourcebin`` element (e.g. ``{"buffer-size": 8_388_608}``).
///         Values must be ``bool``, ``int``, ``float``, ``str``, or ``bytes``.
///     source_properties (dict | None): Properties applied to the inner
///         source element autoplugged by ``urisourcebin`` (e.g. ``rtspsrc``,
///         ``souphttpsrc``) via the ``source-setup`` signal. Same accepted
///         value types as ``bin_properties``. Failures on individual
///         properties surface as :class:`Mp4DemuxerOutput` error events to
///         the callback (they do **not** tear down the pipeline up front).
///
/// Raises:
///     RuntimeError: Pipeline construction failed, URI is missing/malformed,
///         or a *bin* property cannot be applied.
///     TypeError: A ``bin_properties`` / ``source_properties`` value has an
///         unsupported type.
///
/// Example::
///
///     from pathlib import Path
///     from savant_rs.gstreamer import UriDemuxer
///
///     packets = []
///     def on_output(out):
///         if out.is_packet:
///             packets.append(out.as_packet())
///
///     demuxer = UriDemuxer(Path("/data/clip.mp4").as_uri(), on_output)
///     demuxer.wait()
///     for pkt in packets:
///         print(pkt.pts_ns, len(pkt.data))
#[pyclass(name = "UriDemuxer", module = "savant_rs.gstreamer")]
pub struct PyUriDemuxer(Option<UriDemuxer>);

#[pymethods]
impl PyUriDemuxer {
    #[new]
    #[pyo3(signature = (uri, result_callback, parsed = true, bin_properties = None, source_properties = None))]
    fn new(
        py: Python<'_>,
        uri: &str,
        result_callback: Py<PyAny>,
        parsed: bool,
        bin_properties: Option<Bound<'_, PyDict>>,
        source_properties: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let bin_props = match bin_properties {
            Some(d) => pydict_to_props(&d)?,
            None => Vec::new(),
        };
        let src_props = match source_properties {
            Some(d) => pydict_to_props(&d)?,
            None => Vec::new(),
        };

        let mut cfg = UriDemuxerConfig::new(uri).with_parsed(parsed);
        cfg.bin_properties = bin_props;
        cfg.source_properties = src_props;

        let on_output: Arc<dyn Fn(UriDemuxerOutput) + Send + Sync> =
            Arc::new(move |output: UriDemuxerOutput| {
                Python::attach(|py| {
                    let py_output = match PyMp4DemuxerOutput::from_uri_rust(py, output) {
                        Ok(o) => o,
                        Err(e) => {
                            log::error!("UriDemuxer: failed to wrap output: {e}");
                            return;
                        }
                    };
                    if let Err(e) = result_callback.call1(py, (py_output,)) {
                        log::error!("UriDemuxer result_callback error: {e}");
                    }
                });
            });

        let demuxer = py.detach(move || {
            UriDemuxer::new(cfg, move |out| on_output(out)).map_err(uri_demuxer_err)
        })?;

        Ok(Self(Some(demuxer)))
    }

    /// Block until the demuxer reaches EOS, encounters an error, or
    /// :meth:`finish` is called.
    ///
    /// The GIL is released while waiting so the callback can fire.
    fn wait(&self, py: Python<'_>) -> PyResult<()> {
        if let Some(ref inner) = self.0 {
            py.detach(|| inner.wait());
        }
        Ok(())
    }

    /// Block until the demuxer finishes or the timeout expires.
    ///
    /// Args:
    ///     timeout_ms (int): Timeout in milliseconds.
    ///
    /// Returns:
    ///     bool: ``True`` if finished, ``False`` on timeout.
    fn wait_timeout(&self, py: Python<'_>, timeout_ms: u64) -> PyResult<bool> {
        if let Some(ref inner) = self.0 {
            let timeout = Duration::from_millis(timeout_ms);
            Ok(py.detach(|| inner.wait_timeout(timeout)))
        } else {
            Ok(true)
        }
    }

    /// Auto-detected video codec from the stream, or ``None``.
    #[getter]
    fn detected_codec(&self) -> Option<PyVideoCodec> {
        self.0
            .as_ref()
            .and_then(|d| d.detected_codec())
            .map(|c| c.into())
    }

    /// Auto-detected video-stream metadata, or ``None`` if caps have not been
    /// observed yet.
    #[getter]
    fn video_info(&self, py: Python<'_>) -> PyResult<Option<Py<PyVideoInfo>>> {
        self.0
            .as_ref()
            .and_then(|d| d.video_info())
            .map(|inner| Py::new(py, PyVideoInfo(inner)))
            .transpose()
    }

    /// Block until :class:`VideoInfo` is known, the pipeline terminates, or
    /// the timeout expires.
    ///
    /// Args:
    ///     timeout_ms (int): Timeout in milliseconds.
    ///
    /// Returns:
    ///     Optional[VideoInfo]: ``None`` on timeout or if the pipeline ended
    ///     before caps were observed.
    ///
    /// The GIL is released while waiting.
    fn wait_for_video_info(
        &self,
        py: Python<'_>,
        timeout_ms: u64,
    ) -> PyResult<Option<Py<PyVideoInfo>>> {
        let Some(ref inner) = self.0 else {
            return Ok(None);
        };
        let timeout = Duration::from_millis(timeout_ms);
        let info = py.detach(|| inner.wait_for_video_info(timeout));
        info.map(|inner| Py::new(py, PyVideoInfo(inner)))
            .transpose()
    }

    /// Shut down the demuxer pipeline.
    ///
    /// Safe to call multiple times. After this call, no more callbacks
    /// will fire.
    ///
    /// Must **not** be called from within the ``result_callback``.
    fn finish(&mut self, py: Python<'_>) {
        if let Some(mut inner) = self.0.take() {
            py.detach(move || inner.finish());
        }
    }

    /// Whether the demuxer has been finalized.
    #[getter]
    fn is_finished(&self) -> bool {
        self.0.as_ref().map(|d| d.is_finished()).unwrap_or(true)
    }
}

impl Drop for PyUriDemuxer {
    fn drop(&mut self) {
        if let Some(inner) = self.0.take() {
            // Move the demuxer to a detached thread so that the GC thread
            // (which holds the GIL) does not block on GStreamer streaming
            // threads that try to re-acquire the GIL via callbacks.
            std::thread::spawn(move || {
                drop(inner);
            });
        }
    }
}

/// Register the GStreamer Python classes on the given module.
///
/// The unified :class:`Codec` enum ([`PyVideoCodec`]) is also re-exported
/// here so that `savant_rs.gstreamer.Codec` continues to resolve to the
/// same class as `savant_rs.primitives.Codec`.
pub fn register_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVideoCodec>()?;
    m.add_class::<PyMp4Muxer>()?;
    m.add_class::<PyVideoInfo>()?;
    m.add_class::<PyDemuxedPacket>()?;
    m.add_class::<PyMp4DemuxerOutput>()?;
    m.add_class::<PyMp4Demuxer>()?;
    m.add_class::<PyUriDemuxer>()?;
    Ok(())
}
