//! Python bindings for [`deepstream_inputs::flexible_decoder`].

use crate::deepstream::buffer::PySharedBuffer;
use crate::deepstream::enums::PyVideoFormat;
use crate::gstreamer::PyCodec;
use crate::primitives::frame::VideoFrame;
use deepstream_inputs::flexible_decoder::{
    DecodedFrame, DecoderParameters, FlexibleDecoder, FlexibleDecoderConfig, FlexibleDecoderOutput,
    SealedDelivery, SkipReason,
};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::sync::Arc;
use std::time::Duration;

// ─── FlexibleDecoderConfig ──────────────────────────────────────────────

/// Configuration for a :class:`FlexibleDecoder`.
///
/// Args:
///     source_id (str): Bound source_id; frames with a different source_id
///         are rejected.
///     gpu_id (int): GPU device ordinal.
///     pool_size (int): Number of RGBA buffers per internal decoder pool.
#[pyclass(name = "FlexibleDecoderConfig", module = "savant_rs.deepstream")]
pub struct PyFlexibleDecoderConfig(pub(crate) FlexibleDecoderConfig);

#[pymethods]
impl PyFlexibleDecoderConfig {
    #[new]
    fn new(source_id: String, gpu_id: u32, pool_size: u32) -> Self {
        Self(FlexibleDecoderConfig::new(source_id, gpu_id, pool_size))
    }

    /// Set the idle timeout for graceful drain (milliseconds).  Returns a new
    /// config (builder pattern).
    fn with_idle_timeout_ms(&self, ms: u64) -> Self {
        Self(self.0.clone().idle_timeout(Duration::from_millis(ms)))
    }

    /// Set the maximum frames buffered during H.264/HEVC stream detection.
    fn with_detect_buffer_limit(&self, n: usize) -> Self {
        Self(self.0.clone().detect_buffer_limit(n))
    }

    #[getter]
    fn source_id(&self) -> &str {
        &self.0.source_id
    }

    #[getter]
    fn gpu_id(&self) -> u32 {
        self.0.gpu_id
    }

    #[getter]
    fn pool_size(&self) -> u32 {
        self.0.pool_size
    }

    #[getter]
    fn get_idle_timeout_ms(&self) -> u64 {
        self.0.idle_timeout.as_millis() as u64
    }

    #[getter]
    fn get_detect_buffer_limit(&self) -> usize {
        self.0.detect_buffer_limit
    }

    fn __repr__(&self) -> String {
        format!(
            "FlexibleDecoderConfig(source_id={:?}, gpu_id={}, pool_size={}, \
             idle_timeout_ms={}, detect_buffer_limit={})",
            self.0.source_id,
            self.0.gpu_id,
            self.0.pool_size,
            self.0.idle_timeout.as_millis(),
            self.0.detect_buffer_limit,
        )
    }
}

// ─── DecoderParameters ──────────────────────────────────────────────────

/// Codec, width and height snapshot for a decoder session.
///
/// Returned inside :attr:`FlexibleDecoderOutput.parameter_change`.
#[pyclass(name = "DecoderParameters", module = "savant_rs.deepstream")]
pub struct PyDecoderParameters(DecoderParameters);

impl PyDecoderParameters {
    fn from_rust(p: &DecoderParameters) -> Self {
        Self(p.clone())
    }
}

#[pymethods]
impl PyDecoderParameters {
    #[getter]
    fn codec(&self) -> PyCodec {
        self.0.codec.into()
    }

    #[getter]
    fn width(&self) -> i64 {
        self.0.width
    }

    #[getter]
    fn height(&self) -> i64 {
        self.0.height
    }

    fn __repr__(&self) -> String {
        format!(
            "DecoderParameters(codec={:?}, width={}, height={})",
            self.0.codec.name(),
            self.0.width,
            self.0.height,
        )
    }
}

// ─── SkipReason ─────────────────────────────────────────────────────────

/// Why a frame was not decoded.
///
/// Use the ``is_*`` properties to determine the variant, and :attr:`detail`
/// for the human-readable payload of string-carrying variants.
#[pyclass(name = "SkipReason", module = "savant_rs.deepstream")]
pub struct PySkipReason(SkipReason);

impl PySkipReason {
    fn from_rust(r: &SkipReason) -> Self {
        Self(r.clone())
    }
}

#[pymethods]
impl PySkipReason {
    #[getter]
    fn is_source_id_mismatch(&self) -> bool {
        matches!(self.0, SkipReason::SourceIdMismatch { .. })
    }
    #[getter]
    fn is_unsupported_codec(&self) -> bool {
        matches!(self.0, SkipReason::UnsupportedCodec(_))
    }
    #[getter]
    fn is_waiting_for_keyframe(&self) -> bool {
        matches!(self.0, SkipReason::WaitingForKeyframe)
    }
    #[getter]
    fn is_detection_buffer_overflow(&self) -> bool {
        matches!(self.0, SkipReason::DetectionBufferOverflow)
    }
    #[getter]
    fn is_no_payload(&self) -> bool {
        matches!(self.0, SkipReason::NoPayload)
    }
    #[getter]
    fn is_invalid_payload(&self) -> bool {
        matches!(self.0, SkipReason::InvalidPayload(_))
    }
    #[getter]
    fn is_decoder_creation_failed(&self) -> bool {
        matches!(self.0, SkipReason::DecoderCreationFailed(_))
    }

    /// Human-readable detail for string-carrying variants, or ``None``.
    #[getter]
    fn detail(&self) -> Option<String> {
        match &self.0 {
            SkipReason::SourceIdMismatch { expected, actual } => {
                Some(format!("expected {expected:?}, got {actual:?}"))
            }
            SkipReason::UnsupportedCodec(c) => Some(c.as_deref().unwrap_or("(none)").to_string()),
            SkipReason::InvalidPayload(msg) => Some(msg.clone()),
            SkipReason::DecoderCreationFailed(msg) => Some(msg.clone()),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

// ─── DecodedFrame ───────────────────────────────────────────────────────

/// Scalar metadata from a decoded frame.
///
/// The GPU buffer (if not yet taken via :meth:`FlexibleDecoderOutput.take_delivery`)
/// is indicated by :attr:`has_buffer`.
#[pyclass(name = "DecodedFrame", module = "savant_rs.deepstream")]
pub struct PyDecodedFrame {
    frame_id: Option<u128>,
    pts_ns: u64,
    dts_ns: Option<u64>,
    duration_ns: Option<u64>,
    codec: PyCodec,
    format: PyVideoFormat,
    has_buffer: bool,
}

impl PyDecodedFrame {
    fn from_rust(df: &DecodedFrame) -> Self {
        Self {
            frame_id: df.frame_id,
            pts_ns: df.pts_ns,
            dts_ns: df.dts_ns,
            duration_ns: df.duration_ns,
            codec: df.codec.into(),
            format: df.format.into(),
            has_buffer: df.buffer.is_some(),
        }
    }
}

#[pymethods]
impl PyDecodedFrame {
    #[getter]
    fn frame_id(&self) -> Option<u128> {
        self.frame_id
    }
    #[getter]
    fn pts_ns(&self) -> u64 {
        self.pts_ns
    }
    #[getter]
    fn dts_ns(&self) -> Option<u64> {
        self.dts_ns
    }
    #[getter]
    fn duration_ns(&self) -> Option<u64> {
        self.duration_ns
    }
    #[getter]
    fn codec(&self) -> PyCodec {
        self.codec
    }
    #[getter]
    fn format(&self) -> PyVideoFormat {
        self.format
    }
    #[getter]
    fn has_buffer(&self) -> bool {
        self.has_buffer
    }

    fn __repr__(&self) -> String {
        format!(
            "DecodedFrame(frame_id={:?}, pts_ns={}, codec={:?}, format={:?}, has_buffer={})",
            self.frame_id, self.pts_ns, self.codec, self.format, self.has_buffer,
        )
    }
}

// ─── SealedDelivery ─────────────────────────────────────────────────────

/// A ``(VideoFrame, SharedBuffer)`` pair sealed until the associated
/// :class:`FlexibleDecoderOutput` is dropped.
///
/// Call :meth:`unseal` (blocking, GIL released) or :meth:`try_unseal`
/// (non-blocking) to obtain the pair.
#[pyclass(name = "SealedDelivery", module = "savant_rs.deepstream")]
pub struct PySealedDelivery(Option<SealedDelivery>);

impl PySealedDelivery {
    fn from_rust(sealed: SealedDelivery) -> Self {
        Self(Some(sealed))
    }

    fn ensure_alive(&self) -> PyResult<&SealedDelivery> {
        self.0.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("SealedDelivery already consumed")
        })
    }
}

fn delivery_to_py(
    py: Python<'_>,
    pair: Option<(
        savant_core::primitives::frame::VideoFrameProxy,
        deepstream_buffers::SharedBuffer,
    )>,
) -> PyResult<Option<Py<PyTuple>>> {
    match pair {
        Some((proxy, buf)) => {
            let py_frame = VideoFrame(proxy);
            let py_buf = PySharedBuffer::from_rust(buf);
            let tup = PyTuple::new(
                py,
                [
                    py_frame.into_pyobject(py)?.into_any(),
                    py_buf.into_pyobject(py)?.into_any(),
                ],
            )?;
            Ok(Some(tup.unbind()))
        }
        None => Ok(None),
    }
}

#[pymethods]
impl PySealedDelivery {
    /// Whether the seal has been released (non-blocking check).
    fn is_released(&self) -> PyResult<bool> {
        Ok(self.ensure_alive()?.is_released())
    }

    /// Block until the :class:`FlexibleDecoderOutput` is dropped, then
    /// return the ``(VideoFrame, SharedBuffer)`` pair.
    ///
    /// The GIL is released during the blocking wait so the callback
    /// thread (which needs the GIL to drop the output) can proceed.
    ///
    /// Args:
    ///     timeout_ms: Optional timeout in milliseconds.  When ``None``
    ///         (default), blocks indefinitely.  When the timeout expires,
    ///         raises ``TimeoutError``.
    ///
    /// Raises:
    ///     RuntimeError: If already consumed by a previous call.
    ///     TimeoutError: If the timeout expires before the seal is released.
    #[pyo3(signature = (timeout_ms=None))]
    fn unseal(&mut self, py: Python<'_>, timeout_ms: Option<u64>) -> PyResult<Option<Py<PyTuple>>> {
        let sealed = self.0.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("SealedDelivery already consumed")
        })?;
        let pair = match timeout_ms {
            Some(ms) => {
                let timeout = Duration::from_millis(ms);
                match py.detach(move || sealed.unseal_timeout(timeout)) {
                    Ok(pair) => pair,
                    Err(still_sealed) => {
                        self.0 = Some(still_sealed);
                        return Err(pyo3::exceptions::PyTimeoutError::new_err(
                            "unseal timed out",
                        ));
                    }
                }
            }
            None => py.detach(move || sealed.unseal()),
        };
        delivery_to_py(py, pair)
    }

    /// Non-blocking attempt to unseal.
    ///
    /// Returns ``(VideoFrame, SharedBuffer)`` if the seal has been
    /// released, or ``None`` if still sealed.
    ///
    /// Raises:
    ///     RuntimeError: If already consumed by a previous call.
    fn try_unseal(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyTuple>>> {
        let sealed = self.0.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("SealedDelivery already consumed")
        })?;
        match sealed.try_unseal() {
            Ok(pair) => delivery_to_py(py, pair),
            Err(still_sealed) => {
                self.0 = Some(still_sealed);
                Ok(None)
            }
        }
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            Some(s) => format!("SealedDelivery(released={})", s.is_released()),
            None => "SealedDelivery(<consumed>)".to_string(),
        }
    }
}

// ─── Per-variant output classes ─────────────────────────────────────────

/// Decoded frame paired with the submitted :class:`~savant_rs.primitives.VideoFrame`.
///
/// Owns the underlying ``FlexibleDecoderOutput`` so that its ``Drop``
/// implementation releases the seal when this object is garbage-collected.
///
/// Call :meth:`take_delivery` to extract a :class:`SealedDelivery`.
#[pyclass(name = "FrameOutput", module = "savant_rs.deepstream")]
pub struct PyFrameOutput(Option<FlexibleDecoderOutput>);

impl PyFrameOutput {
    fn as_frame_ref(
        &self,
    ) -> PyResult<(
        &savant_core::primitives::frame::VideoFrameProxy,
        &DecodedFrame,
    )> {
        match &self.0 {
            Some(FlexibleDecoderOutput::Frame { frame, decoded, .. }) => Ok((frame, decoded)),
            _ => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "FrameOutput inner already consumed",
            )),
        }
    }
}

#[pymethods]
impl PyFrameOutput {
    /// The submitted :class:`~savant_rs.primitives.VideoFrame`.
    #[getter]
    fn frame(&self) -> PyResult<VideoFrame> {
        let (proxy, _) = self.as_frame_ref()?;
        Ok(VideoFrame(proxy.clone()))
    }

    /// Scalar metadata of the decoded frame.
    #[getter]
    fn decoded_frame(&self) -> PyResult<PyDecodedFrame> {
        let (_, decoded) = self.as_frame_ref()?;
        Ok(PyDecodedFrame::from_rust(decoded))
    }

    /// Extract the sealed ``(VideoFrame, SharedBuffer)`` delivery.
    ///
    /// Returns :class:`SealedDelivery` on the first call.
    /// Subsequent calls return ``None``.
    fn take_delivery(&mut self) -> Option<PySealedDelivery> {
        let out = self.0.as_mut()?;
        let sealed = out.take_delivery()?;
        Some(PySealedDelivery::from_rust(sealed))
    }

    fn __repr__(&self) -> String {
        "FrameOutput(...)".to_string()
    }
}

/// Codec or resolution changed between two decoder sessions.
#[pyclass(name = "ParameterChangeOutput", module = "savant_rs.deepstream")]
pub struct PyParameterChangeOutput {
    old: PyDecoderParameters,
    new: PyDecoderParameters,
}

#[pymethods]
impl PyParameterChangeOutput {
    /// Previous decoder parameters.
    #[getter]
    fn old(&self, py: Python<'_>) -> PyResult<Py<PyDecoderParameters>> {
        Py::new(py, PyDecoderParameters(self.old.0.clone()))
    }

    /// New decoder parameters.
    #[getter]
    fn get_new(&self, py: Python<'_>) -> PyResult<Py<PyDecoderParameters>> {
        Py::new(py, PyDecoderParameters(self.new.0.clone()))
    }

    fn __repr__(&self) -> String {
        format!(
            "ParameterChangeOutput(old={}, new_={})",
            self.old.__repr__(),
            self.new.__repr__(),
        )
    }
}

/// A frame that was rejected (not submitted to the decoder).
#[pyclass(name = "SkippedOutput", module = "savant_rs.deepstream")]
pub struct PySkippedOutput {
    frame: VideoFrame,
    data: Option<Vec<u8>>,
    reason: PySkipReason,
}

#[pymethods]
impl PySkippedOutput {
    /// The rejected :class:`~savant_rs.primitives.VideoFrame`.
    #[getter]
    fn frame(&self) -> VideoFrame {
        self.frame.clone()
    }

    /// Raw payload bytes (if available).
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> Option<Bound<'py, pyo3::types::PyBytes>> {
        self.data.as_ref().map(|d| pyo3::types::PyBytes::new(py, d))
    }

    /// Why the frame was skipped.
    #[getter]
    fn reason(&self, py: Python<'_>) -> PyResult<Py<PySkipReason>> {
        Py::new(py, PySkipReason(self.reason.0.clone()))
    }

    fn __repr__(&self) -> String {
        format!("SkippedOutput(reason={:?})", self.reason.0)
    }
}

/// A decoded frame whose ``frame_id`` had no matching submitted
/// :class:`~savant_rs.primitives.VideoFrame`.
#[pyclass(name = "OrphanFrameOutput", module = "savant_rs.deepstream")]
pub struct PyOrphanFrameOutput(PyDecodedFrame);

#[pymethods]
impl PyOrphanFrameOutput {
    /// Decoded frame metadata.
    #[getter]
    fn decoded_frame(&self) -> PyDecodedFrame {
        PyDecodedFrame {
            frame_id: self.0.frame_id,
            pts_ns: self.0.pts_ns,
            dts_ns: self.0.dts_ns,
            duration_ns: self.0.duration_ns,
            codec: self.0.codec,
            format: self.0.format,
            has_buffer: self.0.has_buffer,
        }
    }

    fn __repr__(&self) -> String {
        "OrphanFrameOutput(...)".to_string()
    }
}

/// Logical per-source end-of-stream.
#[pyclass(name = "SourceEosOutput", module = "savant_rs.deepstream")]
pub struct PySourceEosOutput(String);

#[pymethods]
impl PySourceEosOutput {
    /// Source identifier.
    #[getter]
    fn source_id(&self) -> &str {
        &self.0
    }

    fn __repr__(&self) -> String {
        format!("SourceEosOutput(source_id={:?})", self.0)
    }
}

/// A GStreamer event captured at the pipeline output.
#[pyclass(name = "EventOutput", module = "savant_rs.deepstream")]
pub struct PyEventOutput(String);

#[pymethods]
impl PyEventOutput {
    /// Debug summary of the GStreamer event.
    #[getter]
    fn summary(&self) -> &str {
        &self.0
    }

    fn __repr__(&self) -> String {
        format!("EventOutput({})", self.0)
    }
}

/// An error from the underlying decoder.
#[pyclass(name = "ErrorOutput", module = "savant_rs.deepstream")]
pub struct PyErrorOutput(String);

#[pymethods]
impl PyErrorOutput {
    /// Error message.
    #[getter]
    fn message(&self) -> &str {
        &self.0
    }

    fn __repr__(&self) -> String {
        format!("ErrorOutput({})", self.0)
    }
}

// ─── FlexibleDecoderOutput (container) ──────────────────────────────────

enum OutputVariant {
    Frame(Py<PyFrameOutput>),
    ParameterChange(Py<PyParameterChangeOutput>),
    Skipped(Py<PySkippedOutput>),
    OrphanFrame(Py<PyOrphanFrameOutput>),
    SourceEos(Py<PySourceEosOutput>),
    Event(Py<PyEventOutput>),
    Error(Py<PyErrorOutput>),
}

/// Callback payload from :class:`FlexibleDecoder`.
///
/// Use the ``is_*`` properties to determine the variant, then call the
/// corresponding ``as_*`` method to get a typed output object.
///
/// For ``Frame`` outputs, call ``as_frame().take_delivery()`` to extract a
/// sealed ``(VideoFrame, SharedBuffer)`` pair.  When the :class:`FrameOutput`
/// is dropped (garbage-collected), the seal is released and downstream can
/// unseal.
#[pyclass(name = "FlexibleDecoderOutput", module = "savant_rs.deepstream")]
pub struct PyFlexibleDecoderOutput(OutputVariant);

impl PyFlexibleDecoderOutput {
    pub(crate) fn from_rust(py: Python<'_>, output: FlexibleDecoderOutput) -> PyResult<Self> {
        let variant = if matches!(output, FlexibleDecoderOutput::Frame { .. }) {
            OutputVariant::Frame(Py::new(py, PyFrameOutput(Some(output)))?)
        } else {
            match output {
                FlexibleDecoderOutput::Frame { .. } => unreachable!(),
                FlexibleDecoderOutput::ParameterChange { ref old, ref new } => {
                    OutputVariant::ParameterChange(Py::new(
                        py,
                        PyParameterChangeOutput {
                            old: PyDecoderParameters::from_rust(old),
                            new: PyDecoderParameters::from_rust(new),
                        },
                    )?)
                }
                FlexibleDecoderOutput::Skipped {
                    ref frame,
                    ref data,
                    ref reason,
                } => OutputVariant::Skipped(Py::new(
                    py,
                    PySkippedOutput {
                        frame: VideoFrame(frame.clone()),
                        data: data.clone(),
                        reason: PySkipReason::from_rust(reason),
                    },
                )?),
                FlexibleDecoderOutput::OrphanFrame { ref decoded } => OutputVariant::OrphanFrame(
                    Py::new(py, PyOrphanFrameOutput(PyDecodedFrame::from_rust(decoded)))?,
                ),
                FlexibleDecoderOutput::SourceEos { ref source_id } => {
                    OutputVariant::SourceEos(Py::new(py, PySourceEosOutput(source_id.clone()))?)
                }
                FlexibleDecoderOutput::Event(ref e) => {
                    OutputVariant::Event(Py::new(py, PyEventOutput(format!("{e:?}")))?)
                }
                FlexibleDecoderOutput::Error(ref e) => {
                    OutputVariant::Error(Py::new(py, PyErrorOutput(e.to_string()))?)
                }
            }
        };
        Ok(Self(variant))
    }
}

#[pymethods]
impl PyFlexibleDecoderOutput {
    // ── variant predicates ──

    #[getter]
    fn is_frame(&self) -> bool {
        matches!(self.0, OutputVariant::Frame(_))
    }
    #[getter]
    fn is_parameter_change(&self) -> bool {
        matches!(self.0, OutputVariant::ParameterChange(_))
    }
    #[getter]
    fn is_skipped(&self) -> bool {
        matches!(self.0, OutputVariant::Skipped(_))
    }
    #[getter]
    fn is_orphan_frame(&self) -> bool {
        matches!(self.0, OutputVariant::OrphanFrame(_))
    }
    #[getter]
    fn is_source_eos(&self) -> bool {
        matches!(self.0, OutputVariant::SourceEos(_))
    }
    #[getter]
    fn is_event(&self) -> bool {
        matches!(self.0, OutputVariant::Event(_))
    }
    #[getter]
    fn is_error(&self) -> bool {
        matches!(self.0, OutputVariant::Error(_))
    }

    // ── typed downcast methods ──

    /// Downcast to :class:`FrameOutput`, or ``None``.
    fn as_frame(&self, py: Python<'_>) -> Option<Py<PyFrameOutput>> {
        match &self.0 {
            OutputVariant::Frame(v) => Some(v.clone_ref(py)),
            _ => None,
        }
    }

    /// Downcast to :class:`ParameterChangeOutput`, or ``None``.
    fn as_parameter_change(&self, py: Python<'_>) -> Option<Py<PyParameterChangeOutput>> {
        match &self.0 {
            OutputVariant::ParameterChange(v) => Some(v.clone_ref(py)),
            _ => None,
        }
    }

    /// Downcast to :class:`SkippedOutput`, or ``None``.
    fn as_skipped(&self, py: Python<'_>) -> Option<Py<PySkippedOutput>> {
        match &self.0 {
            OutputVariant::Skipped(v) => Some(v.clone_ref(py)),
            _ => None,
        }
    }

    /// Downcast to :class:`OrphanFrameOutput`, or ``None``.
    fn as_orphan_frame(&self, py: Python<'_>) -> Option<Py<PyOrphanFrameOutput>> {
        match &self.0 {
            OutputVariant::OrphanFrame(v) => Some(v.clone_ref(py)),
            _ => None,
        }
    }

    /// Downcast to :class:`SourceEosOutput`, or ``None``.
    fn as_source_eos(&self, py: Python<'_>) -> Option<Py<PySourceEosOutput>> {
        match &self.0 {
            OutputVariant::SourceEos(v) => Some(v.clone_ref(py)),
            _ => None,
        }
    }

    /// Downcast to :class:`EventOutput`, or ``None``.
    fn as_event(&self, py: Python<'_>) -> Option<Py<PyEventOutput>> {
        match &self.0 {
            OutputVariant::Event(v) => Some(v.clone_ref(py)),
            _ => None,
        }
    }

    /// Downcast to :class:`ErrorOutput`, or ``None``.
    fn as_error(&self, py: Python<'_>) -> Option<Py<PyErrorOutput>> {
        match &self.0 {
            OutputVariant::Error(v) => Some(v.clone_ref(py)),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match &self.0 {
            OutputVariant::Frame(_) => "FlexibleDecoderOutput(Frame)".to_string(),
            OutputVariant::ParameterChange(_) => {
                "FlexibleDecoderOutput(ParameterChange)".to_string()
            }
            OutputVariant::Skipped(_) => "FlexibleDecoderOutput(Skipped)".to_string(),
            OutputVariant::OrphanFrame(_) => "FlexibleDecoderOutput(OrphanFrame)".to_string(),
            OutputVariant::SourceEos(_) => "FlexibleDecoderOutput(SourceEos)".to_string(),
            OutputVariant::Event(_) => "FlexibleDecoderOutput(Event)".to_string(),
            OutputVariant::Error(_) => "FlexibleDecoderOutput(Error)".to_string(),
        }
    }
}

// ─── FlexibleDecoder ────────────────────────────────────────────────────

const SHUT_DOWN_MSG: &str = "FlexibleDecoder is shut down";

/// Single-stream adaptive GPU decoder.
///
/// Wraps the Rust ``FlexibleDecoder`` and delivers all output through the
/// ``result_callback`` supplied at construction.
///
/// Args:
///     config (FlexibleDecoderConfig): Decoder configuration.
///     result_callback: ``Callable[[FlexibleDecoderOutput], None]`` invoked
///         for every decoded frame, parameter change, skip, EOS, or error.
#[pyclass(name = "FlexibleDecoder", module = "savant_rs.deepstream")]
pub struct PyFlexibleDecoder(Option<FlexibleDecoder>);

#[pymethods]
impl PyFlexibleDecoder {
    #[new]
    fn new(
        py: Python<'_>,
        config: &PyFlexibleDecoderConfig,
        result_callback: Py<PyAny>,
    ) -> PyResult<Self> {
        let rust_config = config.0.clone();

        let on_output: Arc<dyn Fn(FlexibleDecoderOutput) + Send + Sync> =
            Arc::new(move |output: FlexibleDecoderOutput| {
                Python::attach(|py| {
                    let py_output = match PyFlexibleDecoderOutput::from_rust(py, output) {
                        Ok(o) => o,
                        Err(e) => {
                            log::error!("FlexibleDecoder: failed to wrap output: {e}");
                            return;
                        }
                    };
                    if let Err(e) = result_callback.call1(py, (py_output,)) {
                        log::error!("FlexibleDecoder result_callback error: {e}");
                    }
                });
            });

        let decoder =
            py.detach(move || FlexibleDecoder::new(rust_config, move |out| on_output(out)));

        Ok(Self(Some(decoder)))
    }

    /// Submit an encoded frame for decoding.
    ///
    /// Args:
    ///     frame (VideoFrame): The video frame with metadata.
    ///     data (Optional[bytes]): Encoded payload.  If ``None``, the frame's
    ///         internal content is used.
    ///
    /// Raises:
    ///     RuntimeError: If the decoder is shut down or an infrastructure
    ///         error occurs.
    #[pyo3(signature = (frame, data=None))]
    fn submit(&self, py: Python<'_>, frame: &VideoFrame, data: Option<Vec<u8>>) -> PyResult<()> {
        let dec = self
            .0
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(SHUT_DOWN_MSG))?;
        let proxy = frame.0.clone();
        py.detach(move || {
            dec.submit(&proxy, data.as_deref())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Inject a logical per-source EOS.
    ///
    /// Args:
    ///     source_id (str): Source identifier.
    ///
    /// Raises:
    ///     RuntimeError: If the decoder is shut down.
    fn source_eos(&self, py: Python<'_>, source_id: &str) -> PyResult<()> {
        let dec = self
            .0
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(SHUT_DOWN_MSG))?;
        let source_id = source_id.to_string();
        py.detach(move || {
            dec.source_eos(&source_id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Drain the current decoder and shut down.
    ///
    /// Raises:
    ///     RuntimeError: If the decoder is already shut down or a drain
    ///         error occurs.
    fn graceful_shutdown(&mut self, py: Python<'_>) -> PyResult<()> {
        let dec = self
            .0
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(SHUT_DOWN_MSG))?;
        py.detach(move || {
            dec.graceful_shutdown()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Immediate teardown — frames in flight are lost.
    fn shutdown(&mut self, py: Python<'_>) -> PyResult<()> {
        let dec = self
            .0
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(SHUT_DOWN_MSG))?;
        py.detach(move || dec.shutdown());
        Ok(())
    }

    fn __repr__(&self) -> &'static str {
        if self.0.is_some() {
            "FlexibleDecoder(running)"
        } else {
            "FlexibleDecoder(shut_down)"
        }
    }
}

// ─── EvictionDecision ───────────────────────────────────────────────────

/// Decision returned by an eviction callback.
///
/// Use the class-level constants :attr:`EVICT` and :attr:`KEEP`.
#[pyclass(
    name = "EvictionDecision",
    module = "savant_rs.deepstream",
    from_py_object
)]
#[derive(Clone)]
pub struct PyEvictionDecision(deepstream_inputs::decoder_pool::EvictionDecision);

#[pymethods]
impl PyEvictionDecision {
    /// Remove the decoder from the pool.
    #[classattr]
    const EVICT: PyEvictionDecision =
        PyEvictionDecision(deepstream_inputs::decoder_pool::EvictionDecision::Evict);

    /// Keep the decoder alive (reset TTL).
    #[classattr]
    const KEEP: PyEvictionDecision =
        PyEvictionDecision(deepstream_inputs::decoder_pool::EvictionDecision::Keep);

    #[getter]
    fn is_evict(&self) -> bool {
        self.0 == deepstream_inputs::decoder_pool::EvictionDecision::Evict
    }

    #[getter]
    fn is_keep(&self) -> bool {
        self.0 == deepstream_inputs::decoder_pool::EvictionDecision::Keep
    }

    fn __repr__(&self) -> &'static str {
        match self.0 {
            deepstream_inputs::decoder_pool::EvictionDecision::Evict => "EvictionDecision.EVICT",
            deepstream_inputs::decoder_pool::EvictionDecision::Keep => "EvictionDecision.KEEP",
        }
    }

    fn __eq__(&self, other: &PyEvictionDecision) -> bool {
        self.0 == other.0
    }
}

// ─── FlexibleDecoderPoolConfig ──────────────────────────────────────────

/// Configuration for a :class:`FlexibleDecoderPool`.
///
/// Args:
///     gpu_id (int): GPU device ordinal.
///     pool_size (int): Number of RGBA buffers per internal decoder pool.
///     eviction_ttl_ms (int): Idle stream TTL in milliseconds before eviction
///         is considered.
#[pyclass(name = "FlexibleDecoderPoolConfig", module = "savant_rs.deepstream")]
pub struct PyFlexibleDecoderPoolConfig(
    pub(crate) deepstream_inputs::decoder_pool::FlexibleDecoderPoolConfig,
);

#[pymethods]
impl PyFlexibleDecoderPoolConfig {
    #[new]
    fn new(gpu_id: u32, pool_size: u32, eviction_ttl_ms: u64) -> Self {
        Self(
            deepstream_inputs::decoder_pool::FlexibleDecoderPoolConfig::new(
                gpu_id,
                pool_size,
                Duration::from_millis(eviction_ttl_ms),
            ),
        )
    }

    /// Set the idle timeout for graceful drain (milliseconds).  Returns a new
    /// config (builder pattern).
    fn with_idle_timeout_ms(&self, ms: u64) -> Self {
        Self(self.0.clone().idle_timeout(Duration::from_millis(ms)))
    }

    /// Set the maximum frames buffered during H.264/HEVC stream detection.
    fn with_detect_buffer_limit(&self, n: usize) -> Self {
        Self(self.0.clone().detect_buffer_limit(n))
    }

    #[getter]
    fn gpu_id(&self) -> u32 {
        self.0.gpu_id
    }

    #[getter]
    fn pool_size(&self) -> u32 {
        self.0.pool_size
    }

    #[getter]
    fn get_idle_timeout_ms(&self) -> u64 {
        self.0.idle_timeout.as_millis() as u64
    }

    #[getter]
    fn get_detect_buffer_limit(&self) -> usize {
        self.0.detect_buffer_limit
    }

    #[getter]
    fn get_eviction_ttl_ms(&self) -> u64 {
        self.0.eviction_ttl.as_millis() as u64
    }

    fn __repr__(&self) -> String {
        format!(
            "FlexibleDecoderPoolConfig(gpu_id={}, pool_size={}, \
             eviction_ttl_ms={}, idle_timeout_ms={}, detect_buffer_limit={})",
            self.0.gpu_id,
            self.0.pool_size,
            self.0.eviction_ttl.as_millis(),
            self.0.idle_timeout.as_millis(),
            self.0.detect_buffer_limit,
        )
    }
}

// ─── FlexibleDecoderPool ────────────────────────────────────────────────

const POOL_SHUT_DOWN_MSG: &str = "FlexibleDecoderPool is shut down";

/// Multi-stream pool of :class:`FlexibleDecoder` instances.
///
/// Routes incoming frames by ``source_id`` to per-stream decoders, creating
/// them on demand.  Idle streams are evicted after ``eviction_ttl_ms``.
///
/// Args:
///     config (FlexibleDecoderPoolConfig): Pool configuration.
///     result_callback: ``Callable[[FlexibleDecoderOutput], None]`` invoked
///         for every decoded output from any stream.
///     eviction_callback: Optional ``Callable[[str], EvictionDecision]``.
///         Called when a stream's TTL expires.  Return :attr:`EvictionDecision.KEEP`
///         to reset the TTL or :attr:`EvictionDecision.EVICT` to remove the stream.
///         When ``None``, all expired streams are evicted automatically.
#[pyclass(name = "FlexibleDecoderPool", module = "savant_rs.deepstream")]
pub struct PyFlexibleDecoderPool(Option<deepstream_inputs::decoder_pool::FlexibleDecoderPool>);

#[pymethods]
impl PyFlexibleDecoderPool {
    #[new]
    #[pyo3(signature = (config, result_callback, eviction_callback=None))]
    fn new(
        py: Python<'_>,
        config: &PyFlexibleDecoderPoolConfig,
        result_callback: Py<PyAny>,
        eviction_callback: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let rust_config = config.0.clone();

        let on_output: Arc<dyn Fn(FlexibleDecoderOutput) + Send + Sync> =
            Arc::new(move |output: FlexibleDecoderOutput| {
                Python::attach(|py| {
                    let py_output = match PyFlexibleDecoderOutput::from_rust(py, output) {
                        Ok(o) => o,
                        Err(e) => {
                            log::error!("FlexibleDecoderPool: failed to wrap output: {e}");
                            return;
                        }
                    };
                    if let Err(e) = result_callback.call1(py, (py_output,)) {
                        log::error!("FlexibleDecoderPool result_callback error: {e}");
                    }
                });
            });

        let pool = if let Some(eviction_cb) = eviction_callback {
            let on_eviction =
                move |source_id: &str| -> deepstream_inputs::decoder_pool::EvictionDecision {
                    Python::attach(|py| match eviction_cb.call1(py, (source_id,)) {
                        Ok(result) => match result.extract::<PyEvictionDecision>(py) {
                            Ok(decision) => decision.0,
                            Err(e) => {
                                log::error!(
                                        "FlexibleDecoderPool eviction_callback returned invalid type: {e}"
                                    );
                                deepstream_inputs::decoder_pool::EvictionDecision::Evict
                            }
                        },
                        Err(e) => {
                            log::error!("FlexibleDecoderPool eviction_callback error: {e}");
                            deepstream_inputs::decoder_pool::EvictionDecision::Evict
                        }
                    })
                };
            let on_out = Arc::clone(&on_output);
            py.detach(move || {
                deepstream_inputs::decoder_pool::FlexibleDecoderPool::with_eviction_callback(
                    rust_config,
                    move |out| on_out(out),
                    on_eviction,
                )
            })
        } else {
            let on_out = Arc::clone(&on_output);
            py.detach(move || {
                deepstream_inputs::decoder_pool::FlexibleDecoderPool::new(rust_config, move |out| {
                    on_out(out)
                })
            })
        };

        Ok(Self(Some(pool)))
    }

    /// Submit an encoded frame for decoding.
    ///
    /// The frame is routed to the per-stream decoder for
    /// ``frame.source_id``.  If none exists, one is created transparently.
    ///
    /// Args:
    ///     frame (VideoFrame): The video frame with metadata.
    ///     data (Optional[bytes]): Encoded payload.  If ``None``, the frame's
    ///         internal content is used.
    ///
    /// Raises:
    ///     RuntimeError: If the pool is shut down or an infrastructure
    ///         error occurs.
    #[pyo3(signature = (frame, data=None))]
    fn submit(&self, py: Python<'_>, frame: &VideoFrame, data: Option<Vec<u8>>) -> PyResult<()> {
        let pool = self
            .0
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(POOL_SHUT_DOWN_MSG))?;
        let proxy = frame.0.clone();
        py.detach(move || {
            pool.submit(&proxy, data.as_deref())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Inject a logical per-source EOS.
    ///
    /// Args:
    ///     source_id (str): Source identifier.
    ///
    /// Raises:
    ///     RuntimeError: If the pool is shut down.
    fn source_eos(&self, py: Python<'_>, source_id: &str) -> PyResult<()> {
        let pool = self
            .0
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(POOL_SHUT_DOWN_MSG))?;
        let source_id = source_id.to_string();
        py.detach(move || {
            pool.source_eos(&source_id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Drain every decoder in the pool and shut down.
    ///
    /// Raises:
    ///     RuntimeError: If the pool is already shut down.
    fn graceful_shutdown(&mut self, py: Python<'_>) -> PyResult<()> {
        let mut pool = self
            .0
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(POOL_SHUT_DOWN_MSG))?;
        py.detach(move || {
            pool.graceful_shutdown()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Immediate teardown — frames in flight are lost.
    fn shutdown(&mut self, py: Python<'_>) -> PyResult<()> {
        let mut pool = self
            .0
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err(POOL_SHUT_DOWN_MSG))?;
        py.detach(move || pool.shutdown());
        Ok(())
    }

    fn __repr__(&self) -> &'static str {
        if self.0.is_some() {
            "FlexibleDecoderPool(running)"
        } else {
            "FlexibleDecoderPool(shut_down)"
        }
    }
}

// ─── Module registration ────────────────────────────────────────────────

/// Register input adapter types on ``savant_rs.deepstream``.
pub fn register_inputs_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFlexibleDecoderConfig>()?;
    m.add_class::<PyDecoderParameters>()?;
    m.add_class::<PySkipReason>()?;
    m.add_class::<PyDecodedFrame>()?;
    m.add_class::<PySealedDelivery>()?;
    m.add_class::<PyFrameOutput>()?;
    m.add_class::<PyParameterChangeOutput>()?;
    m.add_class::<PySkippedOutput>()?;
    m.add_class::<PyOrphanFrameOutput>()?;
    m.add_class::<PySourceEosOutput>()?;
    m.add_class::<PyEventOutput>()?;
    m.add_class::<PyErrorOutput>()?;
    m.add_class::<PyFlexibleDecoderOutput>()?;
    m.add_class::<PyFlexibleDecoder>()?;
    m.add_class::<PyEvictionDecision>()?;
    m.add_class::<PyFlexibleDecoderPoolConfig>()?;
    m.add_class::<PyFlexibleDecoderPool>()?;
    Ok(())
}
