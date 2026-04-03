//! Python bindings for the multi-stream GPU decoder (`deepstream_inputs` crate).

use crate::deepstream::enums::PyInterpolation;
use crate::deepstream::PySharedBuffer;
use crate::gstreamer::PyCodec;
use crate::primitives::eos::EndOfStream;
use crate::primitives::frame::VideoFrame;
use deepstream_inputs::multistream_decoder::{
    DecoderOutput, EvictionVerdict, MultiStreamDecoder, MultiStreamDecoderConfig, MultiStreamError,
    SessionBoundaryEosPolicy, StopReason, SubmitResult, UndecodedReason,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::time::Duration;

fn multistream_error_to_py(e: MultiStreamError) -> PyErr {
    match e {
        MultiStreamError::NoData => PyValueError::new_err(e.to_string()),
        MultiStreamError::QueueFull {
            source_id,
            queue_size,
        } => PyRuntimeError::new_err(format!(
            "QUEUE_FULL source_id={source_id} queue_size={queue_size}"
        )),
        MultiStreamError::UnknownStream(s) => PyValueError::new_err(format!("unknown stream: {s}")),
        MultiStreamError::ChannelDisconnected(s) => {
            PyRuntimeError::new_err(format!("channel disconnected: {s}"))
        }
        _ => PyRuntimeError::new_err(e.to_string()),
    }
}

/// Register multi-stream decoder types on ``savant_rs.deepstream``.
pub fn register_inputs_classes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMultiStreamDecoderConfig>()?;
    m.add_class::<PySubmitResult>()?;
    m.add_class::<PyEvictionVerdict>()?;
    m.add_class::<PyStopReason>()?;
    m.add_class::<PyUndecodedReason>()?;
    m.add_class::<PyDecoderOutput>()?;
    m.add_class::<PyMultiStreamDecoder>()?;
    Ok(())
}

// ─── Config ─────────────────────────────────────────────────────────────

#[pyclass(
    from_py_object,
    name = "MultiStreamDecoderConfig",
    module = "savant_rs.deepstream"
)]
#[derive(Clone)]
pub struct PyMultiStreamDecoderConfig {
    pub(crate) inner: MultiStreamDecoderConfig,
}

#[pymethods]
impl PyMultiStreamDecoderConfig {
    #[new]
    #[pyo3(signature = (
        gpu_id,
        output_pool_size,
        interpolation = PyInterpolation::Bilinear,
        idle_timeout_ms = 30_000u64,
        max_detection_buffer = 30usize,
        per_stream_queue_size = 16usize,
        session_eos_on_codec_change = None,
        session_eos_on_resolution_change = None,
        session_eos_on_timestamp_regress = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        gpu_id: u32,
        output_pool_size: u32,
        interpolation: PyInterpolation,
        idle_timeout_ms: u64,
        max_detection_buffer: usize,
        per_stream_queue_size: usize,
        session_eos_on_codec_change: Option<bool>,
        session_eos_on_resolution_change: Option<bool>,
        session_eos_on_timestamp_regress: Option<bool>,
    ) -> PyResult<Self> {
        let mut inner = MultiStreamDecoderConfig::new(gpu_id, output_pool_size);
        inner.interpolation = interpolation.into();
        inner.idle_timeout = Duration::from_millis(idle_timeout_ms);
        inner.max_detection_buffer = max_detection_buffer;
        inner.per_stream_queue_size = per_stream_queue_size;
        let mut sess = SessionBoundaryEosPolicy::default();
        if let Some(v) = session_eos_on_codec_change {
            sess.on_codec_change = v;
        }
        if let Some(v) = session_eos_on_resolution_change {
            sess.on_resolution_change = v;
        }
        if let Some(v) = session_eos_on_timestamp_regress {
            sess.on_timestamp_regress = v;
        }
        inner.session_boundary_eos = sess;
        Ok(Self { inner })
    }

    #[getter]
    fn gpu_id(&self) -> u32 {
        self.inner.gpu_id
    }
    #[getter]
    fn output_pool_size(&self) -> u32 {
        self.inner.output_pool_size
    }
    #[getter]
    fn interpolation(&self) -> PyInterpolation {
        self.inner.interpolation.into()
    }
    #[getter]
    fn idle_timeout_ms(&self) -> u64 {
        self.inner.idle_timeout.as_millis() as u64
    }
    #[getter]
    fn max_detection_buffer(&self) -> usize {
        self.inner.max_detection_buffer
    }
    #[getter]
    fn per_stream_queue_size(&self) -> usize {
        self.inner.per_stream_queue_size
    }
    #[getter]
    fn session_eos_on_codec_change(&self) -> bool {
        self.inner.session_boundary_eos.on_codec_change
    }
    #[getter]
    fn session_eos_on_resolution_change(&self) -> bool {
        self.inner.session_boundary_eos.on_resolution_change
    }
    #[getter]
    fn session_eos_on_timestamp_regress(&self) -> bool {
        self.inner.session_boundary_eos.on_timestamp_regress
    }
}

// ─── SubmitResult ────────────────────────────────────────────────────────

#[pyclass(name = "SubmitResult", module = "savant_rs.deepstream")]
pub struct PySubmitResult {
    #[pyo3(get)]
    pub queue_depth: usize,
}

impl From<SubmitResult> for PySubmitResult {
    fn from(s: SubmitResult) -> Self {
        Self {
            queue_depth: s.queue_depth,
        }
    }
}

// ─── EvictionVerdict ─────────────────────────────────────────────────────

#[pyclass(
    from_py_object,
    name = "EvictionVerdict",
    module = "savant_rs.deepstream"
)]
#[derive(Clone)]
pub struct PyEvictionVerdict {
    pub(crate) inner: EvictionVerdict,
}

#[pymethods]
impl PyEvictionVerdict {
    #[staticmethod]
    fn approve() -> Self {
        Self {
            inner: EvictionVerdict::Approve,
        }
    }

    #[staticmethod]
    fn extend(timeout_ms: u64) -> Self {
        Self {
            inner: EvictionVerdict::Extend(Duration::from_millis(timeout_ms)),
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            EvictionVerdict::Approve => "EvictionVerdict.approve()".to_string(),
            EvictionVerdict::Extend(d) => format!("EvictionVerdict.extend({} ms)", d.as_millis()),
        }
    }
}

// ─── StopReason ──────────────────────────────────────────────────────────

#[pyclass(name = "StopReason", module = "savant_rs.deepstream")]
pub struct PyStopReason {
    pub(crate) inner: StopReason,
}

#[pymethods]
impl PyStopReason {
    fn is_eos(&self) -> bool {
        matches!(self.inner, StopReason::Eos)
    }
    fn is_idle_eviction(&self) -> bool {
        matches!(self.inner, StopReason::IdleEviction)
    }
    fn is_codec_changed(&self) -> bool {
        matches!(self.inner, StopReason::CodecChanged)
    }
    fn is_resolution_changed(&self) -> bool {
        matches!(self.inner, StopReason::ResolutionChanged)
    }
    fn is_timestamp_regressed(&self) -> bool {
        matches!(self.inner, StopReason::TimestampRegressed)
    }
    fn is_shutdown(&self) -> bool {
        matches!(self.inner, StopReason::Shutdown)
    }
    fn error_message(&self) -> Option<String> {
        match &self.inner {
            StopReason::Error(s) => Some(s.clone()),
            _ => None,
        }
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

// ─── UndecodedReason ─────────────────────────────────────────────────────

#[pyclass(name = "UndecodedReason", module = "savant_rs.deepstream")]
pub struct PyUndecodedReason {
    pub(crate) inner: UndecodedReason,
}

#[pymethods]
impl PyUndecodedReason {
    fn is_unsupported_codec(&self) -> bool {
        matches!(self.inner, UndecodedReason::UnsupportedCodec(_))
    }
    /// If this is ``UnsupportedCodec(Some(name))``, returns ``Some(name)``.
    /// If ``UnsupportedCodec(None)``, returns ``Some(None)`` in Python as ``None`` for inner — use ``is_unsupported_codec`` plus this.
    fn unsupported_codec_name(&self) -> Option<Option<String>> {
        match &self.inner {
            UndecodedReason::UnsupportedCodec(s) => Some(s.clone()),
            _ => None,
        }
    }
    fn is_awaiting_keyframe(&self) -> bool {
        matches!(self.inner, UndecodedReason::AwaitingKeyframe)
    }
    fn detection_failed_message(&self) -> Option<String> {
        match &self.inner {
            UndecodedReason::DetectionFailed(s) => Some(s.clone()),
            _ => None,
        }
    }
    fn decode_error_message(&self) -> Option<String> {
        match &self.inner {
            UndecodedReason::DecodeError(s) => Some(s.clone()),
            _ => None,
        }
    }
    fn is_stream_evicted(&self) -> bool {
        matches!(self.inner, UndecodedReason::StreamEvicted)
    }
    fn is_session_reset(&self) -> bool {
        matches!(self.inner, UndecodedReason::SessionReset)
    }
    fn is_no_payload(&self) -> bool {
        matches!(self.inner, UndecodedReason::NoPayload)
    }
    fn is_external_content(&self) -> bool {
        matches!(self.inner, UndecodedReason::ExternalContent)
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

// ─── DecoderOutput ───────────────────────────────────────────────────────

#[pyclass(name = "DecoderOutput", module = "savant_rs.deepstream")]
pub struct PyDecoderOutput {
    inner: DecoderOutput,
}

impl PyDecoderOutput {
    pub(crate) fn new(inner: DecoderOutput) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyDecoderOutput {
    fn is_decoded(&self) -> bool {
        matches!(self.inner, DecoderOutput::Decoded { .. })
    }
    fn is_undecoded(&self) -> bool {
        matches!(self.inner, DecoderOutput::Undecoded { .. })
    }
    fn is_eos(&self) -> bool {
        matches!(self.inner, DecoderOutput::Eos { .. })
    }
    fn is_stream_started(&self) -> bool {
        matches!(self.inner, DecoderOutput::StreamStarted { .. })
    }
    fn is_stream_stopped(&self) -> bool {
        matches!(self.inner, DecoderOutput::StreamStopped { .. })
    }
    fn is_pipeline_restarted(&self) -> bool {
        matches!(self.inner, DecoderOutput::PipelineRestarted { .. })
    }

    /// Returns ``(VideoFrame, SharedBuffer)``.
    fn as_decoded(&self) -> PyResult<(VideoFrame, PySharedBuffer)> {
        match &self.inner {
            DecoderOutput::Decoded { frame, buffer } => Ok((
                VideoFrame(frame.clone()),
                PySharedBuffer::from_rust(buffer.clone()),
            )),
            _ => Err(PyValueError::new_err("not Decoded")),
        }
    }

    /// Returns ``(VideoFrame, Optional[bytes], UndecodedReason)``.
    fn as_undecoded(
        &self,
        py: Python<'_>,
    ) -> PyResult<(VideoFrame, Option<Py<PyAny>>, PyUndecodedReason)> {
        match &self.inner {
            DecoderOutput::Undecoded {
                frame,
                data,
                reason,
            } => {
                let b = data
                    .as_ref()
                    .map(|v| Py::from(PyBytes::new(py, v.as_slice())));
                Ok((
                    VideoFrame(frame.clone()),
                    b,
                    PyUndecodedReason {
                        inner: reason.clone(),
                    },
                ))
            }
            _ => Err(PyValueError::new_err("not Undecoded")),
        }
    }

    fn as_eos(&self) -> PyResult<String> {
        match &self.inner {
            DecoderOutput::Eos { source_id } => Ok(source_id.clone()),
            _ => Err(PyValueError::new_err("not Eos")),
        }
    }

    fn as_stream_started(&self) -> PyResult<(String, PyCodec)> {
        match &self.inner {
            DecoderOutput::StreamStarted { source_id, codec } => {
                Ok((source_id.clone(), PyCodec::from(*codec)))
            }
            _ => Err(PyValueError::new_err("not StreamStarted")),
        }
    }

    fn as_stream_stopped(&self) -> PyResult<(String, PyStopReason)> {
        match &self.inner {
            DecoderOutput::StreamStopped { source_id, reason } => Ok((
                source_id.clone(),
                PyStopReason {
                    inner: reason.clone(),
                },
            )),
            _ => Err(PyValueError::new_err("not StreamStopped")),
        }
    }

    fn as_pipeline_restarted(&self) -> PyResult<(String, String, usize)> {
        match &self.inner {
            DecoderOutput::PipelineRestarted {
                source_id,
                reason,
                lost_frame_count,
            } => Ok((source_id.clone(), reason.clone(), *lost_frame_count)),
            _ => Err(PyValueError::new_err("not PipelineRestarted")),
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

// ─── MultiStreamDecoder ──────────────────────────────────────────────────

#[pyclass(name = "MultiStreamDecoder", module = "savant_rs.deepstream")]
pub struct PyMultiStreamDecoder {
    inner: Option<MultiStreamDecoder>,
}

#[pymethods]
impl PyMultiStreamDecoder {
    #[new]
    #[pyo3(signature = (config, on_output, on_eviction=None))]
    fn new(
        py: Python<'_>,
        config: PyRef<'_, PyMultiStreamDecoderConfig>,
        on_output: Py<PyAny>,
        on_eviction: Option<Py<PyAny>>,
    ) -> Self {
        let cfg = config.inner.clone();
        let on_out = on_output.clone_ref(py);
        let rust_out = move |o: DecoderOutput| {
            Python::attach(|py| {
                let py_o = PyDecoderOutput::new(o);
                if let Err(e) = on_out.call1(py, (py_o,)) {
                    log::error!("MultiStreamDecoder on_output error: {e}");
                }
            });
        };
        let rust_ev = on_eviction.map(|ev_py| {
            let ev = ev_py.clone_ref(py);
            move |sid: &str| {
                Python::attach(|py| match ev.call1(py, (sid,)) {
                    Ok(obj) => {
                        let bound = obj.into_bound(py);
                        if let Ok(v) = bound.cast::<PyEvictionVerdict>() {
                            return v.borrow().inner.clone();
                        }
                        EvictionVerdict::Approve
                    }
                    Err(e) => {
                        log::error!("on_eviction error: {e}");
                        EvictionVerdict::Approve
                    }
                })
            }
        });
        let inner = MultiStreamDecoder::new(cfg, rust_out, rust_ev);
        Self { inner: Some(inner) }
    }

    #[pyo3(signature = (frame, data=None, timeout_ms=5000u64))]
    fn submit(
        &self,
        py: Python<'_>,
        frame: &VideoFrame,
        data: Option<Bound<'_, PyBytes>>,
        timeout_ms: u64,
    ) -> PyResult<PySubmitResult> {
        let dec = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("MultiStreamDecoder is shut down"))?;
        let f = frame.0.clone();
        let slice = data.as_ref().map(|b| b.as_bytes());
        py.detach(|| {
            dec.submit(f, slice, Duration::from_millis(timeout_ms))
                .map(PySubmitResult::from)
                .map_err(multistream_error_to_py)
        })
    }

    #[pyo3(signature = (frame, data=None))]
    fn try_submit(
        &self,
        frame: &VideoFrame,
        data: Option<Bound<'_, PyBytes>>,
    ) -> PyResult<PySubmitResult> {
        let dec = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("MultiStreamDecoder is shut down"))?;
        let f = frame.0.clone();
        let slice = data.as_ref().map(|b| b.as_bytes());
        dec.try_submit(f, slice)
            .map(PySubmitResult::from)
            .map_err(multistream_error_to_py)
    }

    #[pyo3(signature = (eos, timeout_ms=5000u64))]
    fn submit_eos(&self, py: Python<'_>, eos: &EndOfStream, timeout_ms: u64) -> PyResult<bool> {
        let dec = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("MultiStreamDecoder is shut down"))?;
        py.detach(|| {
            Ok(dec
                .submit_eos(&eos.0, Duration::from_millis(timeout_ms))
                .is_ok())
        })
    }

    fn try_submit_eos(&self, eos: &EndOfStream) -> PyResult<bool> {
        let dec = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("MultiStreamDecoder is shut down"))?;
        Ok(dec.try_submit_eos(&eos.0).is_ok())
    }

    fn active_streams(&self) -> PyResult<Vec<String>> {
        let dec = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("MultiStreamDecoder is shut down"))?;
        Ok(dec.active_streams())
    }

    fn stream_count(&self) -> PyResult<usize> {
        let dec = self
            .inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("MultiStreamDecoder is shut down"))?;
        Ok(dec.stream_count())
    }

    fn shutdown(&mut self, py: Python<'_>) {
        if let Some(mut d) = self.inner.take() {
            py.detach(|| {
                d.shutdown();
            });
        }
    }

    fn __repr__(&self) -> String {
        if self.inner.is_some() {
            "MultiStreamDecoder(running)".to_string()
        } else {
            "MultiStreamDecoder(shut_down)".to_string()
        }
    }
}
