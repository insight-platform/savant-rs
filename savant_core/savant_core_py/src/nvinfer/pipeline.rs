//! PyO3 wrapper for the NvInfer inference engine.
//!
//! Thin bindings over [`deepstream_nvinfer::NvInfer`]: submit and pull outputs via
//! [`recv`](PyNvInfer::recv) / [`recv_timeout`](PyNvInfer::recv_timeout) /
//! [`try_recv`](PyNvInfer::try_recv), matching the Rust API.

use super::config::PyNvInferConfig;
use super::output::PyBatchInferenceOutput;
use super::roi::PyRoi;
use crate::deepstream::{extract_gst_buffer, PySharedBuffer};
use gstreamer as gst;
use deepstream_nvinfer::{NvInfer, NvInferOutput, Roi};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Py;
use std::collections::HashMap;
use std::time::Duration;

/// Extract a `SharedBuffer` from a Python buffer argument.
///
/// If `batch` is a `SharedBuffer`, we consume it via `take_inner()`.
/// If it is a raw pointer (`int`), we fall through to `extract_gst_buffer`
/// which uses `from_glib_full` (already refcount 1).
pub(super) fn take_shared_buffer(
    batch: &Bound<'_, PyAny>,
) -> PyResult<deepstream_buffers::SharedBuffer> {
    if let Ok(mut sb) = batch.extract::<PyRefMut<'_, PySharedBuffer>>() {
        return sb.take_inner();
    }
    let buf = extract_gst_buffer(batch)?;
    Ok(deepstream_buffers::SharedBuffer::from(buf))
}

/// Extract ``Dict[int, List[Roi]]`` from a Python object into Rust.
fn extract_rois(obj: &Bound<'_, PyAny>) -> PyResult<HashMap<u32, Vec<Roi>>> {
    let dict = obj.cast::<PyDict>()?;
    let mut map = HashMap::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let slot: u32 = key.extract()?;
        let py_list: Vec<PyRef<'_, PyRoi>> = value.extract()?;
        let rois: Vec<Roi> = py_list.iter().map(|r| r.to_rust()).collect();
        map.insert(slot, rois);
    }
    Ok(map)
}

enum PyNvInferOutputInner {
    Inference(PyBatchInferenceOutput),
    /// Opaque summary of a downstream GStreamer event (caps, stream-start, etc.).
    Event {
        summary: String,
    },
    Eos {
        source_id: String,
    },
    Error {
        message: String,
    },
}

/// One item from [`NvInfer::recv`] / [`NvInfer::recv_timeout`] / [`NvInfer::try_recv`].
///
/// Discriminate with ``is_inference``, ``is_event``, ``is_eos``, ``is_error`` and use
/// ``as_inference()``, ``event_summary``, ``eos_source_id``, ``error_message`` accordingly.
#[pyclass(name = "NvInferOutput", module = "savant_rs.nvinfer")]
pub struct PyNvInferOutput {
    inner: PyNvInferOutputInner,
}

impl PyNvInferOutput {
    pub(crate) fn from_rust(output: NvInferOutput) -> Self {
        let inner = match output {
            NvInferOutput::Inference(b) => {
                PyNvInferOutputInner::Inference(PyBatchInferenceOutput::from_rust(b))
            }
            NvInferOutput::Event(e) => PyNvInferOutputInner::Event {
                summary: format_gst_event(&e),
            },
            NvInferOutput::Eos { source_id } => PyNvInferOutputInner::Eos { source_id },
            NvInferOutput::Error(e) => PyNvInferOutputInner::Error {
                message: e.to_string(),
            },
        };
        Self { inner }
    }
}

fn format_gst_event(e: &gst::Event) -> String {
    format!("{:?}", e)
}

#[pymethods]
impl PyNvInferOutput {
    #[getter]
    fn is_inference(&self) -> bool {
        matches!(self.inner, PyNvInferOutputInner::Inference(_))
    }

    #[getter]
    fn is_event(&self) -> bool {
        matches!(self.inner, PyNvInferOutputInner::Event { .. })
    }

    #[getter]
    fn is_eos(&self) -> bool {
        matches!(self.inner, PyNvInferOutputInner::Eos { .. })
    }

    #[getter]
    fn is_error(&self) -> bool {
        matches!(self.inner, PyNvInferOutputInner::Error { .. })
    }

    /// Return inference payload if this is an inference result, else ``None``.
    ///
    /// The returned object shares the same underlying batch as this output's inference
    /// payload (Arc-backed); tensor views remain valid while either handle lives.
    fn as_inference(&self) -> Option<PyBatchInferenceOutput> {
        match &self.inner {
            PyNvInferOutputInner::Inference(b) => Some(b.share()),
            _ => None,
        }
    }

    /// Summary string for a GStreamer event output, or ``None``.
    #[getter]
    fn event_summary(&self) -> Option<String> {
        match &self.inner {
            PyNvInferOutputInner::Event { summary } => Some(summary.clone()),
            _ => None,
        }
    }

    /// Source id for logical per-source EOS, or ``None``.
    #[getter]
    fn eos_source_id(&self) -> Option<String> {
        match &self.inner {
            PyNvInferOutputInner::Eos { source_id } => Some(source_id.clone()),
            _ => None,
        }
    }

    /// Error message for pipeline/framework errors, or ``None``.
    #[getter]
    fn error_message(&self) -> Option<String> {
        match &self.inner {
            PyNvInferOutputInner::Error { message } => Some(message.clone()),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            PyNvInferOutputInner::Inference(b) => {
                format!(
                    "NvInferOutput(Inference(num_elements={}))",
                    b.batch_num_elements()
                )
            }
            PyNvInferOutputInner::Event { summary } => {
                format!("NvInferOutput(Event({summary:?}))")
            }
            PyNvInferOutputInner::Eos { source_id } => {
                format!("NvInferOutput(Eos(source_id={source_id:?}))")
            }
            PyNvInferOutputInner::Error { message } => {
                format!("NvInferOutput(Error({message:?}))")
            }
        }
    }
}

/// The NvInfer inference engine.
///
/// Wraps a DeepStream ``nvinfer`` element in an ``appsrc -> [queue] ->
/// nvinfer -> appsink`` GStreamer pipeline.  Outputs are **pulled** with
/// [`recv`](Self::recv), [`recv_timeout`](Self::recv_timeout), or
/// [`try_recv`](Self::try_recv), matching the Rust [`deepstream_nvinfer::NvInfer`] API.
///
/// Args:
///     config (NvInferConfig): Engine configuration.
#[pyclass(name = "NvInfer", module = "savant_rs.nvinfer")]
pub struct PyNvInfer {
    inner: Option<NvInfer>,
}

#[pymethods]
impl PyNvInfer {
    #[new]
    fn new(py: Python<'_>, config: &PyNvInferConfig) -> PyResult<Self> {
        let rust_config = config.inner.clone();
        let engine = py.detach(|| {
            NvInfer::new(rust_config)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;
        Ok(Self {
            inner: Some(engine),
        })
    }

    /// Submit a batched buffer for inference.
    ///
    /// The buffer is **consumed**: the ``SharedBuffer``
    /// becomes invalid after this call.
    ///
    /// Args:
    ///     batch (Union[SharedBuffer, int]): Batched NvBufSurface buffer.
    ///     rois (Optional[Dict[int, List[Roi]]]): Per-slot ROI lists.
    ///
    /// Raises:
    ///     RuntimeError: If the engine has been shut down or submission fails.
    #[pyo3(signature = (batch, rois=None))]
    fn submit(
        &self,
        py: Python<'_>,
        batch: &Bound<'_, PyAny>,
        rois: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let engine = self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("NvInfer is shut down"))?;
        let shared = take_shared_buffer(batch)?;
        let rust_rois = rois.map(extract_rois).transpose()?;
        py.detach(|| {
            engine
                .submit(shared, rust_rois.as_ref())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Block until the next output item is available.
    ///
    /// Raises:
    ///     RuntimeError: On channel disconnect ([`deepstream_nvinfer::NvInferError::ChannelDisconnected`]).
    fn recv(&self, py: Python<'_>) -> PyResult<PyNvInferOutput> {
        let engine = self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("NvInfer is shut down"))?;
        let out = py.detach(|| {
            engine
                .recv()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;
        Ok(PyNvInferOutput::from_rust(out))
    }

    /// Block until the next output or ``timeout_ms`` elapses.
    ///
    /// Returns:
    ///     Optional[NvInferOutput]: ``None`` on timeout.
    ///
    /// Raises:
    ///     RuntimeError: On channel disconnect.
    fn recv_timeout(&self, py: Python<'_>, timeout_ms: u64) -> PyResult<Option<PyNvInferOutput>> {
        let engine = self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("NvInfer is shut down"))?;
        let timeout = Duration::from_millis(timeout_ms);
        let out = py.detach(|| {
            engine
                .recv_timeout(timeout)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;
        Ok(out.map(PyNvInferOutput::from_rust))
    }

    /// Non-blocking: return the next output if available.
    ///
    /// Returns:
    ///     Optional[NvInferOutput]: ``None`` if no output is ready.
    ///
    /// Raises:
    ///     RuntimeError: On channel disconnect.
    fn try_recv(&self, py: Python<'_>) -> PyResult<Option<PyNvInferOutput>> {
        let engine = self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("NvInfer is shut down"))?;
        let out = py.detach(|| {
            engine
                .try_recv()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;
        Ok(out.map(PyNvInferOutput::from_rust))
    }

    /// Send a logical per-source EOS marker downstream (custom event).
    fn send_eos(&self, py: Python<'_>, source_id: &str) -> PyResult<()> {
        let engine = self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("NvInfer is shut down"))?;
        py.detach(|| {
            engine
                .send_eos(source_id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Send a custom downstream GStreamer event (string fields only).
    ///
    /// Wraps Rust ``NvInfer::send_event`` with ``gst::event::CustomDownstream``.
    /// For other event types, use the Rust API.
    #[pyo3(signature = (structure_name, string_fields=None))]
    fn send_custom_downstream_event(
        &self,
        py: Python<'_>,
        structure_name: &str,
        string_fields: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let mut b = gst::Structure::builder(structure_name);
        if let Some(d) = string_fields {
            for (k, v) in d.iter() {
                let key: String = k.extract()?;
                let val: String = v.extract()?;
                b = b.field(key.as_str(), val.as_str());
            }
        }
        let structure = b.build();
        let event = gst::event::CustomDownstream::new(structure);
        let engine = self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("NvInfer is shut down"))?;
        py.detach(|| {
            engine
                .send_event(event)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Whether the pipeline has entered a terminal failed state.
    fn is_failed(&self) -> PyResult<bool> {
        let engine = self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("NvInfer is shut down"))?;
        Ok(engine.is_failed())
    }

    /// Graceful shutdown: reject new input, drain outputs within ``timeout_ms``, stop pipeline.
    ///
    /// Returns a list of :class:`NvInferOutput` items (excluding terminal pipeline EOS).
    ///
    /// Raises:
    ///     RuntimeError: If the engine has already been shut down.
    #[pyo3(signature = (timeout_ms))]
    fn graceful_shutdown(
        &mut self,
        py: Python<'_>,
        timeout_ms: u64,
    ) -> PyResult<Vec<Py<PyNvInferOutput>>> {
        let engine = self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("NvInfer is already shut down")
        })?;
        let outs = py.detach(|| {
            engine
                .graceful_shutdown(Duration::from_millis(timeout_ms))
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;
        outs.into_iter()
            .map(|o| Py::new(py, PyNvInferOutput::from_rust(o)))
            .collect()
    }

    /// Abrupt shutdown: stop pipeline without draining.
    ///
    /// Raises:
    ///     RuntimeError: If the engine has already been shut down.
    fn shutdown(&mut self, py: Python<'_>) -> PyResult<()> {
        let engine = self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("NvInfer is already shut down")
        })?;
        py.detach(|| {
            engine
                .shutdown()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn __repr__(&self) -> &'static str {
        if self.inner.is_some() {
            "NvInfer(running)"
        } else {
            "NvInfer(shut_down)"
        }
    }
}
