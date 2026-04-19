//! PyO3 wrapper for NvTracker.
//!
//! Thin bindings over [`deepstream_nvtracker::NvTracker`]: submit and pull outputs via
//! [`recv`](PyNvTracker::recv) / [`recv_timeout`](PyNvTracker::recv_timeout) /
//! [`try_recv`](PyNvTracker::try_recv), matching the Rust API.

use super::config::PyNvTrackerConfig;
use super::output::PyTrackerOutput;
use crate::deepstream::enums::{to_rust_id_kind, PySavantIdMetaKind};
use crate::deepstream::PySharedBuffer;
use crate::nvinfer::roi::PyRoi;
use gstreamer as gst;
use deepstream_nvtracker::{NvTracker, NvTrackerOutput, Roi as RustRoi, TrackedFrame};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Py;
use std::collections::HashMap;
use std::time::Duration;

type PyRoiHandle = Py<PyRoi>;

/// One frame to track: source name, single-surface NVMM buffer, and
/// detections keyed by ``class_id``.
///
/// Args:
///     source (str): Stream name (e.g. ``"cam-1"``).
///     buffer (SharedBuffer): Single-surface NVMM buffer.
///     rois (dict[int, list[Roi]]): Detections grouped by class_id.
#[pyclass(name = "TrackedFrame", module = "savant_rs.nvtracker")]
#[derive(Debug)]
pub struct PyTrackedFrame {
    pub(crate) source: String,
    pub(crate) buffer: Option<deepstream_buffers::SharedBuffer>,
    pub(crate) rois: HashMap<i32, Vec<RustRoi>>,
}

#[pymethods]
impl PyTrackedFrame {
    #[new]
    #[pyo3(signature = (source, buffer, rois))]
    fn new(
        py: Python<'_>,
        source: String,
        mut buffer: PyRefMut<'_, PySharedBuffer>,
        rois: HashMap<i32, Vec<PyRoiHandle>>,
    ) -> PyResult<Self> {
        let shared = buffer.take_inner()?;
        let mut rust_rois: HashMap<i32, Vec<RustRoi>> = HashMap::with_capacity(rois.len());
        for (class_id, py_rois) in rois {
            let mut rs = Vec::with_capacity(py_rois.len());
            for p in py_rois {
                let r = p.bind(py).borrow();
                let rr = r.to_rust();
                rs.push(RustRoi {
                    id: rr.id,
                    bbox: rr.bbox,
                });
            }
            rust_rois.insert(class_id, rs);
        }
        Ok(Self {
            source,
            buffer: Some(shared),
            rois: rust_rois,
        })
    }

    /// Stream name.
    #[getter]
    fn source(&self) -> &str {
        &self.source
    }

    fn __repr__(&self) -> String {
        let n_rois: usize = self.rois.values().map(|v| v.len()).sum();
        format!(
            "TrackedFrame(source={:?}, classes={}, rois={})",
            self.source,
            self.rois.len(),
            n_rois
        )
    }
}

fn extract_frames(frames: Vec<PyRefMut<'_, PyTrackedFrame>>) -> PyResult<Vec<TrackedFrame>> {
    let mut out = Vec::with_capacity(frames.len());
    for mut f in frames {
        let buf = f.buffer.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "TrackedFrame buffer already consumed or not set",
            )
        })?;
        out.push(TrackedFrame {
            source: f.source.clone(),
            buffer: buf,
            rois: f.rois.clone(),
        });
    }
    Ok(out)
}

enum PyNvTrackerOutputInner {
    Tracking(PyTrackerOutput),
    Event { summary: String },
    Eos { source_id: String },
    Error { message: String },
}

/// One item from [`PyNvTracker::recv`] / [`PyNvTracker::recv_timeout`] / [`PyNvTracker::try_recv`].
///
/// Discriminate with ``is_tracking``, ``is_event``, ``is_eos``, ``is_error`` and use
/// ``as_tracking()``, ``event_summary``, ``eos_source_id``, ``error_message`` accordingly.
#[pyclass(name = "NvTrackerOutput", module = "savant_rs.nvtracker")]
pub struct PyNvTrackerOutput {
    inner: PyNvTrackerOutputInner,
}

impl PyNvTrackerOutput {
    pub(crate) fn from_rust(output: NvTrackerOutput) -> Self {
        let inner = match output {
            NvTrackerOutput::Tracking(t) => {
                PyNvTrackerOutputInner::Tracking(PyTrackerOutput::from_rust(t))
            }
            NvTrackerOutput::Event(e) => PyNvTrackerOutputInner::Event {
                summary: format_gst_event(&e),
            },
            NvTrackerOutput::Eos { source_id } => PyNvTrackerOutputInner::Eos { source_id },
            NvTrackerOutput::Error(e) => PyNvTrackerOutputInner::Error {
                message: e.to_string(),
            },
        };
        Self { inner }
    }
}

fn format_gst_event(e: &gst::Event) -> String {
    format!("{e:?}")
}

#[pymethods]
impl PyNvTrackerOutput {
    #[getter]
    fn is_tracking(&self) -> bool {
        matches!(self.inner, PyNvTrackerOutputInner::Tracking(_))
    }

    #[getter]
    fn is_event(&self) -> bool {
        matches!(self.inner, PyNvTrackerOutputInner::Event { .. })
    }

    #[getter]
    fn is_eos(&self) -> bool {
        matches!(self.inner, PyNvTrackerOutputInner::Eos { .. })
    }

    #[getter]
    fn is_error(&self) -> bool {
        matches!(self.inner, PyNvTrackerOutputInner::Error { .. })
    }

    /// Return tracking payload if this is a tracking result, else ``None``.
    fn as_tracking(&self) -> Option<PyTrackerOutput> {
        match &self.inner {
            PyNvTrackerOutputInner::Tracking(t) => Some(t.clone()),
            _ => None,
        }
    }

    #[getter]
    fn event_summary(&self) -> Option<String> {
        match &self.inner {
            PyNvTrackerOutputInner::Event { summary } => Some(summary.clone()),
            _ => None,
        }
    }

    #[getter]
    fn eos_source_id(&self) -> Option<String> {
        match &self.inner {
            PyNvTrackerOutputInner::Eos { source_id } => Some(source_id.clone()),
            _ => None,
        }
    }

    #[getter]
    fn error_message(&self) -> Option<String> {
        match &self.inner {
            PyNvTrackerOutputInner::Error { message } => Some(message.clone()),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            PyNvTrackerOutputInner::Tracking(t) => {
                format!(
                    "NvTrackerOutput(Tracking(current_tracks={}))",
                    t.current_tracks.len()
                )
            }
            PyNvTrackerOutputInner::Event { summary } => {
                format!("NvTrackerOutput(Event({summary:?}))")
            }
            PyNvTrackerOutputInner::Eos { source_id } => {
                format!("NvTrackerOutput(Eos(source_id={source_id:?}))")
            }
            PyNvTrackerOutputInner::Error { message } => {
                format!("NvTrackerOutput(Error({message:?}))")
            }
        }
    }
}

/// DeepStream multi-object tracker pipeline.
///
/// Args:
///     config (NvTrackerConfig): Tracker configuration.
#[pyclass(name = "NvTracker", module = "savant_rs.nvtracker")]
pub struct PyNvTracker {
    inner: Option<NvTracker>,
}

#[pymethods]
impl PyNvTracker {
    #[new]
    fn new(py: Python<'_>, config: &PyNvTrackerConfig) -> PyResult<Self> {
        let rust_config = config.inner.clone();
        let engine = py.detach(|| {
            NvTracker::new(rust_config)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;
        Ok(Self {
            inner: Some(engine),
        })
    }

    /// Submit frames for tracking.
    ///
    /// Args:
    ///     frames: ``List[TrackedFrame]`` — per-frame source, buffer, and detections.
    ///     ids: ``List[Tuple[SavantIdMetaKind, int]]`` — per-frame Savant IDs.
    fn submit(
        &self,
        py: Python<'_>,
        frames: Vec<PyRefMut<'_, PyTrackedFrame>>,
        ids: Vec<(PySavantIdMetaKind, u128)>,
    ) -> PyResult<()> {
        let engine = self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("NvTracker is shut down"))?;
        let rust_frames = extract_frames(frames)?;
        let rust_ids = ids
            .into_iter()
            .map(|(kind, id)| to_rust_id_kind(kind, id))
            .collect();
        py.detach(|| {
            engine
                .submit(&rust_frames, rust_ids)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn recv(&self, py: Python<'_>) -> PyResult<PyNvTrackerOutput> {
        let engine = self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("NvTracker is shut down"))?;
        let out = py.detach(|| {
            engine
                .recv()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;
        Ok(PyNvTrackerOutput::from_rust(out))
    }

    fn recv_timeout(&self, py: Python<'_>, timeout_ms: u64) -> PyResult<Option<PyNvTrackerOutput>> {
        let engine = self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("NvTracker is shut down"))?;
        let timeout = Duration::from_millis(timeout_ms);
        let out = py.detach(|| {
            engine
                .recv_timeout(timeout)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;
        Ok(out.map(PyNvTrackerOutput::from_rust))
    }

    fn try_recv(&self, py: Python<'_>) -> PyResult<Option<PyNvTrackerOutput>> {
        let engine = self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("NvTracker is shut down"))?;
        let out = py.detach(|| {
            engine
                .try_recv()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;
        Ok(out.map(PyNvTrackerOutput::from_rust))
    }

    fn send_eos(&self, py: Python<'_>, source_id: &str) -> PyResult<()> {
        let engine = self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("NvTracker is shut down"))?;
        py.detach(|| {
            engine
                .send_eos(source_id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

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
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("NvTracker is shut down"))?;
        py.detach(|| {
            engine
                .send_event(event)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn is_failed(&self) -> PyResult<bool> {
        let engine = self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("NvTracker is shut down"))?;
        Ok(engine.is_failed())
    }

    fn reset_stream(&self, py: Python<'_>, source_id: &str) -> PyResult<()> {
        let engine = self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("NvTracker is shut down"))?;
        py.detach(|| {
            engine
                .reset_stream(source_id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    #[pyo3(signature = (timeout_ms))]
    fn graceful_shutdown(
        &mut self,
        py: Python<'_>,
        timeout_ms: u64,
    ) -> PyResult<Vec<Py<PyNvTrackerOutput>>> {
        let engine = self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("NvTracker is already shut down")
        })?;
        let outs = py.detach(|| {
            engine
                .graceful_shutdown(Duration::from_millis(timeout_ms))
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;
        outs.into_iter()
            .map(|o| Py::new(py, PyNvTrackerOutput::from_rust(o)))
            .collect()
    }

    fn shutdown(&mut self, py: Python<'_>) -> PyResult<()> {
        let engine = self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("NvTracker is already shut down")
        })?;
        py.detach(|| {
            engine
                .shutdown()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn __repr__(&self) -> &'static str {
        if self.inner.is_some() {
            "NvTracker(running)"
        } else {
            "NvTracker(shut_down)"
        }
    }
}
