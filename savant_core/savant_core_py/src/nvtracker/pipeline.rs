//! PyO3 wrapper for NvTracker.

use super::config::PyNvTrackerConfig;
use super::output::PyTrackerOutput;
use crate::deepstream::enums::{to_rust_id_kind, PySavantIdMetaKind};
use crate::deepstream::PySharedBuffer;
use crate::nvinfer::roi::PyRoi;
use nvtracker::{NvTracker, Roi as RustRoi, TrackedFrame};
use pyo3::prelude::*;
use std::collections::HashMap;

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

/// DeepStream multi-object tracker pipeline.
#[pyclass(name = "NvTracker", module = "savant_rs.nvtracker")]
pub struct PyNvTracker {
    inner: Option<NvTracker>,
}

#[pymethods]
impl PyNvTracker {
    #[new]
    fn new(py: Python<'_>, config: &PyNvTrackerConfig, callback: Py<PyAny>) -> PyResult<Self> {
        let rust_config = config.inner.clone();
        let rust_callback: nvtracker::TrackerCallback = Box::new(move |output| {
            Python::attach(|py| {
                let py_out = PyTrackerOutput::from_rust(output);
                if let Err(e) = callback.call1(py, (py_out,)) {
                    log::error!("NvTracker callback error: {e}");
                }
            });
        });
        let engine = py.detach(|| {
            NvTracker::new(rust_config, rust_callback)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;
        Ok(Self {
            inner: Some(engine),
        })
    }

    /// Asynchronous track.
    ///
    /// Args:
    ///     frames: ``List[TrackedFrame]`` — per-frame source, buffer, and detections.
    ///     ids: ``List[Tuple[SavantIdMetaKind, int]]`` — per-frame Savant IDs.
    fn track(
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
                .track(&rust_frames, rust_ids)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Synchronous track (blocks until result or ``operation_timeout``).
    fn track_sync(
        &self,
        py: Python<'_>,
        frames: Vec<PyRefMut<'_, PyTrackedFrame>>,
        ids: Vec<(PySavantIdMetaKind, u128)>,
    ) -> PyResult<PyTrackerOutput> {
        let engine = self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("NvTracker is shut down"))?;
        let rust_frames = extract_frames(frames)?;
        let rust_ids = ids
            .into_iter()
            .map(|(kind, id)| to_rust_id_kind(kind, id))
            .collect();
        let output = py.detach(|| {
            engine
                .track_sync(&rust_frames, rust_ids)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;
        Ok(PyTrackerOutput::from_rust(output))
    }

    /// Send stream-reset for ``source_id`` (hashed to pad index).
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

    fn shutdown(&mut self, py: Python<'_>) -> PyResult<()> {
        let mut engine = self.inner.take().ok_or_else(|| {
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
