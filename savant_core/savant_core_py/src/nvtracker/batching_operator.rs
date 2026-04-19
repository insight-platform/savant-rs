//! PyO3 bindings for the [`NvTrackerBatchingOperator`](deepstream_nvtracker::NvTrackerBatchingOperator).

use super::config::PyNvTrackerConfig;
use super::output::{PyMiscTrackData, PyTrackedObject};
use crate::deepstream::enums::{from_rust_id_kind, to_rust_id_kind, PySavantIdMetaKind};
use crate::deepstream::{extract_gst_buffer, PySharedBuffer};
use crate::nvinfer::roi::PyRoi;
use crate::primitives::frame::VideoFrame;
use deepstream_buffers::SavantIdMetaKind;
use deepstream_nvtracker::{
    NvTrackerBatchingOperator, NvTrackerBatchingOperatorConfig, SealedDeliveries,
    TrackerBatchFormationCallback, TrackerBatchFormationResult, TrackerOperatorOutput,
    TrackerOperatorTrackingOutput,
};
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::Py;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// Extract a `SharedBuffer` from a Python buffer argument.
fn take_shared_buffer(batch: &Bound<'_, PyAny>) -> PyResult<deepstream_buffers::SharedBuffer> {
    if let Ok(mut sb) = batch.extract::<PyRefMut<'_, PySharedBuffer>>() {
        return sb.take_inner();
    }
    let buf = extract_gst_buffer(batch)?;
    Ok(deepstream_buffers::SharedBuffer::from(buf))
}

type SharedTrackingBatch = Arc<Mutex<Option<TrackerOperatorTrackingOutput>>>;
type PyRoiHandle = Py<PyRoi>;

/// Configuration for the NvTrackerBatchingOperator batching layer.
#[pyclass(
    name = "NvTrackerBatchingOperatorConfig",
    module = "savant_rs.nvtracker"
)]
pub struct PyNvTrackerBatchingOperatorConfig {
    pub(crate) inner: NvTrackerBatchingOperatorConfig,
}

#[pymethods]
impl PyNvTrackerBatchingOperatorConfig {
    #[new]
    #[pyo3(signature = (max_batch_size, max_batch_wait_ms, nvtracker_config, *, pending_batch_timeout_ms=60000))]
    fn new(
        max_batch_size: usize,
        max_batch_wait_ms: u64,
        nvtracker_config: &PyNvTrackerConfig,
        pending_batch_timeout_ms: u64,
    ) -> Self {
        Self {
            inner: NvTrackerBatchingOperatorConfig {
                max_batch_size,
                max_batch_wait: Duration::from_millis(max_batch_wait_ms),
                nvtracker: nvtracker_config.inner.clone(),
                pending_batch_timeout: Duration::from_millis(pending_batch_timeout_ms),
            },
        }
    }

    #[getter]
    fn max_batch_size(&self) -> usize {
        self.inner.max_batch_size
    }

    #[getter]
    fn max_batch_wait_ms(&self) -> u64 {
        self.inner.max_batch_wait.as_millis() as u64
    }

    /// Pending batch timeout (milliseconds).
    #[getter]
    fn pending_batch_timeout_ms(&self) -> u64 {
        self.inner.pending_batch_timeout.as_millis() as u64
    }

    #[getter]
    fn nvtracker_config(&self) -> PyNvTrackerConfig {
        PyNvTrackerConfig {
            inner: self.inner.nvtracker.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "NvTrackerBatchingOperatorConfig(max_batch_size={}, max_batch_wait_ms={}, gpu_id={})",
            self.inner.max_batch_size,
            self.inner.max_batch_wait.as_millis(),
            self.inner.nvtracker.gpu_id,
        )
    }
}

/// Result returned by the batch formation callback.
#[pyclass(name = "TrackerBatchFormationResult", module = "savant_rs.nvtracker")]
pub struct PyTrackerBatchFormationResult {
    pub(super) inner: TrackerBatchFormationResult,
}

#[pymethods]
impl PyTrackerBatchFormationResult {
    #[new]
    fn new(
        py: Python<'_>,
        ids: Vec<(PySavantIdMetaKind, u128)>,
        rois: Vec<HashMap<i32, Vec<PyRoiHandle>>>,
    ) -> Self {
        let rust_ids: Vec<SavantIdMetaKind> = ids
            .into_iter()
            .map(|(kind, id)| to_rust_id_kind(kind, id))
            .collect();

        let rust_rois = rois
            .into_iter()
            .map(|class_map| {
                class_map
                    .into_iter()
                    .map(|(class_id, py_rois)| {
                        let converted = py_rois
                            .into_iter()
                            .map(|r| {
                                let roi = r.bind(py).borrow().to_rust();
                                deepstream_nvtracker::Roi {
                                    id: roi.id,
                                    bbox: roi.bbox,
                                }
                            })
                            .collect::<Vec<_>>();
                        (class_id, converted)
                    })
                    .collect::<HashMap<_, _>>()
            })
            .collect::<Vec<_>>();

        Self {
            inner: TrackerBatchFormationResult {
                ids: rust_ids,
                rois: rust_rois,
            },
        }
    }

    /// Per-frame Savant IDs as ``(SavantIdMetaKind, int)`` tuples.
    #[getter]
    fn ids(&self) -> Vec<(PySavantIdMetaKind, u128)> {
        self.inner.ids.iter().map(from_rust_id_kind).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "TrackerBatchFormationResult(ids_len={}, rois_len={})",
            self.inner.ids.len(),
            self.inner.rois.len(),
        )
    }
}

/// Per-frame tracking result (callback view — no direct buffer access).
#[pyclass(name = "TrackerOperatorFrameOutput", module = "savant_rs.nvtracker")]
pub struct PyTrackerOperatorFrameOutput {
    shared: SharedTrackingBatch,
    slot_idx: usize,
    frame: savant_core::primitives::frame::VideoFrameProxy,
    num_tracks: usize,
}

#[pymethods]
impl PyTrackerOperatorFrameOutput {
    /// The original ``VideoFrame`` submitted for this frame.
    #[getter]
    fn frame(&self) -> VideoFrame {
        VideoFrame(self.frame.clone())
    }

    /// Current tracked objects for this frame.
    #[getter]
    fn tracked_objects(&self) -> PyResult<Vec<PyTrackedObject>> {
        let guard = self.shared.lock();
        let output = guard.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("TrackerOperatorOutput has been released")
        })?;
        Ok(output.frames()[self.slot_idx]
            .tracked_objects
            .iter()
            .cloned()
            .map(PyTrackedObject::from)
            .collect())
    }

    /// Shadow tracks relevant to this frame source.
    #[getter]
    fn shadow_tracks(&self) -> PyResult<Vec<PyMiscTrackData>> {
        let guard = self.shared.lock();
        let output = guard.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("TrackerOperatorOutput has been released")
        })?;
        Ok(output.frames()[self.slot_idx]
            .shadow_tracks
            .iter()
            .cloned()
            .map(PyMiscTrackData::from)
            .collect())
    }

    /// Terminated tracks relevant to this frame source.
    #[getter]
    fn terminated_tracks(&self) -> PyResult<Vec<PyMiscTrackData>> {
        let guard = self.shared.lock();
        let output = guard.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("TrackerOperatorOutput has been released")
        })?;
        Ok(output.frames()[self.slot_idx]
            .terminated_tracks
            .iter()
            .cloned()
            .map(PyMiscTrackData::from)
            .collect())
    }

    /// Past-frame data relevant to this frame source.
    #[getter]
    fn past_frame_data(&self) -> PyResult<Vec<PyMiscTrackData>> {
        let guard = self.shared.lock();
        let output = guard.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("TrackerOperatorOutput has been released")
        })?;
        Ok(output.frames()[self.slot_idx]
            .past_frame_data
            .iter()
            .cloned()
            .map(PyMiscTrackData::from)
            .collect())
    }

    fn __repr__(&self) -> String {
        format!(
            "TrackerOperatorFrameOutput(source={:?}, num_tracks={})",
            self.frame.get_source_id(),
            self.num_tracks,
        )
    }
}

/// A batch of ``(VideoFrame, SharedBuffer)`` pairs sealed until the
/// associated ``TrackerOperatorOutput`` is dropped.
#[pyclass(name = "SealedDeliveries", module = "savant_rs.nvtracker")]
pub struct PySealedDeliveries {
    inner: Option<SealedDeliveries>,
}

impl PySealedDeliveries {
    pub(crate) fn from_rust(sealed: SealedDeliveries) -> Self {
        Self {
            inner: Some(sealed),
        }
    }

    fn ensure_alive(&self) -> PyResult<&SealedDeliveries> {
        self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("SealedDeliveries already consumed")
        })
    }
}

#[pymethods]
impl PySealedDeliveries {
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.ensure_alive()?.len())
    }

    fn is_empty(&self) -> PyResult<bool> {
        Ok(self.ensure_alive()?.is_empty())
    }

    fn is_released(&self) -> PyResult<bool> {
        Ok(self.ensure_alive()?.is_released())
    }

    #[pyo3(signature = (timeout_ms=None))]
    fn unseal(&mut self, py: Python<'_>, timeout_ms: Option<u64>) -> PyResult<Py<PyList>> {
        let sealed = self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("SealedDeliveries already consumed")
        })?;
        let pairs = match timeout_ms {
            Some(ms) => {
                let timeout = Duration::from_millis(ms);
                match py.detach(move || sealed.unseal_timeout(timeout)) {
                    Ok(pairs) => pairs,
                    Err(still_sealed) => {
                        self.inner = Some(still_sealed);
                        return Err(pyo3::exceptions::PyTimeoutError::new_err(
                            "unseal timed out",
                        ));
                    }
                }
            }
            None => py.detach(move || sealed.unseal()),
        };
        let list = PyList::empty(py);
        for (frame, buffer) in pairs {
            let py_frame = VideoFrame(frame);
            let py_buf = PySharedBuffer::from_rust(buffer);
            list.append((py_frame.into_pyobject(py)?, py_buf.into_pyobject(py)?))?;
        }
        Ok(list.unbind())
    }

    fn try_unseal(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyList>>> {
        let sealed = self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("SealedDeliveries already consumed")
        })?;
        match sealed.try_unseal() {
            Ok(pairs) => {
                let list = PyList::empty(py);
                for (frame, buffer) in pairs {
                    let py_frame = VideoFrame(frame);
                    let py_buf = PySharedBuffer::from_rust(buffer);
                    list.append((py_frame.into_pyobject(py)?, py_buf.into_pyobject(py)?))?;
                }
                Ok(Some(list.unbind()))
            }
            Err(still_sealed) => {
                self.inner = Some(still_sealed);
                Ok(None)
            }
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            Some(s) => format!(
                "SealedDeliveries(len={}, released={})",
                s.len(),
                s.is_released()
            ),
            None => "SealedDeliveries(consumed)".to_string(),
        }
    }
}

enum PyTrackerOperatorOutputInner {
    Batch {
        shared: SharedTrackingBatch,
        num_frames: usize,
    },
    Eos {
        source_id: String,
    },
    Error {
        message: String,
    },
}

/// Callback payload from :class:`NvTrackerBatchingOperator`.
#[pyclass(name = "TrackerOperatorOutput", module = "savant_rs.nvtracker")]
pub struct PyTrackerOperatorOutput {
    inner: PyTrackerOperatorOutputInner,
}

impl PyTrackerOperatorOutput {
    pub(crate) fn from_rust(output: TrackerOperatorOutput) -> Self {
        let inner = match output {
            TrackerOperatorOutput::Tracking(t) => {
                let num_frames = t.frames().len();
                PyTrackerOperatorOutputInner::Batch {
                    shared: Arc::new(Mutex::new(Some(t))),
                    num_frames,
                }
            }
            TrackerOperatorOutput::Eos { source_id } => {
                PyTrackerOperatorOutputInner::Eos { source_id }
            }
            TrackerOperatorOutput::Error(e) => PyTrackerOperatorOutputInner::Error {
                message: e.to_string(),
            },
        };
        Self { inner }
    }
}

#[pymethods]
impl PyTrackerOperatorOutput {
    #[getter]
    fn is_tracking(&self) -> bool {
        matches!(self.inner, PyTrackerOperatorOutputInner::Batch { .. })
    }

    #[getter]
    fn is_eos(&self) -> bool {
        matches!(self.inner, PyTrackerOperatorOutputInner::Eos { .. })
    }

    #[getter]
    fn is_error(&self) -> bool {
        matches!(self.inner, PyTrackerOperatorOutputInner::Error { .. })
    }

    #[getter]
    fn eos_source_id(&self) -> Option<String> {
        match &self.inner {
            PyTrackerOperatorOutputInner::Eos { source_id } => Some(source_id.clone()),
            _ => None,
        }
    }

    #[getter]
    fn error_message(&self) -> Option<String> {
        match &self.inner {
            PyTrackerOperatorOutputInner::Error { message } => Some(message.clone()),
            _ => None,
        }
    }

    /// Per-frame outputs (tracking results only — no direct buffer access).
    #[getter]
    fn frames(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let list = PyList::empty(py);
        if let PyTrackerOperatorOutputInner::Batch { shared, .. } = &self.inner {
            let guard = shared.lock();
            let output = guard.as_ref().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("TrackerOperatorOutput has been released")
            })?;
            for (idx, frame_output) in output.frames().iter().enumerate() {
                let py_frame = PyTrackerOperatorFrameOutput {
                    shared: shared.clone(),
                    slot_idx: idx,
                    frame: frame_output.frame.clone(),
                    num_tracks: frame_output.tracked_objects.len(),
                };
                list.append(Py::new(py, py_frame)?)?;
            }
        }
        Ok(list.unbind())
    }

    /// Number of frames in the batch (0 for EOS / error).
    #[getter]
    fn num_frames(&self) -> usize {
        match &self.inner {
            PyTrackerOperatorOutputInner::Batch { num_frames, .. } => *num_frames,
            _ => 0,
        }
    }

    /// Extract sealed deliveries while keeping tracking data alive.
    ///
    /// Returns a :class:`SealedDeliveries` on the first call.
    /// Subsequent calls return ``None``.
    fn take_deliveries(&self) -> PyResult<Option<PySealedDeliveries>> {
        match &self.inner {
            PyTrackerOperatorOutputInner::Batch { shared, .. } => {
                let mut guard = shared.lock();
                let output = guard.as_mut().ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err(
                        "TrackerOperatorOutput has been released",
                    )
                })?;
                Ok(output.take_deliveries().map(PySealedDeliveries::from_rust))
            }
            _ => Ok(None),
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            PyTrackerOperatorOutputInner::Batch { num_frames, .. } => {
                format!("TrackerOperatorOutput(Tracking(num_frames={num_frames}))")
            }
            PyTrackerOperatorOutputInner::Eos { source_id } => {
                format!("TrackerOperatorOutput(Eos(source_id={source_id:?}))")
            }
            PyTrackerOperatorOutputInner::Error { message } => {
                format!("TrackerOperatorOutput(Error({message:?}))")
            }
        }
    }
}

/// Higher-level batching layer over ``NvTracker``.
#[pyclass(name = "NvTrackerBatchingOperator", module = "savant_rs.nvtracker")]
pub struct PyNvTrackerBatchingOperator {
    inner: Option<NvTrackerBatchingOperator>,
}

#[pymethods]
impl PyNvTrackerBatchingOperator {
    #[new]
    fn new(
        py: Python<'_>,
        config: &PyNvTrackerBatchingOperatorConfig,
        batch_formation_callback: Py<PyAny>,
        result_callback: Py<PyAny>,
    ) -> PyResult<Self> {
        let rust_config = config.inner.clone();

        let batch_cb: TrackerBatchFormationCallback = Arc::new(move |frames| {
            Python::attach(|py| {
                let py_frames: Vec<VideoFrame> =
                    frames.iter().map(|f| VideoFrame(f.clone())).collect();

                let result = match batch_formation_callback.call1(py, (py_frames,)) {
                    Ok(r) => r,
                    Err(e) => {
                        log::error!("batch_formation_callback raised: {e}");
                        return TrackerBatchFormationResult {
                            ids: vec![],
                            rois: vec![],
                        };
                    }
                };

                let bound = result.bind(py);
                match bound.extract::<PyRef<'_, PyTrackerBatchFormationResult>>() {
                    Ok(py_result) => TrackerBatchFormationResult {
                        ids: py_result.inner.ids.clone(),
                        rois: py_result.inner.rois.clone(),
                    },
                    Err(e) => {
                        log::error!(
                            "batch_formation_callback must return TrackerBatchFormationResult: {e}"
                        );
                        TrackerBatchFormationResult {
                            ids: vec![],
                            rois: vec![],
                        }
                    }
                }
            })
        });

        let result_cb: deepstream_nvtracker::TrackerOperatorResultCallback =
            Box::new(move |output: TrackerOperatorOutput| {
                Python::attach(|py| {
                    let py_output = PyTrackerOperatorOutput::from_rust(output);
                    if let Err(e) = result_callback.call1(py, (py_output,)) {
                        log::error!("NvTrackerBatchingOperator result_callback error: {e}");
                    }
                });
            });

        let operator = py.detach(|| {
            NvTrackerBatchingOperator::new(rust_config, batch_cb, result_cb)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;

        Ok(Self {
            inner: Some(operator),
        })
    }

    /// Add a single frame for batched tracking.
    fn add_frame(
        &self,
        py: Python<'_>,
        frame: &VideoFrame,
        buffer: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let op = self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("NvTrackerBatchingOperator is shut down")
        })?;
        let shared_buf = take_shared_buffer(buffer)?;
        let proxy = frame.0.clone();
        py.detach(|| {
            op.add_frame(proxy, shared_buf)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Submit the current partial batch immediately (if non-empty).
    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        let op = self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("NvTrackerBatchingOperator is shut down")
        })?;
        py.detach(|| {
            op.flush()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Send stream-reset for ``source_id``.
    fn reset_stream(&self, py: Python<'_>, source_id: &str) -> PyResult<()> {
        let op = self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("NvTrackerBatchingOperator is shut down")
        })?;
        py.detach(|| {
            op.reset_stream(source_id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Forward logical per-source EOS to the inner tracker.
    fn send_eos(&self, py: Python<'_>, source_id: &str) -> PyResult<()> {
        let op = self.inner.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("NvTrackerBatchingOperator is shut down")
        })?;
        py.detach(|| {
            op.send_eos(source_id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    #[pyo3(signature = (timeout_ms))]
    fn graceful_shutdown(
        &mut self,
        py: Python<'_>,
        timeout_ms: u64,
    ) -> PyResult<Vec<Py<PyTrackerOperatorOutput>>> {
        let mut op = self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "NvTrackerBatchingOperator is already shut down",
            )
        })?;
        let outs = py.detach(|| {
            op.graceful_shutdown(Duration::from_millis(timeout_ms))
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;
        outs.into_iter()
            .map(|o| Py::new(py, PyTrackerOperatorOutput::from_rust(o)))
            .collect()
    }

    /// Flush pending frames, stop timer thread, and shut down NvTracker.
    fn shutdown(&mut self, py: Python<'_>) -> PyResult<()> {
        let mut op = self.inner.take().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "NvTrackerBatchingOperator is already shut down",
            )
        })?;
        py.detach(|| {
            op.shutdown()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn __repr__(&self) -> &'static str {
        if self.inner.is_some() {
            "NvTrackerBatchingOperator(running)"
        } else {
            "NvTrackerBatchingOperator(shut_down)"
        }
    }
}
