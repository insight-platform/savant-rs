//! PyO3 wrapper for the NvInfer inference engine.

use super::config::PyNvInferConfig;
use super::output::PyBatchInferenceOutput;
use super::roi::PyRoi;
use crate::deepstream::{extract_gst_buffer, PySharedBuffer};
use nvinfer::{NvInfer, Roi};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;

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

/// The NvInfer inference engine.
///
/// Wraps a DeepStream ``nvinfer`` element in an ``appsrc -> [queue] ->
/// nvinfer -> appsink`` GStreamer pipeline.  Supports both asynchronous
/// (callback) and synchronous (``infer_sync``) inference.
///
/// Args:
///     config (NvInferConfig): Engine configuration.
///     callback (Callable[[BatchInferenceOutput], None]): Callback invoked
///         when asynchronous inference completes.
#[pyclass(name = "NvInfer", module = "savant_rs.nvinfer")]
pub struct PyNvInfer {
    inner: Option<NvInfer>,
}

#[pymethods]
impl PyNvInfer {
    #[new]
    fn new(py: Python<'_>, config: &PyNvInferConfig, callback: Py<PyAny>) -> PyResult<Self> {
        let rust_config = config.inner.clone();
        let rust_callback: nvinfer::pipeline::InferCallback = Box::new(move |output| {
            Python::attach(|py| {
                let py_output = PyBatchInferenceOutput::from_rust(output);
                if let Err(e) = callback.call1(py, (py_output,)) {
                    log::error!("NvInfer callback error: {e}");
                }
            });
        });
        let engine = py.detach(|| {
            NvInfer::new(rust_config, rust_callback)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;
        Ok(Self {
            inner: Some(engine),
        })
    }

    /// Submit a batched buffer for asynchronous inference.
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

    /// Synchronous inference -- blocks until results arrive (up to 30 s).
    ///
    /// The buffer is **consumed**: the ``SharedBuffer``
    /// becomes invalid after this call.
    ///
    /// Args:
    ///     batch (Union[SharedBuffer, int]): Batched NvBufSurface buffer.
    ///     rois (Optional[Dict[int, List[Roi]]]): Per-slot ROI lists.
    ///
    /// Returns:
    ///     BatchInferenceOutput: Inference results.
    ///
    /// Raises:
    ///     RuntimeError: If the engine has been shut down, submission fails,
    ///         or inference times out.
    #[pyo3(signature = (batch, rois=None))]
    fn infer_sync(
        &self,
        py: Python<'_>,
        batch: &Bound<'_, PyAny>,
        rois: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyBatchInferenceOutput> {
        let engine = self
            .inner
            .as_ref()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("NvInfer is shut down"))?;
        let shared = take_shared_buffer(batch)?;
        let rust_rois = rois.map(extract_rois).transpose()?;
        let output = py.detach(|| {
            engine
                .infer_sync(shared, rust_rois.as_ref())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })?;
        Ok(PyBatchInferenceOutput::from_rust(output))
    }

    /// Graceful shutdown: send EOS, drain, stop pipeline.
    ///
    /// Raises:
    ///     RuntimeError: If the engine has already been shut down.
    fn shutdown(&mut self, py: Python<'_>) -> PyResult<()> {
        let mut engine = self.inner.take().ok_or_else(|| {
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
