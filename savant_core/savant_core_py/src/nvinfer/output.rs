//! PyO3 wrappers for nvinfer output types.
//!
//! `BatchInferenceOutput` owns the GStreamer sample that keeps tensor data
//! alive.  We store it in `Arc<Mutex<Option<...>>>` so that child objects
//! (`ElementOutput`, `TensorView`) can share the lifetime.

use super::enums::PyDataType;
use nvinfer::{BatchInferenceOutput, DataType};
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyBytes, PyList};
use std::sync::Arc;

type SharedOutput = Arc<Mutex<Option<BatchInferenceOutput>>>;

/// Tensor dimensions and total element count.
#[pyclass(name = "InferDims", module = "savant_rs.nvinfer", skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct PyInferDims {
    dims: Vec<u32>,
    num_elements: u32,
}

#[pymethods]
impl PyInferDims {
    /// Shape along each axis.
    #[getter]
    fn dimensions(&self) -> Vec<u32> {
        self.dims.clone()
    }

    /// Total number of elements (product of dimensions).
    #[getter]
    fn num_elements(&self) -> u32 {
        self.num_elements
    }

    fn __repr__(&self) -> String {
        format!(
            "InferDims(dimensions={:?}, num_elements={})",
            self.dims, self.num_elements
        )
    }
}

/// Zero-copy view into a single output tensor.
///
/// Valid while the parent ``BatchInferenceOutput`` is alive.  Call
/// ``as_bytes()`` or ``as_numpy()`` to copy data out.
#[pyclass(name = "TensorView", module = "savant_rs.nvinfer")]
pub struct PyTensorView {
    shared: SharedOutput,
    element_idx: usize,
    tensor_idx: usize,
    cached_name: String,
    cached_dims: PyInferDims,
    cached_data_type: PyDataType,
    cached_byte_length: usize,
}

#[pymethods]
impl PyTensorView {
    /// Output layer name.
    #[getter]
    fn name(&self) -> &str {
        &self.cached_name
    }

    /// Tensor dimensions.
    #[getter]
    fn dims(&self) -> PyInferDims {
        self.cached_dims.clone()
    }

    /// Data type of tensor elements.
    #[getter]
    fn data_type(&self) -> PyDataType {
        self.cached_data_type
    }

    /// Byte length of the tensor.
    #[getter]
    fn byte_length(&self) -> usize {
        self.cached_byte_length
    }

    /// Copy host-side tensor data as raw bytes.
    ///
    /// Returns:
    ///     bytes: Raw tensor data copied from host memory.
    ///
    /// Raises:
    ///     RuntimeError: If the parent ``BatchInferenceOutput`` has been
    ///         dropped or the host pointer is null.
    fn as_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let guard = self.shared.lock();
        let output = guard.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "BatchInferenceOutput has been released; tensor data is no longer valid",
            )
        })?;
        let tensor = &output.elements()[self.element_idx].tensors[self.tensor_idx];
        if tensor.host_ptr.is_null() || tensor.byte_length == 0 {
            return Ok(PyBytes::new(py, &[]));
        }
        let data =
            unsafe { std::slice::from_raw_parts(tensor.host_ptr as *const u8, tensor.byte_length) };
        Ok(PyBytes::new(py, data))
    }

    /// Copy host-side tensor data as a numpy array.
    ///
    /// The dtype is inferred from ``data_type``: FLOAT -> float32,
    /// HALF -> float16, INT8 -> int8, INT32 -> int32.
    ///
    /// Returns:
    ///     numpy.ndarray: 1-D numpy array with the tensor data.
    ///
    /// Raises:
    ///     RuntimeError: If the parent ``BatchInferenceOutput`` has been
    ///         dropped, the host pointer is null, or numpy is not available.
    fn as_numpy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let guard = self.shared.lock();
        let output = guard.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "BatchInferenceOutput has been released; tensor data is no longer valid",
            )
        })?;
        let tensor = &output.elements()[self.element_idx].tensors[self.tensor_idx];
        if tensor.host_ptr.is_null() || tensor.byte_length == 0 {
            let np = py.import("numpy")?;
            let empty = np.call_method1("array", (Vec::<f32>::new(),))?;
            return Ok(empty.unbind());
        }

        let np = py.import("numpy")?;
        let (dtype_str, elem_count) = match tensor.data_type {
            DataType::Float => ("float32", tensor.byte_length / 4),
            DataType::Half => ("float16", tensor.byte_length / 2),
            DataType::Int8 => ("int8", tensor.byte_length),
            DataType::Int32 => ("int32", tensor.byte_length / 4),
        };

        let raw =
            unsafe { std::slice::from_raw_parts(tensor.host_ptr as *const u8, tensor.byte_length) };
        let py_bytes = PyBytes::new(py, raw);

        let frombuffer = np.getattr("frombuffer")?;
        let dtype = np.getattr("dtype")?.call1((dtype_str,))?;
        let arr = frombuffer.call(
            (),
            Some(&[("buffer", py_bytes.as_any()), ("dtype", &dtype)].into_py_dict(py)?),
        )?;

        let _ = elem_count;
        Ok(arr.unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorView(name={:?}, dims={:?}, data_type={}, byte_length={})",
            self.cached_name,
            self.cached_dims.dims,
            self.cached_data_type.repr_str(),
            self.cached_byte_length,
        )
    }
}

/// Per-element inference output for one ROI in one frame.
#[pyclass(name = "ElementOutput", module = "savant_rs.nvinfer")]
pub struct PyElementOutput {
    shared: SharedOutput,
    element_idx: usize,
    cached_frame_id: Option<i64>,
    cached_roi_id: Option<i64>,
    cached_num_tensors: usize,
}

#[pymethods]
impl PyElementOutput {
    /// User-provided frame ID (if present).
    #[getter]
    fn frame_id(&self) -> Option<i64> {
        self.cached_frame_id
    }

    /// ROI identifier from ``Roi.id``.  ``None`` when no explicit ROIs were
    /// supplied and the full frame was used.
    #[getter]
    fn roi_id(&self) -> Option<i64> {
        self.cached_roi_id
    }

    /// Output tensors for this element.
    #[getter]
    fn tensors(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let guard = self.shared.lock();
        let output = guard.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("BatchInferenceOutput has been released")
        })?;
        let elem = &output.elements()[self.element_idx];
        let list = PyList::empty(py);
        for (tidx, tv) in elem.tensors.iter().enumerate() {
            let py_tv = PyTensorView {
                shared: self.shared.clone(),
                element_idx: self.element_idx,
                tensor_idx: tidx,
                cached_name: tv.name.clone(),
                cached_dims: PyInferDims {
                    dims: tv.dims.dimensions.clone(),
                    num_elements: tv.dims.num_elements,
                },
                cached_data_type: tv.data_type.into(),
                cached_byte_length: tv.byte_length,
            };
            let obj = Py::new(py, py_tv)?;
            list.append(obj)?;
        }
        Ok(list.unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "ElementOutput(frame_id={:?}, roi_id={:?}, num_tensors={})",
            self.cached_frame_id, self.cached_roi_id, self.cached_num_tensors,
        )
    }
}

/// Owns the GStreamer sample and exposes per-ROI inference outputs.
///
/// Tensor data remains valid as long as this object (or any child
/// ``TensorView``) is alive.
#[pyclass(name = "BatchInferenceOutput", module = "savant_rs.nvinfer")]
pub struct PyBatchInferenceOutput {
    shared: SharedOutput,
    cached_batch_id: u64,
    cached_num_elements: usize,
}

impl PyBatchInferenceOutput {
    /// Wrap a Rust `BatchInferenceOutput` for Python.
    pub(crate) fn from_rust(output: BatchInferenceOutput) -> Self {
        let batch_id = output.batch_id();
        let num_elements = output.num_elements();
        Self {
            shared: Arc::new(Mutex::new(Some(output))),
            cached_batch_id: batch_id,
            cached_num_elements: num_elements,
        }
    }
}

#[pymethods]
impl PyBatchInferenceOutput {
    /// User-provided batch ID.
    #[getter]
    fn batch_id(&self) -> u64 {
        self.cached_batch_id
    }

    /// Number of elements in the batch.
    #[getter]
    fn num_elements(&self) -> usize {
        self.cached_num_elements
    }

    /// Per-element outputs (one per ROI per frame).
    #[getter]
    fn elements(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let guard = self.shared.lock();
        let output = guard.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("BatchInferenceOutput has been released")
        })?;
        let list = PyList::empty(py);
        for (eidx, elem) in output.elements().iter().enumerate() {
            let py_elem = PyElementOutput {
                shared: self.shared.clone(),
                element_idx: eidx,
                cached_frame_id: elem.frame_id,
                cached_roi_id: elem.roi_id,
                cached_num_tensors: elem.tensors.len(),
            };
            let obj = Py::new(py, py_elem)?;
            list.append(obj)?;
        }
        Ok(list.unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "BatchInferenceOutput(batch_id={}, num_elements={})",
            self.cached_batch_id, self.cached_num_elements,
        )
    }
}
