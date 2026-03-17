//! PyO3 wrappers for nvinfer output types.
//!
//! `BatchInferenceOutput` owns the GStreamer sample that keeps tensor data
//! alive.  We store it in `Arc<Mutex<Option<...>>>` so that child objects
//! (`ElementOutput`, `TensorView`) can share the lifetime.

use super::enums::PyDataType;
use nvinfer::BatchInferenceOutput;
use parking_lot::Mutex;
use pyo3::prelude::*;
use pyo3::types::PyList;
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
/// Exposes ``host_ptr`` and ``device_ptr`` as plain integer addresses so
/// that Python callers can construct framework-native tensors (NumPy, CuPy,
/// PyTorch) without any data copy on the Rust side.
///
/// Valid while the parent ``BatchInferenceOutput`` is alive.
#[pyclass(name = "TensorView", module = "savant_rs.nvinfer")]
pub struct PyTensorView {
    #[allow(dead_code)]
    shared: SharedOutput,
    name: String,
    dims: PyInferDims,
    data_type: PyDataType,
    byte_length: usize,
    host_ptr: usize,
    device_ptr: usize,
}

#[pymethods]
impl PyTensorView {
    /// Output layer name.
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// Tensor dimensions.
    #[getter]
    fn dims(&self) -> PyInferDims {
        self.dims.clone()
    }

    /// Data type of tensor elements.
    #[getter]
    fn data_type(&self) -> PyDataType {
        self.data_type
    }

    /// Byte length of the tensor.
    #[getter]
    fn byte_length(&self) -> usize {
        self.byte_length
    }

    /// Host (CPU) memory address of the tensor data, or 0 if unavailable.
    #[getter]
    fn host_ptr(&self) -> usize {
        self.host_ptr
    }

    /// Device (GPU) memory address of the tensor data, or 0 if unavailable.
    #[getter]
    fn device_ptr(&self) -> usize {
        self.device_ptr
    }

    /// NumPy-compatible dtype string (``"float32"``, ``"float16"``,
    /// ``"int8"``, ``"int32"``).
    #[getter]
    fn numpy_dtype(&self) -> &'static str {
        match self.data_type {
            PyDataType::Float => "float32",
            PyDataType::Half => "float16",
            PyDataType::Int8 => "int8",
            PyDataType::Int32 => "int32",
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorView(name={:?}, dims={:?}, data_type={}, byte_length={}, \
             host_ptr=0x{:x}, device_ptr=0x{:x})",
            self.name,
            self.dims.dims,
            self.data_type.repr_str(),
            self.byte_length,
            self.host_ptr,
            self.device_ptr,
        )
    }
}

/// Per-element inference output for one ROI in one frame.
#[pyclass(name = "ElementOutput", module = "savant_rs.nvinfer")]
pub struct PyElementOutput {
    shared: SharedOutput,
    element_idx: usize,
    frame_id: Option<i64>,
    roi_id: Option<i64>,
    num_tensors: usize,
}

#[pymethods]
impl PyElementOutput {
    /// User-provided frame ID (if present).
    #[getter]
    fn frame_id(&self) -> Option<i64> {
        self.frame_id
    }

    /// ROI identifier from ``Roi.id``.  ``None`` when no explicit ROIs were
    /// supplied and the full frame was used.
    #[getter]
    fn roi_id(&self) -> Option<i64> {
        self.roi_id
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
        for tv in elem.tensors.iter() {
            let py_tv = PyTensorView {
                shared: self.shared.clone(),
                name: tv.name.clone(),
                dims: PyInferDims {
                    dims: tv.dims.dimensions.clone(),
                    num_elements: tv.dims.num_elements,
                },
                data_type: tv.data_type.into(),
                byte_length: tv.byte_length,
                host_ptr: tv.host_ptr as usize,
                device_ptr: tv.device_ptr as usize,
            };
            let obj = Py::new(py, py_tv)?;
            list.append(obj)?;
        }
        Ok(list.unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "ElementOutput(frame_id={:?}, roi_id={:?}, num_tensors={})",
            self.frame_id, self.roi_id, self.num_tensors,
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
    num_elements: usize,
}

impl PyBatchInferenceOutput {
    /// Wrap a Rust `BatchInferenceOutput` for Python.
    pub(crate) fn from_rust(output: BatchInferenceOutput) -> Self {
        let num_elements = output.num_elements();
        Self {
            shared: Arc::new(Mutex::new(Some(output))),
            num_elements,
        }
    }
}

#[pymethods]
impl PyBatchInferenceOutput {
    /// Number of elements in the batch.
    #[getter]
    fn num_elements(&self) -> usize {
        self.num_elements
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
                frame_id: elem.frame_id,
                roi_id: elem.roi_id,
                num_tensors: elem.tensors.len(),
            };
            let obj = Py::new(py, py_elem)?;
            list.append(obj)?;
        }
        Ok(list.unbind())
    }

    /// Get the output GStreamer buffer.
    fn buffer(&self) -> PyResult<crate::deepstream::PyDsNvBufSurfaceGstBuffer> {
        let guard = self.shared.lock();
        let output = guard.as_ref().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("BatchInferenceOutput has been released")
        })?;
        let shared = output.buffer();
        let buf = shared.into_buffer().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "Cannot extract buffer: outstanding references",
            )
        })?;
        Ok(crate::deepstream::PyDsNvBufSurfaceGstBuffer::new(buf))
    }

    fn __repr__(&self) -> String {
        format!("BatchInferenceOutput(num_elements={})", self.num_elements,)
    }
}
