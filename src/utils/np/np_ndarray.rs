use crate::utils::np::ElementType;
use ndarray::{ArrayBase, IxDyn, OwnedRepr};
use numpy::{PyArray, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use std::ops::Deref;
use std::sync::Arc;

type DynamicArray<T> = ArrayBase<OwnedRepr<T>, IxDyn>;

#[derive(Debug, Clone)]
pub enum NDarrayVariant {
    Float64(DynamicArray<f64>),
    Float32(DynamicArray<f32>),
    Int64(DynamicArray<i64>),
    Int32(DynamicArray<i32>),
    Int16(DynamicArray<i16>),
    Int8(DynamicArray<i8>),
    UnsignedInt64(DynamicArray<u64>),
    UnsignedInt32(DynamicArray<u32>),
    UnsignedInt16(DynamicArray<u16>),
    UnsignedInt8(DynamicArray<u8>),
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct NDarray {
    inner: Arc<NDarrayVariant>,
}

#[pymethods]
impl NDarray {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:#?}", self.inner.deref())
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl NDarray {
    pub fn from_fp64(m: DynamicArray<f64>) -> Self {
        Self {
            inner: Arc::new(NDarrayVariant::Float64(m)),
        }
    }
    pub fn from_fp32(m: DynamicArray<f32>) -> Self {
        Self {
            inner: Arc::new(NDarrayVariant::Float32(m)),
        }
    }

    pub fn from_i64(m: DynamicArray<i64>) -> Self {
        Self {
            inner: Arc::new(NDarrayVariant::Int64(m)),
        }
    }

    pub fn from_i32(m: DynamicArray<i32>) -> Self {
        Self {
            inner: Arc::new(NDarrayVariant::Int32(m)),
        }
    }

    pub fn from_i16(m: DynamicArray<i16>) -> Self {
        Self {
            inner: Arc::new(NDarrayVariant::Int16(m)),
        }
    }

    pub fn from_i8(m: DynamicArray<i8>) -> Self {
        Self {
            inner: Arc::new(NDarrayVariant::Int8(m)),
        }
    }

    pub fn from_u64(m: DynamicArray<u64>) -> Self {
        Self {
            inner: Arc::new(NDarrayVariant::UnsignedInt64(m)),
        }
    }

    pub fn from_u32(m: DynamicArray<u32>) -> Self {
        Self {
            inner: Arc::new(NDarrayVariant::UnsignedInt32(m)),
        }
    }

    pub fn from_u16(m: DynamicArray<u16>) -> Self {
        Self {
            inner: Arc::new(NDarrayVariant::UnsignedInt16(m)),
        }
    }

    pub fn from_u8(m: DynamicArray<u8>) -> Self {
        Self {
            inner: Arc::new(NDarrayVariant::UnsignedInt8(m)),
        }
    }
}

pub fn ndarray_to_np<T: ElementType>(m: &DynamicArray<T>) -> PyObject {
    let arr = m.clone();
    Python::with_gil(|py| {
        let arr = PyArray::from_array(py, &arr);
        arr.into_py(py)
    })
}

pub fn np_to_ndarray<T: ElementType>(arr: PyReadonlyArrayDyn<T>) -> PyResult<DynamicArray<T>> {
    let arr = arr.as_array().to_owned();
    Python::with_gil(|py| py.allow_threads(|| Ok(arr)))
}

#[pyfunction]
#[pyo3(name = "np_to_ndarray")]
pub fn np_to_ndarray_py(arr: &PyAny) -> PyResult<PyObject> {
    if let Ok(arr) = arr.downcast::<PyArray<f32, IxDyn>>() {
        let m = np_to_ndarray(arr.readonly()).map(NDarray::from_fp32)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<f64, IxDyn>>() {
        let m = np_to_ndarray(arr.readonly()).map(NDarray::from_fp64)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<i8, IxDyn>>() {
        let m = np_to_ndarray(arr.readonly()).map(NDarray::from_i8)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<i16, IxDyn>>() {
        let m = np_to_ndarray(arr.readonly()).map(NDarray::from_i16)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<i32, IxDyn>>() {
        let m = np_to_ndarray(arr.readonly()).map(NDarray::from_i32)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<i64, IxDyn>>() {
        let m = np_to_ndarray(arr.readonly()).map(NDarray::from_i64)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<u8, IxDyn>>() {
        let m = np_to_ndarray(arr.readonly()).map(NDarray::from_u8)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<u16, IxDyn>>() {
        let m = np_to_ndarray(arr.readonly()).map(NDarray::from_u16)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<u32, IxDyn>>() {
        let m = np_to_ndarray(arr.readonly()).map(NDarray::from_u32)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<u64, IxDyn>>() {
        let m = np_to_ndarray(arr.readonly()).map(NDarray::from_u64)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected ndarray of type f32/64, i8/16/32/64, or u8/16/32/64",
    ))
}

#[pyfunction]
#[pyo3(name = "ndarray_to_np")]
pub fn ndarray_to_np_py(m: &PyAny) -> PyResult<PyObject> {
    if let Ok(m) = m.extract::<NDarray>() {
        let m = match m.inner.deref() {
            NDarrayVariant::Float64(m) => ndarray_to_np(m),
            NDarrayVariant::Float32(m) => ndarray_to_np(m),
            NDarrayVariant::Int64(m) => ndarray_to_np(m),
            NDarrayVariant::Int32(m) => ndarray_to_np(m),
            NDarrayVariant::Int16(m) => ndarray_to_np(m),
            NDarrayVariant::Int8(m) => ndarray_to_np(m),
            NDarrayVariant::UnsignedInt64(m) => ndarray_to_np(m),
            NDarrayVariant::UnsignedInt32(m) => ndarray_to_np(m),
            NDarrayVariant::UnsignedInt16(m) => ndarray_to_np(m),
            NDarrayVariant::UnsignedInt8(m) => ndarray_to_np(m),
        };
        return Ok(m);
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected ndarray of type f32/64, i8/16/32/i64, or u8/16/32/64",
    ))
}
