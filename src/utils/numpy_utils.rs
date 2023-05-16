use nalgebra::Scalar;
use numpy::ndarray::ArrayD;
use numpy::{IxDyn, PyArray, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

macro_rules! pretty_print {
    ($arr:expr) => {{
        let indent = 4;
        let prefix = String::from_utf8(vec![b' '; indent]).unwrap();
        let mut result_els = vec!["".to_string()];
        for i in 0..$arr.nrows() {
            let mut row_els = vec![];
            for j in 0..$arr.ncols() {
                row_els.push(format!("{:12.3}", $arr[(i, j)]));
            }
            let row_str = row_els.into_iter().collect::<Vec<_>>().join(" ");
            let row_str = format!("{}{}", prefix, row_str);
            result_els.push(row_str);
        }
        result_els.into_iter().collect::<Vec<_>>().join("\n")
    }};
}

pub trait ConvF64 {
    fn conv_f64(self) -> f64;
}

pub trait RConvF64 {
    fn conv_from_f64(f: f64) -> Self;
}

pub trait ElementType: numpy::Element + Scalar + Copy + Clone + Debug {}

impl ElementType for f32 {}
impl ElementType for f64 {}
impl ElementType for i32 {}
impl ElementType for i64 {}

pub struct Matrix<T: ElementType> {
    inner: Arc<Mutex<nalgebra::DMatrix<T>>>,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Float64Matrix {
    inner: Arc<Mutex<nalgebra::DMatrix<f64>>>,
}

#[pymethods]
impl Float64Matrix {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        let m = self.inner.lock().unwrap();
        pretty_print!(m)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Float32Matrix {
    inner: Arc<Mutex<nalgebra::DMatrix<f32>>>,
}

#[pymethods]
impl Float32Matrix {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        let m = self.inner.lock().unwrap();
        pretty_print!(m)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Int32Matrix {
    inner: Arc<Mutex<nalgebra::DMatrix<i32>>>,
}

#[pymethods]
impl Int32Matrix {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        let m = self.inner.lock().unwrap();
        pretty_print!(m)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Int64Matrix {
    inner: Arc<Mutex<nalgebra::DMatrix<i64>>>,
}

#[pymethods]
impl Int64Matrix {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        let m = self.inner.lock().unwrap();
        pretty_print!(m)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

pub fn matrix_to_ndarray<T: ElementType>(m: Matrix<T>) -> PyObject {
    let m = m.inner.lock().unwrap();
    let arr =
        ArrayD::<T>::from_shape_vec(IxDyn(&[m.nrows(), m.ncols()]), m.as_slice().to_vec()).unwrap();
    Python::with_gil(|py| {
        let arr = PyArray::from_array(py, &arr);
        arr.into_py(py)
    })
}

pub fn ndarray_to_matrix<T: ElementType>(arr: PyReadonlyArrayDyn<T>) -> PyResult<Matrix<T>> {
    let shape = arr.shape().to_vec();
    let slice = arr.as_slice();
    let slice = match slice {
        Ok(slice) => slice,
        Err(_) => {
            return Err(PyValueError::new_err(
                "Non-contiguous array cannot be converted to DMatrix",
            ))
        }
    }
    .to_vec();

    Python::with_gil(|py| {
        py.allow_threads(|| {
            let inner = nalgebra::DMatrix::from_vec(shape[0], shape[1], slice);
            Ok(Matrix {
                inner: Arc::new(Mutex::new(inner)),
            })
        })
    })
}

#[pyfunction]
#[pyo3(name = "ndarray_to_matrix")]
pub fn ndarray_to_matrix_py(arr: &PyAny) -> PyResult<PyObject> {
    if let Ok(arr) = arr.downcast::<PyArray<f32, IxDyn>>() {
        let m = ndarray_to_matrix(arr.readonly()).map(|m| Float32Matrix { inner: m.inner })?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<f64, IxDyn>>() {
        let m = ndarray_to_matrix(arr.readonly()).map(|m| Float64Matrix { inner: m.inner })?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<i32, IxDyn>>() {
        let m = ndarray_to_matrix(arr.readonly()).map(|m| Int32Matrix { inner: m.inner })?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<i64, IxDyn>>() {
        let m = ndarray_to_matrix(arr.readonly()).map(|m| Int64Matrix { inner: m.inner })?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected ndarray of type f32, f64, i32 or i64",
    ))
}

#[pyfunction]
#[pyo3(name = "matrix_to_ndarray")]
pub fn matrix_to_ndarray_py(m: &PyAny) -> PyResult<PyObject> {
    if let Ok(m) = m.extract::<Float32Matrix>() {
        let m = matrix_to_ndarray(Matrix { inner: m.inner });
        return Ok(m);
    }

    if let Ok(m) = m.extract::<Float64Matrix>() {
        let m = matrix_to_ndarray(Matrix { inner: m.inner });
        return Ok(m);
    }

    if let Ok(m) = m.extract::<Int32Matrix>() {
        let m = matrix_to_ndarray(Matrix { inner: m.inner });
        return Ok(m);
    }

    if let Ok(m) = m.extract::<Int64Matrix>() {
        let m = matrix_to_ndarray(Matrix { inner: m.inner });
        return Ok(m);
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected ndarray of type f32, f64, i32 or i64",
    ))
}
