use nalgebra::{DMatrix, Scalar};
use numpy::ndarray::ArrayD;
use numpy::{IxDyn, PyArray, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fmt::Debug;
use std::ops::Deref;
use std::sync::Arc;

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
impl ElementType for i8 {}
impl ElementType for i16 {}
impl ElementType for i32 {}
impl ElementType for i64 {}
impl ElementType for u8 {}
impl ElementType for u16 {}
impl ElementType for u32 {}
impl ElementType for u64 {}

#[derive(Debug, Clone)]
pub enum MatrixVariant {
    Float64(DMatrix<f64>),
    Float32(DMatrix<f32>),
    Int64(DMatrix<i64>),
    Int32(DMatrix<i32>),
    Int16(DMatrix<i16>),
    Int8(DMatrix<i8>),
    UnsignedInt64(DMatrix<u64>),
    UnsignedInt32(DMatrix<u32>),
    UnsignedInt16(DMatrix<u16>),
    UnsignedInt8(DMatrix<u8>),
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Matrix {
    inner: Arc<MatrixVariant>,
}

impl Matrix {
    pub fn from_fp64(m: DMatrix<f64>) -> Self {
        Self {
            inner: Arc::new(MatrixVariant::Float64(m)),
        }
    }
    pub fn from_fp32(m: DMatrix<f32>) -> Self {
        Self {
            inner: Arc::new(MatrixVariant::Float32(m)),
        }
    }

    pub fn from_i64(m: DMatrix<i64>) -> Self {
        Self {
            inner: Arc::new(MatrixVariant::Int64(m)),
        }
    }

    pub fn from_i32(m: DMatrix<i32>) -> Self {
        Self {
            inner: Arc::new(MatrixVariant::Int32(m)),
        }
    }

    pub fn from_i16(m: DMatrix<i16>) -> Self {
        Self {
            inner: Arc::new(MatrixVariant::Int16(m)),
        }
    }

    pub fn from_i8(m: DMatrix<i8>) -> Self {
        Self {
            inner: Arc::new(MatrixVariant::Int8(m)),
        }
    }

    pub fn from_u64(m: DMatrix<u64>) -> Self {
        Self {
            inner: Arc::new(MatrixVariant::UnsignedInt64(m)),
        }
    }

    pub fn from_u32(m: DMatrix<u32>) -> Self {
        Self {
            inner: Arc::new(MatrixVariant::UnsignedInt32(m)),
        }
    }

    pub fn from_u16(m: DMatrix<u16>) -> Self {
        Self {
            inner: Arc::new(MatrixVariant::UnsignedInt16(m)),
        }
    }

    pub fn from_u8(m: DMatrix<u8>) -> Self {
        Self {
            inner: Arc::new(MatrixVariant::UnsignedInt8(m)),
        }
    }
}

#[pymethods]
impl Matrix {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        match self.inner.deref() {
            MatrixVariant::Float64(m) => {
                pretty_print!(m)
            }
            MatrixVariant::Float32(m) => {
                pretty_print!(m)
            }
            MatrixVariant::Int64(m) => {
                pretty_print!(m)
            }
            MatrixVariant::Int32(m) => {
                pretty_print!(m)
            }

            MatrixVariant::Int16(m) => {
                pretty_print!(m)
            }
            MatrixVariant::Int8(m) => {
                pretty_print!(m)
            }
            MatrixVariant::UnsignedInt64(m) => {
                pretty_print!(m)
            }
            MatrixVariant::UnsignedInt32(m) => {
                pretty_print!(m)
            }
            MatrixVariant::UnsignedInt16(m) => {
                pretty_print!(m)
            }
            MatrixVariant::UnsignedInt8(m) => {
                pretty_print!(m)
            }
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

pub fn matrix_to_ndarray<T: ElementType>(m: &DMatrix<T>) -> PyObject {
    let arr =
        ArrayD::<T>::from_shape_vec(IxDyn(&[m.nrows(), m.ncols()]), m.as_slice().to_vec()).unwrap();

    Python::with_gil(|py| {
        let arr = PyArray::from_array(py, &arr);
        arr.into_py(py)
    })
}

pub fn ndarray_to_matrix<T: ElementType>(arr: PyReadonlyArrayDyn<T>) -> PyResult<DMatrix<T>> {
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
        py.allow_threads(|| Ok(nalgebra::DMatrix::from_vec(shape[0], shape[1], slice)))
    })
}

#[pyfunction]
#[pyo3(name = "ndarray_to_matrix")]
pub fn ndarray_to_matrix_py(arr: &PyAny) -> PyResult<PyObject> {
    if let Ok(arr) = arr.downcast::<PyArray<f32, IxDyn>>() {
        let m = ndarray_to_matrix(arr.readonly()).map(Matrix::from_fp32)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<f64, IxDyn>>() {
        let m = ndarray_to_matrix(arr.readonly()).map(Matrix::from_fp64)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<i8, IxDyn>>() {
        let m = ndarray_to_matrix(arr.readonly()).map(Matrix::from_i8)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<i16, IxDyn>>() {
        let m = ndarray_to_matrix(arr.readonly()).map(Matrix::from_i16)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<i32, IxDyn>>() {
        let m = ndarray_to_matrix(arr.readonly()).map(Matrix::from_i32)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<i64, IxDyn>>() {
        let m = ndarray_to_matrix(arr.readonly()).map(Matrix::from_i64)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<u8, IxDyn>>() {
        let m = ndarray_to_matrix(arr.readonly()).map(Matrix::from_u8)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<u16, IxDyn>>() {
        let m = ndarray_to_matrix(arr.readonly()).map(Matrix::from_u16)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<u32, IxDyn>>() {
        let m = ndarray_to_matrix(arr.readonly()).map(Matrix::from_u32)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<u64, IxDyn>>() {
        let m = ndarray_to_matrix(arr.readonly()).map(Matrix::from_u64)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected ndarray of type f32/64, i8/16/32/64, or u8/16/32/64",
    ))
}

#[pyfunction]
#[pyo3(name = "matrix_to_ndarray")]
pub fn matrix_to_ndarray_py(m: &PyAny) -> PyResult<PyObject> {
    if let Ok(m) = m.extract::<Matrix>() {
        let m = match m.inner.deref() {
            MatrixVariant::Float64(m) => matrix_to_ndarray(m),
            MatrixVariant::Float32(m) => matrix_to_ndarray(m),
            MatrixVariant::Int64(m) => matrix_to_ndarray(m),
            MatrixVariant::Int32(m) => matrix_to_ndarray(m),
            MatrixVariant::Int16(m) => matrix_to_ndarray(m),
            MatrixVariant::Int8(m) => matrix_to_ndarray(m),
            MatrixVariant::UnsignedInt64(m) => matrix_to_ndarray(m),
            MatrixVariant::UnsignedInt32(m) => matrix_to_ndarray(m),
            MatrixVariant::UnsignedInt16(m) => matrix_to_ndarray(m),
            MatrixVariant::UnsignedInt8(m) => matrix_to_ndarray(m),
        };
        return Ok(m);
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected ndarray of type f32, f64, i32 or i64",
    ))
}
