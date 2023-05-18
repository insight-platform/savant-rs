use crate::utils::np::ElementType;
use nalgebra::DMatrix;
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

#[derive(Debug, Clone)]
pub enum NalgebraDMatrixVariant {
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
pub struct NalgebraDMatrix {
    inner: Arc<NalgebraDMatrixVariant>,
}

impl NalgebraDMatrix {
    pub fn from_fp64(m: DMatrix<f64>) -> Self {
        Self {
            inner: Arc::new(NalgebraDMatrixVariant::Float64(m)),
        }
    }
    pub fn from_fp32(m: DMatrix<f32>) -> Self {
        Self {
            inner: Arc::new(NalgebraDMatrixVariant::Float32(m)),
        }
    }

    pub fn from_i64(m: DMatrix<i64>) -> Self {
        Self {
            inner: Arc::new(NalgebraDMatrixVariant::Int64(m)),
        }
    }

    pub fn from_i32(m: DMatrix<i32>) -> Self {
        Self {
            inner: Arc::new(NalgebraDMatrixVariant::Int32(m)),
        }
    }

    pub fn from_i16(m: DMatrix<i16>) -> Self {
        Self {
            inner: Arc::new(NalgebraDMatrixVariant::Int16(m)),
        }
    }

    pub fn from_i8(m: DMatrix<i8>) -> Self {
        Self {
            inner: Arc::new(NalgebraDMatrixVariant::Int8(m)),
        }
    }

    pub fn from_u64(m: DMatrix<u64>) -> Self {
        Self {
            inner: Arc::new(NalgebraDMatrixVariant::UnsignedInt64(m)),
        }
    }

    pub fn from_u32(m: DMatrix<u32>) -> Self {
        Self {
            inner: Arc::new(NalgebraDMatrixVariant::UnsignedInt32(m)),
        }
    }

    pub fn from_u16(m: DMatrix<u16>) -> Self {
        Self {
            inner: Arc::new(NalgebraDMatrixVariant::UnsignedInt16(m)),
        }
    }

    pub fn from_u8(m: DMatrix<u8>) -> Self {
        Self {
            inner: Arc::new(NalgebraDMatrixVariant::UnsignedInt8(m)),
        }
    }
}

#[pymethods]
impl NalgebraDMatrix {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        match self.inner.deref() {
            NalgebraDMatrixVariant::Float64(m) => {
                pretty_print!(m)
            }
            NalgebraDMatrixVariant::Float32(m) => {
                pretty_print!(m)
            }
            NalgebraDMatrixVariant::Int64(m) => {
                pretty_print!(m)
            }
            NalgebraDMatrixVariant::Int32(m) => {
                pretty_print!(m)
            }

            NalgebraDMatrixVariant::Int16(m) => {
                pretty_print!(m)
            }
            NalgebraDMatrixVariant::Int8(m) => {
                pretty_print!(m)
            }
            NalgebraDMatrixVariant::UnsignedInt64(m) => {
                pretty_print!(m)
            }
            NalgebraDMatrixVariant::UnsignedInt32(m) => {
                pretty_print!(m)
            }
            NalgebraDMatrixVariant::UnsignedInt16(m) => {
                pretty_print!(m)
            }
            NalgebraDMatrixVariant::UnsignedInt8(m) => {
                pretty_print!(m)
            }
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

pub fn matrix_to_np<T: ElementType>(m: &DMatrix<T>) -> PyObject {
    let arr =
        ArrayD::<T>::from_shape_vec(IxDyn(&[m.nrows(), m.ncols()]), m.as_slice().to_vec()).unwrap();

    Python::with_gil(|py| {
        let arr = PyArray::from_array(py, &arr);
        arr.into_py(py)
    })
}

pub fn np_to_matrix<T: ElementType>(arr: PyReadonlyArrayDyn<T>) -> PyResult<DMatrix<T>> {
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
#[pyo3(name = "np_to_matrix")]
pub fn np_to_matrix_py(arr: &PyAny) -> PyResult<PyObject> {
    if let Ok(arr) = arr.downcast::<PyArray<f32, IxDyn>>() {
        let m = np_to_matrix(arr.readonly()).map(NalgebraDMatrix::from_fp32)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<f64, IxDyn>>() {
        let m = np_to_matrix(arr.readonly()).map(NalgebraDMatrix::from_fp64)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<i8, IxDyn>>() {
        let m = np_to_matrix(arr.readonly()).map(NalgebraDMatrix::from_i8)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<i16, IxDyn>>() {
        let m = np_to_matrix(arr.readonly()).map(NalgebraDMatrix::from_i16)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<i32, IxDyn>>() {
        let m = np_to_matrix(arr.readonly()).map(NalgebraDMatrix::from_i32)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<i64, IxDyn>>() {
        let m = np_to_matrix(arr.readonly()).map(NalgebraDMatrix::from_i64)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<u8, IxDyn>>() {
        let m = np_to_matrix(arr.readonly()).map(NalgebraDMatrix::from_u8)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<u16, IxDyn>>() {
        let m = np_to_matrix(arr.readonly()).map(NalgebraDMatrix::from_u16)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<u32, IxDyn>>() {
        let m = np_to_matrix(arr.readonly()).map(NalgebraDMatrix::from_u32)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    if let Ok(arr) = arr.downcast::<PyArray<u64, IxDyn>>() {
        let m = np_to_matrix(arr.readonly()).map(NalgebraDMatrix::from_u64)?;
        return Python::with_gil(|py| Ok(m.into_py(py)));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected ndarray of type f32/64, i8/16/32/64, or u8/16/32/64",
    ))
}

#[pyfunction]
#[pyo3(name = "matrix_to_np")]
pub fn matrix_to_np_py(m: &PyAny) -> PyResult<PyObject> {
    if let Ok(m) = m.extract::<NalgebraDMatrix>() {
        let m = match m.inner.deref() {
            NalgebraDMatrixVariant::Float64(m) => matrix_to_np(m),
            NalgebraDMatrixVariant::Float32(m) => matrix_to_np(m),
            NalgebraDMatrixVariant::Int64(m) => matrix_to_np(m),
            NalgebraDMatrixVariant::Int32(m) => matrix_to_np(m),
            NalgebraDMatrixVariant::Int16(m) => matrix_to_np(m),
            NalgebraDMatrixVariant::Int8(m) => matrix_to_np(m),
            NalgebraDMatrixVariant::UnsignedInt64(m) => matrix_to_np(m),
            NalgebraDMatrixVariant::UnsignedInt32(m) => matrix_to_np(m),
            NalgebraDMatrixVariant::UnsignedInt16(m) => matrix_to_np(m),
            NalgebraDMatrixVariant::UnsignedInt8(m) => matrix_to_np(m),
        };
        return Ok(m);
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected ndarray of type f32/64, i8/16/32/64, or u8/16/32/64",
    ))
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_cast() {
        let m = nalgebra::DMatrix::<i8>::from_row_slice(2, 2, &[1, 2, 3, 4]);
        let _n = m.clone().cast::<f64>();
        let _n = m.clone().cast::<f32>();
        let _n = m.cast::<i64>();
    }
}
