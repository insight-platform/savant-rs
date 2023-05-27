use crate::primitives::message::video::object::query::filter;
use crate::primitives::message::video::object::query::py::QueryWrapper;
use crate::primitives::Object;
use crate::utils::python::no_gil;
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use std::sync::Arc;

#[pyclass]
#[derive(Clone, Debug)]
#[repr(C)]
pub struct VectorView {
    pub(crate) inner: Arc<Vec<Object>>,
}

impl From<Vec<Object>> for VectorView {
    fn from(value: Vec<Object>) -> Self {
        VectorView {
            inner: Arc::new(value),
        }
    }
}

#[pymethods]
impl VectorView {
    fn __getitem__(&self, index: usize) -> PyResult<Object> {
        self.inner
            .get(index)
            .ok_or(PyIndexError::new_err("index out of range"))
            .map(|x| x.clone())
    }

    #[getter]
    fn memory_handle(&self) -> usize {
        self as *const Self as usize
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.inner.len())
    }

    #[pyo3(name = "filter")]
    fn filter_gil(&self, q: QueryWrapper) -> VectorView {
        no_gil(|| VectorView {
            inner: Arc::new(filter(self.inner.as_ref(), &q.inner)),
        })
    }
}
