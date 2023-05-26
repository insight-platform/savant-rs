use crate::primitives::message::video::object::query::filter;
use crate::primitives::message::video::object::query::py::QueryWrapper;
use crate::primitives::Object;
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use std::sync::Arc;

#[pyclass]
#[derive(Clone, Debug)]
#[repr(C)]
pub struct ObjectVector {
    pub(crate) inner: Arc<Vec<Object>>,
}

impl From<Vec<Object>> for ObjectVector {
    fn from(value: Vec<Object>) -> Self {
        ObjectVector {
            inner: Arc::new(value),
        }
    }
}

#[pymethods]
impl ObjectVector {
    fn __getitem__(&self, index: usize) -> PyResult<Object> {
        self.inner
            .get(index)
            .ok_or(PyIndexError::new_err("index out of range"))
            .map(|x| x.clone())
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.inner.len())
    }

    fn filter(&self, q: QueryWrapper) -> ObjectVector {
        ObjectVector {
            inner: Arc::new(filter(self.inner.as_ref(), &q.inner)),
        }
    }
}

#[no_mangle]
pub extern "C" fn object_vector_len(this: *const ObjectVector) -> usize {
    let this = unsafe { &*this };
    this.inner.len()
}
