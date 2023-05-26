use crate::primitives::message::video::object::query::filter;
use crate::primitives::message::video::object::query::py::QueryWrapper;
use crate::primitives::message::video::object::InferenceObjectMeta;
use crate::primitives::Object;
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
    fn raw_memory_address(&self) -> usize {
        self as *const Self as usize
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.inner.len())
    }

    fn filter(&self, q: QueryWrapper) -> VectorView {
        VectorView {
            inner: Arc::new(filter(self.inner.as_ref(), &q.inner)),
        }
    }
}

/// Returns the object vector length.
/// # Safety
/// This function is unsafe because it dereferences a raw pointer.
///
#[no_mangle]
pub unsafe extern "C" fn object_vector_len(handle: usize) -> usize {
    let this = unsafe { &*(handle as *const VectorView) };
    this.inner.len()
}

/// Returns the object vector length.
/// # Safety
/// This function is unsafe because it dereferences a raw pointer.
///
#[no_mangle]
pub unsafe extern "C" fn get_inference_meta(handle: usize, pos: usize) -> InferenceObjectMeta {
    let this = unsafe { &*(handle as *const VectorView) };
    (&this.inner[pos]).into()
}
