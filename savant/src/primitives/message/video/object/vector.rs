use crate::primitives::message::video::query::py::QueryWrapper;
use crate::primitives::message::video::query::{filter, foreach_udf, map_udf, partition};
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
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:#?}", self.inner)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

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

    #[pyo3(name = "partition")]
    fn partition_gil(&self, q: QueryWrapper) -> (VectorView, VectorView) {
        no_gil(|| {
            let (a, b) = partition(self.inner.as_ref(), &q.inner);
            (
                VectorView { inner: Arc::new(a) },
                VectorView { inner: Arc::new(b) },
            )
        })
    }

    #[pyo3(name = "map_udf")]
    fn map_udf_gil(&self, udf: String) -> PyResult<VectorView> {
        no_gil(|| {
            map_udf(
                self.inner.as_ref().iter().collect::<Vec<_>>().as_slice(),
                &udf,
            )
            .map(|x| VectorView { inner: Arc::new(x) })
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    #[pyo3(name = "foreach_udf")]
    fn foreach_udf_gil(&self, udf: String) -> PyResult<Vec<()>> {
        no_gil(|| {
            foreach_udf(
                self.inner.as_ref().iter().collect::<Vec<_>>().as_slice(),
                &udf,
            )
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }
}
