use pyo3::prelude::*;
use std::sync::Arc;

#[pyclass]
#[derive(Clone, Debug)]
pub struct ByteBuffer {
    inner: Arc<Vec<u8>>,
    checksum: Option<u32>,
}

#[pymethods]
impl ByteBuffer {
    #[new]
    pub fn new(v: Vec<u8>, checksum: Option<u32>) -> Self {
        Self {
            inner: Arc::new(v),
            checksum,
        }
    }

    #[getter]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    #[getter]
    pub fn checksum(&self) -> Option<u32> {
        self.checksum
    }

    #[getter]
    pub fn bytes(&self) -> Vec<u8> {
        (*self.inner).clone()
    }
}
