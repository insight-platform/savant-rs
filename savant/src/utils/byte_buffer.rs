use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Arc;

/// A fast copy-free byte buffer intended to efficiently exchange data between Rust & Python
///
/// Parameters
/// ----------
/// v: Vec<u8>
///   The byte buffer.
/// checksum: Optional[int]
///   The checksum of the byte buffer.
///
/// Returns
/// -------
/// ByteBuffer
///   The byte buffer.
///
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
    #[pyo3(name = "bytes")]
    pub fn bytes_py(&self) -> PyObject {
        Python::with_gil(|py| {
            let bytes = PyBytes::new(py, self.inner.as_slice());
            PyObject::from(bytes)
        })
    }
}

impl ByteBuffer {
    pub fn bytes(&self) -> &[u8] {
        self.inner.as_slice()
    }
}
