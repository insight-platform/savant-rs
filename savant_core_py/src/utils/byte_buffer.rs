use crate::attach;
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
    #[pyo3(signature = (v, checksum=None))]
    fn create(v: &Bound<'_, PyBytes>, checksum: Option<u32>) -> PyResult<Self> {
        Ok(Self::new(v.as_bytes().to_vec(), checksum))
    }

    /// Returns the length of the byte buffer.
    ///
    /// Returns
    /// -------
    /// int
    ///   The length of the byte buffer.
    ///
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.inner.len())
    }

    /// Returns whether the byte buffer is empty.
    ///
    /// Returns
    /// -------
    /// bool
    ///   Whether the byte buffer is empty.
    ///
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the checksum of the byte buffer.
    ///
    /// Returns
    /// -------
    /// None or int
    ///   The checksum of the byte buffer. If not set when creating the byte buffer, ``None`` is returned.
    ///
    #[getter]
    pub fn checksum(&self) -> Option<u32> {
        self.checksum
    }

    /// Returns the bytes as Python ``bytes``.
    ///
    /// Returns
    /// -------
    /// bytes
    ///   The bytes as Python ``bytes``.
    ///
    #[getter]
    #[pyo3(name = "bytes")]
    pub fn bytes_py(&self) -> Py<PyAny> {
        attach!(|py| {
            let bytes = PyBytes::new(py, self.inner.as_slice());
            Py::from(bytes)
        })
    }
}

impl ByteBuffer {
    pub fn from_arc(v: Arc<Vec<u8>>, checksum: Option<u32>) -> Self {
        Self { inner: v, checksum }
    }

    pub fn bytes(&self) -> &[u8] {
        self.inner.as_slice()
    }

    pub fn new(v: Vec<u8>, checksum: Option<u32>) -> Self {
        Self {
            inner: Arc::new(v),
            checksum,
        }
    }
}
