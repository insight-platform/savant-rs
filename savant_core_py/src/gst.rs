use parking_lot::RwLock;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

#[pyclass]
#[derive(Clone)]
pub struct GstBuffer(Arc<RwLock<gst::Buffer>>);

impl GstBuffer {
    pub fn new(buffer: gst::Buffer) -> Self {
        Self(Arc::new(RwLock::new(buffer)))
    }

    pub fn extract(self) -> anyhow::Result<gst::Buffer> {
        let lock = Arc::try_unwrap(self.0).map_err(|_| {
            anyhow::anyhow!("Could not extract GstBuffer because multiple object references exist")
        })?;
        Ok(lock.into_inner())
    }
}

#[pymethods]
impl GstBuffer {
    #[getter]
    pub fn pts(&self) -> Option<u64> {
        let bind = self.0.read();
        bind.pts().map(|pts| pts.nseconds())
    }

    #[getter]
    pub fn dts(&self) -> Option<u64> {
        let bind = self.0.read();
        bind.dts().map(|dts| dts.nseconds())
    }

    #[getter]
    pub fn dts_or_pts(&self) -> Option<u64> {
        let bind = self.0.read();
        bind.dts_or_pts().map(|dts_or_pts| dts_or_pts.nseconds())
    }

    #[getter]
    pub fn duration(&self) -> Option<u64> {
        let bind = self.0.read();
        bind.duration().map(|duration| duration.nseconds())
    }

    #[getter]
    pub fn is_writable(&self) -> bool {
        let bind = self.0.read();
        bind.is_writable()
    }

    #[getter]
    pub fn flags(&self) -> u32 {
        let bind = self.0.read();
        bind.flags().bits()
    }

    #[getter]
    pub fn maxsize(&self) -> usize {
        let bind = self.0.read();
        bind.maxsize()
    }

    #[getter]
    pub fn n_memory(&self) -> usize {
        let bind = self.0.read();
        bind.n_memory()
    }

    #[getter]
    pub fn offset(&self) -> u64 {
        let bind = self.0.read();
        bind.offset()
    }

    #[getter]
    pub fn offset_end(&self) -> u64 {
        let bind = self.0.read();
        bind.offset_end()
    }

    #[getter]
    pub fn size(&self) -> usize {
        let bind = self.0.read();
        bind.size()
    }

    pub fn copy(&self) -> Self {
        let bind = self.0.read();
        let new = Arc::new(RwLock::new(bind.copy()));
        Self(new)
    }

    pub fn copy_deep(&self) -> PyResult<Self> {
        let bind = self.0.read();
        let new_buf = bind
            .copy_deep()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self(Arc::new(RwLock::new(new_buf))))
    }
}
