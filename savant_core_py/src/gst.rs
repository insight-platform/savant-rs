use crate::err_to_pyo3;
use gst::Buffer;
use gst::BufferFlags;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use savant_core::gstreamer::GstBuffer as RustGstBuffer;

#[pyclass(eq, eq_int)]
#[derive(Clone, PartialEq, Debug)]
pub enum FlowResult {
    CustomSuccess2,
    CustomSuccess1,
    CustomSuccess,
    Ok,
    NotLinked,
    Flushing,
    Eos,
    NotNegotiated,
    Error,
    NotSupported,
    CustomError,
    CustomError1,
    CustomError2,
}

#[pyclass(eq, eq_int)]
#[derive(Clone, PartialEq, Debug)]
pub enum InvocationReason {
    Buffer,
    SinkEvent,
    SourceEvent,
}

#[pyclass]
#[derive(Clone)]
pub struct GstBuffer(RustGstBuffer);

impl GstBuffer {
    pub fn from_gst_buffer(buffer: Buffer) -> Self {
        Self(RustGstBuffer::from(buffer))
    }

    pub fn extract(self) -> anyhow::Result<Buffer> {
        self.0.extract()
    }
}

#[pymethods]
impl GstBuffer {
    #[getter]
    pub fn raw_pointer(&self) -> usize {
        self.0.raw_pointer()
    }

    #[getter]
    pub fn pts(&self) -> Option<u64> {
        self.0.pts_ns()
    }

    #[new]
    pub fn create_py() -> Self {
        Self(RustGstBuffer::new())
    }

    #[setter]
    pub fn set_pts(&self, pts: u64) {
        self.0.set_pts_ns(pts);
    }

    #[getter]
    pub fn dts(&self) -> Option<u64> {
        self.0.dts_ns()
    }

    #[setter]
    pub fn set_dts(&self, dts: u64) {
        self.0.set_dts_ns(dts);
    }

    #[getter]
    pub fn dts_or_pts(&self) -> Option<u64> {
        self.0.dts_or_pts_ns()
    }

    #[getter]
    pub fn duration(&self) -> Option<u64> {
        self.0.duration_ns()
    }

    #[setter]
    pub fn set_duration(&self, duration: u64) {
        self.0.set_duration_ns(duration);
    }

    #[getter]
    pub fn is_writable(&self) -> bool {
        self.0.is_writable()
    }

    #[getter]
    pub fn flags(&self) -> u32 {
        self.0.flags().bits()
    }

    #[setter]
    pub fn set_flags(&self, flags: u32) {
        self.0.set_flags(BufferFlags::from_bits_retain(flags));
    }

    pub fn unset_flags(&self, flags: u32) {
        self.0.unset_flags(BufferFlags::from_bits_retain(flags));
    }

    #[getter]
    pub fn maxsize(&self) -> usize {
        self.0.maxsize()
    }

    #[getter]
    pub fn n_memory(&self) -> usize {
        self.0.n_memory()
    }

    #[getter]
    pub fn offset(&self) -> u64 {
        self.0.offset()
    }

    #[setter]
    pub fn set_offset(&self, offset: u64) {
        self.0.set_offset(offset);
    }

    #[getter]
    pub fn offset_end(&self) -> u64 {
        self.0.offset_end()
    }

    #[setter]
    pub fn set_offset_end(&self, offset_end: u64) {
        self.0.set_offset_end(offset_end);
    }

    #[getter]
    pub fn size(&self) -> usize {
        self.0.size()
    }

    #[setter]
    pub fn set_size(&self, size: usize) {
        self.0.set_size(size);
    }

    pub fn copy(&self) -> Self {
        Self(self.0.copy())
    }

    pub fn copy_deep(&self) -> PyResult<Self> {
        let res = err_to_pyo3!(self.0.copy_deep(), PyRuntimeError)?;
        Ok(Self(res))
    }

    pub fn append(&self, buffer: GstBuffer) {
        self.0.append(buffer.0);
    }

    #[getter]
    pub fn get_id_meta(&self) -> Option<Vec<i64>> {
        self.0.get_id_meta()
    }

    pub fn replace_id_meta(&self, ids: Vec<i64>) -> PyResult<Option<Vec<i64>>> {
        err_to_pyo3!(self.0.replace_id_meta(ids), PyRuntimeError)
    }

    pub fn clear_id_meta(&self) -> PyResult<Option<Vec<i64>>> {
        err_to_pyo3!(self.0.clear_id_meta(), PyRuntimeError)
    }
}
