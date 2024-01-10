use crate::primitives::message::Message;
use crate::release_gil;
use crate::zmq::configs::{ReaderConfig, WriterConfig};
use crate::zmq::results;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyBytes;
use pyo3::{pyclass, pymethods, PyObject, PyResult};
use savant_core::transport::zeromq;

#[pyclass]
pub struct Writer(Option<zeromq::SyncWriter>, WriterConfig);

#[pymethods]
impl Writer {
    #[new]
    pub fn new(config: WriterConfig) -> PyResult<Self> {
        Ok(Self(None, config))
    }

    pub fn is_started(&self) -> bool {
        if self.0.is_none() {
            return false;
        }
        let writer = self.0.as_ref().unwrap();
        writer.is_started()
    }

    pub fn start(&mut self) -> PyResult<()> {
        if self.0.is_some() {
            return Err(PyRuntimeError::new_err("Writer is already started."));
        }
        self.0 = Some(
            zeromq::SyncWriter::new(&self.1 .0)
                .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?,
        );
        Ok(())
    }

    pub fn shutdown(&mut self) -> PyResult<()> {
        if self.0.is_none() {
            return Err(PyRuntimeError::new_err("Writer is not started."));
        }
        let writer = self.0.take().unwrap();
        writer
            .shutdown()
            .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?;
        Ok(())
    }

    pub fn send_eos(&mut self, topic: &str) -> PyResult<PyObject> {
        if self.0.is_none() {
            return Err(PyRuntimeError::new_err("Writer is not started."));
        }
        let writer = self.0.as_ref().unwrap();
        let res = release_gil!(true, || {
            writer
                .send_eos(topic)
                .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
        })?;
        results::process_writer_result(res)
    }

    pub fn send_message(
        &mut self,
        topic: &str,
        message: &Message,
        extra: &PyBytes,
    ) -> PyResult<PyObject> {
        if self.0.is_none() {
            return Err(PyRuntimeError::new_err("Writer is not started."));
        }
        let writer = self.0.as_ref().unwrap();
        let bytes = extra.as_bytes();
        let res = release_gil!(true, || {
            writer
                .send_message(topic, &message.0, &[bytes])
                .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
        })?;
        results::process_writer_result(res)
    }
}

#[pyclass]
pub struct Reader(Option<zeromq::SyncReader>, ReaderConfig);

#[pymethods]
impl Reader {
    #[new]
    pub fn new(config: ReaderConfig) -> PyResult<Self> {
        Ok(Self(None, config))
    }

    pub fn start(&mut self) -> PyResult<()> {
        if self.0.is_some() {
            return Err(PyRuntimeError::new_err("Reader is already started."));
        }
        self.0 = Some(
            zeromq::SyncReader::new(&self.1 .0)
                .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?,
        );
        Ok(())
    }

    pub fn is_started(&self) -> bool {
        if self.0.is_none() {
            return false;
        }
        let reader = self.0.as_ref().unwrap();
        reader.is_started()
    }

    pub fn shutdown(&mut self) -> PyResult<()> {
        if self.0.is_none() {
            return Err(PyRuntimeError::new_err("Reader is not started."));
        }
        let reader = self.0.take().unwrap();
        reader
            .shutdown()
            .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?;
        Ok(())
    }

    pub fn receive(&self) -> PyResult<PyObject> {
        if self.0.is_none() {
            return Err(PyRuntimeError::new_err("Reader is not started."));
        }
        let reader = self.0.as_ref().unwrap();
        let res = release_gil!(true, || {
            reader
                .receive()
                .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
        })?;
        results::process_reader_result(res)
    }
}
