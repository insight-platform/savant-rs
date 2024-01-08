use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::primitives::message::Message;
use crate::release_gil;
use crate::zmq::basic_types::{ReaderSocketType, TopicPrefixSpec, WriterSocketType};
use crate::zmq::configs::{ReaderConfig, ReaderConfigBuilder, WriterConfig, WriterConfigBuilder};
use crate::zmq::results::{
    ReaderResultEndOfStream, ReaderResultMessage, ReaderResultPrefixMismatch, ReaderResultTimeout,
    WriterResultAck, WriterResultAckTimeout, WriterResultSendTimeout, WriterResultSuccess,
};
use savant_core::transport::zeromq as rust_zmq;

pub mod basic_types;
pub mod configs;
pub mod results;

#[pyclass]
pub struct Writer(Option<rust_zmq::SyncWriter>, WriterConfig);

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
            return Err(PyValueError::new_err("Writer is already started."));
        }
        self.0 = Some(
            rust_zmq::SyncWriter::new(&self.1 .0)
                .map_err(|e| PyValueError::new_err(format!("{:?}", e)))?,
        );
        Ok(())
    }

    pub fn shutdown(&mut self) -> PyResult<()> {
        if self.0.is_none() {
            return Err(PyValueError::new_err("Writer is not started."));
        }
        let writer = self.0.take().unwrap();
        writer
            .shutdown()
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;
        Ok(())
    }

    pub fn send_eos(&mut self, topic: &str) -> PyResult<PyObject> {
        if self.0.is_none() {
            return Err(PyValueError::new_err("Writer is not started."));
        }
        let writer = self.0.as_ref().unwrap();
        let res = release_gil!(true, || {
            writer
                .send_eos(topic)
                .map_err(|e| PyValueError::new_err(format!("{:?}", e)))
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
            return Err(PyValueError::new_err("Writer is not started."));
        }
        let writer = self.0.as_ref().unwrap();
        let bytes = extra.as_bytes();
        let res = release_gil!(true, || {
            writer
                .send_message(topic, &message.0, &[bytes])
                .map_err(|e| PyValueError::new_err(format!("{:?}", e)))
        })?;
        results::process_writer_result(res)
    }
}

#[pyclass]
pub struct Reader(Option<rust_zmq::SyncReader>, ReaderConfig);

#[pymethods]
impl Reader {
    #[new]
    pub fn new(config: ReaderConfig) -> PyResult<Self> {
        Ok(Self(None, config))
    }

    pub fn start(&mut self) -> PyResult<()> {
        if self.0.is_some() {
            return Err(PyValueError::new_err("Reader is already started."));
        }
        self.0 = Some(
            rust_zmq::SyncReader::new(&self.1 .0)
                .map_err(|e| PyValueError::new_err(format!("{:?}", e)))?,
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
            return Err(PyValueError::new_err("Reader is not started."));
        }
        let reader = self.0.take().unwrap();
        reader
            .shutdown()
            .map_err(|e| PyValueError::new_err(format!("{:?}", e)))?;
        Ok(())
    }

    pub fn receive(&self) -> PyResult<PyObject> {
        if self.0.is_none() {
            return Err(PyValueError::new_err("Reader is not started."));
        }
        let reader = self.0.as_ref().unwrap();
        let res = release_gil!(true, || {
            reader
                .receive()
                .map_err(|e| PyValueError::new_err(format!("{:?}", e)))
        })?;
        results::process_reader_result(res)
    }
}

#[pymodule]
pub fn zmq(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<WriterSocketType>()?;
    m.add_class::<WriterConfigBuilder>()?;
    m.add_class::<WriterConfig>()?;
    m.add_class::<WriterResultSendTimeout>()?;
    m.add_class::<WriterResultAckTimeout>()?;
    m.add_class::<WriterResultAck>()?;
    m.add_class::<WriterResultSuccess>()?;
    m.add_class::<Writer>()?;

    m.add_class::<ReaderSocketType>()?;
    m.add_class::<TopicPrefixSpec>()?;
    m.add_class::<ReaderConfigBuilder>()?;
    m.add_class::<ReaderConfig>()?;
    m.add_class::<ReaderResultMessage>()?;
    m.add_class::<ReaderResultEndOfStream>()?;
    m.add_class::<ReaderResultTimeout>()?;
    m.add_class::<ReaderResultPrefixMismatch>()?;
    m.add_class::<Reader>()?;

    Ok(())
}
