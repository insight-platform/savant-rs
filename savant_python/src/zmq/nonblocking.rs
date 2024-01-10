use crate::primitives::message::Message;
use crate::release_gil;
use crate::zmq::configs::{ReaderConfig, WriterConfig};
use crate::zmq::results;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyBytes;
use pyo3::{pyclass, pymethods, PyObject, PyResult};
use savant_core::transport::zeromq;

#[pyclass]
pub struct NonBlockingReader(zeromq::NonBlockingReader);

#[pymethods]
impl NonBlockingReader {
    #[new]
    pub fn new(config: ReaderConfig, results_queue_size: usize) -> PyResult<Self> {
        Ok(Self(
            zeromq::NonBlockingReader::new(&config.0, results_queue_size)
                .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?,
        ))
    }

    pub fn start(&mut self) -> PyResult<()> {
        if self.0.is_started() {
            return Err(PyRuntimeError::new_err("Reader is already started."));
        }
        self.0
            .start()
            .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?;
        Ok(())
    }

    pub fn is_started(&self) -> bool {
        self.0.is_started()
    }

    pub fn is_shutdown(&self) -> bool {
        self.0.is_shutdown()
    }

    pub fn shutdown(&mut self) -> PyResult<()> {
        self.0
            .shutdown()
            .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
    }

    pub fn receive(&self) -> PyResult<PyObject> {
        let res = self
            .0
            .receive()
            .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?;
        results::process_reader_result(res)
    }

    pub fn try_receive(&self) -> PyResult<Option<PyObject>> {
        match self.0.try_receive() {
            None => Ok(None),
            Some(res) => match res {
                Ok(res) => Ok(Some(results::process_reader_result(res)?)),
                Err(e) => Err(PyRuntimeError::new_err(format!("{:?}", e))),
            },
        }
    }
}

#[pyclass]
pub struct NonBlockingWriter(zeromq::NonBlockingWriter);

#[pyclass]
pub struct WriteOperationResult(zeromq::WriteOperationResult);

#[pymethods]
impl WriteOperationResult {
    pub fn get(&self) -> PyResult<PyObject> {
        results::process_writer_result(release_gil!(true, || self.0.get()).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to get write operation result: {:?}", e))
        })?)
    }

    pub fn try_get(&self) -> PyResult<Option<PyObject>> {
        match self.0.try_get() {
            Ok(Some(res)) => {
                let res = res.map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "Failed to get write operation result: {:?}",
                        e
                    ))
                })?;
                Ok(Some(results::process_writer_result(res)?))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to get write operation result: {:?}",
                e
            ))),
        }
    }
}

#[pymethods]
impl NonBlockingWriter {
    #[new]
    pub fn new(config: WriterConfig, max_infight_messages: usize) -> PyResult<Self> {
        Ok(Self(
            zeromq::NonBlockingWriter::new(&config.0, max_infight_messages)
                .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?,
        ))
    }

    pub fn is_started(&self) -> bool {
        self.0.is_started()
    }

    pub fn is_shutdown(&self) -> bool {
        self.0.is_shutdown()
    }

    pub fn start(&mut self) -> PyResult<()> {
        self.0
            .start()
            .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
    }

    pub fn shutdown(&mut self) -> PyResult<()> {
        self.0
            .shutdown()
            .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
    }

    pub fn send_eos(&mut self, topic: &str) -> PyResult<WriteOperationResult> {
        Ok(WriteOperationResult(self.0.send_eos(topic).map_err(
            |e| PyRuntimeError::new_err(format!("{:?}", e)),
        )?))
    }

    pub fn send_message(
        &mut self,
        topic: &str,
        message: &Message,
        extra: &PyBytes,
    ) -> PyResult<WriteOperationResult> {
        let bytes = extra.as_bytes();
        Ok(WriteOperationResult(
            self.0
                .send_message(topic, &message.0, &[bytes])
                .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?,
        ))
    }
}
