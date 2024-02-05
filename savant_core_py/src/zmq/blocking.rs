use crate::primitives::message::Message;
use crate::release_gil;
use crate::zmq::configs::{ReaderConfig, WriterConfig};
use crate::zmq::results;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyBytes;
use pyo3::{pyclass, pymethods, PyObject, PyResult};
use savant_core::transport::zeromq;

/// Blocking Writer with GIL release on long-lasting `send_*` operations.
///
/// Parameters
/// ----------
/// config : WriterConfig
///   Writer configuration.
///   
#[pyclass]
pub struct BlockingWriter(Option<zeromq::SyncWriter>, WriterConfig);

#[pymethods]
impl BlockingWriter {
    #[new]
    pub fn new(config: WriterConfig) -> PyResult<Self> {
        Ok(Self(None, config))
    }

    /// Returns `true` if the writer is started.
    ///
    pub fn is_started(&self) -> bool {
        if self.0.is_none() {
            return false;
        }
        let writer = self.0.as_ref().unwrap();
        writer.is_started()
    }

    /// Starts the writer. If the writer is already started, returns an error.
    ///
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

    /// Shuts down the writer. If the writer is not started, returns an error.
    ///
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

    /// Sends EOS to the specified topic. If the writer is not started, returns an error.
    /// Releases GIL while waiting for the result.
    ///
    /// Parameters
    /// ----------
    /// topic : str
    ///   Topic to send EOS to.
    ///
    /// Returns
    /// -------
    /// :py:class:`WriterResultAck`
    /// :py:class:`WriterResultAckTimeout`
    /// :py:class:`WriterResultSendTimeout`
    /// :py:class:`WriterResultSuccess`
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///   When underlying ZeroMQ writer fails and no longer functional. Usually means that the
    ///   writer must be restarted.
    ///
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

    /// Sends a message to the specified topic. If the writer is not started, returns an error.
    /// Releases GIL while waiting for the result.
    ///
    /// Parameters
    /// ----------
    /// topic : str
    ///   Topic to send EOS to.
    /// message : :py:class:`savant_rs.utils.serialization.Message`
    ///   Message to send.
    /// extra : bytes
    ///   Extra data to send with the message.
    ///
    /// Returns
    /// -------
    /// :py:class:`WriterResultAck`
    /// :py:class:`WriterResultAckTimeout`
    /// :py:class:`WriterResultSendTimeout`
    /// :py:class:`WriterResultSuccess`
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///   When underlying ZeroMQ writer fails and no longer functional. Usually means that the
    ///   writer must be restarted.
    ///
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

/// Blocking Reader with GIL release on long-lasting `receive` operations.
///
/// Parameters
/// ----------
/// config : :py:class:`ReaderConfig`
///   Reader configuration.
///
#[pyclass]
pub struct BlockingReader(Option<zeromq::SyncReader>, ReaderConfig);

#[pymethods]
impl BlockingReader {
    #[new]
    pub fn new(config: ReaderConfig) -> PyResult<Self> {
        Ok(Self(None, config))
    }

    /// Starts the reader. If the reader is already started, returns an error.
    ///
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

    /// Returns `true` if the reader is started.
    ///
    pub fn is_started(&self) -> bool {
        if self.0.is_none() {
            return false;
        }
        let reader = self.0.as_ref().unwrap();
        reader.is_started()
    }

    /// Shuts down the reader. If the reader is not started, returns an error.
    ///
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

    /// Receives a message. Blocks until a message is received. Releases GIL while waiting for the
    /// result.
    ///
    /// Returns
    /// -------
    /// :py:class:`ReaderResultEndOfStream`
    /// :py:class:`ReaderResultMessage`
    /// :py:class:`ReaderResultTimeout`
    /// :py:class:`ReaderResultPrefixMismatch`
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///   When the reader receives an error. Generally means that the reader is no longer
    ///   usable and should be shutdown.
    ///
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
