use crate::detach;
use crate::primitives::message::Message;
use crate::zmq::configs::{ReaderConfig, WriterConfig};
use crate::zmq::results;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::{PyBytes, PyBytesMethods};
use pyo3::{pyclass, pymethods, Bound, Py, PyAny, PyResult};
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
                .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))?,
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
            .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))?;
        Ok(())
    }

    /// Sends EOS to the specified source_id. If the writer is not started, returns an error.
    /// Releases GIL while waiting for the result.
    ///
    /// Parameters
    /// ----------
    /// source_id : str
    ///   Topic to send EOS to.
    /// topic : str
    ///   Topic to send EOS to. If not provided, the source_id will be used as the topic.
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
    #[pyo3(signature = (source_id, topic=None))]
    pub fn send_eos(&mut self, source_id: &str, topic: Option<&str>) -> PyResult<Py<PyAny>> {
        if self.0.is_none() {
            return Err(PyRuntimeError::new_err("Writer is not started."));
        }
        let writer = self.0.as_ref().unwrap();
        let topic = topic.unwrap_or(source_id);
        let res = detach!(true, || {
            writer
                .send_eos_with_topic(topic, source_id)
                .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))
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
        message: &mut Message,
        extra: &Bound<'_, PyBytes>,
    ) -> PyResult<Py<PyAny>> {
        if self.0.is_none() {
            return Err(PyRuntimeError::new_err("Writer is not started."));
        }
        let writer = self.0.as_ref().unwrap();
        let bytes = extra.as_bytes();
        let res = detach!(true, || {
            writer
                .send_message(topic, &mut message.0, &[bytes])
                .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))
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
                .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))?,
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
            .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))?;
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
    pub fn receive(&self) -> PyResult<Py<PyAny>> {
        if self.0.is_none() {
            return Err(PyRuntimeError::new_err("Reader is not started."));
        }
        let reader = self.0.as_ref().unwrap();
        let res = detach!(true, || {
            reader
                .receive()
                .map_err(|e| PyRuntimeError::new_err(format!("{e:?}")))
        })?;
        results::process_reader_result(res)
    }

    /// Blacklists source
    ///
    /// Parameters
    /// ----------
    /// source_id : bytes
    ///   Source ID to blacklist.
    ///
    pub fn blacklist_source(&self, source_id: &Bound<'_, PyBytes>) {
        if self.0.is_none() {
            return;
        }
        let reader = self.0.as_ref().unwrap();
        let bytes = source_id.as_bytes();
        reader.blacklist_source(bytes);
    }

    /// Returns `true` if the source is blacklisted.
    ///
    /// Parameters
    /// ----------
    /// source_id : bytes
    ///  Source ID to check.
    ///
    /// Returns
    /// -------
    /// bool
    ///   `true` if the source is blacklisted.
    ///
    pub fn is_blacklisted(&self, source_id: &Bound<'_, PyBytes>) -> bool {
        if self.0.is_none() {
            return false;
        }
        let reader = self.0.as_ref().unwrap();
        let bytes = source_id.as_bytes();
        reader.is_blacklisted(bytes)
    }
}
