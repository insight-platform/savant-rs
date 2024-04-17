use crate::primitives::message::Message;
use crate::release_gil;
use crate::zmq::configs::{ReaderConfig, WriterConfig};
use crate::zmq::results;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyBytes;
use pyo3::{pyclass, pymethods, PyObject, PyResult};
use savant_core::transport::zeromq;

/// A non-blocking reader. Does not release GIL when uses `receive` convenience method, which is blocking.
/// For non-blocking operations use `try_receive`.
///
/// Parameters
/// ----------
/// config : ReaderConfig
///   Reader configuration.
/// results_queue_size : int
///   Size of the queue for storing results. If the queue is full, the reader's internal operation will block.
///   and depending on the socket type can cause either drop or backpressure. The user can use `enqueued_results` to
///   check the number of enqueued results and read them either with `receive` or `try_receive`.
///
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

    /// Starts the reader. If the reader is already started, returns an error.
    ///
    pub fn start(&mut self) -> PyResult<()> {
        if self.0.is_started() {
            return Err(PyRuntimeError::new_err("Reader is already started."));
        }
        self.0
            .start()
            .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?;
        Ok(())
    }

    /// Returns the number of enqueued results. Those results require fetching to avoid the reader
    /// blocking.
    ///
    pub fn enqueued_results(&self) -> usize {
        self.0.enqueued_results()
    }

    /// Returns `true` if the reader is started.
    ///
    pub fn is_started(&self) -> bool {
        self.0.is_started()
    }

    /// Returns `true` if the reader is shutdown.
    ///
    pub fn is_shutdown(&self) -> bool {
        self.0.is_shutdown()
    }

    /// Shuts down the reader. If the reader is already shutdown, returns an error.
    ///
    pub fn shutdown(&mut self) -> PyResult<()> {
        self.0
            .shutdown()
            .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
    }

    /// Receives a message. Blocks until a message is received. Does not release GIL.
    /// This is a convenience method which normally should not be used with such a reader.
    /// For non-blocking operations use `try_receive`.
    ///
    /// Returns
    /// -------
    /// :py:class:`ReaderResultEndOfStream`
    /// :py:class:`ReaderResultMessage`
    /// :py:class:`ReaderResultPrefixMismatch`
    /// :py:class:`ReaderResultTimeout`
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///   When the reader receives an error. Generally means that the reader is no longer
    ///   usable and should be shutdown.
    ///
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

    /// Blacklists source
    ///
    /// Parameters
    /// ----------
    /// source_id : bytes
    ///   Source ID to blacklist.
    ///
    pub fn blacklist_source(&self, source_id: &PyBytes) {
        let bytes = source_id.as_bytes();
        self.0.blacklist_source(bytes);
    }
}

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

/// A non-blocking writer. Sends the message to the internal command queue with returning a
/// Future-like object :py:class:`WriteOperationResult`. The user can use `get` or `try_get` to get the result of the operation.
///
/// Parameters
/// ----------
/// config : WriterConfig
///   Writer configuration.
/// max_infight_messages : int
///   Maximum number of inflight messages. If the number of inflight messages is equal to this value, the writer's internal operation will block.
///
#[pyclass]
pub struct NonBlockingWriter(zeromq::NonBlockingWriter);

#[pymethods]
impl NonBlockingWriter {
    #[new]
    pub fn new(config: WriterConfig, max_infight_messages: usize) -> PyResult<Self> {
        Ok(Self(
            zeromq::NonBlockingWriter::new(&config.0, max_infight_messages)
                .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?,
        ))
    }

    /// Returns `true` if the writer is started.
    ///
    pub fn is_started(&self) -> bool {
        self.0.is_started()
    }

    /// Returns `true` if the writer is shutdown.
    ///
    pub fn is_shutdown(&self) -> bool {
        self.0.is_shutdown()
    }

    /// Returns the number of inflight messages.
    pub fn inflight_messages(&self) -> usize {
        self.0.inflight_messages()
    }

    /// Returns `true` if the writer has capacity to send more messages.
    pub fn has_capacity(&self) -> bool {
        self.0.has_capacity()
    }

    /// Starts the writer. If the writer is already started, returns an error.
    ///
    pub fn start(&mut self) -> PyResult<()> {
        self.0
            .start()
            .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
    }

    /// Shuts down the writer. If the writer is already shutdown, returns an error.
    ///
    pub fn shutdown(&mut self) -> PyResult<()> {
        self.0
            .shutdown()
            .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
    }

    /// Sends EOS to the specified topic.
    ///
    /// Parameters
    /// ----------
    /// topic : str
    ///   Topic to send EOS to.
    ///
    /// Returns
    /// -------
    /// :py:class:`WriteOperationResult`
    ///   Write operation result - a future-like object
    ///   which can be used to get the result of the operation.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///   When the writer receives an error. Generally means that the writer is no longer
    ///   usable and should be shutdown.
    ///
    pub fn send_eos(&mut self, topic: &str) -> PyResult<WriteOperationResult> {
        Ok(WriteOperationResult(self.0.send_eos(topic).map_err(
            |e| PyRuntimeError::new_err(format!("{:?}", e)),
        )?))
    }

    /// Sends a message to the specified topic.
    ///
    /// Parameters
    /// ----------
    /// topic : str
    ///   Topic to send the message to.
    /// message : :py:class:`savant_rs.utils.serialization.Message`
    ///   Message to send.
    /// extra : bytes
    ///   Extra data to send with the message.
    ///
    /// Returns
    /// -------
    /// :py:class:`WriteOperationResult`
    ///   Write operation result - a future-like object
    ///   which can be used to get the result of the operation.
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///   When the writer receives an error. Generally means that the writer is no longer
    ///   usable and should be shutdown.
    ///
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
