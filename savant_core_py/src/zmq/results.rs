use crate::primitives::message::Message;
use crate::with_gil;
use pyo3::types::PyBytes;
use pyo3::{pyclass, pymethods, IntoPy, Py, PyAny, PyObject, PyResult};
use savant_core::transport::zeromq;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Returned when a writer is unable to send a message due to a timeout on ZMQ.
///
#[pyclass]
#[derive(Debug, Clone, Hash)]
pub struct WriterResultSendTimeout;

#[pymethods]
impl WriterResultSendTimeout {
    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Returned when a writer is unable to receive an ack due to a timeout on ZMQ.
/// Contains a field holding the accumulated timeout (receive_retries x receive_timeout).
///
#[pyclass]
#[derive(Debug, Clone, Hash)]
pub struct WriterResultAckTimeout {
    #[pyo3(get)]
    pub timeout: u128,
}

#[pymethods]
impl WriterResultAckTimeout {
    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Returned when a writer is able to send a message and receive an ack.
/// The result is expected for every Req/Rep message and for only for EOS when
/// using Dealer/Router. Pub/Sub does not use acks.
///
#[pyclass]
#[derive(Debug, Clone, Hash)]
pub struct WriterResultAck {
    #[pyo3(get)]
    pub send_retries_spent: i32,
    #[pyo3(get)]
    pub receive_retries_spent: i32,
    #[pyo3(get)]
    pub time_spent: u128,
}

#[pymethods]
impl WriterResultAck {
    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Returned when a writer is able to send a message without receiving an ack.
/// For Dealer/Router when the message is not EOS, for Pub/Sub always. Req/Rep does not
/// use this result.
///
#[pyclass]
#[derive(Debug, Clone, Hash)]
pub struct WriterResultSuccess {
    #[pyo3(get)]
    pub retries_spent: i32,
    #[pyo3(get)]
    pub time_spent: u128,
}

#[pymethods]
impl WriterResultSuccess {
    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Returned when a reader received a message.
///
#[pyclass]
#[derive(Clone)]
pub struct ReaderResultMessage {
    /// The :py:class:`savant_rs.utils.serialization.Message` received.
    #[pyo3(get)]
    pub message: Message,
    /// The topic of the message.
    #[pyo3(get)]
    pub topic: Vec<u8>,
    /// The routing id of the message. The field is only filled for Router socket.
    #[pyo3(get)]
    pub routing_id: Option<Vec<u8>>,
    pub data: Arc<Vec<Vec<u8>>>,
}

#[pymethods]
impl ReaderResultMessage {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!(
            "ReaderResultMessage [ message = {:?}, topic = {:?}, routing_id = {:?}, data = ... ]",
            &self.message.0, &self.topic, &self.routing_id
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Returns the length o the data vector stored in the message.
    fn data_len(&self) -> usize {
        self.data.len()
    }

    /// Returns the data stored in the message at the given index.
    ///
    /// Parameters
    /// ----------
    /// index: int
    ///   The index of the data to return.
    ///
    /// Returns
    /// -------
    ///  bytes
    ///     The data stored in the message at the given index.
    ///  None
    ///     If the index is out of bounds.
    ///
    /// Raises
    /// ------
    /// MemoryError
    ///   if the data cannot be allocated in Python.
    ///
    fn data(&self, index: usize) -> PyResult<Option<PyObject>> {
        if index < self.data.len() {
            with_gil!(|py| {
                let pybytes = PyBytes::new_with(py, self.data[index].len(), |b: &mut [u8]| {
                    b.copy_from_slice(&self.data[index]);
                    Ok(())
                })?;
                Ok(Some(pybytes.into()))
            })
        } else {
            Ok(None)
        }
    }
}

/// Returned when a reader is unable to receive a message due to a timeout on ZMQ.
///
#[pyclass]
#[derive(Debug, Clone, Hash)]
pub struct ReaderResultTimeout;

#[pymethods]
impl ReaderResultTimeout {
    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Returned when a reader is unable to receive a message due to a timeout on ZMQ.
///
#[pyclass]
#[derive(Debug, Clone, Hash)]
pub struct ReaderResultBlacklisted {
    #[pyo3(get)]
    topic: Vec<u8>,
}

#[pymethods]
impl ReaderResultBlacklisted {
    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Returned when a reader received a message not matching the topic prefix configured.
///
#[pyclass]
#[derive(Debug, Clone, Hash)]
pub struct ReaderResultPrefixMismatch {
    /// The topic of the message.
    #[pyo3(get)]
    pub topic: Vec<u8>,
    /// The routing id of the message. The field is only filled for Router socket.
    #[pyo3(get)]
    pub routing_id: Option<Vec<u8>>,
}

#[pymethods]
impl ReaderResultPrefixMismatch {
    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

pub(crate) fn process_writer_result(res: zeromq::WriterResult) -> PyResult<PyObject> {
    with_gil!(|py| {
        match res {
            zeromq::WriterResult::Success {
                retries_spent,
                time_spent,
            } => Ok(WriterResultSuccess {
                retries_spent,
                time_spent,
            }
            .into_py(py)),
            zeromq::WriterResult::Ack {
                send_retries_spent,
                receive_retries_spent,
                time_spent,
            } => Ok(WriterResultAck {
                send_retries_spent,
                receive_retries_spent,
                time_spent,
            }
            .into_py(py)),
            zeromq::WriterResult::AckTimeout(timeout) => {
                Ok(WriterResultAckTimeout { timeout }.into_py(py))
            }
            zeromq::WriterResult::SendTimeout => Ok(WriterResultSendTimeout {}.into_py(py)),
        }
    })
}

pub(crate) fn process_reader_result(res: zeromq::ReaderResult) -> PyResult<PyObject> {
    with_gil!(|py| {
        match res {
            zeromq::ReaderResult::Blacklisted(topic) => {
                Ok(ReaderResultBlacklisted { topic }.into_py(py))
            }
            zeromq::ReaderResult::Message {
                message,
                topic,
                routing_id,
                data,
            } => Ok(ReaderResultMessage {
                message: Message(*message),
                topic,
                routing_id,
                data: Arc::new(data),
            }
            .into_py(py)),
            zeromq::ReaderResult::Timeout => Ok(ReaderResultTimeout {}.into_py(py)),
            zeromq::ReaderResult::PrefixMismatch { topic, routing_id } => {
                Ok(ReaderResultPrefixMismatch { topic, routing_id }.into_py(py))
            }
            zeromq::ReaderResult::RoutingIdMismatch { topic, routing_id } => {
                Ok(ReaderResultPrefixMismatch { topic, routing_id }.into_py(py))
            }
            zeromq::ReaderResult::TooShort(data) => Ok(data.into_py(py)),
        }
    })
}
