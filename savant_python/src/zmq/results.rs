use crate::primitives::message::Message;
use crate::with_gil;
use pyo3::{pyclass, pymethods, IntoPy, Py, PyAny, PyObject, PyResult};
use savant_core::transport::zeromq;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

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

#[pyclass]
#[derive(Clone)]
pub struct ReaderResultMessage {
    #[pyo3(get)]
    pub message: Message,
    #[pyo3(get)]
    pub topic: Vec<u8>,
    #[pyo3(get)]
    pub routing_id: Option<Vec<u8>>,
    #[pyo3(get)]
    pub data: Vec<Vec<u8>>,
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
}

#[pyclass]
#[derive(Debug, Clone, Hash)]
pub struct ReaderResultEndOfStream {
    #[pyo3(get)]
    pub topic: Vec<u8>,
    #[pyo3(get)]
    pub routing_id: Option<Vec<u8>>,
}

#[pymethods]
impl ReaderResultEndOfStream {
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

#[pyclass]
#[derive(Debug, Clone, Hash)]
pub struct ReaderResultPrefixMismatch {
    #[pyo3(get)]
    pub topic: Vec<u8>,
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
            zeromq::ReaderResult::Message {
                message,
                topic,
                routing_id,
                data,
            } => Ok(ReaderResultMessage {
                message: Message(*message),
                topic,
                routing_id,
                data,
            }
            .into_py(py)),
            zeromq::ReaderResult::EndOfStream { topic, routing_id } => {
                Ok(ReaderResultEndOfStream { topic, routing_id }.into_py(py))
            }
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
