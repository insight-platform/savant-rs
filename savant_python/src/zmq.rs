use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::primitives::message::Message;
use crate::{release_gil, with_gil};
use savant_core::transport::zeromq as rust_zmq;

#[pyclass]
#[derive(Debug, Clone, Hash)]
pub enum WriterSocketType {
    Pub,
    Dealer,
    Req,
}

#[pymethods]
impl WriterSocketType {
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

impl From<rust_zmq::WriterSocketType> for WriterSocketType {
    fn from(socket_type: rust_zmq::WriterSocketType) -> Self {
        match socket_type {
            rust_zmq::WriterSocketType::Pub => Self::Pub,
            rust_zmq::WriterSocketType::Dealer => Self::Dealer,
            rust_zmq::WriterSocketType::Req => Self::Req,
        }
    }
}

impl From<WriterSocketType> for rust_zmq::WriterSocketType {
    fn from(socket_type: WriterSocketType) -> Self {
        match socket_type {
            WriterSocketType::Pub => Self::Pub,
            WriterSocketType::Dealer => Self::Dealer,
            WriterSocketType::Req => Self::Req,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone, Hash)]
pub enum ReaderSocketType {
    Sub,
    Router,
    Rep,
}

#[pymethods]
impl ReaderSocketType {
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

impl From<rust_zmq::ReaderSocketType> for ReaderSocketType {
    fn from(socket_type: rust_zmq::ReaderSocketType) -> Self {
        match socket_type {
            rust_zmq::ReaderSocketType::Sub => Self::Sub,
            rust_zmq::ReaderSocketType::Router => Self::Router,
            rust_zmq::ReaderSocketType::Rep => Self::Rep,
        }
    }
}

impl From<ReaderSocketType> for rust_zmq::ReaderSocketType {
    fn from(socket_type: ReaderSocketType) -> Self {
        match socket_type {
            ReaderSocketType::Sub => Self::Sub,
            ReaderSocketType::Router => Self::Router,
            ReaderSocketType::Rep => Self::Rep,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct TopicPrefixSpec(rust_zmq::TopicPrefixSpec);

#[pymethods]
impl TopicPrefixSpec {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
    #[staticmethod]
    pub fn source_id(id: &str) -> Self {
        Self(rust_zmq::TopicPrefixSpec::SourceId(id.to_string()))
    }

    #[staticmethod]
    pub fn prefix(prefix: &str) -> Self {
        Self(rust_zmq::TopicPrefixSpec::Prefix(prefix.to_string()))
    }

    #[staticmethod]
    pub fn none() -> Self {
        Self(rust_zmq::TopicPrefixSpec::None)
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct WriterConfigBuilder(Option<rust_zmq::WriterConfigBuilder>);

#[pyclass]
#[derive(Debug, Clone)]
pub struct WriterConfig(rust_zmq::WriterConfig);

#[pymethods]
impl WriterConfig {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[pymethods]
impl WriterConfigBuilder {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
    #[new]
    pub fn new(url: &str) -> PyResult<Self> {
        Ok(Self(Some(rust_zmq::WriterConfig::new().url(url).map_err(
            |e| PyValueError::new_err(format!("Failed to set ZeroMQ socket URL: {:?}", e)),
        )?)))
    }

    pub fn with_endpoint(&mut self, endpoint: &str) -> PyResult<()> {
        self.0 = Some(
            self.0
                .take()
                .unwrap()
                .with_endpoint(endpoint)
                .map_err(|e| {
                    PyValueError::new_err(format!("Failed to set ZeroMQ socket endpoint: {:?}", e))
                })?,
        );
        Ok(())
    }

    pub fn with_socket_type(&mut self, socket_type: WriterSocketType) -> PyResult<()> {
        self.0 = Some(
            self.0
                .take()
                .unwrap()
                .with_socket_type(socket_type.into())
                .map_err(|e| {
                    PyValueError::new_err(format!("Failed to set ZeroMQ socket type: {:?}", e))
                })?,
        );
        Ok(())
    }

    pub fn with_bind(&mut self, bind: bool) -> PyResult<()> {
        self.0 = Some(self.0.take().unwrap().with_bind(bind).map_err(|e| {
            PyValueError::new_err(format!("Failed to set ZeroMQ socket bind mode: {:?}", e))
        })?);
        Ok(())
    }

    pub fn with_send_timeout(&mut self, send_timeout: i32) -> PyResult<()> {
        self.0 = Some(
            self.0
                .take()
                .unwrap()
                .with_send_timeout(send_timeout)
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "Failed to set ZeroMQ socket send timeout: {:?}",
                        e
                    ))
                })?,
        );
        Ok(())
    }

    pub fn with_receive_timeout(&mut self, receive_timeout: i32) -> PyResult<()> {
        self.0 = Some(
            self.0
                .take()
                .unwrap()
                .with_receive_timeout(receive_timeout)
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "Failed to set ZeroMQ socket receive timeout: {:?}",
                        e
                    ))
                })?,
        );
        Ok(())
    }

    pub fn with_receive_retries(&mut self, receive_retries: i32) -> PyResult<()> {
        self.0 = Some(
            self.0
                .take()
                .unwrap()
                .with_receive_retries(receive_retries)
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "Failed to set ZeroMQ socket receive retries: {:?}",
                        e
                    ))
                })?,
        );
        Ok(())
    }

    pub fn with_send_retries(&mut self, send_retries: i32) -> PyResult<()> {
        self.0 = Some(
            self.0
                .take()
                .unwrap()
                .with_send_retries(send_retries)
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "Failed to set ZeroMQ socket send retries: {:?}",
                        e
                    ))
                })?,
        );
        Ok(())
    }

    pub fn with_send_hwm(&mut self, send_hwm: i32) -> PyResult<()> {
        self.0 = Some(
            self.0
                .take()
                .unwrap()
                .with_send_hwm(send_hwm)
                .map_err(|e| {
                    PyValueError::new_err(format!("Failed to set ZeroMQ socket send HWM: {:?}", e))
                })?,
        );
        Ok(())
    }

    pub fn with_fix_ipc_permissions(&mut self, fix_ipc_permissions: Option<u32>) -> PyResult<()> {
        self.0 = Some(
            self.0
                .take()
                .unwrap()
                .with_fix_ipc_permissions(fix_ipc_permissions)
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "Failed to set ZeroMQ socket IPC permissions: {:?}",
                        e
                    ))
                })?,
        );
        Ok(())
    }
    pub fn build(&mut self) -> PyResult<WriterConfig> {
        Ok(WriterConfig(self.0.take().unwrap().build().map_err(
            |e| {
                PyValueError::new_err(format!(
                    "Failed to build ZeroMQ socket configuration: {:?}",
                    e
                ))
            },
        )?))
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct ReaderConfigBuilder(Option<rust_zmq::ReaderConfigBuilder>);

#[pyclass]
#[derive(Debug, Clone)]
pub struct ReaderConfig(rust_zmq::ReaderConfig);

#[pymethods]
impl ReaderConfig {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[pymethods]
impl ReaderConfigBuilder {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
    #[new]
    pub fn new(url: &str) -> PyResult<Self> {
        Ok(Self(Some(rust_zmq::ReaderConfig::new().url(url).map_err(
            |e| PyValueError::new_err(format!("Failed to set ZeroMQ socket URL: {:?}", e)),
        )?)))
    }

    pub fn with_endpoint(&mut self, endpoint: &str) -> PyResult<()> {
        self.0 = Some(
            self.0
                .take()
                .unwrap()
                .with_endpoint(endpoint)
                .map_err(|e| {
                    PyValueError::new_err(format!("Failed to set ZeroMQ socket endpoint: {:?}", e))
                })?,
        );
        Ok(())
    }

    pub fn with_socket_type(&mut self, socket_type: ReaderSocketType) -> PyResult<()> {
        self.0 = Some(
            self.0
                .take()
                .unwrap()
                .with_socket_type(socket_type.into())
                .map_err(|e| {
                    PyValueError::new_err(format!("Failed to set ZeroMQ socket type: {:?}", e))
                })?,
        );
        Ok(())
    }

    pub fn with_bind(&mut self, bind: bool) -> PyResult<()> {
        self.0 = Some(self.0.take().unwrap().with_bind(bind).map_err(|e| {
            PyValueError::new_err(format!("Failed to set ZeroMQ socket bind mode: {:?}", e))
        })?);
        Ok(())
    }

    pub fn with_receive_timeout(&mut self, receive_timeout: i32) -> PyResult<()> {
        self.0 = Some(
            self.0
                .take()
                .unwrap()
                .with_receive_timeout(receive_timeout)
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "Failed to set ZeroMQ socket receive timeout: {:?}",
                        e
                    ))
                })?,
        );
        Ok(())
    }

    pub fn with_receive_hwm(&mut self, receive_hwm: i32) -> PyResult<()> {
        self.0 = Some(
            self.0
                .take()
                .unwrap()
                .with_receive_hwm(receive_hwm)
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "Failed to set ZeroMQ socket receive HWM: {:?}",
                        e
                    ))
                })?,
        );
        Ok(())
    }

    pub fn with_topic_prefix_spec(&mut self, topic_prefix_spec: &TopicPrefixSpec) -> PyResult<()> {
        self.0 = Some(
            self.0
                .take()
                .unwrap()
                .with_topic_prefix_spec(topic_prefix_spec.0.clone())
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "Failed to set ZeroMQ socket topic prefix spec: {:?}",
                        e
                    ))
                })?,
        );
        Ok(())
    }

    pub fn with_routing_cache_size(&mut self, size: usize) -> PyResult<()> {
        self.0 = Some(
            self.0
                .take()
                .unwrap()
                .with_routing_cache_size(size)
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "Failed to set ZeroMQ socket routing cache size: {:?}",
                        e
                    ))
                })?,
        );
        Ok(())
    }

    pub fn with_fix_ipc_permissions(&mut self, permissions: Option<u32>) -> PyResult<()> {
        self.0 = Some(
            self.0
                .take()
                .unwrap()
                .with_fix_ipc_permissions(permissions)
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "Failed to set ZeroMQ socket IPC permissions: {:?}",
                        e
                    ))
                })?,
        );
        Ok(())
    }

    pub fn build(&mut self) -> PyResult<ReaderConfig> {
        Ok(ReaderConfig(self.0.take().unwrap().build().map_err(
            |e| {
                PyValueError::new_err(format!(
                    "Failed to build ZeroMQ socket configuration: {:?}",
                    e
                ))
            },
        )?))
    }
}

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
    timeout: u128,
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
    send_retries_spent: i32,
    #[pyo3(get)]
    receive_retries_spent: i32,
    #[pyo3(get)]
    time_spent: u128,
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
    retries_spent: i32,
    #[pyo3(get)]
    time_spent: u128,
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

// pub enum ReaderResult {
//     Message {
//         message: Box<Message>,
//         topic: Vec<u8>,
//         routing_id: Option<Vec<u8>>,
//         data: Vec<Vec<u8>>,
//     },
//     EndOfStream {
//         topic: Vec<u8>,
//         routing_id: Option<Vec<u8>>,
//     },
//     Timeout,
//     PrefixMismatch {
//         topic: Vec<u8>,
//         routing_id: Option<Vec<u8>>,
//     },
//     RoutingIdMismatch {
//         topic: Vec<u8>,
//         routing_id: Option<Vec<u8>>,
//     },
//     TooShort(Vec<Vec<u8>>),
// }

#[pyclass]
#[derive(Clone)]
pub struct ReaderResultMessage {
    #[pyo3(get)]
    message: Message,
    #[pyo3(get)]
    topic: Vec<u8>,
    #[pyo3(get)]
    routing_id: Option<Vec<u8>>,
    #[pyo3(get)]
    data: Vec<Vec<u8>>,
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
    topic: Vec<u8>,
    #[pyo3(get)]
    routing_id: Option<Vec<u8>>,
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
    topic: Vec<u8>,
    #[pyo3(get)]
    routing_id: Option<Vec<u8>>,
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

fn process_writer_result(res: rust_zmq::WriterResult) -> PyResult<PyObject> {
    with_gil!(|py| {
        match res {
            rust_zmq::WriterResult::Success {
                retries_spent,
                time_spent,
            } => Ok(WriterResultSuccess {
                retries_spent,
                time_spent,
            }
            .into_py(py)),
            rust_zmq::WriterResult::Ack {
                send_retries_spent,
                receive_retries_spent,
                time_spent,
            } => Ok(WriterResultAck {
                send_retries_spent,
                receive_retries_spent,
                time_spent,
            }
            .into_py(py)),
            rust_zmq::WriterResult::AckTimeout(timeout) => {
                Ok(WriterResultAckTimeout { timeout }.into_py(py))
            }
            rust_zmq::WriterResult::SendTimeout => Ok(WriterResultSendTimeout {}.into_py(py)),
        }
    })
}

fn process_reader_result(res: rust_zmq::ReaderResult) -> PyResult<PyObject> {
    with_gil!(|py| {
        match res {
            rust_zmq::ReaderResult::Message {
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
            rust_zmq::ReaderResult::EndOfStream { topic, routing_id } => {
                Ok(ReaderResultEndOfStream { topic, routing_id }.into_py(py))
            }
            rust_zmq::ReaderResult::Timeout => Ok(ReaderResultTimeout {}.into_py(py)),
            rust_zmq::ReaderResult::PrefixMismatch { topic, routing_id } => {
                Ok(ReaderResultPrefixMismatch { topic, routing_id }.into_py(py))
            }
            rust_zmq::ReaderResult::RoutingIdMismatch { topic, routing_id } => {
                Ok(ReaderResultPrefixMismatch { topic, routing_id }.into_py(py))
            }
            rust_zmq::ReaderResult::TooShort(data) => Ok(data.into_py(py)),
        }
    })
}

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
        process_writer_result(res)
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
        process_writer_result(res)
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
        process_reader_result(res)
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
