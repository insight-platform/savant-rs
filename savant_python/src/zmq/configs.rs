use crate::zmq::basic_types::{ReaderSocketType, TopicPrefixSpec, WriterSocketType};
use pyo3::exceptions::PyValueError;
use pyo3::{pyclass, pymethods, Py, PyAny, PyResult};
use savant_core::transport::zeromq;

/// A builder class for a writer config
#[pyclass]
#[derive(Debug, Clone)]
pub struct WriterConfigBuilder(Option<zeromq::WriterConfigBuilder>);

/// A writer configuration
#[pyclass]
#[derive(Debug, Clone)]
pub struct WriterConfig(pub(crate) zeromq::WriterConfig);

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

    /// Creates a new configuration builder based on the provided URL
    ///
    /// The URL can have the following formats:
    ///
    ///   * ``tcp://1.2.3.4:5678``
    ///   * ``ipc:///tmp/test``
    ///   * ``(pub|req|dealer)+(bind|connect):(tcp|ipc)://...``
    ///
    /// Parameters
    /// ----------
    /// url: str
    ///   The URL to use
    ///
    /// Returns
    /// -------
    /// WriterConfigBuilder
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the URL is invalid
    ///
    #[staticmethod]
    fn constructor(url: &str) -> PyResult<Self> {
        Self::new(url)
    }
    #[new]
    pub fn new(url: &str) -> PyResult<Self> {
        Ok(Self(Some(zeromq::WriterConfig::new().url(url).map_err(
            |e| PyValueError::new_err(format!("Failed to set ZeroMQ socket URL: {:?}", e)),
        )?)))
    }

    /// Sets the endpoint for the socket
    ///
    /// Parameters
    /// ----------
    /// endpoint: str
    ///   The endpoint to use in the format ``ipc://`` or ``tcp://``
    ///
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

    /// Sets the socket type
    ///
    /// Parameters
    /// ----------
    /// socket_type: :py:class:`WriterSocketType`
    ///   The socket type to use
    ///
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
pub struct ReaderConfigBuilder(Option<zeromq::ReaderConfigBuilder>);

#[pyclass]
#[derive(Debug, Clone)]
pub struct ReaderConfig(pub(crate) zeromq::ReaderConfig);

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
        Ok(Self(Some(zeromq::ReaderConfig::new().url(url).map_err(
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
