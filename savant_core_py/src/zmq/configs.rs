use crate::zmq::basic_types::{ReaderSocketType, TopicPrefixSpec, WriterSocketType};
use pyo3::exceptions::PyValueError;
use pyo3::{pyclass, pymethods, Py, PyAny, PyResult};
use savant_core::transport::zeromq;
use std::num::NonZeroU64;

/// Creates a new configuration builder based on the provided URL.
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
#[pyclass]
#[derive(Debug, Clone)]
pub struct WriterConfigBuilder(Option<zeromq::WriterConfigBuilder>);

/// A writer configuration
#[pyclass]
#[derive(Debug, Clone)]
pub struct WriterConfig(pub(crate) zeromq::WriterConfig);

#[pymethods]
impl WriterConfig {
    #[getter]
    fn endpoint(&self) -> String {
        self.0.endpoint().to_string()
    }

    #[getter]
    fn socket_type(&self) -> WriterSocketType {
        self.0.socket_type().clone().into()
    }

    #[getter]
    fn bind(&self) -> bool {
        *self.0.bind()
    }

    #[getter]
    fn send_timeout(&self) -> i32 {
        *self.0.send_timeout()
    }

    #[getter]
    fn receive_timeout(&self) -> i32 {
        *self.0.receive_timeout()
    }

    #[getter]
    fn receive_retries(&self) -> i32 {
        *self.0.receive_retries()
    }

    #[getter]
    fn send_retries(&self) -> i32 {
        *self.0.send_retries()
    }

    #[getter]
    fn send_hwm(&self) -> i32 {
        *self.0.send_hwm()
    }

    #[getter]
    fn receive_hwm(&self) -> i32 {
        *self.0.receive_hwm()
    }

    #[getter]
    fn fix_ipc_permissions(&self) -> Option<u32> {
        *self.0.fix_ipc_permissions()
    }

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
        Ok(Self(Some(zeromq::WriterConfig::new().url(url).map_err(
            |e| PyValueError::new_err(format!("Failed to set ZeroMQ socket URL: {:?}", e)),
        )?)))
    }

    /// Sets the socket type
    ///
    /// Parameters
    /// ----------
    /// socket_type: :py:class:`WriterSocketType`
    ///   The socket type to use, defaults to ``WriterSocketType.Dealer``
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

    /// Specifies either socket binds or connects
    ///
    /// Parameters
    /// ----------
    /// bind: bool
    ///   If ``True``, the socket will bind, otherwise it will connect. Defaults to ``True``
    ///
    pub fn with_bind(&mut self, bind: bool) -> PyResult<()> {
        self.0 = Some(self.0.take().unwrap().with_bind(bind).map_err(|e| {
            PyValueError::new_err(format!("Failed to set ZeroMQ socket bind mode: {:?}", e))
        })?);
        Ok(())
    }

    /// Sets the send timeout for the ZeroMQ socket
    ///
    /// Parameters
    /// ----------
    /// send_timeout: int
    ///   The send timeout in milliseconds. Defaults to ``5000``
    ///
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

    /// Sets the receive timeout for the ZeroMQ socket
    ///
    /// Parameters
    /// ----------
    /// receive_timeout: int
    ///   The receive timeout in milliseconds. Defaults to ``1000``
    ///
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

    /// Sets the number of receive retries for the ZeroMQ socket when the other side replies.
    ///
    ///   * Dealer-Router pair: only on EOS
    ///   * Req-Rep pair: on every message
    ///   * Pub-Sub pair: never
    ///
    /// Parameters
    /// ----------
    /// receive_retries: int
    ///   The number of receive retries, defaults to ``3``.
    ///
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

    /// Sets the number of send retries for the ZeroMQ socket when the other side does not accept the message.
    ///
    /// Parameters
    /// ----------
    /// send_retries: int
    ///   The number of send retries, defaults to ``3``.
    ///
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

    /// Sets the send high water mark value for the ZeroMQ socket.
    ///
    /// Parameters
    /// ----------
    /// send_hwm: int
    ///   The send high water mark value, defaults to ``50``.
    ///
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

    /// Sets the receive high water mark value for the ZeroMQ socket.
    ///
    /// Parameters
    /// ----------
    /// receive_hwm: int
    ///   The receive high water mark value, defaults to ``50``.
    ///
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

    /// Fixes permissions to specified access mask for a domain unix socket. It may be required for sockets
    /// mapped to host volumes
    ///
    /// Parameters
    /// ----------
    /// fix_ipc_permissions: int
    ///   The access mask to set, defaults to ``0o777``.
    ///
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

    /// Builds the configuration
    ///
    /// Returns
    /// -------
    /// WriterConfig
    ///   The built configuration
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the configuration is invalid
    ///
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

/// Creates a new configuration builder based on the provided URL.
/// The URL can have the following formats:
///
///   * ``tcp://1.2.3.4:5678``
///   * ``ipc:///tmp/test``
///   * ``(sub|rep|router)+(bind|connect):(tcp|ipc)://...``
///
/// Parameters
/// ----------
/// url: str
///   The URL to use
///
/// Returns
/// -------
/// ReaderConfigBuilder
///
/// Raises
/// ------
/// ValueError
///   If the URL is invalid
///
#[pyclass]
#[derive(Debug, Clone)]
pub struct ReaderConfigBuilder(Option<zeromq::ReaderConfigBuilder>);

/// A reader configuration
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

    #[getter]
    fn endpoint(&self) -> String {
        self.0.endpoint().to_string()
    }

    #[getter]
    fn socket_type(&self) -> ReaderSocketType {
        self.0.socket_type().clone().into()
    }

    #[getter]
    fn bind(&self) -> bool {
        *self.0.bind()
    }

    #[getter]
    fn receive_timeout(&self) -> i32 {
        *self.0.receive_timeout()
    }

    #[getter]
    fn receive_hwm(&self) -> i32 {
        *self.0.receive_hwm()
    }

    #[getter]
    fn topic_prefix_spec(&self) -> TopicPrefixSpec {
        TopicPrefixSpec(self.0.topic_prefix_spec().clone())
    }

    #[getter]
    fn routing_cache_size(&self) -> usize {
        *self.0.routing_cache_size()
    }

    #[getter]
    fn fix_ipc_permissions(&self) -> Option<u32> {
        *self.0.fix_ipc_permissions()
    }

    #[getter]
    fn source_blacklist_size(&self) -> u64 {
        *self.0.source_blacklist_size()
    }

    #[getter]
    fn source_blacklist_ttl(&self) -> u64 {
        *self.0.source_blacklist_ttl()
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

    /// Sets the socket type
    ///
    /// Parameters
    /// ----------
    /// socket_type: :py:class:`ReaderSocketType`
    ///   The socket type to use, defaults to ``ReaderSocketType.Sub``
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the socket type is double set. Defaults to ``ReaderSocketType.Router``
    ///
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

    /// Specifies either socket binds or connects
    ///
    /// Parameters
    /// ----------
    /// bind: bool
    ///   If ``True``, the socket will bind, otherwise it will connect. Defaults to ``True``
    ///
    pub fn with_bind(&mut self, bind: bool) -> PyResult<()> {
        self.0 = Some(self.0.take().unwrap().with_bind(bind).map_err(|e| {
            PyValueError::new_err(format!("Failed to set ZeroMQ socket bind mode: {:?}", e))
        })?);
        Ok(())
    }

    /// Sets the receive timeout for the ZeroMQ socket.
    ///
    /// Parameters
    /// ----------
    /// receive_timeout: int
    ///   The receive timeout in milliseconds. Defaults to ``1000``
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the receive timeout is double set
    ///
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

    /// Sets the receive HWM on the ZeroMQ socket.
    ///
    /// Parameters
    /// ----------
    /// receive_hwm: int
    ///   The receive HWM, defaults to ``50``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the receive HWM is double set
    ///
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

    /// Sets the topic prefix spec for the ZeroMQ socket.
    ///
    /// Parameters
    /// ----------
    /// topic_prefix_spec: :py:class:`TopicPrefixSpec`
    ///   The topic prefix spec, defaults to ``TopicPrefixSpec.none()``
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the topic prefix spec is double set
    ///
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

    /// Sets the routing cache size for the ZeroMQ socket. The cache is used to track dealer-router connections
    /// chaos when the dealers reconnect with the same topic.
    ///
    /// Parameters
    /// ----------
    /// size: int
    ///   The routing cache size, defaults to ``512``.
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

    /// Fixes permissions to specified access mask for a domain unix socket. It may be required for sockets
    /// mapped to host volumes
    ///
    /// Parameters
    /// ----------
    /// fix_ipc_permissions: int
    ///   The access mask to set, defaults to ``0o777``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the permissions are double set
    ///
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

    /// Builds the configuration
    ///
    /// Returns
    /// -------
    /// ReaderConfig
    ///   The built configuration
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If the configuration is invalid
    ///
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

    /// Sets the source blacklist size for the ZeroMQ socket. The blacklist is used to block sources sending wrong data
    ///
    /// Parameters
    /// ----------
    /// size: int
    ///  The source blacklist size, defaults to ``1024``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///  If the source blacklist size is double set
    ///
    pub fn with_source_blacklist_size(&mut self, size: u64) -> PyResult<()> {
        self.0 = Some(
            self.0
                .take()
                .unwrap()
                .with_source_blacklist_size(NonZeroU64::new(size).ok_or(PyValueError::new_err(
                    "Failed to set ZeroMQ socket source blacklist size: size must be non-zero",
                ))?)
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "Failed to set ZeroMQ socket source blacklist size: {:?}",
                        e
                    ))
                })?,
        );
        Ok(())
    }

    /// Sets the source blacklist TTL for the ZeroMQ socket. The blacklist is used to track the sources sending wrong data
    ///
    /// Parameters
    /// ----------
    /// ttl: int
    ///  The source blacklist TTL, defaults to ``60000``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///  If the source blacklist TTL is double set
    ///
    pub fn with_source_blacklist_ttl(&mut self, ttl: u64) -> PyResult<()> {
        self.0 = Some(
            self.0
                .take()
                .unwrap()
                .with_source_blacklist_ttl(NonZeroU64::new(ttl).ok_or(PyValueError::new_err(
                    "Failed to set ZeroMQ socket source blacklist TTL: TTL must be non-zero",
                ))?)
                .map_err(|e| {
                    PyValueError::new_err(format!(
                        "Failed to set ZeroMQ socket source blacklist TTL: {:?}",
                        e
                    ))
                })?,
        );
        Ok(())
    }
}
