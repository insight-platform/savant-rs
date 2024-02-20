use super::{
    parse_zmq_socket_uri, ReaderSocketType, SocketType, TopicPrefixSpec, IPC_PERMISSIONS,
    RECEIVE_HWM, RECEIVE_TIMEOUT, ROUTING_ID_CACHE_SIZE,
};
use crate::utils::default_once::DefaultOnceCell;
use anyhow::bail;

#[derive(Clone, Debug, Default)]
pub struct ReaderConfig(ReaderConfigBuilder);

impl ReaderConfig {
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> ReaderConfigBuilder {
        ReaderConfigBuilder::default()
    }

    pub fn endpoint(&self) -> &String {
        self.0.endpoint.get_or_init()
    }

    pub fn socket_type(&self) -> &ReaderSocketType {
        self.0.socket_type.get_or_init()
    }

    pub fn bind(&self) -> &bool {
        self.0.bind.get_or_init()
    }

    pub fn receive_timeout(&self) -> &i32 {
        self.0.receive_timeout.get_or_init()
    }

    pub fn receive_hwm(&self) -> &i32 {
        self.0.receive_hwm.get_or_init()
    }

    pub fn topic_prefix_spec(&self) -> &TopicPrefixSpec {
        self.0.topic_prefix_spec.get_or_init()
    }

    pub fn routing_cache_size(&self) -> &usize {
        self.0.routing_ids_cache_size.get_or_init()
    }

    pub fn fix_ipc_permissions(&self) -> &Option<u32> {
        self.0.fix_ipc_permissions.get_or_init()
    }
}

#[derive(Clone, Debug)]
pub struct ReaderConfigBuilder {
    endpoint: DefaultOnceCell<String>,
    socket_type: DefaultOnceCell<ReaderSocketType>,
    bind: DefaultOnceCell<bool>,
    receive_timeout: DefaultOnceCell<i32>,
    receive_hwm: DefaultOnceCell<i32>,
    topic_prefix_spec: DefaultOnceCell<TopicPrefixSpec>,
    routing_ids_cache_size: DefaultOnceCell<usize>,
    fix_ipc_permissions: DefaultOnceCell<Option<u32>>,
}

impl Default for ReaderConfigBuilder {
    fn default() -> Self {
        Self {
            endpoint: DefaultOnceCell::new(String::new()),
            socket_type: DefaultOnceCell::new(ReaderSocketType::Router),
            bind: DefaultOnceCell::new(true),
            receive_timeout: DefaultOnceCell::new(RECEIVE_TIMEOUT),
            receive_hwm: DefaultOnceCell::new(RECEIVE_HWM),
            topic_prefix_spec: DefaultOnceCell::new(TopicPrefixSpec::None),
            routing_ids_cache_size: DefaultOnceCell::new(ROUTING_ID_CACHE_SIZE),
            fix_ipc_permissions: DefaultOnceCell::new(Some(IPC_PERMISSIONS)),
        }
    }
}

impl ReaderConfigBuilder {
    pub fn build(self) -> anyhow::Result<ReaderConfig> {
        if self.endpoint.get_or_init().is_empty() {
            bail!("ZeroMQ endpoint is not set");
        }
        Ok(ReaderConfig(self))
    }
    pub fn url(self, url: &str) -> anyhow::Result<Self> {
        let uri = parse_zmq_socket_uri(url.to_string())?;
        self.endpoint.set(uri.endpoint)?;
        if let Some(bind) = uri.bind {
            self.bind.set(bind)?;
        }
        if let Some(socket_type) = uri.socket_type {
            self.socket_type.set(match socket_type {
                SocketType::Reader(socket_type) => socket_type,
                _ => bail!("Invalid socket type for reader: {:?}", socket_type),
            })?;
        }
        Ok(self)
    }

    pub fn with_socket_type(self, socket_type: ReaderSocketType) -> anyhow::Result<Self> {
        self.socket_type.set(socket_type)?;
        Ok(self)
    }

    pub fn with_receive_timeout(self, receive_timeout: i32) -> anyhow::Result<Self> {
        if receive_timeout <= 0 {
            bail!("Receive timeout must be non-negative");
        }
        self.receive_timeout.set(receive_timeout)?;
        Ok(self)
    }

    pub fn with_receive_hwm(self, receive_hwm: i32) -> anyhow::Result<Self> {
        if receive_hwm <= 0 {
            bail!("Receive HWM must be non-negative.");
        }
        self.receive_hwm.set(receive_hwm)?;
        Ok(self)
    }

    pub fn with_topic_prefix_spec(self, prefix: TopicPrefixSpec) -> anyhow::Result<Self> {
        self.topic_prefix_spec.set(prefix)?;
        Ok(self)
    }

    pub fn with_routing_cache_size(self, size: usize) -> anyhow::Result<Self> {
        self.routing_ids_cache_size.set(size)?;
        Ok(self)
    }

    pub fn with_bind(self, bind: bool) -> anyhow::Result<Self> {
        self.bind.set(bind)?;
        Ok(self)
    }

    pub fn with_fix_ipc_permissions(self, permissions: Option<u32>) -> anyhow::Result<Self> {
        if !self.bind.get_or_init() {
            bail!("IPC permissions can only be set for bind sockets.");
        }
        self.fix_ipc_permissions.set(permissions)?;
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use crate::transport::zeromq::reader_config::ReaderConfig;
    use crate::transport::zeromq::ReaderSocketType;

    #[test]
    fn test_reader_config_with_endpoint() -> anyhow::Result<()> {
        let url = String::from("tcp://1.1.1.1:1234");
        let config = ReaderConfig::new().url(&url)?.build()?;
        assert_eq!(config.endpoint(), &url);
        Ok(())
    }

    #[test]
    fn test_duplicate_configuration_fails() -> anyhow::Result<()> {
        let config = ReaderConfig::new().url("tcp://1.1.1.1:1234")?;
        assert!(config.url("tcp://1.1.1.1:1234").is_err());
        Ok(())
    }

    #[test]
    fn test_build_empty_config_fails() -> anyhow::Result<()> {
        let config = ReaderConfig::new();
        assert!(config.build().is_err());
        Ok(())
    }

    #[test]
    fn test_full_uri() -> anyhow::Result<()> {
        let endpoint = String::from("ipc:///abc/def");
        let url = format!("sub+connect:{}", endpoint);
        let config = ReaderConfig::new().url(&url)?.build()?;
        assert_eq!(config.endpoint(), &endpoint);
        assert_eq!(config.bind(), &false);
        assert_eq!(config.socket_type(), &ReaderSocketType::Sub);
        Ok(())
    }

    #[test]
    fn test_writer_results_in_error() -> anyhow::Result<()> {
        let endpoint = String::from("ipc:///abc/def");
        let url = format!("pub+connect:{}", endpoint);
        let config = ReaderConfig::new().url(&url);
        assert!(config.is_err());
        Ok(())
    }

    #[test]
    fn set_fix_perms_without_bind_fails() -> anyhow::Result<()> {
        let config = ReaderConfig::new()
            .with_bind(false)?
            .with_fix_ipc_permissions(Some(0777));
        assert!(config.is_err());
        Ok(())
    }

    #[test]
    fn set_fix_ipc_permissions_with_bind_ok() -> anyhow::Result<()> {
        let _ = ReaderConfig::new()
            .with_bind(true)?
            .with_fix_ipc_permissions(Some(0777))?;
        Ok(())
    }
}
