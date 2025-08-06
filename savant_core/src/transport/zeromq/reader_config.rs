use super::{
    parse_zmq_socket_uri, ReaderSocketType, SocketType, TopicPrefixSpec, IPC_PERMISSIONS,
    RECEIVE_HWM, RECEIVE_TIMEOUT, ROUTING_ID_CACHE_SIZE, SOURCE_BLACKLIST_CACHE_EXPIRATION,
    SOURCE_BLACKLIST_CACHE_SIZE,
};
use crate::utils::default_once::DefaultOnceCell;
use anyhow::bail;
use hashbrown::HashMap;
use std::{num::NonZeroU64, str::FromStr};

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

    pub fn source_blacklist_size(&self) -> &u64 {
        self.0.source_blacklist_size.get_or_init()
    }

    pub fn source_blacklist_ttl(&self) -> &u64 {
        self.0.source_blacklist_ttl.get_or_init()
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
    source_blacklist_size: DefaultOnceCell<u64>,
    source_blacklist_ttl: DefaultOnceCell<u64>,
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
            source_blacklist_size: DefaultOnceCell::new(SOURCE_BLACKLIST_CACHE_SIZE),
            source_blacklist_ttl: DefaultOnceCell::new(SOURCE_BLACKLIST_CACHE_EXPIRATION),
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

    #[cfg(test)]
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

    pub fn with_source_blacklist_size(self, size: NonZeroU64) -> anyhow::Result<Self> {
        self.source_blacklist_size.set(size.get())?;
        Ok(self)
    }

    pub fn with_source_blacklist_ttl(self, ttl: NonZeroU64) -> anyhow::Result<Self> {
        self.source_blacklist_ttl.set(ttl.get())?;
        Ok(self)
    }

    pub fn with_map_config(self, map: HashMap<String, String>) -> anyhow::Result<Self> {
        // Handle URL validation more efficiently
        match (self.endpoint.is_initialized(), map.get("url")) {
            (true, Some(_)) => bail!("The 'url' already set. Exclude it from the map."),
            (false, None) => {
                bail!("The 'url' field value is required before building the reader config")
            }
            (_, url_opt) => {
                let mut builder = if let Some(url) = url_opt {
                    self.url(url)?
                } else {
                    self
                };

                // Process remaining configuration options
                for (key, value) in map.iter().filter(|(k, _)| *k != "url") {
                    builder = match key.as_str() {
                        "receive_timeout" => builder.with_receive_timeout(value.parse()?),
                        "receive_hwm" => builder.with_receive_hwm(value.parse()?),
                        "topic_prefix_spec" => {
                            builder.with_topic_prefix_spec(TopicPrefixSpec::from_str(value)?)
                        }
                        "routing_ids_cache_size" => builder.with_routing_cache_size(value.parse()?),
                        "fix_ipc_permissions" => {
                            builder.with_fix_ipc_permissions(Some(u32::from_str_radix(value, 8)?))
                        }
                        "source_blacklist_size" => {
                            let size = NonZeroU64::new(value.parse()?).ok_or_else(|| {
                                anyhow::anyhow!("Source blacklist size must be non-zero")
                            })?;
                            builder.with_source_blacklist_size(size)
                        }
                        "source_blacklist_ttl" => {
                            let ttl = NonZeroU64::new(value.parse()?).ok_or_else(|| {
                                anyhow::anyhow!("Source blacklist ttl must be non-zero")
                            })?;
                            builder.with_source_blacklist_ttl(ttl)
                        }
                        _ => bail!("Invalid field: {}", key),
                    }?;
                }
                Ok(builder)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use hashbrown::HashMap;

    use crate::transport::zeromq::reader_config::ReaderConfig;
    use crate::transport::zeromq::{ReaderSocketType, TopicPrefixSpec};

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
        let url = format!("sub+connect:{endpoint}");
        let config = ReaderConfig::new().url(&url)?.build()?;
        assert_eq!(config.endpoint(), &endpoint);
        assert_eq!(config.bind(), &false);
        assert_eq!(config.socket_type(), &ReaderSocketType::Sub);
        Ok(())
    }

    #[test]
    fn test_writer_results_in_error() -> anyhow::Result<()> {
        let endpoint = String::from("ipc:///abc/def");
        let url = format!("pub+connect:{endpoint}");
        let config = ReaderConfig::new().url(&url);
        assert!(config.is_err());
        Ok(())
    }

    #[test]
    fn set_fix_perms_without_bind_fails() -> anyhow::Result<()> {
        let config = ReaderConfig::new()
            .with_bind(false)?
            .with_fix_ipc_permissions(Some(0o777));
        assert!(config.is_err());
        Ok(())
    }

    #[test]
    fn set_fix_ipc_permissions_with_bind_ok() -> anyhow::Result<()> {
        let _ = ReaderConfig::new()
            .with_bind(true)?
            .with_fix_ipc_permissions(Some(0o777))?;
        Ok(())
    }

    #[test]
    fn test_try_from_map_full_config() -> anyhow::Result<()> {
        let map = HashMap::from([
            ("url".to_string(), "sub+bind:tcp://1.1.1.1:1234".to_string()),
            ("receive_timeout".to_string(), "1000".to_string()),
            ("receive_hwm".to_string(), "1000".to_string()),
            (
                "topic_prefix_spec".to_string(),
                "source{source}".to_string(),
            ),
            ("routing_ids_cache_size".to_string(), "1000".to_string()),
            ("fix_ipc_permissions".to_string(), "0753".to_string()),
            ("source_blacklist_size".to_string(), "1000".to_string()),
            ("source_blacklist_ttl".to_string(), "1000".to_string()),
        ]);
        let builder = ReaderConfig::new().with_map_config(map)?;
        let config = builder.build()?;
        assert_eq!(config.endpoint().as_str(), "tcp://1.1.1.1:1234");
        assert_eq!(config.bind(), &true);
        assert_eq!(config.socket_type(), &ReaderSocketType::Sub);
        assert_eq!(config.receive_timeout(), &1000);
        assert_eq!(config.receive_hwm(), &1000);
        assert!(matches!(
            config.topic_prefix_spec(),
            TopicPrefixSpec::SourceId(source) if source == "source"
        ));
        assert_eq!(config.routing_cache_size(), &1000);
        assert_eq!(config.fix_ipc_permissions(), &Some(0o753));
        assert_eq!(config.source_blacklist_size(), &1000);
        assert_eq!(config.source_blacklist_ttl(), &1000);
        Ok(())
    }

    #[test]
    fn test_try_from_map_partial_double_url_set_config() -> anyhow::Result<()> {
        let builder = ReaderConfig::new().url("sub+bind:tcp://1.1.1.1:1234")?;
        let map = HashMap::from([("url".to_string(), "sub+bind:tcp://1.1.1.1:1234".to_string())]);
        let builder = builder.with_map_config(map);
        assert!(builder.is_err());
        Ok(())
    }

    #[test]
    fn test_try_from_map_no_url_at_all_config() -> anyhow::Result<()> {
        let map = HashMap::from([("receive_timeout".to_string(), "1000".to_string())]);
        let builder = ReaderConfig::new().with_map_config(map);
        assert!(builder.is_err());
        Ok(())
    }

    #[test]
    fn test_try_from_map_explicit_url_and_partial_config() -> anyhow::Result<()> {
        let map = HashMap::from([
            ("receive_timeout".to_string(), "1000".to_string()),
            ("fix_ipc_permissions".to_string(), "0755".to_string()),
        ]);
        let _ = ReaderConfig::new()
            .url("sub+bind:tcp://1.1.1.1:1234")?
            .with_map_config(map)?;
        Ok(())
    }
}
