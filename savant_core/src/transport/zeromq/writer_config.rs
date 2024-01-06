use super::{
    parse_zmq_socket_uri, SocketType, WriterSocketType, IPC_PERMISSIONS, REQ_RECEIVE_RETRIES,
    SENDER_RECEIVE_TIMEOUT, SEND_HWM, SEND_TIMEOUT,
};
use anyhow::bail;
use savant_utils::default_once::DefaultOnceCell;

#[derive(Clone, Debug, Default)]
pub struct WriterConfig(WriterConfigBuilder);

impl WriterConfig {
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> WriterConfigBuilder {
        WriterConfigBuilder::default()
    }

    pub fn endpoint(&self) -> &String {
        self.0.endpoint.get_or_init()
    }

    pub fn socket_type(&self) -> &WriterSocketType {
        self.0.socket_type.get_or_init()
    }

    pub fn bind(&self) -> &bool {
        self.0.bind.get_or_init()
    }

    pub fn send_timeout(&self) -> &i32 {
        self.0.send_timeout.get_or_init()
    }

    pub fn receive_timeout(&self) -> &i32 {
        self.0.receive_timeout.get_or_init()
    }

    pub fn receive_retries(&self) -> &i32 {
        self.0.receive_retries.get_or_init()
    }

    pub fn send_hwm(&self) -> &i32 {
        self.0.send_hwm.get_or_init()
    }

    pub fn fix_ipc_permissions(&self) -> &Option<u32> {
        self.0.fix_ipc_permissions.get_or_init()
    }
}

#[derive(Clone, Debug)]
pub struct WriterConfigBuilder {
    endpoint: DefaultOnceCell<String>,
    socket_type: DefaultOnceCell<WriterSocketType>,
    bind: DefaultOnceCell<bool>,
    send_timeout: DefaultOnceCell<i32>,
    receive_timeout: DefaultOnceCell<i32>,
    receive_retries: DefaultOnceCell<i32>,
    send_hwm: DefaultOnceCell<i32>,
    fix_ipc_permissions: DefaultOnceCell<Option<u32>>,
}

impl Default for WriterConfigBuilder {
    fn default() -> Self {
        Self {
            endpoint: DefaultOnceCell::new(String::new()),
            socket_type: DefaultOnceCell::new(WriterSocketType::Dealer),
            bind: DefaultOnceCell::new(true),
            send_timeout: DefaultOnceCell::new(SEND_TIMEOUT),
            receive_timeout: DefaultOnceCell::new(SENDER_RECEIVE_TIMEOUT),
            receive_retries: DefaultOnceCell::new(REQ_RECEIVE_RETRIES),
            send_hwm: DefaultOnceCell::new(SEND_HWM),
            fix_ipc_permissions: DefaultOnceCell::new(Some(IPC_PERMISSIONS)),
        }
    }
}

impl WriterConfigBuilder {
    pub fn build(self) -> anyhow::Result<WriterConfig> {
        if self.endpoint.get_or_init().is_empty() {
            bail!("ZeroMQ endpoint is not set");
        }
        Ok(WriterConfig(self))
    }
    pub fn url(self, url: &str) -> anyhow::Result<Self> {
        let uri = parse_zmq_socket_uri(url.to_string())?;
        self.endpoint.set(uri.endpoint)?;
        if let Some(bind) = uri.bind {
            self.bind.set(bind)?;
        }
        if let Some(socket_type) = uri.socket_type {
            self.socket_type.set(match socket_type {
                SocketType::Writer(socket_type) => socket_type,
                _ => bail!("Invalid socket type for writer: {:?}", socket_type),
            })?;
        }
        Ok(self)
    }

    pub fn with_endpoint(self, endpoint: &str) -> anyhow::Result<Self> {
        self.endpoint.set(endpoint.to_string())?;
        Ok(self)
    }

    pub fn with_socket_type(self, socket_type: WriterSocketType) -> anyhow::Result<Self> {
        self.socket_type.set(socket_type)?;
        Ok(self)
    }

    pub fn with_bind(self, bind: bool) -> anyhow::Result<Self> {
        self.bind.set(bind)?;
        Ok(self)
    }

    pub fn with_send_timeout(self, send_timeout: i32) -> anyhow::Result<Self> {
        if send_timeout <= 0 {
            bail!("Send timeout must be non-negative");
        }
        self.send_timeout.set(send_timeout)?;
        Ok(self)
    }

    pub fn with_receive_timeout(self, receive_timeout: i32) -> anyhow::Result<Self> {
        if receive_timeout <= 0 {
            bail!("Receive timeout must be non-negative");
        }
        self.receive_timeout.set(receive_timeout)?;
        Ok(self)
    }

    pub fn with_receive_retries(self, receive_retries: i32) -> anyhow::Result<Self> {
        if receive_retries < 0 {
            bail!("Receive retries must be non-negative");
        }
        self.receive_retries.set(receive_retries)?;
        Ok(self)
    }

    pub fn with_send_hwm(self, send_hwm: i32) -> anyhow::Result<Self> {
        if send_hwm <= 0 {
            bail!("Receive HWM must be non-negative.");
        }
        self.send_hwm.set(send_hwm)?;
        Ok(self)
    }

    pub fn with_fix_ipc_permissions(
        self,
        fix_ipc_permissions: Option<u32>,
    ) -> anyhow::Result<Self> {
        self.fix_ipc_permissions.set(fix_ipc_permissions)?;
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use crate::transport::zeromq::writer_config::WriterConfig;
    use crate::transport::zeromq::WriterSocketType;

    #[test]
    fn test_writer_config_with_endpoint() -> anyhow::Result<()> {
        let url = "tcp:///abc";
        let config = WriterConfig::new().with_endpoint(&url)?.build()?;
        assert_eq!(config.endpoint(), url);
        Ok(())
    }

    #[test]
    fn test_duplicate_configuration_fails() -> anyhow::Result<()> {
        let config = WriterConfig::new().with_endpoint("tcp:///abc")?;
        assert!(config.with_endpoint("tcp:///abc").is_err());
        Ok(())
    }

    #[test]
    fn test_build_empty_config_fails() -> anyhow::Result<()> {
        let config = WriterConfig::new();
        assert!(config.build().is_err());
        Ok(())
    }

    #[test]
    fn test_full_uri() -> anyhow::Result<()> {
        let endpoint = "ipc:///abc/def";
        let url = format!("pub+connect:{}", endpoint);
        let config = WriterConfig::new().url(&url)?.build()?;
        assert_eq!(config.endpoint(), &endpoint);
        assert_eq!(config.bind(), &false);
        assert_eq!(config.socket_type(), &WriterSocketType::Pub);
        Ok(())
    }

    #[test]
    fn test_reader_results_in_error() -> anyhow::Result<()> {
        let endpoint = String::from("ipc:///abc/def");
        let url = format!("sub+connect:{}", endpoint);
        let config = WriterConfig::new().url(&url);
        assert!(config.is_err());
        Ok(())
    }
}
