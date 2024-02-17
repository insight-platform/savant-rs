use super::{
    parse_zmq_socket_uri, SocketType, WriterSocketType, ACK_RECEIVE_RETRIES, IPC_PERMISSIONS,
    RECEIVE_HWM, SENDER_RECEIVE_TIMEOUT, SEND_HWM, SEND_RETRIES, SEND_TIMEOUT,
};
use crate::utils::default_once::DefaultOnceCell;
use anyhow::bail;

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

    pub fn send_retries(&self) -> &i32 {
        self.0.send_retries.get_or_init()
    }

    pub fn send_hwm(&self) -> &i32 {
        self.0.send_hwm.get_or_init()
    }

    pub fn receive_hwm(&self) -> &i32 {
        self.0.receive_hwm.get_or_init()
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
    send_retries: DefaultOnceCell<i32>,
    receive_timeout: DefaultOnceCell<i32>,
    receive_retries: DefaultOnceCell<i32>,
    send_hwm: DefaultOnceCell<i32>,
    receive_hwm: DefaultOnceCell<i32>,
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
            receive_retries: DefaultOnceCell::new(ACK_RECEIVE_RETRIES),
            send_retries: DefaultOnceCell::new(SEND_RETRIES),
            send_hwm: DefaultOnceCell::new(SEND_HWM),
            receive_hwm: DefaultOnceCell::new(RECEIVE_HWM),
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

    pub fn with_socket_type(self, socket_type: WriterSocketType) -> anyhow::Result<Self> {
        self.socket_type.set(socket_type)?;
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

    pub fn with_send_retries(self, send_retries: i32) -> anyhow::Result<Self> {
        if send_retries < 0 {
            bail!("Send retries must be non-negative");
        }
        self.send_retries.set(send_retries)?;
        Ok(self)
    }

    pub fn with_send_hwm(self, send_hwm: i32) -> anyhow::Result<Self> {
        if send_hwm <= 0 {
            bail!("Receive HWM must be non-negative.");
        }
        self.send_hwm.set(send_hwm)?;
        Ok(self)
    }

    pub fn with_receive_hwm(self, receive_hwm: i32) -> anyhow::Result<Self> {
        if receive_hwm <= 0 {
            bail!("Receive HWM must be non-negative.");
        }
        self.receive_hwm.set(receive_hwm)?;
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
    use crate::transport::zeromq::writer_config::WriterConfig;
    use crate::transport::zeromq::WriterSocketType;

    #[test]
    fn test_duplicate_configuration_fails() -> anyhow::Result<()> {
        let config = WriterConfig::new().url("tcp:///abc")?;
        assert!(config.url("tcp:///abc").is_err());
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

    #[test]
    fn set_fix_perms_without_bind_fails() -> anyhow::Result<()> {
        let config = WriterConfig::new()
            .with_bind(false)?
            .with_fix_ipc_permissions(Some(0777));
        assert!(config.is_err());
        Ok(())
    }

    #[test]
    fn set_fix_ipc_permissions_with_bind_ok() -> anyhow::Result<()> {
        let _ = WriterConfig::new()
            .with_bind(true)?
            .with_fix_ipc_permissions(Some(0777))?;
        Ok(())
    }
}
