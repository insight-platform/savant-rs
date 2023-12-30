use crate::transport::zeromq::{ReaderConfig, ReaderSocketType, RoutingIdFilter, ZMQ_ACK_LINGER};
use anyhow::bail;
use std::os::unix::prelude::PermissionsExt;

pub struct Reader {
    context: Option<zmq::Context>,
    socket: Option<zmq::Socket>,
    routing_id_filter: RoutingIdFilter,
}

impl From<&ReaderSocketType> for zmq::SocketType {
    fn from(socket_type: &ReaderSocketType) -> Self {
        match socket_type {
            ReaderSocketType::Sub => zmq::SocketType::SUB,
            ReaderSocketType::Router => zmq::SocketType::ROUTER,
            ReaderSocketType::Rep => zmq::SocketType::REP,
        }
    }
}

impl Reader {
    fn create_ipc_dirs(endpoint: &str) -> anyhow::Result<()> {
        let endpoint = endpoint.strip_prefix("ipc://").unwrap();
        if endpoint.is_empty() {
            bail!("Invalid IPC endpoint: {}", endpoint);
        }
        let path = std::path::Path::new(endpoint);
        if !path.is_file() {
            bail!("IPC endpoint is not a file: {}", endpoint);
        }
        let parent = path.parent().unwrap();
        std::fs::create_dir_all(parent)?;
        Ok(())
    }

    fn set_ipc_permissions(endpoint: &str, permissions: u32) -> anyhow::Result<()> {
        let endpoint = endpoint.strip_prefix("ipc://").unwrap();
        if endpoint.is_empty() {
            bail!("Invalid IPC endpoint: {}", endpoint);
        }
        let path = std::path::Path::new(endpoint);
        if !path.is_file() {
            bail!("IPC endpoint is not a file: {}", endpoint);
        }
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(permissions))?;
        Ok(())
    }

    pub fn new(config: &ReaderConfig) -> anyhow::Result<Self> {
        let context = zmq::Context::new();
        let socket = context.socket(config.socket_type().into())?;

        socket.set_rcvhwm(*config.receive_hwm())?;
        socket.set_rcvtimeo(*config.receive_timeout())?;
        socket.set_linger(ZMQ_ACK_LINGER)?;
        socket.set_subscribe(&config.topic_prefix().get().as_bytes())?;

        if config.endpoint().starts_with("ipc://") {
            Self::create_ipc_dirs(config.endpoint())?;
            if let Some(permissions) = config.fix_ipc_permissions() {
                Self::set_ipc_permissions(config.endpoint(), *permissions)?;
            }
        }

        if *config.bind() {
            socket.bind(config.endpoint())?;
        } else {
            socket.connect(config.endpoint())?;
        }

        Ok(Self {
            context: Some(context),
            socket: Some(socket),
            routing_id_filter: RoutingIdFilter::new(*config.routing_ids_cache_size())?,
        })
    }

    pub fn destroy(&mut self) -> anyhow::Result<()> {
        self.socket.take();
        self.context.take();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test() {
        assert_eq!(1, 1);
    }
}
