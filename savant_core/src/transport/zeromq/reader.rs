use crate::message::Message;
use crate::transport::zeromq::reader_config::Protocol;
use crate::transport::zeromq::{
    ReaderConfig, ReaderSocketType, RoutingIdFilter, CONFIRMATION_MESSAGE, END_OF_STREAM_MESSAGE,
    ZMQ_ACK_LINGER,
};
use anyhow::bail;
use log::{debug, info, warn};
use std::mem;
use std::os::unix::prelude::PermissionsExt;
use zmq::SocketType;

pub struct Reader {
    context: Option<zmq::Context>,
    config: ReaderConfig,
    socket: Option<Socket>,
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

#[derive(Debug)]
pub enum ReaderResult {
    Message {
        message: Message,
        topic: Vec<u8>,
        routing_id: Option<Vec<u8>>,
        data: Vec<Vec<u8>>,
    },
    EndOfStream {
        routing_id: Option<Vec<u8>>,
    },
    Timeout,
    PrefixMismatch {
        topic: Vec<u8>,
        routing_id: Option<Vec<u8>>,
    },
    RoutingIdMismatch {
        topic: Vec<u8>,
        routing_id: Option<Vec<u8>>,
    },
}

impl ReaderResult {
    pub fn message(
        message: Message,
        topic: &Vec<u8>,
        routing_id: &Option<&Vec<u8>>,
        data: &[Vec<u8>],
    ) -> Self {
        Self::Message {
            message,
            topic: topic.clone(),
            routing_id: routing_id.cloned(),
            data: data.to_vec(),
        }
    }
    pub fn end_of_stream(routing_id: &Option<&Vec<u8>>) -> Self {
        Self::EndOfStream {
            routing_id: routing_id.cloned(),
        }
    }

    pub fn prefix_mismatch(topic: &Vec<u8>, routing_id: &Option<&Vec<u8>>) -> Self {
        Self::PrefixMismatch {
            topic: topic.clone(),
            routing_id: routing_id.cloned(),
        }
    }

    pub fn routing_id_mismatch(topic: &Vec<u8>, routing_id: &Option<&Vec<u8>>) -> Self {
        Self::RoutingIdMismatch {
            topic: topic.clone(),
            routing_id: routing_id.cloned(),
        }
    }
}

enum Socket {
    ZmqSocket(zmq::Socket),
    MockSocket(Vec<Vec<u8>>),
}

impl Socket {
    fn send_multipart(&mut self, parts: &[&[u8]], flags: i32) -> anyhow::Result<()> {
        match self {
            Socket::ZmqSocket(socket) => socket.send_multipart(parts, flags).map_err(|e| e.into()),
            Socket::MockSocket(data) => {
                data.clear();
                data.extend(parts.iter().map(|p| p.to_vec()));
                Ok(())
            }
        }
    }

    fn send(&mut self, m: &[u8], flags: i32) -> anyhow::Result<()> {
        match self {
            Socket::ZmqSocket(socket) => socket.send(m, flags).map_err(|e| e.into()),
            Socket::MockSocket(data) => {
                data.clear();
                data.push(m.to_vec());
                Ok(())
            }
        }
    }

    fn recv_multipart(&mut self, flags: i32) -> Result<Vec<Vec<u8>>, zmq::Error> {
        match self {
            Socket::ZmqSocket(socket) => socket.recv_multipart(flags),
            Socket::MockSocket(data) => {
                let data = mem::replace(data, vec![]);
                Ok(data)
            }
        }
    }

    fn set_rcvhwm(&self, hwm: i32) -> anyhow::Result<()> {
        match self {
            Socket::ZmqSocket(socket) => socket.set_rcvhwm(hwm).map_err(|e| e.into()),
            Socket::MockSocket(_) => Ok(()),
        }
    }

    fn set_rcvtimeo(&self, timeout: i32) -> anyhow::Result<()> {
        match self {
            Socket::ZmqSocket(socket) => socket.set_rcvtimeo(timeout).map_err(|e| e.into()),
            Socket::MockSocket(_) => Ok(()),
        }
    }

    fn set_linger(&self, linger: i32) -> anyhow::Result<()> {
        match self {
            Socket::ZmqSocket(socket) => socket.set_linger(linger).map_err(|e| e.into()),
            Socket::MockSocket(_) => Ok(()),
        }
    }

    fn set_subscribe(&self, topic: &[u8]) -> anyhow::Result<()> {
        match self {
            Socket::ZmqSocket(socket) => socket.set_subscribe(topic).map_err(|e| e.into()),
            Socket::MockSocket(_) => Ok(()),
        }
    }

    fn bind(&self, endpoint: &str) -> anyhow::Result<()> {
        match self {
            Socket::ZmqSocket(socket) => socket.bind(endpoint).map_err(|e| e.into()),
            Socket::MockSocket(_) => Ok(()),
        }
    }

    fn connect(&self, endpoint: &str) -> anyhow::Result<()> {
        match self {
            Socket::ZmqSocket(socket) => socket.connect(endpoint).map_err(|e| e.into()),
            Socket::MockSocket(_) => Ok(()),
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

    #[cfg(not(test))]
    fn new_socket(config: &ReaderConfig, context: &zmq::Context) -> anyhow::Result<Socket> {
        Ok(Socket::ZmqSocket(
            context.socket(config.socket_type().into())?,
        ))
    }
    #[cfg(test)]
    fn new_socket(config: &ReaderConfig, context: &zmq::Context) -> anyhow::Result<Socket> {
        Ok(Socket::MockSocket(vec![]))
    }

    pub fn new(config: &ReaderConfig) -> anyhow::Result<Self> {
        let context = zmq::Context::new();
        let socket: Socket = Self::new_socket(config, &context)?;

        socket.set_rcvhwm(*config.receive_hwm())?;
        socket.set_rcvtimeo(*config.receive_timeout())?;
        socket.set_linger(ZMQ_ACK_LINGER)?;
        socket.set_subscribe(&config.topic_prefix_spec().get().as_bytes())?;

        #[cfg(not(test))]
        {
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
        }

        Ok(Self {
            context: Some(context),
            config: config.clone(),
            socket: Some(socket),
            routing_id_filter: RoutingIdFilter::new(*config.routing_ids_cache_size())?,
        })
    }

    pub fn destroy(&mut self) -> anyhow::Result<()> {
        info!(
            "Destroying ZeroMQ socket for endpoint {}",
            self.config.endpoint()
        );
        self.socket.take();
        self.context.take();
        info!(
            "ZeroMQ socket for endpoint {} destroyed",
            self.config.endpoint()
        );
        Ok(())
    }

    pub fn is_alive(&self) -> bool {
        self.socket.is_some()
    }

    pub fn receive(&mut self) -> anyhow::Result<ReaderResult> {
        if self.socket.is_none() {
            bail!(
                "ZeroMQ socket for endpoint {} is no longer available, because it was destroyed.",
                self.config.endpoint()
            );
        }
        let socket = self.socket.as_mut().unwrap();
        let parts = socket.recv_multipart(0);
        if let Err(e) = parts {
            warn!(
                "Failed to receive message from ZeroMQ socket. Error is {:?}",
                e
            );
            if let zmq::Error::EAGAIN = e {
                return Ok(ReaderResult::Timeout);
            } else {
                bail!(
                    "Failed to receive message from ZeroMQ socket. Error is {:?}",
                    e
                );
            }
        }

        let parts = parts.unwrap();

        let (routing_id, message) = if self.config.socket_type() == &ReaderSocketType::Router {
            let routing_id = &parts[0];
            let message = &parts[1..];
            (Some(routing_id), message)
        } else {
            (None, &parts[..])
        };

        if message[0] == END_OF_STREAM_MESSAGE {
            debug!(
                "Received end of stream message from ZeroMQ socket for endpoint {}",
                self.config.endpoint()
            );
            if let Some(routing_id) = routing_id {
                socket.send_multipart(&[routing_id, CONFIRMATION_MESSAGE], 0)?;
            } else {
                socket.send(CONFIRMATION_MESSAGE, 0)?;
            }

            if self.config.socket_type() == &ReaderSocketType::Rep {
                socket.send(CONFIRMATION_MESSAGE, 0)?;
            }
            return Ok(ReaderResult::end_of_stream(&routing_id));
        }

        let topic = &message[0];
        if message.len() < 2 {
            bail!(
                "Received message with invalid number of parts from ZeroMQ socket for endpoint {}. Expected at least 2 parts, but got {}",
                self.config.endpoint(),
                message.len()
            );
        }
        if !self.config.topic_prefix_spec().matches(topic) {
            debug!(
                "Received message with invalid topic from ZeroMQ socket for endpoint {}. Expected topic to match spec {:?}, but got {:?}",
                self.config.endpoint(),
                self.config.topic_prefix_spec(),
                topic
            );

            return Ok(ReaderResult::PrefixMismatch {
                topic: topic.to_vec(),
                routing_id: routing_id.cloned(),
            });
        }

        if self.routing_id_filter.allow(topic, &routing_id) {
            Ok(ReaderResult::Message {
                message: match self.config.protocol() {
                    Protocol::SavantRs => crate::message::load_message(&message[1]),
                    Protocol::Protobuf => crate::protobuf::deserialize(&message[1])?,
                },
                topic: topic.to_vec(),
                routing_id: routing_id.cloned(),
                data: if message.len() > 2 {
                    message[2..].to_vec()
                } else {
                    vec![]
                },
            })
        } else {
            debug!(
                "Received message with invalid routing ID from ZeroMQ socket for endpoint {}. Got topic = {:?}, routing_id = {:?}",
                self.config.endpoint(),
                topic,
                routing_id
            );

            Ok(ReaderResult::routing_id_mismatch(topic, &routing_id))
        }
    }
}

#[cfg(test)]
mod tests {
    mod router_tests {
        use crate::message::Message;
        use crate::primitives::eos::EndOfStream;
        use crate::transport::zeromq::reader::ReaderResult;
        use crate::transport::zeromq::{Reader, ReaderConfig, TopicPrefixSpec};

        #[test]
        fn test_ok() -> anyhow::Result<()> {
            let conf = ReaderConfig::new()
                .with_endpoint("router+bind:ipc:///tmp/test")?
                .with_topic_prefix_spec(TopicPrefixSpec::SourceId("topic".into()))?
                .build()?;

            let mut reader = Reader::new(&conf)?;
            let message = Message::end_of_stream(EndOfStream::new("test".to_string()));
            let binary = crate::message::save_message(&message);

            reader
                .socket
                .as_mut()
                .unwrap()
                .send_multipart(&[b"routing-id", b"topic", &binary, &vec![0x0, 0x1, 0x2]], 0)?;

            let m = reader.receive()?;
            assert!(matches!(
                &m,
                ReaderResult::Message {
                    message,
                    topic,
                    routing_id,
                    data
                } if message.is_end_of_stream() && topic == b"topic" && routing_id == &Some(b"routing-id".to_vec()) && data == &vec![vec![0x0, 0x1, 0x2]]
            ));
            Ok(())
        }
    }
}
