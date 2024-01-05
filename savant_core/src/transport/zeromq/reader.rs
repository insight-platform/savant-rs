use crate::message::Message;
use crate::transport::zeromq::reader_config::Protocol;
use crate::transport::zeromq::{
    ReaderConfig, ReaderSocketType, RoutingIdFilter, Socket, CONFIRMATION_MESSAGE,
    END_OF_STREAM_MESSAGE, ZMQ_ACK_LINGER,
};
use crate::TEST_ENV;
use anyhow::bail;
use log::{debug, info, warn};
use std::os::unix::prelude::PermissionsExt;

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
        message: Box<Message>,
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
    TooShort(Vec<Vec<u8>>),
}

impl ReaderResult {
    pub fn message(
        message: Message,
        topic: &[u8],
        routing_id: &Option<&Vec<u8>>,
        data: &[Vec<u8>],
    ) -> Self {
        Self::Message {
            message: Box::new(message),
            topic: topic.to_vec(),
            routing_id: routing_id.cloned(),
            data: data.to_vec(),
        }
    }
    pub fn end_of_stream(routing_id: &Option<&Vec<u8>>) -> Self {
        Self::EndOfStream {
            routing_id: routing_id.cloned(),
        }
    }

    pub fn prefix_mismatch(topic: &[u8], routing_id: &Option<&Vec<u8>>) -> Self {
        Self::PrefixMismatch {
            topic: topic.to_vec(),
            routing_id: routing_id.cloned(),
        }
    }

    pub fn routing_id_mismatch(topic: &[u8], routing_id: &Option<&Vec<u8>>) -> Self {
        Self::RoutingIdMismatch {
            topic: topic.to_vec(),
            routing_id: routing_id.cloned(),
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
    fn new_socket(_config: &ReaderConfig, _context: &zmq::Context) -> anyhow::Result<Socket> {
        Ok(Socket::MockSocket(vec![]))
    }

    pub fn new(config: &ReaderConfig) -> anyhow::Result<Self> {
        let context = zmq::Context::new();
        let socket: Socket = Self::new_socket(config, &context)?;

        socket.set_rcvhwm(*config.receive_hwm())?;
        socket.set_rcvtimeo(*config.receive_timeout())?;
        socket.set_linger(ZMQ_ACK_LINGER)?;
        socket.set_subscribe(config.topic_prefix_spec().get().as_bytes())?;

        if !TEST_ENV && config.endpoint().starts_with("ipc://") {
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

        let min_required_parts = match self.config.socket_type() {
            ReaderSocketType::Sub => 1,
            ReaderSocketType::Router => 2,
            ReaderSocketType::Rep => 1,
        };

        if parts.len() < min_required_parts {
            warn!(
                target: "savant_rs.zeromq.reader",
                "Received message with invalid number of parts from ZeroMQ socket for endpoint {}. Expected at least {} parts, but got {}",
                self.config.endpoint(),
                min_required_parts,
                parts.len()
            );
            return Ok(ReaderResult::TooShort(parts));
        };

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

            return Ok(ReaderResult::end_of_stream(&routing_id));
        }

        if self.config.socket_type() == &ReaderSocketType::Rep {
            socket.send(CONFIRMATION_MESSAGE, 0)?;
        }

        if message.len() < 2 {
            info!(
                "Received message with invalid number of parts from ZeroMQ socket for endpoint {}. Expected at least 2 parts, but got {}",
                self.config.endpoint(),
                message.len()
            );
            return Ok(ReaderResult::TooShort(parts));
        }
        let topic = &message[0];
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
                message: Box::new(match self.config.protocol() {
                    Protocol::SavantRs => crate::message::load_message(&message[1]),
                    Protocol::Protobuf => crate::protobuf::deserialize(&message[1])?,
                }),
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
        use crate::transport::zeromq::{
            Reader, ReaderConfig, TopicPrefixSpec, CONFIRMATION_MESSAGE, END_OF_STREAM_MESSAGE,
        };

        #[test]
        fn test_ok() -> anyhow::Result<()> {
            let conf = ReaderConfig::new()
                .url("router+bind:ipc:///tmp/test")?
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
            assert_eq!(
                reader.socket.as_mut().unwrap().take_buffer(),
                Vec::<Vec<u8>>::new()
            );
            Ok(())
        }
        #[test]
        fn test_empty_multipart() -> anyhow::Result<()> {
            let conf = ReaderConfig::new()
                .url("router+bind:ipc:///tmp/test")?
                .with_topic_prefix_spec(TopicPrefixSpec::SourceId("topic".into()))?
                .build()?;

            let mut reader = Reader::new(&conf)?;
            reader.socket.as_mut().unwrap().send_multipart(&[], 0)?;
            let m = reader.receive()?;
            assert!(matches!(m, ReaderResult::TooShort(_)));
            assert_eq!(
                reader.socket.as_mut().unwrap().take_buffer(),
                Vec::<Vec<u8>>::new()
            );
            Ok(())
        }

        #[test]
        fn test_too_short_routing_id() -> anyhow::Result<()> {
            let conf = ReaderConfig::new()
                .url("router+bind:ipc:///tmp/test")?
                .with_topic_prefix_spec(TopicPrefixSpec::SourceId("topic".into()))?
                .build()?;

            let mut reader = Reader::new(&conf)?;
            reader
                .socket
                .as_mut()
                .unwrap()
                .send_multipart(&[b"routing-id"], 0)?;
            let m = reader.receive()?;
            assert!(matches!(m, ReaderResult::TooShort(_)));
            assert_eq!(
                reader.socket.as_mut().unwrap().take_buffer(),
                Vec::<Vec<u8>>::new()
            );
            Ok(())
        }

        #[test]
        fn test_eos() -> anyhow::Result<()> {
            let conf = ReaderConfig::new()
                .url("router+bind:ipc:///tmp/test")?
                .with_topic_prefix_spec(TopicPrefixSpec::SourceId("topic".into()))?
                .build()?;

            let mut reader = Reader::new(&conf)?;
            reader
                .socket
                .as_mut()
                .unwrap()
                .send_multipart(&[b"routing-id", END_OF_STREAM_MESSAGE], 0)?;
            let m = reader.receive()?;
            assert!(matches!(m, ReaderResult::EndOfStream { .. }));
            assert_eq!(
                reader.socket.as_mut().unwrap().take_buffer(),
                vec![b"routing-id", CONFIRMATION_MESSAGE]
            );
            Ok(())
        }

        #[test]
        fn test_too_short_routing_and_topic() -> anyhow::Result<()> {
            let conf = ReaderConfig::new()
                .url("router+bind:ipc:///tmp/test")?
                .with_topic_prefix_spec(TopicPrefixSpec::SourceId("topic".into()))?
                .build()?;

            let mut reader = Reader::new(&conf)?;
            reader
                .socket
                .as_mut()
                .unwrap()
                .send_multipart(&[b"routing-id", b"topic"], 0)?;
            let m = reader.receive()?;
            assert!(matches!(m, ReaderResult::TooShort(_)));
            assert_eq!(
                reader.socket.as_mut().unwrap().take_buffer(),
                Vec::<Vec<u8>>::new()
            );
            Ok(())
        }

        #[test]
        fn test_prefix_mismatch_source_id() -> anyhow::Result<()> {
            let conf = ReaderConfig::new()
                .url("router+bind:ipc:///tmp/test")?
                .with_topic_prefix_spec(TopicPrefixSpec::SourceId("topic2".into()))?
                .build()?;

            let message = Message::end_of_stream(EndOfStream::new("test".to_string()));
            let binary = crate::message::save_message(&message);

            let mut reader = Reader::new(&conf)?;
            reader
                .socket
                .as_mut()
                .unwrap()
                .send_multipart(&[b"routing-id", b"topic", binary.as_slice()], 0)?;

            let m = reader.receive()?;

            assert!(matches!(
                m,
                ReaderResult::PrefixMismatch { topic, routing_id } if topic == b"topic" && routing_id == Some(b"routing-id".to_vec())
            ));

            reader
                .socket
                .as_mut()
                .unwrap()
                .send_multipart(&[b"routing-id", b"topic22", binary.as_slice()], 0)?;

            let m = reader.receive()?;

            assert!(matches!(
                m,
                ReaderResult::PrefixMismatch { topic, routing_id } if topic == b"topic22" && routing_id == Some(b"routing-id".to_vec())
            ));

            reader
                .socket
                .as_mut()
                .unwrap()
                .send_multipart(&[b"routing-id", b"topic2", binary.as_slice()], 0)?;

            let m = reader.receive()?;

            assert!(matches!(m, ReaderResult::Message { .. }));
            Ok(())
        }

        #[test]
        fn test_prefix_mismatch_prefix() -> anyhow::Result<()> {
            let conf = ReaderConfig::new()
                .url("router+bind:ipc:///tmp/test")?
                .with_topic_prefix_spec(TopicPrefixSpec::Prefix("topic2".into()))?
                .build()?;

            let message = Message::end_of_stream(EndOfStream::new("test".to_string()));
            let binary = crate::message::save_message(&message);

            let mut reader = Reader::new(&conf)?;
            reader
                .socket
                .as_mut()
                .unwrap()
                .send_multipart(&[b"routing-id", b"topic", binary.as_slice()], 0)?;

            let m = reader.receive()?;

            assert!(matches!(
                m,
                ReaderResult::PrefixMismatch { topic, routing_id } if topic == b"topic" && routing_id == Some(b"routing-id".to_vec())
            ));

            reader
                .socket
                .as_mut()
                .unwrap()
                .send_multipart(&[b"routing-id", b"topic22", binary.as_slice()], 0)?;

            let m = reader.receive()?;

            assert!(matches!(m, ReaderResult::Message { .. }));

            reader
                .socket
                .as_mut()
                .unwrap()
                .send_multipart(&[b"routing-id", b"topic2", binary.as_slice()], 0)?;

            let m = reader.receive()?;

            assert!(matches!(m, ReaderResult::Message { .. }));
            Ok(())
        }

        #[test]
        fn test_message_and_extra_parts() -> anyhow::Result<()> {
            let conf = ReaderConfig::new()
                .url("router+bind:ipc:///tmp/test")?
                .build()
                .unwrap();

            let message = Message::end_of_stream(EndOfStream::new("test".to_string()));
            let binary = crate::message::save_message(&message);

            let mut reader = Reader::new(&conf).unwrap();
            reader
                .socket
                .as_mut()
                .unwrap()
                .send_multipart(&[b"routing-id", b"topic2", binary.as_slice(), b"extra"], 0)
                .unwrap();

            let m = reader.receive().unwrap();

            assert!(matches!(
                m,
                ReaderResult::Message {
                    message,
                    topic,
                    routing_id,
                    data
                } if message.is_end_of_stream() && topic == b"topic2" && routing_id == Some(b"routing-id".to_vec()) && data == vec![b"extra"]
            ));
            Ok(())
        }
    }

    mod rep_tests {
        use crate::message::Message;
        use crate::primitives::eos::EndOfStream;
        use crate::transport::zeromq::reader::ReaderResult;
        use crate::transport::zeromq::{
            Reader, ReaderConfig, TopicPrefixSpec, CONFIRMATION_MESSAGE,
        };

        #[test]
        fn test_ok() -> anyhow::Result<()> {
            let conf = ReaderConfig::new()
                .url("rep+bind:ipc:///tmp/test")?
                .with_topic_prefix_spec(TopicPrefixSpec::SourceId("topic".into()))?
                .build()?;

            let mut reader = Reader::new(&conf)?;
            let message = Message::end_of_stream(EndOfStream::new("test".to_string()));
            let binary = crate::message::save_message(&message);

            reader
                .socket
                .as_mut()
                .unwrap()
                .send_multipart(&[b"topic", &binary, &vec![0x0, 0x1, 0x2]], 0)?;

            let m = reader.receive()?;
            assert!(matches!(
                &m,
                ReaderResult::Message {
                    message,
                    topic,
                    routing_id,
                    data
                } if message.is_end_of_stream() && topic == b"topic" && routing_id == &None && data == &vec![vec![0x0, 0x1, 0x2]]
            ));
            assert_eq!(
                reader.socket.as_mut().unwrap().take_buffer(),
                vec![CONFIRMATION_MESSAGE]
            );
            Ok(())
        }
    }
}
