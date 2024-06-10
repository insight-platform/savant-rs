use anyhow::bail;
use log::{debug, error, info, warn};
use lru::LruCache;
use std::num::NonZeroUsize;
use std::str::from_utf8;
use zmq::Context;

use crate::message::Message;
use crate::transport::zeromq::{
    create_ipc_dirs, set_ipc_permissions, MockSocketResponder, ReaderConfig, ReaderSocketType,
    RoutingIdFilter, Socket, SocketProvider, CONFIRMATION_MESSAGE, ZMQ_LINGER,
};
use crate::utils::bytes_to_hex_string;

pub struct Reader<R: MockSocketResponder, P: SocketProvider<R>> {
    context: Option<Context>,
    config: ReaderConfig,
    socket: Option<Socket<R>>,
    routing_id_filter: RoutingIdFilter,
    source_blacklist_cache: LruCache<Vec<u8>, u64>,
    phony: std::marker::PhantomData<P>,
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
    Blacklisted(Vec<u8>),
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

impl<R: MockSocketResponder, P: SocketProvider<R> + Default> Reader<R, P> {
    pub fn new(config: &ReaderConfig) -> anyhow::Result<Self> {
        let context = Context::new();
        let socket_provider = P::default();
        let socket = socket_provider.new_socket(&context, config.socket_type().into())?;

        socket.set_rcvhwm(*config.receive_hwm())?;
        socket.set_rcvtimeo(*config.receive_timeout())?;
        socket.set_linger(ZMQ_LINGER)?;

        if config.socket_type() == &ReaderSocketType::Sub {
            socket.set_subscribe(config.topic_prefix_spec().get().as_bytes())?;
        }

        if *config.bind() {
            if matches!(&socket, Socket::ZmqSocket(_)) && config.endpoint().starts_with("ipc://") {
                create_ipc_dirs(config.endpoint())?;
            }

            socket.bind(config.endpoint())?;

            if matches!(&socket, Socket::ZmqSocket(_)) && config.endpoint().starts_with("ipc://") {
                if let Some(permissions) = config.fix_ipc_permissions() {
                    set_ipc_permissions(config.endpoint(), *permissions)?;
                }
            }
        } else {
            socket.connect(config.endpoint())?;
        }

        Ok(Self {
            context: Some(context),
            config: config.clone(),
            socket: Some(socket),
            routing_id_filter: RoutingIdFilter::new(*config.routing_cache_size())?,
            source_blacklist_cache: LruCache::new(
                NonZeroUsize::new(*config.source_blacklist_size() as usize).unwrap(),
            ),
            phony: std::marker::PhantomData,
        })
    }

    pub fn destroy(&mut self) -> anyhow::Result<()> {
        info!(
            target: "savant_rs::zeromq::reader",
            "Destroying ZeroMQ socket for endpoint {}",
            self.config.endpoint()
        );
        self.socket.take();
        self.context.take();
        info!(
            target: "savant_rs::zeromq::reader",
            "ZeroMQ socket for endpoint {} destroyed",
            self.config.endpoint()
        );
        Ok(())
    }

    pub fn is_alive(&self) -> bool {
        self.socket.is_some()
    }

    pub fn blacklist_source(&mut self, source: &[u8]) {
        info!(
            target: "savant_rs::zeromq::reader",
            "Blacklisting source '{}' for endpoint '{}'",
            from_utf8(source).unwrap_or(&bytes_to_hex_string(source)),
            self.config.endpoint()
        );
        let black_list_until = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + *self.config.source_blacklist_ttl();
        self.source_blacklist_cache
            .put(source.to_vec(), black_list_until);
    }

    pub fn is_blacklisted(&mut self, source: &[u8]) -> bool {
        debug!(
            target: "savant_rs::zeromq::reader",
            "Checking if source '{}' is blacklisted for endpoint '{}'",
            from_utf8(source).unwrap_or(&bytes_to_hex_string(source)),
            self.config.endpoint()
        );
        let val = self.source_blacklist_cache.get(&source.to_vec());
        if let Some(val) = val {
            if val
                > &std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            {
                debug!(
                    target: "savant_rs::zeromq::reader::is_blacklisted",
                    "Source '{}' is blacklisted for endpoint '{}'",
                    from_utf8(source).unwrap_or(&bytes_to_hex_string(source)),
                    self.config.endpoint()
                );
                return true;
            } else {
                self.source_blacklist_cache.pop(&source.to_vec());
            }
        }
        false
    }

    pub fn receive(&mut self) -> anyhow::Result<ReaderResult> {
        if self.socket.is_none() {
            bail!(
                "ZeroMQ socket for endpoint {} is no longer available, because it was destroyed.",
                self.config.endpoint()
            );
        }
        debug!(
            target: "savant_rs::zeromq::reader",
            "Waiting for message from ZeroMQ socket for endpoint {}",
            self.config.endpoint());

        let parts = {
            let socket = self.socket.as_mut().unwrap();
            socket.recv_multipart(0)
        };

        debug!(
            target: "savant_rs::zeromq::reader",
            "Received message from ZeroMQ socket for endpoint {}",
            self.config.endpoint());
        if let Err(e) = parts {
            if let zmq::Error::EAGAIN = e {
                debug!(
                    target: "savant_rs::zeromq::reader",
                    "Failed to receive message from ZeroMQ socket due to timeout (EAGAIN)"
                );
                return Ok(ReaderResult::Timeout);
            } else {
                error!(
                    target: "savant_rs::zeromq::reader",
                    "Failed to receive message from ZeroMQ socket. Error is [{}] {:?}",
                    e.to_raw(), e
                );
                bail!(
                    "Failed to receive message from ZeroMQ socket. Error is [{}] {:?}",
                    e.to_raw(),
                    e
                );
            }
        }

        let parts = parts.unwrap();

        let min_required_parts = match self.config.socket_type() {
            ReaderSocketType::Sub => 2,
            ReaderSocketType::Router => 3,
            ReaderSocketType::Rep => 2,
        };

        if parts.len() < min_required_parts {
            warn!(
                target: "savant_rs::zeromq::reader",
                "Received message with invalid number of parts from ZeroMQ socket for endpoint {}. Expected at least {} parts, but got {}",
                self.config.endpoint(),
                min_required_parts,
                parts.len()
            );
            return Ok(ReaderResult::TooShort(parts));
        };

        let (routing_id, topic, command, extra) =
            if self.config.socket_type() == &ReaderSocketType::Router {
                let routing_id = &parts[0];
                let topic = &parts[1];
                let command = &parts[2];
                let message = &parts[3..];
                (Some(routing_id), topic, command, message)
            } else {
                (None, &parts[0], &parts[1], &parts[2..])
            };
        if self.is_blacklisted(topic) {
            debug!(
                target: "savant_rs::zeromq::reader",
                "Received message from blacklisted source {:?} from ZeroMQ socket for endpoint {}",
                from_utf8(topic).unwrap_or(&bytes_to_hex_string(topic)),
                self.config.endpoint()
            );
            let socket = self.socket.as_mut().unwrap();
            if self.config.socket_type() == &ReaderSocketType::Rep {
                socket.send(CONFIRMATION_MESSAGE, 0)?;
            }

            return Ok(ReaderResult::Blacklisted(topic.clone()));
        }

        let message = Box::new(crate::protobuf::deserialize(command)?);

        if message.is_end_of_stream() {
            if self.config.socket_type() != &ReaderSocketType::Sub {
                debug!(
                    target: "savant_rs::zeromq::reader",
                    "Received end of stream message from ZeroMQ socket for endpoint {}",
                    self.config.endpoint()
                );
                let socket = self.socket.as_mut().unwrap();
                if let Some(routing_id) = routing_id {
                    socket.send_multipart(&[routing_id, CONFIRMATION_MESSAGE], 0)?;
                } else {
                    socket.send(CONFIRMATION_MESSAGE, 0)?;
                }
            }

            return Ok(ReaderResult::Message {
                message,
                topic: topic.clone(),
                routing_id: routing_id.cloned(),
                data: vec![],
            });
        }

        if !self.config.topic_prefix_spec().matches(topic) {
            debug!(
                target: "savant_rs::zeromq::reader",
                "Received message with invalid topic from ZeroMQ socket for endpoint {}. Expected topic to match spec {:?}, but got {}",
                self.config.endpoint(),
                self.config.topic_prefix_spec(),
                from_utf8(topic).unwrap_or(&bytes_to_hex_string(topic))
            );
            let socket = self.socket.as_mut().unwrap();
            if self.config.socket_type() == &ReaderSocketType::Rep {
                socket.send(CONFIRMATION_MESSAGE, 0)?;
            }

            return Ok(ReaderResult::PrefixMismatch {
                topic: topic.clone(),
                routing_id: routing_id.cloned(),
            });
        }

        if self.config.socket_type() == &ReaderSocketType::Rep {
            let socket = self.socket.as_mut().unwrap();
            socket.send(CONFIRMATION_MESSAGE, 0)?;
        }

        if self.routing_id_filter.allow(topic, &routing_id) {
            Ok(ReaderResult::Message {
                message,
                topic: topic.clone(),
                routing_id: routing_id.cloned(),
                data: extra.iter().map(|e| e.to_vec()).collect(),
            })
        } else {
            debug!(
                target: "savant_rs::zeromq::reader",
                "Received message with invalid routing ID from ZeroMQ socket for endpoint {}. Got topic = {}, routing_id = {}",
                self.config.endpoint(),
                from_utf8(topic).unwrap_or(&bytes_to_hex_string(topic)),
                routing_id.map(|r| bytes_to_hex_string(r)).unwrap_or(String::new())
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
        use crate::primitives::userdata::UserData;
        use crate::protobuf::serialize;
        use crate::transport::zeromq::reader::ReaderResult;
        use crate::transport::zeromq::{
            MockSocketProvider, NoopResponder, Reader, ReaderConfig, TopicPrefixSpec,
            CONFIRMATION_MESSAGE,
        };

        #[test]
        fn test_ok() -> anyhow::Result<()> {
            let conf = ReaderConfig::new()
                .url("router+bind:ipc:///tmp/test")?
                .with_topic_prefix_spec(TopicPrefixSpec::SourceId("topic".into()))?
                .build()?;

            let mut reader = Reader::<NoopResponder, MockSocketProvider>::new(&conf)?;
            let message = Message::user_data(UserData::new("test"));
            let binary = crate::message::save_message(&message)?;

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
                } if message.is_user_data() && topic == b"topic" && routing_id == &Some(b"routing-id".to_vec()) && data == &vec![vec![0x0, 0x1, 0x2]]
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

            let mut reader = Reader::<NoopResponder, MockSocketProvider>::new(&conf)?;
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

            let mut reader = Reader::<NoopResponder, MockSocketProvider>::new(&conf)?;
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

            let mut reader = Reader::<NoopResponder, MockSocketProvider>::new(&conf)?;
            reader.socket.as_mut().unwrap().send_multipart(
                &[
                    b"routing-id",
                    b"topic",
                    &serialize(&Message::end_of_stream(EndOfStream::new("topic".into())))?,
                ],
                0,
            )?;
            let m = reader.receive()?;
            assert!(
                matches!(m, ReaderResult::Message {message,topic,routing_id,data }
                    if message.is_end_of_stream() && topic == b"topic" && routing_id == Some(b"routing-id".to_vec()) && data.is_empty())
            );
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

            let mut reader = Reader::<NoopResponder, MockSocketProvider>::new(&conf)?;
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

            let message = Message::user_data(UserData::new("test"));
            let binary = crate::message::save_message(&message)?;

            let mut reader = Reader::<NoopResponder, MockSocketProvider>::new(&conf)?;
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

            let message = Message::user_data(UserData::new("test"));
            let binary = crate::message::save_message(&message)?;

            let mut reader = Reader::<NoopResponder, MockSocketProvider>::new(&conf)?;
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

            let message = Message::user_data(UserData::new("test"));
            let binary = crate::message::save_message(&message)?;

            let mut reader = Reader::<NoopResponder, MockSocketProvider>::new(&conf).unwrap();
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
                } if message.is_user_data() && topic == b"topic2" && routing_id == Some(b"routing-id".to_vec()) && data == vec![b"extra"]
            ));
            Ok(())
        }
    }

    mod rep_tests {
        use crate::message::Message;
        use crate::primitives::userdata::UserData;
        use crate::transport::zeromq::reader::ReaderResult;
        use crate::transport::zeromq::{
            MockSocketProvider, NoopResponder, Reader, ReaderConfig, TopicPrefixSpec,
            CONFIRMATION_MESSAGE,
        };

        #[test]
        fn test_ok() -> anyhow::Result<()> {
            let conf = ReaderConfig::new()
                .url("rep+bind:ipc:///tmp/test")?
                .with_topic_prefix_spec(TopicPrefixSpec::SourceId("topic".into()))?
                .build()?;

            let mut reader = Reader::<NoopResponder, MockSocketProvider>::new(&conf)?;
            let message = Message::user_data(UserData::new("topic"));
            let binary = crate::message::save_message(&message)?;

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
                } if message.is_user_data() && topic == b"topic" && routing_id == &None && data == &vec![vec![0x0, 0x1, 0x2]]
            ));
            assert_eq!(
                reader.socket.as_mut().unwrap().take_buffer(),
                vec![CONFIRMATION_MESSAGE]
            );
            Ok(())
        }
    }

    mod blacklist_tests {
        use crate::message::Message;
        use crate::primitives::userdata::UserData;
        use crate::transport::zeromq::reader::ReaderResult;
        use crate::transport::zeromq::{
            MockSocketProvider, NoopResponder, Reader, ReaderConfig, TopicPrefixSpec,
            CONFIRMATION_MESSAGE,
        };
        use std::num::NonZeroU64;

        #[test]
        fn test_blocks() -> anyhow::Result<()> {
            let conf = ReaderConfig::new()
                .url("rep+bind:ipc:///tmp/test")?
                .with_topic_prefix_spec(TopicPrefixSpec::SourceId("topic".into()))?
                .with_source_blacklist_ttl(NonZeroU64::new(1).unwrap())?
                .build()?;

            let mut reader = Reader::<NoopResponder, MockSocketProvider>::new(&conf)?;
            let message = Message::user_data(UserData::new("topic"));
            let binary = crate::message::save_message(&message)?;

            reader
                .socket
                .as_mut()
                .unwrap()
                .send_multipart(&[b"topic", &binary, &vec![0x0, 0x1, 0x2]], 0)?;
            reader.blacklist_source(b"topic");
            assert!(reader.is_blacklisted(b"topic"));

            let m = reader.receive()?;
            assert!(matches!(
                &m,
                ReaderResult::Blacklisted(topic) if topic == b"topic"
            ));
            assert_eq!(
                reader.socket.as_mut().unwrap().take_buffer(),
                vec![CONFIRMATION_MESSAGE]
            );
            Ok(())
        }

        #[test]
        fn test_passes() -> anyhow::Result<()> {
            let conf = ReaderConfig::new()
                .url("rep+bind:ipc:///tmp/test")?
                .with_topic_prefix_spec(TopicPrefixSpec::SourceId("topic".into()))?
                .with_source_blacklist_ttl(NonZeroU64::new(1).unwrap())?
                .build()?;

            let mut reader = Reader::<NoopResponder, MockSocketProvider>::new(&conf)?;
            let message = Message::user_data(UserData::new("topic"));
            let binary = crate::message::save_message(&message)?;

            reader
                .socket
                .as_mut()
                .unwrap()
                .send_multipart(&[b"topic", &binary, &vec![0x0, 0x1, 0x2]], 0)?;
            reader.blacklist_source(b"topic");
            assert!(reader.is_blacklisted(b"topic"));

            std::thread::sleep(std::time::Duration::from_millis(1100));
            assert!(!reader.is_blacklisted(b"topic"));
            let m = reader.receive()?;
            assert!(matches!(
                &m,
                ReaderResult::Message {
                    message,
                    topic,
                    routing_id,
                    data
                } if message.is_user_data() && topic == b"topic" && routing_id == &None && data == &vec![vec![0x0, 0x1, 0x2]]
            ));
            assert_eq!(
                reader.socket.as_mut().unwrap().take_buffer(),
                vec![CONFIRMATION_MESSAGE]
            );
            Ok(())
        }
    }
}
