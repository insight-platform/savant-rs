use anyhow::bail;
use log::{debug, error, info, warn};
use lru::LruCache;
use parking_lot::Mutex;
use std::num::NonZeroUsize;
use std::str::from_utf8;
use zmq::Context;

use crate::message::{Message, SeqStore};
use crate::transport::zeromq::{
    create_ipc_dirs, set_ipc_permissions, MockSocketResponder, ReaderConfig, ReaderSocketType,
    RoutingIdFilter, Socket, SocketProvider, CONFIRMATION_MESSAGE, ZMQ_LINGER,
};
use crate::utils::bytes_to_hex_string;

pub struct Reader<R: MockSocketResponder, P: SocketProvider<R>> {
    seq_store: SeqStore,
    context: Mutex<Option<Context>>,
    config: ReaderConfig,
    socket: Mutex<Option<Socket<R>>>,
    routing_id_filter: Mutex<RoutingIdFilter>,
    source_blacklist_cache: Mutex<LruCache<Vec<u8>, u64>>,
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
    MessageVersionMismatch {
        topic: Vec<u8>,
        routing_id: Option<Vec<u8>>,
        sender_version: String,
        expected_version: String,
    },
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

    pub fn message_version_mismatch(
        topic: &[u8],
        routing_id: &Option<&Vec<u8>>,
        sender_version: &str,
        expected_version: &str,
    ) -> Self {
        Self::MessageVersionMismatch {
            topic: topic.to_vec(),
            routing_id: routing_id.cloned(),
            sender_version: sender_version.to_string(),
            expected_version: expected_version.to_string(),
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

        info!(
            target: "savant_rs::zeromq::reader",
            "ZMQ Reader bind={}/type={:?} is started on endpoint {}",
            config.bind(),config.socket_type(), config.endpoint()
        );

        Ok(Self {
            seq_store: SeqStore::new(),
            context: Mutex::new(Some(context)),
            config: config.clone(),
            socket: Mutex::new(Some(socket)),
            routing_id_filter: Mutex::new(RoutingIdFilter::new(*config.routing_cache_size())?),
            source_blacklist_cache: Mutex::new(LruCache::new(
                NonZeroUsize::new(*config.source_blacklist_size() as usize).ok_or(
                    anyhow::anyhow!("Source blacklist cache size must be greater than 0"),
                )?,
            )),
            phony: std::marker::PhantomData,
        })
    }

    pub fn destroy(&self) -> anyhow::Result<()> {
        info!(
            target: "savant_rs::zeromq::reader::destroy",
            "Destroying ZeroMQ socket for endpoint {}",
            self.config.endpoint()
        );
        self.socket.lock().take();
        self.context.lock().take();
        info!(
            target: "savant_rs::zeromq::reader::destroy",
            "ZeroMQ socket for endpoint {} destroyed",
            self.config.endpoint()
        );
        Ok(())
    }

    pub fn is_alive(&self) -> bool {
        self.socket.lock().is_some()
    }

    pub fn blacklist_source(&self, source: &[u8]) {
        info!(
            target: "savant_rs::zeromq::reader::blacklist_source",
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
            .lock()
            .put(source.to_vec(), black_list_until);
    }

    pub fn is_blacklisted(&self, source: &[u8]) -> bool {
        debug!(
            target: "savant_rs::zeromq::reader::is_blacklisted",
            "Checking if source '{}' is blacklisted for endpoint '{}'",
            from_utf8(source).unwrap_or(&bytes_to_hex_string(source)),
            self.config.endpoint()
        );
        let mut source_blacklist_cache_bind = self.source_blacklist_cache.lock();
        let val = source_blacklist_cache_bind.get(&source.to_vec());
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
                source_blacklist_cache_bind.pop(&source.to_vec());
            }
        }
        false
    }

    pub fn receive(&mut self) -> anyhow::Result<ReaderResult> {
        if self.socket.lock().is_none() {
            bail!(
                "ZeroMQ socket for endpoint {} is no longer available, because it was destroyed.",
                self.config.endpoint()
            );
        }
        debug!(
            target: "savant_rs::zeromq::reader::receive::before_recv",
            "Waiting for message from ZeroMQ socket for endpoint {}",
            self.config.endpoint());

        let parts = {
            let mut bind = self.socket.lock();
            let socket = bind.as_mut().unwrap();
            socket.recv_multipart(0)
        };

        debug!(
            target: "savant_rs::zeromq::reader::receive::after_recv",
            "Received message from ZeroMQ socket for endpoint {}",
            self.config.endpoint());

        if let Err(e) = parts {
            if let zmq::Error::EAGAIN = e {
                debug!(
                    target: "savant_rs::zeromq::reader::receive::eagain",
                    "Failed to receive message from ZeroMQ socket due to timeout (EAGAIN)"
                );
                return Ok(ReaderResult::Timeout);
            } else {
                error!(
                    target: "savant_rs::zeromq::reader::receive::error",
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
                target: "savant_rs::zeromq::reader::receive::too_short",
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
                target: "savant_rs::zeromq::reader::receive::blacklisted",
                "Received message from blacklisted source {:?} from ZeroMQ socket for endpoint {}",
                from_utf8(topic).unwrap_or(&bytes_to_hex_string(topic)),
                self.config.endpoint()
            );
            let mut bind = self.socket.lock();
            let socket = bind.as_mut().unwrap();
            if self.config.socket_type() == &ReaderSocketType::Rep {
                socket.send(CONFIRMATION_MESSAGE, 0)?;
            }

            return Ok(ReaderResult::Blacklisted(topic.clone()));
        }

        let message = Box::new(crate::protobuf::deserialize(command)?);

        if message.meta.protocol_version != savant_protobuf::version() {
            warn!(
                target: "savant_rs::zeromq::reader::receive::version_mismatch",
                "Message protocol version mismatch: message version={:?}, program expects version={:?}", message.meta.protocol_version, savant_protobuf::version()
            );
            // blacklist the sender
            self.blacklist_source(topic);
            return Ok(ReaderResult::MessageVersionMismatch {
                topic: topic.clone(),
                routing_id: routing_id.cloned(),
                sender_version: message.meta.protocol_version.clone(),
                expected_version: savant_protobuf::version().to_string(),
            });
        }

        self.seq_store.validate_seq_id(&message);

        if message.is_end_of_stream() {
            if self.config.socket_type() != &ReaderSocketType::Sub {
                debug!(
                    target: "savant_rs::zeromq::reader::receive::eos",
                    "Received end of stream message from ZeroMQ socket for endpoint {}",
                    self.config.endpoint()
                );
                let mut bind = self.socket.lock();
                let socket = bind.as_mut().unwrap();
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
                target: "savant_rs::zeromq::reader::receive::prefix_mismatch",
                "Received message with invalid topic from ZeroMQ socket for endpoint {}. Expected topic to match spec {:?}, but got {}",
                self.config.endpoint(),
                self.config.topic_prefix_spec(),
                from_utf8(topic).unwrap_or(&bytes_to_hex_string(topic))
            );
            let mut bind = self.socket.lock();
            let socket = bind.as_mut().unwrap();
            if self.config.socket_type() == &ReaderSocketType::Rep {
                socket.send(CONFIRMATION_MESSAGE, 0)?;
            }

            return Ok(ReaderResult::PrefixMismatch {
                topic: topic.clone(),
                routing_id: routing_id.cloned(),
            });
        }

        if self.config.socket_type() == &ReaderSocketType::Rep {
            let mut bind = self.socket.lock();
            let socket = bind.as_mut().unwrap();
            socket.send(CONFIRMATION_MESSAGE, 0)?;
        }

        if self.routing_id_filter.lock().allow(topic, &routing_id) {
            Ok(ReaderResult::Message {
                message,
                topic: topic.clone(),
                routing_id: routing_id.cloned(),
                data: extra.iter().map(|e| e.to_vec()).collect(),
            })
        } else {
            debug!(
                target: "savant_rs::zeromq::reader::receive::routing_id_mismatch",
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
                .lock()
                .as_mut()
                .unwrap()
                .send_multipart(&[b"routing-id", b"topic", &binary, &[0x0, 0x1, 0x2]], 0)?;

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
                reader.socket.lock().as_mut().unwrap().take_buffer(),
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
            reader
                .socket
                .lock()
                .as_mut()
                .unwrap()
                .send_multipart(&[], 0)?;
            let m = reader.receive()?;
            assert!(matches!(m, ReaderResult::TooShort(_)));
            assert_eq!(
                reader.socket.lock().as_mut().unwrap().take_buffer(),
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
                .lock()
                .as_mut()
                .unwrap()
                .send_multipart(&[b"routing-id"], 0)?;
            let m = reader.receive()?;
            assert!(matches!(m, ReaderResult::TooShort(_)));
            assert_eq!(
                reader.socket.lock().as_mut().unwrap().take_buffer(),
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
            reader.socket.lock().as_mut().unwrap().send_multipart(
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
                reader.socket.lock().as_mut().unwrap().take_buffer(),
                vec![b"routing-id", CONFIRMATION_MESSAGE]
            );
            Ok(())
        }

        #[test]
        fn test_version_mismatch() -> anyhow::Result<()> {
            let conf = ReaderConfig::new()
                .url("router+bind:ipc:///tmp/test")?
                .with_topic_prefix_spec(TopicPrefixSpec::SourceId("topic".into()))?
                .build()?;

            let mut reader = Reader::<NoopResponder, MockSocketProvider>::new(&conf)?;
            let mut message = Message::end_of_stream(EndOfStream::new("topic".into()));
            message.set_protocol_version("0.0.0".to_string());
            reader
                .socket
                .lock()
                .as_mut()
                .unwrap()
                .send_multipart(&[b"routing-id", b"topic", &serialize(&message)?], 0)?;
            let m = reader.receive()?;
            assert!(
                matches!(m, ReaderResult::MessageVersionMismatch {topic,routing_id,sender_version,expected_version }
                    if topic == b"topic" && routing_id == Some(b"routing-id".to_vec()) && sender_version == "0.0.0" && expected_version == savant_protobuf::version())
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
                .lock()
                .as_mut()
                .unwrap()
                .send_multipart(&[b"routing-id", b"topic"], 0)?;
            let m = reader.receive()?;
            assert!(matches!(m, ReaderResult::TooShort(_)));
            assert_eq!(
                reader.socket.lock().as_mut().unwrap().take_buffer(),
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
                .lock()
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
                .lock()
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
                .lock()
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
                .lock()
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
                .lock()
                .as_mut()
                .unwrap()
                .send_multipart(&[b"routing-id", b"topic22", binary.as_slice()], 0)?;

            let m = reader.receive()?;

            assert!(matches!(m, ReaderResult::Message { .. }));

            reader
                .socket
                .lock()
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
                .lock()
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
                .lock()
                .as_mut()
                .unwrap()
                .send_multipart(&[b"topic", &binary, &[0x0, 0x1, 0x2]], 0)?;

            let m = reader.receive()?;
            assert!(matches!(
                &m,
                ReaderResult::Message {
                    message,
                    topic,
                    routing_id,
                    data
                } if message.is_user_data() && topic == b"topic" && routing_id.is_none() && data == &vec![vec![0x0, 0x1, 0x2]]
            ));
            assert_eq!(
                reader.socket.lock().as_mut().unwrap().take_buffer(),
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
                .lock()
                .as_mut()
                .unwrap()
                .send_multipart(&[b"topic", &binary, &[0x0, 0x1, 0x2]], 0)?;
            reader.blacklist_source(b"topic");
            assert!(reader.is_blacklisted(b"topic"));

            let m = reader.receive()?;
            assert!(matches!(
                &m,
                ReaderResult::Blacklisted(topic) if topic == b"topic"
            ));
            assert_eq!(
                reader.socket.lock().as_mut().unwrap().take_buffer(),
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
                .lock()
                .as_mut()
                .unwrap()
                .send_multipart(&[b"topic", &binary, &[0x0, 0x1, 0x2]], 0)?;
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
                } if message.is_user_data() && topic == b"topic" && routing_id.is_none() && data == &vec![vec![0x0, 0x1, 0x2]]
            ));
            assert_eq!(
                reader.socket.lock().as_mut().unwrap().take_buffer(),
                vec![CONFIRMATION_MESSAGE]
            );
            Ok(())
        }
    }
}
