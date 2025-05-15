use std::num::NonZero;

use crate::configuration::ServiceConfiguration;
use log::debug;
use lru::LruCache;
use savant_core::{
    message::Message,
    transport::zeromq::{NonBlockingReader, ReaderResult},
};
use savant_services_common::topic_to_string;

struct IngressStream {
    name: String,
    socket: NonBlockingReader,
    handler: Option<String>,
}

pub struct IngressMessage {
    pub topic: String,
    pub message: Message,
    pub payload: Vec<Vec<u8>>,
}

impl IngressStream {
    pub fn new(name: String, socket: NonBlockingReader, handler: Option<String>) -> Self {
        Self {
            name,
            socket,
            handler,
        }
    }
}

pub struct Ingress {
    streams: Vec<IngressStream>,
    affinity_cache: LruCache<String, usize>,
}

impl Ingress {
    pub fn new(config: &ServiceConfiguration) -> anyhow::Result<Self> {
        let mut streams = Vec::new();
        for ingress in &config.ingress {
            let socket = NonBlockingReader::try_from(&ingress.socket)?;
            let stream = IngressStream::new(ingress.name.clone(), socket, ingress.handler.clone());
            streams.push(stream);
        }
        let affinity_cache_size = config.common.source_affinity_cache_size.unwrap();

        let affinity_cache = LruCache::new(NonZero::new(affinity_cache_size).unwrap());
        Ok(Self {
            streams,
            affinity_cache,
        })
    }

    pub fn get(&self) -> anyhow::Result<Vec<IngressMessage>> {
        let mut messages = Vec::new();
        for stream in &self.streams {
            let message = stream.socket.try_receive();
            if message.is_none() {
                continue;
            }
            let message_rr = message.unwrap()?;
            match message_rr {
                ReaderResult::Message {
                    message,
                    topic,
                    routing_id,
                    data,
                } => todo!(),
                ReaderResult::Timeout => {
                    debug!(
                        target: "router::ingress::get",
                        "Timeout receiving message, waiting for next message."
                    );
                }
                ReaderResult::PrefixMismatch { topic, routing_id } => {
                    log::warn!(
                        target: "router::ingress::get",
                        "Received message with mismatched prefix: topic: {:?}, routing_id: {:?}",
                        topic_to_string(&topic),
                        topic_to_string(routing_id.as_ref().unwrap_or(&Vec::new()))
                    );
                }
                ReaderResult::RoutingIdMismatch { topic, routing_id } => {
                    log::warn!(
                        target: "router::ingress::get",
                        "Received message with mismatched routing_id: topic: {:?}, routing_id: {:?}",
                        topic_to_string(&topic),
                        topic_to_string(routing_id.as_ref().unwrap_or(&Vec::new()))
                    );
                }
                ReaderResult::TooShort(m) => {
                    log::warn!(
                        target: "router::ingress::get",
                        "Received message that was too short: {:?}",
                        m
                    );
                }
                ReaderResult::MessageVersionMismatch {
                    topic,
                    routing_id,
                    sender_version,
                    expected_version,
                } => {
                    log::warn!(
                        target: "router::ingress::get",
                        "Received message with mismatched version: topic: {:?}, routing_id: {:?}, sender_version: {:?}, expected_version: {:?}",
                        topic_to_string(&topic),
                        topic_to_string(routing_id.as_ref().unwrap_or(&Vec::new())),
                        sender_version,
                        expected_version
                    );
                }
                ReaderResult::Blacklisted(items) => {
                    log::warn!(
                        target: "router::ingress::get",
                        "Received blacklisted message: {:?}",
                        items
                    );
                }
            }
        }
        Ok(messages)
    }

    fn get_affinity(&mut self, source_id: &str) -> Option<usize> {
        let affinity = self.affinity_cache.get(source_id);
        affinity.cloned()
    }
}
