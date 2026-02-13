use crate::configuration::{EosPolicy, ServiceConfiguration};
use log::debug;
use pyo3::{types::PyBytes, Python};
use savant_core::{
    message::Message,
    transport::zeromq::{NonBlockingReader, ReaderResult},
};
use savant_core_py::primitives::message::Message as PyMessage;
use savant_core_py::REGISTERED_HANDLERS;
use savant_services_common::topic_to_string;

struct IngressStream {
    name: String,
    socket: NonBlockingReader,
    eos_policy: Option<EosPolicy>,
}

pub struct IngressMessage {
    pub ingress_name: String,
    pub topic: String,
    pub message: Box<Message>,
    pub data: Vec<Vec<u8>>,
}

impl IngressStream {
    pub fn new(name: String, socket: NonBlockingReader, eos_policy: Option<EosPolicy>) -> Self {
        Self {
            name,
            socket,
            eos_policy,
        }
    }
}

pub struct Ingress {
    streams: Vec<IngressStream>,
    on_unsupported_message: Option<String>,
}

impl Ingress {
    pub fn new(config: &ServiceConfiguration) -> anyhow::Result<Self> {
        let mut streams = Vec::new();
        for ingress in &config.ingress {
            let socket = NonBlockingReader::try_from(&ingress.socket)?;
            let stream =
                IngressStream::new(ingress.name.clone(), socket, ingress.eos_policy.clone());
            streams.push(stream);
        }
        Ok(Self {
            streams,
            on_unsupported_message: config.common.callbacks.on_unsupported_message.clone(),
        })
    }

    pub fn get(&mut self) -> anyhow::Result<Vec<IngressMessage>> {
        let mut messages = Vec::new();
        for stream in &mut self.streams {
            let message = stream.socket.try_receive();
            if message.is_none() {
                continue;
            }
            let message_rr = message.unwrap()?;
            match message_rr {
                ReaderResult::Message {
                    message,
                    topic,
                    routing_id: _,
                    data,
                } => {
                    let topic = topic_to_string(&topic);
                    let ingress_stream_name = &stream.name;
                    let eos_policy_opt = &stream.eos_policy;
                    if message.is_video_frame()
                        || (message.is_end_of_stream()
                            && matches!(eos_policy_opt, Some(EosPolicy::Allow)))
                    {
                        let message = IngressMessage {
                            ingress_name: ingress_stream_name.clone(),
                            topic,
                            message,
                            data,
                        };
                        messages.push(message);
                    } else if let Some(on_unsupported_message) = &self.on_unsupported_message {
                        let message = PyMessage::new(*message);
                        Python::attach(|py| {
                            let handlers_bind = REGISTERED_HANDLERS.read();
                            let handler = handlers_bind
                                .get(on_unsupported_message.as_str())
                                .unwrap_or_else(|| {
                                    panic!("Python handler '{}' not found", on_unsupported_message)
                                });
                            let mut pydata = Vec::new();
                            for d in data {
                                pydata.push(
                                    PyBytes::new_with(py, d.len(), |b: &mut [u8]| {
                                        b.copy_from_slice(&d);
                                        Ok(())
                                    })
                                    .expect("Failed to create PyBytes"),
                                );
                            }
                            handler.call1(py, (ingress_stream_name, &topic, message, &pydata))
                        })?;
                    }
                }
                ReaderResult::Timeout => {
                    debug!(
                        target: "meta_merge::ingress::get",
                        "Timeout receiving message, waiting for next message."
                    );
                }
                ReaderResult::PrefixMismatch { topic, routing_id } => {
                    log::warn!(
                        target: "meta_merge::ingress::get",
                        "Received message with mismatched prefix: topic: {:?}, routing_id: {:?}",
                        topic_to_string(&topic),
                        topic_to_string(routing_id.as_ref().unwrap_or(&Vec::new()))
                    );
                }
                ReaderResult::RoutingIdMismatch { topic, routing_id } => {
                    log::warn!(
                        target: "meta_merge::ingress::get",
                        "Received message with mismatched routing_id: topic: {:?}, routing_id: {:?}",
                        topic_to_string(&topic),
                        topic_to_string(routing_id.as_ref().unwrap_or(&Vec::new()))
                    );
                }
                ReaderResult::TooShort(m) => {
                    log::warn!(
                        target: "meta_merge::ingress::get",
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
                        target: "meta_merge::ingress::get",
                        "Received message with mismatched version: topic: {:?}, routing_id: {:?}, sender_version: {:?}, expected_version: {:?}",
                        topic_to_string(&topic),
                        topic_to_string(routing_id.as_ref().unwrap_or(&Vec::new())),
                        sender_version,
                        expected_version
                    );
                }
                ReaderResult::Blacklisted(items) => {
                    log::warn!(
                        target: "meta_merge::ingress::get",
                        "Received blacklisted message: {:?}",
                        items
                    );
                }
            }
        }
        Ok(messages)
    }
}
