use crate::configuration::ServiceConfiguration;
use log::debug;
use pyo3::Python;
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
    handler: Option<String>,
}

pub struct IngressMessage {
    pub topic: String,
    pub message: Box<Message>,
    pub data: Vec<Vec<u8>>,
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
}

impl Ingress {
    pub fn new(config: &ServiceConfiguration) -> anyhow::Result<Self> {
        let mut streams = Vec::new();
        for ingress in &config.ingress {
            let socket = NonBlockingReader::try_from(&ingress.socket)?;
            let stream = IngressStream::new(ingress.name.clone(), socket, ingress.handler.clone());
            streams.push(stream);
        }
        Ok(Self { streams })
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
                    routing_id: _,
                    data,
                } => {
                    let topic = topic_to_string(&topic);
                    let ingress_stream_name = &stream.name;
                    let handler_name_opt = &stream.handler;
                    let message = if let Some(handler) = handler_name_opt {
                        let message = PyMessage::new(*message);
                        Python::with_gil(|py| {
                            let handlers_bind = REGISTERED_HANDLERS.read();
                            let handler = handlers_bind
                                .get(handler.as_str())
                                .unwrap_or_else(|| panic!("Handler {} not found", handler));
                            let res = handler.call1(py, (ingress_stream_name, &topic, message))?;
                            // is none, drop message
                            if res.is_none(py) {
                                Ok::<Option<Box<Message>>, anyhow::Error>(None)
                            } else {
                                let new_message = res.extract::<PyMessage>(py)?.extract();
                                Ok::<Option<Box<Message>>, anyhow::Error>(Some(Box::new(
                                    new_message,
                                )))
                            }
                        })?
                    } else {
                        Some(Box::new(*message))
                    };
                    if let Some(message) = message {
                        let message = IngressMessage {
                            topic,
                            message,
                            data,
                        };
                        messages.push(message);
                    }
                }
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
}
