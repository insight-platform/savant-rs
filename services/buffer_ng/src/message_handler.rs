use std::{borrow::Cow, sync::Arc, time::Duration};

use crate::rocksdb::PersistentQueueWithCapacity;
use anyhow::{Context, Result};
use bitcode::{Decode, Encode};
use parking_lot::Mutex;
use pyo3::{Py, PyAny, Python};
use savant_core::{
    message::{load_message, save_message, Message},
    transport::zeromq::{NonBlockingWriter, WriterResult},
};
use savant_core_py::primitives::message::Message as PyMessage;

#[derive(Debug, Clone, Decode, Encode)]
struct StoredMessage {
    topic: String,
    message_bytes: Vec<u8>,
    data: Vec<Vec<u8>>,
}

pub struct MessageWriter {
    pub python_handler: Option<Py<PyAny>>,
    pub queue: Arc<Mutex<PersistentQueueWithCapacity>>,
}

impl MessageWriter {
    pub fn new(
        queue: Arc<Mutex<PersistentQueueWithCapacity>>,
        python_handler: Option<Py<PyAny>>,
    ) -> Self {
        Self {
            python_handler,
            queue,
        }
    }

    pub fn push(&mut self, topic: &String, message: Message, data: Vec<Vec<u8>>) -> Result<()> {
        let (topic, message) = if let Some(handler) = &self.python_handler {
            let message = PyMessage::new(message);
            let topic = topic.to_string();
            Python::attach(|py| {
                let res = handler.call1(py, (&topic, message))?;
                let (topic, message) = res.extract::<(String, PyMessage)>(py)?;
                Ok::<(Cow<String>, Cow<Message>), anyhow::Error>((
                    Cow::Owned(topic),
                    Cow::Owned(message.extract()),
                ))
            })?
        } else {
            (Cow::Borrowed(topic), Cow::Borrowed(&message))
        };

        let message_bytes = save_message(&message)?;
        let stored_message = StoredMessage {
            topic: topic.to_string(),
            message_bytes,
            data,
        };

        let serialized_message = bitcode::encode(&stored_message);
        let mut q = self.queue.lock();
        let res = q.push(&[&serialized_message]);
        if let Err(e) = res {
            log::warn!(
                target: "buffer_ng::message_handler::writer",
                "Failed to push message to queue: {:?}",
                e
            );
        }
        if q.is_high_utilization() {
            log::warn!(
                target: "buffer_ng::message_handler::writer",
                "Queue is high utilization. Current length: {}, payload size: {}, High watermark: {}, Max elements: {}",
                q.len(),
                q.payload_size(),
                q.high_watermark(),
                q.max_elements()
            );
        }
        Ok(())
    }
}

pub struct MessageHandler {
    pub queue: Arc<Mutex<PersistentQueueWithCapacity>>,
    pub writer: NonBlockingWriter,
    pub idle_sleep: Duration,
    pub last_writer_result: Option<WriterResult>,
}

impl MessageHandler {
    pub fn new(
        queue: Arc<Mutex<PersistentQueueWithCapacity>>,
        writer: NonBlockingWriter,
        idle_sleep: Duration,
    ) -> Self {
        Self {
            queue,
            writer,
            idle_sleep,
            last_writer_result: None,
        }
    }

    pub fn process_stored_message(&mut self) -> Result<()> {
        let read = self
            .queue
            .lock()
            .pop(1)
            .with_context(|| "Failed to pop message from queue")?;
        if read.is_empty() {
            std::thread::sleep(self.idle_sleep);
            self.log_last_writer_result();
            return Ok(());
        }

        let stored_message: StoredMessage =
            bitcode::decode(&read[0]).with_context(|| "Failed to decode stored message")?;

        let data_refs = stored_message
            .data
            .iter()
            .map(|d| d.as_slice())
            .collect::<Vec<_>>();

        self.log_last_writer_result();

        let res = self
            .writer
            .send_message(
                &stored_message.topic,
                &load_message(&stored_message.message_bytes),
                &data_refs,
            )
            .with_context(|| "Failed to send message")?;

        self.last_writer_result = Some(res.get()?);

        Ok(())
    }

    fn log_last_writer_result(&mut self) {
        if let Some(res) = &self.last_writer_result {
            match res {
                WriterResult::SendTimeout => {
                    log::warn!("Send timeout occurred!");
                }
                WriterResult::AckTimeout(timeout) => {
                    log::warn!("Ack timeout occurred! Timeout: {timeout}");
                }
                WriterResult::Ack {
                    send_retries_spent,
                    receive_retries_spent,
                    time_spent,
                } => {
                    log::debug!("Ack occurred! Send retries spent: {send_retries_spent}, Receive retries spent: {receive_retries_spent}, Time spent: {time_spent}");
                }
                WriterResult::Success {
                    retries_spent,
                    time_spent,
                } => {
                    log::debug!(
                        "Success occurred! Retries spent: {retries_spent}, Time spent: {time_spent}"
                    );
                }
            }
        }
        self.last_writer_result = None;
    }
}
