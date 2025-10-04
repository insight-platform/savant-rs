use std::{sync::Arc, time::Duration};

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

type MangledMessage = Option<(String, Message)>;

fn handle_with_python(
    handler: &Py<PyAny>,
    topic: String,
    message: Message,
) -> Result<MangledMessage> {
    let message = PyMessage::new(message);
    let topic = topic.to_string();
    let res = Python::attach(|py| {
        let res = handler.call1(py, (&topic, message))?;
        if res.is_none(py) {
            Ok::<MangledMessage, anyhow::Error>(None)
        } else {
            let (topic, message) = res.extract::<(String, PyMessage)>(py)?;
            Ok::<MangledMessage, anyhow::Error>(Some((topic, message.extract())))
        }
    })?;
    if let Some((topic, message)) = res {
        if topic.is_empty() {
            log::warn!("Topic is empty! The message will be discarded");
            return Ok(None);
        }
        Ok(Some((topic, message)))
    } else {
        Ok(None)
    }
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

    pub fn push(&mut self, topic: String, message: Message, data: Vec<Vec<u8>>) -> Result<()> {
        let (topic, message) = if let Some(handler) = &self.python_handler {
            let res = handle_with_python(handler, topic, message)?;
            if res.is_none() {
                return Ok(());
            }
            res.unwrap()
        } else {
            (topic, message)
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
    pub python_handler: Option<Py<PyAny>>,
}

impl MessageHandler {
    pub fn new(
        queue: Arc<Mutex<PersistentQueueWithCapacity>>,
        writer: NonBlockingWriter,
        idle_sleep: Duration,
        python_handler: Option<Py<PyAny>>,
    ) -> Self {
        Self {
            queue,
            writer,
            idle_sleep,
            last_writer_result: None,
            python_handler,
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

        self.log_last_writer_result();

        let message = load_message(&stored_message.message_bytes);
        let (topic, message) = if let Some(handler) = &self.python_handler {
            let res = handle_with_python(handler, stored_message.topic, message)?;
            if res.is_none() {
                return Ok(());
            }
            res.unwrap()
        } else {
            (stored_message.topic, message)
        };

        let data_refs = stored_message
            .data
            .iter()
            .map(|d| d.as_slice())
            .collect::<Vec<_>>();

        let res = self
            .writer
            .send_message(&topic, &message, &data_refs)
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
