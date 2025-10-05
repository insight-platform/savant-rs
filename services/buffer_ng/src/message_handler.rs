use std::{
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use crate::{
    metric_collector::{CommonMetrics, EgressMetrics, IngressMetrics},
    rocksdb::PersistentQueueWithCapacity,
};
use anyhow::{Context, Result};
use bitcode::{Decode, Encode};
use parking_lot::Mutex;
use pyo3::{Py, PyAny, Python};
use savant_core::{
    message::{load_message, save_message, Message},
    transport::zeromq::{NonBlockingWriter, WriterResult},
};
use savant_core_py::primitives::message::Message as PyMessage;
use savant_services_common::fps_meter::FpsMeter;

#[derive(Debug, Clone, Decode, Encode)]
struct StoredMessage {
    topic: String,
    message_bytes: Vec<u8>,
    data: Vec<Vec<u8>>,
}

pub struct MessageWriter {
    pub python_handler: Option<Py<PyAny>>,
    pub queue: Arc<Mutex<PersistentQueueWithCapacity>>,
    pub metrics: IngressMetrics,
    pub ingress_fps_meter: Arc<Mutex<FpsMeter>>,
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
        ingress_fps_meter: Arc<Mutex<FpsMeter>>,
        common_metrics: &CommonMetrics,
    ) -> Self {
        Self {
            python_handler,
            queue,
            metrics: IngressMetrics::new(common_metrics),
            ingress_fps_meter,
        }
    }

    pub fn get_metrics(&self) -> IngressMetrics {
        self.metrics.clone()
    }

    pub fn push(&mut self, topic: String, message: Message, data: Vec<Vec<u8>>) -> Result<()> {
        self.ingress_fps_meter.lock().increment();

        self.metrics.received_messages.lock().inc(1, &[])?;

        self.metrics.last_received_message.lock().set(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
            &[],
        )?;

        let (topic, message) = if let Some(handler) = &self.python_handler {
            let res = handle_with_python(handler, topic, message)?;
            if res.is_none() {
                self.metrics
                    .ingress_python_none_messages
                    .lock()
                    .inc(1, &[])?;
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
        {
            let mut q = self.queue.lock();
            let res = q.push(&[&serialized_message]);
            if let Err(e) = res {
                log::warn!(
                    target: "buffer_ng::message_handler::writer",
                    "Failed to push message to queue: {:?}",
                    e
                );

                self.metrics.dropped_messages.lock().inc(1, &[])?;

                self.metrics.last_dropped_message.lock().set(
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs_f64(),
                    &[],
                )?;
            }

            self.metrics.buffer_size.lock().set(q.len() as f64, &[])?;

            self.metrics
                .payload_size
                .lock()
                .set(q.payload_size() as f64, &[])?;

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
        }
        self.metrics.pushed_messages.lock().inc(1, &[])?;

        self.metrics.last_pushed_message.lock().set(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
            &[],
        )?;

        Ok(())
    }
}

pub struct MessageHandler {
    pub queue: Arc<Mutex<PersistentQueueWithCapacity>>,
    pub writer: NonBlockingWriter,
    pub idle_sleep: Duration,
    pub last_writer_result: Option<WriterResult>,
    pub python_handler: Option<Py<PyAny>>,
    pub metrics: EgressMetrics,
    pub egress_fps_meter: Arc<Mutex<FpsMeter>>,
}

impl MessageHandler {
    pub fn new(
        queue: Arc<Mutex<PersistentQueueWithCapacity>>,
        writer: NonBlockingWriter,
        idle_sleep: Duration,
        python_handler: Option<Py<PyAny>>,
        egress_fps_meter: Arc<Mutex<FpsMeter>>,
        common_metrics: &CommonMetrics,
    ) -> Self {
        Self {
            queue,
            writer,
            idle_sleep,
            last_writer_result: None,
            python_handler,
            metrics: EgressMetrics::new(common_metrics),
            egress_fps_meter,
        }
    }

    pub fn get_metrics(&self) -> EgressMetrics {
        self.metrics.clone()
    }

    pub fn process_stored_message(&mut self) -> Result<()> {
        let read = {
            let mut q = self.queue.lock();

            let read = q
                .pop(1)
                .with_context(|| "Failed to pop message from queue")?;

            self.metrics.buffer_size.lock().set(q.len() as f64, &[])?;

            self.metrics
                .payload_size
                .lock()
                .set(q.payload_size() as f64, &[])?;

            read
        };

        if read.is_empty() {
            std::thread::sleep(self.idle_sleep);
            self.log_last_writer_result()?;
            return Ok(());
        }

        self.metrics.popped_messages.lock().inc(1, &[])?;

        self.metrics.last_popped_message.lock().set(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs_f64(),
            &[],
        )?;

        let stored_message: StoredMessage =
            bitcode::decode(&read[0]).with_context(|| "Failed to decode stored message")?;

        self.log_last_writer_result()?;

        let message = load_message(&stored_message.message_bytes);
        let (topic, message) = if let Some(handler) = &self.python_handler {
            let res = handle_with_python(handler, stored_message.topic, message)?;
            if res.is_none() {
                self.metrics
                    .egress_python_none_messages
                    .lock()
                    .inc(1, &[])?;
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

    fn log_last_writer_result(&mut self) -> Result<()> {
        if let Some(res) = &self.last_writer_result {
            match res {
                WriterResult::SendTimeout => {
                    log::warn!("Send timeout occurred!");

                    self.metrics
                        .undelivered_messages
                        .lock()
                        .inc(1, &["send_timeout"])?;

                    self.metrics.last_undelivered_message.lock().set(
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs_f64(),
                        &["send_timeout"],
                    )?;
                }
                WriterResult::AckTimeout(timeout) => {
                    log::warn!("Ack timeout occurred! Timeout: {timeout}");

                    self.metrics
                        .undelivered_messages
                        .lock()
                        .inc(1, &["ack_timeout"])?;

                    self.metrics.last_undelivered_message.lock().set(
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs_f64(),
                        &["ack_timeout"],
                    )?;
                }
                WriterResult::Ack {
                    send_retries_spent,
                    receive_retries_spent,
                    time_spent,
                } => {
                    log::debug!("Ack occurred! Send retries spent: {send_retries_spent}, Receive retries spent: {receive_retries_spent}, Time spent: {time_spent}");

                    self.metrics.sent_messages.lock().inc(1, &["ack_success"])?;

                    self.metrics.last_sent_message.lock().set(
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs_f64(),
                        &["ack_success"],
                    )?;

                    self.egress_fps_meter.lock().increment();
                }
                WriterResult::Success {
                    retries_spent,
                    time_spent,
                } => {
                    log::debug!(
                        "Success occurred! Retries spent: {retries_spent}, Time spent: {time_spent}"
                    );
                    self.metrics
                        .sent_messages
                        .lock()
                        .inc(1, &["send_success"])?;

                    self.metrics.last_sent_message.lock().set(
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs_f64(),
                        &["send_success"],
                    )?;

                    self.egress_fps_meter.lock().increment();
                }
            }
        }
        self.last_writer_result = None;
        Ok(())
    }
}
