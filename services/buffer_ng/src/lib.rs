pub mod configuration;
pub mod message_handler;
pub mod metric_collector;
pub mod rocksdb;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::Result;
use log::debug;
use parking_lot::Mutex;
use pyo3::{Py, PyAny};
use ::rocksdb::Options;

use crate::configuration::ServiceConfiguration;
use crate::message_handler::{MessageHandler, MessageWriter};
use crate::metric_collector::CommonMetrics;
use crate::rocksdb::PersistentQueueWithCapacity;
use savant_core::transport::zeromq::{NonBlockingReader, NonBlockingWriter, ReaderResult};
use savant_services_common::{fps_meter::FpsMeter, topic_to_string};

/// Run the buffer_ng service loop.
///
/// Creates ingress reader and egress writer from the configuration, sets up
/// the RocksDB-backed persistent queue, then enters the processing loop.
///
/// `ingress_handler` is invoked on every message **after** it is received
/// (before buffering).  `egress_handler` is invoked on every message
/// **before** it is sent (after popping from the buffer).  Either can be
/// `None` to skip that stage.
///
/// When `shutdown` is `Some`, both the ingress and egress loops check the
/// flag on every iteration and exit when it is set to `true`.  When `None`,
/// the loop runs indefinitely.
pub fn run_service_loop(
    conf: &ServiceConfiguration,
    ingress_handler: Option<Py<PyAny>>,
    egress_handler: Option<Py<PyAny>>,
    shutdown: Option<Arc<AtomicBool>>,
) -> Result<()> {
    let reader = NonBlockingReader::try_from(&conf.ingress.socket)?;
    let writer = NonBlockingWriter::try_from(&conf.egress.socket)?;
    let db_opts = Options::default();
    if conf.common.buffer.reset_on_start {
        PersistentQueueWithCapacity::remove_db(&conf.common.buffer.path)?;
    }
    let queue = PersistentQueueWithCapacity::new(
        &conf.common.buffer.path,
        conf.common.buffer.max_length,
        conf.common.buffer.full_threshold_percentage,
        db_opts,
    )?;
    log::info!(
        "Buffer initialized, path: {}, max length: {}, \
        full threshold: {}, reset on start: {}, \
        current length: {}, current disk size: {}",
        conf.common.buffer.path,
        conf.common.buffer.max_length,
        conf.common.buffer.full_threshold_percentage,
        conf.common.buffer.reset_on_start,
        queue.len(),
        queue.disk_size()?
    );

    let queue = Arc::new(Mutex::new(queue));
    let ingress_fps_meter = Arc::new(Mutex::new(FpsMeter::default()));
    let egress_fps_meter = Arc::new(Mutex::new(FpsMeter::default()));
    let common_metrics = CommonMetrics::new();

    let mut message_writer = MessageWriter::new(
        queue.clone(),
        ingress_handler,
        ingress_fps_meter,
        &common_metrics,
    );

    let mut message_handler = MessageHandler::new(
        queue,
        writer,
        conf.common.idle_sleep,
        egress_handler,
        egress_fps_meter,
        &common_metrics,
    );

    let shutdown_egress = shutdown.clone();
    std::thread::spawn(move || loop {
        if let Some(ref flag) = shutdown_egress {
            if flag.load(Ordering::SeqCst) {
                break;
            }
        }
        let res = message_handler.process_stored_message();
        if let Err(e) = res {
            log::warn!(
                target: "buffer_ng::message_handler",
                "Failed to process message delivery: {:?}",
                e
            );
        }
    });

    loop {
        if let Some(ref flag) = shutdown {
            if flag.load(Ordering::SeqCst) {
                debug!("Shutdown flag set, exiting service loop");
                break;
            }
        }

        let message = reader.receive()?;
        match message {
            ReaderResult::Message {
                message,
                topic,
                routing_id: _,
                data,
            } => {
                message_writer.push(topic_to_string(&topic), *message, data)?;
            }
            ReaderResult::Timeout => {
                debug!(
                    target: "buffer_ng::ingress",
                    "Timeout receiving message, waiting for next message."
                );
            }
            ReaderResult::PrefixMismatch { topic, routing_id } => {
                log::warn!(
                    target: "buffer_ng::ingress",
                    "Received message with mismatched prefix: topic: {:?}, routing_id: {:?}",
                    topic_to_string(&topic),
                    topic_to_string(routing_id.as_ref().unwrap_or(&Vec::new()))
                );
            }
            ReaderResult::RoutingIdMismatch { topic, routing_id } => {
                log::warn!(
                    target: "buffer_ng::ingress",
                    "Received message with mismatched routing_id: topic: {:?}, routing_id: {:?}",
                    topic_to_string(&topic),
                    topic_to_string(routing_id.as_ref().unwrap_or(&Vec::new()))
                );
            }
            ReaderResult::TooShort(m) => {
                log::warn!(
                    target: "buffer_ng::ingress",
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
                    target: "buffer_ng::ingress",
                    "Received message with mismatched version: topic: {:?}, routing_id: {:?}, sender_version: {:?}, expected_version: {:?}",
                    topic_to_string(&topic),
                    topic_to_string(routing_id.as_ref().unwrap_or(&Vec::new())),
                    sender_version,
                    expected_version
                );
            }
            ReaderResult::Blacklisted(items) => {
                log::warn!(
                    target: "buffer_ng::ingress",
                    "Received blacklisted message: {:?}",
                    items
                );
            }
        }
    }

    Ok(())
}
