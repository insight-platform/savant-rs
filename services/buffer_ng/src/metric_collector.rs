use std::{
    sync::Arc,
    time::{Duration, SystemTime},
};

use anyhow::Result;
use parking_lot::Mutex;
use savant_core::metrics::{
    get_or_create_counter_family, get_or_create_gauge_family, SharedCounterFamily,
    SharedGaugeFamily,
};
use savant_services_common::fps_meter::FpsMeter;

#[derive(Clone)]
pub struct CommonMetrics {
    pub buffer_size: SharedGaugeFamily,
    pub payload_size: SharedGaugeFamily,
}

impl CommonMetrics {
    pub fn new() -> Self {
        Self {
            buffer_size: get_or_create_gauge_family(
                "buffer_size",
                Some("Number of messages in the buffer"),
                &[],
                None,
            ),
            payload_size: get_or_create_gauge_family(
                "payload_size",
                Some("Size of messages in the buffer"),
                &[],
                None,
            ),
        }
    }
}

#[derive(Clone)]
pub struct IngressMetrics {
    pub ingress_python_none_messages: SharedCounterFamily,
    pub received_messages: SharedCounterFamily,
    pub last_received_message: SharedGaugeFamily,
    pub pushed_messages: SharedCounterFamily,
    pub last_pushed_message: SharedGaugeFamily,
    pub dropped_messages: SharedCounterFamily,
    pub last_dropped_message: SharedGaugeFamily,
    pub buffer_size: SharedGaugeFamily,
    pub payload_size: SharedGaugeFamily,
}

impl IngressMetrics {
    pub fn new(common_metrics: &CommonMetrics) -> Self {
        Self {
            ingress_python_none_messages: get_or_create_counter_family(
                "ingress_python_none_messages",
                Some("Number of messages dropped by the adapter"),
                &[],
                None,
            ),
            received_messages: get_or_create_counter_family(
                "received_messages",
                Some("Number of messages received by the adapter"),
                &[],
                None,
            ),
            last_received_message: get_or_create_gauge_family(
                "last_received_message",
                Some("Number of messages received by the adapter"),
                &[],
                None,
            ),
            pushed_messages: get_or_create_counter_family(
                "pushed_messages",
                Some("Number of messages pushed by the adapter"),
                &[],
                None,
            ),
            last_pushed_message: get_or_create_gauge_family(
                "last_pushed_message",
                Some("Number of messages pushed by the adapter"),
                &[],
                None,
            ),
            dropped_messages: get_or_create_counter_family(
                "dropped_messages",
                Some("Number of messages dropped by the adapter"),
                &[],
                None,
            ),
            last_dropped_message: get_or_create_gauge_family(
                "last_dropped_message",
                Some("Number of messages dropped by the adapter"),
                &[],
                None,
            ),
            buffer_size: common_metrics.buffer_size.clone(),
            payload_size: common_metrics.payload_size.clone(),
        }
    }
}

#[derive(Clone)]
pub struct EgressMetrics {
    pub egress_python_none_messages: SharedCounterFamily,
    pub popped_messages: SharedCounterFamily,
    pub last_popped_message: SharedGaugeFamily,
    pub sent_messages: SharedCounterFamily,
    pub last_sent_message: SharedGaugeFamily,
    pub undelivered_messages: SharedCounterFamily,
    pub last_undelivered_message: SharedGaugeFamily,
    pub buffer_size: SharedGaugeFamily,
    pub payload_size: SharedGaugeFamily,
}

impl EgressMetrics {
    pub fn new(common_metrics: &CommonMetrics) -> Self {
        Self {
            egress_python_none_messages: get_or_create_counter_family(
                "egress_python_none_messages",
                Some("Number of messages dropped by the adapter"),
                &[],
                None,
            ),
            popped_messages: get_or_create_counter_family(
                "popped_messages",
                Some("Number of messages popped from the buffer"),
                &[],
                None,
            ),
            last_popped_message: get_or_create_gauge_family(
                "last_popped_message",
                Some("Number of messages popped from the buffer"),
                &[],
                None,
            ),
            sent_messages: get_or_create_counter_family(
                "sent_messages",
                Some("Number of messages sent to the sink ZeroMQ socket"),
                &["reason"],
                None,
            ),
            last_sent_message: get_or_create_gauge_family(
                "last_sent_message",
                Some("Number of messages sent to the sink ZeroMQ socket"),
                &["reason"],
                None,
            ),
            undelivered_messages: get_or_create_counter_family(
                "undelivered_messages",
                Some("Number of messages undelivered to the sink ZeroMQ socket"),
                &["reason"],
                None,
            ),
            last_undelivered_message: get_or_create_gauge_family(
                "last_undelivered_message",
                Some("Number of messages undelivered to the sink ZeroMQ socket"),
                &["reason"],
                None,
            ),
            buffer_size: common_metrics.buffer_size.clone(),
            payload_size: common_metrics.payload_size.clone(),
        }
    }
}

pub struct StatsLogger {
    ingress_fps_meter: Arc<Mutex<FpsMeter>>,
    egress_fps_meter: Arc<Mutex<FpsMeter>>,
    ingress_metrics: IngressMetrics,
    egress_metrics: EgressMetrics,
    stats_log_interval: Duration,
    last_stats_log: SystemTime,
}

impl StatsLogger {
    pub fn new(
        ingress_fps_meter: Arc<Mutex<FpsMeter>>,
        egress_fps_meter: Arc<Mutex<FpsMeter>>,
        ingress_metrics: IngressMetrics,
        egress_metrics: EgressMetrics,
        stats_period: Duration,
    ) -> Self {
        Self {
            ingress_fps_meter,
            egress_fps_meter,
            ingress_metrics,
            egress_metrics,
            stats_log_interval: stats_period,
            last_stats_log: SystemTime::now(),
        }
    }

    pub fn log_stats(&mut self) -> Result<()> {
        let now = SystemTime::now();
        if now.duration_since(self.last_stats_log).unwrap() < self.stats_log_interval {
            return Ok(());
        }
        self.last_stats_log = now;
        let ingress_fps = self.ingress_fps_meter.lock().get_fps();

        let buffer_size = self
            .ingress_metrics
            .buffer_size
            .lock()
            .get(&[])?
            .unwrap_or(0.0);

        let buffer_payload_size = self
            .ingress_metrics
            .payload_size
            .lock()
            .get(&[])?
            .unwrap_or(0.0);

        let received_messages = self
            .ingress_metrics
            .received_messages
            .lock()
            .get(&[])?
            .unwrap_or(0);

        let pushed_messages = self
            .ingress_metrics
            .pushed_messages
            .lock()
            .get(&[])?
            .unwrap_or(0);

        let dropped_messages = self
            .ingress_metrics
            .dropped_messages
            .lock()
            .get(&[])?
            .unwrap_or(0);

        let ingress_python_none_messages = self
            .ingress_metrics
            .ingress_python_none_messages
            .lock()
            .get(&[])?
            .unwrap_or(0);

        let egress_fps = self.egress_fps_meter.lock().get_fps();

        let egress_python_none_messages = self
            .egress_metrics
            .egress_python_none_messages
            .lock()
            .get(&[])?
            .unwrap_or(0);

        let (sent_messages_send_success, sent_messages_ack_success) = {
            let lock = self.egress_metrics.sent_messages.lock();
            (
                lock.get(&["send_success"])?.unwrap_or(0),
                lock.get(&["ack_success"])?.unwrap_or(0),
            )
        };

        let (undelivered_messages_send_timeout, undelivered_messages_ack_timeout) = {
            let lock = self.egress_metrics.undelivered_messages.lock();
            (
                lock.get(&["send_timeout"])?.unwrap_or(0),
                lock.get(&["ack_timeout"])?.unwrap_or(0),
            )
        };

        log::info!(
            "Buffer size: {}, \
             Buffer payload size: {}, \
             Ingress FPS: {:.2}, \
             Received messages: {}, \
             Pushed messages: {}, \
             Queue full dropped messages: {}, \
             Ingress Python none messages: {}, \
             Egress FPS: {:.2}, \
             Sent messages [success]: {}, \
             Sent messages [ack success]: {}, \
             Undelivered messages [sendtimeout]: {}, \
             Undelivered messages [ack timeout]: {}, \
             Egress Python none messages: {}",
            buffer_size,
            buffer_payload_size,
            ingress_fps,
            received_messages,
            pushed_messages,
            dropped_messages,
            ingress_python_none_messages,
            egress_fps,
            sent_messages_send_success,
            sent_messages_ack_success,
            undelivered_messages_send_timeout,
            undelivered_messages_ack_timeout,
            egress_python_none_messages,
        );

        Ok(())
    }
}
