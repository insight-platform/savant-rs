use anyhow::Result;
use log::info;
use savant_core::message::Message;
use savant_core::transport::zeromq::{
    NonBlockingWriter, WriteOperationResult, WriterConfigBuilder,
};
use serde::{Deserialize, Serialize};
use std::thread;
use std::time::Duration;

pub mod cache;

#[derive(Debug, Serialize, Deserialize, Default)]
pub enum WriterSocketType {
    #[default]
    Dealer,
    Pub,
    Req,
}

#[derive(Debug, Serialize, Deserialize, Hash, PartialEq, Eq, Clone)]
pub struct SinkConfiguration {
    pub(crate) url: String,
    pub(crate) send_timeout: Duration,
    pub(crate) send_retries: usize,
    pub(crate) receive_timeout: Duration,
    pub(crate) receive_retries: usize,
    pub(crate) send_hwm: usize,
    pub(crate) receive_hwm: usize,
    pub(crate) inflight_ops: usize,
}

impl Default for SinkConfiguration {
    fn default() -> Self {
        Self {
            url: String::from("dealer+connect:ipc:///tmp/in"),
            send_timeout: Duration::from_secs(1),
            send_retries: 3,
            receive_timeout: Duration::from_secs(1),
            receive_retries: 3,
            send_hwm: 1000,
            receive_hwm: 1000,
            inflight_ops: 100,
        }
    }
}

#[allow(clippy::too_many_arguments)]
impl SinkConfiguration {
    pub fn new(
        url: &str,
        send_timeout: Duration,
        send_retries: usize,
        receive_timeout: Duration,
        receive_retries: usize,
        send_hwm: usize,
        receive_hwm: usize,
        inflight_ops: usize,
    ) -> Self {
        Self {
            url: url.to_string(),
            send_timeout,
            send_retries,
            receive_timeout,
            receive_retries,
            send_hwm,
            receive_hwm,
            inflight_ops,
        }
    }

    #[cfg(test)]
    pub fn test_dealer_connect_sink() -> Self {
        Self::new(
            "dealer+connect:ipc:///tmp/in",
            Duration::from_secs(1),
            3,
            Duration::from_secs(1),
            3,
            1000,
            100,
            100,
        )
    }
}

impl TryFrom<&SinkConfiguration> for NonBlockingWriter {
    type Error = anyhow::Error;

    fn try_from(configuration: &SinkConfiguration) -> Result<Self, Self::Error> {
        let conf = WriterConfigBuilder::default()
            .url(&configuration.url)?
            .with_receive_timeout(configuration.receive_timeout.as_millis() as i32)?
            .with_send_timeout(configuration.send_timeout.as_millis() as i32)?
            .with_receive_retries(configuration.receive_retries as i32)?
            .with_send_retries(configuration.send_retries as i32)?
            .with_receive_hwm(configuration.receive_hwm as i32)?
            .with_send_hwm(configuration.send_hwm as i32)?
            .build()?;

        // if *conf.bind() {
        //     bail!("JobWriter configuration must be a connect socket.");
        // }

        let mut w = NonBlockingWriter::new(&conf, configuration.inflight_ops)?;
        w.start()?;
        Ok(w)
    }
}

pub struct JobWriter(pub Option<NonBlockingWriter>);

impl JobWriter {
    pub fn new(w: NonBlockingWriter) -> Self {
        Self(Some(w))
    }

    pub fn send_eos(&self, topic: &str) -> Result<WriteOperationResult> {
        self.0
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Writer is not available."))?
            .send_eos(topic)
    }

    pub fn send_message(
        &self,
        topic: &str,
        message: &Message,
        payload: &[&[u8]],
    ) -> Result<WriteOperationResult> {
        self.0
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Writer is not available."))?
            .send_message(topic, message, payload)
    }
}

impl From<NonBlockingWriter> for JobWriter {
    fn from(w: NonBlockingWriter) -> Self {
        Self::new(w)
    }
}

impl Drop for JobWriter {
    fn drop(&mut self) {
        let w = self.0.take();
        thread::spawn(move || {
            if let Some(mut w) = w {
                info!(target: "relay::db::writer::shutdown",
                    "Shutting down writer");
                w.shutdown().expect("Failed to shutdown writer");
            }
        });
    }
}
