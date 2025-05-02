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
pub struct SinkOptions {
    pub send_timeout: Duration,
    pub send_retries: usize,
    pub receive_timeout: Duration,
    pub receive_retries: usize,
    pub send_hwm: usize,
    pub receive_hwm: usize,
    pub inflight_ops: usize,
}

impl Default for SinkOptions {
    fn default() -> Self {
        Self {
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

#[derive(Debug, Serialize, Deserialize, Hash, PartialEq, Eq, Clone)]
pub struct SinkConfiguration {
    pub url: String,
    pub options: Option<SinkOptions>,
}

impl Default for SinkConfiguration {
    fn default() -> Self {
        Self {
            url: String::from("dealer+connect:ipc:///tmp/in"),
            options: Some(SinkOptions::default()),
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
            options: Some(SinkOptions {
                send_timeout,
                send_retries,
                receive_timeout,
                receive_retries,
                send_hwm,
                receive_hwm,
                inflight_ops,
            }),
        }
    }
}

impl TryFrom<&SinkConfiguration> for NonBlockingWriter {
    type Error = anyhow::Error;

    fn try_from(configuration: &SinkConfiguration) -> Result<Self, Self::Error> {
        let default_options = SinkOptions::default();
        let options = configuration.options.as_ref().unwrap_or(&default_options);
        let conf = WriterConfigBuilder::default()
            .url(&configuration.url)?
            .with_receive_timeout(options.receive_timeout.as_millis() as i32)?
            .with_send_timeout(options.send_timeout.as_millis() as i32)?
            .with_receive_retries(options.receive_retries as i32)?
            .with_send_retries(options.send_retries as i32)?
            .with_receive_hwm(options.receive_hwm as i32)?
            .with_send_hwm(options.send_hwm as i32)?
            .build()?;

        let mut w = NonBlockingWriter::new(&conf, options.inflight_ops)?;
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
