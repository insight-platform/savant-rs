use crate::transport::zeromq::{
    NoopResponder, Writer, WriterConfig, WriterResult, ZmqSocketProvider,
};
use parking_lot::Mutex;
use std::sync::Arc;

#[derive(Clone)]
pub struct SyncWriter(Arc<Mutex<Writer<NoopResponder, ZmqSocketProvider>>>);

impl SyncWriter {
    pub fn new(config: &WriterConfig) -> anyhow::Result<Self> {
        Ok(Self(Arc::new(Mutex::new(Writer::new(config)?))))
    }

    pub fn send_eos(&self, source_id: &str) -> anyhow::Result<WriterResult> {
        let mut writer = self.0.lock();
        writer.send_eos(source_id)
    }

    pub fn send_eos_with_topic(
        &self,
        topic: &str,
        source_id: &str,
    ) -> anyhow::Result<WriterResult> {
        let mut writer = self.0.lock();
        writer.send_eos_with_topic(topic, source_id)
    }

    pub fn send_message(
        &self,
        topic: &str,
        message: &mut crate::message::Message,
        data: &[&[u8]],
    ) -> anyhow::Result<WriterResult> {
        let mut writer = self.0.lock();
        writer.send_message(topic, message, data)
    }

    pub fn is_started(&self) -> bool {
        let writer = self.0.lock();
        writer.is_started()
    }

    pub fn shutdown(&self) -> anyhow::Result<()> {
        let mut writer = self.0.lock();
        writer.destroy()
    }
}
