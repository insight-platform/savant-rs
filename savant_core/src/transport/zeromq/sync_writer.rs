use crate::transport::zeromq::{NoopResponder, Writer, WriterResult, ZmqSocketProvider};
use parking_lot::Mutex;
use std::sync::Arc;

#[derive(Clone)]
pub struct SyncWriter(Arc<Mutex<Writer<NoopResponder, ZmqSocketProvider>>>);

impl SyncWriter {
    pub fn new(writer: Writer<NoopResponder, ZmqSocketProvider>) -> Self {
        Self(Arc::new(Mutex::new(writer)))
    }

    pub fn send_eos(&self, topic: &str) -> anyhow::Result<WriterResult> {
        let mut writer = self.0.lock();
        writer.send_eos(topic)
    }

    pub fn send_message(
        &self,
        topic: &str,
        message: &crate::message::Message,
        data: &[&[u8]],
    ) -> anyhow::Result<WriterResult> {
        let mut writer = self.0.lock();
        writer.send_message(topic, message, data)
    }

    pub fn is_alive(&self) -> bool {
        let writer = self.0.lock();
        writer.is_alive()
    }

    pub fn destroy(&self) -> anyhow::Result<()> {
        let mut writer = self.0.lock();
        writer.destroy()
    }
}
