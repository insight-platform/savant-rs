use crate::transport::zeromq::reader::ReaderResult;
use crate::transport::zeromq::{NoopResponder, Reader, ReaderConfig, ZmqSocketProvider};
use parking_lot::Mutex;
use std::sync::Arc;

#[derive(Clone)]
pub struct SyncReader(Arc<Mutex<Reader<NoopResponder, ZmqSocketProvider>>>);

impl SyncReader {
    pub fn new(config: &ReaderConfig) -> anyhow::Result<Self> {
        Ok(Self(Arc::new(Mutex::new(Reader::new(config)?))))
    }

    pub fn receive(&self) -> anyhow::Result<ReaderResult> {
        let mut reader = self.0.lock();
        reader.receive()
    }

    pub fn is_started(&self) -> bool {
        let reader = self.0.lock();
        reader.is_alive()
    }

    pub fn shutdown(&self) -> anyhow::Result<()> {
        let mut reader = self.0.lock();
        reader.destroy()
    }

    pub fn blacklist_source(&self, source_id: &[u8]) {
        let mut reader = self.0.lock();
        reader.blacklist_source(source_id);
    }

    pub fn is_blacklisted(&self, source_id: &[u8]) -> bool {
        let mut reader = self.0.lock();
        reader.is_blacklisted(source_id)
    }
}
