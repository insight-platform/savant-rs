use crate::transport::zeromq::reader::ReaderResult;
use crate::transport::zeromq::{NoopResponder, Reader, ReaderConfig, ZmqSocketProvider};
use std::sync::Arc;

#[derive(Clone)]
pub struct SyncReader(Arc<Reader<NoopResponder, ZmqSocketProvider>>);

impl SyncReader {
    pub fn new(config: &ReaderConfig) -> anyhow::Result<Self> {
        Ok(Self(Arc::new(Reader::new(config)?)))
    }

    pub fn receive(&self) -> anyhow::Result<ReaderResult> {
        self.0.receive()
    }

    pub fn is_started(&self) -> bool {
        self.0.is_alive()
    }

    pub fn shutdown(&self) -> anyhow::Result<()> {
        self.0.destroy()
    }

    pub fn blacklist_source(&self, source_id: &[u8]) {
        self.0.blacklist_source(source_id);
    }

    pub fn is_blacklisted(&self, source_id: &[u8]) -> bool {
        self.0.is_blacklisted(source_id)
    }
}
