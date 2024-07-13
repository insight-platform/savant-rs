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
        let reader = &self.0;
        reader.receive()
    }

    pub fn is_started(&self) -> bool {
        let reader = &self.0;
        reader.is_alive()
    }

    pub fn shutdown(&self) -> anyhow::Result<()> {
        let reader = &self.0;
        reader.destroy()
    }

    pub fn blacklist_source(&self, source_id: &[u8]) {
        let reader = &self.0;
        reader.blacklist_source(source_id);
    }

    pub fn is_blacklisted(&self, source_id: &[u8]) -> bool {
        let reader = &self.0;
        reader.is_blacklisted(source_id)
    }
}
