use crate::transport::zeromq::reader::{ReaderResult, SourceBlacklist};
use crate::transport::zeromq::{NoopResponder, Reader, ReaderConfig, ZmqSocketProvider};
use parking_lot::Mutex;
use std::sync::Arc;

/// A [`Reader`] that can be shared between threads.
///
/// `receive` holds the reader lock for up to the receive timeout, so the source
/// blacklist is reached through its own handle and never waits on that lock.
#[derive(Clone)]
pub struct SyncReader(
    Arc<Mutex<Reader<NoopResponder, ZmqSocketProvider>>>,
    Arc<SourceBlacklist>,
);

impl SyncReader {
    pub fn new(config: &ReaderConfig) -> anyhow::Result<Self> {
        let reader = Reader::new(config)?;
        let source_blacklist = reader.source_blacklist();
        Ok(Self(Arc::new(Mutex::new(reader)), source_blacklist))
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
        let reader = self.0.lock();
        reader.destroy()
    }

    pub fn blacklist_source(&self, source_id: &[u8]) {
        self.1.blacklist_source(source_id);
    }

    pub fn is_blacklisted(&self, source_id: &[u8]) -> bool {
        self.1.is_blacklisted(source_id)
    }

    /// Whether some thread holds the reader lock, i.e. is inside `receive`.
    #[cfg(test)]
    pub(crate) fn is_receiving(&self) -> bool {
        self.0.is_locked()
    }
}

#[cfg(test)]
mod tests {
    use super::SyncReader;
    use crate::message::Message;
    use crate::test::gen_frame;
    use crate::transport::zeromq::reader::ReaderResult;
    use crate::transport::zeromq::writer_config::WriterConfig;
    use crate::transport::zeromq::{
        NoopResponder, ReaderConfig, TopicPrefixSpec, Writer, ZmqSocketProvider,
    };
    use std::time::{Duration, Instant};

    fn wait_until_receiving(reader: &SyncReader) {
        let deadline = Instant::now() + Duration::from_secs(1);
        while !reader.is_receiving() {
            assert!(
                Instant::now() < deadline,
                "receive never took the reader lock"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    // issue: https://github.com/insight-platform/savant-rs/issues/321
    #[test]
    fn test_blacklist_does_not_wait_for_receive() -> anyhow::Result<()> {
        let conf = ReaderConfig::new()
            .url("router+bind:ipc:///tmp/test/sync-reader-blacklist")?
            .with_topic_prefix_spec(TopicPrefixSpec::SourceId("topic".into()))?
            .with_receive_timeout(2000)?
            .build()?;
        let reader = SyncReader::new(&conf)?;
        let receiving = reader.clone();
        let receiver = std::thread::spawn(move || receiving.receive());
        wait_until_receiving(&reader);

        let started = Instant::now();
        reader.blacklist_source(b"topic");
        assert!(reader.is_blacklisted(b"topic"));
        assert!(!reader.is_blacklisted(b"other"));
        assert!(started.elapsed() < Duration::from_millis(500));

        let result = receiver.join().unwrap()?;
        assert!(matches!(result, ReaderResult::Timeout));
        reader.shutdown()?;
        Ok(())
    }

    #[test]
    fn test_blacklist_is_the_one_receive_uses() -> anyhow::Result<()> {
        let path = "/tmp/test/sync-reader-blacklist-shared";
        std::fs::remove_dir_all(path).unwrap_or_default();
        let reader = SyncReader::new(
            &ReaderConfig::new()
                .url(&format!("rep+bind:ipc://{path}"))?
                .with_fix_ipc_permissions(Some(0o777))?
                .build()?,
        )?;
        let mut writer = Writer::<NoopResponder, ZmqSocketProvider>::new(
            &WriterConfig::new()
                .url(&format!("req+connect:ipc://{path}"))?
                .build()?,
        )?;

        reader.blacklist_source(b"test");
        let receiving = reader.clone();
        let receiver = std::thread::spawn(move || receiving.receive());
        let mut message = Message::video_frame(&gen_frame());
        writer.send_message("test", &mut message, &[])?;

        let result = receiver.join().unwrap()?;
        assert!(matches!(result, ReaderResult::Blacklisted(topic) if topic == b"test"));
        reader.shutdown()?;
        Ok(())
    }
}
