use crate::transport::zeromq::reader::ReaderResult;
use crate::transport::zeromq::{ReaderConfig, SyncReader};
use crossbeam::channel::Receiver;
use std::sync::{Arc, OnceLock};

pub struct NonBlockingReader {
    config: ReaderConfig,
    thread: Option<std::thread::JoinHandle<()>>,
    receiver: Option<Receiver<anyhow::Result<ReaderResult>>>,
    is_started: OnceLock<()>,
    is_shutdown: Arc<OnceLock<()>>,
    results_queue_size: usize,
}

impl NonBlockingReader {
    pub fn new(config: &ReaderConfig, results_queue_size: usize) -> anyhow::Result<Self> {
        Ok(Self {
            config: config.clone(),
            thread: None,
            receiver: None,
            is_started: OnceLock::new(),
            is_shutdown: Arc::new(OnceLock::new()),
            results_queue_size,
        })
    }

    pub fn start(&mut self) -> anyhow::Result<()> {
        if self.is_shutdown() {
            anyhow::bail!("Reader is shutdown.");
        }
        if self.is_started() {
            anyhow::bail!("Reader is already started.");
        }
        _ = self.is_started.set(());
        let (sender, receiver) = crossbeam::channel::bounded(self.results_queue_size);
        let reader = SyncReader::new(&self.config)?;
        let is_shutdown = self.is_shutdown.clone();
        let thread = std::thread::spawn(move || loop {
            let res = reader.receive();
            if sender.send(res).is_err() || is_shutdown.get().is_some() {
                _ = is_shutdown.set(());
                break;
            }
        });
        self.thread = Some(thread);
        self.receiver = Some(receiver);
        Ok(())
    }

    pub fn enqueued_results(&self) -> usize {
        if let Some(receiver) = &self.receiver {
            receiver.len()
        } else {
            0
        }
    }

    pub fn is_shutdown(&self) -> bool {
        self.is_shutdown.get().is_some()
    }

    pub fn is_started(&self) -> bool {
        self.is_started.get().is_some()
    }

    pub fn shutdown(&mut self) -> anyhow::Result<()> {
        if self.is_shutdown() {
            anyhow::bail!("Reader is shutdown.");
        }
        if !self.is_started() {
            anyhow::bail!("Reader is not started.");
        }
        if let Some(thread) = self.thread.take() {
            _ = self.is_shutdown.set(());
            thread
                .join()
                .map_err(|_| anyhow::anyhow!("Failed to join thread."))?;
        } else {
            anyhow::bail!("Reader is not running.");
        }
        Ok(())
    }

    pub fn receive(&self) -> anyhow::Result<ReaderResult> {
        if !self.is_started() {
            anyhow::bail!("Reader is not started.");
        }
        if self.is_shutdown() {
            anyhow::bail!("Reader is shutdown.");
        }
        if let Some(receiver) = &self.receiver {
            receiver
                .recv()
                .map_err(|e| anyhow::anyhow!("Failed to receive message: {:?}", e))?
        } else {
            anyhow::bail!("Reader is not running.");
        }
    }

    pub fn try_receive(&self) -> Option<anyhow::Result<ReaderResult>> {
        if !self.is_started() {
            return Some(Err(anyhow::anyhow!("Reader is not started.")));
        }
        if self.is_shutdown() {
            return Some(Err(anyhow::anyhow!("Reader is shutdown.")));
        }
        if let Some(receiver) = &self.receiver {
            match receiver.try_recv() {
                Ok(res) => Some(res),
                Err(e) => match e {
                    crossbeam::channel::TryRecvError::Empty => None,
                    crossbeam::channel::TryRecvError::Disconnected => {
                        Some(Err(anyhow::anyhow!("Failed to receive message: {:?}", e)))
                    }
                },
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::transport::zeromq::reader::ReaderResult;
    use crate::transport::zeromq::{ReaderConfig, TopicPrefixSpec};

    #[test]
    fn test_blocking_idling() -> anyhow::Result<()> {
        let conf = ReaderConfig::new()
            .url("router+bind:ipc:///tmp/test/blocking-reader-idling")?
            .with_topic_prefix_spec(TopicPrefixSpec::SourceId("topic".into()))?
            .with_receive_timeout(100)?
            .build()?;
        let mut reader = super::NonBlockingReader::new(&conf, 1)?;
        reader.start()?;
        let now = std::time::Instant::now();
        let recv = reader.receive();
        let elapsed = now.elapsed().as_millis();
        assert!(elapsed >= 100 && elapsed < 200);
        assert!(recv.is_ok());
        let recv = recv.unwrap();
        assert!(matches!(recv, ReaderResult::Timeout));
        reader.shutdown()?;
        Ok(())
    }

    #[test]
    fn test_nonblocking_idling() -> anyhow::Result<()> {
        let conf = ReaderConfig::new()
            .url("router+bind:ipc:///tmp/test/nonblocking-reader-idling")?
            .with_topic_prefix_spec(TopicPrefixSpec::SourceId("topic".into()))?
            .with_receive_timeout(100)?
            .build()?;
        let mut reader = super::NonBlockingReader::new(&conf, 1)?;
        reader.start()?;
        let now = std::time::Instant::now();
        let recv = reader.try_receive();
        let elapsed = now.elapsed().as_millis();
        assert!(elapsed < 100);
        assert!(recv.is_none());
        reader.shutdown()?;
        Ok(())
    }

    #[test]
    fn test_double_start() -> anyhow::Result<()> {
        let conf = ReaderConfig::new()
            .url("router+bind:ipc:///tmp/test/nonblocking-reader-double-start")?
            .with_topic_prefix_spec(TopicPrefixSpec::SourceId("topic".into()))?
            .with_receive_timeout(100)?
            .build()?;
        let mut reader = super::NonBlockingReader::new(&conf, 1)?;
        reader.start()?;
        let res = reader.start();
        assert!(res.is_err());
        reader.shutdown()?;
        Ok(())
    }

    #[test]
    fn test_double_shutdown() -> anyhow::Result<()> {
        let conf = ReaderConfig::new()
            .url("router+bind:ipc:///tmp/test/nonblocking-reader-double-shutdown")?
            .with_topic_prefix_spec(TopicPrefixSpec::SourceId("topic".into()))?
            .with_receive_timeout(100)?
            .build()?;
        let mut reader = super::NonBlockingReader::new(&conf, 1)?;
        reader.start()?;
        reader.shutdown()?;
        let res = reader.shutdown();
        assert!(res.is_err());
        Ok(())
    }

    #[test]
    fn test_shutdown_without_start() -> anyhow::Result<()> {
        let conf = ReaderConfig::new()
            .url("router+bind:ipc:///tmp/test/nonblocking-reader-shutdown-without-start")?
            .with_topic_prefix_spec(TopicPrefixSpec::SourceId("topic".into()))?
            .with_receive_timeout(100)?
            .build()?;
        let mut reader = super::NonBlockingReader::new(&conf, 1)?;
        let res = reader.shutdown();
        assert!(res.is_err());
        Ok(())
    }

    #[test]
    fn test_recv_without_start() -> anyhow::Result<()> {
        let conf = ReaderConfig::new()
            .url("router+bind:ipc:///tmp/test/nonblocking-reader-recv-without-start")?
            .with_topic_prefix_spec(TopicPrefixSpec::SourceId("topic".into()))?
            .with_receive_timeout(100)?
            .build()?;
        let reader = super::NonBlockingReader::new(&conf, 1)?;
        let res = reader.receive();
        assert!(res.is_err());
        let res = reader.try_receive().unwrap();
        assert!(res.is_err());
        Ok(())
    }
}
