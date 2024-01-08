use crate::transport::zeromq::reader::ReaderResult;
use crate::transport::zeromq::{ReaderConfig, SyncReader};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, OnceLock};

pub struct NonblockingReader {
    config: ReaderConfig,
    thread: Option<std::thread::JoinHandle<()>>,
    receiver: Option<Receiver<anyhow::Result<ReaderResult>>>,
    is_shutdown: Arc<OnceLock<()>>,
}

impl NonblockingReader {
    pub fn new(config: &ReaderConfig) -> anyhow::Result<Self> {
        Ok(Self {
            config: config.clone(),
            thread: None,
            receiver: None,
            is_shutdown: Arc::new(OnceLock::new()),
        })
    }

    pub fn start(&mut self) -> anyhow::Result<()> {
        let (sender, receiver) = std::sync::mpsc::channel();
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

    pub fn is_shutdown(&self) -> bool {
        self.is_shutdown.get().is_some()
    }

    pub fn shutdown(&mut self) -> anyhow::Result<()> {
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
        if self.is_shutdown() {
            return None;
        }
        if let Some(receiver) = &self.receiver {
            match receiver.try_recv() {
                Ok(res) => Some(res),
                Err(e) => match e {
                    std::sync::mpsc::TryRecvError::Empty => None,
                    std::sync::mpsc::TryRecvError::Disconnected => {
                        Some(Err(anyhow::anyhow!("Failed to receive message: {:?}", e)))
                    }
                },
            }
        } else {
            None
        }
    }
}
