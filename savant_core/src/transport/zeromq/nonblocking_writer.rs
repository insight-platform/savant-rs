use crate::message::Message;
use crate::primitives::eos::EndOfStream;
use crate::transport::zeromq::{SyncWriter, WriterConfig, WriterResult};
use crossbeam::channel::{Receiver, Sender, TryRecvError};
use std::cell::OnceCell;
use std::sync::{Arc, OnceLock};

enum Command {
    Message(
        String,
        Box<Message>,
        Vec<Vec<u8>>,
        Sender<anyhow::Result<WriterResult>>,
    ),
    Shutdown,
}

pub struct WriteOperationResult(Option<Receiver<anyhow::Result<WriterResult>>>);

impl WriteOperationResult {
    pub fn get(&self) -> anyhow::Result<WriterResult> {
        if let Some(receiver) = &self.0 {
            receiver.recv()?
        } else {
            anyhow::bail!("Write operation result is no longer available.")
        }
    }
    pub fn try_get(&self) -> anyhow::Result<Option<anyhow::Result<WriterResult>>> {
        if let Some(receiver) = &self.0 {
            match receiver.try_recv() {
                Ok(res) => Ok(Some(res)),
                Err(e) => match e {
                    TryRecvError::Empty => Ok(None),
                    TryRecvError::Disconnected => {
                        Err(anyhow::anyhow!("Failed to receive message: {:?}", e))
                    }
                },
            }
        } else {
            anyhow::bail!("Write operation result is no longer available.")
        }
    }
}

pub struct NonblockingWriter {
    config: WriterConfig,
    max_inflight_messages: usize,
    thread: Option<std::thread::JoinHandle<anyhow::Result<()>>>,
    ops_queue: Option<Sender<Command>>,
    is_started: OnceCell<()>,
    is_shutdown: Arc<OnceLock<()>>,
}

impl NonblockingWriter {
    pub fn new(config: &WriterConfig, max_inflight_messages: usize) -> anyhow::Result<Self> {
        Ok(Self {
            config: config.clone(),
            max_inflight_messages,
            thread: None,
            ops_queue: None,
            is_started: OnceCell::new(),
            is_shutdown: Arc::new(OnceLock::new()),
        })
    }

    pub fn is_started(&self) -> bool {
        self.is_started.get().is_some()
    }

    pub fn is_shutdown(&self) -> bool {
        self.is_shutdown.get().is_some()
    }

    pub fn shutdown(&mut self) -> anyhow::Result<()> {
        if self.is_shutdown() {
            anyhow::bail!("Writer is shutdown.");
        }
        if !self.is_started() {
            anyhow::bail!("Writer is not started.");
        }
        self.ops_queue
            .as_ref()
            .unwrap()
            .send(Command::Shutdown)
            .map_err(|e| anyhow::anyhow!("Failed to send shutdown command to channel: {:?}", e))?;
        if let Some(thread) = self.thread.take() {
            _ = self.is_shutdown.set(());
            thread
                .join()
                .map_err(|_| anyhow::anyhow!("Failed to join thread."))??;
        } else {
            anyhow::bail!("Writer is not running.");
        }
        Ok(())
    }

    pub fn start(&mut self) -> anyhow::Result<()> {
        if self.is_shutdown() {
            anyhow::bail!("Writer is shutdown.");
        }
        if self.is_started() {
            anyhow::bail!("Writer is already started.");
        }
        _ = self.is_started.set(());
        let (sender, receiver) = crossbeam::channel::bounded(self.max_inflight_messages);
        let is_shutdown = self.is_shutdown.clone();
        let writer = SyncWriter::new(&self.config)?;
        let thread = std::thread::spawn(move || {
            loop {
                let command = receiver.recv()?;
                if is_shutdown.get().is_some() {
                    break;
                }
                match command {
                    Command::Message(topic, message, payload, resp_channel) => {
                        let _ = resp_channel.send(writer.send_message(
                            &topic,
                            &message,
                            &payload.iter().map(|e| e.as_slice()).collect::<Vec<_>>(),
                        ));
                    }
                    Command::Shutdown => {
                        break;
                    }
                }
            }
            Ok(())
        });
        self.thread = Some(thread);
        self.ops_queue = Some(sender);
        Ok(())
    }

    pub fn send_eos(&self, topic: &str) -> anyhow::Result<WriteOperationResult> {
        if !self.is_started() {
            anyhow::bail!("Writer is not started.");
        }
        let (resp_sender, resp_receiver) = crossbeam::channel::bounded(1);
        self.ops_queue.as_ref().unwrap().send(Command::Message(
            topic.to_string(),
            Box::new(Message::end_of_stream(EndOfStream::new(topic.to_string()))),
            vec![],
            resp_sender,
        ))?;

        Ok(WriteOperationResult(Some(resp_receiver)))
    }

    pub fn send_message(
        &self,
        topic: &str,
        message: &Message,
        payload: &[&[u8]],
    ) -> anyhow::Result<WriteOperationResult> {
        if !self.is_started() {
            anyhow::bail!("Writer is not started.");
        }
        let (resp_sender, resp_receiver) = crossbeam::channel::bounded(1);
        self.ops_queue.as_ref().unwrap().send(Command::Message(
            topic.to_string(),
            Box::new(message.clone()),
            payload.iter().map(|e| e.to_vec()).collect(),
            resp_sender,
        ))?;

        Ok(WriteOperationResult(Some(resp_receiver)))
    }
}

#[cfg(test)]
mod tests {
    use crate::message::Message;
    use crate::primitives::userdata::UserData;
    use crate::transport::zeromq::reader::ReaderResult;
    use crate::transport::zeromq::{
        NonblockingReader, NonblockingWriter, ReaderConfig, WriterConfig, WriterResult,
    };

    #[test]
    fn test_send_message_to_reader() -> anyhow::Result<()> {
        let mut reader = NonblockingReader::new(
            &ReaderConfig::new()
                .url("rep+bind:ipc:///tmp/test/req-rep-nowhere-writer")?
                .build()?,
        )?;
        reader.start(1)?;
        let mut writer = NonblockingWriter::new(
            &WriterConfig::new()
                .url("req+connect:ipc:///tmp/test/req-rep-nowhere-writer")?
                .with_send_timeout(100)?
                .with_receive_timeout(100)?
                .with_send_retries(1)?
                .with_receive_retries(1)?
                .build()?,
            1,
        )?;
        writer.start()?;
        let channel = writer.send_message(
            "test",
            &Message::user_data(UserData::new("test")),
            &[b"test"],
        )?;
        let result = channel.get()?;
        assert!(
            matches!(result, WriterResult::Ack {send_retries_spent,receive_retries_spent,time_spent: _}
                if send_retries_spent == 0 && receive_retries_spent == 0)
        );
        let reader_result = reader.receive()?;
        assert!(
            matches!(reader_result, ReaderResult::Message {message,topic,routing_id,data}
                if message.is_user_data() && topic == b"test" && routing_id.is_none() && data == vec![b"test".to_vec()]
            )
        );
        writer.shutdown()?;
        reader.shutdown()?;
        Ok(())
    }
}
