use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use savant_core::message::Message;
use savant_core::transport::zeromq::{
    NonBlockingReader, NonBlockingWriter, ReaderResult, WriterResult,
};
use tokio::sync::Mutex;

use crate::store::rocksdb::RocksDbStore;
use crate::store::{Store, SyncRocksDbStore};
use savant_services_common::topic_to_string;

#[derive(Debug)]
struct StreamStats {
    packet_counter: u64,
    byte_counter: u64,
}

struct StreamProcessor<T: Store> {
    db: Arc<Mutex<T>>,
    input: NonBlockingReader,
    output: Option<NonBlockingWriter>,
    stats: StreamStats,
    last_stats: Instant,
    stats_period: Duration,
    send_metadata_only: bool,
    stop_flag: bool,
}

impl<T> StreamProcessor<T>
where
    T: Store,
{
    pub fn new(
        db: Arc<Mutex<T>>,
        input: NonBlockingReader,
        output: Option<NonBlockingWriter>,
        stats_period: Duration,
        send_metadata_only: bool,
    ) -> Self {
        Self {
            db,
            input,
            output,
            stats: StreamStats {
                packet_counter: 0,
                byte_counter: 0,
            },
            stats_period,
            last_stats: Instant::now(),
            send_metadata_only,
            stop_flag: false,
        }
    }

    async fn receive_message(&mut self) -> Result<ReaderResult> {
        loop {
            if Instant::now() - self.last_stats > self.stats_period {
                log::info!(
                    target: "replay::db::stream_processor::receive_message",
                    "Packets: {}, Bytes: {}",
                    self.stats.packet_counter,
                    self.stats.byte_counter
                );
                self.last_stats = Instant::now();
            }
            let message = self.input.try_receive();
            if message.is_none() {
                log::trace!(
                    target: "replay::db::stream_processor::receive_message",
                    "No message received, sleeping for 1ms."
                );
                tokio_timerfd::sleep(Duration::from_millis(1)).await?;
                continue;
            }
            log::debug!(
                target: "replay::db::stream_processor::receive_message",
                "Received message."
            );
            return message.unwrap();
        }
    }

    async fn send_message(&mut self, topic: &str, message: &Message, data: &[&[u8]]) -> Result<()> {
        if self.output.is_none() {
            log::debug!(target: "replay::db::stream_processor::send_message",
                "No output writer, skipping."
            );
            return Ok(());
        }
        let output = self.output.as_mut().unwrap();
        loop {
            let res = output.send_message(topic, message, data)?;
            loop {
                let send_res = res.try_get()?;
                if send_res.is_none() {
                    tokio_timerfd::sleep(Duration::from_micros(50)).await?;
                    continue;
                }
                let send_res = send_res.unwrap()?;
                match send_res {
                    WriterResult::SendTimeout => {
                        log::warn!(
                            target: "replay::db::stream_processor::send_message",
                            "Send timeout, retrying sending {}.", message.meta().seq_id);
                        break;
                    }
                    WriterResult::AckTimeout(_) => {
                        log::warn!(
                            target: "replay::db::stream_processor::send_message",
                            "Ack timeout, retrying sending {:?}.", message.meta().seq_id);
                        break;
                    }
                    WriterResult::Ack { .. } => {
                        log::debug!(
                            target: "replay::db::stream_processor::send_message",
                            "Message ack received for {:?}.", message
                        );
                        return Ok(());
                    }
                    WriterResult::Success { .. } => {
                        log::debug!(
                            target: "replay::db::stream_processor::send_message",
                            "Message {:?} sent.", message
                        );
                        return Ok(());
                    }
                }
            }
        }
    }

    pub async fn run_once(&mut self) -> Result<()> {
        let message = self.receive_message().await;
        match message {
            Ok(m) => match m {
                ReaderResult::Blacklisted(topic) => {
                    log::debug!(
                        target: "replay::db::stream_processor::run_once",
                        "Received blacklisted message: {}", topic_to_string(&topic));
                }
                ReaderResult::Message {
                    message,
                    topic,
                    routing_id,
                    data,
                } => {
                    self.stats.packet_counter += 1;
                    self.stats.byte_counter += data.iter().map(|v| v.len() as u64).sum::<u64>();
                    log::debug!(
                        target: "replay::db::stream_processor::run_once",
                        "Received message: topic: {}, routing_id: {}, message: {:?}",
                        topic_to_string(&topic),
                        topic_to_string(routing_id.as_ref().unwrap_or(&Vec::new())),
                        message
                    );
                    if message.is_video_frame()
                        || message.is_user_data()
                        || message.is_end_of_stream()
                    {
                        log::debug!(
                            target: "replay::db::stream_processor::run_once",
                            "Adding message {:?} to database.", &message
                        );
                        self.db
                            .lock()
                            .await
                            .add_message(&message, &topic, &data)
                            .await?;
                    }
                    let data_slice = if self.send_metadata_only {
                        vec![]
                    } else {
                        data.iter().map(|v| v.as_slice()).collect::<Vec<&[u8]>>()
                    };

                    self.send_message(std::str::from_utf8(&topic)?, &message, &data_slice)
                        .await?;
                }
                ReaderResult::Timeout => {
                    log::debug!(
                        target: "replay::db::stream_processor::run_once",
                        "Timeout receiving message, waiting for next message.");
                }
                ReaderResult::PrefixMismatch { topic, routing_id } => {
                    log::warn!(
                        target: "replay::db::stream_processor::run_once",
                        "Received message with mismatched prefix: topic: {:?}, routing_id: {:?}",
                        topic_to_string(&topic),
                        topic_to_string(routing_id.as_ref().unwrap_or(&Vec::new()))
                    );
                }
                ReaderResult::RoutingIdMismatch { topic, routing_id } => {
                    log::warn!(
                        target: "replay::db::stream_processor::run_once",
                        "Received message with mismatched routing_id: topic: {:?}, routing_id: {:?}",
                        topic_to_string(&topic),
                        topic_to_string(routing_id.as_ref().unwrap_or(&Vec::new()))
                    );
                }
                ReaderResult::TooShort(m) => {
                    log::warn!(
                        target: "replay::db::stream_processor::run_once",
                        "Received message that was too short: {:?}", m);
                }
                ReaderResult::MessageVersionMismatch {
                    topic,
                    routing_id,
                    sender_version,
                    expected_version,
                } => {
                    log::warn!(
                        target: "replay::db::stream_processor::run_once",
                        "Received message with mismatched version: topic: {:?}, routing_id: {:?}, sender_version: {:?}, expected_version: {:?}",
                        topic_to_string(&topic),
                        topic_to_string(routing_id.as_ref().unwrap_or(&Vec::new())),
                        sender_version,
                        expected_version
                    );
                }
            },
            Err(e) => {
                bail!("Error receiving message: {:?}", e);
            }
        }
        Ok(())
    }

    pub async fn run(&mut self) -> Result<()> {
        loop {
            if self.stop_flag {
                return Ok(());
            }
            self.run_once().await?
        }
    }

    pub fn stop(&mut self) {
        self.stop_flag = true;
    }
}

pub struct RocksDbStreamProcessor(StreamProcessor<RocksDbStore>);

impl RocksDbStreamProcessor {
    pub fn new(
        db: SyncRocksDbStore,
        input: NonBlockingReader,
        output: Option<NonBlockingWriter>,
        stats_period: Duration,
        send_metadata_only: bool,
    ) -> Self {
        Self(StreamProcessor::new(
            db,
            input,
            output,
            stats_period,
            send_metadata_only,
        ))
    }

    pub async fn run_once(&mut self) -> Result<()> {
        self.0.run_once().await
    }

    pub async fn run(&mut self) -> Result<()> {
        self.0.run().await
    }

    pub fn stop(&mut self) {
        self.0.stop();
    }

    pub fn store(&self) -> Arc<Mutex<RocksDbStore>> {
        self.0.db.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use anyhow::Result;
    use savant_core::transport::zeromq::{
        NonBlockingReader, NonBlockingWriter, ReaderConfig, ReaderResult, WriterConfig,
    };
    use tokio::sync::Mutex;

    use crate::store::rocksdb::RocksDbStore;
    use crate::store::{gen_properly_filled_frame, Store};
    use crate::stream_processor::StreamProcessor;

    #[tokio::test]
    async fn test_stream_processor() -> Result<()> {
        let dir = tempfile::TempDir::new()?;
        let path = dir.path();
        let db = RocksDbStore::new(path, Duration::from_secs(60))?;

        let mut in_reader = NonBlockingReader::new(
            &ReaderConfig::new()
                .url(&format!(
                    "router+bind:ipc://{}/in",
                    path.to_str()
                        .expect("Cannot convert path to a valid string")
                ))?
                .with_fix_ipc_permissions(Some(0o777))?
                .build()?,
            100,
        )?;
        in_reader.start()?;
        tokio_timerfd::sleep(Duration::from_millis(100)).await?;

        let mut in_writer = NonBlockingWriter::new(
            &WriterConfig::new()
                .url(&format!(
                    "dealer+connect:ipc://{}/in",
                    path.to_str()
                        .expect("Cannot convert path to a valid string")
                ))?
                .build()?,
            100,
        )?;
        in_writer.start()?;
        tokio_timerfd::sleep(Duration::from_millis(100)).await?;

        let mut out_reader = NonBlockingReader::new(
            &ReaderConfig::new()
                .url(&format!(
                    "router+bind:ipc://{}/out",
                    path.to_str()
                        .expect("Cannot convert path to a valid string")
                ))?
                .with_fix_ipc_permissions(Some(0o777))?
                .build()?,
            100,
        )?;
        out_reader.start()?;
        tokio_timerfd::sleep(Duration::from_millis(100)).await?;

        let mut out_writer = NonBlockingWriter::new(
            &WriterConfig::new()
                .url(&format!(
                    "dealer+connect:ipc://{}/out",
                    path.to_str()
                        .expect("Cannot convert path to a valid string")
                ))?
                .build()?,
            100,
        )?;
        out_writer.start()?;
        tokio_timerfd::sleep(Duration::from_millis(100)).await?;

        let db = Arc::new(Mutex::new(db));
        let mut processor = StreamProcessor::new(
            db.clone(),
            in_reader,
            Some(out_writer),
            Duration::from_secs(30),
            false,
        );

        let f = gen_properly_filled_frame(true);
        let uuid = f.get_uuid_u128();
        let m1 = f.to_message();
        in_writer.send_message("test", &m1, &[&[0x01]])?;
        processor.run_once().await?;
        let res = out_reader.receive()?;
        let (m2, _, _) = db.lock().await.get_message("test", 0).await?.unwrap();
        assert_eq!(uuid, m2.as_video_frame().unwrap().get_uuid_u128());
        match res {
            ReaderResult::Blacklisted(topic) => {
                panic!("Blacklisted message: {:?}", topic);
            }
            ReaderResult::Message {
                message,
                topic,
                routing_id: _,
                data,
            } => {
                assert_eq!(message.as_video_frame().unwrap().get_uuid_u128(), uuid);
                assert_eq!(topic, b"test");
                assert_eq!(data, vec![vec![0x01]]);
            }
            ReaderResult::Timeout => {
                panic!("Timeout");
            }
            ReaderResult::PrefixMismatch { .. } => {
                panic!("Prefix mismatch");
            }
            ReaderResult::RoutingIdMismatch { .. } => {
                panic!("Routing ID mismatch");
            }
            ReaderResult::TooShort(_) => {
                panic!("Too short");
            }
            ReaderResult::MessageVersionMismatch {
                topic,
                routing_id,
                sender_version,
                expected_version,
            } => panic!("Message version mismatch: topic: {:?}, routing_id: {:?}, sender_version: {:?}, expected_version: {:?}", topic, routing_id, sender_version, expected_version),
        }
        Ok(())
    }
}
