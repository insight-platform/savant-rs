use std::mem::size_of;
use std::path::Path;
use std::time::{Duration, SystemTime};
use std::{fs, io};

use anyhow::{bail, Result};
use bincode::config::{BigEndian, Configuration};
use bincode::{Decode, Encode};
use hashbrown::HashMap;
use log::info;
use md5::{Digest, Md5};
use rocksdb::{ColumnFamilyDescriptor, Direction, Options, ReadOptions, SliceTransform, DB};
use savant_core::message::{load_message, save_message, Message};
use uuid::Uuid;

use crate::get_keyframe_boundary;
use crate::store::JobOffset;

const CF_MESSAGE_DB: &str = "message_db";
const CF_KEYFRAME_DB: &str = "keyframes_db";
const CF_INDEX_DB: &str = "index_db";

pub(crate) fn dir_size(path: &Path) -> io::Result<usize> {
    fn dir_size(mut dir: fs::ReadDir) -> io::Result<usize> {
        dir.try_fold(0, |acc, file| {
            let file = file?;
            let size = match file.metadata()? {
                data if data.is_dir() => dir_size(fs::read_dir(file.path())?)?,
                data => data.len() as usize,
            };
            Ok(acc + size)
        })
    }
    dir_size(fs::read_dir(Path::new(path))?)
}

type MessageKeyIndex = usize;
#[derive(Encode, Decode, PartialEq, Debug)]
struct MessageKey {
    pub(crate) source_md5: [u8; 16],
    pub(crate) index: MessageKeyIndex,
}

#[derive(Encode, Decode, PartialEq, Debug)]
struct MessageValue {
    message: Vec<u8>,
    topic: Vec<u8>,
    data: Vec<Vec<u8>>,
}

#[derive(Encode, Decode, PartialEq, Debug)]
struct IndexKey {
    pub(crate) source_md5: [u8; 16],
}

#[derive(Encode, Decode, PartialEq, Debug)]
struct IndexValue {
    pub(crate) index: usize,
}

type KeyFrameUUID = u128;
#[derive(Encode, Decode, PartialEq, Debug)]
struct KeyframeKey {
    pub(crate) source_md5: [u8; 16],
    pub(crate) keyframe_uuid: KeyFrameUUID,
}

#[derive(Encode, Decode, PartialEq, Debug)]
struct KeyframeValue {
    pub(crate) index: usize,
    pub(crate) ts: u128,
}

pub struct RocksDbStore {
    db: DB,
    source_id_hashes: HashMap<String, [u8; 16]>,
    resident_index_values: HashMap<String, usize>,
    configuration: Configuration<BigEndian>,
}

impl RocksDbStore {
    fn get_source_hash(&mut self, source_id: &str) -> Result<[u8; 16]> {
        if let Some(hash) = self.source_id_hashes.get(source_id) {
            return Ok(*hash);
        }

        let mut hasher = Md5::new();
        hasher.update(source_id.as_bytes());
        let hash = hasher.finalize();
        let mut hash_bytes = [0; 16];
        hash_bytes.copy_from_slice(&hash[..]);
        self.source_id_hashes
            .insert(source_id.to_string(), hash_bytes);
        Ok(hash_bytes)
    }

    pub fn new(path: &Path, ttl: Duration, max_total_wal_size: u64) -> Result<Self> {
        let configuration = bincode::config::standard().with_big_endian();

        let mut cf_opts = Options::default();
        cf_opts.set_max_write_buffer_number(16);
        cf_opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(
            size_of::<MessageKey>() - size_of::<MessageKeyIndex>(),
        ));
        cf_opts.set_compaction_style(rocksdb::DBCompactionStyle::Universal);
        let cf_message = ColumnFamilyDescriptor::new(CF_MESSAGE_DB, cf_opts);

        let mut cf_opts = Options::default();
        cf_opts.set_max_write_buffer_number(16);
        cf_opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(
            size_of::<KeyframeKey>() - size_of::<KeyFrameUUID>(),
        ));
        cf_opts.set_compaction_style(rocksdb::DBCompactionStyle::Universal);
        let cf_keyframe = ColumnFamilyDescriptor::new(CF_KEYFRAME_DB, cf_opts);

        let mut cf_opts = Options::default();
        cf_opts.set_compaction_style(rocksdb::DBCompactionStyle::Universal);
        let cf_index = ColumnFamilyDescriptor::new(CF_INDEX_DB, cf_opts);

        let mut db_opts = Options::default();
        db_opts.create_missing_column_families(true);
        db_opts.set_max_total_wal_size(max_total_wal_size);
        db_opts.create_if_missing(true);

        info!("Opening RocksDB database at {:?} with TTL: {:?}", path, ttl);
        let db = DB::open_cf_descriptors_with_ttl(
            &db_opts,
            path,
            vec![cf_message, cf_keyframe, cf_index],
            ttl,
        )?;

        Ok(Self {
            db,
            source_id_hashes: HashMap::new(),
            resident_index_values: HashMap::new(),
            configuration,
        })
    }

    pub fn remove_db(path: &Path) -> Result<()> {
        Ok(DB::destroy(&Options::default(), path)?)
    }

    pub fn optimize(&self) -> Result<()> {
        self.db.compact_range(None::<&[u8]>, None::<&[u8]>);
        Ok(())
    }

    pub fn disk_size(&self, path: &Path) -> Result<usize> {
        Ok(dir_size(path)?)
    }
}

impl super::Store for RocksDbStore {
    fn current_index_value(&mut self, source_id: &str) -> Result<usize> {
        if let Some(index) = self.resident_index_values.get(source_id) {
            return Ok(*index);
        }

        let key = IndexKey {
            source_md5: self.get_source_hash(source_id)?,
        };

        let key_bytes = bincode::encode_to_vec(key, self.configuration)?;
        let cf = self.db.cf_handle(CF_INDEX_DB).unwrap();
        let value_bytes = self.db.get_cf(cf, key_bytes)?;
        let value = if let Some(value_bytes) = value_bytes {
            let value: IndexValue = bincode::decode_from_slice(&value_bytes, self.configuration)?.0;
            value.index
        } else {
            0
        };

        self.resident_index_values
            .insert(source_id.to_string(), value);
        Ok(value)
    }

    async fn add_message(
        &mut self,
        message: &Message,
        topic: &[u8],
        data: &[Vec<u8>],
    ) -> Result<usize> {
        let mut frame_opt = None;
        let source_id = if message.is_video_frame() {
            let frame = message.as_video_frame().unwrap();
            let source_id = frame.get_source_id();
            if matches!(frame.get_keyframe(), Some(true)) {
                frame_opt = Some(frame);
            }
            source_id
        } else if message.is_end_of_stream() {
            let eos = message.as_end_of_stream().unwrap();
            eos.source_id.clone()
        } else if message.is_user_data() {
            let user_data = message.as_user_data().unwrap();
            user_data.get_source_id().to_string()
        } else {
            bail!("Unsupported message type. Only video frames, end of stream, and user data are supported.");
        };
        let source_hash = self.get_source_hash(&source_id)?;
        let index = self.current_index_value(&source_id)?;
        let mut batch = rocksdb::WriteBatch::default();

        let message_key = MessageKey {
            source_md5: source_hash,
            index,
        };

        let message_value = MessageValue {
            message: save_message(message)?,
            topic: topic.to_vec(),
            data: data.to_vec(),
        };

        let message_key_bytes = bincode::encode_to_vec(message_key, self.configuration)?;
        let message_value_bytes = bincode::encode_to_vec(message_value, self.configuration)?;
        let cf = self
            .db
            .cf_handle(CF_MESSAGE_DB)
            .expect("CF_MESSAGE_DB not found");
        batch.put_cf(cf, &message_key_bytes, &message_value_bytes);

        if let Some(f) = frame_opt {
            let key = KeyframeKey {
                source_md5: source_hash,
                keyframe_uuid: f.get_uuid_u128(),
            };
            let key_bytes = bincode::encode_to_vec(key, self.configuration)?;
            let ts = f.get_creation_timestamp_ns();
            let value = KeyframeValue { index, ts };
            let value_bytes = bincode::encode_to_vec(value, self.configuration)?;
            let cf = self
                .db
                .cf_handle(CF_KEYFRAME_DB)
                .expect("CF_KEYFRAME_DB not found");
            batch.put_cf(cf, &key_bytes, &value_bytes);
        }

        let index_key = IndexKey {
            source_md5: source_hash,
        };

        let index = index + 1;
        let index_value = IndexValue { index };

        let index_key_bytes = bincode::encode_to_vec(index_key, self.configuration)?;
        let index_value_bytes = bincode::encode_to_vec(index_value, self.configuration)?;
        let cf = self
            .db
            .cf_handle(CF_INDEX_DB)
            .expect("CF_INDEX_DB not found");
        batch.put_cf(cf, &index_key_bytes, &index_value_bytes);

        self.db.write(batch)?;
        self.resident_index_values.insert(source_id, index);

        Ok(index - 1)
    }

    async fn get_message(
        &mut self,
        source_id: &str,
        index: usize,
    ) -> Result<Option<(Message, Vec<u8>, Vec<Vec<u8>>)>> {
        let current_index = self.current_index_value(source_id)?;
        if index > current_index {
            return Ok(None);
        }
        let source_hash = self.get_source_hash(source_id)?;
        let key = MessageKey {
            source_md5: source_hash,
            index,
        };
        let key_bytes = bincode::encode_to_vec(key, self.configuration)?;
        let cf = self
            .db
            .cf_handle(CF_MESSAGE_DB)
            .expect("CF_MESSAGE_DB not found");
        let value_bytes = self.db.get_cf(cf, key_bytes)?;
        if let Some(value_bytes) = value_bytes {
            let message_value: MessageValue =
                bincode::decode_from_slice(&value_bytes, self.configuration)?.0;
            let message = load_message(&message_value.message);
            Ok(Some((message, message_value.topic, message_value.data)))
        } else {
            Ok(None)
        }
    }

    async fn get_first(
        &mut self,
        source_id: &str,
        keyframe_uuid: Uuid,
        before: &JobOffset,
    ) -> Result<Option<usize>> {
        let source_hash = self.get_source_hash(source_id)?;
        let key = KeyframeKey {
            source_md5: source_hash,
            keyframe_uuid: keyframe_uuid.as_u128(),
        };
        let key_bytes = bincode::encode_to_vec(&key, self.configuration)?;
        let cf = self
            .db
            .cf_handle(CF_KEYFRAME_DB)
            .expect("CF_KEYFRAME_DB not found");
        let value_bytes = self.db.get_cf(cf, key_bytes)?;

        if let Some(value_bytes) = value_bytes {
            let value: KeyframeValue =
                bincode::decode_from_slice(&value_bytes, self.configuration)?.0;

            let req_index = value.index;
            let req_pts = value.ts;

            let cf = self
                .db
                .cf_handle(CF_KEYFRAME_DB)
                .expect("CF_KEYFRAME_DB not found");

            let key_bytes = bincode::encode_to_vec(&key, self.configuration)?;

            let mut opts = ReadOptions::default();
            opts.set_prefix_same_as_start(true);
            let iter = self.db.iterator_cf_opt(
                cf,
                opts,
                rocksdb::IteratorMode::From(&key_bytes, Direction::Reverse),
            );

            let mut current_index = req_index;
            let mut block_counter = 0;
            for res in iter {
                let (_, v) = res?;
                let value: KeyframeValue = bincode::decode_from_slice(&v, self.configuration)?.0;
                current_index = value.index;
                let current_pts = value.ts;
                match before {
                    JobOffset::Blocks(blocks_before) => {
                        if block_counter >= *blocks_before {
                            break;
                        }
                        block_counter += 1;
                    }
                    JobOffset::Seconds(seconds_before) => {
                        if req_pts - current_pts
                            > Duration::from_secs_f64(*seconds_before).as_nanos()
                        {
                            break;
                        }
                    }
                }
            }
            Ok(Some(current_index))
        } else {
            Ok(None)
        }
    }

    async fn find_keyframes(
        &mut self,
        source_id: &str,
        from: Option<u64>,
        to: Option<u64>,
        limit: usize,
    ) -> Result<Vec<Uuid>> {
        let from_uuid = get_keyframe_boundary(from.map(|f| f.saturating_sub(1)), 0).as_u128();
        let to_uuid = get_keyframe_boundary(to.map(|t| t.saturating_add(1)), {
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)?
                .as_secs()
                + 1
        })
        .as_u128();
        let source_hash = self.get_source_hash(source_id)?;

        let cf = self
            .db
            .cf_handle(CF_KEYFRAME_DB)
            .expect("CF_KEYFRAME_DB not found");

        let key = KeyframeKey {
            source_md5: source_hash,
            keyframe_uuid: from_uuid,
        };

        let key_bytes = bincode::encode_to_vec(key, self.configuration)?;

        let mut opts = ReadOptions::default();
        opts.set_prefix_same_as_start(true);
        let iter = self.db.iterator_cf_opt(
            cf,
            opts,
            rocksdb::IteratorMode::From(&key_bytes, Direction::Forward),
        );
        let mut keyframes = vec![];
        for k in iter {
            let (k, _) = k?;
            let key: KeyframeKey = bincode::decode_from_slice(&k, self.configuration)?.0;
            if key.keyframe_uuid > to_uuid {
                break;
            }
            keyframes.push(Uuid::from_u128(key.keyframe_uuid));
            if keyframes.len() >= limit {
                break;
            }
        }
        Ok(keyframes)
    }
}

#[cfg(test)]
mod tests {
    use savant_core::test::gen_frame;
    use std::ops::Div;
    use tokio_timerfd::sleep;

    use crate::store::{gen_properly_filled_frame, Store};

    use super::*;

    #[tokio::test]
    async fn test_rocksdb_init() -> Result<()> {
        let dir = tempfile::TempDir::new()?;
        let path = dir.path();
        let mut db = RocksDbStore::new(path, Duration::from_secs(60), 1024 * 1024 * 1024)?;

        let source_id1 = "test_source_id-1";
        let source_id2 = "test_source_id-2";
        let source_hash1 = db.get_source_hash(source_id1)?;
        let source_hash2 = db.get_source_hash(source_id2)?;
        assert_ne!(source_hash1, source_hash2);
        let source_hash1_1 = db.get_source_hash(source_id1)?;
        assert_eq!(source_hash1, source_hash1_1);

        let index = db.current_index_value(source_id1)?;
        assert_eq!(index, 0);
        let index = db.current_index_value(source_id1)?;
        assert_eq!(index, 0);

        let disk_size = db.disk_size(path)?;
        assert!(disk_size > 0);

        let _ = RocksDbStore::remove_db(path);
        Ok(())
    }

    #[tokio::test]
    async fn test_load_message() -> Result<()> {
        let dir = tempfile::TempDir::new()?;
        let path = dir.path();
        let frame = gen_frame();
        let source_id = frame.get_source_id();
        {
            let mut db = RocksDbStore::new(path, Duration::from_secs(60), 1024 * 1024 * 1024)?;
            let m = frame.to_message();
            let id = db.add_message(&m, &[], &[]).await?;
            let (m2, _, _) = db.get_message(&source_id, id).await?.unwrap();
            assert_eq!(
                m.as_video_frame().unwrap().get_uuid_u128(),
                m2.as_video_frame().unwrap().get_uuid_u128()
            );
        }
        {
            let mut db = RocksDbStore::new(path, Duration::from_secs(60), 1024 * 1024 * 1024)?;
            let (_, _, _) = db.get_message(&source_id, 0).await?.unwrap();
            assert_eq!(db.current_index_value(&source_id)?, 1);
            let m = db.get_message(&source_id, 1).await?;
            assert!(m.is_none());
        }
        let _ = RocksDbStore::remove_db(path);
        Ok(())
    }

    #[tokio::test]
    async fn test_find_starting_block_with_offset_in_blocks() -> Result<()> {
        const DELTA_FRAMES: usize = 10;
        let dir = tempfile::TempDir::new()?;
        let path = dir.path();
        let mut db = RocksDbStore::new(path, Duration::from_secs(60), 1024 * 1024 * 1024)?;

        // add to stream 1 (0)
        let f = gen_properly_filled_frame(true);
        let source_id = f.get_source_id();
        let first_indx = db.add_message(&f.to_message(), &[], &[]).await?;
        for _ in 0..DELTA_FRAMES {
            let f = gen_properly_filled_frame(false);
            db.add_message(&f.to_message(), &[], &[]).await?;
        }

        // add to stream 1 (11)
        let f = gen_properly_filled_frame(true);
        let second_indx = db.add_message(&f.to_message(), &[], &[]).await?;
        for _ in 0..DELTA_FRAMES {
            let f = gen_properly_filled_frame(false);
            db.add_message(&f.to_message(), &[], &[]).await?;
        }

        // add to stream 2 (22)
        let mut other_source_frame = gen_properly_filled_frame(true);
        let other_source_id = "other_source_id";
        other_source_frame.set_source_id(other_source_id);
        db.add_message(&other_source_frame.to_message(), &[], &[])
            .await?;
        for _ in 0..DELTA_FRAMES {
            let mut f = gen_properly_filled_frame(false);
            f.set_source_id(other_source_id);
            db.add_message(&f.to_message(), &[], &[]).await?;
        }

        // add to stream 1 (33)
        let f = gen_properly_filled_frame(true);
        let uuid_f3 = f.get_uuid();
        let last_indx = db.add_message(&f.to_message(), &[], &[]).await?;
        for _ in 0..DELTA_FRAMES {
            let f = gen_properly_filled_frame(false);
            db.add_message(&f.to_message(), &[], &[]).await?;
        }

        let first = db
            .get_first(&source_id, uuid_f3, &JobOffset::Blocks(0))
            .await?
            .unwrap();
        assert_eq!(first, last_indx);

        let first = db
            .get_first(&source_id, uuid_f3, &JobOffset::Blocks(1))
            .await?
            .unwrap();
        assert_eq!(first, second_indx);

        let first = db
            .get_first(&source_id, uuid_f3, &JobOffset::Blocks(2))
            .await?
            .unwrap();
        assert_eq!(first, first_indx);

        Ok(())
    }

    #[tokio::test]
    async fn test_find_first_block_in_duration() -> Result<()> {
        let dir = tempfile::TempDir::new()?;
        let path = dir.path();
        let mut db = RocksDbStore::new(path, Duration::from_secs(60), 1024 * 1024 * 1024)?;
        let f = gen_properly_filled_frame(true);
        let source_id = f.get_source_id();
        db.add_message(&f.to_message(), &[], &[]).await?;
        sleep(Duration::from_millis(10)).await?;
        let f = gen_properly_filled_frame(true);
        db.add_message(&f.to_message(), &[], &[]).await?;
        sleep(Duration::from_millis(10)).await?;
        let f = gen_properly_filled_frame(true);
        let uuid_f3 = f.get_uuid();
        db.add_message(&f.to_message(), &[], &[]).await?;

        let first = db
            .get_first(&source_id, uuid_f3, &JobOffset::Seconds(0.005))
            .await?
            .unwrap();
        assert_eq!(first, 1);

        let first = db
            .get_first(&source_id, uuid_f3, &JobOffset::Seconds(0.015))
            .await?
            .unwrap();
        assert_eq!(first, 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_keyframes() -> Result<()> {
        let dir = tempfile::TempDir::new()?;
        let path = dir.path();
        let mut db = RocksDbStore::new(path, Duration::from_secs(60), 1024 * 1024 * 1024)?;
        let mut keyframes = vec![];
        const N: usize = 20;
        let source = gen_properly_filled_frame(true).get_source_id();
        for i in 0..N {
            let f = gen_properly_filled_frame(true);
            db.add_message(&f.to_message(), &[], &[]).await?;
            keyframes.push(f.get_uuid());
            if i + 1 == N.div(2) {
                sleep(Duration::from_secs(2)).await?;
            }
        }
        let first = keyframes[1].get_timestamp().unwrap().to_unix().0;
        let last = keyframes[N - 2].get_timestamp().unwrap().to_unix().0;
        let res_keyframes = db
            .find_keyframes(&source, Some(first), Some(last), N)
            .await?;
        assert_eq!(res_keyframes.len(), N);

        let first = keyframes[N.div(2) + 1].get_timestamp().unwrap().to_unix().0;

        let res_keyframes = db.find_keyframes(&source, Some(first), None, N).await?;
        assert_eq!(res_keyframes.len(), N.div_ceil(2));

        let last = keyframes[N.div(2) - 1].get_timestamp().unwrap().to_unix().0;

        let res_keyframes = db.find_keyframes(&source, None, Some(last), N).await?;
        assert_eq!(res_keyframes.len(), N.div_ceil(2));

        let res_keyframes = db.find_keyframes(&source, None, None, 2).await?;
        assert_eq!(res_keyframes.len(), 2);

        Ok(())
    }

    #[tokio::test]
    async fn test_future_message() -> Result<()> {
        let dir = tempfile::TempDir::new()?;
        let path = dir.path();
        let mut db = RocksDbStore::new(path, Duration::from_secs(60), 1024 * 1024 * 1024)?;
        let source_id = "test_source_id";
        let message = gen_properly_filled_frame(true).to_message();
        db.add_message(&message, &[], &[]).await?;
        let message = gen_properly_filled_frame(true).to_message();
        db.add_message(&message, &[], &[]).await?;

        let now = std::time::Instant::now();
        let res = db.get_message(source_id, 10).await?;
        let elapsed = now.elapsed();
        assert!(res.is_none());
        // Should return very quickly, under 1ms
        assert!(elapsed.as_millis() < 1);
        Ok(())
    }
}
