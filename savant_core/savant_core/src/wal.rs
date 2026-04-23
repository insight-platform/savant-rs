use std::fs::{self, File, OpenOptions};
use std::io::{self, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use log::warn;

const U32_BYTE_LEN: usize = 4;
const U64_BYTE_LEN: usize = 8;

const RECORD_HEADER_LEN: usize = U32_BYTE_LEN + U64_BYTE_LEN;
// write_segment_id, write_offset, read_segment_id, read_offset, space_stat, element_count
const META_SIZE: usize = U64_BYTE_LEN * 6;
const META_FILE_NAME: &str = "meta.bin";

const MAX_ALLOWED_ELEMENTS: u64 = u64::MAX - 100;

const SEGMENTS_DIR: &str = "segments";

#[derive(Debug)]
pub struct PersistentQueueWithCapacity {
    path: String,
    write_segment_id: u64,
    write_offset: u64,
    read_segment_id: u64,
    read_offset: u64,
    space_stat: u64,
    element_count: u64,
    segment_max_bytes: u64,
    max_elements: u64,
    high_watermark: usize,
}

fn encode_u64(v: u64) -> [u8; U64_BYTE_LEN] {
    v.to_le_bytes()
}

fn decode_u64(buf: &[u8]) -> u64 {
    let mut bytes = [0u8; U64_BYTE_LEN];
    bytes.copy_from_slice(buf);
    u64::from_le_bytes(bytes)
}

fn encode_u32(v: u32) -> [u8; U32_BYTE_LEN] {
    v.to_le_bytes()
}

fn decode_u32(buf: &[u8]) -> u32 {
    let mut bytes = [0u8; U32_BYTE_LEN];
    bytes.copy_from_slice(buf);
    u32::from_le_bytes(bytes)
}

fn hash(len: &[u8], payload: &[u8]) -> u32 {
    let mut h = crc32fast::Hasher::new();
    h.update(len);
    h.update(payload);
    h.finalize()
}

fn write_record(writer: &mut impl Write, payload: &[u8]) -> io::Result<()> {
    let len_bytes = encode_u64(payload.len() as u64);

    writer.write_all(&encode_u32(hash(&len_bytes, payload)))?;
    writer.write_all(&len_bytes)?;
    writer.write_all(payload)
}

fn read_record(reader: &mut impl Read) -> io::Result<Option<Vec<u8>>> {
    let mut header_buf = [0u8; RECORD_HEADER_LEN];
    if let Err(e) = reader.read_exact(&mut header_buf) {
        if e.kind() == ErrorKind::UnexpectedEof {
            return Ok(None);
        }
        return Err(e);
    }

    let stored_hash = decode_u32(&header_buf[0..4]);
    let len_bytes = &header_buf[4..12];
    let payload_len = decode_u64(len_bytes) as usize;
    let mut payload = vec![0u8; payload_len];
    reader.read_exact(&mut payload)?;

    if stored_hash != hash(len_bytes, &payload) {
        return Err(io::Error::new(ErrorKind::InvalidData, "invalid checksum"));
    }

    Ok(Some(payload))
}

fn read_meta(path: &Path) -> io::Result<(u64, u64, u64, u64, u64, u64)> {
    let meta_file = path.join(META_FILE_NAME);
    let metadata = match fs::read(meta_file) {
        Ok(metadata) => metadata,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok((0u64, 0, 0, 0, 0, 0)),
        Err(e) => return Err(e),
    };
    let write_segment_id = decode_u64(&metadata[0..8]);
    let write_offset = decode_u64(&metadata[8..16]);
    let read_segment_id = decode_u64(&metadata[16..24]);
    let read_offset = decode_u64(&metadata[24..32]);
    let space_stat = decode_u64(&metadata[32..40]);
    let element_count = decode_u64(&metadata[40..48]);

    Ok((
        write_segment_id,
        write_offset,
        read_segment_id,
        read_offset,
        space_stat,
        element_count,
    ))
}

fn segment_path(path: &str, segment_id: u64) -> PathBuf {
    Path::new(path)
        .join(SEGMENTS_DIR)
        .join(format!("{:020}", segment_id))
}

pub fn dir_size(path: &String) -> io::Result<usize> {
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

fn write_meta(
    path: &Path,
    write_segment_id: u64,
    write_offset: u64,
    read_segment_id: u64,
    read_offset: u64,
    space_stat: u64,
    element_count: u64,
) -> io::Result<()> {
    let mut buf = [0u8; META_SIZE];
    buf[0..8].copy_from_slice(&encode_u64(write_segment_id));
    buf[8..16].copy_from_slice(&encode_u64(write_offset));
    buf[16..24].copy_from_slice(&encode_u64(read_segment_id));
    buf[24..32].copy_from_slice(&encode_u64(read_offset));
    buf[32..40].copy_from_slice(&encode_u64(space_stat));
    buf[40..48].copy_from_slice(&encode_u64(element_count));

    let tmp_file = path.join("meta.tmp");
    fs::write(&tmp_file, buf)?;

    let meta_file = path.join(META_FILE_NAME);
    fs::rename(tmp_file, meta_file)
}

impl PersistentQueueWithCapacity {
    pub fn new(
        path: &str,
        max_elements: usize,
        high_watermark_pct: usize,
        segment_max_bytes: usize,
    ) -> Result<Self> {
        if max_elements > MAX_ALLOWED_ELEMENTS as usize {
            return Err(anyhow!(
                "max_elements can't be greater than {}",
                MAX_ALLOWED_ELEMENTS
            ));
        }

        let p: &Path = Path::new(path);
        fs::create_dir_all(p.join(SEGMENTS_DIR))?;
        let (
            write_segment_id,
            write_offset,
            read_segment_id,
            read_offset,
            space_stat,
            element_count,
        ) = read_meta(p)?;

        let last_segment = segment_path(path, write_segment_id);
        if last_segment.exists() {
            OpenOptions::new()
                .write(true)
                .open(last_segment)?
                .set_len(write_offset)?;
        } // need to scan records from the start and truncate at the first invalid checksum?

        Ok(Self {
            path: path.to_string(),
            write_segment_id,
            write_offset,
            read_segment_id,
            read_offset,
            space_stat,
            element_count,
            segment_max_bytes: segment_max_bytes as u64,
            max_elements: max_elements as u64,
            high_watermark: (high_watermark_pct as f64 / 100.0 * max_elements as f64).ceil()
                as usize,
        })
    }

    pub fn remove_db(path: &str) -> Result<()> {
        Ok(fs::remove_dir_all(path)?)
    }

    pub fn disk_size(&self) -> Result<usize> {
        Ok(dir_size(&self.path)?)
    }

    pub fn len(&self) -> usize {
        self.element_count as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn payload_size(&self) -> u64 {
        self.space_stat
    }

    pub fn max_elements(&self) -> usize {
        self.max_elements as usize
    }

    pub fn high_watermark(&self) -> usize {
        self.high_watermark
    }

    pub fn is_high_utilization(&self) -> bool {
        self.len() > self.high_watermark()
    }

    pub fn push(&mut self, values: &[&[u8]]) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }

        if self.len() + values.len() > self.max_elements() {
            return Err(anyhow!("Queue is full"));
        }

        let mut segment_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(segment_path(&self.path, self.write_segment_id))?;

        for value in values {
            if self.write_offset >= self.segment_max_bytes {
                self.write_segment_id += 1;
                self.write_offset = 0;
                segment_file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(segment_path(&self.path, self.write_segment_id))?;
            }
            write_record(&mut segment_file, *value)?;
            self.write_offset = segment_file.stream_position()?;
        } // todo add hot cash?

        self.space_stat += values.iter().map(|v| v.len() as u64).sum::<u64>();
        self.element_count += values.len() as u64;

        write_meta(
            Path::new(&self.path),
            self.write_segment_id,
            self.write_offset,
            self.read_segment_id,
            self.read_offset,
            self.space_stat,
            self.element_count,
        )?;

        Ok(())
    }

    pub fn pop(&mut self, max_elts: usize) -> Result<Vec<Vec<u8>>> {
        if self.is_empty() || max_elts == 0 {
            return Ok(vec![]);
        }

        let mut result: Vec<Vec<u8>> = Vec::with_capacity(max_elts);

        let mut data_file = File::open(segment_path(&self.path, self.read_segment_id))?;
        data_file.seek(SeekFrom::Start(self.read_offset))?;

        let mut read_count = 0;
        loop {
            if read_count >= max_elts {
                break;
            }
            match read_record(&mut data_file) {
                Ok(Some(v)) => {
                    result.push(v);
                    read_count += 1;
                }
                Ok(None) => {
                    if self.read_segment_id < self.write_segment_id {
                        // todo remove file here or implement cleanup as a separate task
                        self.read_segment_id += 1;
                        self.read_offset = 0;
                        data_file = File::open(segment_path(&self.path, self.read_segment_id))?;
                    } else {
                        break;
                    }
                }
                Err(e) if e.kind() == ErrorKind::InvalidData => {
                    warn!("corrupted record detected, skipping current segment");
                    if self.read_segment_id < self.write_segment_id {
                        // element_count and space_stat become approximate from this point (keep in segment header? or stats.bin file?)
                        self.read_segment_id += 1;
                        self.read_offset = 0;
                        data_file = File::open(segment_path(&self.path, self.read_segment_id))?;
                    } else {
                        break;
                    }
                }
                Err(e) => return Err(e.into()),
            }
        }

        if !result.is_empty() {
            self.read_offset = data_file.stream_position()?;
            self.space_stat -= result.iter().map(|v| v.len() as u64).sum::<u64>();
            self.element_count -= result.len() as u64;
        }

        write_meta(
            Path::new(&self.path),
            self.write_segment_id,
            self.write_offset,
            self.read_segment_id,
            self.read_offset,
            self.space_stat,
            self.element_count,
        )?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_queue(path: &str, max: usize, pct: usize) -> PersistentQueueWithCapacity {
        let _ = PersistentQueueWithCapacity::remove_db(path);
        PersistentQueueWithCapacity::new(path, max, pct, 10).unwrap()
    }

    #[test]
    fn test_normal_ops() {
        let path = "/tmp/wal_test1";
        let mut q = fresh_queue(path, 100, 90);

        q.push(&[&[1, 2, 3]]).unwrap();
        q.push(&[&[4, 5, 6]]).unwrap();
        q.push(&[&[7, 8, 9]]).unwrap();

        assert_eq!(q.len(), 3);
        assert_eq!(q.payload_size(), 9);

        assert_eq!(q.pop(1).unwrap(), vec![vec![1, 2, 3]]);
        assert_eq!(q.pop(1).unwrap(), vec![vec![4, 5, 6]]);
        assert_eq!(q.pop(1).unwrap(), vec![vec![7, 8, 9]]);

        assert!(q.is_empty());
        assert_eq!(q.payload_size(), 0);

        PersistentQueueWithCapacity::remove_db(path).unwrap();
    }

    #[test]
    fn test_capacity() {
        let path = "/tmp/wal_test2";
        let mut q = fresh_queue(path, 2, 90);

        q.push(&[&[1u8]]).unwrap();
        q.push(&[&[2u8]]).unwrap();
        assert!(q.push(&[&[3u8]]).is_err(), "should fail: queue full");

        PersistentQueueWithCapacity::remove_db(path).unwrap();
    }

    #[test]
    fn test_durability() {
        let path = "/tmp/wal_test3";
        let _ = PersistentQueueWithCapacity::remove_db(path);

        {
            let mut q = PersistentQueueWithCapacity::new(path, 100, 90, 10).unwrap();
            q.push(&[&[10u8, 20u8], &[30u8, 40u8]]).unwrap();
        }

        {
            let mut q = PersistentQueueWithCapacity::new(path, 100, 90, 10).unwrap();
            assert_eq!(q.len(), 2);
            assert_eq!(q.payload_size(), 4);
            let v = q.pop(2).unwrap();
            assert_eq!(v, vec![vec![10u8, 20u8], vec![30u8, 40u8]]);
        }

        PersistentQueueWithCapacity::remove_db(path).unwrap();
    }

    #[test]
    fn test_pop_partial() {
        let path = "/tmp/wal_test4";
        let mut q = fresh_queue(path, 100, 90);
        q.push(&[&[1u8], &[2u8]]).unwrap();

        let v = q.pop(10).unwrap();
        assert_eq!(v.len(), 2);
        assert!(q.is_empty());

        PersistentQueueWithCapacity::remove_db(path).unwrap();
    }

    #[test]
    fn test_high_watermark() {
        let path = "/tmp/wal_test5";
        let _ = PersistentQueueWithCapacity::remove_db(path);
        let mut q = PersistentQueueWithCapacity::new(path, 10, 50, 10).unwrap();

        for i in 0u8..5 {
            q.push(&[&[i]]).unwrap();
            assert!(!q.is_high_utilization());
        }
        q.push(&[&[99u8]]).unwrap();
        assert!(q.is_high_utilization());

        PersistentQueueWithCapacity::remove_db(path).unwrap();
    }

    // todo add more tests
}
