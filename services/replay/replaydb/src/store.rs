pub mod rocksdb;

use anyhow::Result;
use savant_core::message::Message;
use savant_core::primitives::frame::VideoFrameProxy;
use savant_core::test::gen_frame;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::Mutex;
use uuid::Uuid;

pub fn gen_properly_filled_frame(kf: bool) -> VideoFrameProxy {
    let mut f = gen_frame();
    let (tbn, tbd) = (1, 1_000_000);
    let now_nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let now = now_nanos as f64 / 1e9f64;
    let pts = (now * tbd as f64 / tbn as f64) as i64;
    f.set_pts(pts);
    f.set_creation_timestamp_ns(now_nanos);
    f.set_time_base((tbn, tbd));
    f.set_keyframe(Some(kf));
    f
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum JobOffset {
    #[serde(rename = "blocks")]
    Blocks(usize),
    #[serde(rename = "seconds")]
    Seconds(f64),
}

pub type SyncRocksDbStore = Arc<Mutex<rocksdb::RocksDbStore>>;

pub(crate) trait Store {
    fn current_index_value(&mut self, source_id: &str) -> Result<usize>;

    async fn add_message(
        &mut self,
        message: &Message,
        topic: &[u8],
        data: &[Vec<u8>],
    ) -> Result<usize>;

    #[allow(clippy::type_complexity)]
    async fn get_message(
        &mut self,
        source_id: &str,
        position: usize,
    ) -> Result<Option<(Message, Vec<u8>, Vec<Vec<u8>>)>>;

    async fn get_first(
        &mut self,
        source_id: &str,
        keyframe_uuid: Uuid,
        before: &JobOffset,
    ) -> Result<Option<usize>>;

    async fn find_keyframes(
        &mut self,
        source_id: &str,
        from: Option<u64>,
        to: Option<u64>,
        limit: usize,
    ) -> Result<Vec<Uuid>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{best_ts, get_keyframe_boundary};
    use savant_core::primitives::eos::EndOfStream;

    struct SampleStore {
        keyframes: Vec<(Uuid, usize)>,
        messages: Vec<Message>,
    }

    impl Store for SampleStore {
        fn current_index_value(&mut self, _source_id: &str) -> Result<usize> {
            Ok(self.messages.len())
        }

        async fn add_message(
            &mut self,
            message: &Message,
            _topic: &[u8],
            _data: &[Vec<u8>],
        ) -> Result<usize> {
            let current_len = self.messages.len();
            if message.is_video_frame() {
                let f = message.as_video_frame().unwrap();
                if let Some(true) = f.get_keyframe() {
                    self.keyframes.push((f.get_uuid(), current_len));
                }
            }
            self.messages.push(message.clone());
            Ok(current_len)
        }

        async fn get_message(
            &mut self,
            _: &str,
            id: usize,
        ) -> Result<Option<(Message, Vec<u8>, Vec<Vec<u8>>)>> {
            Ok(Some((self.messages[id].clone(), vec![], vec![])))
        }

        async fn get_first(
            &mut self,
            _: &str,
            keyframe_uuid: Uuid,
            before: &JobOffset,
        ) -> Result<Option<usize>> {
            let idx = self.keyframes.iter().position(|(u, _)| u == &keyframe_uuid);
            if idx.is_none() {
                return Ok(None);
            }
            let idx = idx.unwrap();

            Ok(Some(match before {
                JobOffset::Blocks(blocks_before) => {
                    if idx < *blocks_before {
                        self.keyframes[0].1
                    } else {
                        self.keyframes[idx - blocks_before].1
                    }
                }
                JobOffset::Seconds(seconds_before) => {
                    let frame = self.messages[self.keyframes[idx].1]
                        .as_video_frame()
                        .unwrap();
                    let current_pts = best_ts(&frame) as u64;
                    let time_base = frame.get_time_base();
                    let before_scaled =
                        (seconds_before * time_base.1 as f64 / time_base.0 as f64) as u64;
                    let mut i = self.keyframes[idx].1 - 1;
                    while i > 0 {
                        if self.messages[i].is_video_frame() {
                            let f = self.messages[i].as_video_frame().unwrap();
                            if let Some(true) = f.get_keyframe() {
                                let pts = best_ts(&f) as u64;
                                if current_pts - pts > before_scaled {
                                    break;
                                }
                            }
                        }
                        i -= 1;
                    }
                    i
                }
            }))
        }

        async fn find_keyframes(
            &mut self,
            _source_id: &str,
            from: Option<u64>,
            to: Option<u64>,
            _limit: usize,
        ) -> Result<Vec<Uuid>> {
            let from_uuid = get_keyframe_boundary(from, 0);
            let to_uuid = get_keyframe_boundary(to, u64::MAX);
            Ok(self
                .keyframes
                .iter()
                .filter(|(u, _)| from_uuid <= *u && *u <= to_uuid)
                .map(|(u, _)| *u)
                .collect())
        }
    }

    #[tokio::test]
    async fn test_sample_store() -> Result<()> {
        let mut store = SampleStore {
            keyframes: Vec::new(),
            messages: Vec::new(),
        };

        let mut f = gen_frame();
        f.set_keyframe(Some(true));
        f.set_time_base((1, 1));
        f.set_pts(0);
        store.add_message(&f.to_message(), &[], &[]).await?;
        store
            .add_message(
                &Message::end_of_stream(EndOfStream::new(String::from(""))),
                &[],
                &[],
            )
            .await?;
        let mut f = gen_frame();
        f.set_keyframe(Some(false));
        f.set_time_base((1, 1));
        f.set_pts(1);
        store.add_message(&f.to_message(), &[], &[]).await?;

        let mut f = gen_frame();
        f.set_keyframe(Some(false));
        f.set_time_base((1, 1));
        f.set_pts(2);
        store.add_message(&f.to_message(), &[], &[]).await?;

        let mut f = gen_frame();
        f.set_keyframe(Some(true));
        f.set_time_base((1, 1));
        f.set_pts(3);
        store.add_message(&f.to_message(), &[], &[]).await?;
        store
            .add_message(
                &Message::end_of_stream(EndOfStream::new(String::from(""))),
                &[],
                &[],
            )
            .await?;
        let mut f = gen_frame();
        f.set_keyframe(Some(false));
        f.set_time_base((1, 1));
        f.set_pts(4);
        store.add_message(&f.to_message(), &[], &[]).await?;

        let mut f = gen_frame();
        let u = f.get_uuid();
        f.set_keyframe(Some(true));
        f.set_time_base((1, 1));
        f.set_pts(5);
        store.add_message(&f.to_message(), &[], &[]).await?;

        let mut f = gen_frame();
        f.set_keyframe(Some(false));
        f.set_time_base((1, 1));
        f.set_pts(6);
        store.add_message(&f.to_message(), &[], &[]).await?;

        let first = store.get_first("", u, &JobOffset::Blocks(1)).await?;
        assert_eq!(first, Some(4));
        let (m, _, _) = store.get_message("", first.unwrap()).await?.unwrap();
        assert!(matches!(
            m.as_video_frame().unwrap().get_keyframe(),
            Some(true)
        ));

        let first_ts = store.get_first("", u, &JobOffset::Seconds(5.0)).await?;
        assert_eq!(first_ts, Some(0));
        let (m, _, _) = store.get_message("", first_ts.unwrap()).await?.unwrap();

        assert!(matches!(
            m.as_video_frame().unwrap().get_keyframe(),
            Some(true)
        ));

        Ok(())
    }
}
