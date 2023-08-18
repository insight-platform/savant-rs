use crate::match_query::MatchQuery;
use crate::primitives::frame::{VideoFrame, VideoFrameProxy};
use crate::primitives::object::VideoObjectProxy;
use crate::primitives::AttributeMethods;
use rkyv::{with::Skip, Archive, Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Archive, Deserialize, Serialize, Debug, Clone, Default)]
#[archive(check_bytes)]
pub struct VideoFrameBatch {
    offline_frames: Vec<(i64, VideoFrame)>,
    #[with(Skip)]
    pub(crate) frames: HashMap<i64, VideoFrameProxy>,
}

impl VideoFrameBatch {
    pub fn deep_copy(&self) -> Self {
        let frames = self
            .frames
            .iter()
            .map(|(id, frame)| (*id, frame.deep_copy()))
            .collect();

        Self {
            offline_frames: Default::default(),
            frames,
        }
    }

    pub(crate) fn prepare_after_load(&mut self) {
        let offline_frames = std::mem::take(&mut self.offline_frames);
        for (id, inner) in offline_frames.into_iter() {
            let frame = VideoFrameProxy::from_inner(inner);
            frame.restore_from_snapshot();
            self.frames.insert(id, frame);
        }
    }

    pub(crate) fn prepare_before_save(&mut self) {
        self.offline_frames.clear();
        for (id, frame) in self.frames.iter() {
            let frame = frame.deep_copy();
            frame.exclude_temporary_attributes();
            frame.get_all_objects().iter().for_each(|o| {
                o.exclude_temporary_attributes();
            });
            frame.make_snapshot();
            let inner = Arc::try_unwrap(frame.inner).unwrap().into_inner(); //..into_inner();
            self.offline_frames.push((*id, *inner));
        }
    }

    pub fn snapshot(&mut self) {
        self.prepare_before_save();
    }

    pub fn restore(&mut self) {
        self.prepare_after_load();
    }

    pub fn access_objects(&self, q: &MatchQuery) -> hashbrown::HashMap<i64, Vec<VideoObjectProxy>> {
        self.frames
            .iter()
            .map(|(id, frame)| (*id, frame.access_objects(q)))
            .collect()
    }

    pub fn delete_objects(&mut self, q: &MatchQuery) {
        self.frames.iter_mut().for_each(|(_, frame)| {
            frame.delete_objects(q);
        });
    }

    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, id: i64, frame: VideoFrameProxy) {
        self.frames.insert(id, frame);
    }

    pub fn get(&self, id: i64) -> Option<VideoFrameProxy> {
        self.frames.get(&id).cloned()
    }

    pub fn del(&mut self, id: i64) -> Option<VideoFrameProxy> {
        self.frames.remove(&id)
    }
}
