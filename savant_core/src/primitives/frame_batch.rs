use crate::match_query::MatchQuery;
use crate::primitives::frame::VideoFrameProxy;
use crate::primitives::object::VideoObjectProxy;
use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct VideoFrameBatch {
    pub(crate) frames: HashMap<i64, VideoFrameProxy>,
}

impl VideoFrameBatch {
    pub fn exclude_all_temporary_attributes(&mut self) {
        self.frames.iter_mut().for_each(|(_, frame)| {
            frame.exclude_all_temporary_attributes();
        });
    }
    pub fn smart_copy(&self) -> Self {
        let frames = self
            .frames
            .iter()
            .map(|(id, frame)| (*id, frame.smart_copy()))
            .collect();

        Self { frames }
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

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            frames: HashMap::with_capacity(capacity),
        }
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

    pub fn frames(&self) -> &HashMap<i64, VideoFrameProxy> {
        &self.frames
    }
}
