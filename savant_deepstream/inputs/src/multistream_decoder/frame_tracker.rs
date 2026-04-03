//! Bounded map from `frame_id` (`uuid`) to `VideoFrameProxy` for async decode completion.

use savant_core::primitives::frame::VideoFrameProxy;
use std::collections::HashMap;
use std::collections::VecDeque;

/// Tracks pending frames for correlation with `NvDecoder` output.
pub struct FrameTracker {
    map: HashMap<u128, VideoFrameProxy>,
    order: VecDeque<u128>,
    capacity: usize,
}

impl FrameTracker {
    pub fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            capacity: capacity.max(1),
        }
    }

    /// Insert proxy; if over capacity, removes oldest and returns it (caller should emit Undecoded).
    #[allow(clippy::map_entry)]
    pub fn insert(&mut self, id: u128, frame: VideoFrameProxy) -> Option<VideoFrameProxy> {
        let mut evicted = None;
        if self.map.contains_key(&id) {
            self.map.insert(id, frame);
            return None;
        }
        while self.map.len() >= self.capacity && !self.order.is_empty() {
            if let Some(old) = self.order.pop_front() {
                evicted = self.map.remove(&old);
            }
        }
        self.order.push_back(id);
        self.map.insert(id, frame);
        evicted
    }

    pub fn remove(&mut self, id: u128) -> Option<VideoFrameProxy> {
        let v = self.map.remove(&id);
        if v.is_some() {
            self.order.retain(|x| *x != id);
        }
        v
    }

    #[allow(dead_code)] // reserved for explicit flush / shutdown paths
    pub fn drain(&mut self) -> Vec<(u128, VideoFrameProxy)> {
        let mut out = Vec::new();
        for id in self.order.drain(..) {
            if let Some(f) = self.map.remove(&id) {
                out.push((id, f));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use savant_core::primitives::frame::VideoFrameContent;
    use savant_core::primitives::frame::VideoFrameProxy;
    use savant_core::primitives::frame::VideoFrameTranscodingMethod;
    use savant_core::primitives::video_codec::VideoCodec;

    fn dummy_frame(source: &str, _uuid: u128) -> VideoFrameProxy {
        VideoFrameProxy::new(
            source,
            (30, 1),
            64,
            64,
            VideoFrameContent::Internal(vec![1, 2, 3]),
            VideoFrameTranscodingMethod::Encoded,
            Some(VideoCodec::H264),
            None,
            (1, 1),
            0,
            None,
            None,
        )
        .expect("frame")
    }

    #[test]
    fn evicts_oldest_when_full() {
        let mut t = FrameTracker::new(2);
        assert!(t.insert(1, dummy_frame("s", 1)).is_none());
        assert!(t.insert(2, dummy_frame("s", 2)).is_none());
        let e = t.insert(3, dummy_frame("s", 3));
        assert!(e.is_some());
        assert!(!t.map.contains_key(&1));
        assert!(t.map.contains_key(&3));
    }
}
