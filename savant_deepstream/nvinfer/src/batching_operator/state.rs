use deepstream_buffers::SharedBuffer;
use savant_core::primitives::frame::VideoFrameProxy;
use std::collections::HashSet;
use std::time::Instant;

pub(super) struct BatchState {
    pub(super) frames: Vec<(VideoFrameProxy, SharedBuffer)>,
    pub(super) sources: HashSet<String>,
    pub(super) deadline: Option<Instant>,
}

impl BatchState {
    pub(super) fn new() -> Self {
        Self {
            frames: Vec::new(),
            sources: HashSet::new(),
            deadline: None,
        }
    }

    pub(super) fn take(&mut self) -> Vec<(VideoFrameProxy, SharedBuffer)> {
        self.sources.clear();
        self.deadline = None;
        std::mem::take(&mut self.frames)
    }

    pub(super) fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}
