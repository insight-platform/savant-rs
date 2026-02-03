use std::time::Duration;

use hashbrown::HashMap;
use pyo3::{types::PyList, Py};
use savant_core_py::primitives::frame::VideoFrame;
use uuid::Uuid;

use crate::egress::payload::EgressItem;

mod merge_queue;
mod payload;

pub struct Egress {
    queues: HashMap<String, merge_queue::MergeQueue>,
    max_duration: Duration,
}

impl Egress {
    pub fn new(max_duration: Duration) -> Self {
        Self {
            queues: HashMap::new(),
            max_duration,
        }
    }

    pub fn push_frame(
        &mut self,
        frame: VideoFrame,
        data: Py<PyList>,
        labels: Py<PyList>,
    ) -> anyhow::Result<()> {
        let source_id = frame.0.get_source_id();
        let queue = self
            .queues
            .entry(source_id)
            .or_insert(merge_queue::MergeQueue::new(self.max_duration));
        queue.push_frame(frame, data, labels)?;
        Ok(())
    }

    pub fn push_eos(
        &mut self,
        source_id: String,
        data: Py<PyList>,
        labels: Py<PyList>,
    ) -> anyhow::Result<()> {
        let queue = self
            .queues
            .entry(source_id)
            .or_insert(merge_queue::MergeQueue::new(self.max_duration));
        queue.push_eos(data, labels)?;
        Ok(())
    }

    pub fn take_frame(&mut self, source_id: String, uuid: Uuid) -> anyhow::Result<EgressItem> {
        let queue = self
            .queues
            .entry(source_id)
            .or_insert(merge_queue::MergeQueue::new(self.max_duration));
        let item = queue.take_frame(uuid)?;
        Ok(item)
    }

    pub fn put_frame(&mut self, source_id: String, item: EgressItem) -> anyhow::Result<()> {
        let queue = self
            .queues
            .entry(source_id)
            .or_insert(merge_queue::MergeQueue::new(self.max_duration));
        queue.put_frame(item)?;
        Ok(())
    }

    pub fn set_frame_ready(&mut self, source_id: String, uuid: Uuid) -> anyhow::Result<()> {
        let queue = self
            .queues
            .entry(source_id)
            .or_insert(merge_queue::MergeQueue::new(self.max_duration));
        queue.set_frame_ready(uuid)?;
        Ok(())
    }

    fn is_head_ready(&mut self, source_id: String) -> bool {
        let queue = self
            .queues
            .entry(source_id)
            .or_insert(merge_queue::MergeQueue::new(self.max_duration));
        queue.is_head_ready()
    }

    pub fn fetch_ready(&mut self) -> anyhow::Result<Vec<(String, EgressItem, Option<EgressItem>)>> {
        let mut heads = Vec::new();
        for (source_id, queue) in self.queues.iter_mut() {
            if !queue.is_head_ready() {
                continue;
            }

            let (item, eos) = queue.fetch_head()?;
            heads.push((source_id.clone(), item, eos));
        }
        Ok(heads)
    }
}
