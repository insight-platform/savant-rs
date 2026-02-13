use std::time::Duration;

use hashbrown::HashMap;
use pyo3::{types::PyList, PyErr, Python};
use savant_core_py::primitives::frame::VideoFrame;
use uuid::Uuid;

use crate::egress::{
    merge_queue::{HeadReadyReason, MergeQueue, MergeQueueError},
    payload::EgressItem,
};

/// A ready head: (source_id, frame, optional EOS, reason).
pub type ReadyHead = (String, EgressItem, Option<EgressItem>, HeadReadyReason);

#[derive(Debug, thiserror::Error)]
pub enum EgressError {
    #[error("Failed in Python context: {0}")]
    PythonError(#[from] PyErr),
    #[error("Failed to push frame. Merge queue push frame error: {0}")]
    PushFrameError(MergeQueueError),
    #[error("Failed to push EOS. Merge queue push EOS error: {0}")]
    PushEosError(MergeQueueError),
    #[error("Failed to take frame. Merge queue take frame error: {0}")]
    TakeFrameError(MergeQueueError),
    #[error("Failed to put frame. Merge queue put frame error: {0}")]
    PutFrameError(MergeQueueError),
    #[error("Failed to fetch head. Merge queue fetch head error: {0}")]
    FetchHeadError(MergeQueueError),
}

pub struct Egress {
    queues: HashMap<String, MergeQueue>,
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
        data: Vec<Vec<u8>>,
        labels: Vec<String>,
    ) -> Result<(), EgressError> {
        let source_id = frame.0.get_source_id();
        let queue = self
            .queues
            .entry(source_id)
            .or_insert(MergeQueue::new(self.max_duration));
        Python::attach(|py| {
            let data = PyList::new(py, data)?.unbind();
            let labels = PyList::new(py, labels)?.unbind();
            queue
                .push_frame(frame, data, labels)
                .map_err(EgressError::PushFrameError)?;
            Ok(())
        })
    }

    pub fn push_eos(
        &mut self,
        source_id: String,
        data: Vec<Vec<u8>>,
        labels: Vec<String>,
    ) -> Result<(), EgressError> {
        let queue = self
            .queues
            .entry(source_id)
            .or_insert(MergeQueue::new(self.max_duration));
        Python::attach(|py| {
            let data = PyList::new(py, data)?.unbind();
            let labels = PyList::new(py, labels)?.unbind();
            queue
                .push_eos(data, labels)
                .map_err(EgressError::PushEosError)?;
            Ok(())
        })
    }

    pub fn take_frame(&mut self, source_id: String, uuid: Uuid) -> Result<EgressItem, EgressError> {
        let queue = self
            .queues
            .entry(source_id)
            .or_insert(MergeQueue::new(self.max_duration));
        let item = queue
            .take_frame(uuid)
            .map_err(EgressError::TakeFrameError)?;
        Ok(item)
    }

    pub fn put_frame(&mut self, source_id: String, item: EgressItem) -> Result<(), EgressError> {
        let queue = self
            .queues
            .entry(source_id)
            .or_insert(MergeQueue::new(self.max_duration));
        queue.put_frame(item).map_err(EgressError::PutFrameError)?;
        Ok(())
    }

    pub fn set_frame_ready(&mut self, source_id: String, uuid: Uuid) -> anyhow::Result<()> {
        let queue = self
            .queues
            .entry(source_id)
            .or_insert(MergeQueue::new(self.max_duration));
        queue.set_frame_ready(uuid)?;
        Ok(())
    }

    pub fn fetch_ready(&mut self) -> Result<Vec<ReadyHead>, EgressError> {
        let mut heads = Vec::new();
        for (source_id, queue) in self.queues.iter_mut() {
            if queue.is_head_ready().is_none() {
                continue;
            }

            let (item, eos, reason) = queue.fetch_head().map_err(EgressError::FetchHeadError)?;
            heads.push((source_id.clone(), item, eos, reason));
        }
        Ok(heads)
    }
}
