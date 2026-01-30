use std::{
    cell::RefCell,
    collections::VecDeque,
    rc::Rc,
    time::{Duration, SystemTime},
};

use hashbrown::HashMap;
use pyo3::types::{PyBytes, PyList, PyString};
use savant_core_py::primitives::{eos::EndOfStream, frame::VideoFrame};
use uuid::Uuid;

use anyhow::{anyhow, bail, Result};

#[derive(Debug)]
enum EgressMessage {
    VideoFrame(VideoFrame),
    EndOfStream(EndOfStream),
}
struct EgressItem {
    message: EgressMessage,
    data: PyList,
    labels: PyList,
    arrival_time: SystemTime,
    is_ready: bool,
}

type EgressItemRef = Rc<RefCell<Option<EgressItem>>>;

struct EgressQueue {
    max_duration: Duration,
    index: HashMap<Uuid, EgressItemRef>,
    items: VecDeque<EgressItemRef>,
}

impl EgressQueue {
    pub fn new(max_duration: Duration) -> Self {
        Self {
            max_duration,
            index: HashMap::new(),
            items: VecDeque::new(),
        }
    }

    pub fn push_frame(&mut self, video_frame: VideoFrame, data: PyList, labels: PyList) -> Uuid {
        let uuid = video_frame.0.get_uuid();
        let item = EgressItem {
            message: EgressMessage::VideoFrame(video_frame),
            data,
            labels,
            arrival_time: SystemTime::now(),
            is_ready: false,
        };
        let index_obj = Rc::new(RefCell::new(Some(item)));
        let queue_obj = index_obj.clone();
        self.index.insert(uuid, index_obj);
        self.items.push_back(queue_obj);
        uuid
    }

    pub fn push_eos(&mut self, eos: EndOfStream, data: PyList, labels: PyList) {
        let item = EgressItem {
            message: EgressMessage::EndOfStream(eos),
            data,
            labels,
            arrival_time: SystemTime::now(),
            is_ready: true,
        };
        self.items.push_back(Rc::new(RefCell::new(Some(item))));
    }

    pub fn take_frame(&mut self, uuid: Uuid) -> Result<EgressItem> {
        let item = self
            .index
            .get(&uuid)
            .ok_or_else(|| anyhow!("Item with UUID {} not found", uuid))?
            .borrow_mut()
            .take()
            .ok_or_else(|| anyhow!("Item with UUID {} is already taken", uuid))?;
        Ok(item)
    }

    pub fn put_frame(&mut self, item: EgressItem) -> Result<()> {
        let uuid = match &item.message {
            EgressMessage::VideoFrame(video_frame) => video_frame.0.get_uuid(),
            _ => bail!(
                "Invalid message type: {:?}, the operation is supported only for video frames",
                item.message
            ),
        };
        let index_obj = self
            .index
            .get(&uuid)
            .ok_or_else(|| anyhow!("Item with UUID {} not found", uuid))?;
        index_obj.borrow_mut().replace(item);
        Ok(())
    }

    pub fn is_head_ready(&mut self) -> bool {
        if let Some(item) = self.items.front() {
            if let Some(item) = item.borrow().as_ref() {
                item.arrival_time + self.max_duration < SystemTime::now() || item.is_ready
            } else {
                false
            }
        } else {
            false
        }
    }

    pub fn fetch_head(&mut self) -> Option<EgressItem> {
        let front = self.items.pop_front()?;
        if let EgressMessage::VideoFrame(video_frame) = &front.borrow().as_ref()?.message {
            self.index.remove(&video_frame.0.get_uuid());
        }
        let mut bind = front.borrow_mut();
        bind.take()
    }
}
