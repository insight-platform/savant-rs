use std::{
    cell::RefCell,
    collections::VecDeque,
    rc::Rc,
    time::{Duration, SystemTime},
};

use hashbrown::HashMap;
use pyo3::{types::PyList, Py};
#[cfg(not(test))]
use savant_core::utils::clock::clock::now as system_now;
#[cfg(test)]
use savant_core::utils::clock::mock_clock::now as system_now;
use savant_core_py::primitives::{eos::EndOfStream, frame::VideoFrame};
use uuid::Uuid;

use anyhow::{anyhow, bail, Result};

#[derive(Debug)]
enum EgressMessage {
    VideoFrame(VideoFrame),
    EndOfStream(EndOfStream),
}
struct EgressItem {
    pub message: EgressMessage,
    pub data: Py<PyList>,
    pub labels: Py<PyList>,
    pub arrival_time: SystemTime,
    pub is_ready: bool,
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

    pub fn size(&self) -> usize {
        self.items.len()
    }

    pub fn push_frame(
        &mut self,
        video_frame: VideoFrame,
        data: Py<PyList>,
        labels: Py<PyList>,
    ) -> Uuid {
        let uuid = video_frame.0.get_uuid();
        let item = EgressItem {
            message: EgressMessage::VideoFrame(video_frame),
            data,
            labels,
            arrival_time: system_now(),
            is_ready: false,
        };
        let index_obj = Rc::new(RefCell::new(Some(item)));
        let queue_obj = index_obj.clone();
        self.index.insert(uuid, index_obj);
        self.items.push_back(queue_obj);
        uuid
    }

    pub fn push_eos(&mut self, eos: EndOfStream, data: Py<PyList>, labels: Py<PyList>) {
        let item = EgressItem {
            message: EgressMessage::EndOfStream(eos),
            data,
            labels,
            arrival_time: system_now(),
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

    pub fn set_frame_ready(&mut self, uuid: Uuid) -> Result<()> {
        let item = self
            .index
            .get(&uuid)
            .ok_or_else(|| anyhow!("Item with UUID {} not found", uuid))?;
        item.borrow_mut()
            .as_mut()
            .ok_or_else(|| anyhow!("Item with UUID {} is already taken", uuid))?
            .is_ready = true;
        Ok(())
    }

    pub fn is_head_ready(&mut self) -> bool {
        if let Some(item) = self.items.front() {
            if let Some(item) = item.borrow().as_ref() {
                item.arrival_time + self.max_duration < system_now() || item.is_ready
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

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use pyo3::Python;
    use savant_core::utils::clock::mock_clock::{advance_time_ms, reset_time};
    use savant_core_py::primitives::eos::EndOfStream;
    use savant_core_py::test::utils::gen_frame;

    #[test]
    fn test_egress_queue() -> Result<()> {
        reset_time();
        let mut queue = EgressQueue::new(Duration::from_secs(1));
        let ready_frame = gen_frame();
        let expired_frame = gen_frame();
        let eos = EndOfStream::new("test".to_string());
        Python::attach(|py| {
            let uuid_ready = ready_frame.0.get_uuid();
            let uuid_ready_pushed = queue.push_frame(
                ready_frame,
                PyList::empty(py).unbind(),
                PyList::empty(py).unbind(),
            );
            assert_eq!(uuid_ready_pushed, uuid_ready);

            let uuid_expired = expired_frame.0.get_uuid();
            let uuid_expired_pushed = queue.push_frame(
                expired_frame,
                PyList::empty(py).unbind(),
                PyList::empty(py).unbind(),
            );
            assert_eq!(uuid_expired_pushed, uuid_expired);

            queue.push_eos(eos, PyList::empty(py).unbind(), PyList::empty(py).unbind());
            assert_eq!(queue.size(), 3);

            /// take and return frame
            let egress_item = queue.take_frame(uuid_ready)?;
            assert!(matches!(egress_item.message, EgressMessage::VideoFrame(_)));
            // try to take one more time
            let egress_item_err = queue.take_frame(uuid_ready);
            assert!(egress_item_err.is_err());
            queue.put_frame(egress_item)?;

            queue.set_frame_ready(uuid_ready)?;
            let head = queue.fetch_head();
            assert!(head.is_some());
            assert!(matches!(
                head.unwrap().message,
                EgressMessage::VideoFrame(_)
            ));

            assert!(queue.size() == 2);
            assert!(!queue.is_head_ready());
            advance_time_ms(1001);
            assert!(queue.is_head_ready());
            let head = queue.fetch_head();
            assert!(head.is_some());
            assert!(matches!(
                head.unwrap().message,
                EgressMessage::VideoFrame(_)
            ));

            assert!(queue.size() == 1);
            assert!(queue.is_head_ready()); // EOS is ready immediately
            let head = queue.fetch_head();
            assert!(head.is_some());
            assert!(matches!(
                head.unwrap().message,
                EgressMessage::EndOfStream(_)
            ));
            assert!(queue.size() == 0);
            Ok::<(), anyhow::Error>(())
        })?;
        Ok(())
    }
}
