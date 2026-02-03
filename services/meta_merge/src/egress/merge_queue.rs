use std::{
    cell::RefCell,
    collections::BTreeMap,
    rc::Rc,
    time::{Duration, SystemTime},
};

use pyo3::{types::PyList, Py};
#[cfg(not(test))]
use savant_core::utils::clock::clock::now as system_now;
#[cfg(test)]
use savant_core::utils::clock::mock_clock::now as system_now;
use savant_core_py::primitives::frame::VideoFrame;
use uuid::Uuid;

use super::payload::*;

#[derive(Debug, thiserror::Error)]
pub(super) enum MergeQueueError {
    #[error("Head is invalid, it is already taken")]
    HeadTaken,
    #[error("Queue is empty")]
    QueueIsEmpty,
    #[error("Frame is late and cannot be pushed to the merge queue.")]
    LateFrame(Uuid),
    #[error("Frame {:?} already exists", .0)]
    FrameAlreadyExists(Uuid),
    #[error("Failed to set EOS. Payload set error: {0}")]
    SetEosError(PayloadError),
    #[error("Item {:?} not found", .0)]
    ItemNotFound(Uuid),
    #[error("Failed to take frame. Payload take error: {0}")]
    TakeFrameError(PayloadError),
    #[error("Invalid message type: {:?}", .0)]
    InvalidMessageType(EgressMessage),
    #[error("Failed to set frame. Payload set error: {0}")]
    SetFrameError(PayloadError),
    #[error("Failed to set frame ready. Payload set frame ready error: {0}")]
    SetFrameReadyError(PayloadError),
    #[error("Head is not ready")]
    HeadNotReady,
    #[error("Failed to deconstruct payload. Payload deconstruct error: {0}")]
    PayloadDeconstructError(PayloadError),
    #[error("Invalid item UUID: expected {:?}, got {:?}", .0, .1)]
    InvalidItemUuid(Uuid, Option<Uuid>),
}

pub struct MergeQueue {
    max_duration: Duration,
    index: BTreeMap<Uuid, Payload>,
}

impl MergeQueue {
    pub fn new(max_duration: Duration) -> Self {
        Self {
            max_duration,
            index: BTreeMap::new(),
        }
    }

    pub fn size(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    pub(self) fn is_late(&self, uuid7: Uuid) -> Result<bool, MergeQueueError> {
        let uuid_as_u128 = uuid7.as_u128();
        let head = match self.front_uuid() {
            Ok(uuid) => uuid,
            Err(MergeQueueError::QueueIsEmpty) => return Ok(false),
            Err(e) => return Err(e),
        };
        let head_as_u128 = head.as_u128();
        if head_as_u128 > uuid_as_u128 {
            return Ok(true);
        }
        Ok(false)
    }

    pub(self) fn front_uuid(&self) -> Result<Uuid, MergeQueueError> {
        if let Some((uuid, _)) = self.index.iter().next() {
            return Ok(*uuid);
        }
        Err(MergeQueueError::QueueIsEmpty)
    }

    pub fn push_frame(
        &mut self,
        video_frame: VideoFrame,
        data: Py<PyList>,
        labels: Py<PyList>,
    ) -> Result<Uuid, MergeQueueError> {
        let late = self.is_late(video_frame.0.get_uuid())?;
        let uuid = video_frame.0.get_uuid();

        if late {
            return Err(MergeQueueError::LateFrame(uuid));
        }

        if let Some(_payload) = self.index.get(&uuid) {
            return Err(MergeQueueError::FrameAlreadyExists(uuid));
        }

        let item = EgressItem {
            uuid: Some(uuid),
            message: EgressMessage::VideoFrame(video_frame),
            data,
            labels,
            arrival_time: system_now(),
            is_ready: false,
        };
        self.index.insert(uuid, Payload::new(item, None));
        Ok(uuid)
    }

    pub fn push_eos(
        &mut self,
        data: Py<PyList>,
        labels: Py<PyList>,
    ) -> Result<Uuid, MergeQueueError> {
        let k = {
            self.index
                .last_key_value()
                .ok_or_else(|| MergeQueueError::QueueIsEmpty)?
                .0
                .clone()
        };

        let item = EgressItem {
            uuid: Some(k),
            message: EgressMessage::EndOfStream,
            data,
            labels,
            arrival_time: system_now(),
            is_ready: true,
        };

        let slot = self
            .index
            .get_mut(&k)
            .ok_or_else(|| MergeQueueError::QueueIsEmpty)?;
        if let Err(e) = slot.set_eos(item) {
            return Err(MergeQueueError::SetEosError(e));
        }
        Ok(k)
    }

    pub fn take_frame(&mut self, uuid: Uuid) -> Result<EgressItem, MergeQueueError> {
        let item = self
            .index
            .get_mut(&uuid)
            .ok_or_else(|| MergeQueueError::ItemNotFound(uuid))?
            .take_frame()
            .map_err(|e| MergeQueueError::TakeFrameError(e))?;
        Ok(item)
    }

    pub fn put_frame(&mut self, item: EgressItem) -> Result<(), MergeQueueError> {
        let uuid = match &item.message {
            EgressMessage::VideoFrame(video_frame) => video_frame.0.get_uuid(),
            _ => return Err(MergeQueueError::InvalidMessageType(item.message)),
        };
        if item.uuid.is_none() || !matches!(item.uuid, Some(uuid) if uuid == uuid) {
            return Err(MergeQueueError::InvalidItemUuid(uuid, item.uuid));
        }
        self.index
            .get_mut(&uuid)
            .ok_or_else(|| MergeQueueError::ItemNotFound(uuid))?
            .set_frame(item)
            .map_err(|e| MergeQueueError::SetFrameError(e))?;
        Ok(())
    }

    pub fn set_frame_ready(&mut self, uuid: Uuid) -> Result<(), MergeQueueError> {
        let item_ref_opt = self
            .index
            .get_mut(&uuid)
            .ok_or_else(|| MergeQueueError::ItemNotFound(uuid))?
            .set_frame_ready()
            .map_err(|e| MergeQueueError::SetFrameReadyError(e))?;
        Ok(())
    }

    pub fn is_head_ready(&mut self) -> bool {
        if let Some(payload) = self.index.values().next() {
            return payload.is_frame_ready()
                || (payload.frame.is_some()
                    && payload.frame.as_ref().unwrap().arrival_time + self.max_duration
                        < system_now());
        }
        false
    }

    pub fn fetch_head(&mut self) -> Result<(EgressItem, Option<EgressItem>), MergeQueueError> {
        if !self.is_head_ready() {
            return Err(MergeQueueError::HeadNotReady);
        }

        let head = self
            .index
            .iter()
            .next()
            .ok_or_else(|| MergeQueueError::QueueIsEmpty)?;
        let uuid = *head.0;
        let payload = self
            .index
            .remove(&uuid)
            .ok_or_else(|| MergeQueueError::ItemNotFound(uuid))?;
        let (frame, eos) = payload
            .deconstruct()
            .map_err(|e| MergeQueueError::PayloadDeconstructError(e))?;

        Ok((frame.unwrap(), eos))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use pyo3::Python;
    use savant_core::utils::clock::mock_clock::{advance_time_ms, reset_time};
    use savant_core_py::test::utils::gen_frame;

    #[test]
    fn test_merge_queue() -> Result<()> {
        reset_time();
        let mut queue = MergeQueue::new(Duration::from_secs(1));
        let late_frame = gen_frame();
        let ready_frame = gen_frame();
        let ready_dup = ready_frame.clone();
        let expired_frame = gen_frame();
        Python::attach(|py| {
            let uuid_ready = ready_frame.0.get_uuid();
            let uuid_ready_pushed = queue.push_frame(
                ready_frame,
                PyList::empty(py).unbind(),
                PyList::empty(py).unbind(),
            )?;
            assert_eq!(uuid_ready_pushed, uuid_ready);
            let res = queue.is_late(uuid_ready);
            assert!(matches!(res, Ok(false)));

            let err = queue.push_frame(
                ready_dup,
                PyList::empty(py).unbind(),
                PyList::empty(py).unbind(),
            );
            assert!(err.is_err());
            assert!(matches!(
                err.err(),
                Some(MergeQueueError::FrameAlreadyExists(uuid)) if uuid == uuid_ready
            ));

            let uuid_expired = expired_frame.0.get_uuid();
            let uuid_expired_pushed = queue.push_frame(
                expired_frame,
                PyList::empty(py).unbind(),
                PyList::empty(py).unbind(),
            )?;
            assert_eq!(uuid_expired_pushed, uuid_expired);

            let late_err = queue.push_frame(
                late_frame,
                PyList::empty(py).unbind(),
                PyList::empty(py).unbind(),
            );
            assert!(late_err.is_err());

            queue.push_eos(PyList::empty(py).unbind(), PyList::empty(py).unbind())?;
            assert_eq!(queue.size(), 2);

            // take and return frame
            let ready_item = queue.take_frame(uuid_ready)?;
            assert!(matches!(ready_item.message, EgressMessage::VideoFrame(_)));
            // try to take one more time (must fail)
            let double_take_ready_item = queue.take_frame(uuid_ready);
            assert!(double_take_ready_item.is_err());
            let head = queue.fetch_head();
            assert!(head.is_err());

            let expired_item = queue.take_frame(uuid_expired)?;
            assert!(matches!(expired_item.message, EgressMessage::VideoFrame(_)));
            let double_take_expired_item = queue.take_frame(uuid_expired);
            assert!(double_take_expired_item.is_err());

            queue.put_frame(ready_item)?;
            queue.put_frame(expired_item)?;

            let head_uuid = queue.front_uuid();
            assert!(matches!(head_uuid, Ok(uuid) if uuid == uuid_ready));

            queue.set_frame_ready(uuid_ready)?;
            let head = queue.fetch_head();
            assert!(head.is_ok());
            let (f, eos) = head.unwrap();
            assert!(matches!(f.message, EgressMessage::VideoFrame(_)));
            assert!(eos.is_none());

            let is_late = queue.is_late(uuid_ready)?;
            assert!(is_late);

            assert!(queue.size() == 1);
            assert!(!queue.is_head_ready());
            advance_time_ms(1001);
            assert!(queue.is_head_ready());
            let head = queue.fetch_head();
            assert!(head.is_ok());
            let (f, eos) = head.unwrap();
            assert!(matches!(f.message, EgressMessage::VideoFrame(_)));
            assert!(eos.is_some());
            let eos = eos.unwrap();
            assert!(matches!(eos.message, EgressMessage::EndOfStream));

            assert!(queue.size() == 0);
            Ok::<(), anyhow::Error>(())
        })?;
        Ok(())
    }
}
