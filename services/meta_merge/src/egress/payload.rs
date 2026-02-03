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

#[derive(Debug, thiserror::Error)]
pub(super) enum PayloadError {
    #[error("Item is already taken")]
    ItemTaken,
    #[error("Invalid message type, the operation is supported only for video frames")]
    InvalidMessageType,
    #[error("Item {:?} already exists", .0)]
    FrameAlreadyFilled(Uuid),
    #[error("EOS is already filled")]
    EosAlreadyFilled,
}

#[derive(Debug)]
pub(super) enum EgressMessage {
    VideoFrame(VideoFrame),
    EndOfStream,
}

pub(super) struct EgressItem {
    pub uuid: Option<Uuid>,
    pub topic: String,
    pub message: EgressMessage,
    pub data: Py<PyList>,
    pub labels: Py<PyList>,
    pub arrival_time: SystemTime,
    pub is_ready: bool,
}

impl EgressItem {
    pub fn new_frame(
        topic: String,
        frame: VideoFrame,
        data: Py<PyList>,
        labels: Py<PyList>,
    ) -> Self {
        Self {
            uuid: None,
            topic,
            message: EgressMessage::VideoFrame(frame),
            data,
            labels,
            arrival_time: system_now(),
            is_ready: false,
        }
    }

    pub fn as_frame(&self) -> Option<&VideoFrame> {
        if let EgressMessage::VideoFrame(frame) = &self.message {
            Some(&frame)
        } else {
            None
        }
    }

    pub fn new_eos(topic: String, data: Py<PyList>, labels: Py<PyList>) -> Self {
        Self {
            uuid: None,
            topic,
            message: EgressMessage::EndOfStream,
            data,
            labels,
            arrival_time: system_now(),
            is_ready: true,
        }
    }
}

pub(super) type PayloadResult<T> = Result<T, PayloadError>;

pub(super) struct Payload {
    pub(super) frame: Option<EgressItem>,
    pub(super) eos: Option<EgressItem>,
}

impl Payload {
    pub fn new(frame: EgressItem, eos: Option<EgressItem>) -> Self {
        Self {
            frame: Some(frame),
            eos,
        }
    }

    pub fn take_frame(&mut self) -> PayloadResult<EgressItem> {
        if let Some(frame) = self.frame.take() {
            return Ok(frame);
        }
        Err(PayloadError::ItemTaken)
    }

    pub fn set_frame(&mut self, item: EgressItem) -> PayloadResult<()> {
        if let EgressMessage::EndOfStream = item.message {
            return Err(PayloadError::InvalidMessageType);
        }

        if let Some(old) = &self.frame {
            let uuid = match &old.message {
                EgressMessage::VideoFrame(video_frame) => video_frame.0.get_uuid(),
                _ => return Err(PayloadError::InvalidMessageType),
            };
            return Err(PayloadError::FrameAlreadyFilled(uuid));
        }
        self.frame = Some(item);
        Ok(())
    }

    fn take_eos(&mut self) -> Option<EgressItem> {
        self.eos.take()
    }

    pub fn set_eos(&mut self, item: EgressItem) -> PayloadResult<()> {
        if let EgressMessage::VideoFrame(_) = item.message {
            return Err(PayloadError::InvalidMessageType);
        }

        if let Some(_eos) = self.eos.as_ref() {
            return Err(PayloadError::EosAlreadyFilled);
        }
        self.eos = Some(item);
        Ok(())
    }

    pub fn deconstruct(mut self) -> PayloadResult<(Option<EgressItem>, Option<EgressItem>)> {
        Ok((Some(self.take_frame()?), self.take_eos()))
    }

    pub fn uuid(&self) -> PayloadResult<Uuid> {
        if let Some(frame) = &self.frame {
            if let EgressMessage::VideoFrame(video_frame) = &frame.message {
                return Ok(video_frame.0.get_uuid());
            }
        }
        Err(PayloadError::ItemTaken)
    }

    pub fn is_frame_ready(&self) -> bool {
        if let Some(frame) = &self.frame {
            return frame.is_ready;
        }
        false
    }

    pub fn set_frame_ready(&mut self) -> PayloadResult<()> {
        if let Some(frame) = &mut self.frame {
            frame.is_ready = true;
            return Ok(());
        }
        Err(PayloadError::ItemTaken)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use pyo3::Python;
    use savant_core_py::test::utils::gen_frame;

    #[test]
    fn test_payload() -> Result<()> {
        Python::attach(|py| {
            let frame_data = PyList::empty(py).unbind();
            let frame_labels = PyList::empty(py).unbind();
            let frame =
                EgressItem::new_frame("test".to_string(), gen_frame(), frame_data, frame_labels);

            let eos_data = PyList::empty(py).unbind();
            let eos_labels = PyList::empty(py).unbind();
            let eos = EgressItem::new_eos("test".to_string(), eos_data, eos_labels);
            let payload = Payload::new(frame, Some(eos));

            let (frame, eos) = payload.deconstruct()?;
            assert!(frame.is_some());
            assert!(eos.is_some());
            let frame = frame.unwrap();
            let eos = eos.unwrap();
            assert!(matches!(frame.message, EgressMessage::VideoFrame(_)));
            assert!(matches!(eos.message, EgressMessage::EndOfStream));

            let mut payload = Payload::new(frame, Some(eos));
            let _uuid = payload.uuid()?;

            let frame = payload.take_frame()?;
            assert!(matches!(frame.message, EgressMessage::VideoFrame(_)));

            let eos_opt = payload.take_eos();
            assert!(
                matches!(eos_opt.as_ref(), Some(eos) if matches!(eos.message, EgressMessage::EndOfStream))
            );

            let mut payload = Payload::new(frame, eos_opt);
            let new_frame_data = PyList::empty(py).unbind();
            let new_frame_labels = PyList::empty(py).unbind();
            let new_frame = EgressItem::new_frame(
                "test".to_string(),
                gen_frame(),
                new_frame_data,
                new_frame_labels,
            );
            let res = payload.set_frame(new_frame);
            assert!(matches!(res, Err(PayloadError::FrameAlreadyFilled(_))));

            let new_eos_data = PyList::empty(py).unbind();
            let new_eos_labels = PyList::empty(py).unbind();
            let new_eos = EgressItem::new_eos("test".to_string(), new_eos_data, new_eos_labels);
            let res = payload.set_eos(new_eos);
            assert!(matches!(res, Err(PayloadError::EosAlreadyFilled)));

            Ok::<(), anyhow::Error>(())
        })
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use anyhow::Result;
//     use pyo3::Python;
//     use savant_core::utils::clock::mock_clock::{advance_time_ms, reset_time};
//     use savant_core_py::primitives::eos::EndOfStream;
//     use savant_core_py::test::utils::gen_frame;

//     #[test]
//     fn test_merge_queue() -> Result<()> {
//         reset_time();
//         let mut queue = MergeQueue::new(Duration::from_secs(1));
//         let late_frame = gen_frame();
//         let ready_frame = gen_frame();
//         let expired_frame = gen_frame();
//         Python::attach(|py| {
//             let uuid_ready = ready_frame.0.get_uuid();
//             let uuid_ready_pushed = queue.push_frame(
//                 ready_frame,
//                 PyList::empty(py).unbind(),
//                 PyList::empty(py).unbind(),
//             )?;
//             assert_eq!(uuid_ready_pushed, uuid_ready);
//             let res = queue.is_late(uuid_ready);
//             assert!(matches!(res, Ok(false)));

//             let uuid_expired = expired_frame.0.get_uuid();
//             let uuid_expired_pushed = queue.push_frame(
//                 expired_frame,
//                 PyList::empty(py).unbind(),
//                 PyList::empty(py).unbind(),
//             )?;
//             assert_eq!(uuid_expired_pushed, uuid_expired);

//             let late_err = queue.push_frame(
//                 late_frame,
//                 PyList::empty(py).unbind(),
//                 PyList::empty(py).unbind(),
//             );
//             assert!(late_err.is_err());

//             queue.push_eos(PyList::empty(py).unbind(), PyList::empty(py).unbind());
//             assert_eq!(queue.size(), 3);

//             // take and return frame
//             let egress_item = queue.take_frame(uuid_ready)?;
//             assert!(matches!(egress_item.message, EgressMessage::VideoFrame(_)));
//             // try to take one more time (must fail)
//             let egress_item_err = queue.take_frame(uuid_ready);
//             assert!(egress_item_err.is_err());
//             let head_uuid = queue.front_uuid();
//             assert!(head_uuid.is_err());

//             queue.put_frame(egress_item)?;
//             let head_uuid = queue.front_uuid();
//             assert!(head_uuid.is_ok());
//             assert!(matches!(head_uuid, Ok(Some(uuid)) if uuid == uuid_ready));

//             queue.set_frame_ready(uuid_ready)?;
//             let head = queue.fetch_head();
//             assert!(head.is_some());
//             assert!(matches!(
//                 head.unwrap().message,
//                 EgressMessage::VideoFrame(_)
//             ));

//             let is_late = queue.is_late(uuid_ready)?;
//             assert!(is_late);

//             assert!(queue.size() == 2);
//             assert!(!queue.is_head_ready());
//             advance_time_ms(1001);
//             assert!(queue.is_head_ready());
//             let head = queue.fetch_head();
//             assert!(head.is_some());
//             assert!(matches!(
//                 head.unwrap().message,
//                 EgressMessage::VideoFrame(_)
//             ));

//             assert!(queue.size() == 1);
//             assert!(queue.is_head_ready()); // EOS is ready immediately
//             let head = queue.fetch_head();
//             assert!(head.is_some());
//             assert!(matches!(
//                 head.unwrap().message,
//                 EgressMessage::EndOfStream(_)
//             ));
//             assert!(queue.size() == 0);
//             Ok::<(), anyhow::Error>(())
//         })?;
//         Ok(())
//     }
// }
