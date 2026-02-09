use std::time::SystemTime;

use pyo3::{
    exceptions::PyValueError,
    pyclass,
    types::{PyDict, PyList},
    Py, PyResult, Python,
};
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

#[pyclass]
#[pyo3(name = "EgressItem")]
pub struct EgressItemPy {
    pub video_frame: VideoFrame,
    pub state: Py<PyDict>,
    pub data: Py<PyList>,
    pub labels: Py<PyList>,
}

pub struct EgressItem {
    pub uuid: Option<Uuid>,
    pub message: EgressMessage,
    pub state: Py<PyDict>,
    pub data: Py<PyList>,
    pub labels: Py<PyList>,
    pub arrival_time: SystemTime,
    pub is_ready: bool,
}

impl Default for EgressItem {
    fn default() -> Self {
        let (state, data, labels) = Python::attach(|py| {
            (
                PyDict::new(py).unbind(),
                PyList::empty(py).unbind(),
                PyList::empty(py).unbind(),
            )
        });

        Self {
            uuid: None,
            state,
            message: EgressMessage::EndOfStream,
            data,
            labels,
            arrival_time: system_now(),
            is_ready: false,
        }
    }
}

impl EgressItem {
    pub fn new_frame(frame: VideoFrame, data: Py<PyList>, labels: Py<PyList>) -> Self {
        Self {
            uuid: Some(frame.0.get_uuid()),
            message: EgressMessage::VideoFrame(frame),
            data,
            labels,
            ..Default::default()
        }
    }

    pub fn as_frame(&self) -> Option<&VideoFrame> {
        if let EgressMessage::VideoFrame(frame) = &self.message {
            Some(&frame)
        } else {
            None
        }
    }

    pub fn new_eos(data: Py<PyList>, labels: Py<PyList>) -> Self {
        Self {
            data,
            labels,
            is_ready: true,
            ..Default::default()
        }
    }

    pub fn new_eos_with_uuid(uuid: Uuid, data: Py<PyList>, labels: Py<PyList>) -> Self {
        Self {
            uuid: Some(uuid),
            message: EgressMessage::EndOfStream,
            data,
            labels,
            ..Default::default()
        }
    }

    pub fn to_py(&self) -> PyResult<Py<EgressItemPy>> {
        Python::attach(|py| match &self.message {
            EgressMessage::VideoFrame(frame) => Py::new(
                py,
                EgressItemPy {
                    video_frame: frame.clone(),
                    state: self.state.clone_ref(py),
                    data: self.data.clone_ref(py),
                    labels: self.labels.clone_ref(py),
                },
            ),
            _ => Err(PyValueError::new_err(
                "Invalid message type, the operation is supported only for video frames",
            )),
        })
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
            let frame = EgressItem::new_frame(gen_frame(), frame_data, frame_labels);

            let eos_data = PyList::empty(py).unbind();
            let eos_labels = PyList::empty(py).unbind();
            let eos = EgressItem::new_eos(eos_data, eos_labels);
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
            let new_frame = EgressItem::new_frame(gen_frame(), new_frame_data, new_frame_labels);
            let res = payload.set_frame(new_frame);
            assert!(matches!(res, Err(PayloadError::FrameAlreadyFilled(_))));

            let new_eos_data = PyList::empty(py).unbind();
            let new_eos_labels = PyList::empty(py).unbind();
            let new_eos = EgressItem::new_eos(new_eos_data, new_eos_labels);
            let res = payload.set_eos(new_eos);
            assert!(matches!(res, Err(PayloadError::EosAlreadyFilled)));

            Ok::<(), anyhow::Error>(())
        })
    }
}
