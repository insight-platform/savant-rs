use std::time::SystemTime;

use pyo3::{
    exceptions::PyValueError,
    pyclass, pymethods,
    types::{PyDict, PyList},
    Py, PyResult, Python,
};
#[cfg(test)]
use savant_core::utils::clock::mock_clock::now as system_now;
#[cfg(not(test))]
use savant_core::utils::clock::real_clock::now as system_now;
use savant_core_py::primitives::frame::VideoFrame;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum PayloadError {
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
pub enum EgressMessage {
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

#[pymethods]
impl EgressItemPy {
    #[getter]
    fn get_video_frame(&self) -> VideoFrame {
        self.video_frame.clone()
    }

    #[setter]
    fn set_video_frame(&mut self, video_frame: VideoFrame) {
        self.video_frame = video_frame;
    }

    #[getter]
    fn get_state(&self, py: Python<'_>) -> Py<PyDict> {
        self.state.clone_ref(py)
    }

    #[setter]
    fn set_state(&mut self, state: Py<PyDict>) {
        self.state = state;
    }

    #[getter]
    fn get_data(&self, py: Python<'_>) -> Py<PyList> {
        self.data.clone_ref(py)
    }

    #[setter]
    fn set_data(&mut self, data: Py<PyList>) {
        self.data = data;
    }

    #[getter]
    fn get_labels(&self, py: Python<'_>) -> Py<PyList> {
        self.labels.clone_ref(py)
    }

    #[setter]
    fn set_labels(&mut self, labels: Py<PyList>) {
        self.labels = labels;
    }
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
    pub fn new_frame_rust_types(
        frame: VideoFrame,
        data: Vec<Vec<u8>>,
        labels: Vec<String>,
        py: Python<'_>,
    ) -> PyResult<Self> {
        let data = PyList::new(py, data)?.unbind();
        let labels = PyList::new(py, labels)?.unbind();
        Ok(Self {
            uuid: Some(frame.0.get_uuid()),
            message: EgressMessage::VideoFrame(frame),
            data,
            labels,
            ..Default::default()
        })
    }

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
            Some(frame)
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

    pub fn to_py(&self, py: Python<'_>) -> PyResult<Py<EgressItemPy>> {
        match &self.message {
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
        }
    }

    /// Update this EgressItem in-place from a modified EgressItemPy (after Python callback).
    pub fn update_from_py(&mut self, py_item: &EgressItemPy) {
        Python::attach(|py| {
            self.message = EgressMessage::VideoFrame(py_item.video_frame.clone());
            self.state = py_item.state.clone_ref(py);
            self.data = py_item.data.clone_ref(py);
            self.labels = py_item.labels.clone_ref(py);
        });
    }
}

pub type PayloadResult<T> = Result<T, PayloadError>;

pub struct Payload {
    pub frame: Option<EgressItem>,
    pub eos: Option<EgressItem>,
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
    use pyo3::types::{PyAnyMethods, PyDictMethods, PyListMethods};
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

    #[test]
    fn test_egress_item_py_getters_setters() -> Result<()> {
        Python::attach(|py| {
            let frame = gen_frame();
            let data = PyList::empty(py).unbind();
            let labels = PyList::empty(py).unbind();
            let state = PyDict::new(py).unbind();

            let mut py_item = EgressItemPy {
                video_frame: frame.clone(),
                state,
                data,
                labels,
            };

            // Test getter returns the same frame
            let got_frame = py_item.get_video_frame();
            assert_eq!(got_frame.0.get_uuid(), frame.0.get_uuid());

            // Test state getter/setter
            let new_state = PyDict::new(py);
            new_state.set_item("key", "value")?;
            py_item.set_state(new_state.unbind());
            let got_state = py_item.get_state(py);
            let got_state_bound = got_state.bind(py);
            let val: String = got_state_bound.get_item("key")?.unwrap().extract()?;
            assert_eq!(val, "value");

            // Test data getter/setter
            let new_data = PyList::empty(py);
            new_data.append(42)?;
            py_item.set_data(new_data.unbind());
            let got_data = py_item.get_data(py);
            let got_data_bound = got_data.bind(py);
            assert_eq!(got_data_bound.len(), 1);

            // Test labels getter/setter
            let new_labels = PyList::empty(py);
            new_labels.append("label1")?;
            new_labels.append("label2")?;
            py_item.set_labels(new_labels.unbind());
            let got_labels = py_item.get_labels(py);
            let got_labels_bound = got_labels.bind(py);
            assert_eq!(got_labels_bound.len(), 2);

            // Test setter for video_frame
            let new_frame = gen_frame();
            let new_uuid = new_frame.0.get_uuid();
            py_item.set_video_frame(new_frame);
            assert_eq!(py_item.get_video_frame().0.get_uuid(), new_uuid);

            Ok::<(), anyhow::Error>(())
        })
    }

    #[test]
    fn test_egress_item_to_py_roundtrip() -> Result<()> {
        Python::attach(|py| {
            let frame = gen_frame();
            let data = PyList::empty(py).unbind();
            let labels = PyList::empty(py).unbind();
            let mut item = EgressItem::new_frame(frame.clone(), data, labels);

            // Convert to py
            let py_item = item.to_py(py)?;
            let py_item_bound = py_item.bind(py);

            // Modify through Python interface
            let new_state = PyDict::new(py);
            new_state.set_item("merged", true)?;
            py_item_bound.borrow_mut().set_state(new_state.unbind());

            // Update from py
            let py_ref = py_item_bound.borrow();
            item.update_from_py(&py_ref);

            // Verify the state was updated
            let state_bound = item.state.bind(py);
            let merged: bool = state_bound.get_item("merged")?.unwrap().extract()?;
            assert!(merged);

            // Verify frame is preserved
            assert!(matches!(item.message, EgressMessage::VideoFrame(_)));

            // EOS should fail to_py
            let eos = EgressItem::new_eos(PyList::empty(py).unbind(), PyList::empty(py).unbind());
            assert!(eos.to_py(py).is_err());

            Ok::<(), anyhow::Error>(())
        })
    }
}
