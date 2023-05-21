use crate::primitives::message::video::frame::InnerVideoFrame;
use crate::primitives::VideoFrame;
use crate::utils::python::no_gil;
use pyo3::{pyclass, pymethods};
use rkyv::{with::Skip, Archive, Deserialize, Serialize};
use std::collections::HashMap;

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, Clone, Default)]
#[archive(check_bytes)]
pub struct VideoFrameBatch {
    offline_frames: HashMap<i64, InnerVideoFrame>,
    #[with(Skip)]
    pub(crate) frames: HashMap<i64, VideoFrame>,
}

impl VideoFrameBatch {
    pub(crate) fn prepare_after_load(&mut self) {
        let offline_frames = std::mem::take(&mut self.offline_frames);
        for (id, mut frame) in offline_frames.into_iter() {
            frame.prepare_after_load();
            self.frames.insert(id, VideoFrame::from_inner(frame));
        }
    }

    pub(crate) fn prepare_before_save(&mut self) {
        for (id, frame) in self.frames.iter_mut() {
            let mut frame = frame.inner.lock().unwrap();
            frame.prepare_before_save();
            self.offline_frames.insert(*id, frame.as_ref().clone());
        }
    }

    pub fn snapshot(&mut self) {
        self.prepare_before_save();
    }

    pub fn restore(&mut self) {
        self.prepare_after_load();
    }
}

#[pymethods]
impl VideoFrameBatch {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, id: i64, frame: VideoFrame) {
        self.frames.insert(id, frame);
    }

    pub fn get(&self, id: i64) -> Option<VideoFrame> {
        self.frames.get(&id).cloned()
    }

    pub fn del(&mut self, id: i64) -> Option<VideoFrame> {
        self.frames.remove(&id)
    }

    #[pyo3(name = "snapshot")]
    pub fn snapshot_py(&mut self) {
        no_gil(|| self.snapshot())
    }

    #[pyo3(name = "restore")]
    pub fn restore_py(&mut self) {
        no_gil(|| self.restore())
    }
}
