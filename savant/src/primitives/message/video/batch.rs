use crate::primitives::message::video::frame::VideoFrame;
use crate::primitives::message::video::query::py::QueryProxy;
use crate::primitives::message::video::query::Query;
use crate::primitives::{VideoFrameProxy, VideoObjectProxy, VideoObjectsView};
use crate::utils::python::release_gil;
use pyo3::{pyclass, pymethods};
use rayon::prelude::*;
use rkyv::{with::Skip, Archive, Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::Deref;

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, Clone, Default)]
#[archive(check_bytes)]
pub struct VideoFrameBatch {
    offline_frames: HashMap<i64, VideoFrame>,
    #[with(Skip)]
    pub(crate) frames: HashMap<i64, VideoFrameProxy>,
}

impl VideoFrameBatch {
    pub(crate) fn prepare_after_load(&mut self) {
        let offline_frames = std::mem::take(&mut self.offline_frames);
        for (id, inner) in offline_frames.into_iter() {
            let frame = VideoFrameProxy::from_inner(inner);
            frame.restore_from_snapshot();
            self.frames.insert(id, frame);
        }
    }

    pub(crate) fn prepare_before_save(&mut self) {
        self.offline_frames.clear();
        for (id, frame) in self.frames.iter_mut() {
            frame.make_snapshot();
            let inner = frame.inner.read_recursive();
            self.offline_frames.insert(*id, inner.as_ref().clone());
        }
    }

    pub fn snapshot(&mut self) {
        self.prepare_before_save();
    }

    pub fn restore(&mut self) {
        self.prepare_after_load();
    }

    pub fn access_objects(&self, q: &Query) -> HashMap<i64, Vec<VideoObjectProxy>> {
        self.frames
            .par_iter()
            .map(|(id, frame)| (*id, frame.access_objects(q)))
            .collect()
    }

    pub fn delete_objects(&mut self, q: &Query) {
        self.frames.par_iter_mut().for_each(|(_, frame)| {
            frame.delete_objects(q);
        });
    }
}

#[pymethods]
impl VideoFrameBatch {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, id: i64, frame: VideoFrameProxy) {
        self.frames.insert(id, frame);
    }

    pub fn get(&self, id: i64) -> Option<VideoFrameProxy> {
        self.frames.get(&id).cloned()
    }

    pub fn del(&mut self, id: i64) -> Option<VideoFrameProxy> {
        self.frames.remove(&id)
    }

    #[pyo3(name = "snapshot")]
    pub fn snapshot_gil(&mut self) {
        release_gil(|| self.snapshot())
    }

    #[pyo3(name = "restore")]
    pub fn restore_gil(&mut self) {
        release_gil(|| self.restore())
    }

    #[pyo3(name = "access_objects")]
    pub fn access_objects_gil(&self, q: QueryProxy) -> HashMap<i64, VideoObjectsView> {
        release_gil(|| {
            self.access_objects(q.inner.deref())
                .into_iter()
                .map(|(id, x)| (id, x.into()))
                .collect()
        })
    }

    #[pyo3(name = "delete_objects")]
    pub fn delete_objects_gil(&mut self, q: QueryProxy) {
        release_gil(|| self.delete_objects(q.inner.deref()))
    }
}
