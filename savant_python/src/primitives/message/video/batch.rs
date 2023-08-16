use crate::primitives::message::video::frame::VideoFrame;
use crate::primitives::message::video::match_query::MatchQuery;
use crate::primitives::message::video::object::VideoObject;
use crate::primitives::message::video::objects_view::VideoObjectsView;
use crate::release_gil;
use pyo3::{pyclass, pymethods};
use savant_core::primitives::rust;
use std::collections::HashMap;

#[pyclass]
pub struct VideoFrameBatch(pub(crate) rust::VideoFrameBatch);

impl Default for VideoFrameBatch {
    fn default() -> Self {
        Self(rust::VideoFrameBatch::new())
    }
}

#[pymethods]
impl VideoFrameBatch {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, id: i64, frame: VideoFrame) {
        self.0.add(id, frame.0);
    }

    pub fn get(&self, id: i64) -> Option<VideoFrame> {
        self.0.get(id).map(|x| VideoFrame(x.clone()))
    }

    pub fn del(&mut self, id: i64) -> Option<VideoFrame> {
        self.0.del(id).map(|x| VideoFrame(x.clone()))
    }

    #[pyo3(name = "snapshot")]
    #[pyo3(signature = (no_gil = true))]
    pub fn snapshot_gil(&mut self, no_gil: bool) {
        release_gil!(no_gil, || self.0.snapshot())
    }

    #[pyo3(name = "restore")]
    #[pyo3(signature = (no_gil = true))]
    pub fn restore_gil(&mut self, no_gil: bool) {
        release_gil!(no_gil, || self.0.restore())
    }

    #[pyo3(name = "access_objects")]
    #[pyo3(signature = (q, no_gil = true))]
    pub fn access_objects_gil(
        &self,
        q: MatchQuery,
        no_gil: bool,
    ) -> HashMap<i64, VideoObjectsView> {
        release_gil!(no_gil, || {
            self.0
                .access_objects(&q.0)
                .into_iter()
                .map(|(id, x)| {
                    (
                        id,
                        x.into_iter().map(VideoObject).collect::<Vec<_>>().into(),
                    )
                })
                .collect::<HashMap<_, _>>()
        })
    }

    #[pyo3(name = "delete_objects")]
    #[pyo3(signature = (q, no_gil = true))]
    pub fn delete_objects_gil(&mut self, q: MatchQuery, no_gil: bool) {
        release_gil!(no_gil, || self.0.delete_objects(&q.0))
    }
}