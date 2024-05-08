use crate::match_query::MatchQuery;
use crate::primitives::frame::VideoFrame;
use crate::primitives::object::BorrowedVideoObject;
use crate::primitives::objects_view::VideoObjectsView;
use crate::{release_gil, with_gil};
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::{PyBytes, PyBytesMethods};
use pyo3::{pyclass, pymethods, Bound, PyObject, PyResult};
use savant_core::primitives::rust;
use savant_core::protobuf::{from_pb, ToProtobuf};
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

    pub fn del_(&mut self, id: i64) -> Option<VideoFrame> {
        self.0.del(id).map(|x| VideoFrame(x.clone()))
    }

    #[pyo3(name = "access_objects")]
    #[pyo3(signature = (q, no_gil = true))]
    pub fn access_objects_gil(
        &self,
        q: &MatchQuery,
        no_gil: bool,
    ) -> HashMap<i64, VideoObjectsView> {
        release_gil!(no_gil, || {
            self.0
                .access_objects(&q.0)
                .into_iter()
                .map(|(id, x)| {
                    (
                        id,
                        x.into_iter()
                            .map(BorrowedVideoObject)
                            .collect::<Vec<_>>()
                            .into(),
                    )
                })
                .collect::<HashMap<_, _>>()
        })
    }

    #[pyo3(name = "delete_objects")]
    #[pyo3(signature = (q, no_gil = true))]
    pub fn delete_objects_gil(&mut self, q: &MatchQuery, no_gil: bool) {
        release_gil!(no_gil, || self.0.delete_objects(&q.0))
    }

    #[pyo3(name = "to_protobuf")]
    #[pyo3(signature = (no_gil = true))]
    fn to_protobuf_gil(&self, no_gil: bool) -> PyResult<PyObject> {
        let bytes = release_gil!(no_gil, || {
            self.0.to_pb().map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "Failed to serialize video frame batch to protobuf: {}",
                    e
                ))
            })
        })?;
        with_gil!(|py| {
            let bytes = PyBytes::new_bound(py, &bytes);
            Ok(PyObject::from(bytes))
        })
    }

    #[staticmethod]
    #[pyo3(name = "from_protobuf")]
    #[pyo3(signature = (bytes, no_gil = true))]
    fn from_protobuf_gil(bytes: &Bound<'_, PyBytes>, no_gil: bool) -> PyResult<Self> {
        let bytes = bytes.as_bytes();
        release_gil!(no_gil, || {
            let obj =
                from_pb::<savant_core::protobuf::VideoFrameBatch, rust::VideoFrameBatch>(bytes)
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!(
                            "Failed to deserialize video frame batch from protobuf: {}",
                            e
                        ))
                    })?;
            Ok(Self(obj))
        })
    }

    fn ids(&self) -> Vec<i64> {
        self.0.frames().keys().copied().collect()
    }

    fn frames(&self) -> Vec<VideoFrame> {
        self.0.frames().values().map(|x| VideoFrame(x.clone())).collect()
    }
}
