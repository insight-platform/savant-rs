use crate::primitives::bbox::BBOX_UNDEFINED;
use crate::primitives::message::video::query::match_query::{
    batch_filter, batch_foreach_udf, batch_map_udf, batch_partition, filter, foreach_udf, map_udf,
    partition,
};
use crate::primitives::message::video::query::py::MatchQueryProxy;
use crate::primitives::{RBBox, VideoObjectProxy};
use crate::release_gil;
use crate::utils::{
    ndarray_to_bboxes, ndarray_to_rotated_bboxes, rotated_bboxes_to_ndarray, BBoxFormat,
};
use ndarray::IxDyn;
use numpy::{PyArray, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyIndexError, PyRuntimeError};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::iter::zip;
use std::sync::Arc;

pub type VideoObjectsViewBatch = HashMap<i64, VideoObjectsView>;

/// Determines which object bbox is a subject of the operation
///
#[pyclass]
#[derive(Clone, Debug, Copy)]
#[repr(C)]
pub enum VideoObjectBBoxType {
    Detection,
    TrackingInfo,
}

#[pyclass]
#[derive(Clone, Debug)]
#[repr(C)]
pub struct VideoObjectsView {
    pub(crate) inner: Arc<Vec<VideoObjectProxy>>,
}

impl From<Vec<VideoObjectProxy>> for VideoObjectsView {
    fn from(value: Vec<VideoObjectProxy>) -> Self {
        VideoObjectsView {
            inner: Arc::new(value),
        }
    }
}

impl VideoObjectsView {
    fn fill_boxes_gil(&self, boxes: Vec<RBBox>, kind: &VideoObjectBBoxType) {
        let it = zip(self.inner.iter(), boxes);
        match kind {
            VideoObjectBBoxType::Detection => it.for_each(|(o, b)| o.set_detection_bbox(b)),
            VideoObjectBBoxType::TrackingInfo => it.for_each(|(o, b)| o.set_track_box(b)),
        }
    }
}

#[pymethods]
impl VideoObjectsView {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:#?}", self.inner)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __getitem__(&self, index: usize) -> PyResult<VideoObjectProxy> {
        self.inner
            .get(index)
            .ok_or(PyIndexError::new_err("index out of range"))
            .map(|x| x.clone())
    }

    #[getter]
    fn memory_handle(&self) -> usize {
        self as *const Self as usize
    }

    fn __len__(&self) -> PyResult<usize> {
        Ok(self.inner.len())
    }

    #[getter]
    #[pyo3(name = "ids")]
    fn ids_py(&self) -> Vec<i64> {
        self.inner.iter().map(|x| x.get_id()).collect()
    }

    #[pyo3(name = "detached_copy")]
    fn detached_copy_py(&self) -> VideoObjectsView {
        release_gil!(|| VideoObjectsView {
            inner: Arc::new(self.inner.iter().map(|x| x.detached_copy()).collect()),
        })
    }

    #[pyo3(name = "rotated_boxes_as_numpy")]
    fn rotated_boxes_as_numpy_gil(&self, kind: &VideoObjectBBoxType) -> Py<PyArray<f64, IxDyn>> {
        let boxes = match kind {
            VideoObjectBBoxType::Detection => self
                .inner
                .iter()
                .map(|x| x.get_detection_box())
                .collect::<Vec<_>>(),
            VideoObjectBBoxType::TrackingInfo => self
                .inner
                .iter()
                .flat_map(|o| o.get_track_box().or(Some(BBOX_UNDEFINED.clone())))
                .collect::<Vec<_>>(),
        };
        rotated_bboxes_to_ndarray(boxes)
    }

    #[getter]
    #[pyo3(name = "track_ids")]
    pub fn track_ids_gil(&self) -> Vec<Option<i64>> {
        self.inner
            .iter()
            .map(|o| o.get_track_id())
            .collect::<Vec<_>>()
    }

    #[pyo3(name = "sorted_by_id")]
    pub fn sorted_by_id_gil(&self) -> VideoObjectsView {
        let mut objects = self.inner.as_ref().clone();
        objects.sort_by_key(|o| o.get_id());
        VideoObjectsView {
            inner: Arc::new(objects),
        }
    }

    #[pyo3(name = "update_from_numpy_boxes")]
    pub fn update_from_numpy_boxes_gil(
        &mut self,
        np_boxes: PyReadonlyArrayDyn<f64>,
        format: &BBoxFormat,
        kind: &VideoObjectBBoxType,
    ) {
        let boxes = ndarray_to_bboxes(&np_boxes, format)
            .into_iter()
            .map(|x| x.inner)
            .collect::<Vec<_>>();

        self.fill_boxes_gil(boxes, kind);
    }

    #[pyo3(name = "update_from_numpy_rotated_boxes")]
    pub fn update_from_numpy_rotated_boxes_gil(
        &mut self,
        np_boxes: PyReadonlyArrayDyn<f64>,
        kind: &VideoObjectBBoxType,
    ) {
        let boxes = ndarray_to_rotated_bboxes(&np_boxes);
        self.fill_boxes_gil(boxes, kind);
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub(crate) struct QueryFunctions;

#[pymethods]
impl QueryFunctions {
    #[staticmethod]
    #[pyo3(name = "filter")]
    pub(crate) fn filter_gil(v: &VideoObjectsView, q: &MatchQueryProxy) -> VideoObjectsView {
        release_gil!(|| VideoObjectsView {
            inner: Arc::new(filter(v.inner.as_ref(), &q.inner)),
        })
    }

    #[staticmethod]
    #[pyo3(name = "batch_filter")]
    pub(crate) fn batch_filter_gil(
        v: VideoObjectsViewBatch,
        q: &MatchQueryProxy,
    ) -> VideoObjectsViewBatch {
        release_gil!(|| {
            let m = v
                .iter()
                .map(|(id, v)| (*id, v.inner.to_vec()))
                .collect::<HashMap<_, _>>();
            batch_filter(&m, &q.inner)
                .into_iter()
                .map(|(id, v)| (id, VideoObjectsView { inner: Arc::new(v) }))
                .collect()
        })
    }

    #[staticmethod]
    #[pyo3(name = "partition")]
    pub(crate) fn partition_gil(
        v: &VideoObjectsView,
        q: &MatchQueryProxy,
    ) -> (VideoObjectsView, VideoObjectsView) {
        release_gil!(|| {
            let (a, b) = partition(v.inner.as_ref(), &q.inner);
            (
                VideoObjectsView { inner: Arc::new(a) },
                VideoObjectsView { inner: Arc::new(b) },
            )
        })
    }

    #[staticmethod]
    #[pyo3(name = "batch_partition")]
    pub(crate) fn batch_partition_gil(
        v: VideoObjectsViewBatch,
        q: &MatchQueryProxy,
    ) -> (VideoObjectsViewBatch, VideoObjectsViewBatch) {
        release_gil!(|| {
            let m = v
                .iter()
                .map(|(id, v)| (*id, v.inner.to_vec()))
                .collect::<HashMap<_, _>>();
            let (a, b) = batch_partition(m, &q.inner);

            (
                a.into_iter()
                    .map(|(id, v)| (id, VideoObjectsView { inner: Arc::new(v) }))
                    .collect(),
                b.into_iter()
                    .map(|(id, v)| (id, VideoObjectsView { inner: Arc::new(v) }))
                    .collect(),
            )
        })
    }

    #[staticmethod]
    #[pyo3(name = "map_udf")]
    pub(crate) fn map_udf_gil(v: &VideoObjectsView, udf: String) -> PyResult<VideoObjectsView> {
        release_gil!(|| {
            map_udf(v.inner.as_ref().iter().collect::<Vec<_>>().as_slice(), &udf)
                .map(|x| VideoObjectsView { inner: Arc::new(x) })
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    #[staticmethod]
    #[pyo3(name = "batch_map_udf")]
    pub(crate) fn batch_map_udf_gil(
        v: VideoObjectsViewBatch,
        udf: String,
    ) -> PyResult<VideoObjectsViewBatch> {
        release_gil!(|| {
            let m = v
                .iter()
                .map(|(id, v)| (*id, v.inner.to_vec()))
                .collect::<HashMap<_, _>>();

            Ok(batch_map_udf(&m, &udf)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .into_iter()
                .map(|(id, v)| (id, VideoObjectsView { inner: Arc::new(v) }))
                .collect())
        })
    }

    #[staticmethod]
    #[pyo3(name = "foreach_udf")]
    pub(crate) fn foreach_udf_gil(v: &VideoObjectsView, udf: String) -> PyResult<()> {
        release_gil!(|| {
            let res = foreach_udf(v.inner.as_ref().iter().collect::<Vec<_>>().as_slice(), &udf);

            for r in res {
                r.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            }
            Ok(())
        })
    }

    #[staticmethod]
    #[pyo3(name = "batch_foreach_udf")]
    pub(crate) fn batch_foreach_udf_gil(v: VideoObjectsViewBatch, udf: String) -> PyResult<()> {
        release_gil!(|| {
            let m = v
                .iter()
                .map(|(id, v)| (*id, v.inner.to_vec()))
                .collect::<HashMap<_, _>>();

            let res = batch_foreach_udf(&m, &udf);
            for (i, r) in res {
                for e in r {
                    e.map_err(|e| {
                        PyRuntimeError::new_err(format!("Batch frame Id={}, Error: {}", i, e))
                    })?;
                }
            }
            Ok(())
        })
    }
}
