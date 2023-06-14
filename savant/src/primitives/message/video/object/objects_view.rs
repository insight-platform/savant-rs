use crate::capi::BBOX_UNDEFINED;
use crate::primitives::message::video::query::py::QueryWrapper;
use crate::primitives::message::video::query::{filter, foreach_udf, map_udf, partition};
use crate::primitives::{RBBox, VideoObject};
use crate::utils::python::no_gil;
use crate::utils::{
    ndarray_to_bboxes, ndarray_to_rotated_bboxes, rotated_bboxes_to_ndarray, BBoxFormat,
};
use ndarray::IxDyn;
use numpy::{PyArray, PyReadonlyArrayDyn};
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;
use std::iter::zip;
use std::sync::Arc;

/// Determines which object bbox is a subject of the operation
///
#[pyclass]
#[derive(Clone, Debug)]
pub enum VideoObjectBBoxKind {
    Detection,
    TrackingInfo,
}

#[pyclass]
#[derive(Clone, Debug)]
#[repr(C)]
pub struct VideoObjectsView {
    pub(crate) inner: Arc<Vec<VideoObject>>,
}

impl From<Vec<VideoObject>> for VideoObjectsView {
    fn from(value: Vec<VideoObject>) -> Self {
        VideoObjectsView {
            inner: Arc::new(value),
        }
    }
}

impl VideoObjectsView {
    fn fill_boxes_gil(&self, boxes: Vec<RBBox>, kind: VideoObjectBBoxKind) {
        no_gil(|| {
            let it = zip(self.inner.iter(), boxes);
            match kind {
                VideoObjectBBoxKind::Detection => it.for_each(|(o, b)| o.set_bbox(b)),
                VideoObjectBBoxKind::TrackingInfo => it.for_each(|(o, b)| o.update_track_bbox(b)),
            }
        })
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

    fn __getitem__(&self, index: usize) -> PyResult<VideoObject> {
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

    #[pyo3(name = "filter")]
    fn filter_gil(&self, q: QueryWrapper) -> VideoObjectsView {
        no_gil(|| VideoObjectsView {
            inner: Arc::new(filter(self.inner.as_ref(), &q.inner)),
        })
    }

    #[pyo3(name = "partition")]
    fn partition_gil(&self, q: QueryWrapper) -> (VideoObjectsView, VideoObjectsView) {
        no_gil(|| {
            let (a, b) = partition(self.inner.as_ref(), &q.inner);
            (
                VideoObjectsView { inner: Arc::new(a) },
                VideoObjectsView { inner: Arc::new(b) },
            )
        })
    }

    #[pyo3(name = "map_udf")]
    fn map_udf_gil(&self, udf: String) -> PyResult<VideoObjectsView> {
        no_gil(|| {
            map_udf(
                self.inner.as_ref().iter().collect::<Vec<_>>().as_slice(),
                &udf,
            )
            .map(|x| VideoObjectsView { inner: Arc::new(x) })
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    #[pyo3(name = "foreach_udf")]
    fn foreach_udf_gil(&self, udf: String) -> PyResult<Vec<()>> {
        no_gil(|| {
            foreach_udf(
                self.inner.as_ref().iter().collect::<Vec<_>>().as_slice(),
                &udf,
            )
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    #[getter]
    #[pyo3(name = "ids")]
    fn ids_py(&self) -> Vec<i64> {
        no_gil(|| self.inner.iter().map(|x| x.get_id()).collect())
    }

    #[pyo3(name = "detached_copy")]
    fn detached_copy_py(&self) -> VideoObjectsView {
        no_gil(|| VideoObjectsView {
            inner: Arc::new(self.inner.iter().map(|x| x.detached_copy()).collect()),
        })
    }

    #[pyo3(name = "rotated_boxes_as_numpy")]
    fn rotated_boxes_as_numpy_gil(&self, kind: VideoObjectBBoxKind) -> Py<PyArray<f64, IxDyn>> {
        let boxes = no_gil(|| match kind {
            VideoObjectBBoxKind::Detection => {
                self.inner.iter().map(|x| x.get_bbox()).collect::<Vec<_>>()
            }
            VideoObjectBBoxKind::TrackingInfo => self
                .inner
                .iter()
                .flat_map(|o| {
                    o.get_tracking_data()
                        .map(|t| t.bounding_box)
                        .or(Some(BBOX_UNDEFINED))
                })
                .collect::<Vec<_>>(),
        });
        rotated_bboxes_to_ndarray(boxes)
    }

    #[getter]
    #[pyo3(name = "track_ids")]
    pub fn track_ids_gil(&self) -> Vec<Option<i64>> {
        no_gil(|| {
            self.inner
                .iter()
                .map(|o| o.get_tracking_data().map(|t| t.id))
                .collect::<Vec<_>>()
        })
    }

    #[pyo3(name = "sorted_by_id")]
    pub fn sorted_by_id_gil(&self) -> VideoObjectsView {
        no_gil(|| {
            let mut objects = self.inner.as_ref().clone();
            objects.sort_by_key(|o| o.get_id());
            VideoObjectsView {
                inner: Arc::new(objects),
            }
        })
    }

    #[pyo3(name = "update_from_numpy_boxes")]
    pub fn update_from_numpy_boxes_gil(
        &mut self,
        np_boxes: PyReadonlyArrayDyn<f64>,
        format: BBoxFormat,
        kind: VideoObjectBBoxKind,
    ) {
        let boxes = ndarray_to_bboxes(np_boxes, format)
            .into_iter()
            .map(|x| x.inner)
            .collect::<Vec<_>>();

        self.fill_boxes_gil(boxes, kind);
    }

    #[pyo3(name = "update_from_numpy_rotated_boxes")]
    pub fn update_from_numpy_rotated_boxes_gil(
        &mut self,
        np_boxes: PyReadonlyArrayDyn<f64>,
        kind: VideoObjectBBoxKind,
    ) {
        let boxes = ndarray_to_rotated_bboxes(np_boxes);
        self.fill_boxes_gil(boxes, kind);
    }
}