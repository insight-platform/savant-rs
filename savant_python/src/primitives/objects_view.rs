use crate::match_query::MatchQuery;
use crate::primitives::object::VideoObject;
use crate::release_gil;
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use savant_core::match_query::*;
use savant_core::primitives::rust::VideoObjectProxy;
use std::collections::HashMap;
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

impl From<VideoObjectBBoxType> for savant_core::primitives::object::VideoObjectBBoxType {
    fn from(value: VideoObjectBBoxType) -> Self {
        match value {
            VideoObjectBBoxType::Detection => {
                savant_core::primitives::object::VideoObjectBBoxType::Detection
            }
            VideoObjectBBoxType::TrackingInfo => {
                savant_core::primitives::object::VideoObjectBBoxType::TrackingInfo
            }
        }
    }
}

impl From<savant_core::primitives::object::VideoObjectBBoxType> for VideoObjectBBoxType {
    fn from(value: savant_core::primitives::object::VideoObjectBBoxType) -> Self {
        match value {
            savant_core::primitives::object::VideoObjectBBoxType::Detection => {
                VideoObjectBBoxType::Detection
            }
            savant_core::primitives::object::VideoObjectBBoxType::TrackingInfo => {
                VideoObjectBBoxType::TrackingInfo
            }
        }
    }
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

impl From<Vec<VideoObjectProxy>> for VideoObjectsView {
    fn from(value: Vec<VideoObjectProxy>) -> Self {
        VideoObjectsView {
            inner: Arc::new(value.into_iter().map(VideoObject).collect()),
        }
    }
}

#[pymethods]
impl VideoObjectsView {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
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

    #[getter]
    #[pyo3(name = "ids")]
    fn ids_py(&self) -> Vec<i64> {
        self.inner.iter().map(|x| x.get_id()).collect()
    }

    #[pyo3(name = "detached_copy")]
    #[pyo3(signature = (no_gil = true))]
    fn detached_copy_gil(&self, no_gil: bool) -> VideoObjectsView {
        release_gil!(no_gil, || VideoObjectsView {
            inner: Arc::new(self.inner.iter().map(|x| x.detached_copy()).collect()),
        })
    }

    // fn rotated_boxes_as_numpy(&self, kind: &VideoObjectBBoxType) -> Py<PyArray<f64, IxDyn>> {
    //     let boxes = match kind {
    //         VideoObjectBBoxType::Detection => self
    //             .inner
    //             .iter()
    //             .map(|x| x.get_detection_box())
    //             .collect::<Vec<_>>(),
    //         VideoObjectBBoxType::TrackingInfo => self
    //             .inner
    //             .iter()
    //             .flat_map(|o| o.get_track_box().or(Some(BBOX_UNDEFINED.clone())))
    //             .collect::<Vec<_>>(),
    //     };
    //     rotated_bboxes_to_ndarray(boxes)
    // }

    #[getter]
    #[pyo3(name = "track_ids")]
    pub fn track_ids_py(&self) -> Vec<Option<i64>> {
        self.inner
            .iter()
            .map(|o| o.get_track_id())
            .collect::<Vec<_>>()
    }

    #[pyo3(name = "sorted_by_id")]
    pub fn sorted_by_id_py(&self) -> VideoObjectsView {
        let mut objects = self.inner.as_ref().clone();
        objects.sort_by_key(|o| o.get_id());
        VideoObjectsView {
            inner: Arc::new(objects),
        }
    }

    /// Ensures object box coordinates reside within the frame and not negative
    ///
    /// GIL is optionally released during the call
    ///
    /// Params
    /// ------
    /// width: float
    ///   Frame width
    /// height: float
    ///   Frame height
    /// box_type: :py:class:`VideoObjectBBoxType`
    ///   Type of the box to check (detection box or track box)
    /// no_gil: bool
    ///   Whether to release GIL during the call (true means release)
    ///
    /// Returns
    /// -------
    /// None
    ///   Raises exception if any of the boxes is out of frame
    ///
    /// Raises
    /// ------
    /// ValueError
    ///   If any of the boxes is out of frame
    ///
    #[pyo3(name = "check_frame_fit")]
    #[pyo3(signature = (width, height, box_type, no_gil = true))]
    pub fn check_frame_fit_gil(
        &self,
        width: f32,
        height: f32,
        box_type: VideoObjectBBoxType,
        no_gil: bool,
    ) -> PyResult<()> {
        release_gil!(no_gil, || {
            let objs = self.inner.iter().map(|o| o.0.clone()).collect::<Vec<_>>();
            savant_core::primitives::frame::VideoFrameProxy::check_frame_fit(
                &objs,
                width,
                height,
                box_type.into(),
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    // #[pyo3(name = "update_from_numpy_boxes")]
    // pub fn update_from_numpy_boxes_py(
    //     &mut self,
    //     np_boxes: PyReadonlyArrayDyn<f64>,
    //     format: &BBoxFormat,
    //     kind: &VideoObjectBBoxType,
    // ) {
    //     let boxes = ndarray_to_bboxes(&np_boxes, format)
    //         .into_iter()
    //         .map(|x| x.inner)
    //         .collect::<Vec<_>>();
    //
    //     self.fill_boxes(boxes, kind);
    // }

    // #[pyo3(name = "update_from_numpy_rotated_boxes")]
    // pub fn update_from_numpy_rotated_boxes_py(
    //     &mut self,
    //     np_boxes: PyReadonlyArrayDyn<f64>,
    //     kind: &VideoObjectBBoxType,
    // ) {
    //     let boxes = ndarray_to_rotated_bboxes(&np_boxes);
    //     self.fill_boxes(boxes, kind);
    // }
}

#[pyclass]
#[derive(Clone, Debug)]
pub(crate) struct QueryFunctions;

#[pymethods]
impl QueryFunctions {
    #[staticmethod]
    #[pyo3(name = "filter")]
    #[pyo3(signature = (v, q, no_gil = true))]
    pub(crate) fn filter_gil(
        v: VideoObjectsView,
        q: &MatchQuery,
        no_gil: bool,
    ) -> VideoObjectsView {
        release_gil!(no_gil, || {
            let objs = v.inner.iter().map(|o| o.0.clone()).collect::<Vec<_>>();
            VideoObjectsView::from(filter(&objs, &q.0))
        })
    }

    // #[staticmethod]
    // #[pyo3(name = "batch_filter")]
    // #[pyo3(signature = (v, q, no_gil = true))]
    // pub(crate) fn batch_filter_gil(
    //     v: VideoObjectsViewBatch,
    //     q: &MatchQueryProxy,
    //     no_gil: bool,
    // ) -> VideoObjectsViewBatch {
    //     release_gil!(no_gil, || {
    //         let m = v
    //             .iter()
    //             .map(|(id, v)| (*id, v.inner.to_vec()))
    //             .collect::<HashMap<_, _>>();
    //         batch_filter(&m, &q.inner)
    //             .into_iter()
    //             .map(|(id, v)| (id, VideoObjectsView { inner: Arc::new(v) }))
    //             .collect::<VideoObjectsViewBatch>()
    //     })
    // }

    #[staticmethod]
    #[pyo3(name = "partition")]
    #[pyo3(signature = (v, q, no_gil = true))]
    pub(crate) fn partition_gil(
        v: VideoObjectsView,
        q: &MatchQuery,
        no_gil: bool,
    ) -> (VideoObjectsView, VideoObjectsView) {
        release_gil!(no_gil, || {
            let objs = v.inner.iter().map(|o| o.0.clone()).collect::<Vec<_>>();
            let (a, b) = partition(&objs, &q.0);
            (a.into(), b.into())
        })
    }

    // #[staticmethod]
    // #[pyo3(name = "batch_partition")]
    // #[pyo3(signature = (v, q, no_gil = true))]
    // pub(crate) fn batch_partition_gil(
    //     v: VideoObjectsViewBatch,
    //     q: &MatchQueryProxy,
    //     no_gil: bool,
    // ) -> (VideoObjectsViewBatch, VideoObjectsViewBatch) {
    //     release_gil!(no_gil, || {
    //         let m = v
    //             .iter()
    //             .map(|(id, v)| (*id, v.inner.to_vec()))
    //             .collect::<HashMap<_, _>>();
    //         let (a, b) = batch_partition(m, &q.inner);
    //
    //         (
    //             a.into_iter()
    //                 .map(|(id, v)| (id, VideoObjectsView { inner: Arc::new(v) }))
    //                 .collect(),
    //             b.into_iter()
    //                 .map(|(id, v)| (id, VideoObjectsView { inner: Arc::new(v) }))
    //                 .collect(),
    //         )
    //     })
    // }

    #[staticmethod]
    #[pyo3(name = "map_udf")]
    #[pyo3(signature = (v, udf, no_gil = true))]
    pub(crate) fn map_udf_gil(
        v: &VideoObjectsView,
        udf: String,
        no_gil: bool,
    ) -> PyResult<VideoObjectsView> {
        let res = release_gil!(no_gil, || {
            let objs = v.inner.iter().map(|o| o.0.clone()).collect::<Vec<_>>();
            map_udf(&objs, &udf)
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(res.into())
    }

    // #[staticmethod]
    // #[pyo3(name = "batch_map_udf")]
    // #[pyo3(signature = (v, udf, no_gil = true))]
    // pub(crate) fn batch_map_udf_gil(
    //     v: VideoObjectsViewBatch,
    //     udf: String,
    //     no_gil: bool,
    // ) -> PyResult<VideoObjectsViewBatch> {
    //     release_gil!(no_gil, || {
    //         let m = v
    //             .iter()
    //             .map(|(id, v)| (*id, v.inner.to_vec()))
    //             .collect::<HashMap<_, _>>();
    //
    //         Ok(batch_map_udf(&m, &udf)
    //             .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    //             .into_iter()
    //             .map(|(id, v)| (id, VideoObjectsView { inner: Arc::new(v) }))
    //             .collect())
    //     })
    // }

    #[staticmethod]
    #[pyo3(name = "foreach_udf")]
    #[pyo3(signature = (v, udf, no_gil = true))]
    pub(crate) fn foreach_udf_gil(v: &VideoObjectsView, udf: String, no_gil: bool) -> PyResult<()> {
        release_gil!(no_gil, || {
            let objs = v.inner.iter().map(|o| o.0.clone()).collect::<Vec<_>>();
            let res = foreach_udf(&objs, &udf);

            for r in res {
                r.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            }

            Ok(())
        })
    }

    // #[staticmethod]
    // #[pyo3(name = "batch_foreach_udf")]
    // #[pyo3(signature = (v, udf, no_gil = true))]
    // pub(crate) fn batch_foreach_udf_gil(
    //     v: VideoObjectsViewBatch,
    //     udf: String,
    //     no_gil: bool,
    // ) -> PyResult<()> {
    //     release_gil!(no_gil, || {
    //         let m = v
    //             .iter()
    //             .map(|(id, v)| (*id, v.inner.to_vec()))
    //             .collect::<HashMap<_, _>>();
    //
    //         let res = batch_foreach_udf(&m, &udf);
    //         for (i, r) in res {
    //             for e in r {
    //                 e.map_err(|e| {
    //                     PyRuntimeError::new_err(format!("Batch frame Id={}, Error: {}", i, e))
    //                 })?;
    //             }
    //         }
    //         Ok(())
    //     })
    // }
}
