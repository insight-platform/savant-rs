use crate::primitives::message::video::object::VideoObject;
use crate::primitives::proxy::{StrongInnerType, UpgradeableWeakInner, WeakInner};
use crate::primitives::{PaddingDraw, PolygonalArea, PythonBBox, RBBox, VideoObjectBBoxType};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pyclass::CompareOp;

/// See the documentation of :py:class:`savant_rs.primitives.geometry.RBBox` for meaning of the properties and methods.
///
#[pyclass]
#[derive(Clone, Debug)]
pub struct VideoObjectRBBoxProxy {
    object: WeakInner<VideoObject>,
    kind: VideoObjectBBoxType,
}

impl VideoObjectRBBoxProxy {
    pub fn new(object: StrongInnerType<VideoObject>, kind: VideoObjectBBoxType) -> Self {
        Self {
            object: WeakInner::new(object),
            kind,
        }
    }
}

impl VideoObjectRBBoxProxy {
    fn get_object(&self) -> StrongInnerType<VideoObject> {
        self.object.get_or_fail()
    }
}

#[pymethods]
impl VideoObjectRBBoxProxy {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[getter]
    fn get_area(&self) -> f64 {
        let kind = self.kind;
        self.get_object().read().bbox_ref(kind).get_area()
    }

    #[pyo3(name = "copy")]
    fn copy_py(&self) -> RBBox {
        let kind = self.kind;
        self.get_object().read().bbox_ref(kind).copy_py()
    }

    fn new_padded(&self, padding: &PaddingDraw) -> RBBox {
        let kind = self.kind;
        self.get_object().read().bbox_ref(kind).new_padded(padding)
    }

    fn is_modified(&self) -> bool {
        let kind = self.kind;
        self.get_object().read().bbox_ref(kind).is_modified()
    }

    #[pyo3(name = "visual_box")]
    pub fn get_visual_box_gil(&self, padding: &PaddingDraw, border_width: i64) -> RBBox {
        let kind = self.kind;
        self.get_object()
            .read()
            .bbox_ref(kind)
            .get_visual_box_gil(padding, border_width)
    }

    /// Access the ratio between width and height.
    ///
    #[getter]
    pub fn get_width_to_height_ratio(&self) -> f64 {
        let kind = self.kind;
        self.get_object()
            .read()
            .bbox_ref(kind)
            .get_width_to_height_ratio()
    }

    #[getter]
    fn get_xc(&self) -> f64 {
        let kind = self.kind;
        self.get_object().read().bbox_ref(kind).get_xc()
    }

    #[setter]
    fn set_xc(&self, xc: f64) {
        let kind = self.kind;
        self.get_object().write().bbox_mut(kind).set_xc(xc);
    }

    #[getter]
    fn get_yc(&self) -> f64 {
        let kind = self.kind;
        self.get_object().read().bbox_ref(kind).get_yc()
    }

    #[setter]
    fn set_yc(&self, yc: f64) {
        let kind = self.kind;
        self.get_object().write().bbox_mut(kind).set_yc(yc);
    }

    #[getter]
    fn get_width(&self) -> f64 {
        let kind = self.kind;
        self.get_object().read().bbox_ref(kind).get_width()
    }

    #[setter]
    fn set_width(&self, width: f64) {
        let kind = self.kind;
        self.get_object().write().bbox_mut(kind).set_width(width);
    }

    #[getter]
    fn get_height(&self) -> f64 {
        let kind = self.kind;
        self.get_object().read().bbox_ref(kind).get_height()
    }

    #[setter]
    fn set_height(&self, height: f64) {
        let kind = self.kind;
        self.get_object().write().bbox_mut(kind).set_height(height);
    }

    #[getter]
    fn get_angle(&self) -> Option<f64> {
        let kind = self.kind;
        self.get_object().read().bbox_ref(kind).get_angle()
    }

    #[setter]
    fn set_angle(&self, angle: Option<f64>) {
        let kind = self.kind;
        self.get_object().write().bbox_mut(kind).set_angle(angle);
    }

    pub fn scale(&self, scale_x: f64, scale_y: f64) {
        let kind = self.kind;
        self.get_object()
            .write()
            .bbox_mut(kind)
            .scale_gil(scale_x, scale_y);
    }

    pub fn shift(&self, shift_x: f64, shift_y: f64) {
        let kind = self.kind;
        self.get_object()
            .write()
            .bbox_mut(kind)
            .shift(shift_x, shift_y);
    }

    #[getter]
    pub fn get_vertices(&self) -> Vec<(f64, f64)> {
        let kind = self.kind;
        self.get_object().read().bbox_ref(kind).get_vertices_gil()
    }

    #[getter]
    pub fn get_vertices_rounded(&self) -> Vec<(f64, f64)> {
        let kind = self.kind;
        self.get_object()
            .read()
            .bbox_ref(kind)
            .get_vertices_rounded_gil()
    }

    #[getter]
    pub fn get_vertices_int(&self) -> Vec<(i64, i64)> {
        let kind = self.kind;
        self.get_object()
            .read()
            .bbox_ref(kind)
            .get_vertices_int_gil()
    }

    #[getter]
    pub fn get_as_polygonal_area(&self) -> PolygonalArea {
        let kind = self.kind;
        self.get_object()
            .read()
            .bbox_ref(kind)
            .get_as_polygonal_area_gil()
    }

    #[getter]
    pub fn get_wrapping_box(&self) -> PythonBBox {
        let kind = self.kind;
        self.get_object()
            .read()
            .bbox_ref(kind)
            .get_wrapping_box_gil()
    }

    pub fn get_visual_box(&self, padding: &PaddingDraw, border_width: i64) -> RBBox {
        let kind = self.kind;
        self.get_object()
            .read()
            .bbox_ref(kind)
            .get_visual_box_gil(padding, border_width)
    }

    #[pyo3(name = "eq")]
    pub fn geometric_eq(&self, other: &PyAny) -> PyResult<bool> {
        let kind = self.kind;
        let ob1 = self.get_object();
        let br1 = ob1.read();
        let o1 = br1.bbox_ref(kind);

        if let Ok(other) = other.extract::<Self>() {
            let ob2 = other.get_object();
            let br2 = ob2.read();
            let o2 = br2.bbox_ref(kind);
            Ok(o1.geometric_eq(o2))
        } else if let Ok(other) = other.extract::<RBBox>() {
            Ok(o1.geometric_eq(&other))
        } else {
            Err(PyValueError::new_err(
                "Not a VideoObjectRBBoxProxy or RBBox",
            ))
        }
    }

    pub fn almost_eq(&self, other: &PyAny, eps: f64) -> PyResult<bool> {
        let kind = self.kind;
        let ob1 = self.get_object();
        let br1 = ob1.read();
        let o1 = br1.bbox_ref(kind);

        if let Ok(other) = other.extract::<Self>() {
            let ob2 = other.get_object();
            let br2 = ob2.read();
            let o2 = br2.bbox_ref(kind);
            Ok(o1.almost_eq(o2, eps))
        } else if let Ok(other) = other.extract::<RBBox>() {
            Ok(o1.almost_eq(&other, eps))
        } else {
            Err(PyValueError::new_err(
                "Not a VideoObjectRBBoxProxy or RBBox",
            ))
        }
    }

    pub fn iou(&self, other: &PyAny) -> PyResult<f64> {
        let kind = self.kind;
        let ob1 = self.get_object();
        let br1 = ob1.read();
        let o1 = br1.bbox_ref(kind);

        if let Ok(other) = other.extract::<Self>() {
            let ob2 = other.get_object();
            let br2 = ob2.read();
            let o2 = br2.bbox_ref(kind);
            o1.iou_gil(o2)
        } else if let Ok(other) = other.extract::<RBBox>() {
            o1.iou_gil(&other)
        } else {
            Err(PyValueError::new_err(
                "Not a VideoObjectRBBoxProxy or RBBox",
            ))
        }
    }

    pub fn ios(&self, other: &PyAny) -> PyResult<f64> {
        let kind = self.kind;
        let ob1 = self.get_object();
        let br1 = ob1.read();
        let o1 = br1.bbox_ref(kind);

        if let Ok(other) = other.extract::<Self>() {
            let ob2 = other.get_object();
            let br2 = ob2.read();
            let o2 = br2.bbox_ref(kind);
            o1.ios_gil(o2)
        } else if let Ok(other) = other.extract::<RBBox>() {
            o1.ios_gil(&other)
        } else {
            Err(PyValueError::new_err(
                "Not a VideoObjectRBBoxProxy or RBBox",
            ))
        }
    }

    pub fn ioo(&self, other: &PyAny) -> PyResult<f64> {
        let kind = self.kind;
        let ob1 = self.get_object();
        let br1 = ob1.read();
        let o1 = br1.bbox_ref(kind);

        if let Ok(other) = other.extract::<Self>() {
            let ob2 = other.get_object();
            let br2 = ob2.read();
            let o2 = br2.bbox_ref(kind);
            o1.ioo_gil(o2)
        } else if let Ok(other) = other.extract::<RBBox>() {
            o1.ioo_gil(&other)
        } else {
            Err(PyValueError::new_err(
                "Not a VideoObjectRBBoxProxy or RBBox",
            ))
        }
    }

    fn __richcmp__(&self, other: &PyAny, op: CompareOp) -> PyResult<bool> {
        let kind = self.kind;
        let ob1 = self.get_object();
        let br1 = ob1.read();
        let o1 = br1.bbox_ref(kind);
        if let Ok(other) = other.extract::<Self>() {
            let ob2 = other.get_object();
            let br2 = ob2.read();
            let o2 = br2.bbox_ref(kind);
            o1.__richcmp__(o2, op)
        } else if let Ok(other) = other.extract::<RBBox>() {
            o1.__richcmp__(&other, op)
        } else {
            Err(PyValueError::new_err(
                "Not a VideoObjectRBBoxProxy or RBBox",
            ))
        }
    }
}
