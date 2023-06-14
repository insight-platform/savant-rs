use crate::primitives::message::video::object::InnerVideoObject;
use crate::primitives::proxy::{StrongInnerType, UpgradeableWeakInner, WeakInner};
use crate::primitives::{PolygonalArea, PythonBBox, VideoObjectBBoxKind};
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone, Debug)]
pub struct VideoObjectRBBoxProxy {
    object: WeakInner<InnerVideoObject>,
    kind: VideoObjectBBoxKind,
}

impl VideoObjectRBBoxProxy {
    pub fn new(object: StrongInnerType<InnerVideoObject>, kind: VideoObjectBBoxKind) -> Self {
        Self {
            object: WeakInner::new(object),
            kind,
        }
    }
}

impl VideoObjectRBBoxProxy {
    fn get_object(&self) -> StrongInnerType<InnerVideoObject> {
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
    fn get_xc(&self) -> f64 {
        let kind = self.kind.clone();
        self.get_object().read().bbox_ref(kind).xc
    }

    #[setter]
    fn set_xc(&self, xc: f64) {
        let kind = self.kind.clone();
        self.get_object().write().bbox_mut(kind).set_xc(xc);
    }

    #[getter]
    fn get_yc(&self) -> f64 {
        let kind = self.kind.clone();
        self.get_object().read().bbox_ref(kind).yc
    }

    #[setter]
    fn set_yc(&self, yc: f64) {
        let kind = self.kind.clone();
        self.get_object().write().bbox_mut(kind).set_yc(yc);
    }

    #[getter]
    fn get_width(&self) -> f64 {
        let kind = self.kind.clone();
        self.get_object().read().bbox_ref(kind).width
    }

    #[setter]
    fn set_width(&self, width: f64) {
        let kind = self.kind.clone();
        self.get_object().write().bbox_mut(kind).set_width(width);
    }

    #[getter]
    fn get_height(&self) -> f64 {
        let kind = self.kind.clone();
        self.get_object().read().bbox_ref(kind).height
    }

    #[setter]
    fn set_height(&self, height: f64) {
        let kind = self.kind.clone();
        self.get_object().write().bbox_mut(kind).set_height(height);
    }

    #[getter]
    fn get_angle(&self) -> Option<f64> {
        let kind = self.kind.clone();
        self.get_object().read().bbox_ref(kind).angle
    }

    #[setter]
    fn set_angle(&self, angle: Option<f64>) {
        let kind = self.kind.clone();
        self.get_object().write().bbox_mut(kind).set_angle(angle);
    }

    pub fn scale(&self, scale_x: f64, scale_y: f64) {
        let kind = self.kind.clone();
        self.get_object()
            .write()
            .bbox_mut(kind)
            .scale_gil(scale_x, scale_y);
    }

    pub fn vertices(&self) -> Vec<(f64, f64)> {
        let kind = self.kind.clone();
        self.get_object().read().bbox_ref(kind).vertices_gil()
    }

    pub fn vertices_rounded(&self) -> Vec<(f64, f64)> {
        let kind = self.kind.clone();
        self.get_object()
            .read()
            .bbox_ref(kind)
            .vertices_rounded_gil()
    }

    pub fn vertices_int(&self) -> Vec<(i64, i64)> {
        let kind = self.kind.clone();
        self.get_object().read().bbox_ref(kind).vertices_int_gil()
    }

    pub fn as_polygonal_area(&self) -> PolygonalArea {
        let kind = self.kind.clone();
        self.get_object()
            .read()
            .bbox_ref(kind)
            .as_polygonal_area_gil()
    }

    pub fn wrapping_box(&self) -> PythonBBox {
        let kind = self.kind.clone();
        self.get_object().read().bbox_ref(kind).wrapping_box_gil()
    }

    pub fn as_graphical_wrapping_box(
        &self,
        padding: f64,
        border_width: f64,
        max_x: f64,
        max_y: f64,
    ) -> PythonBBox {
        let kind = self.kind.clone();
        self.get_object()
            .read()
            .bbox_ref(kind)
            .as_graphical_wrapping_box_gil(padding, border_width, max_x, max_y)
    }
}
