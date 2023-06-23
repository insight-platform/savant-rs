pub mod context;

use crate::capi::BBOX_ELEMENT_UNDEFINED;
use crate::primitives::to_json_value::ToSerdeJsonValue;
use crate::primitives::{PaddingDraw, Point, PolygonalArea};
use crate::utils::python::no_gil;
use crate::utils::round_2_digits;
use anyhow::bail;
use geo::{Area, BooleanOps};
use lazy_static::lazy_static;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::pyclass::CompareOp;
use pyo3::{pyclass, pymethods, Py, PyAny, PyResult};
use rkyv::{Archive, Deserialize, Serialize};
use std::f64::consts::PI;

pub const EPS: f64 = 0.00001;

lazy_static! {
    pub static ref BBOX_UNDEFINED: RBBox = RBBox::new(
        BBOX_ELEMENT_UNDEFINED,
        BBOX_ELEMENT_UNDEFINED,
        BBOX_ELEMENT_UNDEFINED,
        BBOX_ELEMENT_UNDEFINED,
        None,
    );
}

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct RBBox {
    xc: f64,
    yc: f64,
    width: f64,
    height: f64,
    angle: Option<f64>,
    has_modifications: bool,
}

impl Default for RBBox {
    fn default() -> Self {
        Self {
            xc: 0.0,
            yc: 0.0,
            width: 0.0,
            height: 0.0,
            angle: None,
            has_modifications: false,
        }
    }
}

impl ToSerdeJsonValue for RBBox {
    fn to_serde_json_value(&self) -> serde_json::Value {
        serde_json::json!({
            "xc": self.xc,
            "yc": self.yc,
            "width": self.width,
            "height": self.height,
            "angle": self.angle,
        })
    }
}

#[pymethods]
impl RBBox {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    pub fn area(&self) -> f64 {
        self.width * self.height
    }

    #[pyo3(name = "eq")]
    pub fn geometric_eq(&self, other: &Self) -> bool {
        self.xc == other.xc
            && self.yc == other.yc
            && self.width == other.width
            && self.height == other.height
            && self.angle == other.angle
    }

    pub fn almost_eq(&self, other: &Self, eps: f64) -> bool {
        (self.xc - other.xc).abs() < eps
            && (self.yc - other.yc).abs() < eps
            && (self.width - other.width).abs() < eps
            && (self.height - other.height).abs() < eps
            && (self.angle.unwrap_or(0.0) - other.angle.unwrap_or(0.0)).abs() < eps
    }

    pub(crate) fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge => Err(
                PyNotImplementedError::new_err("Comparison ops Ge/Gt/Le/Lt are not implemented"),
            ),
            CompareOp::Eq => Ok(self.geometric_eq(other)),
            CompareOp::Ne => Ok(!self.geometric_eq(other)),
        }
    }

    #[getter]
    pub fn get_xc(&self) -> f64 {
        self.xc
    }

    #[getter]
    pub fn get_yc(&self) -> f64 {
        self.yc
    }

    #[getter]
    pub fn get_width(&self) -> f64 {
        self.width
    }

    #[getter]
    pub fn get_height(&self) -> f64 {
        self.height
    }

    #[getter]
    pub fn get_width_to_height_ratio(&self) -> f64 {
        self.width / self.height
    }

    #[getter]
    pub fn get_angle(&self) -> Option<f64> {
        self.angle
    }

    pub fn is_modified(&self) -> bool {
        self.has_modifications
    }

    pub fn reset_modifications(&mut self) {
        self.has_modifications = false;
    }

    #[setter]
    pub fn set_xc(&mut self, xc: f64) {
        self.xc = xc;
        self.has_modifications = true;
    }

    #[setter]
    pub fn set_yc(&mut self, yc: f64) {
        self.yc = yc;
        self.has_modifications = true;
    }

    #[setter]
    pub fn set_width(&mut self, width: f64) {
        self.width = width;
        self.has_modifications = true;
    }

    #[setter]
    pub fn set_height(&mut self, height: f64) {
        self.height = height;
        self.has_modifications = true;
    }

    #[setter]
    pub fn set_angle(&mut self, angle: Option<f64>) {
        self.angle = angle;
        self.has_modifications = true;
    }

    #[new]
    pub fn new(xc: f64, yc: f64, width: f64, height: f64, angle: Option<f64>) -> Self {
        Self {
            xc,
            yc,
            width,
            height,
            angle,
            has_modifications: false,
        }
    }

    #[pyo3(name = "scale")]
    pub fn scale_gil(&mut self, scale_x: f64, scale_y: f64) {
        no_gil(|| {
            self.scale(scale_x, scale_y);
        })
    }

    #[getter]
    #[pyo3(name = "vertices")]
    pub fn vertices_gil(&self) -> Vec<(f64, f64)> {
        no_gil(|| self.vertices())
    }

    #[getter]
    #[pyo3(name = "vertices_rounded")]
    pub fn vertices_rounded_gil(&self) -> Vec<(f64, f64)> {
        no_gil(|| self.vertices_rounded())
    }

    #[getter]
    #[pyo3(name = "vertices_int")]
    pub fn vertices_int_gil(&self) -> Vec<(i64, i64)> {
        no_gil(|| self.vertices_int())
    }

    #[pyo3(name = "as_polygonal_area")]
    pub fn as_polygonal_area_gil(&self) -> PolygonalArea {
        no_gil(|| self.as_polygonal_area())
    }

    #[getter]
    #[pyo3(name = "wrapping_box")]
    pub fn wrapping_box_gil(&self) -> PythonBBox {
        no_gil(|| self.wrapping_bbox())
    }

    #[pyo3(name = "visual_box")]
    pub fn visual_box_gil(&self, padding: &PaddingDraw, border_width: i64) -> RBBox {
        no_gil(|| self.visual_bbox(padding, border_width))
    }

    pub fn new_padded(&self, padding: &PaddingDraw) -> Self {
        let (left, right, top, bottom) = (
            padding.left as f64,
            padding.right as f64,
            padding.top as f64,
            padding.bottom as f64,
        );

        let angle_rad = self.angle.unwrap_or(0.0) * PI / 180.0;
        let cos_theta = angle_rad.cos();
        let sin_theta = angle_rad.sin();

        let xc = self.xc + ((right - left) * cos_theta - (bottom - top) * sin_theta) / 2.0;
        let yc = self.yc + ((right - left) * sin_theta + (bottom - top) * cos_theta) / 2.0;
        let height = self.height + top + bottom;
        let width = self.width + left + right;
        let angle = self.angle;

        Self {
            xc,
            yc,
            width,
            height,
            angle,
            has_modifications: false,
        }
    }

    /// Returns a copy of the RBBox object
    ///
    /// Returns
    /// -------
    /// :py:class:`RBBox`
    ///    A copy of the RBBox object
    ///
    ///
    #[pyo3(name = "copy")]
    pub fn copy_py(&self) -> Self {
        let mut new_self = self.clone();
        new_self.has_modifications = false;
        new_self
    }

    #[pyo3(name = "iou")]
    pub(crate) fn iou_gil(&self, other: &Self) -> PyResult<f64> {
        no_gil(|| {
            self.iou(other)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }
}

impl RBBox {
    pub fn iou(&self, other: &Self) -> anyhow::Result<f64> {
        if self.area() < EPS || other.area() < EPS {
            bail!("Area of one of the bounding boxes is zero. Division by zero is not allowed.");
        }

        let mut area1 = self.as_polygonal_area();
        let poly1 = area1.get_polygon();
        let mut area2 = other.as_polygonal_area();
        let poly2 = area2.get_polygon();
        let union = poly1.union(&poly2).unsigned_area();
        if union < EPS {
            bail!("Union of two bounding boxes is zero. Division by zero is not allowed.",)
        }

        let intersection = poly1.intersection(&poly2).unsigned_area();
        Ok(intersection / union)
    }

    pub fn scale(&mut self, scale_x: f64, scale_y: f64) {
        self.has_modifications = true;
        match self.angle {
            None => {
                self.xc *= scale_x;
                self.yc *= scale_y;
                self.width *= scale_x;
                self.height *= scale_y;
            }
            Some(angle) => {
                let scale_x2 = scale_x * scale_x;
                let scale_y2 = scale_y * scale_y;
                let cotan = (angle * PI / 180.0).tan().powi(-1);
                let cotan_2 = cotan * cotan;
                let scale_angle =
                    (scale_x * angle.signum() / (scale_x2 + scale_y2 * cotan_2).sqrt()).acos();
                let nscale_height = ((scale_x2 + scale_y2 * cotan_2) / (1.0 + cotan_2)).sqrt();
                let ayh = 1.0 / ((90.0 - angle) / 180.0 * std::f64::consts::PI).tan();
                let nscale_width = ((scale_x2 + scale_y2 * ayh * ayh) / (1.0 + ayh * ayh)).sqrt();

                self.angle = Some(90.0 - (scale_angle * 180.0 / std::f64::consts::PI));
                self.xc *= scale_x;
                self.yc *= scale_y;
                self.width *= nscale_width;
                self.height *= nscale_height;
            }
        }
    }

    pub fn vertices(&self) -> Vec<(f64, f64)> {
        let angle = self.angle.unwrap_or(0.0);
        let angle = angle * std::f64::consts::PI / 180.0;
        let cos = angle.cos();
        let sin = angle.sin();
        let x = self.xc;
        let y = self.yc;
        let w = self.width / 2.0;
        let h = self.height / 2.0;
        vec![
            (x + w * cos - h * sin, y + w * sin + h * cos),
            (x + w * cos + h * sin, y + w * sin - h * cos),
            (x - w * cos + h * sin, y - w * sin - h * cos),
            (x - w * cos - h * sin, y - w * sin + h * cos),
        ]
    }

    pub fn vertices_rounded(&self) -> Vec<(f64, f64)> {
        self.vertices()
            .into_iter()
            .map(|(x, y)| (round_2_digits(x), round_2_digits(y)))
            .collect::<Vec<_>>()
    }

    pub fn vertices_int(&self) -> Vec<(i64, i64)> {
        self.vertices()
            .into_iter()
            .map(|(x, y)| (x as i64, y as i64))
            .collect::<Vec<_>>()
    }

    pub fn as_polygonal_area(&self) -> PolygonalArea {
        PolygonalArea::new(
            self.vertices()
                .into_iter()
                .map(|(x, y)| Point::new(x, y))
                .collect::<Vec<_>>(),
            None,
        )
    }

    pub fn wrapping_bbox(&self) -> PythonBBox {
        if self.angle.is_none() {
            PythonBBox::new(self.xc, self.yc, self.width, self.height)
        } else {
            let mut vertices = self.vertices();
            let (initial_vtx_x, initial_vtx_y) = vertices.pop().unwrap();
            let (mut min_x, mut min_y, mut max_x, mut max_y) =
                (initial_vtx_x, initial_vtx_y, initial_vtx_x, initial_vtx_y);
            for v in vertices {
                let (vtx_x, vtx_y) = v;
                if vtx_x < min_x {
                    min_x = vtx_x;
                }
                if vtx_x > max_x {
                    max_x = vtx_x;
                }
                if vtx_y < min_y {
                    min_y = vtx_y;
                }
                if vtx_y > max_y {
                    max_y = vtx_y;
                }
            }
            PythonBBox::new(
                (min_x + max_x) / 2.0,
                (min_y + max_y) / 2.0,
                max_x - min_x,
                max_y - min_y,
            )
        }
    }

    pub fn visual_bbox(&self, padding: &PaddingDraw, border_width: i64) -> RBBox {
        assert!(border_width >= 0);
        let padding_with_border = PaddingDraw::new(
            padding.left + border_width,
            padding.top + border_width,
            padding.right + border_width,
            padding.bottom + border_width,
        );

        self.new_padded(&padding_with_border)
    }
}

#[pyclass]
#[derive(Clone, Debug)]
#[pyo3(name = "BBox")]
pub struct PythonBBox {
    pub(crate) inner: RBBox,
}

impl PythonBBox {
    pub fn visual_bbox(
        &self,
        padding: &PaddingDraw,
        border_width: i64,
        max_x: f64,
        max_y: f64,
    ) -> PythonBBox {
        assert!(border_width >= 0 && max_x >= 0.0 && max_y >= 0.0);

        let padding_with_border = PaddingDraw::new(
            padding.left + border_width,
            padding.top + border_width,
            padding.right + border_width,
            padding.bottom + border_width,
        );

        let bbox = self.new_padded(&padding_with_border);

        let left = 0.0f64.max(bbox.get_left()).floor();
        let top = 0.0f64.max(bbox.get_top()).floor();
        let right = max_x.min(bbox.get_right()).ceil();
        let bottom = max_y.min(bbox.get_bottom()).ceil();

        let mut width = 1.0f64.max(right - left);
        if width as i64 % 2 != 0 {
            width += 1.0;
        }

        let mut height = 1.0f64.max(bottom - top);
        if height as i64 % 2 != 0 {
            height += 1.0;
        }

        PythonBBox::new(left + width / 2.0, top + height / 2.0, width, height)
    }
}

#[pymethods]
impl PythonBBox {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    #[pyo3(name = "eq")]
    pub fn geometric_eq(&self, other: &Self) -> bool {
        self.inner.geometric_eq(&other.inner)
    }

    pub fn almost_eq(&self, other: &Self, eps: f64) -> bool {
        self.inner.almost_eq(&other.inner, eps)
    }

    pub(crate) fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        self.inner.__richcmp__(&other.inner, op)
    }

    #[pyo3(name = "iou")]
    fn iou_gil(&self, other: &Self) -> PyResult<f64> {
        self.inner.iou_gil(&other.inner)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn is_modified(&self) -> bool {
        self.inner.has_modifications
    }

    #[new]
    pub fn new(xc: f64, yc: f64, width: f64, height: f64) -> Self {
        Self {
            inner: RBBox::new(xc, yc, width, height, None),
        }
    }

    #[staticmethod]
    pub fn ltrb(left: f64, top: f64, right: f64, bottom: f64) -> Self {
        let width = right - left;
        let height = bottom - top;

        let xc = (left + right) / 2.0;
        let yc = (top + bottom) / 2.0;

        Self {
            inner: RBBox::new(xc, yc, width, height, None),
        }
    }

    #[staticmethod]
    pub fn ltwh(left: f64, top: f64, width: f64, height: f64) -> Self {
        let xc = left + width / 2.0;
        let yc = top + height / 2.0;

        Self {
            inner: RBBox::new(xc, yc, width, height, None),
        }
    }

    #[getter]
    pub fn get_xc(&self) -> f64 {
        self.inner.get_xc()
    }

    #[setter]
    pub fn set_xc(&mut self, xc: f64) {
        self.inner.set_xc(xc);
    }

    #[getter]
    pub fn get_yc(&self) -> f64 {
        self.inner.get_yc()
    }

    #[setter]
    pub fn set_yc(&mut self, yc: f64) {
        self.inner.set_yc(yc);
    }

    #[getter]
    pub fn get_width(&self) -> f64 {
        self.inner.get_width()
    }

    #[setter]
    pub fn set_width(&mut self, width: f64) {
        self.inner.set_width(width);
    }

    #[getter]
    pub fn get_height(&self) -> f64 {
        self.inner.get_height()
    }

    #[setter]
    pub fn set_height(&mut self, height: f64) {
        self.inner.set_height(height);
    }

    #[getter]
    pub fn get_top(&self) -> f64 {
        self.inner.get_yc() - self.inner.get_height() / 2.0
    }

    #[setter]
    pub fn set_top(&mut self, top: f64) {
        self.inner.set_yc(top + self.inner.get_height() / 2.0);
    }

    #[getter]
    pub fn get_left(&self) -> f64 {
        self.inner.get_xc() - self.inner.get_width() / 2.0
    }

    #[setter]
    pub fn set_left(&mut self, left: f64) {
        self.inner.set_xc(left + self.inner.get_width() / 2.0);
    }

    #[getter]
    pub fn get_right(&self) -> f64 {
        self.inner.get_xc() + self.inner.get_width() / 2.0
    }

    #[getter]
    pub fn get_bottom(&self) -> f64 {
        self.inner.get_yc() + self.inner.get_height() / 2.0
    }

    #[getter]
    #[pyo3(name = "vertices")]
    pub fn vertices_gil(&self) -> Vec<(f64, f64)> {
        no_gil(|| self.inner.vertices())
    }

    #[getter]
    #[pyo3(name = "vertices_rounded")]
    pub fn vertices_rounded_gil(&self) -> Vec<(f64, f64)> {
        no_gil(|| self.inner.vertices_rounded())
    }

    #[getter]
    #[pyo3(name = "vertices_int")]
    pub fn vertices_int_gil(&self) -> Vec<(i64, i64)> {
        no_gil(|| self.inner.vertices_int())
    }

    #[getter]
    #[pyo3(name = "wrapping_box")]
    pub fn wrapping_box_gil(&self) -> PythonBBox {
        no_gil(|| self.inner.wrapping_bbox())
    }

    #[pyo3(name = "visual_box")]
    pub fn visual_box_gil(
        &self,
        padding: &PaddingDraw,
        border_width: i64,
        max_x: f64,
        max_y: f64,
    ) -> PythonBBox {
        no_gil(|| self.visual_bbox(padding, border_width, max_x, max_y))
    }

    pub fn as_ltrb(&self) -> (f64, f64, f64, f64) {
        let top = self.get_top();
        let left = self.get_left();
        let bottom = self.get_bottom();
        let right = self.get_right();
        (left, top, right, bottom)
    }

    pub fn as_ltrb_int(&self) -> (i64, i64, i64, i64) {
        let top = self.get_top().floor();
        let left = self.get_left().floor();
        let bottom = self.get_bottom().ceil();
        let right = self.get_right().ceil();
        (left as i64, top as i64, right as i64, bottom as i64)
    }

    pub fn as_ltwh(&self) -> (f64, f64, f64, f64) {
        let top = self.get_top();
        let left = self.get_left();
        let width = self.get_width();
        let height = self.get_height();
        (left, top, width, height)
    }

    pub fn as_ltwh_int(&self) -> (i64, i64, i64, i64) {
        let top = self.get_top().floor();
        let left = self.get_left().floor();
        let width = self.get_width().ceil();
        let height = self.get_height().ceil();
        (left as i64, top as i64, width as i64, height as i64)
    }

    pub fn as_xcycwh(&self) -> (f64, f64, f64, f64) {
        let xc = self.get_xc();
        let yc = self.get_yc();
        let width = self.get_width();
        let height = self.get_height();
        (xc, yc, width, height)
    }

    pub fn as_xcycwh_int(&self) -> (i64, i64, i64, i64) {
        let xc = self.get_xc();
        let yc = self.get_yc();
        let width = self.get_width();
        let height = self.get_height();
        (xc as i64, yc as i64, width as i64, height as i64)
    }

    pub fn as_rbbox(&self) -> RBBox {
        self.inner.clone()
    }

    #[pyo3(name = "scale")]
    pub fn scale_gil(&mut self, scale_x: f64, scale_y: f64) {
        no_gil(|| {
            self.inner.scale(scale_x, scale_y);
        })
    }

    #[pyo3(name = "as_polygonal_area")]
    pub fn as_polygonal_area_gil(&self) -> PolygonalArea {
        no_gil(|| self.inner.as_polygonal_area())
    }

    /// Returns a copy of the BBox object
    ///
    /// Returns
    /// -------
    /// :py:class:`BBox`
    ///    A copy of the BBox object
    ///
    ///
    #[pyo3(name = "copy")]
    pub fn copy_py(&self) -> Self {
        let mut new_self = self.clone();
        new_self.inner.reset_modifications();
        new_self
    }

    pub fn new_padded(&self, padding: &PaddingDraw) -> Self {
        let inner_copy = self.inner.clone();
        let padded = inner_copy.new_padded(padding);
        Self { inner: padded }
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::{PaddingDraw, RBBox};
    use crate::utils::round_2_digits;
    use pyo3::basic::CompareOp;

    #[test]
    fn test_scale_no_angle() {
        let mut bbox = RBBox::new(0.0, 0.0, 100.0, 100.0, None);
        bbox.scale(2.0, 2.0);
        assert_eq!(bbox.xc, 0.0);
        assert_eq!(bbox.yc, 0.0);
        assert_eq!(bbox.width, 200.0);
        assert_eq!(bbox.height, 200.0);
        assert_eq!(bbox.angle, None);
    }

    #[test]
    fn test_scale_with_angle() {
        let mut bbox = RBBox::new(0.0, 0.0, 100.0, 100.0, Some(45.0));
        bbox.scale(2.0, 3.0);
        //dbg!(&bbox);
        assert_eq!(bbox.xc, 0.0);
        assert_eq!(bbox.yc, 0.0);
        assert_eq!(round_2_digits(bbox.width), 254.95);
        assert_eq!(round_2_digits(bbox.height), 254.95);
        assert_eq!(bbox.angle.map(round_2_digits), Some(33.69));
    }

    #[test]
    fn test_vertices() {
        let bbox = RBBox::new(0.0, 0.0, 100.0, 100.0, Some(45.0));
        let vertices = bbox.vertices_rounded();
        assert_eq!(vertices.len(), 4);
        assert_eq!(vertices[0], (0.0, 70.71));
        assert_eq!(vertices[1], (70.71, 0.0));
        assert_eq!(vertices[2], (-0.0, -70.71));
        assert_eq!(vertices[3], (-70.71, 0.0));
    }

    #[test]
    fn test_wrapping_bbox() {
        let bbox = RBBox::new(0.0, 0.0, 100.0, 100.0, Some(45.0));
        let wrapped = bbox.wrapping_bbox();
        assert_eq!(wrapped.inner.xc, 0.0);
        assert_eq!(wrapped.inner.yc, 0.0);
        assert_eq!(round_2_digits(wrapped.inner.width), 141.42);
        assert_eq!(round_2_digits(wrapped.inner.height), 141.42);
        assert_eq!(wrapped.inner.angle, None);

        let bbox = RBBox::new(0.0, 0.0, 50.0, 100.0, None);
        let wrapped = bbox.wrapping_bbox();
        assert_eq!(wrapped.inner.xc, 0.0);
        assert_eq!(wrapped.inner.yc, 0.0);
        assert_eq!(round_2_digits(wrapped.inner.width), 50.0);
        assert_eq!(round_2_digits(wrapped.inner.height), 100.0);
        assert_eq!(wrapped.inner.angle, None);

        let bbox = RBBox::new(0.0, 0.0, 50.0, 100.0, Some(90.0));
        let wrapped = bbox.wrapping_bbox();
        assert_eq!(wrapped.inner.xc, 0.0);
        assert_eq!(wrapped.inner.yc, 0.0);
        assert_eq!(round_2_digits(wrapped.inner.width), 100.0);
        assert_eq!(round_2_digits(wrapped.inner.height), 50.0);
        assert_eq!(wrapped.inner.angle, None);
    }

    fn get_bbox(angle: Option<f64>) -> RBBox {
        RBBox::new(0.0, 0.0, 100.0, 100.0, angle)
    }

    #[test]
    fn check_modifications() {
        let mut bb = get_bbox(Some(45.0));
        bb.set_xc(10.0);
        assert!(bb.is_modified());

        let mut bb = get_bbox(Some(45.0));
        bb.set_yc(10.0);
        assert!(bb.is_modified());

        let mut bb = get_bbox(Some(45.0));
        bb.set_width(10.0);
        assert!(bb.is_modified());

        let mut bb = get_bbox(Some(45.0));
        bb.set_height(10.0);
        assert!(bb.is_modified());

        let mut bb = get_bbox(Some(45.0));
        bb.set_angle(Some(10.0));
        assert!(bb.is_modified());

        let mut bb = get_bbox(Some(45.0));
        bb.set_angle(None);
        assert!(bb.is_modified());

        let mut bb = get_bbox(Some(45.0));
        bb.scale(2.0, 2.0);
        assert!(bb.is_modified());
    }

    #[test]
    fn test_padded_axis_aligned() {
        let bb = get_bbox(None);
        let padded = bb.new_padded(&PaddingDraw::new(0, 0, 0, 0));
        assert_eq!(padded.get_xc(), bb.get_xc());
        assert_eq!(padded.get_yc(), bb.get_yc());
        assert_eq!(padded.get_width(), bb.get_width());
        assert_eq!(padded.get_height(), bb.get_height());

        let bb = get_bbox(None);
        let padded = bb.new_padded(&PaddingDraw::new(2, 0, 0, 0));
        assert_eq!(padded.get_xc(), bb.get_xc() - 1.0);
        assert_eq!(padded.get_yc(), bb.get_yc());
        assert_eq!(padded.get_width(), bb.get_width() + 2.0);
        assert_eq!(padded.get_height(), bb.get_height());

        let bb = get_bbox(None);
        let padded = bb.new_padded(&PaddingDraw::new(0, 2, 0, 0));
        assert_eq!(padded.get_xc(), bb.get_xc());
        assert_eq!(padded.get_yc(), bb.get_yc() - 1.0);
        assert_eq!(padded.get_width(), bb.get_width());
        assert_eq!(padded.get_height(), bb.get_height() + 2.0);

        let bb = get_bbox(None);
        let padded = bb.new_padded(&PaddingDraw::new(2, 0, 4, 0));
        assert_eq!(padded.get_xc(), bb.get_xc() + 1.0);
        assert_eq!(padded.get_yc(), bb.get_yc());
        assert_eq!(padded.get_width(), bb.get_width() + 6.0);
        assert_eq!(padded.get_height(), bb.get_height());
    }

    #[test]
    fn test_padded_rotated() {
        let bb = get_bbox(Some(90.0));
        let padded = bb.new_padded(&PaddingDraw::new(2, 0, 0, 0));
        assert_eq!(round_2_digits(padded.get_xc()), bb.get_xc());
        assert_eq!(round_2_digits(padded.get_yc()), bb.get_yc() - 1.0);
        assert_eq!(padded.get_width(), bb.get_width() + 2.0);
        assert_eq!(padded.get_height(), bb.get_height());
    }

    #[test]
    fn test_eq() {
        let bb1 = get_bbox(Some(45.0));
        let bb2 = get_bbox(Some(45.0));
        assert_eq!(bb1, bb2);

        let bb1 = get_bbox(Some(45.0));
        let bb2 = get_bbox(Some(90.0));
        assert_ne!(bb1, bb2);

        let bb1 = get_bbox(Some(45.0));
        let bb2 = get_bbox(None);
        assert_ne!(bb1, bb2);
    }

    #[test]
    fn test_almost_eq() {
        let bb1 = get_bbox(Some(45.0));
        let bb2 = get_bbox(Some(45.05));
        assert!(bb1.almost_eq(&bb2, 0.1));

        let bb1 = get_bbox(Some(45.0));
        let bb2 = get_bbox(Some(90.0));
        assert!(!bb1.almost_eq(&bb2, 0.1));

        let bb1 = get_bbox(Some(45.0));
        let bb2 = get_bbox(None);
        assert!(!bb1.almost_eq(&bb2, 0.1));
    }

    #[test]
    fn test_richcmp_ok() {
        let bb1 = get_bbox(Some(45.0));
        let bb2 = get_bbox(Some(45.0));
        assert!(matches!(bb1.__richcmp__(&bb2, CompareOp::Eq), Ok(true)));
        assert!(matches!(bb1.__richcmp__(&bb2, CompareOp::Ne), Ok(false)));
        assert!(matches!(bb1.__richcmp__(&bb2, CompareOp::Lt), Err(_)));
        assert!(matches!(bb1.__richcmp__(&bb2, CompareOp::Le), Err(_)));
        assert!(matches!(bb1.__richcmp__(&bb2, CompareOp::Gt), Err(_)));
        assert!(matches!(bb1.__richcmp__(&bb2, CompareOp::Ge), Err(_)));
    }
}
