pub mod context;
pub mod transformations;

use crate::capi::BBOX_ELEMENT_UNDEFINED;
use crate::primitives::message::video::object::VideoObject;
use crate::primitives::to_json_value::ToSerdeJsonValue;
use crate::primitives::{PaddingDraw, Point, PolygonalArea};
use crate::release_gil;
use crate::utils::round_2_digits;
use anyhow::bail;
use geo::{Area, BooleanOps};
use lazy_static::lazy_static;
use parking_lot::RwLock;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::pyclass::CompareOp;
use pyo3::{pyclass, pymethods, Py, PyAny, PyResult};
use rkyv::{Archive, Deserialize, Serialize};
use std::f32::consts::PI;
use std::sync::Arc;

pub const EPS: f32 = 0.00001;

lazy_static! {
    pub static ref BBOX_UNDEFINED: RBBox = RBBox::new(
        BBOX_ELEMENT_UNDEFINED,
        BBOX_ELEMENT_UNDEFINED,
        BBOX_ELEMENT_UNDEFINED,
        BBOX_ELEMENT_UNDEFINED,
        None,
    );
}

/// Allows configuring what kind of Intersection over Something to use.
///
/// IoU - Intersection over Union
/// IoSelf - Intersection over Self (Intersection / Area of Self)
/// IoOther - Intersection over Other (Intersection / Area of Other)
///
#[pyclass]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename = "bbox.metric.type")]
pub enum BBoxMetricType {
    IoU,
    IoSelf,
    IoOther,
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct OwnedRBBoxData {
    pub xc: f32,
    pub yc: f32,
    pub width: f32,
    pub height: f32,
    pub angle: Option<f32>,
    pub has_modifications: bool,
}

impl TryFrom<RBBox> for OwnedRBBoxData {
    type Error = anyhow::Error;

    fn try_from(value: RBBox) -> Result<Self, Self::Error> {
        OwnedRBBoxData::try_from(&value)
    }
}

impl TryFrom<&RBBox> for OwnedRBBoxData {
    type Error = anyhow::Error;

    fn try_from(value: &RBBox) -> Result<Self, Self::Error> {
        match &value.data {
            BBoxVariant::Owned(d) => Ok(d.clone()),
            BBoxVariant::BorrowedDetectionBox(d) => Ok(d.read().detection_box.clone()),
            BBoxVariant::BorrowedTrackingBox(d) => d.read().track_box.as_ref().map_or_else(
                || Err(anyhow::anyhow!("Cannot convert tracking box to RBBoxData")),
                |t| Ok(t.clone()),
            ),
        }
    }
}

impl Default for OwnedRBBoxData {
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

impl ToSerdeJsonValue for OwnedRBBoxData {
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

#[derive(Debug, Clone)]
enum BBoxVariant {
    Owned(OwnedRBBoxData),
    BorrowedDetectionBox(Arc<RwLock<VideoObject>>),
    BorrowedTrackingBox(Arc<RwLock<VideoObject>>),
}

/// Represents a bounding box with an optional rotation angle in degrees.
///
#[pyclass]
#[derive(Debug, Clone)]
pub struct RBBox {
    data: BBoxVariant,
}

impl PartialEq for RBBox {
    fn eq(&self, other: &Self) -> bool {
        self.geometric_eq(other)
    }
}

impl ToSerdeJsonValue for RBBox {
    fn to_serde_json_value(&self) -> serde_json::Value {
        serde_json::json!({
            "xc": self.get_xc(),
            "yc": self.get_yc(),
            "width": self.get_width(),
            "height": self.get_height(),
            "angle": self.get_angle(),
        })
    }
}

impl RBBox {
    pub fn new_from_data(data: OwnedRBBoxData) -> Self {
        Self {
            data: BBoxVariant::Owned(data),
        }
    }

    pub fn borrowed_detection_box(object: Arc<RwLock<VideoObject>>) -> Self {
        Self {
            data: BBoxVariant::BorrowedDetectionBox(object),
        }
    }

    pub fn borrowed_track_box(object: Arc<RwLock<VideoObject>>) -> Self {
        Self {
            data: BBoxVariant::BorrowedTrackingBox(object),
        }
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

    /// Computes the area of bbox
    ///
    /// Returns
    /// -------
    /// float
    ///   area of bbox
    ///
    #[getter]
    pub fn get_area(&self) -> f32 {
        self.get_width() * self.get_height()
    }

    /// Compares boxes geometrically.
    ///
    /// Parameters
    /// ----------
    /// other : :py:class:`RBBox`
    ///   other bbox to compare with
    ///
    /// Returns
    /// -------
    /// bool
    ///   True if boxes are geometrically equal, False otherwise.
    ///
    #[pyo3(name = "eq")]
    pub fn geometric_eq(&self, other: &Self) -> bool {
        self.get_xc() == other.get_xc()
            && self.get_yc() == other.get_yc()
            && self.get_width() == other.get_width()
            && self.get_height() == other.get_height()
            && self.get_angle() == other.get_angle()
    }

    /// Compares boxes geometrically with given precision.
    ///
    /// Parameters
    /// ----------
    /// other : :py:class:`RBBox`
    ///   other bbox to compare with
    /// eps : float
    ///   precision
    ///
    /// Returns
    /// -------
    /// bool
    ///   True if boxes are geometrically equal, False otherwise.
    ///
    pub fn almost_eq(&self, other: &Self, eps: f32) -> bool {
        (self.get_xc() - other.get_xc()).abs() < eps
            && (self.get_yc() - other.get_yc()).abs() < eps
            && (self.get_width() - other.get_width()).abs() < eps
            && (self.get_height() - other.get_height()).abs() < eps
            && (self.get_angle().unwrap_or(0.0) - other.get_angle().unwrap_or(0.0)).abs() < eps
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

    /// Access and change the x center of the bbox.
    ///
    #[getter]
    pub fn get_xc(&self) -> f32 {
        match &self.data {
            BBoxVariant::Owned(d) => d.xc,
            BBoxVariant::BorrowedDetectionBox(d) => d.read_recursive().detection_box.xc,
            BBoxVariant::BorrowedTrackingBox(d) => d
                .read_recursive()
                .track_box
                .as_ref()
                .map(|t| t.xc)
                .unwrap_or(0.0),
        }
    }

    /// Access and change the y center of the bbox.
    ///
    #[getter]
    pub fn get_yc(&self) -> f32 {
        match &self.data {
            BBoxVariant::Owned(d) => d.yc,
            BBoxVariant::BorrowedDetectionBox(d) => d.read_recursive().detection_box.yc,
            BBoxVariant::BorrowedTrackingBox(d) => d
                .read_recursive()
                .track_box
                .as_ref()
                .map(|t| t.yc)
                .unwrap_or(0.0),
        }
    }

    /// Access and change the width of the bbox.
    ///
    #[getter]
    pub fn get_width(&self) -> f32 {
        match &self.data {
            BBoxVariant::Owned(d) => d.width,
            BBoxVariant::BorrowedDetectionBox(d) => d.read_recursive().detection_box.width,
            BBoxVariant::BorrowedTrackingBox(d) => d
                .read_recursive()
                .track_box
                .as_ref()
                .map(|t| t.width)
                .unwrap_or(0.0),
        }
    }

    /// Access and change the height of the bbox.
    ///
    #[getter]
    pub fn get_height(&self) -> f32 {
        match &self.data {
            BBoxVariant::Owned(d) => d.height,
            BBoxVariant::BorrowedDetectionBox(d) => d.read_recursive().detection_box.height,
            BBoxVariant::BorrowedTrackingBox(d) => d
                .read_recursive()
                .track_box
                .as_ref()
                .map(|t| t.height)
                .unwrap_or(0.0),
        }
    }

    /// Access and change the angle of the bbox. To unset the angle use None as a value.
    ///
    #[getter]
    pub fn get_angle(&self) -> Option<f32> {
        match &self.data {
            BBoxVariant::Owned(d) => d.angle,
            BBoxVariant::BorrowedDetectionBox(d) => d.read_recursive().detection_box.angle,
            BBoxVariant::BorrowedTrackingBox(d) => d
                .read_recursive()
                .track_box
                .as_ref()
                .map(|t| t.angle)
                .unwrap_or(None),
        }
    }

    /// Access the ratio between width and height.
    ///
    #[getter]
    pub fn get_width_to_height_ratio(&self) -> f32 {
        let height = self.get_height();
        if height == 0.0 {
            // TODO: should we return an error here?
            return -1.0;
        }
        self.get_width() / self.get_height()
    }

    /// Flag indicating if the bbox has been modified.
    ///
    pub fn is_modified(&self) -> bool {
        match &self.data {
            BBoxVariant::Owned(d) => d.has_modifications,
            BBoxVariant::BorrowedDetectionBox(d) => {
                d.read_recursive().detection_box.has_modifications
            }
            BBoxVariant::BorrowedTrackingBox(d) => d
                .read_recursive()
                .track_box
                .as_ref()
                .map(|t| t.has_modifications)
                .unwrap_or(false),
        }
    }

    /// Resets the modification flag.
    ///
    pub fn set_modifications(&mut self, value: bool) {
        match &mut self.data {
            BBoxVariant::Owned(d) => d.has_modifications = value,
            BBoxVariant::BorrowedDetectionBox(d) => {
                let mut lock = d.write();
                lock.detection_box.has_modifications = value;
            }
            BBoxVariant::BorrowedTrackingBox(d) => {
                let mut lock = d.write();
                if let Some(b) = &mut lock.track_box {
                    b.has_modifications = value;
                }
            }
        }
    }

    #[setter]
    pub fn set_xc(&mut self, xc: f32) {
        match &mut self.data {
            BBoxVariant::Owned(d) => {
                d.xc = xc;
                d.has_modifications = true;
            }
            BBoxVariant::BorrowedDetectionBox(d) => {
                let mut lock = d.write();
                lock.detection_box.xc = xc;
                lock.detection_box.has_modifications = true;
            }
            BBoxVariant::BorrowedTrackingBox(d) => {
                let mut lock = d.write();
                if let Some(b) = &mut lock.track_box {
                    b.xc = xc;
                    b.has_modifications = true;
                }
            }
        }
    }

    #[setter]
    pub fn set_yc(&mut self, yc: f32) {
        match &mut self.data {
            BBoxVariant::Owned(d) => {
                d.yc = yc;
                d.has_modifications = true;
            }
            BBoxVariant::BorrowedDetectionBox(d) => {
                let mut lock = d.write();
                lock.detection_box.yc = yc;
                lock.detection_box.has_modifications = true;
            }
            BBoxVariant::BorrowedTrackingBox(d) => {
                let mut lock = d.write();
                if let Some(b) = &mut lock.track_box {
                    b.yc = yc;
                    b.has_modifications = true;
                }
            }
        }
    }

    #[setter]
    pub fn set_width(&mut self, width: f32) {
        match &mut self.data {
            BBoxVariant::Owned(d) => {
                d.width = width;
                d.has_modifications = true;
            }
            BBoxVariant::BorrowedDetectionBox(d) => {
                let mut lock = d.write();
                lock.detection_box.width = width;
                lock.detection_box.has_modifications = true;
            }
            BBoxVariant::BorrowedTrackingBox(d) => {
                let mut lock = d.write();
                if let Some(b) = &mut lock.track_box {
                    b.width = width;
                    b.has_modifications = true;
                }
            }
        }
    }

    #[setter]
    pub fn set_height(&mut self, height: f32) {
        match &mut self.data {
            BBoxVariant::Owned(d) => {
                d.height = height;
                d.has_modifications = true;
            }
            BBoxVariant::BorrowedDetectionBox(d) => {
                let mut lock = d.write();
                lock.detection_box.height = height;
                lock.detection_box.has_modifications = true;
            }
            BBoxVariant::BorrowedTrackingBox(d) => {
                let mut lock = d.write();
                if let Some(b) = &mut lock.track_box {
                    b.height = height;
                    b.has_modifications = true;
                }
            }
        }
    }

    #[setter]
    pub fn set_angle(&mut self, angle: Option<f32>) {
        match &mut self.data {
            BBoxVariant::Owned(d) => {
                d.angle = angle;
                d.has_modifications = true;
            }
            BBoxVariant::BorrowedDetectionBox(d) => {
                let mut lock = d.write();
                lock.detection_box.angle = angle;
                lock.detection_box.has_modifications = true;
            }
            BBoxVariant::BorrowedTrackingBox(d) => {
                let mut lock = d.write();
                if let Some(b) = &mut lock.track_box {
                    b.angle = angle;
                    b.has_modifications = true;
                }
            }
        }
    }

    /// Alias to the ``__init__`` method.
    ///
    /// Parameters
    /// ----------
    /// xc : float
    ///   x center of bbox
    /// yc : float
    ///   y center of bbox
    /// width : float
    ///   width of bbox
    /// height : float
    ///   height of bbox
    /// angle : float, optional
    ///   angle of bbox
    ///
    /// Returns
    /// -------
    /// :py:class:`RBBox`
    ///  new bbox
    ///
    #[staticmethod]
    fn constructor(xc: f32, yc: f32, width: f32, height: f32, angle: Option<f32>) -> Self {
        Self::new(xc, yc, width, height, angle)
    }

    #[new]
    pub fn new(xc: f32, yc: f32, width: f32, height: f32, angle: Option<f32>) -> Self {
        Self {
            data: BBoxVariant::Owned(OwnedRBBoxData {
                xc,
                yc,
                width,
                height,
                angle,
                has_modifications: false,
            }),
        }
    }

    /// Scales the bbox by given factors. The function is GIL-free. Scaling happens in-place.
    ///
    #[pyo3(name = "scale")]
    pub fn scale_py(&mut self, scale_x: f32, scale_y: f32) {
        self.scale(scale_x, scale_y)
    }

    /// Returns vertices of the bbox. The property is GIL-free.
    ///
    /// Returns
    /// -------
    /// list of tuples (float, float)
    ///   vertices of the bbox
    ///
    #[getter]
    #[pyo3(name = "vertices")]
    pub fn get_vertices_gil(&self) -> Vec<(f32, f32)> {
        self.get_vertices()
    }

    /// Returns vertices of the bbox rounded to 2 decimal digits. The property is GIL-free.
    ///
    /// Returns
    /// -------
    /// list of tuples (float, float)
    ///   vertices of the bbox rounded to 2 decimal digits
    ///
    #[getter]
    #[pyo3(name = "vertices_rounded")]
    pub fn get_vertices_rounded_gil(&self) -> Vec<(f32, f32)> {
        self.get_vertices_rounded()
    }

    /// Returns vertices of the bbox rounded to integers. The property is GIL-free.
    ///
    /// Returns
    /// -------
    /// list of tuples (int, int)
    ///   vertices of the bbox rounded to integers
    ///
    #[getter]
    #[pyo3(name = "vertices_int")]
    pub fn get_vertices_int_gil(&self) -> Vec<(i64, i64)> {
        self.get_vertices_int()
    }

    /// Returns bbox as a polygonal area. The property is GIL-free.
    ///
    /// Returns
    /// -------
    /// :py:class:`PolygonalArea`
    ///   polygonal area of the bbox
    ///
    #[pyo3(name = "as_polygonal_area")]
    pub fn get_as_polygonal_area_gil(&self) -> PolygonalArea {
        self.get_as_polygonal_area()
    }

    /// Returns axis-aligned bounding box wrapping the bbox. The property is GIL-free.
    ///
    /// Returns
    /// -------
    /// :py:class:`BBox`
    ///   axis-aligned bounding box wrapping the bbox
    ///
    #[getter]
    #[pyo3(name = "wrapping_box")]
    pub fn get_wrapping_box_gil(&self) -> PythonBBox {
        self.get_wrapping_bbox()
    }

    /// Returns rotated bounding box wrapping the bbox with geometry corrected with respect to the padding and border width. The property is GIL-free.
    ///
    /// Parameters
    /// ----------
    /// padding : :py:class:`savant_rs.draw_spec.PaddingDraw`
    ///   padding of the bbox
    /// border_width : int
    ///   border width of the bbox
    ///
    /// Returns
    /// -------
    /// :py:class:`RBBox`
    ///   wrapping bbox
    ///
    #[pyo3(name = "visual_box")]
    pub fn get_visual_box_gil(&self, padding: &PaddingDraw, border_width: i64) -> PyResult<RBBox> {
        release_gil!(|| self.get_visual_bbox(padding, border_width))
    }

    /// Returns rotated bounding box wrapping the bbox. The property is GIL-free.
    ///
    /// Parameters
    /// ----------
    /// padding : :py:class:`savant_rs.draw_spec.PaddingDraw`
    ///   padding of the bbox
    ///
    /// Returns
    /// -------
    /// :py:class:`RBBox`
    ///   wrapping bbox
    ///
    #[pyo3(name = "new_padded")]
    pub fn new_padded_gil(&self, padding: &PaddingDraw) -> Self {
        release_gil!(|| self.new_padded(padding))
    }

    /// Returns a copy of the RBBox object. Modification flag is reset.
    ///
    /// Returns
    /// -------
    /// :py:class:`RBBox`
    ///    A copy of the RBBox object
    ///
    ///
    #[pyo3(name = "copy")]
    pub fn copy_py(&self) -> Self {
        let data = self
            .try_into()
            .expect("Failed to convert RBBox to RBBoxData");

        let mut new_self = Self {
            data: BBoxVariant::Owned(data),
        };

        new_self.set_modifications(false);
        new_self
    }

    /// Calculates the intersection over union (IoU) of two rotated bounding boxes. The function is GIL-free.
    ///
    /// Parameters
    /// ----------
    /// other : :py:class:`RBBox`
    ///   other rotated bounding box
    ///
    /// Returns
    /// -------
    /// float
    ///   intersection over union (IoU) of two rotated bounding boxes
    ///
    #[pyo3(name = "iou")]
    pub(crate) fn iou_gil(&self, other: &Self) -> PyResult<f64> {
        release_gil!(|| {
            self.iou(other)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    /// Calculates the intersection over self (IoS) of two rotated bounding boxes. The function is GIL-free.
    ///
    /// Parameters
    /// ----------
    /// other : :py:class:`RBBox`
    ///   other rotated bounding box
    ///
    /// Returns
    /// -------
    /// float
    ///   intersection over self (IoS) of two rotated bounding boxes
    ///
    #[pyo3(name = "ios")]
    pub(crate) fn ios_gil(&self, other: &Self) -> PyResult<f64> {
        release_gil!(|| {
            self.ios(other)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    /// Calculates the intersection over other (IoO) of two rotated bounding boxes. The function is GIL-free.
    ///
    /// Parameters
    /// ----------
    /// other : :py:class:`RBBox`
    ///   other rotated bounding box
    ///
    /// Returns
    /// -------
    /// float
    ///   intersection over other (IoO) of two rotated bounding boxes
    ///
    #[pyo3(name = "ioo")]
    pub(crate) fn ioo_gil(&self, other: &Self) -> PyResult<f64> {
        release_gil!(|| {
            self.ioo(other)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    /// Shifts the center of the rotated bounding box by the given amount.
    ///
    /// Parameters
    /// ----------
    /// dx : float
    ///   amount to shift the center of the rotated bounding box in the x-direction
    /// dy : float
    ///   amount to shift the center of the rotated bounding box in the y-direction
    ///
    /// Returns
    /// -------
    /// None
    ///
    pub fn shift(&mut self, dx: f32, dy: f32) {
        match &mut self.data {
            BBoxVariant::Owned(d) => {
                d.xc += dx;
                d.yc += dy;
                d.has_modifications = true;
            }
            BBoxVariant::BorrowedDetectionBox(d) => {
                let mut lock = d.write();
                lock.detection_box.xc += dx;
                lock.detection_box.yc += dy;
                lock.detection_box.has_modifications = true;
            }
            BBoxVariant::BorrowedTrackingBox(d) => {
                let mut lock = d.write();
                if let Some(track_box) = &mut lock.track_box {
                    track_box.xc += dx;
                    track_box.yc += dy;
                    track_box.has_modifications = true;
                }
            }
        }
    }

    /// Creates a new object from (left, top, right, bottom) coordinates.
    ///
    #[staticmethod]
    pub fn ltrb(left: f32, top: f32, right: f32, bottom: f32) -> Self {
        let width = right - left;
        let height = bottom - top;

        let xc = (left + right) / 2.0;
        let yc = (top + bottom) / 2.0;

        Self::new(xc, yc, width, height, None)
    }

    /// Creates a new object from (left, top, width, height) coordinates.
    ///
    #[staticmethod]
    pub fn ltwh(left: f32, top: f32, width: f32, height: f32) -> Self {
        let xc = left + width / 2.0;
        let yc = top + height / 2.0;
        RBBox::new(xc, yc, width, height, None)
    }

    #[getter]
    pub fn get_top(&self) -> PyResult<f32> {
        if self.get_angle().unwrap_or(0.0) == 0.0 {
            Ok(self.get_yc() - self.get_height() / 2.0)
        } else {
            Err(PyValueError::new_err(
                "Cannot get top for rotated bounding box",
            ))
        }
    }

    #[setter]
    pub fn set_top(&mut self, top: f32) -> PyResult<()> {
        if self.get_angle().unwrap_or(0.0) == 0.0 {
            self.set_modifications(true);
            let h = self.get_height();
            self.set_yc(top + h / 2.0);
            Ok(())
        } else {
            Err(PyValueError::new_err(
                "Cannot set top for rotated bounding box",
            ))
        }
    }

    #[getter]
    pub fn get_left(&self) -> PyResult<f32> {
        if self.get_angle().unwrap_or(0.0) == 0.0 {
            Ok(self.get_xc() - self.get_width() / 2.0)
        } else {
            Err(PyValueError::new_err(
                "Cannot get left for rotated bounding box",
            ))
        }
    }

    #[setter]
    pub fn set_left(&mut self, left: f32) -> PyResult<()> {
        if self.get_angle().unwrap_or(0.0) == 0.0 {
            self.set_modifications(true);
            let w = self.get_width();
            self.set_xc(left + w / 2.0);
            Ok(())
        } else {
            Err(PyValueError::new_err(
                "Cannot set left for rotated bounding box",
            ))
        }
    }

    #[getter]
    pub fn get_right(&self) -> PyResult<f32> {
        if self.get_angle().unwrap_or(0.0) == 0.0 {
            Ok(self.get_xc() + self.get_width() / 2.0)
        } else {
            Err(PyValueError::new_err(
                "Cannot get right for rotated bounding box",
            ))
        }
    }

    #[getter]
    pub fn get_bottom(&self) -> PyResult<f32> {
        if self.get_angle().unwrap_or(0.0) == 0.0 {
            Ok(self.get_yc() + self.get_height() / 2.0)
        } else {
            Err(PyValueError::new_err(
                "Cannot get bottom for rotated bounding box",
            ))
        }
    }

    /// Returns (left, top, right, bottom) coordinates.
    ///
    pub fn as_ltrb(&self) -> PyResult<(f32, f32, f32, f32)> {
        if self.get_angle().unwrap_or(0.0) != 0.0 {
            return Err(PyValueError::new_err(
                "Cannot get left, top, width, height for rotated bounding box",
            ));
        }
        let top = self.get_top()?;
        let left = self.get_left()?;
        let bottom = self.get_bottom()?;
        let right = self.get_right()?;

        Ok((left, top, right, bottom))
    }

    /// Returns (left, top, right, bottom) coordinates rounded to integers.
    ///
    pub fn as_ltrb_int(&self) -> PyResult<(i64, i64, i64, i64)> {
        if self.get_angle().unwrap_or(0.0) != 0.0 {
            return Err(PyValueError::new_err(
                "Cannot get left, top, width, height for rotated bounding box",
            ));
        }
        let top = self.get_top()?.floor();
        let left = self.get_left()?.floor();
        let bottom = self.get_bottom()?.ceil();
        let right = self.get_right()?.ceil();

        Ok((left as i64, top as i64, right as i64, bottom as i64))
    }

    /// Returns (left, top, width, height) coordinates.
    ///
    pub fn as_ltwh(&self) -> PyResult<(f32, f32, f32, f32)> {
        if self.get_angle().unwrap_or(0.0) != 0.0 {
            return Err(PyValueError::new_err(
                "Cannot get left, top, width, height for rotated bounding box",
            ));
        }
        let top = self.get_top()?;
        let left = self.get_left()?;
        let width = self.get_width();
        let height = self.get_height();
        Ok((left, top, width, height))
    }

    /// Returns (left, top, width, height) coordinates rounded to integers.
    ///
    pub fn as_ltwh_int(&self) -> PyResult<(i64, i64, i64, i64)> {
        if self.get_angle().unwrap_or(0.0) != 0.0 {
            return Err(PyValueError::new_err(
                "Cannot get left, top, width, height for rotated bounding box",
            ));
        }
        let top = self.get_top()?.floor();
        let left = self.get_left()?.floor();
        let width = self.get_width().ceil();
        let height = self.get_height().ceil();
        Ok((left as i64, top as i64, width as i64, height as i64))
    }

    /// Returns (xc, yc, width, height) coordinates.
    ///
    pub fn as_xcycwh(&self) -> (f32, f32, f32, f32) {
        let xc = self.get_xc();
        let yc = self.get_yc();
        let width = self.get_width();
        let height = self.get_height();
        (xc, yc, width, height)
    }

    /// Returns (xc, yc, width, height) coordinates rounded to integers.
    ///
    pub fn as_xcycwh_int(&self) -> (i64, i64, i64, i64) {
        let xc = self.get_xc();
        let yc = self.get_yc();
        let width = self.get_width();
        let height = self.get_height();
        (xc as i64, yc as i64, width as i64, height as i64)
    }
}

impl RBBox {
    pub fn new_padded(&self, padding: &PaddingDraw) -> Self {
        let (left, right, top, bottom) = (
            padding.left as f32,
            padding.right as f32,
            padding.top as f32,
            padding.bottom as f32,
        );

        let xc = self.get_xc();
        let yc = self.get_yc();
        let width = self.get_width();
        let height = self.get_height();
        let angle = self.get_angle();

        let angle_rad = angle.unwrap_or(0.0) * PI / 180.0;
        let cos_theta = angle_rad.cos();
        let sin_theta = angle_rad.sin();

        let xc = xc + ((right - left) * cos_theta - (bottom - top) * sin_theta) / 2.0;
        let yc = yc + ((right - left) * sin_theta + (bottom - top) * cos_theta) / 2.0;
        let height = height + top + bottom;
        let width = width + left + right;

        Self::new(xc, yc, width, height, angle)
    }

    pub fn ios(&self, other: &Self) -> anyhow::Result<f64> {
        if self.get_area() < EPS || other.get_area() < EPS {
            bail!("Area of one of the bounding boxes is zero. Division by zero is not allowed.");
        }

        let mut area1 = self.get_as_polygonal_area();
        let poly1 = area1.get_polygon();
        let mut area2 = other.get_as_polygonal_area();
        let poly2 = area2.get_polygon();

        let intersection = poly1.intersection(&poly2).unsigned_area();
        Ok(intersection / self.get_area() as f64)
    }

    pub fn ioo(&self, other: &Self) -> anyhow::Result<f64> {
        if self.get_area() < EPS || other.get_area() < EPS {
            bail!("Area of one of the bounding boxes is zero. Division by zero is not allowed.");
        }

        let mut area1 = self.get_as_polygonal_area();
        let poly1 = area1.get_polygon();
        let mut area2 = other.get_as_polygonal_area();
        let poly2 = area2.get_polygon();

        let intersection = poly1.intersection(&poly2).unsigned_area();
        Ok(intersection / other.get_area() as f64)
    }

    pub fn iou(&self, other: &Self) -> anyhow::Result<f64> {
        if self.get_area() < EPS || other.get_area() < EPS {
            bail!("Area of one of the bounding boxes is zero. Division by zero is not allowed.");
        }

        let mut area1 = self.get_as_polygonal_area();
        let poly1 = area1.get_polygon();
        let mut area2 = other.get_as_polygonal_area();
        let poly2 = area2.get_polygon();
        let union = poly1.union(&poly2).unsigned_area();
        if union < EPS as f64 {
            bail!("Union of two bounding boxes is zero. Division by zero is not allowed.",)
        }

        let intersection = poly1.intersection(&poly2).unsigned_area();
        Ok(intersection / union)
    }

    pub fn scale(&mut self, scale_x: f32, scale_y: f32) {
        let angle = self.get_angle().unwrap_or(0.0);
        let xc = self.get_xc();
        let yc = self.get_yc();
        let width = self.get_width();
        let height = self.get_height();

        if angle % 90.0 == 0.0 {
            self.set_xc(xc * scale_x);
            self.set_yc(yc * scale_y);
            self.set_width(width * scale_x);
            self.set_height(height * scale_y);
        } else {
            let scale_x2 = scale_x * scale_x;
            let scale_y2 = scale_y * scale_y;
            let cotan = (angle * PI / 180.0).tan().powi(-1);
            let cotan_2 = cotan * cotan;
            let scale_angle =
                (scale_x * angle.signum() / (scale_x2 + scale_y2 * cotan_2).sqrt()).acos();
            let nscale_height = ((scale_x2 + scale_y2 * cotan_2) / (1.0 + cotan_2)).sqrt();
            let ayh = 1.0 / ((90.0 - angle) / 180.0 * PI).tan();
            let nscale_width = ((scale_x2 + scale_y2 * ayh * ayh) / (1.0 + ayh * ayh)).sqrt();

            self.set_angle(Some(90.0 - (scale_angle * 180.0 / PI)));
            self.set_xc(xc * scale_x);
            self.set_yc(yc * scale_y);
            self.set_width(width * nscale_width);
            self.set_height(height * nscale_height);
        }
    }

    pub fn get_vertices(&self) -> Vec<(f32, f32)> {
        let angle = self.get_angle().unwrap_or(0.0);
        let angle = angle * PI / 180.0;
        let cos = angle.cos();
        let sin = angle.sin();
        let x = self.get_xc();
        let y = self.get_yc();
        let w = self.get_width() / 2.0;
        let h = self.get_height() / 2.0;

        vec![
            (x + w * cos - h * sin, y + w * sin + h * cos),
            (x + w * cos + h * sin, y + w * sin - h * cos),
            (x - w * cos + h * sin, y - w * sin - h * cos),
            (x - w * cos - h * sin, y - w * sin + h * cos),
        ]
    }

    pub fn get_vertices_rounded(&self) -> Vec<(f32, f32)> {
        self.get_vertices()
            .into_iter()
            .map(|(x, y)| (round_2_digits(x), round_2_digits(y)))
            .collect::<Vec<_>>()
    }

    pub fn get_vertices_int(&self) -> Vec<(i64, i64)> {
        self.get_vertices()
            .into_iter()
            .map(|(x, y)| (x as i64, y as i64))
            .collect::<Vec<_>>()
    }

    pub fn get_as_polygonal_area(&self) -> PolygonalArea {
        PolygonalArea::new(
            self.get_vertices()
                .into_iter()
                .map(|(x, y)| Point::new(x as f64, y as f64))
                .collect::<Vec<_>>(),
            None,
        )
    }

    pub fn get_wrapping_bbox(&self) -> PythonBBox {
        if self.get_angle().is_none() {
            PythonBBox::new(
                self.get_xc(),
                self.get_yc(),
                self.get_width(),
                self.get_height(),
            )
        } else {
            let mut vertices = self.get_vertices();
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

    pub fn get_visual_bbox(&self, padding: &PaddingDraw, border_width: i64) -> PyResult<RBBox> {
        if border_width < 0 {
            return Err(PyValueError::new_err(
                "border_width must be greater than or equal to 0",
            ));
        }
        let padding_with_border = PaddingDraw::new(
            padding.left + border_width,
            padding.top + border_width,
            padding.right + border_width,
            padding.bottom + border_width,
        )?;

        Ok(self.new_padded(&padding_with_border))
    }
}

#[pyclass]
#[derive(Clone, Debug)]
#[pyo3(name = "BBox")]
pub struct PythonBBox {
    pub(crate) inner: RBBox,
}

impl PythonBBox {}

/// Auxiliary class representing :py:class:`RBBox` without an angle.
///
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

    pub fn almost_eq(&self, other: &Self, eps: f32) -> bool {
        self.inner.almost_eq(&other.inner, eps)
    }

    pub(crate) fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        self.inner.__richcmp__(&other.inner, op)
    }

    #[pyo3(name = "iou")]
    fn iou_gil(&self, other: &Self) -> PyResult<f64> {
        self.inner.iou_gil(&other.inner)
    }

    #[pyo3(name = "ios")]
    fn ios_gil(&self, other: &Self) -> PyResult<f64> {
        self.inner.ios_gil(&other.inner)
    }

    #[pyo3(name = "ioo")]
    fn ioo_gil(&self, other: &Self) -> PyResult<f64> {
        self.inner.ioo_gil(&other.inner)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn is_modified(&self) -> bool {
        self.inner.is_modified()
    }

    /// Creates a new object. Alias to the ``__init__`` method.
    ///
    #[staticmethod]
    fn constructor(xc: f32, yc: f32, width: f32, height: f32) -> Self {
        Self::new(xc, yc, width, height)
    }

    #[new]
    pub fn new(xc: f32, yc: f32, width: f32, height: f32) -> Self {
        Self {
            inner: RBBox::new(xc, yc, width, height, None),
        }
    }

    /// Creates a new object from (left, top, right, bottom) coordinates.
    ///
    #[staticmethod]
    pub fn ltrb(left: f32, top: f32, right: f32, bottom: f32) -> Self {
        Self {
            inner: RBBox::ltrb(left, top, right, bottom),
        }
    }

    /// Creates a new object from (left, top, width, height) coordinates.
    ///
    #[staticmethod]
    pub fn ltwh(left: f32, top: f32, width: f32, height: f32) -> Self {
        Self {
            inner: RBBox::ltwh(left, top, width, height),
        }
    }

    #[getter]
    pub fn get_xc(&self) -> f32 {
        self.inner.get_xc()
    }

    #[setter]
    pub fn set_xc(&mut self, xc: f32) {
        self.inner.set_xc(xc);
    }

    #[getter]
    pub fn get_yc(&self) -> f32 {
        self.inner.get_yc()
    }

    #[setter]
    pub fn set_yc(&mut self, yc: f32) {
        self.inner.set_yc(yc);
    }

    #[getter]
    pub fn get_width(&self) -> f32 {
        self.inner.get_width()
    }

    #[setter]
    pub fn set_width(&mut self, width: f32) {
        self.inner.set_width(width);
    }

    #[getter]
    pub fn get_height(&self) -> f32 {
        self.inner.get_height()
    }

    #[setter]
    pub fn set_height(&mut self, height: f32) {
        self.inner.set_height(height);
    }

    #[getter]
    pub fn get_top(&self) -> f32 {
        self.inner.get_top().unwrap()
    }

    #[setter]
    pub fn set_top(&mut self, top: f32) -> PyResult<()> {
        self.inner.set_top(top)
    }

    #[getter]
    pub fn get_left(&self) -> f32 {
        self.inner.get_left().unwrap()
    }

    #[setter]
    pub fn set_left(&mut self, left: f32) -> PyResult<()> {
        self.inner.set_left(left)
    }

    #[getter]
    pub fn get_right(&self) -> f32 {
        self.inner.get_right().unwrap()
    }

    #[getter]
    pub fn get_bottom(&self) -> f32 {
        self.inner.get_bottom().unwrap()
    }

    #[getter]
    pub fn get_vertices(&self) -> Vec<(f32, f32)> {
        self.inner.get_vertices()
    }

    #[getter]
    pub fn get_vertices_rounded(&self) -> Vec<(f32, f32)> {
        self.inner.get_vertices_rounded()
    }

    #[getter]
    pub fn get_vertices_int(&self) -> Vec<(i64, i64)> {
        self.inner.get_vertices_int()
    }

    #[getter]
    pub fn get_wrapping_box(&self) -> PythonBBox {
        self.inner.get_wrapping_bbox()
    }

    pub fn visual_box(
        &self,
        padding: &PaddingDraw,
        border_width: i64,
        max_x: f32,
        max_y: f32,
    ) -> PyResult<PythonBBox> {
        if !(border_width >= 0 && max_x >= 0.0 && max_y >= 0.0) {
            return Err(PyValueError::new_err(
                "border_width, max_x and max_y must be greater than or equal to 0",
            ));
        }

        let padding_with_border = PaddingDraw::new(
            padding.left + border_width,
            padding.top + border_width,
            padding.right + border_width,
            padding.bottom + border_width,
        )?;

        let bbox = self.new_padded(&padding_with_border);

        let left = 0.0f32.max(bbox.get_left()).floor();
        let top = 0.0f32.max(bbox.get_top()).floor();
        let right = max_x.min(bbox.get_right()).ceil();
        let bottom = max_y.min(bbox.get_bottom()).ceil();

        let mut width = 1.0f32.max(right - left);
        if width as i64 % 2 != 0 {
            width += 1.0;
        }

        let mut height = 1.0f32.max(bottom - top);
        if height as i64 % 2 != 0 {
            height += 1.0;
        }

        Ok(PythonBBox::new(
            left + width / 2.0,
            top + height / 2.0,
            width,
            height,
        ))
    }

    /// Returns (left, top, right, bottom) coordinates.
    ///
    pub fn as_ltrb(&self) -> (f32, f32, f32, f32) {
        self.inner.as_ltrb().unwrap()
    }

    /// Returns (left, top, right, bottom) coordinates rounded to integers.
    ///
    pub fn as_ltrb_int(&self) -> (i64, i64, i64, i64) {
        self.inner.as_ltrb_int().unwrap()
    }

    /// Returns (left, top, width, height) coordinates.
    ///
    pub fn as_ltwh(&self) -> (f32, f32, f32, f32) {
        self.inner.as_ltwh().unwrap()
    }

    /// Returns (left, top, width, height) coordinates rounded to integers.
    ///
    pub fn as_ltwh_int(&self) -> (i64, i64, i64, i64) {
        self.inner.as_ltwh_int().unwrap()
    }

    /// Returns (xc, yc, width, height) coordinates.
    ///
    pub fn as_xcycwh(&self) -> (f32, f32, f32, f32) {
        self.inner.as_xcycwh()
    }

    /// Returns (xc, yc, width, height) coordinates rounded to integers.
    ///
    pub fn as_xcycwh_int(&self) -> (i64, i64, i64, i64) {
        self.inner.as_xcycwh_int()
    }

    /// Converts the :py:class:`BBox` to a :py:class:`RBBox`.
    ///
    pub fn as_rbbox(&self) -> RBBox {
        self.inner.clone()
    }

    #[pyo3(name = "scale")]
    pub fn scale_gil(&mut self, scale_x: f32, scale_y: f32) {
        self.inner.scale(scale_x, scale_y);
    }

    pub fn shift(&mut self, dx: f32, dy: f32) {
        self.inner.shift(dx, dy);
    }

    pub fn as_polygonal_area(&self) -> PolygonalArea {
        self.inner.get_as_polygonal_area()
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
        new_self.inner.set_modifications(false);
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
        assert_eq!(bbox.get_xc(), 0.0);
        assert_eq!(bbox.get_yc(), 0.0);
        assert_eq!(bbox.get_width(), 200.0);
        assert_eq!(bbox.get_height(), 200.0);
        assert_eq!(bbox.get_angle(), None);
    }

    #[test]
    fn test_scale_with_angle() {
        let mut bbox = RBBox::new(0.0, 0.0, 100.0, 100.0, Some(45.0));
        bbox.scale(2.0, 3.0);
        //dbg!(&bbox);
        assert_eq!(bbox.get_xc(), 0.0);
        assert_eq!(bbox.get_yc(), 0.0);
        assert_eq!(round_2_digits(bbox.get_width()), 254.95);
        assert_eq!(round_2_digits(bbox.get_height()), 254.95);
        assert_eq!(bbox.get_angle().map(round_2_digits), Some(33.69));
    }

    #[test]
    fn test_vertices() {
        let bbox = RBBox::new(0.0, 0.0, 100.0, 100.0, Some(45.0));
        let vertices = bbox.get_vertices_rounded();
        assert_eq!(vertices.len(), 4);
        assert_eq!(vertices[0], (0.0, 70.71));
        assert_eq!(vertices[1], (70.71, 0.0));
        assert_eq!(vertices[2], (-0.0, -70.71));
        assert_eq!(vertices[3], (-70.71, 0.0));
    }

    #[test]
    fn test_wrapping_bbox() {
        let bbox = RBBox::new(0.0, 0.0, 100.0, 100.0, Some(45.0));
        let wrapped = bbox.get_wrapping_bbox();
        assert_eq!(wrapped.inner.get_xc(), 0.0);
        assert_eq!(wrapped.inner.get_yc(), 0.0);
        assert_eq!(round_2_digits(wrapped.inner.get_width()), 141.42);
        assert_eq!(round_2_digits(wrapped.inner.get_height()), 141.42);
        assert_eq!(wrapped.inner.get_angle(), None);

        let bbox = RBBox::new(0.0, 0.0, 50.0, 100.0, None);
        let wrapped = bbox.get_wrapping_bbox();
        assert_eq!(wrapped.inner.get_xc(), 0.0);
        assert_eq!(wrapped.inner.get_yc(), 0.0);
        assert_eq!(round_2_digits(wrapped.inner.get_width()), 50.0);
        assert_eq!(round_2_digits(wrapped.inner.get_height()), 100.0);
        assert_eq!(wrapped.inner.get_angle(), None);

        let bbox = RBBox::new(0.0, 0.0, 50.0, 100.0, Some(90.0));
        let wrapped = bbox.get_wrapping_bbox();
        assert_eq!(wrapped.inner.get_xc(), 0.0);
        assert_eq!(wrapped.inner.get_yc(), 0.0);
        assert_eq!(round_2_digits(wrapped.inner.get_width()), 100.0);
        assert_eq!(round_2_digits(wrapped.inner.get_height()), 50.0);
        assert_eq!(wrapped.inner.get_angle(), None);
    }

    fn get_bbox(angle: Option<f32>) -> RBBox {
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
        let padded = bb.new_padded(&PaddingDraw::new(0, 0, 0, 0).unwrap());
        assert_eq!(padded.get_xc(), bb.get_xc());
        assert_eq!(padded.get_yc(), bb.get_yc());
        assert_eq!(padded.get_width(), bb.get_width());
        assert_eq!(padded.get_height(), bb.get_height());

        let bb = get_bbox(None);
        let padded = bb.new_padded(&PaddingDraw::new(2, 0, 0, 0).unwrap());
        assert_eq!(padded.get_xc(), bb.get_xc() - 1.0);
        assert_eq!(padded.get_yc(), bb.get_yc());
        assert_eq!(padded.get_width(), bb.get_width() + 2.0);
        assert_eq!(padded.get_height(), bb.get_height());

        let bb = get_bbox(None);
        let padded = bb.new_padded(&PaddingDraw::new(0, 2, 0, 0).unwrap());
        assert_eq!(padded.get_xc(), bb.get_xc());
        assert_eq!(padded.get_yc(), bb.get_yc() - 1.0);
        assert_eq!(padded.get_width(), bb.get_width());
        assert_eq!(padded.get_height(), bb.get_height() + 2.0);

        let bb = get_bbox(None);
        let padded = bb.new_padded(&PaddingDraw::new(2, 0, 4, 0).unwrap());
        assert_eq!(padded.get_xc(), bb.get_xc() + 1.0);
        assert_eq!(padded.get_yc(), bb.get_yc());
        assert_eq!(padded.get_width(), bb.get_width() + 6.0);
        assert_eq!(padded.get_height(), bb.get_height());
    }

    #[test]
    fn test_padded_rotated() {
        let bb = get_bbox(Some(90.0));
        let padded = bb.new_padded(&PaddingDraw::new(2, 0, 0, 0).unwrap());
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

    #[test]
    fn test_shift() {
        let mut bb = get_bbox(Some(45.0));
        bb.shift(10.0, 20.0);
        assert_eq!(bb.get_xc(), 10.0);
        assert_eq!(bb.get_yc(), 20.0);
    }

    #[test]
    fn test_various_reprs_non_zero_angle() {
        let mut bb = get_bbox(Some(45.0));
        assert!(bb.as_ltrb().is_err());
        assert!(bb.as_ltrb_int().is_err());
        assert!(bb.as_ltwh().is_err());
        assert!(bb.as_ltwh_int().is_err());
        assert!(bb.get_top().is_err());
        assert!(bb.get_left().is_err());
        assert!(bb.get_bottom().is_err());
        assert!(bb.get_right().is_err());
        assert!(bb.set_top(11.0).is_err());
        assert!(bb.set_left(12.0).is_err());
        assert!(!bb.is_modified());
    }

    #[test]
    fn test_various_reprs_zero_angle() {
        let mut bb = get_bbox(Some(0.0));
        assert!(bb.as_ltrb().is_ok());
        assert!(bb.as_ltrb_int().is_ok());
        assert!(bb.as_ltwh().is_ok());
        assert!(bb.as_ltwh_int().is_ok());
        assert!(bb.get_top().is_ok());
        assert!(bb.get_left().is_ok());
        assert!(bb.get_bottom().is_ok());
        assert!(bb.get_right().is_ok());
        assert!(bb.set_top(11.0).is_ok());
        assert!(bb.set_left(12.0).is_ok());
        assert!(bb.is_modified());
    }

    #[test]
    fn test_various_reprs_none_angle() {
        let mut bb = get_bbox(None);
        assert!(bb.as_ltrb().is_ok());
        assert!(bb.as_ltrb_int().is_ok());
        assert!(bb.as_ltwh().is_ok());
        assert!(bb.as_ltwh_int().is_ok());
        assert!(bb.get_top().is_ok());
        assert!(bb.get_left().is_ok());
        assert!(bb.get_bottom().is_ok());
        assert!(bb.get_right().is_ok());
        assert!(bb.set_top(11.0).is_ok());
        assert!(bb.set_left(12.0).is_ok());
        assert!(bb.is_modified());
    }

    #[test]
    fn test_reprs_correct_values() {
        let mut bb = get_bbox(None);
        bb.set_xc(10.0);
        bb.set_yc(20.0);
        bb.set_width(30.0);
        bb.set_height(40.0);
        let left = bb.get_left().unwrap();
        assert_eq!(left, 10.0 - 30.0 / 2.0);
        let top = bb.get_top().unwrap();
        assert_eq!(top, 20.0 - 40.0 / 2.0);
        let right = bb.get_right().unwrap();
        assert_eq!(right, 10.0 + 30.0 / 2.0);
        let bottom = bb.get_bottom().unwrap();
        assert_eq!(bottom, 20.0 + 40.0 / 2.0);
        let width = bb.get_width();
        let height = bb.get_height();
        let ltrb = bb.as_ltrb().unwrap();
        assert_eq!(ltrb, (left, top, right, bottom));
        let ltrb_int = bb.as_ltrb_int().unwrap();
        assert_eq!(
            ltrb_int,
            (
                left.floor() as i64,
                top.floor() as i64,
                right.ceil() as i64,
                bottom.ceil() as i64
            )
        );

        let ltwh = bb.as_ltwh().unwrap();
        assert_eq!(ltwh, (left, top, width, height));
        let ltwh_int = bb.as_ltwh_int().unwrap();
        assert_eq!(
            ltwh_int,
            (
                left.floor() as i64,
                top.floor() as i64,
                width.ceil() as i64,
                height.ceil() as i64
            )
        );
    }
}
