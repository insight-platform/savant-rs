use crate::primitives::{PaddingDraw, PolygonalArea};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::pyclass::CompareOp;
use pyo3::{pyclass, pymethods, Py, PyAny, PyResult};
use savant_core::json_api::ToSerdeJsonValue;
use savant_core::primitives::{rust, OwnedRBBoxData};

/// Allows configuring what kind of Intersection over Something to use.
///
/// IoU - Intersection over Union
/// IoSelf - Intersection over Self (Intersection / Area of Self)
/// IoOther - Intersection over Other (Intersection / Area of Other)
///
#[pyclass]
#[derive(Debug, Clone)]
pub(crate) enum BBoxMetricType {
    IoU,
    IoSelf,
    IoOther,
}

impl From<BBoxMetricType> for rust::BBoxMetricType {
    fn from(value: BBoxMetricType) -> Self {
        match value {
            BBoxMetricType::IoU => rust::BBoxMetricType::IoU,
            BBoxMetricType::IoSelf => rust::BBoxMetricType::IoSelf,
            BBoxMetricType::IoOther => rust::BBoxMetricType::IoOther,
        }
    }
}

impl From<rust::BBoxMetricType> for BBoxMetricType {
    fn from(value: rust::BBoxMetricType) -> Self {
        match value {
            rust::BBoxMetricType::IoU => BBoxMetricType::IoU,
            rust::BBoxMetricType::IoSelf => BBoxMetricType::IoSelf,
            rust::BBoxMetricType::IoOther => BBoxMetricType::IoOther,
        }
    }
}

/// Represents a bounding box with an optional rotation angle in degrees.
///
#[pyclass]
#[derive(Debug, Clone)]
pub struct RBBox(pub(crate) rust::RBBox);

impl PartialEq for RBBox {
    fn eq(&self, other: &Self) -> bool {
        self.0.geometric_eq(&other.0)
    }
}

impl ToSerdeJsonValue for RBBox {
    fn to_serde_json_value(&self) -> serde_json::Value {
        self.0.to_serde_json_value()
    }
}

#[pymethods]
impl RBBox {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", &self.0)
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
        self.0.get_area()
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
        self.0.geometric_eq(&other.0)
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
        self.0.almost_eq(&other.0, eps)
    }

    pub(crate) fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge => Err(
                PyNotImplementedError::new_err("Comparison ops Ge/Gt/Le/Lt are not implemented"),
            ),
            CompareOp::Eq => Ok(self.0.geometric_eq(&other.0)),
            CompareOp::Ne => Ok(!self.0.geometric_eq(&other.0)),
        }
    }

    /// Access and change the x center of the bbox.
    ///
    #[getter]
    pub fn get_xc(&self) -> f32 {
        self.0.get_xc()
    }

    /// Access and change the y center of the bbox.
    ///
    #[getter]
    pub fn get_yc(&self) -> f32 {
        self.0.get_yc()
    }

    /// Access and change the width of the bbox.
    ///
    #[getter]
    pub fn get_width(&self) -> f32 {
        self.0.get_width()
    }

    /// Access and change the height of the bbox.
    ///
    #[getter]
    pub fn get_height(&self) -> f32 {
        self.0.get_height()
    }

    /// Access and change the angle of the bbox. To unset the angle use None as a value.
    ///
    #[getter]
    pub fn get_angle(&self) -> Option<f32> {
        self.0.get_angle()
    }

    /// Access the ratio between width and height.
    ///
    #[getter]
    pub fn get_width_to_height_ratio(&self) -> f32 {
        self.0.get_width_to_height_ratio()
    }

    /// Flag indicating if the bbox has been modified.
    ///
    pub fn is_modified(&self) -> bool {
        self.0.is_modified()
    }

    /// Resets the modification flag.
    ///
    pub fn set_modifications(&mut self, value: bool) {
        self.0.set_modifications(value)
    }

    #[setter]
    pub fn set_xc(&mut self, xc: f32) {
        self.0.set_xc(xc)
    }

    #[setter]
    pub fn set_yc(&mut self, yc: f32) {
        self.0.set_yc(yc)
    }

    #[setter]
    pub fn set_width(&mut self, width: f32) {
        self.0.set_width(width)
    }

    #[setter]
    pub fn set_height(&mut self, height: f32) {
        self.0.set_height(height)
    }

    #[setter]
    pub fn set_angle(&mut self, angle: Option<f32>) {
        self.0.set_angle(angle)
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
        Self(rust::RBBox::new(xc, yc, width, height, angle))
    }

    /// Scales the bbox by given factors. The function is GIL-free. Scaling happens in-place.
    ///
    pub fn scale(&mut self, scale_x: f32, scale_y: f32) {
        self.0.scale(scale_x, scale_y)
    }

    /// Returns vertices of the bbox. The property is GIL-free.
    ///
    /// Returns
    /// -------
    /// list of tuples (float, float)
    ///   vertices of the bbox
    ///
    #[getter]
    pub fn get_vertices(&self) -> Vec<(f32, f32)> {
        self.0.get_vertices()
    }

    /// Returns vertices of the bbox rounded to 2 decimal digits. The property is GIL-free.
    ///
    /// Returns
    /// -------
    /// list of tuples (float, float)
    ///   vertices of the bbox rounded to 2 decimal digits
    ///
    #[getter]
    pub fn get_vertices_rounded(&self) -> Vec<(f32, f32)> {
        self.0.get_vertices_rounded()
    }

    /// Returns vertices of the bbox rounded to integers. The property is GIL-free.
    ///
    /// Returns
    /// -------
    /// list of tuples (int, int)
    ///   vertices of the bbox rounded to integers
    ///
    #[getter]
    pub fn get_vertices_int(&self) -> Vec<(i64, i64)> {
        self.0.get_vertices_int()
    }

    /// Returns bbox as a polygonal area. The property is GIL-free.
    ///
    /// Returns
    /// -------
    /// :py:class:`PolygonalArea`
    ///   polygonal area of the bbox
    ///
    pub fn as_polygonal_area(&self) -> PolygonalArea {
        PolygonalArea(self.0.get_as_polygonal_area())
    }

    /// Returns axis-aligned bounding box wrapping the bbox. The property is GIL-free.
    ///
    /// Returns
    /// -------
    /// :py:class:`BBox`
    ///   axis-aligned bounding box wrapping the bbox
    ///
    #[getter]
    pub fn get_wrapping_box(&self) -> BBox {
        let bb = self.0.get_wrapping_bbox();
        BBox::new(bb.get_xc(), bb.get_yc(), bb.get_width(), bb.get_height())
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
    pub fn get_visual_box(&self, padding: &PaddingDraw, border_width: i64) -> PyResult<RBBox> {
        let rust_padding = &padding.0;
        let rbbox_res = self.0.get_visual_bbox(rust_padding, border_width);
        rbbox_res
            .map(RBBox)
            .map_err(|e| {
            PyValueError::new_err(format!(
                "Failed to get visual box for bbox: {:?}, padding: {:?}, border_width: {}, error: {}",
                self.0, padding, border_width, e
            ))
        })
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
    pub fn new_padded(&self, padding: &PaddingDraw) -> Self {
        let padding = &padding.0;
        let rbbox = self.0.new_padded(padding);
        RBBox(rbbox)
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
        let data: OwnedRBBoxData = OwnedRBBoxData::from(&self.0);

        let mut new_self = Self(rust::RBBox::from(data));

        new_self.set_modifications(false);
        new_self
    }

    /// Calculates the intersection over union (IoU) of two rotated bounding boxes.
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
    pub(crate) fn iou(&self, other: &Self) -> PyResult<f32> {
        self.0
            .iou(&other.0)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Calculates the intersection over self (IoS) of two rotated bounding boxes.
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
    pub(crate) fn ios(&self, other: &Self) -> PyResult<f32> {
        self.0
            .ios(&other.0)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Calculates the intersection over other (IoO) of two rotated bounding boxes.
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
    pub(crate) fn ioo(&self, other: &Self) -> PyResult<f32> {
        self.0
            .ioo(&other.0)
            .map_err(|e| PyValueError::new_err(e.to_string()))
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
        self.0.shift(dx, dy)
    }

    /// Creates a new object from (left, top, right, bottom) coordinates.
    ///
    #[staticmethod]
    pub fn ltrb(left: f32, top: f32, right: f32, bottom: f32) -> Self {
        Self(rust::RBBox::ltrb(left, top, right, bottom))
    }

    /// Creates a new object from (left, top, width, height) coordinates.
    ///
    #[staticmethod]
    pub fn ltwh(left: f32, top: f32, width: f32, height: f32) -> Self {
        Self(rust::RBBox::ltwh(left, top, width, height))
    }

    #[getter]
    pub fn get_top(&self) -> PyResult<f32> {
        self.0
            .get_top()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[setter]
    pub fn set_top(&mut self, top: f32) -> PyResult<()> {
        self.0
            .set_top(top)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    pub fn get_left(&self) -> PyResult<f32> {
        self.0
            .get_left()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[setter]
    pub fn set_left(&mut self, left: f32) -> PyResult<()> {
        self.0
            .set_left(left)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    pub fn get_right(&self) -> PyResult<f32> {
        self.0
            .get_right()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    pub fn get_bottom(&self) -> PyResult<f32> {
        self.0
            .get_bottom()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Returns (left, top, right, bottom) coordinates.
    ///
    pub fn as_ltrb(&self) -> PyResult<(f32, f32, f32, f32)> {
        self.0
            .as_ltrb()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Returns (left, top, right, bottom) coordinates rounded to integers.
    ///
    pub fn as_ltrb_int(&self) -> PyResult<(i64, i64, i64, i64)> {
        self.0
            .as_ltrb_int()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Returns (left, top, width, height) coordinates.
    ///
    pub fn as_ltwh(&self) -> PyResult<(f32, f32, f32, f32)> {
        self.0
            .as_ltwh()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Returns (left, top, width, height) coordinates rounded to integers.
    ///
    pub fn as_ltwh_int(&self) -> PyResult<(i64, i64, i64, i64)> {
        self.0
            .as_ltwh_int()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Returns (xc, yc, width, height) coordinates.
    ///
    pub fn as_xcycwh(&self) -> (f32, f32, f32, f32) {
        self.0.as_xcycwh()
    }

    /// Returns (xc, yc, width, height) coordinates rounded to integers.
    ///
    pub fn as_xcycwh_int(&self) -> (i64, i64, i64, i64) {
        self.0.as_xcycwh_int()
    }
}

#[pyclass]
#[derive(Clone, Debug)]
#[pyo3(name = "BBox")]
pub struct BBox {
    pub(crate) inner: RBBox,
}

impl BBox {}

/// Auxiliary class representing :py:class:`RBBox` without an angle.
///
#[pymethods]
impl BBox {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner.__repr__())
    }

    fn __str__(&self) -> String {
        self.__repr__()
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

    fn iou(&self, other: &Self) -> PyResult<f32> {
        self.inner.iou(&other.inner)
    }

    fn ios(&self, other: &Self) -> PyResult<f32> {
        self.inner.ios(&other.inner)
    }

    fn ioo(&self, other: &Self) -> PyResult<f32> {
        self.inner.ioo(&other.inner)
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
    pub fn get_wrapping_box(&self) -> BBox {
        self.inner.get_wrapping_box()
    }

    pub fn visual_box(
        &self,
        padding: &PaddingDraw,
        border_width: i64,
        max_x: f32,
        max_y: f32,
    ) -> PyResult<BBox> {
        if !(border_width >= 0 && max_x >= 0.0 && max_y >= 0.0) {
            return Err(PyValueError::new_err(
                "border_width, max_x and max_y must be greater than or equal to 0",
            ));
        }

        let padding_with_border = PaddingDraw::new(
            padding.left() + border_width,
            padding.top() + border_width,
            padding.right() + border_width,
            padding.bottom() + border_width,
        )?;

        let bbox = self.new_padded(&padding_with_border);

        let left = 2.0f32.max(bbox.get_left()).ceil();
        let top = 2.0f32.max(bbox.get_top()).ceil();
        let right = (max_x - 2.0).min(bbox.get_right()).floor();
        let bottom = (max_y - 2.0).min(bbox.get_bottom()).floor();

        let mut width = 1.0f32.max(right - left);
        if width as i64 % 2 != 0 {
            width -= 1.0;
        }

        let mut height = 1.0f32.max(bottom - top);
        if height as i64 % 2 != 0 {
            height -= 1.0;
        }

        Ok(BBox::new(
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
    pub fn scale_py(&mut self, scale_x: f32, scale_y: f32) {
        self.inner.scale(scale_x, scale_y);
    }

    pub fn shift(&mut self, dx: f32, dy: f32) {
        self.inner.shift(dx, dy);
    }

    pub fn as_polygonal_area(&self) -> PolygonalArea {
        self.inner.as_polygonal_area()
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

#[pyclass]
#[derive(Debug, Clone, Copy)]
pub(crate) struct VideoObjectBBoxTransformation(pub(crate) rust::VideoObjectBBoxTransformation);

#[pymethods]
impl VideoObjectBBoxTransformation {
    #[staticmethod]
    fn scale(x: f32, y: f32) -> Self {
        Self(rust::VideoObjectBBoxTransformation::Scale(x, y))
    }

    #[staticmethod]
    fn shift(x: f32, y: f32) -> Self {
        Self(rust::VideoObjectBBoxTransformation::Shift(x, y))
    }
}
