use crate::primitives::to_json_value::ToSerdeJsonValue;
use crate::primitives::{Intersection, Point, PolygonalArea, RBBox};
use pyo3::types::PyBytes;
use pyo3::{pyclass, pymethods, Py, PyAny};
use rkyv::{Archive, Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone, Default)]
#[archive(check_bytes)]
pub enum AttributeValueVariant {
    Bytes(Vec<i64>, Vec<u8>),
    String(String),
    StringVector(Vec<String>),
    Integer(i64),
    IntegerVector(Vec<i64>),
    Float(f64),
    FloatVector(Vec<f64>),
    Boolean(bool),
    BooleanVector(Vec<bool>),
    BBox(RBBox),
    BBoxVector(Vec<RBBox>),
    Point(Point),
    PointVector(Vec<Point>),
    Polygon(PolygonalArea),
    PolygonVector(Vec<PolygonalArea>),
    Intersection(Intersection),
    #[default]
    None,
}

/// Represents attribute value types for matching
///
#[pyclass]
#[derive(Debug, Clone)]
pub enum NativeAttributeValueType {
    Bytes,
    String,
    StringVector,
    Integer,
    IntegerVector,
    Float,
    FloatVector,
    Boolean,
    BooleanVector,
    BBox,
    BBoxVector,
    Point,
    PointVector,
    Polygon,
    PolygonVector,
    Intersection,
    None,
}

impl ToSerdeJsonValue for AttributeValueVariant {
    fn to_serde_json_value(&self) -> serde_json::Value {
        match self {
            AttributeValueVariant::Bytes(dims, blob) => serde_json::json!({
                "dims": dims,
                "blob": blob,
            }),
            AttributeValueVariant::String(s) => serde_json::json!({
                "string": s,
            }),
            AttributeValueVariant::StringVector(v) => serde_json::json!({
                "string_vector": v,
            }),
            AttributeValueVariant::Integer(i) => serde_json::json!({
                "integer": i,
            }),
            AttributeValueVariant::IntegerVector(v) => serde_json::json!({
                "integer_vector": v,
            }),
            AttributeValueVariant::Float(f) => serde_json::json!({
                "float": f,
            }),
            AttributeValueVariant::FloatVector(v) => serde_json::json!({
                "float_vector": v,
            }),
            AttributeValueVariant::Boolean(b) => serde_json::json!({
                "boolean": b,
            }),
            AttributeValueVariant::BooleanVector(v) => serde_json::json!({
                "boolean_vector": v,
            }),
            AttributeValueVariant::BBox(b) => serde_json::json!({
                "bbox": b.to_serde_json_value(),
            }),
            AttributeValueVariant::BBoxVector(v) => serde_json::json!({
                "bbox_vector": v.iter().map(|b| b.to_serde_json_value()).collect::<Vec<_>>(),
            }),
            AttributeValueVariant::Point(p) => serde_json::json!({
                "point": p.to_serde_json_value(),
            }),
            AttributeValueVariant::PointVector(v) => serde_json::json!({
                "point_vector": v.iter().map(|p| p.to_serde_json_value()).collect::<Vec<_>>(),
            }),
            AttributeValueVariant::Polygon(p) => serde_json::json!({
                "polygon": p.to_serde_json_value(),
            }),
            AttributeValueVariant::PolygonVector(v) => serde_json::json!({
                "polygon_vector": v.iter().map(|p| p.to_serde_json_value()).collect::<Vec<_>>(),
            }),
            AttributeValueVariant::Intersection(i) => serde_json::json!({
                "intersection": i.to_serde_json_value(),
            }),
            AttributeValueVariant::None => serde_json::json!({
                "none": null,
            }),
        }
    }
}

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone, Default)]
#[archive(check_bytes)]
pub struct AttributeValue {
    #[pyo3(get, set)]
    pub confidence: Option<f64>,
    pub(crate) v: AttributeValueVariant,
}

impl ToSerdeJsonValue for AttributeValue {
    fn to_serde_json_value(&self) -> serde_json::Value {
        serde_json::json!({
            "confidence": self.confidence,
            "value": self.v.to_serde_json_value(),
        })
    }
}

#[pymethods]
impl AttributeValue {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Returns the confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`NativeAttributeValueType`
    ///    The type of the attribute value.
    ///
    #[getter]
    fn get_value_type(&self) -> NativeAttributeValueType {
        match &self.v {
            AttributeValueVariant::Bytes(_, _) => NativeAttributeValueType::Bytes,
            AttributeValueVariant::String(_) => NativeAttributeValueType::String,
            AttributeValueVariant::StringVector(_) => NativeAttributeValueType::StringVector,
            AttributeValueVariant::Integer(_) => NativeAttributeValueType::Integer,
            AttributeValueVariant::IntegerVector(_) => NativeAttributeValueType::IntegerVector,
            AttributeValueVariant::Float(_) => NativeAttributeValueType::Float,
            AttributeValueVariant::FloatVector(_) => NativeAttributeValueType::FloatVector,
            AttributeValueVariant::Boolean(_) => NativeAttributeValueType::Boolean,
            AttributeValueVariant::BooleanVector(_) => NativeAttributeValueType::BooleanVector,
            AttributeValueVariant::BBox(_) => NativeAttributeValueType::BBox,
            AttributeValueVariant::BBoxVector(_) => NativeAttributeValueType::BBoxVector,
            AttributeValueVariant::Point(_) => NativeAttributeValueType::Point,
            AttributeValueVariant::PointVector(_) => NativeAttributeValueType::PointVector,
            AttributeValueVariant::Polygon(_) => NativeAttributeValueType::Polygon,
            AttributeValueVariant::PolygonVector(_) => NativeAttributeValueType::PolygonVector,
            AttributeValueVariant::Intersection(_) => NativeAttributeValueType::Intersection,
            AttributeValueVariant::None => NativeAttributeValueType::None,
        }
    }

    /// Creates a new attribute value of type :class:`savant_rs.primitives.geometry.Intersection`.
    ///
    /// Parameters
    /// ----------
    /// int : :class:`savant_rs.primitives.geometry.Intersection`
    ///   The intersection value.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (int, confidence = None))]
    pub fn intersection(int: Intersection, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::Intersection(int),
        }
    }

    /// Creates a new attribute value of type None
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    pub fn none() -> Self {
        Self {
            confidence: None,
            v: AttributeValueVariant::None,
        }
    }

    /// Creates a new attribute value of blob type.
    ///
    /// Parameters
    /// ----------
    /// dims : list of int
    ///   The dimensions of the blob.
    /// blob : List[int]
    ///   The blob.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   
    #[staticmethod]
    #[pyo3(signature = (dims, blob, confidence = None))]
    pub fn bytes_from_list(dims: Vec<i64>, blob: Vec<u8>, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::Bytes(dims, blob),
        }
    }

    /// Creates a new attribute value of blob type.
    ///
    /// Parameters
    /// ----------
    /// dims : list of int
    ///   The dimensions of the blob.
    /// blob : bytes
    ///   The blob.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   
    #[staticmethod]
    #[pyo3(signature = (dims, blob, confidence = None))]
    pub fn bytes(dims: Vec<i64>, blob: &PyBytes, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::Bytes(dims, blob.as_bytes().to_vec()),
        }
    }

    /// Creates a new attribute value of string type.
    ///
    /// Parameters
    /// ----------
    /// s : str
    ///   The string.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (s, confidence = None))]
    pub fn string(s: String, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::String(s),
        }
    }

    /// Creates a new attribute value of list of strings type.
    ///
    /// Parameters
    /// ----------
    /// ss : List[str]
    ///   The list of strings.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (ss, confidence = None))]
    pub fn strings(ss: Vec<String>, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::StringVector(ss),
        }
    }

    /// Creates a new attribute value of integer type.
    ///
    /// Parameters
    /// ----------
    /// i : int
    ///   The integer value.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (i, confidence = None))]
    pub fn integer(i: i64, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::Integer(i),
        }
    }

    /// Creates a new attribute value of list of integers type.
    ///
    /// Parameters
    /// ----------
    /// ii : List[int]
    ///   The list of integers.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (ii, confidence = None))]
    pub fn integers(ii: Vec<i64>, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::IntegerVector(ii),
        }
    }

    /// Creates a new attribute value of float type.
    ///
    /// Parameters
    /// ----------
    /// f : float
    ///   The float value.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (f, confidence = None))]
    pub fn float(f: f64, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::Float(f),
        }
    }

    /// Creates a new attribute value of list of floats type.
    ///
    /// Parameters
    /// ----------
    /// ff : List[float]
    ///   The list of floats.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (ff, confidence = None))]
    pub fn floats(ff: Vec<f64>, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::FloatVector(ff),
        }
    }

    /// Creates a new attribute value of boolean type.
    ///
    /// Parameters
    /// ----------
    /// b : bool
    ///   The boolean value.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (b, confidence = None))]
    pub fn boolean(b: bool, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::Boolean(b),
        }
    }

    /// Creates a new attribute value of list of booleans type.
    ///
    /// Parameters
    /// ----------
    /// bb : List[bool]
    ///   The list of booleans.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (bb, confidence = None))]
    pub fn booleans(bb: Vec<bool>, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::BooleanVector(bb),
        }
    }

    /// Creates a new attribute value of bounding box type.
    ///
    /// Parameters
    /// ----------
    /// bbox : :class:`savant_rs.primitives.geometry.RBBox`
    ///   The bounding box.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (bbox, confidence = None))]
    pub fn bbox(bbox: RBBox, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::BBox(bbox),
        }
    }

    /// Creates a new attribute value of list of bounding boxes type.
    ///
    /// Parameters
    /// ----------
    /// bboxes : List[:class:`savant_rs.primitives.geometry.RBBox`]
    ///   The list of bounding boxes.
    /// confidence : float, optional
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (bboxes, confidence = None))]
    pub fn bboxes(bboxes: Vec<RBBox>, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::BBoxVector(bboxes),
        }
    }

    /// Creates a new attribute value of point type.
    ///
    /// Parameters
    /// ----------
    /// point : :class:`savant_rs.primitives.geometry.Point`
    ///   The point.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (point, confidence = None))]
    pub fn point(point: Point, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::Point(point),
        }
    }

    /// Creates a new attribute value of list of points type.
    ///
    /// Parameters
    /// ----------
    /// points : List[:class:`savant_rs.primitives.geometry.Point`]
    ///   The list of points.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (points, confidence = None))]
    pub fn points(points: Vec<Point>, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::PointVector(points),
        }
    }

    /// Creates a new attribute value of polygon type.
    ///
    /// Parameters
    /// ----------
    /// polygon : :class:`savant_rs.primitives.geometry.PolygonalArea`
    ///   The polygon.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (polygon, confidence = None))]
    pub fn polygon(polygon: PolygonalArea, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::Polygon(polygon),
        }
    }

    /// Creates a new attribute value of list of polygons type.
    ///
    /// Parameters
    /// ----------
    /// polygons : List[:class:`savant_rs.primitives.geometry.PolygonalArea`]
    ///   The list of polygons.
    /// confidence : float, optional
    ///   The confidence of the attribute value.
    ///
    /// Returns
    /// -------
    /// :class:`AttributeValue`
    ///   The attribute value.
    ///
    #[staticmethod]
    #[pyo3(signature = (polygons, confidence = None))]
    pub fn polygons(polygons: Vec<PolygonalArea>, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::PolygonVector(polygons),
        }
    }

    /// Checks if the attribute valus if of None type.
    ///
    pub fn is_none(&self) -> bool {
        matches!(&self.v, AttributeValueVariant::None)
    }

    /// Returns the value of attribute as ``(dims, bytes)`` tuple or None if not a bytes type.
    ///
    /// Returns
    /// -------
    /// Optional[Tuple[List[int], bytes]]
    ///   The value of attribute as ``(dims, bytes)`` tuple or None if not a bytes type.
    ///
    pub fn as_bytes(&self) -> Option<(Vec<i64>, Vec<u8>)> {
        match &self.v {
            AttributeValueVariant::Bytes(dims, bytes) => Some((dims.clone(), bytes.clone())),
            _ => None,
        }
    }

    /// Returns the value of attribute as an :class:`savant_rs.primitives.geometry.Intersection` or None if not an intersection type.
    ///
    /// Returns
    /// -------
    /// Optional[:class:`savant_rs.primitives.geometry.Intersection`]
    ///   The value of attribute as an :class:`savant_rs.primitives.geometry.Intersection` or None if not an intersection type.
    ///
    pub fn as_intersection(&self) -> Option<Intersection> {
        match &self.v {
            AttributeValueVariant::Intersection(i) => Some(i.clone()),
            _ => None,
        }
    }

    /// Returns the value of attribute as a string or None if not a string type.
    ///
    /// Returns
    /// -------
    /// Optional[str]
    ///   The value of attribute as a string or None if not a string type.
    ///
    pub fn as_string(&self) -> Option<String> {
        match &self.v {
            AttributeValueVariant::String(s) => Some(s.clone()),
            _ => None,
        }
    }

    /// Returns the value of attribute as a list of strings or None if not a list of strings type.
    ///
    /// Returns
    /// -------
    /// Optional[List[str]]
    ///   The value of attribute as a list of strings or None if not a list of strings type.
    ///
    pub fn as_strings(&self) -> Option<Vec<String>> {
        match &self.v {
            AttributeValueVariant::StringVector(s) => Some(s.clone()),
            _ => None,
        }
    }

    /// Returns the value of attribute as an integer or None if not an integer type.
    ///
    /// Returns
    /// -------
    /// Optional[int]
    ///   The value of attribute as an integer or None if not an integer type.
    ///
    pub fn as_integer(&self) -> Option<i64> {
        match &self.v {
            AttributeValueVariant::Integer(i) => Some(*i),
            _ => None,
        }
    }

    /// Returns the value of attribute as a list of integers or None if not a list of integers type.
    ///
    /// Returns
    /// -------
    /// Optional[List[int]]
    ///   The value of attribute as a list of integers or None if not a list of integers type.
    ///
    pub fn as_integers(&self) -> Option<Vec<i64>> {
        match &self.v {
            AttributeValueVariant::IntegerVector(i) => Some(i.clone()),
            _ => None,
        }
    }

    /// Returns the value of attribute as a float or None if not a float type.
    ///
    /// Returns
    /// -------
    /// Optional[float]
    ///   The value of attribute as a float or None if not a float type.
    ///
    pub fn as_float(&self) -> Option<f64> {
        match &self.v {
            AttributeValueVariant::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Returns the value of attribute as a list of floats or None if not a list of floats type.
    ///
    /// Returns
    /// -------
    /// Optional[List[float]]
    ///   The value of attribute as a list of floats or None if not a list of floats type.
    ///
    pub fn as_floats(&self) -> Option<Vec<f64>> {
        match &self.v {
            AttributeValueVariant::FloatVector(f) => Some(f.clone()),
            _ => None,
        }
    }

    /// Returns the value of attribute as a boolean or None if not a boolean type.
    ///
    /// Returns
    /// -------
    /// Optional[bool]
    ///   The value of attribute as a boolean or None if not a boolean type.
    ///
    pub fn as_boolean(&self) -> Option<bool> {
        match &self.v {
            AttributeValueVariant::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    /// Returns the value of attribute as a list of booleans or None if not a list of booleans type.
    ///
    /// Returns
    /// -------
    /// Optional[List[bool]]
    ///   The value of attribute as a list of booleans or None if not a list of booleans type.
    ///
    pub fn as_booleans(&self) -> Option<Vec<bool>> {
        match &self.v {
            AttributeValueVariant::BooleanVector(b) => Some(b.clone()),
            _ => None,
        }
    }

    /// Returns the value of attribute as a :class:`savant_rs.primitives.geometry.RBBox` or None if not a bounding box type.
    ///
    /// Returns
    /// -------
    /// Optional[:class:`savant_rs.primitives.geometry.RBBox`]
    ///   The value of attribute as a :class:`savant_rs.primitives.geometry.RBBox` or None if not a bounding box type.
    ///
    pub fn as_bbox(&self) -> Option<RBBox> {
        match &self.v {
            AttributeValueVariant::BBox(bbox) => Some(bbox.clone()),
            _ => None,
        }
    }

    /// Returns the value of attribute as a list of :class:`savant_rs.primitives.geometry.RBBox` or None if not a list of bounding boxes type.
    ///
    /// Returns
    /// -------
    /// Optional[List[:class:`savant_rs.primitives.geometry.RBBox`]]
    ///   The value of attribute as a list of :class:`savant_rs.primitives.geometry.RBBox` or None if not a list of bounding boxes type.
    ///
    pub fn as_bboxes(&self) -> Option<Vec<RBBox>> {
        match &self.v {
            AttributeValueVariant::BBoxVector(bbox) => Some(bbox.clone()),
            _ => None,
        }
    }

    /// Returns the value of attribute as a :class:`savant_rs.primitives.geometry.Point` or None if not a point type.
    ///
    /// Returns
    /// -------
    /// Optional[:class:`savant_rs.primitives.geometry.Point`]
    ///   The value of attribute as a :class:`savant_rs.primitives.geometry.Point` or None if not a point type.
    ///
    pub fn as_point(&self) -> Option<Point> {
        match &self.v {
            AttributeValueVariant::Point(point) => Some(point.clone()),
            _ => None,
        }
    }

    /// Returns the value of attribute as a list of :class:`savant_rs.primitives.geometry.Point` or None if not a list of points type.
    ///
    /// Returns
    /// -------
    /// Optional[List[:class:`savant_rs.primitives.geometry.Point`]]
    ///   The value of attribute as a list of :class:`savant_rs.primitives.geometry.Point` or None if not a list of points type.
    ///
    pub fn as_points(&self) -> Option<Vec<Point>> {
        match &self.v {
            AttributeValueVariant::PointVector(point) => Some(point.clone()),
            _ => None,
        }
    }

    /// Returns the value of attribute as a :class:`savant_rs.primitives.geometry.PolygonalArea` or None if not a polygon type.
    ///
    /// Returns
    /// -------
    /// Optional[:class:`savant_rs.primitives.geometry.PolygonalArea`]
    ///   The value of attribute as a :class:`savant_rs.primitives.geometry.PolygonalArea` or None if not a polygon type.
    ///
    pub fn as_polygon(&self) -> Option<PolygonalArea> {
        match &self.v {
            AttributeValueVariant::Polygon(polygon) => Some(polygon.clone()),
            _ => None,
        }
    }

    /// Returns the value of attribute as a list of :class:`savant_rs.primitives.geometry.PolygonalArea` or None if not a list of polygons type.
    ///
    /// Returns
    /// -------
    /// Optional[List[:class:`savant_rs.primitives.geometry.PolygonalArea`]]
    ///   The value of attribute as a list of :class:`savant_rs.primitives.geometry.PolygonalArea` or None if not a list of polygons type.
    ///
    pub fn as_polygons(&self) -> Option<Vec<PolygonalArea>> {
        match &self.v {
            AttributeValueVariant::PolygonVector(polygon) => Some(polygon.clone()),
            _ => None,
        }
    }
}

/// Attribute represents a specific knowledge about certain entity. The attribute is identified by ``(creator, label)`` pair which is unique within the entity.
/// The attribute value is a list of values, each of which has a confidence score. The attribute may include additional information in the form of a hint.
/// There are two kinds of attributes: persistent and non-persistent. Persistent attributes are serialized, while non-persistent are not.
///
/// The list nature of attribute values is used to represent complex values of the same attribute.
/// For example, the attribute ``(person_profiler, bio)`` may include values in the form ``["Age", 32, "Gender", None, "Height", 186]``. Each element of the
/// list is :class:`AttributeValue`.
///
#[pyclass]
#[derive(
    Archive, Deserialize, Serialize, Debug, PartialEq, Clone, derive_builder::Builder, Default,
)]
#[archive(check_bytes)]
pub struct Attribute {
    #[pyo3(get)]
    pub creator: String,
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub values: Vec<AttributeValue>,
    #[pyo3(get)]
    pub hint: Option<String>,
    #[pyo3(get)]
    #[builder(default = "true")]
    pub is_persistent: bool,
}

impl ToSerdeJsonValue for Attribute {
    fn to_serde_json_value(&self) -> serde_json::Value {
        serde_json::json!({
            "creator": self.creator,
            "name": self.name,
            "values": self.values.iter().map(|v| v.to_serde_json_value()).collect::<Vec<_>>(),
            "hint": self.hint,
        })
    }
}

#[pymethods]
impl Attribute {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[new]
    #[pyo3(signature = (creator, name , values, hint = None, is_persistent = true))]
    pub fn new(
        creator: String,
        name: String,
        values: Vec<AttributeValue>,
        hint: Option<String>,
        is_persistent: bool,
    ) -> Self {
        Self {
            is_persistent,
            creator,
            name,
            values,
            hint,
        }
    }

    /// Alias to constructor method. Creates a persistent attribute.
    ///
    /// Parameters
    /// ----------
    /// creator : str
    ///   The creator of the attribute.
    /// name : str
    ///   The name of the attribute.
    /// values : List[:class:`AttributeValue`]
    ///   The values of the attribute.
    /// hint : str, optional
    ///   The hint of the attribute. The hint is a user-defined string that may contain additional information about the attribute.
    ///
    /// Returns
    /// -------
    /// :class:`Attribute`
    ///   The created attribute.
    ///
    #[staticmethod]
    pub fn persistent(
        creator: String,
        name: String,
        values: Vec<AttributeValue>,
        hint: Option<String>,
    ) -> Self {
        Self {
            is_persistent: true,
            creator,
            name,
            values,
            hint,
        }
    }

    /// Alias to constructor method for non-persistent attributes.
    ///
    /// Parameters
    /// ----------
    /// creator : str
    ///   The creator of the attribute.
    /// name : str
    ///   The name of the attribute.
    /// values : List[:class:`AttributeValue`]
    ///   The values of the attribute.
    /// hint : str, optional
    ///   The hint of the attribute. The hint is a user-defined string that may contain additional information about the attribute.
    ///
    /// Returns
    /// -------
    /// :class:`Attribute`
    ///   The created attribute.
    ///
    #[staticmethod]
    pub fn temporary(
        creator: String,
        name: String,
        values: Vec<AttributeValue>,
        hint: Option<String>,
    ) -> Self {
        Self {
            is_persistent: false,
            creator,
            name,
            values,
            hint,
        }
    }

    /// Returns ``True`` if the attribute is persistent, ``False`` otherwise.
    ///
    /// Returns
    /// -------
    /// bool
    ///   ``True`` if the attribute is persistent, ``False`` otherwise.
    ///
    pub fn is_temporary(&self) -> bool {
        !self.is_persistent
    }

    /// Changes the attribute to be persistent.
    ///
    /// Returns
    /// -------
    /// None
    ///   The attribute is changed in-place.
    ///
    pub fn make_persistent(&mut self) {
        self.is_persistent = true;
    }

    /// Changes the attribute to be non-persistent.
    ///
    /// Returns
    /// -------
    /// None
    ///   The attribute is changed in-place.
    ///
    pub fn make_temporary(&mut self) {
        self.is_persistent = false;
    }

    /// Returns the creator of the attribute.
    ///
    /// Returns
    /// -------
    /// str
    ///   The creator of the attribute.
    ///
    #[getter]
    pub fn get_creator(&self) -> String {
        self.creator.clone()
    }

    /// Returns the name of the attribute.
    ///
    /// Returns
    /// -------
    /// str
    ///   The name of the attribute.
    ///
    #[getter]
    pub fn get_name(&self) -> String {
        self.name.clone()
    }

    /// Returns the values of the attribute. The values are returned as copies, changing them will not change the attribute. To change the values of the
    /// attribute, use assignment to the ``values`` attribute.
    ///
    /// Returns
    /// -------
    /// List[:class:`AttributeValue`]
    ///   The values of the attribute.
    ///
    #[getter]
    pub fn get_values(&self) -> Vec<AttributeValue> {
        self.values.clone()
    }

    /// Returns the hint of the attribute.
    ///
    /// Returns
    /// -------
    /// str or None
    ///   The hint of the attribute or ``None`` if no hint is set.
    ///
    #[getter]
    pub fn get_hint(&self) -> Option<String> {
        self.hint.clone()
    }

    /// Sets the hint of the attribute.
    ///
    /// Parameters
    /// ----------
    /// hint : str or None
    ///   The hint of the attribute or ``None`` if no hint is set.
    ///
    #[setter]
    pub fn set_hint(&mut self, hint: Option<String>) {
        self.hint = hint;
    }

    /// Sets the values of the attribute.
    ///
    /// Parameters
    /// ----------
    /// values : List[:class:`AttributeValue`]
    ///   The values of the attribute.
    ///
    #[setter]
    pub fn set_values(&mut self, values: Vec<AttributeValue>) {
        self.values = values;
    }
}

pub trait AttributeMethods {
    fn exclude_temporary_attributes(&self) -> Vec<Attribute>;
    fn restore_attributes(&self, attributes: Vec<Attribute>);
    fn get_attributes(&self) -> Vec<(String, String)>;
    fn get_attribute(&self, creator: String, name: String) -> Option<Attribute>;
    fn delete_attribute(&self, creator: String, name: String) -> Option<Attribute>;
    fn set_attribute(&self, attribute: Attribute) -> Option<Attribute>;
    fn clear_attributes(&self);
    fn delete_attributes(&self, creator: Option<String>, names: Vec<String>);
    fn find_attributes(
        &self,
        creator: Option<String>,
        names: Vec<String>,
        hint: Option<String>,
    ) -> Vec<(String, String)>;
}

pub trait Attributive: Send {
    fn get_attributes_ref(&self) -> &HashMap<(String, String), Attribute>;
    fn get_attributes_ref_mut(&mut self) -> &mut HashMap<(String, String), Attribute>;
    fn take_attributes(&mut self) -> HashMap<(String, String), Attribute>;
    fn place_attributes(&mut self, attributes: HashMap<(String, String), Attribute>);

    fn exclude_temporary_attributes(&mut self) -> Vec<Attribute> {
        let attributes = self.take_attributes();
        let (retained, removed): (Vec<Attribute>, Vec<Attribute>) =
            attributes.into_values().partition(|a| !a.is_temporary());

        self.place_attributes(
            retained
                .into_iter()
                .map(|a| ((a.creator.clone(), a.name.clone()), a))
                .collect(),
        );

        removed
    }

    fn restore_attributes(&mut self, attributes: Vec<Attribute>) {
        let attrs = self.get_attributes_ref_mut();
        attributes.into_iter().for_each(|a| {
            attrs.insert((a.creator.clone(), a.name.clone()), a);
        })
    }

    fn get_attributes(&self) -> Vec<(String, String)> {
        self.get_attributes_ref()
            .iter()
            .map(|((creator, name), _)| (creator.clone(), name.clone()))
            .collect()
    }

    fn get_attribute(&self, creator: String, name: String) -> Option<Attribute> {
        self.get_attributes_ref().get(&(creator, name)).cloned()
    }

    fn delete_attribute(&mut self, creator: String, name: String) -> Option<Attribute> {
        self.get_attributes_ref_mut().remove(&(creator, name))
    }

    fn set_attribute(&mut self, attribute: Attribute) -> Option<Attribute> {
        self.get_attributes_ref_mut().insert(
            (attribute.creator.clone(), attribute.name.clone()),
            attribute,
        )
    }

    fn clear_attributes(&mut self) {
        self.get_attributes_ref_mut().clear();
    }

    fn delete_attributes(&mut self, creator: Option<String>, names: Vec<String>) {
        self.get_attributes_ref_mut().retain(|(c, label), _| {
            if let Some(creator) = &creator {
                if c != creator {
                    return true;
                }
            }

            if !names.is_empty() && !names.contains(label) {
                return true;
            }

            false
        });
    }

    fn find_attributes(
        &self,
        creator: Option<String>,
        names: Vec<String>,
        hint: Option<String>,
    ) -> Vec<(String, String)> {
        self.get_attributes_ref()
            .iter()
            .filter(|((_, _), a)| {
                if let Some(creator) = &creator {
                    if a.creator != *creator {
                        return false;
                    }
                }

                if !names.is_empty() && !names.contains(&a.name) {
                    return false;
                }

                if let Some(hint) = &hint {
                    if a.hint.as_ref() != Some(hint) {
                        return false;
                    }
                }

                true
            })
            .map(|((c, n), _)| (c.clone(), n.clone()))
            .collect()
    }
}
