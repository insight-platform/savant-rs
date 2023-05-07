use crate::primitives::{BBox, Point, PolygonalArea};
use pyo3::{pyclass, pymethods, Py, PyAny};
use rkyv::{Archive, Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone, Default)]
#[archive(check_bytes)]
pub enum ValueVariant {
    Bytes((Vec<i64>, Vec<u8>)),
    String(String),
    StringVector(Vec<String>),
    Integer(i64),
    IntegerVector(Vec<i64>),
    Float(f64),
    FloatVector(Vec<f64>),
    Boolean(bool),
    BooleanVector(Vec<bool>),
    BBox(BBox),
    BBoxVector(Vec<BBox>),
    Point(Point),
    PointVector(Vec<Point>),
    Polygon(PolygonalArea),
    PolygonVector(Vec<PolygonalArea>),
    KeyPoints(Vec<(Point, HashMap<String, String>)>),
    #[default]
    None,
}

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone, Default)]
#[archive(check_bytes)]
pub struct Value {
    v: ValueVariant,
}

#[pymethods]
impl Value {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[staticmethod]
    pub fn none() -> Self {
        Self {
            v: ValueVariant::None,
        }
    }

    #[staticmethod]
    pub fn bytes(dims: Vec<i64>, blob: Vec<u8>) -> Self {
        Self {
            v: ValueVariant::Bytes((dims, blob)),
        }
    }

    #[staticmethod]
    pub fn string(s: String) -> Self {
        Self {
            v: ValueVariant::String(s),
        }
    }

    #[staticmethod]
    pub fn strings(s: Vec<String>) -> Self {
        Self {
            v: ValueVariant::StringVector(s),
        }
    }

    #[staticmethod]
    pub fn integer(i: i64) -> Self {
        Self {
            v: ValueVariant::Integer(i),
        }
    }

    #[staticmethod]
    pub fn integers(i: Vec<i64>) -> Self {
        Self {
            v: ValueVariant::IntegerVector(i),
        }
    }

    #[staticmethod]
    pub fn float(f: f64) -> Self {
        Self {
            v: ValueVariant::Float(f),
        }
    }

    #[staticmethod]
    pub fn floats(f: Vec<f64>) -> Self {
        Self {
            v: ValueVariant::FloatVector(f),
        }
    }

    #[staticmethod]
    pub fn boolean(b: bool) -> Self {
        Self {
            v: ValueVariant::Boolean(b),
        }
    }

    #[staticmethod]
    pub fn booleans(b: Vec<bool>) -> Self {
        Self {
            v: ValueVariant::BooleanVector(b),
        }
    }

    #[staticmethod]
    pub fn bbox(bbox: BBox) -> Self {
        Self {
            v: ValueVariant::BBox(bbox),
        }
    }

    #[staticmethod]
    pub fn bboxes(bboxes: Vec<BBox>) -> Self {
        Self {
            v: ValueVariant::BBoxVector(bboxes),
        }
    }

    #[staticmethod]
    pub fn point(point: Point) -> Self {
        Self {
            v: ValueVariant::Point(point),
        }
    }

    #[staticmethod]
    pub fn points(points: Vec<Point>) -> Self {
        Self {
            v: ValueVariant::PointVector(points),
        }
    }

    #[staticmethod]
    pub fn polygon(polygon: PolygonalArea) -> Self {
        Self {
            v: ValueVariant::Polygon(polygon),
        }
    }

    #[staticmethod]
    pub fn polygons(polygons: Vec<PolygonalArea>) -> Self {
        Self {
            v: ValueVariant::PolygonVector(polygons),
        }
    }

    #[staticmethod]
    pub fn keypoints(keypoints: Vec<(Point, HashMap<String, String>)>) -> Self {
        Self {
            v: ValueVariant::KeyPoints(keypoints),
        }
    }

    pub fn is_none(&self) -> bool {
        matches!(&self.v, ValueVariant::None)
    }

    pub fn as_bytes(&self) -> Option<(Vec<i64>, Vec<u8>)> {
        match &self.v {
            ValueVariant::Bytes(b) => Some(b.clone()),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<String> {
        match &self.v {
            ValueVariant::String(s) => Some(s.clone()),
            _ => None,
        }
    }

    pub fn as_strings(&self) -> Option<Vec<String>> {
        match &self.v {
            ValueVariant::StringVector(s) => Some(s.clone()),
            _ => None,
        }
    }

    pub fn as_integer(&self) -> Option<i64> {
        match &self.v {
            ValueVariant::Integer(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_integers(&self) -> Option<Vec<i64>> {
        match &self.v {
            ValueVariant::IntegerVector(i) => Some(i.clone()),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match &self.v {
            ValueVariant::Float(f) => Some(*f),
            _ => None,
        }
    }

    pub fn as_floats(&self) -> Option<Vec<f64>> {
        match &self.v {
            ValueVariant::FloatVector(f) => Some(f.clone()),
            _ => None,
        }
    }

    pub fn as_boolean(&self) -> Option<bool> {
        match &self.v {
            ValueVariant::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_booleans(&self) -> Option<Vec<bool>> {
        match &self.v {
            ValueVariant::BooleanVector(b) => Some(b.clone()),
            _ => None,
        }
    }

    pub fn as_bbox(&self) -> Option<BBox> {
        match &self.v {
            ValueVariant::BBox(bbox) => Some(bbox.clone()),
            _ => None,
        }
    }

    pub fn as_bboxes(&self) -> Option<Vec<BBox>> {
        match &self.v {
            ValueVariant::BBoxVector(bbox) => Some(bbox.clone()),
            _ => None,
        }
    }

    pub fn as_point(&self) -> Option<Point> {
        match &self.v {
            ValueVariant::Point(point) => Some(point.clone()),
            _ => None,
        }
    }

    pub fn as_points(&self) -> Option<Vec<Point>> {
        match &self.v {
            ValueVariant::PointVector(point) => Some(point.clone()),
            _ => None,
        }
    }

    pub fn as_polygon(&self) -> Option<PolygonalArea> {
        match &self.v {
            ValueVariant::Polygon(polygon) => Some(polygon.clone()),
            _ => None,
        }
    }

    pub fn as_polygons(&self) -> Option<Vec<PolygonalArea>> {
        match &self.v {
            ValueVariant::PolygonVector(polygon) => Some(polygon.clone()),
            _ => None,
        }
    }

    pub fn as_keypoints(&self) -> Option<Vec<(Point, HashMap<String, String>)>> {
        match &self.v {
            ValueVariant::KeyPoints(keypoints) => Some(keypoints.clone()),
            _ => None,
        }
    }
}

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
    pub value: Value,
    #[pyo3(get)]
    pub confidence: Option<f64>,
    #[pyo3(get)]
    pub hint: Option<String>,
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
    pub fn new(
        creator: String,
        name: String,
        value: Value,
        confidence: Option<f64>,
        hint: Option<String>,
    ) -> Self {
        Self {
            creator,
            name,
            value,
            confidence,
            hint,
        }
    }
}
