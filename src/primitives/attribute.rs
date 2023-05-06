use crate::primitives::{BBox, Point, PolygonalArea};
use pyo3::{pyclass, pymethods, Py, PyAny};
use rkyv::{Archive, Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub enum ValueVariant {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    BBox(BBox),
    Point(Point),
    Polygon(PolygonalArea),
    PolyLine(Vec<Point>),
    KeyPoints(Vec<(Point, HashMap<String, String>)>),
}

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
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
    pub fn new_string(s: String) -> Self {
        Self {
            v: ValueVariant::String(s),
        }
    }

    #[staticmethod]
    pub fn new_integer(i: i64) -> Self {
        Self {
            v: ValueVariant::Integer(i),
        }
    }

    #[staticmethod]
    pub fn new_float(f: f64) -> Self {
        Self {
            v: ValueVariant::Float(f),
        }
    }

    #[staticmethod]
    pub fn new_boolean(b: bool) -> Self {
        Self {
            v: ValueVariant::Boolean(b),
        }
    }

    #[staticmethod]
    pub fn new_bbox(bbox: BBox) -> Self {
        Self {
            v: ValueVariant::BBox(bbox),
        }
    }

    #[staticmethod]
    pub fn new_point(point: Point) -> Self {
        Self {
            v: ValueVariant::Point(point),
        }
    }

    #[staticmethod]
    pub fn new_polygon(polygon: PolygonalArea) -> Self {
        Self {
            v: ValueVariant::Polygon(polygon),
        }
    }

    #[staticmethod]
    pub fn new_polyline(polyline: Vec<Point>) -> Self {
        Self {
            v: ValueVariant::PolyLine(polyline),
        }
    }

    #[staticmethod]
    pub fn new_keypoints(keypoints: Vec<(Point, HashMap<String, String>)>) -> Self {
        Self {
            v: ValueVariant::KeyPoints(keypoints),
        }
    }

    pub fn as_string(&self) -> Option<String> {
        match &self.v {
            ValueVariant::String(s) => Some(s.clone()),
            _ => None,
        }
    }

    pub fn as_integer(&self) -> Option<i64> {
        match &self.v {
            ValueVariant::Integer(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match &self.v {
            ValueVariant::Float(f) => Some(*f),
            _ => None,
        }
    }

    pub fn as_boolean(&self) -> Option<bool> {
        match &self.v {
            ValueVariant::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_bbox(&self) -> Option<BBox> {
        match &self.v {
            ValueVariant::BBox(bbox) => Some(bbox.clone()),
            _ => None,
        }
    }

    pub fn as_point(&self) -> Option<Point> {
        match &self.v {
            ValueVariant::Point(point) => Some(point.clone()),
            _ => None,
        }
    }

    pub fn as_polygon(&self) -> Option<PolygonalArea> {
        match &self.v {
            ValueVariant::Polygon(polygon) => Some(polygon.clone()),
            _ => None,
        }
    }

    pub fn as_polyline(&self) -> Option<Vec<Point>> {
        match &self.v {
            ValueVariant::PolyLine(polyline) => Some(polyline.clone()),
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
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone, derive_builder::Builder)]
#[archive(check_bytes)]
pub struct Attribute {
    #[pyo3(get)]
    pub element_name: String,
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub value: Value,
    #[pyo3(get)]
    pub confidence: Option<f64>,
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
    pub fn new(element_name: String, name: String, value: Value, confidence: Option<f64>) -> Self {
        Self {
            element_name,
            name,
            value,
            confidence,
        }
    }
}
