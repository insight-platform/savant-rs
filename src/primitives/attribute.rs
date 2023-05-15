use crate::primitives::to_json_value::ToSerdeJsonValue;
use crate::primitives::{Intersection, Point, PolygonalArea, RBBox};
use pyo3::{pyclass, pymethods, Py, PyAny};
use rkyv::{Archive, Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone, Default)]
#[archive(check_bytes)]
pub enum ValueVariant {
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

impl ToSerdeJsonValue for ValueVariant {
    fn to_serde_json_value(&self) -> serde_json::Value {
        match self {
            ValueVariant::Bytes(dims, blob) => serde_json::json!({
                "dims": dims,
                "blob": blob,
            }),
            ValueVariant::String(s) => serde_json::json!({
                "string": s,
            }),
            ValueVariant::StringVector(v) => serde_json::json!({
                "string_vector": v,
            }),
            ValueVariant::Integer(i) => serde_json::json!({
                "integer": i,
            }),
            ValueVariant::IntegerVector(v) => serde_json::json!({
                "integer_vector": v,
            }),
            ValueVariant::Float(f) => serde_json::json!({
                "float": f,
            }),
            ValueVariant::FloatVector(v) => serde_json::json!({
                "float_vector": v,
            }),
            ValueVariant::Boolean(b) => serde_json::json!({
                "boolean": b,
            }),
            ValueVariant::BooleanVector(v) => serde_json::json!({
                "boolean_vector": v,
            }),
            ValueVariant::BBox(b) => serde_json::json!({
                "bbox": b.to_serde_json_value(),
            }),
            ValueVariant::BBoxVector(v) => serde_json::json!({
                "bbox_vector": v.iter().map(|b| b.to_serde_json_value()).collect::<Vec<_>>(),
            }),
            ValueVariant::Point(p) => serde_json::json!({
                "point": p.to_serde_json_value(),
            }),
            ValueVariant::PointVector(v) => serde_json::json!({
                "point_vector": v.iter().map(|p| p.to_serde_json_value()).collect::<Vec<_>>(),
            }),
            ValueVariant::Polygon(p) => serde_json::json!({
                "polygon": p.to_serde_json_value(),
            }),
            ValueVariant::PolygonVector(v) => serde_json::json!({
                "polygon_vector": v.iter().map(|p| p.to_serde_json_value()).collect::<Vec<_>>(),
            }),
            ValueVariant::Intersection(i) => serde_json::json!({
                "intersection": i.to_serde_json_value(),
            }),
            ValueVariant::None => serde_json::json!({
                "none": null,
            }),
        }
    }
}

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone, Default)]
#[archive(check_bytes)]
pub struct Value {
    #[pyo3(get, set)]
    pub confidence: Option<f64>,
    v: ValueVariant,
}

impl ToSerdeJsonValue for Value {
    fn to_serde_json_value(&self) -> serde_json::Value {
        serde_json::json!({
            "confidence": self.confidence,
            "value": self.v.to_serde_json_value(),
        })
    }
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
    pub fn intersection(i: Intersection, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: ValueVariant::Intersection(i),
        }
    }

    #[staticmethod]
    pub fn none() -> Self {
        Self {
            confidence: None,
            v: ValueVariant::None,
        }
    }

    #[staticmethod]
    pub fn bytes(dims: Vec<i64>, blob: Vec<u8>, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: ValueVariant::Bytes(dims, blob),
        }
    }

    #[staticmethod]
    pub fn string(s: String, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: ValueVariant::String(s),
        }
    }

    #[staticmethod]
    pub fn strings(s: Vec<String>, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: ValueVariant::StringVector(s),
        }
    }

    #[staticmethod]
    pub fn integer(i: i64, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: ValueVariant::Integer(i),
        }
    }

    #[staticmethod]
    pub fn integers(i: Vec<i64>, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: ValueVariant::IntegerVector(i),
        }
    }

    #[staticmethod]
    pub fn float(f: f64, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: ValueVariant::Float(f),
        }
    }

    #[staticmethod]
    pub fn floats(f: Vec<f64>, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: ValueVariant::FloatVector(f),
        }
    }

    #[staticmethod]
    pub fn boolean(b: bool, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: ValueVariant::Boolean(b),
        }
    }

    #[staticmethod]
    pub fn booleans(b: Vec<bool>, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: ValueVariant::BooleanVector(b),
        }
    }

    #[staticmethod]
    pub fn bbox(bbox: RBBox, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: ValueVariant::BBox(bbox),
        }
    }

    #[staticmethod]
    pub fn bboxes(bboxes: Vec<RBBox>, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: ValueVariant::BBoxVector(bboxes),
        }
    }

    #[staticmethod]
    pub fn point(point: Point, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: ValueVariant::Point(point),
        }
    }

    #[staticmethod]
    pub fn points(points: Vec<Point>, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: ValueVariant::PointVector(points),
        }
    }

    #[staticmethod]
    pub fn polygon(polygon: PolygonalArea, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: ValueVariant::Polygon(polygon),
        }
    }

    #[staticmethod]
    pub fn polygons(polygons: Vec<PolygonalArea>, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: ValueVariant::PolygonVector(polygons),
        }
    }

    pub fn is_none(&self) -> bool {
        matches!(&self.v, ValueVariant::None)
    }

    pub fn as_bytes(&self) -> Option<(Vec<i64>, Vec<u8>)> {
        match &self.v {
            ValueVariant::Bytes(dims, bytes) => Some((dims.clone(), bytes.clone())),
            _ => None,
        }
    }

    pub fn as_intersection(&self) -> Option<Intersection> {
        match &self.v {
            ValueVariant::Intersection(i) => Some(i.clone()),
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

    pub fn as_bbox(&self) -> Option<RBBox> {
        match &self.v {
            ValueVariant::BBox(bbox) => Some(bbox.clone()),
            _ => None,
        }
    }

    pub fn as_bboxes(&self) -> Option<Vec<RBBox>> {
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
    pub values: Vec<Value>,
    #[pyo3(get)]
    pub hint: Option<String>,
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
    pub fn new(creator: String, name: String, values: Vec<Value>, hint: Option<String>) -> Self {
        Self {
            creator,
            name,
            values,
            hint,
        }
    }
}

pub trait InnerAttributes {
    fn get_attributes_ref(&self) -> &HashMap<(String, String), Attribute>;
    fn get_attributes_ref_mut(&mut self) -> &mut HashMap<(String, String), Attribute>;
}

pub trait Attributive<T: InnerAttributes + Send> {
    fn get_inner(&self) -> Arc<Mutex<T>>;

    fn get_attributes(&self) -> Vec<(String, String)> {
        self.get_inner()
            .lock()
            .unwrap()
            .get_attributes_ref()
            .iter()
            .map(|((creator, name), _)| (creator.clone(), name.clone()))
            .collect()
    }

    fn get_attribute(&self, creator: String, name: String) -> Option<Attribute> {
        self.get_inner()
            .lock()
            .unwrap()
            .get_attributes_ref()
            .get(&(creator, name))
            .cloned()
    }

    fn delete_attribute(&mut self, creator: String, name: String) -> Option<Attribute> {
        self.get_inner()
            .lock()
            .unwrap()
            .get_attributes_ref_mut()
            .remove(&(creator, name))
    }

    fn set_attribute(&mut self, attribute: Attribute) -> Option<Attribute> {
        self.get_inner()
            .lock()
            .unwrap()
            .get_attributes_ref_mut()
            .insert(
                (attribute.creator.clone(), attribute.name.clone()),
                attribute,
            )
    }

    #[allow(clippy::let_unit_value)]
    fn clear_attributes(&mut self) {
        self.get_inner()
            .lock()
            .unwrap()
            .get_attributes_ref_mut()
            .clear()
    }

    fn delete_attributes(&mut self, negated: bool, creator: Option<String>, names: Vec<String>) {
        // let inner = self.get_inner();
        self.get_inner()
            .lock()
            .unwrap()
            .get_attributes_ref_mut()
            .retain(|(c, label), _| match creator {
                Some(ref creator) => {
                    ((names.is_empty() || names.contains(label)) && creator == c) ^ !negated
                }
                None => names.contains(label) ^ !negated,
            });
    }

    fn find_attributes(
        &self,
        creator: Option<String>,
        name: Option<String>,
        hint: Option<String>,
    ) -> Vec<(String, String)> {
        self.get_inner()
            .lock()
            .unwrap()
            .get_attributes_ref()
            .iter()
            .filter(|((_, _), a)| {
                if let Some(creator) = &creator {
                    if a.creator != *creator {
                        return false;
                    }
                }

                if let Some(name) = &name {
                    if a.name != *name {
                        return false;
                    }
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
