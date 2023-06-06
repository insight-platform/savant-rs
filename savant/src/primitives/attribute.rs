use crate::primitives::to_json_value::ToSerdeJsonValue;
use crate::primitives::{Intersection, Point, PolygonalArea, RBBox};
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

    #[staticmethod]
    pub fn intersection(i: Intersection, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::Intersection(i),
        }
    }

    #[staticmethod]
    pub fn none() -> Self {
        Self {
            confidence: None,
            v: AttributeValueVariant::None,
        }
    }

    #[staticmethod]
    pub fn bytes(dims: Vec<i64>, blob: Vec<u8>, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::Bytes(dims, blob),
        }
    }

    #[staticmethod]
    pub fn string(s: String, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::String(s),
        }
    }

    #[staticmethod]
    pub fn strings(s: Vec<String>, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::StringVector(s),
        }
    }

    #[staticmethod]
    pub fn integer(i: i64, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::Integer(i),
        }
    }

    #[staticmethod]
    pub fn integers(i: Vec<i64>, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::IntegerVector(i),
        }
    }

    #[staticmethod]
    pub fn float(f: f64, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::Float(f),
        }
    }

    #[staticmethod]
    pub fn floats(f: Vec<f64>, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::FloatVector(f),
        }
    }

    #[staticmethod]
    pub fn boolean(b: bool, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::Boolean(b),
        }
    }

    #[staticmethod]
    pub fn booleans(b: Vec<bool>, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::BooleanVector(b),
        }
    }

    #[staticmethod]
    pub fn bbox(bbox: RBBox, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::BBox(bbox),
        }
    }

    #[staticmethod]
    pub fn bboxes(bboxes: Vec<RBBox>, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::BBoxVector(bboxes),
        }
    }

    #[staticmethod]
    pub fn point(point: Point, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::Point(point),
        }
    }

    #[staticmethod]
    pub fn points(points: Vec<Point>, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::PointVector(points),
        }
    }

    #[staticmethod]
    pub fn polygon(polygon: PolygonalArea, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::Polygon(polygon),
        }
    }

    #[staticmethod]
    pub fn polygons(polygons: Vec<PolygonalArea>, confidence: Option<f64>) -> Self {
        Self {
            confidence,
            v: AttributeValueVariant::PolygonVector(polygons),
        }
    }

    pub fn is_none(&self) -> bool {
        matches!(&self.v, AttributeValueVariant::None)
    }

    pub fn as_bytes(&self) -> Option<(Vec<i64>, Vec<u8>)> {
        match &self.v {
            AttributeValueVariant::Bytes(dims, bytes) => Some((dims.clone(), bytes.clone())),
            _ => None,
        }
    }

    pub fn as_intersection(&self) -> Option<Intersection> {
        match &self.v {
            AttributeValueVariant::Intersection(i) => Some(i.clone()),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<String> {
        match &self.v {
            AttributeValueVariant::String(s) => Some(s.clone()),
            _ => None,
        }
    }

    pub fn as_strings(&self) -> Option<Vec<String>> {
        match &self.v {
            AttributeValueVariant::StringVector(s) => Some(s.clone()),
            _ => None,
        }
    }

    pub fn as_integer(&self) -> Option<i64> {
        match &self.v {
            AttributeValueVariant::Integer(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_integers(&self) -> Option<Vec<i64>> {
        match &self.v {
            AttributeValueVariant::IntegerVector(i) => Some(i.clone()),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match &self.v {
            AttributeValueVariant::Float(f) => Some(*f),
            _ => None,
        }
    }

    pub fn as_floats(&self) -> Option<Vec<f64>> {
        match &self.v {
            AttributeValueVariant::FloatVector(f) => Some(f.clone()),
            _ => None,
        }
    }

    pub fn as_boolean(&self) -> Option<bool> {
        match &self.v {
            AttributeValueVariant::Boolean(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_booleans(&self) -> Option<Vec<bool>> {
        match &self.v {
            AttributeValueVariant::BooleanVector(b) => Some(b.clone()),
            _ => None,
        }
    }

    pub fn as_bbox(&self) -> Option<RBBox> {
        match &self.v {
            AttributeValueVariant::BBox(bbox) => Some(bbox.clone()),
            _ => None,
        }
    }

    pub fn as_bboxes(&self) -> Option<Vec<RBBox>> {
        match &self.v {
            AttributeValueVariant::BBoxVector(bbox) => Some(bbox.clone()),
            _ => None,
        }
    }

    pub fn as_point(&self) -> Option<Point> {
        match &self.v {
            AttributeValueVariant::Point(point) => Some(point.clone()),
            _ => None,
        }
    }

    pub fn as_points(&self) -> Option<Vec<Point>> {
        match &self.v {
            AttributeValueVariant::PointVector(point) => Some(point.clone()),
            _ => None,
        }
    }

    pub fn as_polygon(&self) -> Option<PolygonalArea> {
        match &self.v {
            AttributeValueVariant::Polygon(polygon) => Some(polygon.clone()),
            _ => None,
        }
    }

    pub fn as_polygons(&self) -> Option<Vec<PolygonalArea>> {
        match &self.v {
            AttributeValueVariant::PolygonVector(polygon) => Some(polygon.clone()),
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
    pub fn new(
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

    pub fn is_temporary(&self) -> bool {
        !self.is_persistent
    }

    pub fn make_persistent(&mut self) {
        self.is_persistent = true;
    }

    pub fn make_temporary(&mut self) {
        self.is_persistent = false;
    }

    #[getter]
    pub fn get_creator(&self) -> String {
        self.creator.clone()
    }

    #[getter]
    pub fn get_name(&self) -> String {
        self.name.clone()
    }

    #[getter]
    pub fn get_values(&self) -> Vec<AttributeValue> {
        self.values.clone()
    }

    #[getter]
    pub fn get_hint(&self) -> Option<String> {
        self.hint.clone()
    }

    #[setter]
    pub fn set_hint(&mut self, hint: Option<String>) {
        self.hint = hint;
    }

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
