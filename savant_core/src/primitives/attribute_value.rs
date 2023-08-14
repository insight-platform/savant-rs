use crate::primitives::{Intersection, OwnedRBBoxData, Point, PolygonalArea};
use crate::to_json_value::ToSerdeJsonValue;
use rkyv::{Archive, Deserialize, Serialize};
use serde_json::Value;

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
    BBox(OwnedRBBoxData),
    BBoxVector(Vec<OwnedRBBoxData>),
    Point(Point),
    PointVector(Vec<Point>),
    Polygon(PolygonalArea),
    PolygonVector(Vec<PolygonalArea>),
    Intersection(Intersection),
    #[default]
    None,
}

impl ToSerdeJsonValue for AttributeValueVariant {
    fn to_serde_json_value(&self) -> Value {
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

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone, Default)]
#[archive(check_bytes)]
pub struct AttributeValue {
    pub confidence: Option<f32>,
    pub v: AttributeValueVariant,
}

impl ToSerdeJsonValue for AttributeValue {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!({
            "confidence": self.confidence,
            "value": self.v.to_serde_json_value(),
        })
    }
}

impl AttributeValue {
    pub fn new(v: AttributeValueVariant, confidence: Option<f32>) -> Self {
        Self { confidence, v }
    }
}
