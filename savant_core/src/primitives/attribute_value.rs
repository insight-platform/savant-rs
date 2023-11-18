use crate::json_api::ToSerdeJsonValue;
use crate::primitives::any_object::AnyObject;
use crate::primitives::{Intersection, OwnedRBBoxData, Point, PolygonalArea};
use rkyv::{Archive, Deserialize, Serialize};
use serde_json::Value;

#[derive(
    Archive,
    Deserialize,
    Serialize,
    Debug,
    PartialEq,
    Clone,
    Default,
    serde::Serialize,
    serde::Deserialize,
)]
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
    TemporaryValue(AnyObject),
    #[default]
    None,
}

impl ToSerdeJsonValue for AttributeValueVariant {
    fn to_serde_json_value(&self) -> Value {
        serde_json::to_value(self).unwrap()
    }
}

#[derive(
    Archive,
    Deserialize,
    Serialize,
    Debug,
    PartialEq,
    Clone,
    Default,
    serde::Serialize,
    serde::Deserialize,
)]
#[archive(check_bytes)]
pub struct AttributeValue {
    pub confidence: Option<f32>,
    pub value: AttributeValueVariant,
}

impl ToSerdeJsonValue for AttributeValue {
    fn to_serde_json_value(&self) -> Value {
        serde_json::to_value(self).unwrap()
    }
}

impl AttributeValue {
    pub fn new(value: AttributeValueVariant, confidence: Option<f32>) -> Self {
        Self { confidence, value }
    }

    pub fn get_value(&self) -> &AttributeValueVariant {
        &self.value
    }

    pub fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
}
