use crate::primitives::any_object::AnyObject;
use crate::primitives::{Intersection, Point, PolygonalArea, RBBoxData};
use rkyv::{Archive, Deserialize, Serialize};

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
    BBox(RBBoxData),
    BBoxVector(Vec<RBBoxData>),
    Point(Point),
    PointVector(Vec<Point>),
    Polygon(PolygonalArea),
    PolygonVector(Vec<PolygonalArea>),
    Intersection(Intersection),
    TemporaryValue(AnyObject),
    #[default]
    None,
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

impl AttributeValue {
    pub fn new(value: AttributeValueVariant, confidence: Option<f32>) -> Self {
        Self { confidence, value }
    }

    pub fn float(value: f64, confidence: Option<f32>) -> Self {
        Self::new(AttributeValueVariant::Float(value), confidence)
    }

    pub fn float_vector(value: Vec<f64>, confidence: Option<f32>) -> Self {
        Self::new(AttributeValueVariant::FloatVector(value), confidence)
    }

    pub fn integer(value: i64, confidence: Option<f32>) -> Self {
        Self::new(AttributeValueVariant::Integer(value), confidence)
    }

    pub fn integer_vector(value: Vec<i64>, confidence: Option<f32>) -> Self {
        Self::new(AttributeValueVariant::IntegerVector(value), confidence)
    }

    pub fn string(value: String, confidence: Option<f32>) -> Self {
        Self::new(AttributeValueVariant::String(value), confidence)
    }

    pub fn string_vector(value: Vec<String>, confidence: Option<f32>) -> Self {
        Self::new(AttributeValueVariant::StringVector(value), confidence)
    }

    pub fn boolean(value: bool, confidence: Option<f32>) -> Self {
        Self::new(AttributeValueVariant::Boolean(value), confidence)
    }

    pub fn boolean_vector(value: Vec<bool>, confidence: Option<f32>) -> Self {
        Self::new(AttributeValueVariant::BooleanVector(value), confidence)
    }

    pub fn bbox(value: RBBoxData, confidence: Option<f32>) -> Self {
        Self::new(AttributeValueVariant::BBox(value), confidence)
    }

    pub fn bbox_vector(value: Vec<RBBoxData>, confidence: Option<f32>) -> Self {
        Self::new(AttributeValueVariant::BBoxVector(value), confidence)
    }

    pub fn point(value: Point, confidence: Option<f32>) -> Self {
        Self::new(AttributeValueVariant::Point(value), confidence)
    }

    pub fn point_vector(value: Vec<Point>, confidence: Option<f32>) -> Self {
        Self::new(AttributeValueVariant::PointVector(value), confidence)
    }

    pub fn polygon(value: PolygonalArea, confidence: Option<f32>) -> Self {
        Self::new(AttributeValueVariant::Polygon(value), confidence)
    }

    pub fn polygon_vector(value: Vec<PolygonalArea>, confidence: Option<f32>) -> Self {
        Self::new(AttributeValueVariant::PolygonVector(value), confidence)
    }

    pub fn intersection(value: Intersection, confidence: Option<f32>) -> Self {
        Self::new(AttributeValueVariant::Intersection(value), confidence)
    }

    pub fn temporary_value(value: AnyObject, confidence: Option<f32>) -> Self {
        Self::new(AttributeValueVariant::TemporaryValue(value), confidence)
    }

    pub fn none() -> Self {
        Self::new(AttributeValueVariant::None, None)
    }

    pub fn bytes(value: Vec<i64>, confidence: Option<f32>) -> Self {
        Self::new(AttributeValueVariant::Bytes(value, Vec::new()), confidence)
    }

    pub fn get(&self) -> &AttributeValueVariant {
        &self.value
    }

    pub fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
}
