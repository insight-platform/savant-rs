use crate::json_api::ToSerdeJsonValue;
use crate::primitives::point::Point;
use rkyv::{Archive, Deserialize, Serialize};
use serde_json::Value;

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct Segment {
    pub begin: Point,
    pub end: Point,
}

impl ToSerdeJsonValue for Segment {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!({
            "begin": self.begin.to_serde_json_value(),
            "end": self.end.to_serde_json_value(),
        })
    }
}

impl Segment {
    pub fn new(begin: Point, end: Point) -> Self {
        Self { begin, end }
    }
}

#[derive(
    Archive, Deserialize, Serialize, Debug, PartialEq, Clone, serde::Serialize, serde::Deserialize,
)]
#[archive(check_bytes)]
pub enum IntersectionKind {
    Enter,
    Inside,
    Leave,
    Cross,
    Outside,
}

impl ToSerdeJsonValue for IntersectionKind {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!(format!("{:?}", self))
    }
}

#[derive(
    Archive, Deserialize, Serialize, Debug, PartialEq, Clone, serde::Serialize, serde::Deserialize,
)]
#[archive(check_bytes)]
pub struct Intersection {
    pub kind: IntersectionKind,
    pub edges: Vec<(usize, Option<String>)>,
}

impl ToSerdeJsonValue for Intersection {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!({
            "kind": self.kind.to_serde_json_value(),
            "edges": self.edges,
        })
    }
}

impl Intersection {
    pub fn new(kind: IntersectionKind, edges: Vec<(usize, Option<String>)>) -> Self {
        Self { kind, edges }
    }

    pub fn get_kind(&self) -> IntersectionKind {
        self.kind.clone()
    }

    pub fn get_edges(&self) -> Vec<(usize, Option<String>)> {
        self.edges.clone()
    }
}
