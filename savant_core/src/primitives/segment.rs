use crate::primitives::point::Point;
use rkyv::{Archive, Deserialize, Serialize};

#[derive(
    Archive, Deserialize, Serialize, Debug, PartialEq, Clone, serde::Serialize, serde::Deserialize,
)]
#[archive(check_bytes)]
pub struct Segment {
    pub begin: Point,
    pub end: Point,
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

#[derive(
    Archive, Deserialize, Serialize, Debug, PartialEq, Clone, serde::Serialize, serde::Deserialize,
)]
#[archive(check_bytes)]
pub struct Intersection {
    pub kind: IntersectionKind,
    pub edges: Vec<(usize, Option<String>)>,
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
