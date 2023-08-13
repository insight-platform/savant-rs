use crate::primitives::point::Point;
use pyo3::{pyclass, pymethods, Py, PyAny};
use savant_core::primitives::point::Point as PointRs;
use savant_core::primitives::segment::Intersection as IntersectionRs;
use savant_core::primitives::segment::IntersectionKind as IntersectionKindRs;
use savant_core::primitives::segment::Segment as SegmentRs;
use savant_core::to_json_value::ToSerdeJsonValue;
use serde_json::Value;
use std::mem;

#[pyclass]
#[derive(Debug, PartialEq, Clone)]
pub struct Segment(SegmentRs);

impl ToSerdeJsonValue for Segment {
    fn to_serde_json_value(&self) -> Value {
        self.0.to_serde_json_value()
    }
}

#[pymethods]
impl Segment {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[new]
    pub fn new(begin: Point, end: Point) -> Self {
        Self(SegmentRs::new(
            unsafe { mem::transmute::<Point, PointRs>(begin) },
            unsafe { mem::transmute::<Point, PointRs>(end) },
        ))
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub enum IntersectionKind {
    Enter,
    Inside,
    Leave,
    Cross,
    Outside,
}

impl From<IntersectionKind> for IntersectionKindRs {
    fn from(kind: IntersectionKind) -> Self {
        match kind {
            IntersectionKind::Enter => IntersectionKindRs::Enter,
            IntersectionKind::Inside => IntersectionKindRs::Inside,
            IntersectionKind::Leave => IntersectionKindRs::Leave,
            IntersectionKind::Cross => IntersectionKindRs::Cross,
            IntersectionKind::Outside => IntersectionKindRs::Outside,
        }
    }
}

impl From<IntersectionKindRs> for IntersectionKind {
    fn from(kind: IntersectionKindRs) -> Self {
        match kind {
            IntersectionKindRs::Enter => IntersectionKind::Enter,
            IntersectionKindRs::Inside => IntersectionKind::Inside,
            IntersectionKindRs::Leave => IntersectionKind::Leave,
            IntersectionKindRs::Cross => IntersectionKind::Cross,
            IntersectionKindRs::Outside => IntersectionKind::Outside,
        }
    }
}

impl ToSerdeJsonValue for IntersectionKind {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!(format!("{:?}", self))
    }
}

#[pymethods]
impl IntersectionKind {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Intersection(IntersectionRs);

impl ToSerdeJsonValue for Intersection {
    fn to_serde_json_value(&self) -> Value {
        self.0.to_serde_json_value()
    }
}

#[pymethods]
impl Intersection {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[new]
    pub fn new(kind: IntersectionKind, edges: Vec<(usize, Option<String>)>) -> Self {
        Self(IntersectionRs::new(kind.into(), edges))
    }

    #[getter]
    pub fn get_kind(&self) -> IntersectionKind {
        self.0.kind.clone().into()
    }

    #[getter]
    pub fn get_edges(&self) -> Vec<(usize, Option<String>)> {
        self.0.edges.clone()
    }
}
