use crate::primitives::point::Point;
use pyo3::{pyclass, pymethods, Py, PyAny};
use savant_core::primitives::rust;
use savant_core::to_json_value::ToSerdeJsonValue;
use serde_json::Value;

#[pyclass]
#[derive(Debug, PartialEq, Clone)]
pub struct Segment(pub(crate) rust::Segment);

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
        Self(rust::Segment::new(begin.0, end.0))
    }

    #[getter]
    pub fn begin(&self) -> Point {
        Point(self.0.begin.clone())
    }

    #[getter]
    pub fn end(&self) -> Point {
        Point(self.0.end.clone())
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

impl From<IntersectionKind> for rust::IntersectionKind {
    fn from(kind: IntersectionKind) -> Self {
        match kind {
            IntersectionKind::Enter => rust::IntersectionKind::Enter,
            IntersectionKind::Inside => rust::IntersectionKind::Inside,
            IntersectionKind::Leave => rust::IntersectionKind::Leave,
            IntersectionKind::Cross => rust::IntersectionKind::Cross,
            IntersectionKind::Outside => rust::IntersectionKind::Outside,
        }
    }
}

impl From<rust::IntersectionKind> for IntersectionKind {
    fn from(kind: rust::IntersectionKind) -> Self {
        match kind {
            rust::IntersectionKind::Enter => IntersectionKind::Enter,
            rust::IntersectionKind::Inside => IntersectionKind::Inside,
            rust::IntersectionKind::Leave => IntersectionKind::Leave,
            rust::IntersectionKind::Cross => IntersectionKind::Cross,
            rust::IntersectionKind::Outside => IntersectionKind::Outside,
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
pub struct Intersection(pub(crate) rust::Intersection);

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
        Self(rust::Intersection::new(kind.into(), edges))
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
