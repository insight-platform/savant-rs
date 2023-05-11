use crate::primitives::point::Point;
use pyo3::{pyclass, pymethods, Py, PyAny};
use rkyv::{Archive, Deserialize, Serialize};

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct Segment {
    #[pyo3(get, set)]
    pub begin: Point,
    #[pyo3(get, set)]
    pub end: Point,
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
        Self { begin, end }
    }
}

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub enum IntersectionKind {
    Enter,
    Inside,
    Leave,
    Cross,
    Outside,
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
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct Intersection {
    pub kind: IntersectionKind,
    pub edges: Vec<(usize, Option<String>)>,
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
        Self { kind, edges }
    }

    #[getter]
    pub fn get_kind(&self) -> IntersectionKind {
        self.kind.clone()
    }

    #[getter]
    pub fn get_edges(&self) -> Vec<(usize, Option<String>)> {
        self.edges.clone()
    }
}
