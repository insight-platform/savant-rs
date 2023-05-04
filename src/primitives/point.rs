use pyo3::{pyclass, pymethods, Py, PyAny};
use rkyv::{Archive, Deserialize, Serialize};

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[pymethods]
impl Point {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[new]
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}
