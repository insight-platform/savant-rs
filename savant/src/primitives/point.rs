use crate::primitives::to_json_value::ToSerdeJsonValue;
use pyo3::{pyclass, pymethods, Py, PyAny};
use rkyv::{Archive, Deserialize, Serialize};
use serde_json::Value;

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct Point {
    #[pyo3(get, set)]
    pub x: f64,
    #[pyo3(get, set)]
    pub y: f64,
}

impl ToSerdeJsonValue for Point {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!({
            "x": self.x,
            "y": self.y,
        })
    }
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
