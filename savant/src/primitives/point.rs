use pyo3::{pyclass, pymethods, Py, PyAny};
use savant_core::primitives::rust;
use savant_core::to_json_value::ToSerdeJsonValue;
use serde_json::Value;

#[pyclass]
#[derive(Debug, PartialEq, Clone)]
pub struct Point(rust::Point);

#[pymethods]
impl Point {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[new]
    pub fn new(x: f32, y: f32) -> Self {
        Self(rust::Point::new(x, y))
    }

    #[getter]
    fn get_x(&self) -> f32 {
        self.0.x
    }

    #[setter]
    fn set_x(&mut self, x: f32) {
        self.0.x = x;
    }

    #[getter]
    fn get_y(&self) -> f32 {
        self.0.y
    }

    #[setter]
    fn set_y(&mut self, y: f32) {
        self.0.y = y;
    }
}

impl ToSerdeJsonValue for Point {
    fn to_serde_json_value(&self) -> Value {
        self.0.to_serde_json_value()
    }
}
