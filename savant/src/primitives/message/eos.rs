use crate::primitives::Message;
use pyo3::{pyclass, pymethods, Py, PyAny};
use savant_core::primitives::rust;
use savant_core::to_json_value::ToSerdeJsonValue;

#[pyclass]
#[derive(Debug, Clone)]
pub struct EndOfStream(rust::EndOfStream);

#[pymethods]
impl EndOfStream {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[new]
    pub fn new(source_id: String) -> Self {
        Self(rust::EndOfStream { source_id })
    }

    #[getter]
    pub fn get_source_id(&self) -> String {
        self.0.source_id.clone()
    }

    #[getter]
    pub fn get_json(&self) -> String {
        serde_json::to_string(&self.0.to_serde_json_value()).unwrap()
    }

    pub fn to_message(&self) -> Message {
        Message::end_of_stream(self.clone())
    }
}
