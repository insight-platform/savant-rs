use crate::primitives::message::Message;
use pyo3::{pyclass, pymethods, Py, PyAny};
use savant_core::primitives::rust;

#[pyclass]
#[derive(Debug, Clone)]
pub struct EndOfStream(pub(crate) rust::EndOfStream);

#[pymethods]
impl EndOfStream {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", &self.0)
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
        serde_json::json!(&self.0).to_string()
    }

    pub fn to_message(&self) -> Message {
        Message::end_of_stream(self.clone())
    }
}
