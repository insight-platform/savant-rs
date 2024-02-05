use crate::primitives::message::Message;
use pyo3::{pyclass, pymethods, Py, PyAny};
use savant_core::json_api::ToSerdeJsonValue;
use savant_core::primitives::rust;

#[pyclass]
#[derive(Debug, Clone)]
pub struct Shutdown(pub(crate) rust::Shutdown);

#[pymethods]
impl Shutdown {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", &self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[new]
    pub fn new(auth: &str) -> Self {
        Self(rust::Shutdown::new(auth))
    }

    #[getter]
    pub fn get_auth(&self) -> String {
        self.0.get_auth().to_string()
    }

    #[getter]
    pub fn get_json(&self) -> String {
        serde_json::to_string(&self.0.to_serde_json_value()).unwrap()
    }

    pub fn to_message(&self) -> Message {
        Message::shutdown(self.clone())
    }
}
