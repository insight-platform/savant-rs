use crate::primitives::to_json_value::ToSerdeJsonValue;
use crate::primitives::Message;
use pyo3::{pyclass, pymethods, Py, PyAny};
use rkyv::{Archive, Deserialize, Serialize};
use serde_json::Value;

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct EndOfStream {
    #[pyo3(get, set)]
    pub source_id: String,
}

impl ToSerdeJsonValue for EndOfStream {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!(
        {
            "type": "EndOfStream",
            "source_id": self.source_id,
        })
    }
}

#[pymethods]
impl EndOfStream {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[new]
    pub fn new(source_id: String) -> Self {
        Self { source_id }
    }

    #[getter]
    pub fn get_json(&self) -> String {
        serde_json::to_string(&self.to_serde_json_value()).unwrap()
    }

    pub fn to_message(&self) -> Message {
        Message::end_of_stream(self.clone())
    }
}
