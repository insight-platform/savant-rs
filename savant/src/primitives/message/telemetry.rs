use crate::primitives::attribute::Attributive;
use crate::primitives::to_json_value::ToSerdeJsonValue;
use crate::primitives::{Attribute, Message};
use crate::utils::python::release_gil;
use pyo3::{pyclass, pymethods, Py, PyAny};
use rkyv::{Archive, Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::mem;

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct Telemetry {
    #[pyo3(get, set)]
    pub source_id: String,
    pub attributes: HashMap<(String, String), Attribute>,
}

impl ToSerdeJsonValue for Telemetry {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!(
        {
            "type": "Telemetry",
            "source_id": self.source_id,
            "attributes": self.attributes.values().map(|v| v.to_serde_json_value()).collect::<Vec<_>>(),
        })
    }
}

#[pymethods]
impl Telemetry {
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
        Self {
            source_id,
            attributes: HashMap::new(),
        }
    }

    #[getter]
    pub fn get_json(&self) -> String {
        serde_json::to_string(&self.to_serde_json_value()).unwrap()
    }

    pub fn to_message(&self) -> Message {
        Message::telemetry(self.clone())
    }

    #[pyo3(name = "attributes")]
    pub fn attributes_gil(&self) -> Vec<(String, String)> {
        release_gil(|| self.get_attributes())
    }

    #[pyo3(name = "find_attributes")]
    #[pyo3(signature = (creator=None, names=vec![], hint=None))]
    pub fn find_attributes_gil(
        &self,
        creator: Option<String>,
        names: Vec<String>,
        hint: Option<String>,
    ) -> Vec<(String, String)> {
        release_gil(|| self.find_attributes(creator, names, hint))
    }

    #[pyo3(name = "get_attribute")]
    pub fn get_attribute_gil(&self, creator: String, name: String) -> Option<Attribute> {
        release_gil(|| self.get_attribute(creator, name))
    }

    #[pyo3(signature = (creator=None, names=vec![]))]
    #[pyo3(name = "delete_attributes")]
    pub fn delete_attributes_gil(&mut self, creator: Option<String>, names: Vec<String>) {
        release_gil(|| self.delete_attributes(creator, names))
    }

    #[pyo3(name = "delete_attribute")]
    pub fn delete_attribute_gil(&mut self, creator: String, name: String) -> Option<Attribute> {
        release_gil(|| self.delete_attribute(creator, name))
    }

    #[pyo3(name = "set_attribute")]
    pub fn set_attribute_gil(&mut self, attribute: Attribute) -> Option<Attribute> {
        self.set_attribute(attribute)
    }

    #[pyo3(name = "clear_attributes")]
    pub fn clear_attributes_gil(&mut self) {
        release_gil(|| self.clear_attributes())
    }

    pub fn json(&self) -> String {
        serde_json::to_string(&self.to_serde_json_value()).unwrap()
    }

    pub fn json_pretty(&self) -> String {
        serde_json::to_string_pretty(&self.to_serde_json_value()).unwrap()
    }
}

impl Attributive for Telemetry {
    fn get_attributes_ref(&self) -> &HashMap<(String, String), Attribute> {
        &self.attributes
    }

    fn get_attributes_ref_mut(&mut self) -> &mut HashMap<(String, String), Attribute> {
        &mut self.attributes
    }

    fn take_attributes(&mut self) -> HashMap<(String, String), Attribute> {
        mem::take(&mut self.attributes)
    }

    fn place_attributes(&mut self, attributes: HashMap<(String, String), Attribute>) {
        self.attributes = attributes;
    }
}
