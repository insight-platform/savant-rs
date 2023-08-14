use crate::primitives::{Attribute, Message};
use crate::release_gil;
use pyo3::{pyclass, pymethods, Py, PyAny};
use rkyv::{Archive, Deserialize, Serialize};
use savant_core::primitives::{rust, Attributive};
use savant_core::to_json_value::ToSerdeJsonValue;
use serde_json::Value;
use std::collections::HashMap;
use std::mem;

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct Telemetry {
    #[pyo3(get, set)]
    pub source_id: String,
    pub attributes: HashMap<(String, String), rust::Attribute>,
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
    #[pyo3(signature = (no_gil = true))]
    pub fn attributes_gil(&self, no_gil: bool) -> Vec<(String, String)> {
        release_gil!(no_gil, || self.get_attributes())
    }

    #[pyo3(name = "find_attributes")]
    #[pyo3(signature = (namespace=None, names=vec![], hint=None, no_gil=true))]
    pub fn find_attributes_gil(
        &self,
        namespace: Option<String>,
        names: Vec<String>,
        hint: Option<String>,
        no_gil: bool,
    ) -> Vec<(String, String)> {
        release_gil!(no_gil, || self.find_attributes(namespace, names, hint))
    }

    #[pyo3(name = "get_attribute")]
    #[pyo3(signature = (namespace, name, no_gil=true))]
    pub fn get_attribute_gil(
        &self,
        namespace: String,
        name: String,
        no_gil: bool,
    ) -> Option<Attribute> {
        let res = release_gil!(no_gil, || self.get_attribute(namespace, name));
        unsafe { mem::transmute::<Option<rust::Attribute>, Option<Attribute>>(res) }
    }

    #[pyo3(name = "delete_attributes")]
    #[pyo3(signature = (namespace=None, names=vec![], no_gil=true))]
    pub fn delete_attributes_gil(
        &mut self,
        namespace: Option<String>,
        names: Vec<String>,
        no_gil: bool,
    ) {
        release_gil!(no_gil, || self.delete_attributes(namespace, names))
    }

    #[pyo3(name = "delete_attribute")]
    #[pyo3(signature = (namespace, name, no_gil=true))]
    pub fn delete_attribute_gil(
        &mut self,
        namespace: String,
        name: String,
        no_gil: bool,
    ) -> Option<Attribute> {
        let res = release_gil!(no_gil, || self.delete_attribute(namespace, name));
        unsafe { mem::transmute::<Option<rust::Attribute>, Option<Attribute>>(res) }
    }

    #[pyo3(name = "set_attribute")]
    pub fn set_attribute_py(&mut self, attribute: Attribute) -> Option<Attribute> {
        let attribute = unsafe { mem::transmute::<Attribute, rust::Attribute>(attribute) };
        let res = self.set_attribute(attribute);
        unsafe { mem::transmute::<Option<rust::Attribute>, Option<Attribute>>(res) }
    }

    #[pyo3(name = "clear_attributes")]
    #[pyo3(signature = (no_gil=true))]
    pub fn clear_attributes_gil(&mut self, no_gil: bool) {
        release_gil!(no_gil, || self.clear_attributes())
    }

    pub fn json(&self) -> String {
        serde_json::to_string(&self.to_serde_json_value()).unwrap()
    }

    pub fn json_pretty(&self) -> String {
        serde_json::to_string_pretty(&self.to_serde_json_value()).unwrap()
    }
}

impl Attributive for Telemetry {
    fn get_attributes_ref(&self) -> &HashMap<(String, String), rust::Attribute> {
        &self.attributes
    }

    fn get_attributes_ref_mut(&mut self) -> &mut HashMap<(String, String), rust::Attribute> {
        &mut self.attributes
    }

    fn take_attributes(&mut self) -> HashMap<(String, String), rust::Attribute> {
        mem::take(&mut self.attributes)
    }

    fn place_attributes(&mut self, attributes: HashMap<(String, String), rust::Attribute>) {
        self.attributes = attributes;
    }
}
