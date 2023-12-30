use crate::primitives::attribute::Attribute;
use crate::primitives::message::Message;
use crate::{release_gil, with_gil};
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyBytes;
use pyo3::{pyclass, pymethods, Py, PyAny, PyObject, PyResult};
use savant_core::json_api::ToSerdeJsonValue;
use savant_core::primitives::rust as rust_primitives;
use savant_core::primitives::{rust, Attributive};
use savant_core::protobuf::{from_pb, ToProtobuf};
use serde_json::Value;
use std::mem;

#[pyclass]
#[derive(Debug, Clone)]
pub struct UserData(pub(crate) rust_primitives::UserData);

impl ToSerdeJsonValue for UserData {
    fn to_serde_json_value(&self) -> Value {
        self.0.to_serde_json_value()
    }
}

#[pymethods]
impl UserData {
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
        Self(rust_primitives::UserData::new(source_id))
    }

    #[getter]
    pub fn get_source_id(&self) -> String {
        self.0.get_source_id().to_string()
    }

    #[getter]
    pub fn get_json(&self) -> String {
        self.0.json()
    }

    pub fn to_message(&self) -> Message {
        Message::user_data(self.clone())
    }

    #[pyo3(name = "attributes")]
    #[pyo3(signature = (no_gil = true))]
    pub fn attributes_gil(&self, no_gil: bool) -> Vec<(String, String)> {
        release_gil!(no_gil, || self.0.get_attributes())
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
        release_gil!(no_gil, || self.0.find_attributes(namespace, names, hint))
    }

    #[pyo3(name = "get_attribute")]
    #[pyo3(signature = (namespace, name, no_gil=true))]
    pub fn get_attribute_gil(
        &self,
        namespace: String,
        name: String,
        no_gil: bool,
    ) -> Option<Attribute> {
        let res = release_gil!(no_gil, || self.0.get_attribute(namespace, name));
        res.map(Attribute)
    }

    #[pyo3(name = "delete_attributes")]
    #[pyo3(signature = (namespace=None, names=vec![], no_gil=true))]
    pub fn delete_attributes_gil(
        &mut self,
        namespace: Option<String>,
        names: Vec<String>,
        no_gil: bool,
    ) {
        release_gil!(no_gil, || self.0.delete_attributes(namespace, names))
    }

    #[pyo3(name = "delete_attribute")]
    #[pyo3(signature = (namespace, name, no_gil=true))]
    pub fn delete_attribute_gil(
        &mut self,
        namespace: String,
        name: String,
        no_gil: bool,
    ) -> Option<Attribute> {
        let res = release_gil!(no_gil, || self.0.delete_attribute(namespace, name));
        res.map(Attribute)
    }

    pub fn set_attribute(&mut self, attribute: Attribute) -> Option<Attribute> {
        let res = self.0.set_attribute(attribute.0);
        res.map(Attribute)
    }

    pub fn clear_attributes(&mut self, no_gil: bool) {
        release_gil!(no_gil, || self.0.clear_attributes())
    }

    pub fn exclude_temporary_attributes(&mut self) -> Vec<Attribute> {
        unsafe {
            mem::transmute::<Vec<rust::Attribute>, Vec<Attribute>>(
                self.0.exclude_temporary_attributes(),
            )
        }
    }

    #[getter]
    pub fn json(&self) -> String {
        self.0.json()
    }

    #[getter]
    pub fn json_pretty(&self) -> String {
        self.0.json_pretty()
    }

    #[pyo3(name = "to_protobuf")]
    #[pyo3(signature = (no_gil = true))]
    fn to_protobuf_gil(&self, no_gil: bool) -> PyResult<PyObject> {
        let bytes = release_gil!(no_gil, || {
            self.0.to_pb().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to serialize user data to protobuf: {}", e))
            })
        })?;
        with_gil!(|py| {
            let bytes = PyBytes::new(py, &bytes);
            Ok(PyObject::from(bytes))
        })
    }

    #[staticmethod]
    #[pyo3(name = "from_protobuf")]
    #[pyo3(signature = (bytes, no_gil = true))]
    fn from_protobuf_gil(bytes: &PyBytes, no_gil: bool) -> PyResult<Self> {
        let bytes = bytes.as_bytes();
        release_gil!(no_gil, || {
            let obj =
                from_pb::<savant_core::protobuf::UserData, rust::UserData>(bytes).map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "Failed to deserialize user data from protobuf: {}",
                        e
                    ))
                })?;
            Ok(Self(obj))
        })
    }
}
