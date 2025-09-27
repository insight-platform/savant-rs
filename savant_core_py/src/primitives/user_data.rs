use crate::primitives::attribute::Attribute;
use crate::primitives::attribute_value::AttributeValue;
use crate::primitives::message::Message;
use crate::{attach, detach};
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::{PyBytes, PyBytesMethods};
use pyo3::{pyclass, pymethods, Bound, Py, PyAny, PyResult};
use savant_core::json_api::ToSerdeJsonValue;
use savant_core::primitives::rust as rust_primitives;
use savant_core::primitives::{rust, WithAttributes};
use savant_core::protobuf::{from_pb, ToProtobuf};
use serde_json::Value;

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
    pub fn new(source_id: &str) -> Self {
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

    #[getter]
    pub fn attributes(&self) -> Vec<(String, String)> {
        self.0.get_attributes()
    }

    pub fn get_attribute(&self, namespace: &str, name: &str) -> Option<Attribute> {
        let res = self.0.get_attribute(namespace, name);
        res.map(Attribute)
    }

    pub fn find_attributes_with_ns(&mut self, namespace: &str) -> Vec<(String, String)> {
        self.0.find_attributes_with_ns(namespace)
    }

    pub fn find_attributes_with_names(&mut self, names: Vec<String>) -> Vec<(String, String)> {
        let label_refs = names.iter().map(|v| v.as_ref()).collect::<Vec<&str>>();
        self.0.find_attributes_with_names(&label_refs)
    }

    pub fn find_attributes_with_hints(
        &mut self,
        hints: Vec<Option<String>>,
    ) -> Vec<(String, String)> {
        let hint_opts_refs = hints
            .iter()
            .map(|v| v.as_deref())
            .collect::<Vec<Option<&str>>>();
        let hint_refs = hint_opts_refs.iter().collect::<Vec<_>>();

        self.0.find_attributes_with_hints(&hint_refs)
    }

    pub fn delete_attributes_with_ns(&mut self, namespace: &str) {
        self.0.delete_attributes_with_ns(namespace)
    }

    pub fn delete_attributes_with_names(&mut self, names: Vec<String>) {
        let label_refs = names.iter().map(|v| v.as_ref()).collect::<Vec<&str>>();
        self.0.delete_attributes_with_names(&label_refs)
    }

    pub fn delete_attributes_with_hints(&mut self, hints: Vec<Option<String>>) {
        let hint_opts_refs = hints
            .iter()
            .map(|v| v.as_deref())
            .collect::<Vec<Option<&str>>>();
        let hint_refs = hint_opts_refs.iter().collect::<Vec<_>>();

        self.0.delete_attributes_with_hints(&hint_refs)
    }

    pub fn delete_attribute(&mut self, namespace: &str, name: &str) -> Option<Attribute> {
        let res = self.0.delete_attribute(namespace, name);
        res.map(Attribute)
    }

    pub fn set_attribute(&mut self, attribute: &Attribute) -> Option<Attribute> {
        let res = self.0.set_attribute(attribute.0.clone());
        res.map(Attribute)
    }

    /// Sets new persistent attribute for the user data. If the attribute is already set, it is replaced.
    ///
    /// Parameters
    /// ----------
    /// namespace : str
    ///   Attribute namespace.
    /// name : str
    ///   Attribute name.
    /// hint : str or None
    ///   Attribute hint.
    /// is_hidden : bool
    ///   Attribute hidden flag.
    /// values : List[:py:class:`AttributeValue`] or None
    ///   Attribute values.
    ///
    #[pyo3(signature = (namespace, name, is_hidden = false, hint = None, values = vec![]))]
    pub fn set_persistent_attribute(
        &mut self,
        namespace: &str,
        name: &str,
        is_hidden: bool,
        hint: Option<String>,
        values: Option<Vec<AttributeValue>>,
    ) {
        let values = match values {
            Some(values) => values.into_iter().map(|v| v.0).collect::<Vec<_>>(),
            None => vec![],
        };
        let hint = hint.as_deref();
        self.0
            .set_persistent_attribute(namespace, name, &hint, is_hidden, values)
    }

    /// Sets new temporary attribute for the user data. If the attribute is already set, it is replaced.
    ///
    /// Parameters
    /// ----------
    /// namespace : str
    ///   Attribute namespace.
    /// name : str
    ///   Attribute name.
    /// hint : str or None
    ///   Attribute hint.
    /// is_hidden : bool
    ///   Attribute hidden flag.
    /// values : List[:py:class:`AttributeValue`] or None
    ///   Attribute values.
    ///
    #[pyo3(signature = (namespace, name, is_hidden = false, hint = None, values = vec![]))]
    pub fn set_temporary_attribute(
        &mut self,
        namespace: &str,
        name: &str,
        is_hidden: bool,
        hint: Option<String>,
        values: Option<Vec<AttributeValue>>,
    ) {
        let values = match values {
            Some(values) => values.into_iter().map(|v| v.0).collect::<Vec<_>>(),
            None => vec![],
        };
        let hint = hint.as_deref();
        self.0
            .set_temporary_attribute(namespace, name, &hint, is_hidden, values)
    }

    pub fn clear_attributes(&mut self) {
        self.0.clear_attributes()
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
    fn to_protobuf_gil(&self, no_gil: bool) -> PyResult<Py<PyAny>> {
        let bytes = detach!(no_gil, || {
            self.0.to_pb().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to serialize user data to protobuf: {e}"))
            })
        })?;
        attach!(|py| {
            let bytes = PyBytes::new(py, &bytes);
            Ok(Py::from(bytes))
        })
    }

    #[staticmethod]
    #[pyo3(name = "from_protobuf")]
    #[pyo3(signature = (bytes, no_gil = true))]
    fn from_protobuf_gil(bytes: &Bound<'_, PyBytes>, no_gil: bool) -> PyResult<Self> {
        let bytes = bytes.as_bytes();
        detach!(no_gil, || {
            let obj =
                from_pb::<savant_core::protobuf::UserData, rust::UserData>(bytes).map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "Failed to deserialize user data from protobuf: {e}"
                    ))
                })?;
            Ok(Self(obj))
        })
    }
}
