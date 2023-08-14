use crate::primitives::attribute_value::AttributeValue;
use crate::primitives::attribute_value::AttributeValuesView;
use pyo3::{pyclass, pymethods, Py, PyAny};
use savant_core::primitives::rust;
use savant_core::to_json_value::ToSerdeJsonValue;
use std::mem;
use std::sync::Arc;

/// Attribute represents a specific knowledge about certain entity. The attribute is identified by ``(creator, label)`` pair which is unique within the entity.
/// The attribute value is a list of values, each of which has a confidence score. The attribute may include additional information in the form of a hint.
/// There are two kinds of attributes: persistent and non-persistent. Persistent attributes are serialized, while non-persistent are not.
///
/// The list nature of attribute values is used to represent complex values of the same attribute.
/// For example, the attribute ``(person_profiler, bio)`` may include values in the form ``["Age", 32, "Gender", None, "Height", 186]``. Each element of the
/// list is :class:`AttributeValue`.
///
#[pyclass]
#[derive(Debug, PartialEq, Clone, Default)]
pub struct Attribute(rust::Attribute);

impl Attribute {
    pub fn values(&mut self, vals: Vec<AttributeValue>) -> &mut Self {
        let vals =
            unsafe { mem::transmute::<Vec<AttributeValue>, Vec<rust::AttributeValue>>(vals) };
        self.0.values = Arc::new(vals);
        self
    }
}

impl ToSerdeJsonValue for Attribute {
    fn to_serde_json_value(&self) -> serde_json::Value {
        self.0.to_serde_json_value()
    }
}

#[pymethods]
impl Attribute {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[new]
    #[pyo3(signature = (namespace, name , values, hint = None, is_persistent = true))]
    pub fn new(
        namespace: String,
        name: String,
        values: Vec<AttributeValue>,
        hint: Option<String>,
        is_persistent: bool,
    ) -> Self {
        let values =
            unsafe { mem::transmute::<Vec<AttributeValue>, Vec<rust::AttributeValue>>(values) };

        Self(rust::Attribute::new(
            namespace,
            name,
            values,
            hint,
            is_persistent,
        ))
    }

    /// Alias to constructor method. Creates a persistent attribute.
    ///
    /// Parameters
    /// ----------
    /// namespace : str
    ///   The namespace of the attribute.
    /// name : str
    ///   The name of the attribute.
    /// values : List[:class:`AttributeValue`]
    ///   The values of the attribute.
    /// hint : str, optional
    ///   The hint of the attribute. The hint is a user-defined string that may contain additional information about the attribute.
    ///
    /// Returns
    /// -------
    /// :class:`Attribute`
    ///   The created attribute.
    ///
    #[staticmethod]
    pub fn persistent(
        namespace: String,
        name: String,
        values: Vec<AttributeValue>,
        hint: Option<String>,
    ) -> Self {
        let values =
            unsafe { mem::transmute::<Vec<AttributeValue>, Vec<rust::AttributeValue>>(values) };
        Self(rust::Attribute::new(namespace, name, values, hint, true))
    }

    /// Alias to constructor method for non-persistent attributes.
    ///
    /// Parameters
    /// ----------
    /// namespace : str
    ///   The namespace of the attribute.
    /// name : str
    ///   The name of the attribute.
    /// values : List[:class:`AttributeValue`]
    ///   The values of the attribute.
    /// hint : str, optional
    ///   The hint of the attribute. The hint is a user-defined string that may contain additional information about the attribute.
    ///
    /// Returns
    /// -------
    /// :class:`Attribute`
    ///   The created attribute.
    ///
    #[staticmethod]
    pub fn temporary(
        namespace: String,
        name: String,
        values: Vec<AttributeValue>,
        hint: Option<String>,
    ) -> Self {
        let values =
            unsafe { mem::transmute::<Vec<AttributeValue>, Vec<rust::AttributeValue>>(values) };
        Self(rust::Attribute::new(namespace, name, values, hint, false))
    }

    /// Returns ``True`` if the attribute is persistent, ``False`` otherwise.
    ///
    /// Returns
    /// -------
    /// bool
    ///   ``True`` if the attribute is persistent, ``False`` otherwise.
    ///
    pub fn is_temporary(&self) -> bool {
        !self.0.is_persistent
    }

    /// Changes the attribute to be persistent.
    ///
    /// Returns
    /// -------
    /// None
    ///   The attribute is changed in-place.
    ///
    pub fn make_persistent(&mut self) {
        self.0.is_persistent = true;
    }

    /// Changes the attribute to be non-persistent.
    ///
    /// Returns
    /// -------
    /// None
    ///   The attribute is changed in-place.
    ///
    pub fn make_temporary(&mut self) {
        self.0.is_persistent = false;
    }

    /// Returns the namespace of the attribute.
    ///
    /// Returns
    /// -------
    /// str
    ///   The namespace of the attribute.
    ///
    #[getter]
    pub fn get_namespace(&self) -> String {
        self.0.get_namespace().to_string()
    }

    /// Returns the name of the attribute.
    ///
    /// Returns
    /// -------
    /// str
    ///   The name of the attribute.
    ///
    #[getter]
    pub fn get_name(&self) -> String {
        self.0.get_name().to_string()
    }

    /// Returns the values of the attribute. The values are returned as copies, changing them will not change the attribute. To change the values of the
    /// attribute, use assignment to the ``values`` attribute.
    ///
    /// Returns
    /// -------
    /// List[:class:`AttributeValue`]
    ///   The values of the attribute.
    ///
    #[getter]
    pub fn get_values(&self) -> Vec<AttributeValue> {
        unsafe {
            mem::transmute::<Vec<rust::AttributeValue>, Vec<AttributeValue>>(
                (*self.0.values).clone(),
            )
        }
    }

    /// Returns a link to attributes without retrieving them. It is convenience method if you need to access certain value.
    ///
    /// Returns
    /// -------
    ///
    #[getter]
    pub fn values_view(&self) -> AttributeValuesView {
        AttributeValuesView {
            inner: self.0.values.clone(),
        }
    }

    /// Returns the hint of the attribute.
    ///
    /// Returns
    /// -------
    /// str or None
    ///   The hint of the attribute or ``None`` if no hint is set.
    ///
    #[getter]
    pub fn get_hint(&self) -> Option<String> {
        self.0.hint.clone()
    }

    /// Sets the hint of the attribute.
    ///
    /// Parameters
    /// ----------
    /// hint : str or None
    ///   The hint of the attribute or ``None`` if no hint is set.
    ///
    #[setter]
    pub fn set_hint(&mut self, hint: Option<String>) {
        self.0.hint = hint;
    }

    /// Sets the values of the attribute.
    ///
    /// Parameters
    /// ----------
    /// values : List[:class:`AttributeValue`]
    ///   The values of the attribute.
    ///
    #[setter]
    pub fn set_values(&mut self, values: Vec<AttributeValue>) {
        self.0.values = Arc::new(unsafe {
            mem::transmute::<Vec<AttributeValue>, Vec<rust::AttributeValue>>(values)
        });
    }
}

pub trait AttributeMethods {
    fn exclude_temporary_attributes(&self) -> Vec<Attribute>;
    fn restore_attributes(&self, attributes: Vec<Attribute>);
    fn get_attributes(&self) -> Vec<(String, String)>;
    fn get_attribute(&self, namespace: String, name: String) -> Option<Attribute>;
    fn delete_attribute(&self, namespace: String, name: String) -> Option<Attribute>;
    fn set_attribute(&self, attribute: Attribute) -> Option<Attribute>;
    fn clear_attributes(&self);
    fn delete_attributes(&self, namespace: Option<String>, names: Vec<String>);
    fn find_attributes(
        &self,
        namespace: Option<String>,
        names: Vec<String>,
        hint: Option<String>,
    ) -> Vec<(String, String)>;
}
