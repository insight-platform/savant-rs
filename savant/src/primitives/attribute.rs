use crate::primitives::attribute_value::AttributeValue;
use crate::primitives::attribute_value::AttributeValuesView;
use pyo3::{pyclass, pymethods, Py, PyAny};
use rkyv::{Archive, Deserialize, Serialize};
use savant_core::primitives::attribute_value::AttributeValue as AttributeValueRs;
use savant_core::to_json_value::ToSerdeJsonValue;
use std::collections::HashMap;
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
#[derive(
    Archive, Deserialize, Serialize, Debug, PartialEq, Clone, derive_builder::Builder, Default,
)]
#[archive(check_bytes)]
pub struct Attribute {
    #[pyo3(get)]
    pub namespace: String,
    #[pyo3(get)]
    pub name: String,
    #[builder(setter(custom))]
    pub values: Arc<Vec<AttributeValueRs>>,
    #[pyo3(get)]
    pub hint: Option<String>,
    #[pyo3(get)]
    #[builder(default = "true")]
    pub is_persistent: bool,
}

impl AttributeBuilder {
    pub fn values(&mut self, vals: Vec<AttributeValue>) -> &mut Self {
        let vals =
            unsafe { std::mem::transmute::<Vec<AttributeValue>, Vec<AttributeValueRs>>(vals) };
        self.values = Some(Arc::new(vals));
        self
    }
}

impl ToSerdeJsonValue for Attribute {
    fn to_serde_json_value(&self) -> serde_json::Value {
        serde_json::json!({
            "namespace": self.namespace,
            "name": self.name,
            "values": self.values.iter().map(|v| v.to_serde_json_value()).collect::<Vec<_>>(),
            "hint": self.hint,
        })
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
            unsafe { std::mem::transmute::<Vec<AttributeValue>, Vec<AttributeValueRs>>(values) };
        Self {
            is_persistent,
            namespace,
            name,
            values: Arc::new(values),
            hint,
        }
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
            unsafe { std::mem::transmute::<Vec<AttributeValue>, Vec<AttributeValueRs>>(values) };
        Self {
            is_persistent: true,
            namespace,
            name,
            values: Arc::new(values),
            hint,
        }
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
            unsafe { std::mem::transmute::<Vec<AttributeValue>, Vec<AttributeValueRs>>(values) };
        Self {
            is_persistent: false,
            namespace,
            name,
            values: Arc::new(values),
            hint,
        }
    }

    /// Returns ``True`` if the attribute is persistent, ``False`` otherwise.
    ///
    /// Returns
    /// -------
    /// bool
    ///   ``True`` if the attribute is persistent, ``False`` otherwise.
    ///
    pub fn is_temporary(&self) -> bool {
        !self.is_persistent
    }

    /// Changes the attribute to be persistent.
    ///
    /// Returns
    /// -------
    /// None
    ///   The attribute is changed in-place.
    ///
    pub fn make_persistent(&mut self) {
        self.is_persistent = true;
    }

    /// Changes the attribute to be non-persistent.
    ///
    /// Returns
    /// -------
    /// None
    ///   The attribute is changed in-place.
    ///
    pub fn make_temporary(&mut self) {
        self.is_persistent = false;
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
        self.namespace.clone()
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
        self.name.clone()
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
            mem::transmute::<Vec<AttributeValueRs>, Vec<AttributeValue>>((*self.values).clone())
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
            inner: self.values.clone(),
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
        self.hint.clone()
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
        self.hint = hint;
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
        self.values = Arc::new(unsafe {
            mem::transmute::<Vec<AttributeValue>, Vec<AttributeValueRs>>(values)
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

pub trait Attributive: Send {
    fn get_attributes_ref(&self) -> &HashMap<(String, String), Attribute>;
    fn get_attributes_ref_mut(&mut self) -> &mut HashMap<(String, String), Attribute>;
    fn take_attributes(&mut self) -> HashMap<(String, String), Attribute>;
    fn place_attributes(&mut self, attributes: HashMap<(String, String), Attribute>);

    fn exclude_temporary_attributes(&mut self) -> Vec<Attribute> {
        let attributes = self.take_attributes();
        let (retained, removed): (Vec<Attribute>, Vec<Attribute>) =
            attributes.into_values().partition(|a| !a.is_temporary());

        self.place_attributes(
            retained
                .into_iter()
                .map(|a| ((a.namespace.clone(), a.name.clone()), a))
                .collect(),
        );

        removed
    }

    fn restore_attributes(&mut self, attributes: Vec<Attribute>) {
        let attrs = self.get_attributes_ref_mut();
        attributes.into_iter().for_each(|a| {
            attrs.insert((a.namespace.clone(), a.name.clone()), a);
        })
    }

    fn get_attributes(&self) -> Vec<(String, String)> {
        self.get_attributes_ref()
            .iter()
            .map(|((namespace, name), _)| (namespace.clone(), name.clone()))
            .collect()
    }

    fn get_attribute(&self, namespace: String, name: String) -> Option<Attribute> {
        self.get_attributes_ref().get(&(namespace, name)).cloned()
    }

    fn delete_attribute(&mut self, namespace: String, name: String) -> Option<Attribute> {
        self.get_attributes_ref_mut().remove(&(namespace, name))
    }

    fn set_attribute(&mut self, attribute: Attribute) -> Option<Attribute> {
        self.get_attributes_ref_mut().insert(
            (attribute.namespace.clone(), attribute.name.clone()),
            attribute,
        )
    }

    fn clear_attributes(&mut self) {
        self.get_attributes_ref_mut().clear();
    }

    fn delete_attributes(&mut self, namespace: Option<String>, names: Vec<String>) {
        self.get_attributes_ref_mut().retain(|(c, label), _| {
            if let Some(namespace) = &namespace {
                if c != namespace {
                    return true;
                }
            }

            if !names.is_empty() && !names.contains(label) {
                return true;
            }

            false
        });
    }

    fn find_attributes(
        &self,
        namespace: Option<String>,
        names: Vec<String>,
        hint: Option<String>,
    ) -> Vec<(String, String)> {
        self.get_attributes_ref()
            .iter()
            .filter(|((_, _), a)| {
                if let Some(namespace) = &namespace {
                    if a.namespace != *namespace {
                        return false;
                    }
                }

                if !names.is_empty() && !names.contains(&a.name) {
                    return false;
                }

                if let Some(hint) = &hint {
                    if a.hint.as_ref() != Some(hint) {
                        return false;
                    }
                }

                true
            })
            .map(|((c, n), _)| (c.clone(), n.clone()))
            .collect()
    }
}
