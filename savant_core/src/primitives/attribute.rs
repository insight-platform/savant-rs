use crate::json_api::ToSerdeJsonValue;
use crate::primitives::attribute_value::AttributeValue;
use hashbrown::HashMap;
use rkyv::{Archive, Deserialize, Serialize};
use std::sync::Arc;

/// Attribute represents a specific knowledge about certain entity. The attribute is identified by ``(creator, label)`` pair which is unique within the entity.
/// The attribute value is a list of values, each of which has a confidence score. The attribute may include additional information in the form of a hint.
/// There are two kinds of attributes: persistent and non-persistent. Persistent attributes are serialized, while non-persistent are not.
///
/// The list nature of attribute values is used to represent complex values of the same attribute.
/// For example, the attribute ``(person_profiler, bio)`` may include values in the form ``["Age", 32, "Gender", None, "Height", 186]``. Each element of the
/// list is :class:`AttributeValue`.
///
#[derive(
    Archive,
    Deserialize,
    Serialize,
    Debug,
    PartialEq,
    Clone,
    derive_builder::Builder,
    Default,
    serde::Serialize,
    serde::Deserialize,
)]
#[archive(check_bytes)]
pub struct Attribute {
    pub namespace: String,
    pub name: String,
    #[builder(setter(custom))]
    pub values: Arc<Vec<AttributeValue>>,
    pub hint: Option<String>,
    #[builder(default = "true")]
    pub is_persistent: bool,
    #[builder(default = "false")]
    pub is_hidden: bool,
}

impl AttributeBuilder {
    pub fn values(&mut self, vals: Vec<AttributeValue>) -> &mut Self {
        self.values = Some(Arc::new(vals));
        self
    }
}

impl ToSerdeJsonValue for Attribute {
    fn to_serde_json_value(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap()
    }
}

impl Attribute {
    pub fn new(
        namespace: String,
        name: String,
        values: Vec<AttributeValue>,
        hint: Option<String>,
        is_persistent: bool,
        is_hidden: bool,
    ) -> Self {
        AttributeBuilder::default()
            .is_persistent(is_persistent)
            .is_hidden(is_hidden)
            .name(name)
            .namespace(namespace)
            .values(values)
            .hint(hint)
            .build()
            .unwrap()
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
    pub fn persistent(
        namespace: String,
        name: String,
        values: Vec<AttributeValue>,
        hint: Option<String>,
        is_hidden: bool,
    ) -> Self {
        AttributeBuilder::default()
            .is_persistent(true)
            .is_hidden(is_hidden)
            .name(name)
            .namespace(namespace)
            .values(values)
            .hint(hint)
            .build()
            .unwrap()
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
    pub fn temporary(
        namespace: String,
        name: String,
        values: Vec<AttributeValue>,
        hint: Option<String>,
        is_hidden: bool,
    ) -> Self {
        AttributeBuilder::default()
            .is_persistent(false)
            .is_hidden(is_hidden)
            .name(name)
            .namespace(namespace)
            .values(values)
            .hint(hint)
            .build()
            .unwrap()
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
    pub fn get_namespace(&self) -> &str {
        &self.namespace
    }

    /// Returns the name of the attribute.
    ///
    /// Returns
    /// -------
    /// str
    ///   The name of the attribute.
    ///
    pub fn get_name(&self) -> &str {
        &self.name
    }

    /// Returns the values of the attribute. The values are returned as copies, changing them will not change the attribute. To change the values of the
    /// attribute, use assignment to the ``values`` attribute.
    ///
    /// Returns
    /// -------
    /// List[:class:`AttributeValue`]
    ///   The values of the attribute.
    ///
    pub fn get_values(&self) -> &Vec<AttributeValue> {
        &self.values
    }

    /// Returns the hint of the attribute.
    ///
    /// Returns
    /// -------
    /// str or None
    ///   The hint of the attribute or ``None`` if no hint is set.
    ///
    pub fn get_hint(&self) -> &Option<String> {
        &self.hint
    }

    /// Sets the hint of the attribute.
    ///
    /// Parameters
    /// ----------
    /// hint : str or None
    ///   The hint of the attribute or ``None`` if no hint is set.
    ///
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
    pub fn set_values(&mut self, values: Vec<AttributeValue>) {
        self.values = Arc::new(values);
    }

    pub fn to_json(&self) -> anyhow::Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
}

pub trait AttributeMethods {
    fn exclude_temporary_attributes(&self) -> Vec<Attribute>;
    fn restore_attributes(&self, frame_attributes: Vec<Attribute>);
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
            .filter_map(|((namespace, name), a)| {
                if a.is_hidden {
                    None
                } else {
                    Some((namespace.clone(), name.clone()))
                }
            })
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
                if a.is_hidden {
                    return false;
                }

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
