use crate::json_api::ToSerdeJsonValue;
use crate::primitives::attribute_value::AttributeValue;
use rkyv::{Archive, Deserialize, Serialize};
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
        namespace: &str,
        name: &str,
        values: Vec<AttributeValue>,
        hint: &Option<&str>,
        is_persistent: bool,
        is_hidden: bool,
    ) -> Self {
        AttributeBuilder::default()
            .is_persistent(is_persistent)
            .is_hidden(is_hidden)
            .name(name.to_string())
            .namespace(namespace.to_string())
            .values(values)
            .hint(hint.map(|s| s.to_string()))
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
    fn get_attribute(&self, namespace: &str, name: &str) -> Option<Attribute>;
    fn delete_attribute(&self, namespace: &str, name: &str) -> Option<Attribute>;
    fn set_attribute(&self, attribute: Attribute) -> Option<Attribute>;
    fn clear_attributes(&self);
    fn delete_attributes(&self, namespace: &Option<&str>, names: &[&str]);
    fn find_attributes(
        &self,
        namespace: &Option<&str>,
        names: &[&str],
        hint: &Option<&str>,
    ) -> Vec<(String, String)>;
}

pub trait Attributive: Send {
    fn get_attributes_ref(&self) -> &Vec<Attribute>;
    fn get_attributes_ref_mut(&mut self) -> &mut Vec<Attribute>;
    fn take_attributes(&mut self) -> Vec<Attribute> {
        mem::take(self.get_attributes_ref_mut())
    }
    fn place_attributes(&mut self, mut attributes: Vec<Attribute>) {
        self.get_attributes_ref_mut().append(&mut attributes);
    }

    fn exclude_temporary_attributes(&mut self) -> Vec<Attribute> {
        let attributes = self.take_attributes();
        let (retained, removed): (Vec<Attribute>, Vec<Attribute>) =
            attributes.into_iter().partition(|a| !a.is_temporary());

        self.place_attributes(retained);

        removed
    }

    fn restore_attributes(&mut self, attributes: Vec<Attribute>) {
        attributes.into_iter().for_each(|a| {
            self.set_attribute(a);
        });
    }

    fn get_attributes(&self) -> Vec<(String, String)> {
        self.get_attributes_ref()
            .iter()
            .filter_map(|a| {
                if a.is_hidden {
                    None
                } else {
                    Some((a.namespace.clone(), a.name.clone()))
                }
            })
            .collect()
    }

    fn get_attribute(&self, namespace: &str, name: &str) -> Option<Attribute> {
        self.get_attributes_ref()
            .iter()
            .find(|a| a.namespace == namespace && a.name == name)
            .cloned()
    }

    fn contains_attribute(&self, namespace: &str, name: &str) -> bool {
        self.get_attributes_ref()
            .iter()
            .any(|a| a.namespace == namespace && a.name == name)
    }

    fn delete_attribute(&mut self, namespace: &str, name: &str) -> Option<Attribute> {
        let index = self
            .get_attributes_ref()
            .iter()
            .position(|a| a.namespace == namespace && a.name == name)?;
        Some(self.get_attributes_ref_mut().swap_remove(index))
    }

    fn set_attribute(&mut self, attribute: Attribute) -> Option<Attribute> {
        let index = self
            .get_attributes_ref()
            .iter()
            .position(|a| a.namespace == attribute.namespace && a.name == attribute.name);

        if let Some(index) = index {
            Some(std::mem::replace(
                &mut self.get_attributes_ref_mut()[index],
                attribute,
            ))
        } else {
            self.get_attributes_ref_mut().push(attribute);
            None
        }
    }

    fn clear_attributes(&mut self) {
        self.get_attributes_ref_mut().clear();
    }

    fn delete_attributes(&mut self, namespace: &Option<&str>, names: &[&str]) {
        self.get_attributes_ref_mut().retain(|a| {
            if let Some(namespace) = namespace {
                if a.namespace != *namespace {
                    return true;
                }
            }

            if !names.is_empty() && !names.contains(&a.name.as_str()) {
                return true;
            }

            false
        });
    }

    fn find_attributes(
        &self,
        namespace: &Option<&str>,
        names: &[&str],
        hint: &Option<&str>,
    ) -> Vec<(String, String)> {
        self.get_attributes_ref()
            .iter()
            .filter(|a| {
                if a.is_hidden {
                    return false;
                }

                if let Some(namespace) = &namespace {
                    if a.namespace != *namespace {
                        return false;
                    }
                }

                if !names.is_empty() && !names.contains(&a.name.as_str()) {
                    return false;
                }

                if let (Some(l), Some(r)) = (&hint, &a.hint) {
                    if l != &r.as_str() {
                        return false;
                    }
                }

                true
            })
            .map(|a| (a.namespace.clone(), a.name.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::attribute_value::{AttributeValue, AttributeValueVariant};
    use crate::primitives::{Attribute, Attributive};
    use std::mem;

    #[derive(Default)]
    struct AttrStor {
        attributes: Vec<Attribute>,
    }

    impl Attributive for AttrStor {
        fn get_attributes_ref(&self) -> &Vec<Attribute> {
            &self.attributes
        }

        fn get_attributes_ref_mut(&mut self) -> &mut Vec<Attribute> {
            &mut self.attributes
        }

        fn take_attributes(&mut self) -> Vec<Attribute> {
            mem::take(&mut self.attributes)
        }

        fn place_attributes(&mut self, mut attributes: Vec<Attribute>) {
            self.attributes.append(&mut attributes);
        }
    }

    #[test]
    fn test_get_attribute() {
        let attribute = Attribute::new("system", "test", vec![], &None, true, false);

        let mut t = AttrStor::default();
        t.set_attribute(attribute);

        let attribute = t.get_attribute("system", "test");
        assert!(attribute.is_some());
    }

    #[test]
    fn test_replace_attribute() {
        let attribute = Attribute::new(
            "system",
            "test",
            vec![AttributeValue::float(1.0, None)],
            &None,
            true,
            false,
        );

        let replacement = Attribute::new(
            "system",
            "test",
            vec![AttributeValue::float(2.0, None)],
            &None,
            true,
            false,
        );

        let mut t = AttrStor::default();
        t.set_attribute(attribute);

        let attr_opt = t.set_attribute(replacement);
        let attr = attr_opt.unwrap();
        assert_eq!(t.attributes.len(), 1);
        assert_eq!(attr.values.len(), 1);
        assert!(matches!(
            attr.values[0].get(),
            AttributeValueVariant::Float(v) if *v == 1.0
        ));

        let replacement = t.get_attribute("system", "test");
        let replacement = replacement.unwrap();
        assert_eq!(replacement.values.len(), 1);
        assert!(matches!(
            replacement.values[0].get(),
            AttributeValueVariant::Float(v) if *v == 2.0
        ));
    }

    #[test]
    fn test_delete_attribute() {
        let attribute = Attribute::new("system", "test", vec![], &None, true, false);

        let mut t = AttrStor::default();
        t.set_attribute(attribute);

        let attribute = t.delete_attribute("system", "test");
        assert!(attribute.is_some());
        assert_eq!(t.attributes.len(), 0);
    }

    #[test]
    fn test_contains_attribute() {
        let attribute = Attribute::new("system", "test", vec![], &None, true, false);

        let mut t = AttrStor::default();
        t.set_attribute(attribute);

        assert!(t.contains_attribute("system", "test"));
        assert!(!t.contains_attribute("system", "test2"));
        assert!(!t.contains_attribute("system2", "test"));
    }

    #[test]
    fn test_take_place_attributes() {
        let attribute = Attribute::new("system", "test", vec![], &None, true, false);

        let mut t = AttrStor::default();
        t.set_attribute(attribute.clone());

        let attributes = t.take_attributes();
        assert_eq!(attributes.len(), 1);
        assert_eq!(attributes[0], attribute);
        assert_eq!(t.attributes.len(), 0);

        t.place_attributes(attributes);
        assert_eq!(t.attributes.len(), 1);
        assert_eq!(t.attributes[0], attribute);
    }

    //
    //     #[test]
    //     fn test_find_attributes() {
    //         let t = gen_frame();
    //         let mut attributes = t.find_attributes(&Some("system"), &[], &None);
    //         attributes.sort();
    //         assert_eq!(attributes.len(), 2);
    //         assert_eq!(attributes[0], ("system".to_string(), "test".to_string()));
    //         assert_eq!(attributes[1], ("system".to_string(), "test2".to_string()));
    //
    //         let attributes = t.find_attributes(&Some("system"), &["test"], &None);
    //         assert_eq!(attributes.len(), 1);
    //         assert_eq!(attributes[0], ("system".to_string(), "test".to_string()));
    //
    //         let attributes = t.find_attributes(&Some("system"), &["test"], &Some("test"));
    //         assert_eq!(attributes.len(), 1);
    //         assert_eq!(attributes[0], ("system".to_string(), "test".to_string()));
    //
    //         let mut attributes = t.find_attributes(&None, &[], &Some("test"));
    //         attributes.sort();
    //         assert_eq!(attributes.len(), 2);
    //         assert_eq!(attributes[0], ("system".to_string(), "test".to_string()));
    //         assert_eq!(attributes[1], ("system".to_string(), "test2".to_string()));
    //     }
}
