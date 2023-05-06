pub mod proxy;

use crate::primitives::{Attribute, BBox};
use pyo3::{pyclass, pymethods, Py, PyAny, Python};
use rkyv::{Archive, Deserialize, Serialize};
use std::collections::HashMap;

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct ParentObject {
    #[pyo3(get, set)]
    pub id: i64,
    #[pyo3(get, set)]
    pub model_name: String,
    #[pyo3(get, set)]
    pub label: String,
}

#[pymethods]
impl ParentObject {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[new]
    pub fn new(id: i64, model_name: String, label: String) -> Self {
        Self {
            id,
            model_name,
            label,
        }
    }
}

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone, derive_builder::Builder)]
#[archive(check_bytes)]
pub struct Object {
    pub id: i64,
    #[pyo3(get)]
    pub model_name: String,
    #[pyo3(get)]
    pub label: String,
    #[pyo3(get)]
    pub bbox: BBox,
    pub attributes: HashMap<(String, String), Attribute>,
    #[pyo3(get)]
    pub confidence: Option<f64>,
    #[pyo3(get)]
    pub parent: Option<ParentObject>,
}

#[pymethods]
impl Object {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[new]
    pub fn new(
        id: i64,
        model_name: String,
        label: String,
        bbox: BBox,
        attributes: HashMap<(String, String), Attribute>,
        confidence: Option<f64>,
        parent: Option<ParentObject>,
    ) -> Self {
        Self {
            id,
            model_name,
            label,
            bbox,
            confidence,
            attributes,
            parent,
        }
    }

    pub fn attributes(&self) -> Vec<(String, String)> {
        Python::with_gil(|py| {
            py.allow_threads(|| {
                self.attributes
                    .iter()
                    .map(|((element_name, name), _)| (element_name.clone(), name.clone()))
                    .collect()
            })
        })
    }

    pub fn get_attribute(&self, element_name: String, name: String) -> Option<Attribute> {
        self.attributes.get(&(element_name, name)).cloned()
    }

    pub fn delete_attribute(&mut self, element_name: String, name: String) -> Option<Attribute> {
        self.attributes.remove(&(element_name, name))
    }

    pub fn set_attribute(&mut self, attribute: Attribute) -> Option<Attribute> {
        self.attributes.insert(
            (attribute.element_name.clone(), attribute.name.clone()),
            attribute,
        )
    }

    pub fn clear_attributes(&mut self) {
        self.attributes.clear();
    }

    #[pyo3(signature = (negated=false, element_name=None, names=vec![]))]
    pub fn delete_attributes(
        &mut self,
        negated: bool,
        element_name: Option<String>,
        names: Vec<String>,
    ) {
        Python::with_gil(|py| {
            py.allow_threads(|| {
                self.attributes.retain(|(en, label), _| match element_name {
                    Some(ref element_name) => {
                        ((names.is_empty() || names.contains(label)) && element_name == en)
                            ^ !negated
                    }
                    None => names.contains(label) ^ !negated,
                })
            })
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::{AttributeBuilder, BBox, ObjectBuilder, Value};

    #[test]
    fn test_delete_attributes() {
        pyo3::prepare_freethreaded_python();
        let o = ObjectBuilder::default()
            .id(1)
            .model_name("model".to_string())
            .label("label".to_string())
            .bbox(BBox::new(0.0, 0.0, 1.0, 1.0, None))
            .confidence(Some(0.5))
            .attributes(
                vec![
                    AttributeBuilder::default()
                        .element_name("element_name".to_string())
                        .name("name".to_string())
                        .value(Value::new_string("value".to_string()))
                        .confidence(None)
                        .build()
                        .unwrap(),
                    AttributeBuilder::default()
                        .element_name("element_name".to_string())
                        .name("name2".to_string())
                        .value(Value::new_string("value2".to_string()))
                        .confidence(None)
                        .build()
                        .unwrap(),
                    AttributeBuilder::default()
                        .element_name("element_name2".to_string())
                        .name("name".to_string())
                        .value(Value::new_string("value".to_string()))
                        .confidence(None)
                        .build()
                        .unwrap(),
                ]
                .into_iter()
                .map(|a| ((a.element_name.clone(), a.name.clone()), a))
                .collect(),
            )
            .parent(None)
            .build()
            .unwrap();

        let mut t = o.clone();
        t.delete_attributes(false, None, vec![]);
        assert_eq!(t.attributes.len(), 3);

        let mut t = o.clone();
        t.delete_attributes(true, None, vec![]);
        assert!(t.attributes.is_empty());

        let mut t = o.clone();
        t.delete_attributes(false, Some("element_name".to_string()), vec![]);
        assert_eq!(t.attributes.len(), 1);

        let mut t = o.clone();
        t.delete_attributes(true, Some("element_name".to_string()), vec![]);
        assert_eq!(t.attributes.len(), 2);

        let mut t = o.clone();
        t.delete_attributes(false, None, vec!["name".to_string()]);
        assert_eq!(t.attributes.len(), 1);

        let mut t = o.clone();
        t.delete_attributes(true, None, vec!["name".to_string()]);
        assert_eq!(t.attributes.len(), 2);

        let mut t = o.clone();
        t.delete_attributes(false, None, vec!["name".to_string(), "name2".to_string()]);
        assert_eq!(t.attributes.len(), 0);

        let mut t = o.clone();
        t.delete_attributes(true, None, vec!["name".to_string(), "name2".to_string()]);
        assert_eq!(t.attributes.len(), 3);

        let mut t = o.clone();
        t.delete_attributes(
            false,
            Some("element_name".to_string()),
            vec!["name".to_string(), "name2".to_string()],
        );
        assert_eq!(t.attributes.len(), 1);

        assert_eq!(
            &t.attributes[&("element_name2".to_string(), "name".to_string())],
            &AttributeBuilder::default()
                .element_name("element_name2".to_string())
                .name("name".to_string())
                .value(Value::new_string("value".to_string()))
                .confidence(None)
                .build()
                .unwrap()
        );

        let mut t = o.clone();
        t.delete_attributes(
            true,
            Some("element_name".to_string()),
            vec!["name".to_string(), "name2".to_string()],
        );
        assert_eq!(t.attributes.len(), 2);

        drop(o);
    }
}
