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
    pub creator: String,
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
    pub fn new(id: i64, creator: String, label: String) -> Self {
        Self { id, creator, label }
    }
}

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone, derive_builder::Builder)]
#[archive(check_bytes)]
pub struct Object {
    pub id: i64,
    #[pyo3(get)]
    pub creator: String,
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
        creator: String,
        label: String,
        bbox: BBox,
        attributes: HashMap<(String, String), Attribute>,
        confidence: Option<f64>,
        parent: Option<ParentObject>,
    ) -> Self {
        Self {
            id,
            creator,
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
                    .map(|((creator, name), _)| (creator.clone(), name.clone()))
                    .collect()
            })
        })
    }

    pub fn get_attribute(&self, creator: String, name: String) -> Option<Attribute> {
        self.attributes.get(&(creator, name)).cloned()
    }

    pub fn delete_attribute(&mut self, creator: String, name: String) -> Option<Attribute> {
        self.attributes.remove(&(creator, name))
    }

    pub fn set_attribute(&mut self, attribute: Attribute) -> Option<Attribute> {
        self.attributes.insert(
            (attribute.creator.clone(), attribute.name.clone()),
            attribute,
        )
    }

    pub fn clear_attributes(&mut self) {
        self.attributes.clear();
    }

    #[pyo3(signature = (negated=false, creator=None, names=vec![]))]
    pub fn delete_attributes(
        &mut self,
        negated: bool,
        creator: Option<String>,
        names: Vec<String>,
    ) {
        Python::with_gil(|py| {
            py.allow_threads(|| {
                self.attributes.retain(|(en, label), _| match creator {
                    Some(ref creator) => {
                        ((names.is_empty() || names.contains(label)) && creator == en) ^ !negated
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
            .creator("model".to_string())
            .label("label".to_string())
            .bbox(BBox::new(0.0, 0.0, 1.0, 1.0, None))
            .confidence(Some(0.5))
            .attributes(
                vec![
                    AttributeBuilder::default()
                        .creator("creator".to_string())
                        .name("name".to_string())
                        .value(Value::string("value".to_string()))
                        .confidence(None)
                        .hint(None)
                        .build()
                        .unwrap(),
                    AttributeBuilder::default()
                        .creator("creator".to_string())
                        .name("name2".to_string())
                        .value(Value::string("value2".to_string()))
                        .confidence(None)
                        .hint(None)
                        .build()
                        .unwrap(),
                    AttributeBuilder::default()
                        .creator("creator2".to_string())
                        .name("name".to_string())
                        .value(Value::string("value".to_string()))
                        .confidence(None)
                        .hint(None)
                        .build()
                        .unwrap(),
                ]
                .into_iter()
                .map(|a| ((a.creator.clone(), a.name.clone()), a))
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
        t.delete_attributes(false, Some("creator".to_string()), vec![]);
        assert_eq!(t.attributes.len(), 1);

        let mut t = o.clone();
        t.delete_attributes(true, Some("creator".to_string()), vec![]);
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
            Some("creator".to_string()),
            vec!["name".to_string(), "name2".to_string()],
        );
        assert_eq!(t.attributes.len(), 1);

        assert_eq!(
            &t.attributes[&("creator2".to_string(), "name".to_string())],
            &AttributeBuilder::default()
                .creator("creator2".to_string())
                .name("name".to_string())
                .value(Value::string("value".to_string()))
                .confidence(None)
                .hint(None)
                .build()
                .unwrap()
        );

        let mut t = o.clone();
        t.delete_attributes(
            true,
            Some("creator".to_string()),
            vec!["name".to_string(), "name2".to_string()],
        );
        assert_eq!(t.attributes.len(), 2);

        drop(o);
    }
}
