use crate::primitives::attribute::{Attributive, InnerAttributes};
use crate::primitives::to_json_value::ToSerdeJsonValue;
use crate::primitives::{Attribute, RBBox};
use crate::utils::python::no_gil;
use crate::utils::symbol_mapper::get_object_id;
use pyo3::{pyclass, pymethods, Py, PyAny};
use rkyv::{with::Skip, Archive, Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub mod query;

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone, derive_builder::Builder)]
#[archive(check_bytes)]
pub struct ParentObject {
    #[pyo3(get, set)]
    pub id: i64,
    #[pyo3(get, set)]
    pub creator: String,
    #[pyo3(get, set)]
    pub label: String,
    #[with(Skip)]
    #[builder(default)]
    pub creator_id: Option<i64>,
    #[with(Skip)]
    #[builder(default)]
    pub label_id: Option<i64>,
}

impl Default for ParentObject {
    fn default() -> Self {
        Self {
            id: 0,
            creator: "".to_string(),
            label: "".to_string(),
            creator_id: None,
            label_id: None,
        }
    }
}

impl ToSerdeJsonValue for ParentObject {
    fn to_serde_json_value(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "creator": self.creator,
            "label": self.label,
        })
    }
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
        let (creator_id, label_id) =
            get_object_id(&creator, &label).map_or((None, None), |(c, o)| (Some(c), Some(o)));

        Self {
            id,
            creator,
            label,
            creator_id,
            label_id,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone, PartialEq)]
pub enum Modification {
    Id,
    Creator,
    Label,
    BoundingBox,
    Attributes,
    Confidence,
    Parent,
    TrackId,
}

impl ToSerdeJsonValue for Modification {
    fn to_serde_json_value(&self) -> serde_json::Value {
        serde_json::json!(format!("{:?}", self))
    }
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone, derive_builder::Builder)]
#[archive(check_bytes)]
pub struct InnerObject {
    pub id: i64,
    pub creator: String,
    pub label: String,
    pub bbox: RBBox,
    #[builder(default)]
    pub attributes: HashMap<(String, String), Attribute>,
    #[builder(default)]
    pub confidence: Option<f64>,
    #[builder(default)]
    pub parent: Option<ParentObject>,
    #[builder(default)]
    pub track_id: Option<i64>,
    #[with(Skip)]
    #[builder(default)]
    pub modifications: Vec<Modification>,
    #[with(Skip)]
    #[builder(default)]
    pub creator_id: Option<i64>,
    #[with(Skip)]
    #[builder(default)]
    pub label_id: Option<i64>,
}

impl Default for InnerObject {
    fn default() -> Self {
        Self {
            id: 0,
            creator: "".to_string(),
            label: "".to_string(),
            bbox: RBBox::default(),
            attributes: HashMap::new(),
            confidence: None,
            parent: None,
            track_id: None,
            modifications: Vec::new(),
            creator_id: None,
            label_id: None,
        }
    }
}

impl ToSerdeJsonValue for InnerObject {
    fn to_serde_json_value(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "creator": self.creator,
            "label": self.label,
            "bbox": self.bbox.to_serde_json_value(),
            "attributes": self.attributes.values().map(|v| v.to_serde_json_value()).collect::<Vec<_>>(),
            "confidence": self.confidence,
            "parent": self.parent.as_ref().map(|p| p.to_serde_json_value()),
            "track_id": self.track_id,
            "modifications": self.modifications.iter().map(|m| m.to_serde_json_value()).collect::<Vec<serde_json::Value>>(),
        })
    }
}

impl InnerAttributes for InnerObject {
    fn get_attributes_ref(&self) -> &HashMap<(String, String), Attribute> {
        &self.attributes
    }

    fn get_attributes_ref_mut(&mut self) -> &mut HashMap<(String, String), Attribute> {
        &mut self.attributes
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Object {
    pub(crate) inner: Arc<Mutex<InnerObject>>,
}

impl ToSerdeJsonValue for Object {
    fn to_serde_json_value(&self) -> serde_json::Value {
        self.inner.lock().unwrap().to_serde_json_value()
    }
}

impl Attributive<InnerObject> for Object {
    fn get_inner(&self) -> Arc<Mutex<InnerObject>> {
        self.inner.clone()
    }
}

impl Object {
    #[cfg(test)]
    pub(crate) fn from_inner_object(object: InnerObject) -> Self {
        Self {
            inner: Arc::new(Mutex::new(object)),
        }
    }

    pub(crate) fn from_arc_inner_object(object: Arc<Mutex<InnerObject>>) -> Self {
        Self { inner: object }
    }
}

#[pymethods]
impl Object {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:#?}", self.inner.lock().unwrap())
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[allow(clippy::too_many_arguments)]
    #[new]
    pub fn new(
        id: i64,
        creator: String,
        label: String,
        bbox: RBBox,
        attributes: HashMap<(String, String), Attribute>,
        confidence: Option<f64>,
        parent: Option<ParentObject>,
        track_id: Option<i64>,
    ) -> Self {
        let (creator_id, label_id) =
            get_object_id(&creator, &label).map_or((None, None), |(c, o)| (Some(c), Some(o)));

        let object = InnerObject {
            id,
            creator,
            label,
            bbox,
            attributes,
            confidence,
            parent,
            track_id,
            modifications: Vec::default(),
            creator_id,
            label_id,
        };
        Self {
            inner: Arc::new(Mutex::new(object)),
        }
    }

    #[getter]
    pub fn get_track_id(&self) -> Option<i64> {
        let object = self.inner.lock().unwrap();
        object.track_id
    }

    #[getter]
    pub fn get_id(&self) -> i64 {
        self.inner.lock().unwrap().id
    }

    #[getter]
    pub fn get_creator(&self) -> String {
        self.inner.lock().unwrap().creator.clone()
    }

    #[getter]
    pub fn get_label(&self) -> String {
        self.inner.lock().unwrap().label.clone()
    }

    #[getter]
    pub fn get_bbox(&self) -> crate::primitives::RBBox {
        self.inner.lock().unwrap().bbox.clone()
    }

    #[getter]
    pub fn get_confidence(&self) -> Option<f64> {
        let object = self.inner.lock().unwrap();
        object.confidence
    }

    #[getter]
    pub fn get_parent(&self) -> Option<ParentObject> {
        let object = self.inner.lock().unwrap();
        object.parent.clone()
    }

    #[setter]
    pub fn set_track_id(&mut self, track_id: Option<i64>) {
        let mut object = self.inner.lock().unwrap();
        object.track_id = track_id;
        object.modifications.push(Modification::TrackId);
    }

    #[setter]
    pub fn set_id(&mut self, id: i64) {
        let mut object = self.inner.lock().unwrap();
        object.id = id;
        object.modifications.push(Modification::Id);
    }

    #[setter]
    pub fn set_creator(&mut self, creator: String) {
        let mut object = self.inner.lock().unwrap();
        object.creator = creator;
        object.modifications.push(Modification::Creator);
    }

    #[setter]
    pub fn set_label(&mut self, label: String) {
        let mut object = self.inner.lock().unwrap();
        object.label = label;
        object.modifications.push(Modification::Label);
    }

    #[setter]
    pub fn set_bbox(&mut self, bbox: RBBox) {
        let mut object = self.inner.lock().unwrap();
        object.bbox = bbox;
        object.modifications.push(Modification::BoundingBox);
    }

    #[setter]
    pub fn set_confidence(&mut self, confidence: Option<f64>) {
        let mut object = self.inner.lock().unwrap();
        object.confidence = confidence;
        object.modifications.push(Modification::Confidence);
    }

    #[setter]
    pub fn set_parent(&mut self, parent: Option<ParentObject>) {
        let mut object = self.inner.lock().unwrap();
        object.parent = parent;
        object.modifications.push(Modification::Parent);
    }

    #[getter]
    pub fn attributes(&self) -> Vec<(String, String)> {
        no_gil(|| self.get_attributes())
    }

    #[pyo3(name = "get_attribute")]
    pub fn get_attribute_py(&self, creator: String, name: String) -> Option<Attribute> {
        self.get_attribute(creator, name)
    }

    #[pyo3(name = "delete_attribute")]
    pub fn delete_attribute_py(&mut self, creator: String, name: String) -> Option<Attribute> {
        match self.delete_attribute(creator, name) {
            Some(attribute) => {
                let mut object = self.inner.lock().unwrap();
                object.modifications.push(Modification::Attributes);
                Some(attribute)
            }
            None => None,
        }
    }

    #[pyo3(name = "set_attribute")]
    pub fn set_attribute_py(&mut self, attribute: Attribute) -> Option<Attribute> {
        {
            let mut object = self.inner.lock().unwrap();
            object.modifications.push(Modification::Attributes);
        }
        self.set_attribute(attribute)
    }

    #[pyo3(name = "clear_attributes")]
    pub fn clear_attributes_py(&mut self) {
        {
            let mut object = self.inner.lock().unwrap();
            object.modifications.push(Modification::Attributes);
        }
        self.clear_attributes()
    }

    #[pyo3(signature = (negated=false, creator=None, names=vec![]))]
    #[pyo3(name = "delete_attributes")]
    pub fn delete_attributes_py(
        &mut self,
        negated: bool,
        creator: Option<String>,
        names: Vec<String>,
    ) {
        no_gil(move || {
            {
                let mut object = self.inner.lock().unwrap();
                object.modifications.push(Modification::Attributes);
            }
            self.delete_attributes(negated, creator, names)
        })
    }

    #[pyo3(name = "find_attributes")]
    pub fn find_attributes_py(
        &self,
        creator: Option<String>,
        name: Option<String>,
        hint: Option<String>,
    ) -> Vec<(String, String)> {
        no_gil(|| self.find_attributes(creator, name, hint))
    }

    pub fn take_modifications(&self) -> Vec<Modification> {
        let mut object = self.inner.lock().unwrap();
        std::mem::take(&mut object.modifications)
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::attribute::Attributive;
    use crate::primitives::message::video::object::InnerObjectBuilder;
    use crate::primitives::{AttributeBuilder, Modification, Object, RBBox, Value};

    fn get_object() -> Object {
        Object::from_inner_object(
            InnerObjectBuilder::default()
                .id(1)
                .track_id(None)
                .modifications(vec![])
                .creator("model".to_string())
                .label("label".to_string())
                .bbox(RBBox::new(0.0, 0.0, 1.0, 1.0, None))
                .confidence(Some(0.5))
                .attributes(
                    vec![
                        AttributeBuilder::default()
                            .creator("creator".to_string())
                            .name("name".to_string())
                            .values(vec![Value::string("value".to_string(), None)])
                            .hint(None)
                            .build()
                            .unwrap(),
                        AttributeBuilder::default()
                            .creator("creator".to_string())
                            .name("name2".to_string())
                            .values(vec![Value::string("value2".to_string(), None)])
                            .hint(None)
                            .build()
                            .unwrap(),
                        AttributeBuilder::default()
                            .creator("creator2".to_string())
                            .name("name".to_string())
                            .values(vec![Value::string("value".to_string(), None)])
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
                .unwrap(),
        )
    }

    #[test]
    fn test_delete_attributes() {
        pyo3::prepare_freethreaded_python();

        let mut t = get_object();
        t.delete_attributes(false, None, vec![]);
        assert_eq!(t.inner.lock().unwrap().attributes.len(), 3);

        let mut t = get_object();
        t.delete_attributes(true, None, vec![]);
        assert!(t.inner.lock().unwrap().attributes.is_empty());

        let mut t = get_object();
        t.delete_attributes(false, Some("creator".to_string()), vec![]);
        assert_eq!(t.inner.lock().unwrap().attributes.len(), 1);

        let mut t = get_object();
        t.delete_attributes(true, Some("creator".to_string()), vec![]);
        assert_eq!(t.inner.lock().unwrap().attributes.len(), 2);

        let mut t = get_object();
        t.delete_attributes(false, None, vec!["name".to_string()]);
        assert_eq!(t.inner.lock().unwrap().attributes.len(), 1);

        let mut t = get_object();
        t.delete_attributes(true, None, vec!["name".to_string()]);
        assert_eq!(t.inner.lock().unwrap().attributes.len(), 2);

        let mut t = get_object();
        t.delete_attributes(false, None, vec!["name".to_string(), "name2".to_string()]);
        assert_eq!(t.inner.lock().unwrap().attributes.len(), 0);

        let mut t = get_object();
        t.delete_attributes(true, None, vec!["name".to_string(), "name2".to_string()]);
        assert_eq!(t.inner.lock().unwrap().attributes.len(), 3);

        let mut t = get_object();
        t.delete_attributes(
            false,
            Some("creator".to_string()),
            vec!["name".to_string(), "name2".to_string()],
        );
        assert_eq!(t.inner.lock().unwrap().attributes.len(), 1);

        assert_eq!(
            &t.inner.lock().unwrap().attributes[&("creator2".to_string(), "name".to_string())],
            &AttributeBuilder::default()
                .creator("creator2".to_string())
                .name("name".to_string())
                .values(vec![Value::string("value".to_string(), None)])
                .hint(None)
                .build()
                .unwrap()
        );

        let mut t = get_object();
        t.delete_attributes(
            true,
            Some("creator".to_string()),
            vec!["name".to_string(), "name2".to_string()],
        );
        assert_eq!(t.inner.lock().unwrap().attributes.len(), 2);
    }

    #[test]
    fn test_modifications() {
        let mut t = get_object();
        t.set_label("label2".to_string());
        assert_eq!(t.take_modifications(), vec![Modification::Label]);
        assert_eq!(t.take_modifications(), vec![]);

        t.set_bbox(RBBox::new(0.0, 0.0, 1.0, 1.0, None));
        t.clear_attributes_py();
        assert_eq!(
            t.take_modifications(),
            vec![Modification::BoundingBox, Modification::Attributes]
        );
        assert_eq!(t.take_modifications(), vec![]);
    }
}
