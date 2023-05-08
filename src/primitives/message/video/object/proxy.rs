use crate::primitives::{Attribute, BBox, Object, ParentObject};
use pyo3::{pyclass, pymethods, Py, PyAny};
use std::sync::{Arc, Mutex};

#[pyclass]
#[derive(Debug, Clone)]
pub struct ProxyObject {
    object: Arc<Mutex<Object>>,
}

#[pymethods]
impl ProxyObject {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    pub fn id(&self) -> i64 {
        self.object.lock().unwrap().id
    }

    pub fn creator(&self) -> String {
        self.object.lock().unwrap().creator.clone()
    }

    pub fn label(&self) -> String {
        self.object.lock().unwrap().label.clone()
    }

    pub fn bbox(&self) -> crate::primitives::BBox {
        self.object.lock().unwrap().bbox.clone()
    }

    pub fn confidence(&self) -> Option<f64> {
        let object = self.object.lock().unwrap();
        object.confidence
    }

    pub fn parent(&self) -> Option<ParentObject> {
        let object = self.object.lock().unwrap();
        object.parent.clone()
    }

    pub fn set_id(&mut self, id: i64) {
        let mut object = self.object.lock().unwrap();
        object.id = id;
    }

    pub fn set_creator(&mut self, creator: String) {
        let mut object = self.object.lock().unwrap();
        object.creator = creator;
    }

    pub fn set_label(&mut self, label: String) {
        let mut object = self.object.lock().unwrap();
        object.label = label;
    }

    pub fn set_bbox(&mut self, bbox: BBox) {
        self.object.lock().unwrap().bbox = bbox;
    }

    pub fn set_confidence(&mut self, confidence: Option<f64>) {
        let mut object = self.object.lock().unwrap();
        object.confidence = confidence;
    }

    pub fn set_parent(&mut self, parent: Option<ParentObject>) {
        let mut object = self.object.lock().unwrap();
        object.parent = parent;
    }

    pub fn attributes(&self) -> Vec<(String, String)> {
        let object = self.object.lock().unwrap();
        object.attributes()
    }

    pub fn get_attribute(&self, creator: String, name: String) -> Option<Attribute> {
        let object = self.object.lock().unwrap();
        object.get_attribute(creator, name)
    }

    pub fn delete_attribute(&mut self, creator: String, name: String) -> Option<Attribute> {
        let mut object = self.object.lock().unwrap();
        object.delete_attribute(creator, name)
    }

    pub fn set_attribute(&mut self, attribute: Attribute) -> Option<Attribute> {
        let mut object = self.object.lock().unwrap();
        object.set_attribute(attribute)
    }

    pub fn clear_attributes(&mut self) {
        let mut object = self.object.lock().unwrap();
        object.clear_attributes();
    }

    #[pyo3(signature = (negated=false, creator=None, names=vec![]))]
    pub fn delete_attributes(
        &mut self,
        negated: bool,
        creator: Option<String>,
        names: Vec<String>,
    ) {
        let mut object = self.object.lock().unwrap();
        object.delete_attributes(negated, creator, names);
    }
}

impl ProxyObject {
    pub fn new(object: Arc<Mutex<Object>>) -> Self {
        Self { object }
    }
}
