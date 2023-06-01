use crate::primitives::attribute::{AttributeMethods, Attributive};
use crate::primitives::message::video::frame::BelongingVideoFrame;
use crate::primitives::message::video::object::vector::VectorView;
use crate::primitives::to_json_value::ToSerdeJsonValue;
use crate::primitives::{Attribute, RBBox, VideoFrame};
use crate::utils::python::no_gil;
use crate::utils::symbol_mapper::get_object_id;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use pyo3::{pyclass, pymethods, Py, PyAny};
use rkyv::{with::Skip, Archive, Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

pub mod vector;

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, Clone, derive_builder::Builder)]
#[archive(check_bytes)]
pub struct ObjectTrack {
    #[pyo3(get, set)]
    pub id: i64,
    #[pyo3(get, set)]
    pub bounding_box: RBBox,
}

impl ToSerdeJsonValue for ObjectTrack {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!({
            "track_id": self.id,
            "track_bounding_box": self.bounding_box.to_serde_json_value(),
        })
    }
}

#[pymethods]
impl ObjectTrack {
    #[new]
    pub fn new(track_id: i64, bounding_box: RBBox) -> Self {
        Self {
            id: track_id,
            bounding_box,
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
    Track,
    DrawLabel,
}

impl ToSerdeJsonValue for Modification {
    fn to_serde_json_value(&self) -> serde_json::Value {
        serde_json::json!(format!("{:?}", self))
    }
}

#[derive(Archive, Deserialize, Serialize, Debug, Clone, derive_builder::Builder)]
#[archive(check_bytes)]
pub struct InnerObject {
    pub id: i64,
    pub creator: String,
    pub label: String,
    #[builder(default)]
    pub draw_label: Option<String>,
    pub bbox: RBBox,
    #[builder(default)]
    pub attributes: HashMap<(String, String), Attribute>,
    #[builder(default)]
    pub confidence: Option<f64>,
    #[builder(default)]
    pub(crate) parent_id: Option<i64>,
    #[builder(default)]
    pub track: Option<ObjectTrack>,
    #[with(Skip)]
    #[builder(default)]
    pub modifications: Vec<Modification>,
    #[with(Skip)]
    #[builder(default)]
    pub creator_id: Option<i64>,
    #[with(Skip)]
    #[builder(default)]
    pub label_id: Option<i64>,
    #[with(Skip)]
    #[builder(default)]
    pub(crate) frame: Option<BelongingVideoFrame>,
}

impl Default for InnerObject {
    fn default() -> Self {
        Self {
            id: 0,
            creator: "".to_string(),
            label: "".to_string(),
            draw_label: None,
            bbox: RBBox::default(),
            attributes: HashMap::new(),
            confidence: None,
            parent_id: None,
            track: None,
            modifications: Vec::new(),
            creator_id: None,
            label_id: None,
            frame: None,
        }
    }
}

impl ToSerdeJsonValue for InnerObject {
    fn to_serde_json_value(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id,
            "creator": self.creator,
            "label": self.label,
            "draw_label": self.draw_label,
            "bbox": self.bbox.to_serde_json_value(),
            "attributes": self.attributes.values().map(|v| v.to_serde_json_value()).collect::<Vec<_>>(),
            "confidence": self.confidence,
            "parent": self.parent_id,
            "track": self.track.as_ref().map(|t| t.to_serde_json_value()),
            "modifications": self.modifications.iter().map(|m| m.to_serde_json_value()).collect::<Vec<serde_json::Value>>(),
            "frame": self.get_parent_frame_source(),
        })
    }
}

impl InnerObject {
    pub fn get_parent_frame_source(&self) -> Option<String> {
        self.frame.as_ref().and_then(|f| {
            f.inner
                .upgrade()
                .map(|f| f.read_recursive().source_id.clone())
        })
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Object {
    pub(crate) inner: Arc<RwLock<InnerObject>>,
}

impl ToSerdeJsonValue for Object {
    fn to_serde_json_value(&self) -> serde_json::Value {
        self.inner.read_recursive().to_serde_json_value()
    }
}

impl Attributive for InnerObject {
    fn get_attributes_ref(&self) -> &HashMap<(String, String), Attribute> {
        &self.attributes
    }

    fn get_attributes_ref_mut(&mut self) -> &mut HashMap<(String, String), Attribute> {
        &mut self.attributes
    }

    fn take_attributes(&mut self) -> HashMap<(String, String), Attribute> {
        std::mem::take(&mut self.attributes)
    }

    fn place_attributes(&mut self, attributes: HashMap<(String, String), Attribute>) {
        self.attributes = attributes;
    }
}

impl AttributeMethods for Object {
    fn exclude_temporary_attributes(&self) -> Vec<Attribute> {
        let mut inner = self.inner.write();
        inner.exclude_temporary_attributes()
    }

    fn restore_attributes(&self, attributes: Vec<Attribute>) {
        let mut inner = self.inner.write();
        inner.restore_attributes(attributes)
    }

    fn get_attributes(&self) -> Vec<(String, String)> {
        let inner = self.inner.read_recursive();
        inner.get_attributes()
    }

    fn get_attribute(&self, creator: String, name: String) -> Option<Attribute> {
        let inner = self.inner.read_recursive();
        inner.get_attribute(creator, name)
    }

    fn delete_attribute(&self, creator: String, name: String) -> Option<Attribute> {
        let mut inner = self.inner.write();
        inner.delete_attribute(creator, name)
    }

    fn set_attribute(&self, attribute: Attribute) -> Option<Attribute> {
        let mut inner = self.inner.write();
        inner.set_attribute(attribute)
    }

    fn clear_attributes(&self) {
        let mut inner = self.inner.write();
        inner.clear_attributes()
    }

    fn delete_attributes(&self, negated: bool, creator: Option<String>, names: Vec<String>) {
        let mut inner = self.inner.write();
        inner.delete_attributes(negated, creator, names)
    }

    fn find_attributes(
        &self,
        creator: Option<String>,
        name: Option<String>,
        hint: Option<String>,
    ) -> Vec<(String, String)> {
        let inner = self.inner.read_recursive();
        inner.find_attributes(creator, name, hint)
    }
}

impl Object {
    pub fn get_parent_id(&self) -> Option<i64> {
        let inner = self.inner.read_recursive();
        inner.parent_id
    }

    pub fn get_inner(&self) -> Arc<RwLock<InnerObject>> {
        self.inner.clone()
    }

    pub fn from_inner_object(object: InnerObject) -> Self {
        Self {
            inner: Arc::new(RwLock::new(object)),
        }
    }

    pub fn from_arced_inner_object(object: Arc<RwLock<InnerObject>>) -> Self {
        Self { inner: object }
    }

    pub fn get_inner_read(&self) -> RwLockReadGuard<InnerObject> {
        let inner = self.inner.read_recursive();
        inner
    }

    pub fn get_inner_write(&self) -> RwLockWriteGuard<InnerObject> {
        let inner = self.inner.write();
        inner
    }

    pub(crate) fn set_parent(&self, parent_opt: Option<i64>) {
        if let Some(parent) = parent_opt {
            if self.get_frame().is_none() {
                panic!("Cannot set parent to the object detached from a frame");
            }
            if self.get_id() == parent {
                panic!("Cannot set parent to itself");
            }
            let f = self.get_frame().unwrap();
            if !f.object_exists(parent) {
                panic!("Cannot set parent to the object which cannot be found in the frame");
            }
        }

        let mut inner = self.inner.write();
        inner.parent_id = parent_opt;
        inner.modifications.push(Modification::Parent);
    }

    pub fn get_parent(&self) -> Option<Object> {
        let frame = self.get_frame();
        let id = self.inner.read_recursive().parent_id?;
        match frame {
            Some(f) => f.get_object(id),
            None => None,
        }
    }

    pub fn get_children(&self) -> Vec<Object> {
        let frame = self.get_frame();
        let id = self.get_id();
        match frame {
            Some(f) => f.get_children(id),
            None => Vec::new(),
        }
    }

    pub(crate) fn attach_to_video_frame(&self, frame: VideoFrame) {
        let mut inner = self.inner.write();
        inner.frame = Some(frame.into());
    }
}

#[pymethods]
impl Object {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:#?}", self.inner.read_recursive())
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
        track: Option<ObjectTrack>,
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
            track,
            creator_id,
            label_id,
            ..Default::default()
        };
        Self {
            inner: Arc::new(RwLock::new(object)),
        }
    }

    pub fn is_spoiled(&self) -> bool {
        let inner = self.inner.read_recursive();
        match inner.frame {
            Some(ref f) => f.inner.upgrade().is_none(),
            None => false,
        }
    }

    pub fn is_detached(&self) -> bool {
        let inner = self.inner.read_recursive();
        inner.frame.is_none()
    }

    pub fn detached_copy(&self) -> Self {
        let inner = self.inner.read_recursive();
        let mut new_inner = inner.clone();
        new_inner.parent_id = None;
        new_inner.frame = None;
        Self {
            inner: Arc::new(RwLock::new(new_inner)),
        }
    }

    #[getter]
    #[pyo3(name = "get_children")]
    pub fn get_children_gil(&self) -> VectorView {
        self.get_children().into()
    }

    #[getter]
    pub fn get_track(&self) -> Option<ObjectTrack> {
        let inner = self.inner.read_recursive();
        inner.track.clone()
    }

    pub fn get_frame(&self) -> Option<VideoFrame> {
        let inner = self.inner.read_recursive();
        inner.frame.as_ref().map(|f| f.into())
    }

    #[getter]
    pub fn get_id(&self) -> i64 {
        self.inner.read_recursive().id
    }

    #[getter]
    pub fn get_creator(&self) -> String {
        self.inner.read_recursive().creator.clone()
    }

    #[getter]
    pub fn get_label(&self) -> String {
        self.inner.read_recursive().label.clone()
    }

    #[getter]
    pub fn get_bbox(&self) -> crate::primitives::RBBox {
        self.inner.read_recursive().bbox.clone()
    }

    #[getter]
    pub fn get_confidence(&self) -> Option<f64> {
        let inner = self.inner.read_recursive();
        inner.confidence
    }

    #[getter]
    pub fn draw_label(&self) -> String {
        let inner = self.inner.read_recursive();
        inner.draw_label.as_ref().unwrap_or(&inner.label).clone()
    }

    #[setter]
    pub fn set_draw_label(&self, draw_label: Option<String>) {
        let mut inner = self.inner.write();
        inner.draw_label = draw_label;
        inner.modifications.push(Modification::DrawLabel);
    }

    #[setter]
    pub fn set_track(&self, track: Option<ObjectTrack>) {
        let mut inner = self.inner.write();
        inner.track = track;
        inner.modifications.push(Modification::Track);
    }

    #[setter]
    pub fn set_id(&self, id: i64) {
        assert!(
            !matches!(self.get_frame(), Some(_)),
            "When object is attached to a frame, it is impossible to change its ID"
        );

        let mut inner = self.inner.write();
        inner.id = id;
        inner.modifications.push(Modification::Id);
    }

    #[setter]
    pub fn set_creator(&self, creator: String) {
        let mut inner = self.inner.write();
        inner.creator = creator;
        inner.modifications.push(Modification::Creator);
    }

    #[setter]
    pub fn set_label(&self, label: String) {
        let mut inner = self.inner.write();
        inner.label = label;
        inner.modifications.push(Modification::Label);
    }

    #[setter]
    pub fn set_bbox(&self, bbox: RBBox) {
        let mut inner = self.inner.write();
        inner.bbox = bbox;
        inner.modifications.push(Modification::BoundingBox);
    }

    #[setter]
    pub fn set_confidence(&self, confidence: Option<f64>) {
        let mut inner = self.inner.write();
        inner.confidence = confidence;
        inner.modifications.push(Modification::Confidence);
    }

    #[getter]
    #[pyo3(name = "attributes")]
    pub fn get_attributes_gil(&self) -> Vec<(String, String)> {
        no_gil(|| self.get_attributes())
    }

    #[pyo3(name = "get_attribute")]
    pub fn get_attribute_gil(&self, creator: String, name: String) -> Option<Attribute> {
        self.get_attribute(creator, name)
    }

    #[pyo3(name = "delete_attribute")]
    pub fn delete_attribute_gil(&mut self, creator: String, name: String) -> Option<Attribute> {
        match self.delete_attribute(creator, name) {
            Some(attribute) => {
                let mut object = self.inner.write();
                object.modifications.push(Modification::Attributes);
                Some(attribute)
            }
            None => None,
        }
    }

    #[pyo3(name = "set_attribute")]
    pub fn set_attribute_gil(&mut self, attribute: Attribute) -> Option<Attribute> {
        {
            let mut object = self.inner.write();
            object.modifications.push(Modification::Attributes);
        }
        self.set_attribute(attribute)
    }

    #[pyo3(name = "clear_attributes")]
    pub fn clear_attributes_gil(&mut self) {
        {
            let mut object = self.inner.write();
            object.modifications.push(Modification::Attributes);
        }
        self.clear_attributes()
    }

    #[pyo3(signature = (negated=false, creator=None, names=vec![]))]
    #[pyo3(name = "delete_attributes")]
    pub fn delete_attributes_gil(
        &mut self,
        negated: bool,
        creator: Option<String>,
        names: Vec<String>,
    ) {
        no_gil(move || {
            {
                let mut object = self.inner.write();
                object.modifications.push(Modification::Attributes);
            }
            self.delete_attributes(negated, creator, names)
        })
    }

    #[pyo3(name = "find_attributes")]
    pub fn find_attributes_gil(
        &self,
        creator: Option<String>,
        name: Option<String>,
        hint: Option<String>,
    ) -> Vec<(String, String)> {
        no_gil(|| self.find_attributes(creator, name, hint))
    }

    pub fn take_modifications(&self) -> Vec<Modification> {
        let mut object = self.inner.write();
        std::mem::take(&mut object.modifications)
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::attribute::AttributeMethods;
    use crate::primitives::message::video::object::InnerObjectBuilder;
    use crate::primitives::{AttributeBuilder, Modification, Object, RBBox, Value};
    use crate::test::utils::gen_frame;

    fn get_object() -> Object {
        Object::from_inner_object(
            InnerObjectBuilder::default()
                .id(1)
                .track(None)
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
                .parent_id(None)
                .build()
                .unwrap(),
        )
    }

    #[test]
    fn test_delete_attributes() {
        pyo3::prepare_freethreaded_python();

        let obj = get_object();
        obj.delete_attributes(false, None, vec![]);
        assert_eq!(obj.get_inner_read().attributes.len(), 3);

        let obj = get_object();
        obj.delete_attributes(true, None, vec![]);
        assert!(obj.get_inner_read().attributes.is_empty());

        let obj = get_object();
        obj.delete_attributes(false, Some("creator".to_string()), vec![]);
        assert_eq!(obj.get_inner_read().attributes.len(), 1);

        let obj = get_object();
        obj.delete_attributes(true, Some("creator".to_string()), vec![]);
        assert_eq!(obj.get_inner_read().attributes.len(), 2);

        let obj = get_object();
        obj.delete_attributes(false, None, vec!["name".to_string()]);
        assert_eq!(obj.get_inner_read().attributes.len(), 1);

        let obj = get_object();
        obj.delete_attributes(true, None, vec!["name".to_string()]);
        assert_eq!(obj.get_inner_read().attributes.len(), 2);

        let t = get_object();
        t.delete_attributes(false, None, vec!["name".to_string(), "name2".to_string()]);
        assert_eq!(t.get_inner_read().attributes.len(), 0);

        let obj = get_object();
        obj.delete_attributes(true, None, vec!["name".to_string(), "name2".to_string()]);
        assert_eq!(obj.get_inner_read().attributes.len(), 3);

        let obj = get_object();
        obj.delete_attributes(
            false,
            Some("creator".to_string()),
            vec!["name".to_string(), "name2".to_string()],
        );
        assert_eq!(obj.get_inner_read().attributes.len(), 1);

        assert_eq!(
            &obj.get_inner_read().attributes[&("creator2".to_string(), "name".to_string())],
            &AttributeBuilder::default()
                .creator("creator2".to_string())
                .name("name".to_string())
                .values(vec![Value::string("value".to_string(), None)])
                .hint(None)
                .build()
                .unwrap()
        );

        let obj = get_object();
        obj.delete_attributes(
            true,
            Some("creator".to_string()),
            vec!["name".to_string(), "name2".to_string()],
        );
        assert_eq!(obj.get_inner_read().attributes.len(), 2);
    }

    #[test]
    fn test_modifications() {
        let mut t = get_object();
        t.set_label("label2".to_string());
        assert_eq!(t.take_modifications(), vec![Modification::Label]);
        assert_eq!(t.take_modifications(), vec![]);

        t.set_bbox(RBBox::new(0.0, 0.0, 1.0, 1.0, None));
        t.clear_attributes_gil();
        assert_eq!(
            t.take_modifications(),
            vec![Modification::BoundingBox, Modification::Attributes]
        );
        assert_eq!(t.take_modifications(), vec![]);
    }

    #[test]
    #[should_panic]
    fn self_parent_assignment_panic_trivial() {
        let obj = get_object();
        obj.set_parent(Some(obj.get_id()));
    }

    #[test]
    #[should_panic]
    fn self_parent_assignment_change_id() {
        let obj = get_object();
        let parent = obj.clone();
        parent.set_id(2);
        obj.set_parent(Some(parent.get_id()));
    }

    #[test]
    #[should_panic(expected = "Frame is dropped, you cannot use attached objects anymore")]
    fn try_access_frame_object_after_frame_drop() {
        let f = gen_frame();
        let o = f.get_object(0).unwrap();
        drop(f);
        let _f = o.get_frame().unwrap();
    }

    #[test]
    #[should_panic(expected = "Only detached objects can be attached to a frame.")]
    fn reassign_object_from_dropped_frame_to_new_frame() {
        let f = gen_frame();
        let o = f.get_object(0).unwrap();
        drop(f);
        let f = gen_frame();
        f.delete_objects_by_ids(&[0]);
        f.add_object(&o);
    }

    #[test]
    fn reassign_clean_copy_from_dropped_to_new_frame() {
        let f = gen_frame();
        let o = f.get_object(0).unwrap();
        drop(f);
        let f = gen_frame();
        f.delete_objects_by_ids(&[0]);
        let copy = o.detached_copy();
        assert!(copy.is_detached(), "Clean copy is not attached");
        assert!(!copy.is_spoiled(), "Clean copy must be not spoiled");
        assert!(
            copy.get_parent().is_none(),
            "Clean copy must have no parent"
        );
        f.add_object(&o.detached_copy());
    }
}
