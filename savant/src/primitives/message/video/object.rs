use crate::primitives::attribute::{AttributeMethods, Attributive};
use crate::primitives::message::video::frame::BelongingVideoFrame;
use crate::primitives::message::video::object::objects_view::VideoObjectsView;
use crate::primitives::proxy::video_object_rbbox::VideoObjectRBBoxProxy;
use crate::primitives::proxy::video_object_tracking_data::VideoObjectTrackingDataProxy;
use crate::primitives::to_json_value::ToSerdeJsonValue;
use crate::primitives::{Attribute, RBBox, VideoFrameProxy, VideoObjectBBoxKind};
use crate::utils::python::no_gil;
use crate::utils::symbol_mapper::get_object_id;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use pyo3::exceptions::PyRuntimeError;
use pyo3::{pyclass, pymethods, Py, PyAny, PyResult};
use rkyv::{with::Skip, Archive, Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

pub mod objects_view;

/// Represents tracking data for a single object filled by a tracker.
/// This is a readonly object, you cannot change fields inplace. If you need to change tracking data for
/// an object, you need to create a new instance and fill it. However, if you have requested the access with
/// :py:attr:`VideoObject.tracking_data_ref`, you can change the fields of the returned object inplace (if it is defined).
///
/// The property :py:attr:`VideoObject.tracking_data` operates by value. If you change the fields of the returned object,
/// the changes will not be applied to the original object.
///
#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, Clone, derive_builder::Builder)]
#[archive(check_bytes)]
pub struct VideoObjectTrackingData {
    #[pyo3(get)]
    pub id: i64,
    #[pyo3(get)]
    pub bounding_box: RBBox,
}

impl ToSerdeJsonValue for VideoObjectTrackingData {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!({
            "track_id": self.id,
            "track_bounding_box": self.bounding_box.to_serde_json_value(),
        })
    }
}

#[pymethods]
impl VideoObjectTrackingData {
    #[new]
    pub fn new(track_id: i64, bounding_box: RBBox) -> Self {
        Self {
            id: track_id,
            bounding_box,
        }
    }
}

/// Represents operations happened with a video object. The operations are used to determine which fields of the object
/// should be updated in the backing stores. You don't need to track modifications manually, they are tracked automatically.
///
#[pyclass]
#[derive(Debug, Clone, PartialEq)]
pub enum VideoObjectModification {
    Id,
    Creator,
    Label,
    BoundingBox,
    Attributes,
    Confidence,
    Parent,
    TrackInfo,
    DrawLabel,
}

impl ToSerdeJsonValue for VideoObjectModification {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!(format!("{:?}", self))
    }
}

#[derive(Archive, Deserialize, Serialize, Debug, Clone, derive_builder::Builder)]
#[archive(check_bytes)]
pub struct VideoObject {
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
    pub track_info: Option<VideoObjectTrackingData>,
    #[with(Skip)]
    #[builder(default)]
    pub modifications: Vec<VideoObjectModification>,
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

impl Default for VideoObject {
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
            track_info: None,
            modifications: Vec::new(),
            creator_id: None,
            label_id: None,
            frame: None,
        }
    }
}

impl ToSerdeJsonValue for VideoObject {
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
            "track": self.track_info.as_ref().map(|t| t.to_serde_json_value()),
            "modifications": self.modifications.iter().map(|m| m.to_serde_json_value()).collect::<Vec<serde_json::Value>>(),
            "frame": self.get_parent_frame_source(),
        })
    }
}

impl VideoObject {
    pub fn get_parent_frame_source(&self) -> Option<String> {
        self.frame.as_ref().and_then(|f| {
            f.inner
                .upgrade()
                .map(|f| f.read_recursive().source_id.clone())
        })
    }

    pub(crate) fn bbox_ref(&self, kind: VideoObjectBBoxKind) -> &RBBox {
        match kind {
            VideoObjectBBoxKind::Detection => &self.bbox,
            VideoObjectBBoxKind::TrackingInfo => self
                .track_info
                .as_ref()
                .map(|t| &t.bounding_box)
                .unwrap_or(&self.bbox),
        }
    }

    pub(crate) fn bbox_mut(&mut self, kind: VideoObjectBBoxKind) -> &mut RBBox {
        match kind {
            VideoObjectBBoxKind::Detection => &mut self.bbox,
            VideoObjectBBoxKind::TrackingInfo => self
                .track_info
                .as_mut()
                .map(|t| &mut t.bounding_box)
                .unwrap_or(&mut self.bbox),
        }
    }

    pub(crate) fn add_modification(&mut self, modification: VideoObjectModification) {
        self.modifications.push(modification);
    }
}

/// Represents a video object. The object is a part of a video frame, it includes bounding
/// box, attributes, label, creator label, etc. The objects are always accessible by reference. The only way to
/// copy the object by value is to call :py:meth:`VideoObject.detached_copy`.
///
/// :py:class:`VideoObject` is a part of :py:class:`VideoFrame` and may outlive it if there are references.
///
#[pyclass]
#[derive(Debug, Clone)]
#[pyo3(name = "VideoObject")]
pub struct VideoObjectProxy {
    pub(crate) inner: Arc<RwLock<VideoObject>>,
}

impl ToSerdeJsonValue for VideoObjectProxy {
    fn to_serde_json_value(&self) -> serde_json::Value {
        self.inner.read_recursive().to_serde_json_value()
    }
}

impl Attributive for VideoObject {
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

impl AttributeMethods for VideoObjectProxy {
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

    fn delete_attributes(&self, creator: Option<String>, names: Vec<String>) {
        let mut inner = self.inner.write();
        inner.delete_attributes(creator, names)
    }

    fn find_attributes(
        &self,
        creator: Option<String>,
        names: Vec<String>,
        hint: Option<String>,
    ) -> Vec<(String, String)> {
        let inner = self.inner.read_recursive();
        inner.find_attributes(creator, names, hint)
    }
}

impl VideoObjectProxy {
    pub fn update_track_bbox(&self, bbox: RBBox) {
        let mut inner = self.inner.write();
        if let Some(t) = inner.track_info.as_mut() {
            t.bounding_box = bbox;
        }
    }

    pub fn get_parent_id(&self) -> Option<i64> {
        let inner = self.inner.read_recursive();
        inner.parent_id
    }

    pub fn get_inner(&self) -> Arc<RwLock<VideoObject>> {
        self.inner.clone()
    }

    pub fn from_video_object(object: VideoObject) -> Self {
        Self {
            inner: Arc::new(RwLock::new(object)),
        }
    }

    pub fn from_arced_inner_object(object: Arc<RwLock<VideoObject>>) -> Self {
        Self { inner: object }
    }

    pub fn get_inner_read(&self) -> RwLockReadGuard<VideoObject> {
        let inner = self.inner.read_recursive();
        inner
    }

    pub fn get_inner_write(&self) -> RwLockWriteGuard<VideoObject> {
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
        inner.add_modification(VideoObjectModification::Parent);
    }

    pub fn get_parent(&self) -> Option<VideoObjectProxy> {
        let frame = self.get_frame();
        let id = self.inner.read_recursive().parent_id?;
        match frame {
            Some(f) => f.get_object(id),
            None => None,
        }
    }

    pub fn get_children(&self) -> Vec<VideoObjectProxy> {
        let frame = self.get_frame();
        let id = self.get_id();
        match frame {
            Some(f) => f.get_children(id),
            None => Vec::new(),
        }
    }

    pub(crate) fn attach_to_video_frame(&self, frame: VideoFrameProxy) {
        let mut inner = self.inner.write();
        inner.frame = Some(frame.into());
    }
}

#[pymethods]
impl VideoObjectProxy {
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
        track: Option<VideoObjectTrackingData>,
    ) -> Self {
        let (creator_id, label_id) =
            get_object_id(&creator, &label).map_or((None, None), |(c, o)| (Some(c), Some(o)));

        let object = VideoObject {
            id,
            creator,
            label,
            bbox,
            attributes,
            confidence,
            track_info: track,
            creator_id,
            label_id,
            ..Default::default()
        };
        Self {
            inner: Arc::new(RwLock::new(object)),
        }
    }

    /// Returns object's attributes as a list of tuples ``(creator, name)``.
    ///
    /// Returns
    /// -------
    /// List[Tuple[str, str]]
    ///   List of attribute identifiers as ``(creator, name)``.
    ///
    #[getter]
    #[pyo3(name = "attributes")]
    pub fn get_attributes_gil(&self) -> Vec<(String, String)> {
        no_gil(|| self.get_attributes())
    }

    /// Returns object's bbox by value. Any modifications of the returned value will not affect the object.
    /// When used as setter, allows setting object's bbox by value.
    ///
    /// Returns
    /// -------
    /// :py:class:`savant_rs.primitives.geometry.RBBox`
    ///   Object's bounding box.
    ///
    #[getter]
    pub fn get_bbox(&self) -> RBBox {
        self.inner.read_recursive().bbox.clone()
    }

    /// Accesses object's bbox by reference. The object returned by the method is a special proxy object.
    ///
    /// Returns
    /// -------
    /// :py:class:`VideoObjectRBBoxProxy`
    ///   A proxy object for the object's bbox.
    ///
    #[getter]
    pub fn bbox_ref(&self) -> VideoObjectRBBoxProxy {
        VideoObjectRBBoxProxy::new(self.inner.clone(), VideoObjectBBoxKind::Detection)
    }

    /// Accesses object's children. If the object is detached from a frame, an empty view is returned.
    ///
    /// Returns
    /// -------
    /// :py:class:`VideoObjectsView`
    ///   A view of the object's children.
    ///
    #[getter]
    #[pyo3(name = "children_ref")]
    pub fn children_ref_gil(&self) -> VideoObjectsView {
        self.get_children().into()
    }

    /// Clears all object's attributes.
    ///
    #[pyo3(name = "clear_attributes")]
    pub fn clear_attributes_gil(&mut self) {
        {
            let mut object = self.inner.write();
            object.add_modification(VideoObjectModification::Attributes);
        }
        self.clear_attributes()
    }

    /// Returns object confidence if set. When used as setter, allows setting object's confidence.
    ///
    /// Returns
    /// -------
    /// float or None
    ///   Object's confidence.
    ///
    #[getter]
    pub fn get_confidence(&self) -> Option<f64> {
        let inner = self.inner.read_recursive();
        inner.confidence
    }

    /// Returns object's creator. When used as setter, allows setting object's creator.
    ///
    /// Returns
    /// -------
    /// str
    ///   Object's creator.
    ///
    #[getter]
    pub fn get_creator(&self) -> String {
        self.inner.read_recursive().creator.clone()
    }

    /// Deletes an attribute from the object.
    /// If the attribute is not found, returns None, otherwise returns the deleted attribute.
    ///
    /// Parameters
    /// ----------
    /// creator : str
    ///   Attribute creator.
    /// name : str
    ///   Attribute name.
    ///
    /// Returns
    /// -------
    /// :py:class:`Attribute` or None
    ///   Deleted attribute or None if the attribute is not found.
    ///
    #[pyo3(name = "delete_attribute")]
    pub fn delete_attribute_gil(&mut self, creator: String, name: String) -> Option<Attribute> {
        match self.delete_attribute(creator, name) {
            Some(attribute) => {
                let mut object = self.inner.write();
                object.add_modification(VideoObjectModification::Attributes);
                Some(attribute)
            }
            None => None,
        }
    }

    /// Deletes attributes from the object.
    ///
    /// Parameters
    /// ----------
    /// creator : str or None
    ///   Attribute creator. If None, it is ignored when candidates are selected for removal.
    /// names : List[str]
    ///   Attribute names. If empty, it is ignored when candidates are selected for removal.
    ///
    #[pyo3(signature = (creator=None, names=vec![]))]
    #[pyo3(name = "delete_attributes")]
    pub fn delete_attributes_gil(&mut self, creator: Option<String>, names: Vec<String>) {
        no_gil(move || {
            {
                let mut object = self.inner.write();
                object.add_modification(VideoObjectModification::Attributes);
            }
            self.delete_attributes(creator, names)
        })
    }

    /// Returns a copy of the object with the same properties but detached from the frame and without a parent set.
    ///
    /// Returns
    /// -------
    /// :py:class:`VideoObject`
    ///   A copy of the object.
    ///
    pub fn detached_copy(&self) -> Self {
        let inner = self.inner.read_recursive();
        let mut new_inner = inner.clone();
        new_inner.parent_id = None;
        new_inner.frame = None;
        Self {
            inner: Arc::new(RwLock::new(new_inner)),
        }
    }

    /// Returns object's draw label if set. When used as setter, allows setting object's draw label.
    /// If the draw label is not set, returns object's label.
    ///
    /// Returns
    /// -------
    /// str
    ///   Object's draw label.
    ///
    #[getter]
    pub fn get_draw_label(&self) -> String {
        let inner = self.inner.read_recursive();
        inner.draw_label.as_ref().unwrap_or(&inner.label).clone()
    }

    /// finds and returns names of attributes by expression based on creator, names and hint.
    ///
    /// Parameters
    /// ----------
    /// creator : str or None
    ///   Attribute creator. If None, it is ignored when candidates are selected.
    /// names : List[str]
    ///   Attribute names. If empty, it is ignored when candidates are selected.
    /// hint : str or None
    ///   Hint for the attribute name. If None, it is ignored when candidates are selected.
    ///
    /// Returns
    /// -------
    /// List[Tuple[str, str]]
    ///   List of tuples with attribute creators and names.
    ///
    #[pyo3(name = "find_attributes")]
    #[pyo3(signature = (creator=None, names=vec![], hint=None))]
    pub fn find_attributes_gil(
        &self,
        creator: Option<String>,
        names: Vec<String>,
        hint: Option<String>,
    ) -> Vec<(String, String)> {
        no_gil(|| self.find_attributes(creator, names, hint))
    }

    /// Fetches attribute by creator and name. The attribute is fetched by value, not reference, however attribute's values are fetched as CoW,
    /// until the modification the values are shared. If the attribute is not found, returns None.
    ///
    /// Remember, because the attribute is fetched as a copy,
    /// that changing attribute properties will not change the attribute kept in the object.
    ///
    /// Parameters
    /// ----------
    /// creator : str
    ///   Attribute creator.
    /// name : str
    ///   Attribute name.
    ///
    /// Returns
    /// -------
    /// :py:class:`Attribute` or None
    ///   Attribute or None if the attribute is not found.
    ///
    #[pyo3(name = "get_attribute")]
    pub fn get_attribute_gil(&self, creator: String, name: String) -> Option<Attribute> {
        self.get_attribute(creator, name)
    }

    /// Returns the :py:class:`VideoFrame` reference to a frame the object belongs to.
    ///
    /// Returns
    /// -------
    /// :py:class:`VideoFrame` or None
    ///   A reference to a frame the object belongs to.
    ///
    pub fn get_frame(&self) -> Option<VideoFrameProxy> {
        let inner = self.inner.read_recursive();
        inner.frame.as_ref().map(|f| f.into())
    }

    /// Returns the object's id. The setter causes ``RuntimeError`` when the object is attached to a frame.
    ///
    /// Returns
    /// -------
    /// int
    ///   Object's id.
    ///
    #[getter]
    pub fn get_id(&self) -> i64 {
        self.inner.read_recursive().id
    }

    /// The object is detached if it is not attached to any frame. Such state may cause it impossible to operate with certain object properties.
    ///
    /// Returns
    /// -------
    /// bool
    ///   True if the object is detached, False otherwise.
    ///
    pub fn is_detached(&self) -> bool {
        let inner = self.inner.read_recursive();
        inner.frame.is_none()
    }

    /// The object is spoiled if it is outlived the a belonging frame. Such state may cause it impossible to operate with certain object properties.
    ///
    /// Returns
    /// -------
    /// bool
    ///   True if the object is spoiled, False otherwise.
    pub fn is_spoiled(&self) -> bool {
        let inner = self.inner.read_recursive();
        match inner.frame {
            Some(ref f) => f.inner.upgrade().is_none(),
            None => false,
        }
    }

    /// Returns the object's label. The setter allows setting object's label.
    ///
    /// Returns
    /// -------
    /// str
    ///   Object's label.
    ///
    #[getter]
    pub fn get_label(&self) -> String {
        self.inner.read_recursive().label.clone()
    }

    /// Sets the attribute for the object. If the attribute is already set, it is replaced.
    ///
    /// Parameters
    /// ----------
    /// attribute : :py:class:`Attribute`
    ///   Attribute to set.
    ///
    /// Returns
    /// -------
    /// :py:class:`Attribute` or None
    ///   Attribute that was replaced or None if the attribute was not set.
    ///
    #[pyo3(name = "set_attribute")]
    pub fn set_attribute_gil(&mut self, attribute: &Attribute) -> Option<Attribute> {
        {
            let mut object = self.inner.write();
            object.add_modification(VideoObjectModification::Attributes);
        }
        self.set_attribute(attribute.clone())
    }

    /// Fetches object modifications. The modifications are fetched by value, not reference. The modifications are cleared after the fetch.
    ///
    /// Returns
    /// -------
    /// List[:py:class:`VideoObjectModification`]
    ///   List of object modifications.
    ///
    pub fn take_modifications(&self) -> Vec<VideoObjectModification> {
        let mut object = self.inner.write();
        std::mem::take(&mut object.modifications)
    }

    /// Returns object tracking data if it is available. The data returned by value, not reference.
    ///
    /// Returns
    /// -------
    /// :py:class:`VideoObjectTrackingData` or None
    ///   Object tracking data.
    ///
    #[getter]
    pub fn get_tracking_data(&self) -> Option<VideoObjectTrackingData> {
        let inner = self.inner.read_recursive();
        inner.track_info.clone()
    }

    #[getter]
    pub fn tracking_data_ref(&self) -> VideoObjectTrackingDataProxy {
        VideoObjectTrackingDataProxy::new(self.inner.clone())
    }

    #[setter]
    pub fn set_bbox(&self, bbox: RBBox) {
        let mut inner = self.inner.write();
        inner.bbox = bbox;
        inner.add_modification(VideoObjectModification::BoundingBox);
    }

    #[setter]
    pub fn set_draw_label(&self, draw_label: Option<String>) {
        let mut inner = self.inner.write();
        inner.draw_label = draw_label;
        inner.add_modification(VideoObjectModification::DrawLabel);
    }

    #[setter]
    pub fn set_tracking_data(&self, track: Option<VideoObjectTrackingData>) {
        let mut inner = self.inner.write();
        inner.track_info = track;
        inner.add_modification(VideoObjectModification::TrackInfo);
    }

    #[setter]
    pub fn set_id(&self, id: i64) -> PyResult<()> {
        if matches!(self.get_frame(), Some(_)) {
            return Err(PyRuntimeError::new_err(
                "When object is attached to a frame, it is impossible to change its ID",
            ));
        }

        let mut inner = self.inner.write();
        inner.id = id;
        inner.add_modification(VideoObjectModification::Id);
        Ok(())
    }

    #[setter]
    pub fn set_creator(&self, creator: String) {
        let mut inner = self.inner.write();
        inner.creator = creator;
        inner.add_modification(VideoObjectModification::Creator);
    }

    #[setter]
    pub fn set_label(&self, label: String) {
        let mut inner = self.inner.write();
        inner.label = label;
        inner.add_modification(VideoObjectModification::Label);
    }

    #[setter]
    pub fn set_confidence(&self, confidence: Option<f64>) {
        let mut inner = self.inner.write();
        inner.confidence = confidence;
        inner.add_modification(VideoObjectModification::Confidence);
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::attribute::attribute_value::AttributeValue;
    use crate::primitives::attribute::AttributeMethods;
    use crate::primitives::message::video::object::VideoObjectBuilder;
    use crate::primitives::{AttributeBuilder, RBBox, VideoObjectModification, VideoObjectProxy};
    use crate::test::utils::{gen_frame, s};

    fn get_object() -> VideoObjectProxy {
        VideoObjectProxy::from_video_object(
            VideoObjectBuilder::default()
                .id(1)
                .track_info(None)
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
                            .values(vec![AttributeValue::string("value".to_string(), None)])
                            .hint(None)
                            .build()
                            .unwrap(),
                        AttributeBuilder::default()
                            .creator("creator".to_string())
                            .name("name2".to_string())
                            .values(vec![AttributeValue::string("value2".to_string(), None)])
                            .hint(None)
                            .build()
                            .unwrap(),
                        AttributeBuilder::default()
                            .creator("creator2".to_string())
                            .name("name".to_string())
                            .values(vec![AttributeValue::string("value".to_string(), None)])
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
        obj.delete_attributes(None, vec![]);
        assert_eq!(obj.get_inner_read().attributes.len(), 0);

        let obj = get_object();
        obj.delete_attributes(Some(s("creator")), vec![]);
        assert_eq!(obj.get_attributes().len(), 1);

        let obj = get_object();
        obj.delete_attributes(None, vec![s("name")]);
        assert_eq!(obj.get_inner_read().attributes.len(), 1);

        let t = get_object();
        t.delete_attributes(None, vec![s("name"), s("name2")]);
        assert_eq!(t.get_inner_read().attributes.len(), 0);
    }

    #[test]
    fn test_modifications() {
        let mut t = get_object();
        t.set_label("label2".to_string());
        assert_eq!(t.take_modifications(), vec![VideoObjectModification::Label]);
        assert_eq!(t.take_modifications(), vec![]);

        t.set_bbox(RBBox::new(0.0, 0.0, 1.0, 1.0, None));
        t.clear_attributes_gil();
        assert_eq!(
            t.take_modifications(),
            vec![
                VideoObjectModification::BoundingBox,
                VideoObjectModification::Attributes
            ]
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
        _ = parent.set_id(2);
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
