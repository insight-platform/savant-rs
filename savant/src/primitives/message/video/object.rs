use crate::primitives::attribute::{AttributeMethods, Attributive};
use crate::primitives::bbox::transformations::{
    VideoObjectBBoxTransformation, VideoObjectBBoxTransformationProxy,
};
use crate::primitives::bbox::{OwnedRBBoxData, BBOX_UNDEFINED};
use crate::primitives::message::video::frame::BelongingVideoFrame;
use crate::primitives::message::video::object::objects_view::VideoObjectsView;
use crate::primitives::pyobject::PyObjectMeta;
use crate::primitives::{Attribute, RBBox, VideoFrameProxy};
use crate::release_gil;
use crate::utils::symbol_mapper::get_object_id;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use pyo3::exceptions::PyRuntimeError;
use pyo3::{pyclass, pymethods, Py, PyAny, PyObject, PyResult};
use rkyv::{with::Skip, Archive, Deserialize, Serialize};
use savant_core::to_json_value::ToSerdeJsonValue;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

pub mod context;
pub mod objects_view;

#[pyclass]
#[derive(Debug, Clone)]
pub enum IdCollisionResolutionPolicy {
    GenerateNewId,
    Overwrite,
    Error,
}

/// Represents operations happened with a video object. The operations are used to determine which fields of the object
/// should be updated in the backing stores. You don't need to track modifications manually, they are tracked automatically.
///
#[pyclass]
#[derive(Debug, Clone, PartialEq)]
pub enum VideoObjectModification {
    Id,
    Namespace,
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
    pub namespace: String,
    pub label: String,
    #[builder(default)]
    pub draw_label: Option<String>,
    pub detection_box: OwnedRBBoxData,
    #[builder(default)]
    pub attributes: HashMap<(String, String), Attribute>,
    #[builder(default)]
    pub confidence: Option<f32>,
    #[builder(default)]
    pub(crate) parent_id: Option<i64>,
    #[builder(default)]
    pub(crate) track_box: Option<OwnedRBBoxData>,
    #[builder(default)]
    pub track_id: Option<i64>,
    #[with(Skip)]
    #[builder(default)]
    pub modifications: Vec<VideoObjectModification>,
    #[with(Skip)]
    #[builder(default)]
    pub namespace_id: Option<i64>,
    #[with(Skip)]
    #[builder(default)]
    pub label_id: Option<i64>,
    #[with(Skip)]
    #[builder(default)]
    pub(crate) frame: Option<BelongingVideoFrame>,
    #[with(Skip)]
    #[builder(default)]
    pub(crate) pyobjects: HashMap<(String, String), PyObject>,
}

impl Default for VideoObject {
    fn default() -> Self {
        Self {
            id: 0,
            namespace: "".to_string(),
            label: "".to_string(),
            draw_label: None,
            detection_box: BBOX_UNDEFINED.clone().try_into().unwrap(),
            attributes: HashMap::new(),
            confidence: None,
            parent_id: None,
            track_id: None,
            track_box: None,
            modifications: Vec::new(),
            namespace_id: None,
            label_id: None,
            frame: None,
            pyobjects: HashMap::new(),
        }
    }
}

impl ToSerdeJsonValue for VideoObject {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!({
            "id": self.id,
            "namespace": self.namespace,
            "label": self.label,
            "draw_label": self.draw_label,
            "bbox": self.detection_box.to_serde_json_value(),
            "attributes": self.attributes.values().map(|v| v.to_serde_json_value()).collect::<Vec<_>>(),
            "confidence": self.confidence,
            "parent": self.parent_id,
            "track_id": self.track_id,
            "track_box": self.track_box.as_ref().map(|x| x.to_serde_json_value()),
            "modifications": self.modifications.iter().map(|m| m.to_serde_json_value()).collect::<Vec<serde_json::Value>>(),
            "frame": self.get_parent_frame_source(),
            "pyobjects": "not_implemented",
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

    pub(crate) fn add_modification(&mut self, modification: VideoObjectModification) {
        self.modifications.push(modification);
    }
}

/// Represents a video object. The object is a part of a video frame, it includes bounding
/// box, attributes, label, namespace label, etc. The objects are always accessible by reference. The only way to
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

impl PyObjectMeta for VideoObject {
    fn get_py_objects_ref(&self) -> &HashMap<(String, String), PyObject> {
        &self.pyobjects
    }

    fn get_py_objects_ref_mut(&mut self) -> &mut HashMap<(String, String), PyObject> {
        &mut self.pyobjects
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

    fn get_attribute(&self, namespace: String, name: String) -> Option<Attribute> {
        let inner = self.inner.read_recursive();
        inner.get_attribute(namespace, name)
    }

    fn delete_attribute(&self, namespace: String, name: String) -> Option<Attribute> {
        let mut inner = self.inner.write();
        inner.delete_attribute(namespace, name)
    }

    fn set_attribute(&self, attribute: Attribute) -> Option<Attribute> {
        let mut inner = self.inner.write();
        inner.set_attribute(attribute)
    }

    fn clear_attributes(&self) {
        let mut inner = self.inner.write();
        inner.clear_attributes()
    }

    fn delete_attributes(&self, namespace: Option<String>, names: Vec<String>) {
        let mut inner = self.inner.write();
        inner.delete_attributes(namespace, names)
    }

    fn find_attributes(
        &self,
        namespace: Option<String>,
        names: Vec<String>,
        hint: Option<String>,
    ) -> Vec<(String, String)> {
        let inner = self.inner.read_recursive();
        inner.find_attributes(namespace, names, hint)
    }
}

impl VideoObjectProxy {
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

    pub fn transform_geometry(&self, ops: &Vec<&VideoObjectBBoxTransformation>) {
        for o in ops {
            match o {
                VideoObjectBBoxTransformation::Scale(kx, ky) => {
                    self.get_detection_box().scale(*kx, *ky);
                    if let Some(mut t) = self.get_track_box() {
                        t.scale(*kx, *ky);
                    }
                }
                VideoObjectBBoxTransformation::Shift(dx, dy) => {
                    self.get_detection_box().shift(*dx, *dy);
                    if let Some(mut t) = self.get_track_box() {
                        t.shift(*dx, *dy);
                    }
                }
            }
        }
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

    pub fn get_track_id(&self) -> Option<i64> {
        let inner = self.inner.read_recursive();
        inner.track_id
    }

    #[allow(clippy::too_many_arguments)]
    #[new]
    pub fn new(
        id: i64,
        namespace: String,
        label: String,
        detection_box: RBBox,
        attributes: HashMap<(String, String), Attribute>,
        confidence: Option<f32>,
        track_id: Option<i64>,
        track_box: Option<RBBox>,
    ) -> Self {
        let (namespace_id, label_id) =
            get_object_id(&namespace, &label).map_or((None, None), |(c, o)| (Some(c), Some(o)));

        let object = VideoObject {
            id,
            namespace,
            label,
            detection_box: detection_box
                .try_into()
                .expect("Failed to convert RBBox to RBBoxData"),
            attributes,
            confidence,
            track_id,
            track_box: track_box
                .map(|b| b.try_into().expect("Failed to convert RBBox to RBBoxData")),
            namespace_id,
            label_id,
            ..Default::default()
        };
        Self {
            inner: Arc::new(RwLock::new(object)),
        }
    }

    /// Returns object's attributes as a list of tuples ``(namespace, name)``.
    ///
    /// Returns
    /// -------
    /// List[Tuple[str, str]]
    ///   List of attribute identifiers as ``(namespace, name)``.
    ///
    #[getter]
    #[pyo3(name = "attributes")]
    pub fn get_attributes_py(&self) -> Vec<(String, String)> {
        self.get_attributes()
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
    pub fn get_detection_box(&self) -> RBBox {
        RBBox::borrowed_detection_box(self.inner.clone())
    }

    #[getter]
    pub fn get_track_box(&self) -> Option<RBBox> {
        if self.get_track_id().is_some() {
            Some(RBBox::borrowed_track_box(self.inner.clone()))
        } else {
            None
        }
    }

    /// Accesses object's children. If the object is detached from a frame, an empty view is returned.
    ///
    /// Returns
    /// -------
    /// :py:class:`VideoObjectsView`
    ///   A view of the object's children.
    ///
    #[getter]
    pub fn children_ref(&self) -> VideoObjectsView {
        self.get_children().into()
    }

    /// Clears all object's attributes.
    ///
    #[pyo3(name = "clear_attributes")]
    pub fn clear_attributes_py(&mut self) {
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
    pub fn get_confidence(&self) -> Option<f32> {
        let inner = self.inner.read_recursive();
        inner.confidence
    }

    /// Returns object's namespace. When used as setter, allows setting object's namespace.
    ///
    /// Returns
    /// -------
    /// str
    ///   Object's namespace.
    ///
    #[getter]
    pub fn get_namespace(&self) -> String {
        self.inner.read_recursive().namespace.clone()
    }

    /// Deletes an attribute from the object.
    /// If the attribute is not found, returns None, otherwise returns the deleted attribute.
    ///
    /// Parameters
    /// ----------
    /// namespace : str
    ///   Attribute namespace.
    /// name : str
    ///   Attribute name.
    ///
    /// Returns
    /// -------
    /// :py:class:`Attribute` or None
    ///   Deleted attribute or None if the attribute is not found.
    ///
    #[pyo3(name = "delete_attribute")]
    pub fn delete_attribute_py(&mut self, namespace: String, name: String) -> Option<Attribute> {
        match self.delete_attribute(namespace, name) {
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
    /// namespace : str or None
    ///   Attribute namespace. If None, it is ignored when candidates are selected for removal.
    /// names : List[str]
    ///   Attribute names. If empty, it is ignored when candidates are selected for removal.
    ///
    #[pyo3(name = "delete_attributes")]
    #[pyo3(signature = (namespace=None, names=vec![], no_gil=false))]
    pub fn delete_attributes_gil(
        &mut self,
        namespace: Option<String>,
        names: Vec<String>,
        no_gil: bool,
    ) {
        release_gil!(no_gil, move || {
            {
                let mut object = self.inner.write();
                object.add_modification(VideoObjectModification::Attributes);
            }
            self.delete_attributes(namespace, names)
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

    /// finds and returns names of attributes by expression based on namespace, names and hint.
    ///
    /// Parameters
    /// ----------
    /// namespace : str or None
    ///   Attribute namespace. If None, it is ignored when candidates are selected.
    /// names : List[str]
    ///   Attribute names. If empty, it is ignored when candidates are selected.
    /// hint : str or None
    ///   Hint for the attribute name. If None, it is ignored when candidates are selected.
    ///
    /// Returns
    /// -------
    /// List[Tuple[str, str]]
    ///   List of tuples with attribute namespaces and names.
    ///
    #[pyo3(name = "find_attributes")]
    #[pyo3(signature = (namespace=None, names=vec![], hint=None))]
    pub fn find_attributes_py(
        &self,
        namespace: Option<String>,
        names: Vec<String>,
        hint: Option<String>,
    ) -> Vec<(String, String)> {
        self.find_attributes(namespace, names, hint)
    }

    /// Fetches attribute by namespace and name. The attribute is fetched by value, not reference, however attribute's values are fetched as CoW,
    /// until the modification the values are shared. If the attribute is not found, returns None.
    ///
    /// Remember, because the attribute is fetched as a copy,
    /// that changing attribute properties will not change the attribute kept in the object.
    ///
    /// Parameters
    /// ----------
    /// namespace : str
    ///   Attribute namespace.
    /// name : str
    ///   Attribute name.
    ///
    /// Returns
    /// -------
    /// :py:class:`Attribute` or None
    ///   Attribute or None if the attribute is not found.
    ///
    #[pyo3(name = "get_attribute")]
    pub fn get_attribute_py(&self, namespace: String, name: String) -> Option<Attribute> {
        self.get_attribute(namespace, name)
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
    pub fn set_attribute_py(&mut self, attribute: &Attribute) -> Option<Attribute> {
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

    #[setter]
    pub fn set_detection_box(&self, bbox: RBBox) {
        let mut inner = self.inner.write();
        inner.detection_box = bbox
            .try_into()
            .expect("Failed to convert RBBox to RBBoxData");
        inner.add_modification(VideoObjectModification::BoundingBox);
    }

    pub fn set_track_info(&self, track_id: i64, bbox: RBBox) {
        let mut inner = self.inner.write();
        inner.track_box = Some(
            bbox.try_into()
                .expect("Failed to convert RBBox to RBBoxData"),
        );
        inner.track_id = Some(track_id);
        inner.add_modification(VideoObjectModification::TrackInfo);
    }

    pub fn set_track_box(&self, bbox: RBBox) {
        let mut inner = self.inner.write();
        inner.track_box = Some(
            bbox.try_into()
                .expect("Failed to convert RBBox to RBBoxData"),
        );
        inner.add_modification(VideoObjectModification::TrackInfo);
    }

    pub fn clear_track_info(&self) {
        let mut inner = self.inner.write();
        inner.track_box = None;
        inner.track_id = None;
        inner.add_modification(VideoObjectModification::TrackInfo);
    }

    #[setter]
    pub fn set_draw_label(&self, draw_label: Option<String>) {
        let mut inner = self.inner.write();
        inner.draw_label = draw_label;
        inner.add_modification(VideoObjectModification::DrawLabel);
    }

    #[setter]
    pub fn set_id(&self, id: i64) -> PyResult<()> {
        if self.get_frame().is_some() {
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
    pub fn set_namespace(&self, namespace: String) {
        let mut inner = self.inner.write();
        inner.namespace = namespace;
        inner.add_modification(VideoObjectModification::Namespace);
    }

    #[setter]
    pub fn set_label(&self, label: String) {
        let mut inner = self.inner.write();
        inner.label = label;
        inner.add_modification(VideoObjectModification::Label);
    }

    #[setter]
    pub fn set_confidence(&self, confidence: Option<f32>) {
        let mut inner = self.inner.write();
        inner.confidence = confidence;
        inner.add_modification(VideoObjectModification::Confidence);
    }

    #[pyo3(name = "transform_geometry")]
    #[pyo3(signature = (ops, no_gil = false))]
    fn transform_geometry_gil(&self, ops: Vec<VideoObjectBBoxTransformationProxy>, no_gil: bool) {
        release_gil!(no_gil, || {
            let inner_ops = ops.iter().map(|op| op.get_ref()).collect::<Vec<_>>();
            self.transform_geometry(&inner_ops);
        })
    }

    fn get_pyobject(&self, namespace: String, name: String) -> Option<PyObject> {
        let inner = self.inner.read_recursive();
        inner.get_py_object_by_ref(&namespace, &name)
    }

    fn set_pyobject(
        &self,
        namespace: String,
        name: String,
        pyobject: PyObject,
    ) -> Option<PyObject> {
        let mut inner = self.inner.write();
        inner.set_py_object(&namespace, &name, pyobject)
    }

    fn delete_pyobject(&self, namespace: String, name: String) -> Option<PyObject> {
        let mut inner = self.inner.write();
        inner.del_py_object(&namespace, &name)
    }

    fn list_pyobjects(&self) -> Vec<(String, String)> {
        let inner = self.inner.read_recursive();
        inner.list_py_objects()
    }

    fn clear_pyobjects(&self) {
        let mut inner = self.inner.write();
        inner.clear_py_objects()
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::attribute::attribute_value::AttributeValue;
    use crate::primitives::attribute::AttributeMethods;
    use crate::primitives::bbox::transformations::VideoObjectBBoxTransformation;
    use crate::primitives::message::video::object::VideoObjectBuilder;
    use crate::primitives::{
        AttributeBuilder, IdCollisionResolutionPolicy, RBBox, VideoObjectModification,
        VideoObjectProxy,
    };
    use crate::test::utils::{gen_frame, s};

    fn get_object() -> VideoObjectProxy {
        VideoObjectProxy::from_video_object(
            VideoObjectBuilder::default()
                .id(1)
                .modifications(vec![])
                .namespace("model".to_string())
                .label("label".to_string())
                .detection_box(
                    RBBox::new(0.0, 0.0, 1.0, 1.0, None)
                        .try_into()
                        .expect("Failed to convert RBBox to RBBoxData"),
                )
                .confidence(Some(0.5))
                .attributes(
                    vec![
                        AttributeBuilder::default()
                            .namespace("namespace".to_string())
                            .name("name".to_string())
                            .values(vec![AttributeValue::string("value".to_string(), None)])
                            .hint(None)
                            .build()
                            .unwrap(),
                        AttributeBuilder::default()
                            .namespace("namespace".to_string())
                            .name("name2".to_string())
                            .values(vec![AttributeValue::string("value2".to_string(), None)])
                            .hint(None)
                            .build()
                            .unwrap(),
                        AttributeBuilder::default()
                            .namespace("namespace2".to_string())
                            .name("name".to_string())
                            .values(vec![AttributeValue::string("value".to_string(), None)])
                            .hint(None)
                            .build()
                            .unwrap(),
                    ]
                    .into_iter()
                    .map(|a| ((a.namespace.clone(), a.name.clone()), a))
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
        obj.delete_attributes(Some(s("namespace")), vec![]);
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

        t.set_detection_box(RBBox::new(0.0, 0.0, 1.0, 1.0, None));
        t.clear_attributes_py();
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
        f.add_object(&o, IdCollisionResolutionPolicy::Error)
            .unwrap();
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
        f.add_object(&o.detached_copy(), IdCollisionResolutionPolicy::Error)
            .unwrap();
    }

    #[test]
    fn test_transform_geometry() {
        let o = get_object();
        o.set_track_info(13, RBBox::new(0.0, 0.0, 10.0, 20.0, None));
        let ops = vec![VideoObjectBBoxTransformation::Shift(10.0, 20.0)];
        let ref_ops = ops.iter().map(|op| op).collect();
        o.transform_geometry(&ref_ops);
        let new_bb = o.get_detection_box();
        assert_eq!(new_bb.get_xc(), 10.0);
        assert_eq!(new_bb.get_yc(), 20.0);
        assert_eq!(new_bb.get_width(), 1.0);
        assert_eq!(new_bb.get_height(), 1.0);

        let new_track_bb = o.get_track_box().unwrap();
        assert_eq!(new_track_bb.get_xc(), 10.0);
        assert_eq!(new_track_bb.get_yc(), 20.0);
        assert_eq!(new_track_bb.get_width(), 10.0);
        assert_eq!(new_track_bb.get_height(), 20.0);

        let ops = vec![VideoObjectBBoxTransformation::Scale(2.0, 4.0)];
        let ref_ops = ops.iter().map(|op| op).collect();
        o.transform_geometry(&ref_ops);
        let new_bb = o.get_detection_box();
        assert_eq!(new_bb.get_xc(), 20.0);
        assert_eq!(new_bb.get_yc(), 80.0);
        assert_eq!(new_bb.get_width(), 2.0);
        assert_eq!(new_bb.get_height(), 4.0);

        let new_track_bb = o.get_track_box().unwrap();
        assert_eq!(new_track_bb.get_xc(), 20.0);
        assert_eq!(new_track_bb.get_yc(), 80.0);
        assert_eq!(new_track_bb.get_width(), 20.0);
        assert_eq!(new_track_bb.get_height(), 80.0);
    }
}
