use crate::primitives::bbox::VideoObjectBBoxTransformation;
use crate::primitives::objects_view::VideoObjectsView;
use crate::primitives::{Attribute, RBBox, VideoFrame};
use crate::release_gil;
use pyo3::exceptions::PyRuntimeError;
use pyo3::{pyclass, pymethods, Py, PyAny, PyResult};
use savant_core::json_api::ToSerdeJsonValue;
use savant_core::primitives::{rust, AttributeMethods};
use serde_json::Value;
use std::collections::HashMap;

#[pyclass]
#[derive(Debug, Clone)]
pub enum IdCollisionResolutionPolicy {
    GenerateNewId,
    Overwrite,
    Error,
}

impl From<IdCollisionResolutionPolicy> for rust::IdCollisionResolutionPolicy {
    fn from(value: IdCollisionResolutionPolicy) -> Self {
        match value {
            IdCollisionResolutionPolicy::GenerateNewId => {
                rust::IdCollisionResolutionPolicy::GenerateNewId
            }
            IdCollisionResolutionPolicy::Overwrite => rust::IdCollisionResolutionPolicy::Overwrite,
            IdCollisionResolutionPolicy::Error => rust::IdCollisionResolutionPolicy::Error,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct VideoObject(pub(crate) rust::VideoObjectProxy);

impl ToSerdeJsonValue for VideoObject {
    fn to_serde_json_value(&self) -> Value {
        self.0.to_serde_json_value()
    }
}

#[pymethods]
impl VideoObject {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", &self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    pub fn get_track_id(&self) -> Option<i64> {
        self.0.get_label_id()
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
        Self(rust::VideoObjectProxy::new(
            id,
            namespace,
            label,
            detection_box.0,
            attributes
                .into_iter()
                .map(|(k, v)| (k.clone(), v.0))
                .collect(),
            confidence,
            track_id,
            track_box.map(|b| b.0),
        ))
    }

    /// Returns object's attributes as a list of tuples ``(namespace, name)``.
    ///
    /// Returns
    /// -------
    /// List[Tuple[str, str]]
    ///   List of attribute identifiers as ``(namespace, name)``.
    ///
    #[getter]
    pub fn attributes(&self) -> Vec<(String, String)> {
        self.0.get_attributes()
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
        RBBox(self.0.get_detection_box())
    }

    #[getter]
    pub fn get_track_box(&self) -> Option<RBBox> {
        self.0.get_track_box().map(RBBox)
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
        self.0.get_children().into()
    }

    /// Clears all object's attributes.
    ///
    pub fn clear_attributes(&mut self) {
        self.0.clear_attributes()
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
        self.0.get_confidence()
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
        self.0.get_namespace()
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
    pub fn delete_attribute(&mut self, namespace: String, name: String) -> Option<Attribute> {
        self.0.delete_attribute(namespace, name).map(Attribute)
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
        release_gil!(no_gil, || { self.0.delete_attributes(namespace, names) })
    }

    /// Returns a copy of the object with the same properties but detached from the frame and without a parent set.
    ///
    /// Returns
    /// -------
    /// :py:class:`VideoObject`
    ///   A copy of the object.
    ///
    pub fn detached_copy(&self) -> Self {
        Self(self.0.detached_copy())
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
        self.0.get_draw_label()
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
    #[pyo3(signature = (namespace=None, names=vec![], hint=None))]
    pub fn find_attributes(
        &self,
        namespace: Option<String>,
        names: Vec<String>,
        hint: Option<String>,
    ) -> Vec<(String, String)> {
        self.0.find_attributes(namespace, names, hint)
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
    pub fn get_attribute(&self, namespace: String, name: String) -> Option<Attribute> {
        self.0.get_attribute(namespace, name).map(Attribute)
    }

    /// Returns the :py:class:`VideoFrame` reference to a frame the object belongs to.
    ///
    /// Returns
    /// -------
    /// :py:class:`VideoFrame` or None
    ///   A reference to a frame the object belongs to.
    ///
    pub fn get_frame(&self) -> Option<VideoFrame> {
        self.0.get_frame().map(VideoFrame)
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
        self.0.get_id()
    }

    /// The object is detached if it is not attached to any frame. Such state may cause it impossible to operate with certain object properties.
    ///
    /// Returns
    /// -------
    /// bool
    ///   True if the object is detached, False otherwise.
    ///
    pub fn is_detached(&self) -> bool {
        self.0.is_detached()
    }

    /// The object is spoiled if it is outlived the a belonging frame. Such state may cause it impossible to operate with certain object properties.
    ///
    /// Returns
    /// -------
    /// bool
    ///   True if the object is spoiled, False otherwise.
    pub fn is_spoiled(&self) -> bool {
        self.0.is_spoiled()
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
        self.0.get_label()
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
    pub fn set_attribute(&mut self, attribute: &Attribute) -> Option<Attribute> {
        self.0.set_attribute(attribute.0.clone()).map(Attribute)
    }

    #[setter]
    pub fn set_detection_box(&self, bbox: RBBox) {
        self.0.set_detection_box(bbox.0);
    }

    pub fn set_track_info(&self, track_id: i64, bbox: RBBox) {
        self.0.set_track_info(track_id, bbox.0);
    }

    pub fn set_track_box(&self, bbox: RBBox) {
        self.0.set_track_box(bbox.0);
    }

    pub fn clear_track_info(&self) {
        self.0.clear_track_info()
    }

    #[setter]
    pub fn set_draw_label(&self, draw_label: Option<String>) {
        self.0.set_draw_label(draw_label);
    }

    #[setter]
    pub fn set_id(&self, id: i64) -> PyResult<()> {
        self.0.set_id(id).map_err(|e| {
            PyRuntimeError::new_err(format!(
                "Failed to set object id to {}: {}",
                id,
                e.to_string()
            ))
        })
    }

    #[setter]
    pub fn set_namespace(&self, namespace: String) {
        self.0.set_namespace(namespace);
    }

    #[setter]
    pub fn set_label(&self, label: String) {
        self.0.set_label(label);
    }

    #[setter]
    pub fn set_confidence(&self, confidence: Option<f32>) {
        self.0.set_confidence(confidence);
    }

    #[pyo3(name = "transform_geometry")]
    #[pyo3(signature = (ops, no_gil = false))]
    fn transform_geometry_gil(&self, ops: Vec<VideoObjectBBoxTransformation>, no_gil: bool) {
        release_gil!(no_gil, || {
            let inner_ops = ops.iter().map(|op| op.0).collect::<Vec<_>>();
            self.0.transform_geometry(&inner_ops);
        })
    }
}
