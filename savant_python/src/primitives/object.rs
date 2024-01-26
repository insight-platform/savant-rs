use crate::primitives::attribute_value::AttributeValue;
use crate::primitives::bbox::VideoObjectBBoxTransformation;
use crate::primitives::{Attribute, RBBox};
use crate::{release_gil, with_gil};
use log::warn;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyBytes;
use pyo3::{pyclass, pymethods, Py, PyAny, PyObject, PyResult};
use savant_core::json_api::ToSerdeJsonValue;
use savant_core::primitives::object::{ObjectAccess, ObjectOperations};
use savant_core::primitives::{rust, WithAttributes};
use savant_core::protobuf::{from_pb, ToProtobuf};
use serde_json::Value;

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
pub struct VideoObject(pub(crate) rust::VideoObject);

#[pymethods]
impl VideoObject {
    #[allow(clippy::too_many_arguments)]
    #[new]
    pub fn new(
        id: i64,
        namespace: &str,
        label: &str,
        detection_box: RBBox,
        attributes: Vec<Attribute>,
        confidence: Option<f32>,
        track_id: Option<i64>,
        track_box: Option<RBBox>,
    ) -> Self {
        let object = rust::VideoObjectBuilder::default()
            .id(id)
            .namespace(namespace.to_string())
            .label(label.to_string())
            .detection_box(detection_box.0)
            .attributes(attributes.into_iter().map(|a| a.0).collect())
            .confidence(confidence)
            .track_id(track_id)
            .track_box(track_box.map(|b| b.0))
            .build()
            .unwrap();

        Self(object)
    }

    #[pyo3(name = "to_protobuf")]
    #[pyo3(signature = (no_gil = true))]
    fn to_protobuf_gil(&self, no_gil: bool) -> PyResult<PyObject> {
        let bytes =
            release_gil!(no_gil, || { self.0.with_object_ref(|o| o.to_pb()) }).map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "Failed to serialize video object to protobuf: {}",
                    e
                ))
            })?;
        with_gil!(|py| {
            let bytes = PyBytes::new(py, &bytes);
            Ok(PyObject::from(bytes))
        })
    }

    #[staticmethod]
    #[pyo3(name = "from_protobuf")]
    #[pyo3(signature = (bytes, no_gil = true))]
    fn from_protobuf_gil(bytes: &PyBytes, no_gil: bool) -> PyResult<Self> {
        let bytes = bytes.as_bytes();
        release_gil!(no_gil, || {
            let obj = from_pb::<savant_core::protobuf::VideoObject, rust::VideoObject>(bytes)
                .map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "Failed to deserialize video object from protobuf: {}",
                        e
                    ))
                })?;
            Ok(Self(obj))
        })
    }

    #[getter]
    fn get_namespace(&self) -> String {
        self.0.get_namespace()
    }

    #[getter]
    fn get_label(&self) -> String {
        self.0.get_label()
    }

    #[getter]
    fn get_id(&self) -> i64 {
        self.0.get_id()
    }

    #[getter]
    fn get_detection_box(&self) -> RBBox {
        RBBox(self.0.get_detection_box())
    }

    #[getter]
    fn get_track_id(&self) -> Option<i64> {
        self.0.get_track_id()
    }

    #[getter]
    fn get_track_box(&self) -> Option<RBBox> {
        self.0.get_track_box().map(RBBox)
    }

    #[getter]
    fn get_confidence(&self) -> Option<f32> {
        self.0.get_confidence()
    }

    #[getter]
    fn attributes(&self) -> Vec<(String, String)> {
        self.0.get_attributes()
    }

    fn get_attribute(&self, namespace: &str, name: &str) -> Option<Attribute> {
        self.0.get_attribute(namespace, name).map(Attribute)
    }

    fn set_attribute(&mut self, attribute: &Attribute) -> Option<Attribute> {
        self.0.set_attribute(attribute.0.clone()).map(Attribute)
    }

    fn set_persistent_attribute(
        &mut self,
        namespace: &str,
        name: &str,
        is_hidden: bool,
        hint: Option<String>,
        values: Option<Vec<AttributeValue>>,
    ) {
        let values = match values {
            Some(values) => values.into_iter().map(|v| v.0).collect::<Vec<_>>(),
            None => vec![],
        };
        let hint = hint.as_deref();
        self.0
            .set_persistent_attribute(namespace, name, &hint, is_hidden, values)
    }

    fn set_temporary_attribute(
        &mut self,
        namespace: &str,
        name: &str,
        is_hidden: bool,
        hint: Option<String>,
        values: Option<Vec<AttributeValue>>,
    ) {
        let values = match values {
            Some(values) => values.into_iter().map(|v| v.0).collect::<Vec<_>>(),
            None => vec![],
        };
        let hint = hint.as_deref();
        self.0
            .set_temporary_attribute(namespace, name, &hint, is_hidden, values)
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct BorrowedVideoObject(pub(crate) rust::BorrowedVideoObject);

impl ToSerdeJsonValue for BorrowedVideoObject {
    fn to_serde_json_value(&self) -> Value {
        self.0.to_serde_json_value()
    }
}

#[pymethods]
impl BorrowedVideoObject {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", &self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
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

    /// Clears all object's attributes.
    ///
    pub fn clear_attributes(&mut self) {
        self.0.clear_attributes()
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

    #[setter]
    pub fn set_confidence(&mut self, confidence: Option<f32>) {
        self.0.set_confidence(confidence);
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

    #[getter]
    pub fn get_namespace_id(&self) -> Option<i64> {
        self.0.get_namespace_id()
    }

    #[setter]
    pub fn set_namespace(&mut self, namespace: &str) {
        self.0.set_namespace(namespace);
    }

    #[getter]
    pub fn get_label(&self) -> String {
        self.0.get_label()
    }

    #[getter]
    pub fn get_label_id(&self) -> Option<i64> {
        self.0.get_label_id()
    }
    #[setter]
    pub fn set_label(&mut self, label: &str) {
        self.0.set_label(label);
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
    pub fn delete_attribute(&mut self, namespace: &str, name: &str) -> Option<Attribute> {
        self.0.delete_attribute(namespace, name).map(Attribute)
    }

    pub fn delete_attributes_with_ns(&mut self, namespace: &str) {
        self.0.delete_attributes_with_ns(namespace)
    }

    pub fn delete_attributes_with_names(&mut self, names: Vec<String>) {
        let label_refs = names.iter().map(|v| v.as_ref()).collect::<Vec<&str>>();
        self.0.delete_attributes_with_names(&label_refs)
    }

    pub fn delete_attributes_with_hints(&mut self, hints: Vec<Option<String>>) {
        let hint_opts_refs = hints
            .iter()
            .map(|v| v.as_deref())
            .collect::<Vec<Option<&str>>>();
        let hint_refs = hint_opts_refs.iter().collect::<Vec<_>>();

        self.0.delete_attributes_with_hints(&hint_refs)
    }

    /// Returns a copy of the object with the same properties but detached from the frame and without a parent set.
    ///
    /// Returns
    /// -------
    /// :py:class:`VideoObject`
    ///   A copy of the object.
    ///
    pub fn detached_copy(&self) -> VideoObject {
        VideoObject(self.0.detached_copy())
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
        self.0.calculate_draw_label()
    }

    #[setter]
    pub fn set_draw_label(&mut self, draw_label: Option<String>) {
        self.0.set_draw_label(draw_label);
    }

    pub fn find_attributes_with_ns(&mut self, namespace: &str) -> Vec<(String, String)> {
        self.0.find_attributes_with_ns(namespace)
    }

    pub fn find_attributes_with_names(&mut self, names: Vec<String>) -> Vec<(String, String)> {
        let label_refs = names.iter().map(|v| v.as_ref()).collect::<Vec<&str>>();
        self.0.find_attributes_with_names(&label_refs)
    }
    pub fn find_attributes_with_hints(
        &mut self,
        hints: Vec<Option<String>>,
    ) -> Vec<(String, String)> {
        let hint_opts_refs = hints
            .iter()
            .map(|v| v.as_deref())
            .collect::<Vec<Option<&str>>>();
        let hint_refs = hint_opts_refs.iter().collect::<Vec<_>>();

        self.0.find_attributes_with_hints(&hint_refs)
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
    pub fn get_attribute(&self, namespace: &str, name: &str) -> Option<Attribute> {
        self.0.get_attribute(namespace, name).map(Attribute)
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

    /// Sets new persistent attribute for the object. If the attribute is already set, it is replaced.
    ///
    /// Parameters
    /// ----------
    /// namespace : str
    ///   Attribute namespace.
    /// name : str
    ///   Attribute name.
    /// hint : str or None
    ///   Attribute hint.
    /// is_hidden : bool
    ///   Attribute hidden flag.
    /// values : List[:py:class:`AttributeValue`] or None
    ///   Attribute values.
    ///
    #[pyo3(signature = (namespace, name, is_hidden = false, hint = None, values = vec![]))]
    pub fn set_persistent_attribute(
        &mut self,
        namespace: &str,
        name: &str,
        is_hidden: bool,
        hint: Option<String>,
        values: Option<Vec<AttributeValue>>,
    ) {
        let values = match values {
            Some(values) => values.into_iter().map(|v| v.0).collect::<Vec<_>>(),
            None => vec![],
        };
        let hint = hint.as_deref();
        self.0
            .set_persistent_attribute(namespace, name, &hint, is_hidden, values)
    }

    /// Sets new temporary attribute for the object. If the attribute is already set, it is replaced.
    ///
    /// Parameters
    /// ----------
    /// namespace : str
    ///   Attribute namespace.
    /// name : str
    ///   Attribute name.
    /// hint : str or None
    ///   Attribute hint.
    /// is_hidden : bool
    ///   Attribute hidden flag.
    /// values : List[:py:class:`AttributeValue`] or None
    ///   Attribute values.
    ///
    #[pyo3(signature = (namespace, name, is_hidden = false, hint = None, values = vec![]))]
    pub fn set_temporary_attribute(
        &mut self,
        namespace: &str,
        name: &str,
        is_hidden: bool,
        hint: Option<String>,
        values: Option<Vec<AttributeValue>>,
    ) {
        let values = match values {
            Some(values) => values.into_iter().map(|v| v.0).collect::<Vec<_>>(),
            None => vec![],
        };
        let hint = hint.as_deref();
        self.0
            .set_temporary_attribute(namespace, name, &hint, is_hidden, values)
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

    #[setter]
    pub fn set_detection_box(&mut self, bbox: RBBox) {
        self.0.set_detection_box(bbox.0);
    }

    #[getter]
    pub fn get_track_id(&self) -> Option<i64> {
        warn!("get_track_id is deprecated, use track_id instead");
        self.0.get_track_id()
    }

    #[setter]
    pub fn set_track_id(&mut self, track_id: Option<i64>) {
        self.0.set_track_id(track_id);
    }

    #[getter]
    pub fn get_track_box(&self) -> Option<RBBox> {
        self.0.get_track_box().map(RBBox)
    }

    #[setter]
    pub fn set_track_box(&mut self, bbox: RBBox) {
        self.0.set_track_box(bbox.0);
    }

    pub fn set_track_info(&mut self, track_id: i64, bbox: RBBox) {
        self.0.set_track_info(track_id, bbox.0);
    }

    pub fn clear_track_info(&mut self) {
        self.0.clear_track_info()
    }

    fn transform_geometry(&mut self, ops: Vec<VideoObjectBBoxTransformation>) {
        let inner_ops = ops.iter().map(|op| op.0).collect::<Vec<_>>();
        self.0.transform_geometry(&inner_ops);
    }

    #[pyo3(name = "to_protobuf")]
    #[pyo3(signature = (no_gil = true))]
    fn to_protobuf_gil(&self, no_gil: bool) -> PyResult<PyObject> {
        let bytes =
            release_gil!(no_gil, || { self.0.with_object_ref(|o| o.to_pb()) }).map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "Failed to serialize video object to protobuf: {}",
                    e
                ))
            })?;
        with_gil!(|py| {
            let bytes = PyBytes::new(py, &bytes);
            Ok(PyObject::from(bytes))
        })
    }

    #[getter]
    pub fn memory_handle(&self) -> usize {
        self as *const Self as usize
    }
}
