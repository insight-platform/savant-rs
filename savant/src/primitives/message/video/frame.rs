pub mod context;
pub mod frame_update;

use crate::primitives::attribute::{AttributeMethods, Attributive};
use crate::primitives::bbox::transformations::{
    VideoObjectBBoxTransformation, VideoObjectBBoxTransformationProxy,
};
use crate::primitives::message::video::object::objects_view::VideoObjectsView;
use crate::primitives::message::video::object::VideoObject;
use crate::primitives::message::video::query::match_query::{
    IntExpression, MatchQuery, StringExpression,
};
use crate::primitives::message::video::query::py::MatchQueryProxy;
use crate::primitives::message::TRACE_ID_LEN;
use crate::primitives::pyobject::PyObjectMeta;
use crate::primitives::to_json_value::ToSerdeJsonValue;
use crate::primitives::{
    Attribute, IdCollisionResolutionPolicy, Message, RBBox, SetDrawLabelKind,
    SetDrawLabelKindProxy, VideoFrameUpdate, VideoObjectProxy,
};
use crate::utils::python::release_gil;
use crate::utils::symbol_mapper::{get_model_id, get_object_label};
use anyhow::bail;
use derive_builder::Builder;
use ndarray::IxDyn;
use numpy::PyArray;
use parking_lot::RwLock;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyBytes;
use pyo3::{pyclass, pymethods, Py, PyAny, PyObject, PyResult, Python};
use rayon::prelude::*;
use rkyv::{with::Skip, Archive, Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::mem;
use std::ops::Deref;
use std::sync::{Arc, Weak};

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct ExternalFrame {
    #[pyo3(get, set)]
    pub method: String,
    #[pyo3(get, set)]
    pub location: Option<String>,
}

impl ToSerdeJsonValue for ExternalFrame {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!({
            "method": self.method,
            "location": self.location,
        })
    }
}

#[pymethods]
impl ExternalFrame {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[new]
    pub fn new(method: String, location: Option<String>) -> Self {
        Self { method, location }
    }
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub enum VideoFrameContent {
    External(ExternalFrame),
    Internal(Vec<u8>),
    None,
}

impl ToSerdeJsonValue for VideoFrameContent {
    fn to_serde_json_value(&self) -> Value {
        match self {
            VideoFrameContent::External(data) => {
                serde_json::json!({"external": data.to_serde_json_value()})
            }
            VideoFrameContent::Internal(_) => {
                serde_json::json!({ "internal": Value::Null })
            }
            VideoFrameContent::None => Value::Null,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
#[pyo3(name = "VideoFrameContent")]
pub struct VideoFrameContentProxy {
    pub(crate) inner: Arc<VideoFrameContent>,
}

impl VideoFrameContentProxy {
    pub fn new(inner: VideoFrameContent) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }

    pub fn new_arced(inner: Arc<VideoFrameContent>) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl VideoFrameContentProxy {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[staticmethod]
    pub fn external(method: String, location: Option<String>) -> Self {
        Self::new(VideoFrameContent::External(ExternalFrame::new(
            method, location,
        )))
    }

    #[staticmethod]
    pub fn internal(data: Vec<u8>) -> Self {
        Self::new(VideoFrameContent::Internal(data))
    }

    #[staticmethod]
    pub fn none() -> Self {
        Self::new(VideoFrameContent::None)
    }

    pub fn is_external(&self) -> bool {
        matches!(*self.inner, VideoFrameContent::External(_))
    }

    pub fn is_internal(&self) -> bool {
        matches!(*self.inner, VideoFrameContent::Internal(_))
    }

    pub fn is_none(&self) -> bool {
        matches!(*self.inner, VideoFrameContent::None)
    }

    pub fn get_data(&self) -> PyResult<Vec<u8>> {
        match &*self.inner {
            VideoFrameContent::Internal(data) => Ok(data.clone()),
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "Video data is not stored internally",
            )),
        }
    }

    pub fn get_data_as_bytes(&self) -> PyResult<PyObject> {
        match &*self.inner {
            VideoFrameContent::Internal(data) => Ok(Python::with_gil(|py| {
                let bytes = PyBytes::new(py, data);
                PyObject::from(bytes)
            })),
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "Video data is not stored internally",
            )),
        }
    }

    pub fn get_method(&self) -> PyResult<String> {
        match &*self.inner {
            VideoFrameContent::External(data) => Ok(data.method.clone()),
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "Video data is not stored externally",
            )),
        }
    }

    pub fn get_location(&self) -> PyResult<Option<String>> {
        match &*self.inner {
            VideoFrameContent::External(data) => Ok(data.location.clone()),
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "Video data is not stored externally",
            )),
        }
    }
}

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub enum VideoFrameTranscodingMethod {
    Copy,
    Encoded,
}

impl ToSerdeJsonValue for VideoFrameTranscodingMethod {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!(format!("{:?}", self))
    }
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub enum FrameTransformation {
    InitialSize(u64, u64),
    Scale(u64, u64),
    Padding(u64, u64, u64, u64),
    ResultingSize(u64, u64),
}

impl ToSerdeJsonValue for FrameTransformation {
    fn to_serde_json_value(&self) -> Value {
        match self {
            FrameTransformation::InitialSize(width, height) => {
                serde_json::json!({"initial_size": [width, height]})
            }
            FrameTransformation::Scale(width, height) => {
                serde_json::json!({"scale": [width, height]})
            }
            FrameTransformation::Padding(left, top, right, bottom) => {
                serde_json::json!({"padding": [left, top, right, bottom]})
            }
            FrameTransformation::ResultingSize(width, height) => {
                serde_json::json!({"resulting_size": [width, height]})
            }
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
#[pyo3(name = "FrameTransformation")]
pub struct VideoFrameTransformationProxy {
    pub(crate) inner: FrameTransformation,
}

impl VideoFrameTransformationProxy {
    pub fn new(inner: FrameTransformation) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl VideoFrameTransformationProxy {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[staticmethod]
    pub fn initial_size(width: i64, height: i64) -> Self {
        assert!(width > 0 && height > 0);
        Self {
            inner: FrameTransformation::InitialSize(
                u64::try_from(width).unwrap(),
                u64::try_from(height).unwrap(),
            ),
        }
    }

    #[staticmethod]
    pub fn resulting_size(width: i64, height: i64) -> Self {
        assert!(width > 0 && height > 0);
        Self {
            inner: FrameTransformation::ResultingSize(
                u64::try_from(width).unwrap(),
                u64::try_from(height).unwrap(),
            ),
        }
    }

    #[staticmethod]
    pub fn scale(width: i64, height: i64) -> Self {
        assert!(width > 0 && height > 0);
        Self {
            inner: FrameTransformation::Scale(
                u64::try_from(width).unwrap(),
                u64::try_from(height).unwrap(),
            ),
        }
    }

    #[staticmethod]
    pub fn padding(left: i64, top: i64, right: i64, bottom: i64) -> Self {
        assert!(left >= 0 && top >= 0 && right >= 0 && bottom >= 0);
        Self {
            inner: FrameTransformation::Padding(
                u64::try_from(left).unwrap(),
                u64::try_from(top).unwrap(),
                u64::try_from(right).unwrap(),
                u64::try_from(bottom).unwrap(),
            ),
        }
    }

    #[getter]
    pub fn is_initial_size(&self) -> bool {
        matches!(self.inner, FrameTransformation::InitialSize(_, _))
    }

    #[getter]
    pub fn is_scale(&self) -> bool {
        matches!(self.inner, FrameTransformation::Scale(_, _))
    }

    #[getter]
    pub fn is_padding(&self) -> bool {
        matches!(self.inner, FrameTransformation::Padding(_, _, _, _))
    }

    #[getter]
    pub fn is_resulting_size(&self) -> bool {
        matches!(self.inner, FrameTransformation::ResultingSize(_, _))
    }

    #[getter]
    pub fn as_initial_size(&self) -> Option<(u64, u64)> {
        match &self.inner {
            FrameTransformation::InitialSize(w, h) => Some((*w, *h)),
            _ => None,
        }
    }

    #[getter]
    pub fn as_resulting_size(&self) -> Option<(u64, u64)> {
        match &self.inner {
            FrameTransformation::ResultingSize(w, h) => Some((*w, *h)),
            _ => None,
        }
    }

    #[getter]
    pub fn as_scale(&self) -> Option<(u64, u64)> {
        match &self.inner {
            FrameTransformation::Scale(w, h) => Some((*w, *h)),
            _ => None,
        }
    }

    #[getter]
    pub fn as_padding(&self) -> Option<(u64, u64, u64, u64)> {
        match &self.inner {
            FrameTransformation::Padding(l, t, r, b) => Some((*l, *t, *r, *b)),
            _ => None,
        }
    }
}

#[derive(Archive, Deserialize, Serialize, Debug, Clone, Builder)]
#[archive(check_bytes)]
pub struct VideoFrame {
    pub source_id: String,
    pub framerate: String,
    pub width: i64,
    pub height: i64,
    pub transcoding_method: VideoFrameTranscodingMethod,
    pub codec: Option<String>,
    pub keyframe: Option<bool>,
    #[builder(setter(skip))]
    pub time_base: (i32, i32),
    pub pts: i64,
    #[builder(setter(skip))]
    pub dts: Option<i64>,
    #[builder(setter(skip))]
    pub duration: Option<i64>,
    pub content: Arc<VideoFrameContent>,
    #[builder(setter(skip))]
    pub transformations: Vec<FrameTransformation>,
    #[builder(setter(skip))]
    pub attributes: HashMap<(String, String), Attribute>,
    #[builder(setter(skip))]
    pub offline_objects: HashMap<i64, VideoObject>,
    #[with(Skip)]
    #[builder(setter(skip))]
    pub(crate) resident_objects: HashMap<i64, Arc<RwLock<VideoObject>>>,
    #[with(Skip)]
    #[builder(setter(skip))]
    pub(crate) max_object_id: i64,
    #[with(Skip)]
    #[builder(default)]
    pub(crate) pyobjects: HashMap<(String, String), PyObject>,
    #[with(Skip)]
    #[builder(default)]
    pub(crate) trace_id: [u8; TRACE_ID_LEN],
}

impl Default for VideoFrame {
    fn default() -> Self {
        Self {
            source_id: String::new(),
            framerate: String::new(),
            width: 0,
            height: 0,
            transcoding_method: VideoFrameTranscodingMethod::Copy,
            codec: None,
            keyframe: None,
            time_base: (1, 1000000),
            pts: 0,
            dts: None,
            duration: None,
            content: Arc::new(VideoFrameContent::None),
            transformations: Vec::new(),
            attributes: HashMap::new(),
            offline_objects: HashMap::new(),
            resident_objects: HashMap::new(),
            max_object_id: 0,
            pyobjects: HashMap::new(),
            trace_id: [0; TRACE_ID_LEN],
        }
    }
}

impl ToSerdeJsonValue for VideoFrame {
    fn to_serde_json_value(&self) -> Value {
        use crate::version;
        serde_json::json!(
            {
                "version": version(),
                "type": "VideoFrame",
                "source_id": self.source_id,
                "framerate": self.framerate,
                "width": self.width,
                "height": self.height,
                "transcoding_method": self.transcoding_method.to_serde_json_value(),
                "codec": self.codec,
                "keyframe": self.keyframe,
                "time_base": self.time_base,
                "pts": self.pts,
                "dts": self.dts,
                "duration": self.duration,
                "content": self.content.to_serde_json_value(),
                "transformations": self.transformations.iter().map(|t| t.to_serde_json_value()).collect::<Vec<_>>(),
                "attributes": self.attributes.values().map(|v| v.to_serde_json_value()).collect::<Vec<_>>(),
                "objects": self.resident_objects.values().map(|o| o.read_recursive().to_serde_json_value()).collect::<Vec<_>>(),
                "pyobjects": "not_implemented"
            }
        )
    }
}

impl PyObjectMeta for Box<VideoFrame> {
    fn get_py_objects_ref(&self) -> &HashMap<(String, String), PyObject> {
        &self.pyobjects
    }

    fn get_py_objects_ref_mut(&mut self) -> &mut HashMap<(String, String), PyObject> {
        &mut self.pyobjects
    }
}

impl Attributive for Box<VideoFrame> {
    fn get_attributes_ref(&self) -> &HashMap<(String, String), Attribute> {
        &self.attributes
    }

    fn get_attributes_ref_mut(&mut self) -> &mut HashMap<(String, String), Attribute> {
        &mut self.attributes
    }

    fn take_attributes(&mut self) -> HashMap<(String, String), Attribute> {
        mem::take(&mut self.attributes)
    }

    fn place_attributes(&mut self, attributes: HashMap<(String, String), Attribute>) {
        self.attributes = attributes;
    }
}

impl VideoFrame {
    pub(crate) fn set_trace_id(&mut self, trace_id: [u8; TRACE_ID_LEN]) {
        self.trace_id = trace_id;
    }

    pub(crate) fn get_trace_id(&self) -> [u8; TRACE_ID_LEN] {
        self.trace_id
    }

    fn preserve(&mut self) {
        self.offline_objects = self
            .resident_objects
            .iter()
            .map(|(id, o)| (*id, o.read_recursive().clone()))
            .collect();
    }

    fn restore(&mut self) {
        self.resident_objects = mem::take(&mut self.offline_objects)
            .into_iter()
            .map(|(id, o)| (id, Arc::new(RwLock::new(o))))
            .collect();
    }

    pub fn deep_copy(&self) -> Self {
        let mut frame = self.clone();
        frame.preserve();
        frame.restore();
        frame
    }
}

#[pyclass]
#[derive(Debug, Clone)]
#[repr(C)]
#[pyo3(name = "VideoFrame")]
pub struct VideoFrameProxy {
    pub(crate) inner: Arc<RwLock<Box<VideoFrame>>>,
    pub(crate) is_parallelized: bool,
}

#[pyclass]
#[derive(Clone)]
pub struct BelongingVideoFrame {
    pub(crate) inner: Weak<RwLock<Box<VideoFrame>>>,
}

impl Debug for BelongingVideoFrame {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.inner.upgrade() {
            Some(inner) => f
                .debug_struct("BelongingVideoFrame")
                .field("stream_id", &inner.read_recursive().source_id)
                .finish(),
            None => f.debug_struct("Unset").finish(),
        }
    }
}

impl From<VideoFrameProxy> for BelongingVideoFrame {
    fn from(value: VideoFrameProxy) -> Self {
        Self {
            inner: Arc::downgrade(&value.inner),
        }
    }
}

impl From<BelongingVideoFrame> for VideoFrameProxy {
    fn from(value: BelongingVideoFrame) -> Self {
        Self {
            inner: value
                .inner
                .upgrade()
                .expect("Frame is dropped, you cannot use attached objects anymore"),
            is_parallelized: false,
        }
    }
}

impl From<&BelongingVideoFrame> for VideoFrameProxy {
    fn from(value: &BelongingVideoFrame) -> Self {
        Self {
            inner: value
                .inner
                .upgrade()
                .expect("Frame is dropped, you cannot use attached objects anymore"),
            is_parallelized: false,
        }
    }
}

impl AttributeMethods for VideoFrameProxy {
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

impl VideoFrameProxy {
    fn transform_geometry(&self, ops: &Vec<&VideoObjectBBoxTransformation>) {
        let objs = self.access_objects(&MatchQuery::Idle);
        for obj in objs {
            obj.transform_geometry(ops);
        }
    }

    pub fn get_trace_id(&self) -> [u8; TRACE_ID_LEN] {
        let inner = self.inner.read_recursive();
        inner.get_trace_id()
    }

    pub fn set_trace_id(&self, trace_id: [u8; TRACE_ID_LEN]) {
        let mut inner = self.inner.write();
        inner.set_trace_id(trace_id);
    }

    pub fn deep_copy(&self) -> Self {
        let inner = self.inner.read_recursive();
        let inner_copy = inner.deep_copy();
        drop(inner);
        Self::from_inner(inner_copy)
    }

    pub fn get_inner(&self) -> Arc<RwLock<Box<VideoFrame>>> {
        self.inner.clone()
    }

    pub(crate) fn from_inner(inner: VideoFrame) -> Self {
        VideoFrameProxy {
            inner: Arc::new(RwLock::new(Box::new(inner))),
            is_parallelized: false,
        }
    }

    pub fn access_objects(&self, q: &MatchQuery) -> Vec<VideoObjectProxy> {
        let inner = self.inner.read_recursive();
        let resident_objects = inner.resident_objects.clone();
        drop(inner);

        if self.is_parallelized {
            resident_objects
                .par_iter()
                .filter_map(|(_, o)| {
                    let obj = VideoObjectProxy::from_arced_inner_object(o.clone());
                    if q.execute_with_new_context(&obj) {
                        Some(obj)
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            resident_objects
                .iter()
                .filter_map(|(_, o)| {
                    let obj = VideoObjectProxy::from_arced_inner_object(o.clone());
                    if q.execute_with_new_context(&obj) {
                        Some(obj)
                    } else {
                        None
                    }
                })
                .collect()
        }
    }

    pub fn get_json(&self) -> String {
        serde_json::to_string(&self.to_serde_json_value()).unwrap()
    }

    pub fn get_json_pretty(&self) -> String {
        serde_json::to_string_pretty(&self.to_serde_json_value()).unwrap()
    }

    pub fn access_objects_by_id(&self, ids: &[i64]) -> Vec<VideoObjectProxy> {
        let inner = self.inner.read_recursive();
        let resident_objects = inner.resident_objects.clone();
        drop(inner);

        ids.iter()
            .flat_map(|id| {
                let o = resident_objects
                    .get(id)
                    .map(|o| VideoObjectProxy::from_arced_inner_object(o.clone()));
                o
            })
            .collect()
    }

    pub fn delete_objects_by_ids(&self, ids: &[i64]) -> Vec<VideoObjectProxy> {
        self.clear_parent(&MatchQuery::ParentId(IntExpression::OneOf(ids.to_vec())));
        let mut inner = self.inner.write();
        let objects = mem::take(&mut inner.resident_objects);
        let (retained, removed) = objects.into_iter().partition(|(id, _)| !ids.contains(id));
        inner.resident_objects = retained;
        drop(inner);

        removed
            .into_values()
            .map(|o| {
                let o = VideoObjectProxy::from_arced_inner_object(o);
                o.detached_copy()
            })
            .collect()
    }

    pub fn object_exists(&self, id: i64) -> bool {
        let inner = self.inner.read_recursive();
        inner.resident_objects.contains_key(&id)
    }

    pub fn delete_objects(&self, q: &MatchQuery) -> Vec<VideoObjectProxy> {
        let objs = self.access_objects(q);
        let ids = objs.iter().map(|o| o.get_id()).collect::<Vec<_>>();
        self.delete_objects_by_ids(&ids)
    }

    pub fn get_object(&self, id: i64) -> Option<VideoObjectProxy> {
        let inner = self.inner.read_recursive();
        inner
            .resident_objects
            .get(&id)
            .map(|o| VideoObjectProxy::from_arced_inner_object(o.clone()))
    }

    pub fn make_snapshot(&self) {
        let mut inner = self.inner.write();
        inner.preserve();
    }

    fn fix_object_owned_frame(&self) {
        self.access_objects(&MatchQuery::Idle)
            .iter()
            .for_each(|o| o.attach_to_video_frame(self.clone()));
    }

    pub fn restore_from_snapshot(&self) {
        {
            let inner = self.inner.write();
            let resident_objects = inner.resident_objects.clone();
            drop(inner);

            resident_objects.iter().for_each(|(_, o)| {
                let mut o = o.write();
                o.frame = None
            });

            let mut inner = self.inner.write();
            inner.restore();
        }
        self.fix_object_owned_frame();
    }

    pub fn get_modified_objects(&self) -> Vec<VideoObjectProxy> {
        let inner = self.inner.read_recursive();
        let resident_objects = inner.resident_objects.clone();
        drop(inner);

        resident_objects
            .iter()
            .filter(|(_id, o)| !o.read_recursive().modifications.is_empty())
            .map(|(_id, o)| VideoObjectProxy::from_arced_inner_object(o.clone()))
            .collect()
    }

    pub fn set_draw_label(&self, q: &MatchQuery, label: SetDrawLabelKind) {
        let objects = self.access_objects(q);
        objects.iter().for_each(|o| match &label {
            SetDrawLabelKind::OwnLabel(l) => {
                o.set_draw_label(Some(l.clone()));
            }
            SetDrawLabelKind::ParentLabel(l) => {
                if let Some(p) = o.get_parent().as_ref() {
                    p.set_draw_label(Some(l.clone()));
                }
            }
        });
    }

    pub fn set_parent(&self, q: &MatchQuery, parent: &VideoObjectProxy) -> Vec<VideoObjectProxy> {
        let frame = parent.get_frame();
        assert!(
            frame.is_some() && Arc::ptr_eq(&frame.unwrap().inner, &self.inner),
            "Parent object must be attached to the same frame"
        );
        let objects = self.access_objects(q);
        objects.iter().for_each(|o| {
            o.set_parent(Some(parent.get_id()));
        });

        objects
    }

    pub fn set_parent_by_id(&self, object_id: i64, parent_id: i64) -> anyhow::Result<()> {
        self.get_object(parent_id).ok_or_else(|| {
            anyhow::anyhow!(
                "Parent object with ID {} does not exist in the frame.",
                parent_id
            )
        })?;

        let object = self.get_object(object_id).ok_or_else(|| {
            anyhow::anyhow!("Object with ID {} does not exist in the frame.", object_id)
        })?;

        object.set_parent(Some(parent_id));
        Ok(())
    }

    pub fn clear_parent(&self, q: &MatchQuery) -> Vec<VideoObjectProxy> {
        let objects = self.access_objects(q);
        objects.iter().for_each(|o| {
            o.set_parent(None);
        });
        objects
    }

    pub fn get_children(&self, id: i64) -> Vec<VideoObjectProxy> {
        self.access_objects(&MatchQuery::ParentId(IntExpression::EQ(id)))
    }

    pub fn add_object(
        &self,
        object: &VideoObjectProxy,
        policy: IdCollisionResolutionPolicy,
    ) -> anyhow::Result<()> {
        let parent_id_opt = object.get_parent_id();
        if let Some(parent_id) = parent_id_opt {
            if !self.object_exists(parent_id) {
                bail!(
                    "Parent object with ID {} does not exist in the frame.",
                    parent_id
                );
            }
        }

        if !object.is_detached() {
            bail!("Only detached objects can be attached to a frame.");
        }

        let object_id = object.get_id();
        let new_id = self.get_max_object_id() + 1;
        let mut inner = self.inner.write();
        if inner.resident_objects.contains_key(&object_id) {
            match policy {
                IdCollisionResolutionPolicy::GenerateNewId => {
                    object.set_id(new_id)?;
                    inner.resident_objects.insert(new_id, object.inner.clone());
                }
                IdCollisionResolutionPolicy::Overwrite => {
                    let old = inner.resident_objects.remove(&object_id).unwrap();
                    old.write().frame = None;
                    old.write().parent_id = None;
                    inner
                        .resident_objects
                        .insert(object_id, object.inner.clone());
                }
                IdCollisionResolutionPolicy::Error => {
                    bail!("Object with ID {} already exists in the frame.", object_id);
                }
            }
        } else {
            inner
                .resident_objects
                .insert(object_id, object.inner.clone());
        }

        object.attach_to_video_frame(self.clone());
        let object_id = object.get_id();
        if object_id > inner.max_object_id {
            inner.max_object_id = object_id;
        }
        Ok(())
    }

    pub fn get_max_object_id(&self) -> i64 {
        let inner = self.inner.read_recursive();
        inner.max_object_id
    }

    pub fn update_objects(&self, update: &VideoFrameUpdate) -> anyhow::Result<()> {
        use frame_update::ObjectUpdateCollisionResolutionPolicy::*;
        let other_inner = update.objects.clone();

        let object_query = |o: &VideoObject| {
            crate::primitives::message::video::query::and![
                MatchQuery::Label(StringExpression::EQ(o.label.clone())),
                MatchQuery::Namespace(StringExpression::EQ(o.namespace.clone()))
            ]
        };

        match &update.object_collision_resolution_policy {
            AddForeignObjects => {
                for (mut obj, p) in other_inner {
                    let object_id = self.get_max_object_id() + 1;
                    obj.id = object_id;

                    self.add_object(
                        &VideoObjectProxy::from_video_object(obj),
                        IdCollisionResolutionPolicy::GenerateNewId,
                    )?;
                    if let Some(p) = p {
                        self.set_parent_by_id(object_id, p)?;
                    }
                }
            }
            ErrorIfLabelsCollide => {
                for (mut obj, p) in other_inner {
                    let objs = self.access_objects(&object_query(&obj));
                    if !objs.is_empty() {
                        bail!(
                            "Objects with label '{}' and namespace '{}' already exists in the frame.",
                            obj.label,
                            obj.namespace
                        )
                    }

                    let object_id = self.get_max_object_id() + 1;
                    obj.id = object_id;

                    self.add_object(
                        &VideoObjectProxy::from_video_object(obj),
                        IdCollisionResolutionPolicy::GenerateNewId,
                    )?;
                    if let Some(p) = p {
                        self.set_parent_by_id(object_id, p)?;
                    }
                }
            }
            ReplaceSameLabelObjects => {
                for (mut obj, p) in other_inner {
                    self.delete_objects(&object_query(&obj));

                    let object_id = self.get_max_object_id() + 1;
                    obj.id = object_id;

                    self.add_object(
                        &VideoObjectProxy::from_video_object(obj),
                        IdCollisionResolutionPolicy::GenerateNewId,
                    )?;

                    if let Some(p) = p {
                        self.set_parent_by_id(object_id, p)?;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn update_attributes(&self, update: &VideoFrameUpdate) -> anyhow::Result<()> {
        use frame_update::AttributeUpdateCollisionResolutionPolicy::*;
        match &update.attribute_collision_resolution_policy {
            ReplaceWithForeignWhenDuplicate => {
                let mut inner = self.inner.write();
                let other_inner = update.attributes.clone();
                inner.attributes.extend(
                    other_inner
                        .into_iter()
                        .map(|a| ((a.namespace.clone(), a.name.clone()), a)),
                );
            }
            KeepOwnWhenDuplicate => {
                let mut inner = self.inner.write();
                let other_inner = update.attributes.clone();
                for attr in other_inner {
                    let key = (attr.namespace.clone(), attr.name.clone());
                    inner.attributes.entry(key).or_insert(attr);
                }
            }
            ErrorWhenDuplicate => {
                let mut inner = self.inner.write();
                let other_inner = update.attributes.clone();
                for attr in other_inner {
                    let key = (attr.namespace.clone(), attr.name.clone());
                    if inner.attributes.contains_key(&key) {
                        anyhow::bail!(
                            "Attribute with name '{}' created by '{}' already exists in the frame.",
                            key.1,
                            key.0
                        );
                    }
                    inner.attributes.insert(key, attr);
                }
            }
            PrefixDuplicates(prefix) => {
                let mut inner = self.inner.write();
                let other_inner = update.attributes.clone();
                for attr in other_inner {
                    let key = (attr.namespace.clone(), attr.name.clone());
                    if inner.attributes.contains_key(&key) {
                        let mut new_key = key.clone();
                        new_key.1 = format!("{}{}", prefix, new_key.1);
                        inner.attributes.insert(new_key, attr);
                    } else {
                        inner.attributes.insert(key, attr);
                    }
                }
            }
        }

        Ok(())
    }

    pub fn update(&self, update: &VideoFrameUpdate) -> anyhow::Result<()> {
        self.update_objects(update)?;
        self.update_attributes(update)?;
        Ok(())
    }
}

impl ToSerdeJsonValue for VideoFrameProxy {
    fn to_serde_json_value(&self) -> Value {
        let inner = self.inner.read_recursive().clone();
        inner.to_serde_json_value()
    }
}

#[pymethods]
impl VideoFrameProxy {
    #[pyo3(name = "transform_geometry")]
    fn transform_geometry_gil(&self, ops: Vec<VideoObjectBBoxTransformationProxy>) {
        release_gil(|| {
            let ops_ref = ops.iter().map(|op| op.get_ref()).collect();
            self.transform_geometry(&ops_ref);
        })
    }

    #[setter]
    pub fn set_parallelized(&mut self, is_parallelized: bool) {
        self.is_parallelized = is_parallelized;
    }

    #[getter]
    pub fn get_parallelized(&self) -> bool {
        self.is_parallelized
    }

    #[getter]
    fn memory_handle(&self) -> usize {
        self as *const Self as usize
    }

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
    #[pyo3(
        signature = (source_id, framerate, width, height, content, transcoding_method=VideoFrameTranscodingMethod::Copy, codec=None, keyframe=None, time_base=(1, 1000000), pts=0, dts=None, duration=None)
    )]
    pub fn new(
        source_id: String,
        framerate: String,
        width: i64,
        height: i64,
        content: VideoFrameContentProxy,
        transcoding_method: VideoFrameTranscodingMethod,
        codec: Option<String>,
        keyframe: Option<bool>,
        time_base: (i64, i64),
        pts: i64,
        dts: Option<i64>,
        duration: Option<i64>,
    ) -> Self {
        VideoFrameProxy::from_inner(VideoFrame {
            source_id,
            pts,
            framerate,
            width,
            height,
            time_base: (time_base.0 as i32, time_base.1 as i32),
            dts,
            duration,
            transcoding_method,
            codec,
            keyframe,
            content: content.inner,
            ..Default::default()
        })
    }

    pub fn to_message(&self) -> Message {
        Message::video_frame(self.clone())
    }

    #[getter]
    pub fn get_source_id(&self) -> String {
        self.inner.read_recursive().source_id.clone()
    }

    #[getter]
    #[pyo3(name = "json")]
    pub fn json_gil(&self) -> String {
        release_gil(|| serde_json::to_string(&self.to_serde_json_value()).unwrap())
    }

    #[getter]
    #[pyo3(name = "json_pretty")]
    fn json_pretty_gil(&self) -> String {
        release_gil(|| serde_json::to_string_pretty(&self.to_serde_json_value()).unwrap())
    }

    #[setter]
    pub fn set_source_id(&mut self, source_id: String) {
        let mut inner = self.inner.write();
        inner.source_id = source_id;
    }

    #[setter]
    pub fn set_time_base(&mut self, time_base: (i32, i32)) {
        let mut inner = self.inner.write();
        inner.time_base = time_base;
    }

    #[getter]
    pub fn get_time_base(&self) -> (i32, i32) {
        self.inner.read_recursive().time_base
    }

    #[getter]
    pub fn get_pts(&self) -> i64 {
        self.inner.read_recursive().pts
    }

    #[setter]
    pub fn set_pts(&mut self, pts: i64) {
        assert!(pts >= 0, "pts must be greater than or equal to 0");
        let mut inner = self.inner.write();
        inner.pts = pts;
    }

    #[getter]
    pub fn get_framerate(&self) -> String {
        self.inner.read_recursive().framerate.clone()
    }

    #[setter]
    pub fn set_framerate(&mut self, framerate: String) {
        let mut inner = self.inner.write();
        inner.framerate = framerate;
    }

    #[getter]
    pub fn get_width(&self) -> i64 {
        self.inner.read_recursive().width
    }

    #[setter]
    pub fn set_width(&mut self, width: i64) {
        assert!(width > 0, "width must be greater than 0");
        let mut inner = self.inner.write();
        inner.width = width;
    }

    #[getter]
    pub fn get_height(&self) -> i64 {
        self.inner.read_recursive().height
    }

    #[setter]
    pub fn set_height(&mut self, height: i64) {
        assert!(height > 0, "height must be greater than 0");
        let mut inner = self.inner.write();
        inner.height = height;
    }

    #[getter]
    pub fn get_dts(&self) -> Option<i64> {
        let inner = self.inner.read_recursive();
        inner.dts
    }

    #[setter]
    pub fn set_dts(&mut self, dts: Option<i64>) {
        assert!(
            dts.is_none() || dts.unwrap() >= 0,
            "dts must be greater than or equal to 0"
        );
        let mut inner = self.inner.write();
        inner.dts = dts;
    }

    #[getter]
    pub fn get_duration(&self) -> Option<i64> {
        let inner = self.inner.read_recursive();
        inner.duration
    }

    #[setter]
    pub fn set_duration(&mut self, duration: Option<i64>) {
        assert!(
            duration.is_none() || duration.unwrap() >= 0,
            "duration must be greater than or equal to 0"
        );
        let mut inner = self.inner.write();
        inner.duration = duration;
    }

    #[getter]
    pub fn get_transcoding_method(&self) -> VideoFrameTranscodingMethod {
        let inner = self.inner.read_recursive();
        inner.transcoding_method.clone()
    }

    #[setter]
    pub fn set_transcoding_method(&mut self, transcoding_method: VideoFrameTranscodingMethod) {
        let mut inner = self.inner.write();
        inner.transcoding_method = transcoding_method;
    }

    #[getter]
    pub fn get_codec(&self) -> Option<String> {
        let inner = self.inner.read_recursive();
        inner.codec.clone()
    }

    #[setter]
    pub fn set_codec(&mut self, codec: Option<String>) {
        let mut inner = self.inner.write();
        inner.codec = codec;
    }

    pub fn clear_transformations(&mut self) {
        let mut inner = self.inner.write();
        inner.transformations.clear();
    }

    pub fn add_transformation(&mut self, transformation: &VideoFrameTransformationProxy) {
        let mut inner = self.inner.write();
        inner.transformations.push(transformation.inner.clone());
    }

    #[getter]
    pub fn get_transformations(&self) -> Vec<VideoFrameTransformationProxy> {
        let inner = self.inner.read_recursive();
        inner
            .transformations
            .iter()
            .map(|t| VideoFrameTransformationProxy::new(t.clone()))
            .collect()
    }

    #[getter]
    pub fn get_keyframe(&self) -> Option<bool> {
        let inner = self.inner.read_recursive();
        inner.keyframe
    }

    #[setter]
    pub fn set_keyframe(&mut self, keyframe: Option<bool>) {
        let mut inner = self.inner.write();
        inner.keyframe = keyframe;
    }

    #[getter]
    pub fn get_content(&self) -> VideoFrameContentProxy {
        let inner = self.inner.read_recursive();
        VideoFrameContentProxy::new_arced(inner.content.clone())
    }

    #[setter]
    pub fn set_content(&mut self, content: VideoFrameContentProxy) {
        let mut inner = self.inner.write();
        inner.content = content.inner;
    }

    #[getter]
    #[pyo3(name = "attributes")]
    pub fn attributes_gil(&self) -> Vec<(String, String)> {
        release_gil(|| self.get_attributes())
    }

    #[pyo3(name = "find_attributes")]
    #[pyo3(signature = (namespace=None, names=vec![], hint=None))]
    pub fn find_attributes_gil(
        &self,
        namespace: Option<String>,
        names: Vec<String>,
        hint: Option<String>,
    ) -> Vec<(String, String)> {
        release_gil(|| self.find_attributes(namespace, names, hint))
    }

    #[pyo3(name = "get_attribute")]
    pub fn get_attribute_gil(&self, namespace: String, name: String) -> Option<Attribute> {
        release_gil(|| self.get_attribute(namespace, name))
    }

    #[pyo3(signature = (namespace=None, names=vec![]))]
    #[pyo3(name = "delete_attributes")]
    pub fn delete_attributes_gil(&mut self, namespace: Option<String>, names: Vec<String>) {
        release_gil(|| self.delete_attributes(namespace, names))
    }

    #[pyo3(name = "add_object")]
    pub fn add_object_py(
        &self,
        o: VideoObjectProxy,
        policy: IdCollisionResolutionPolicy,
    ) -> PyResult<()> {
        release_gil(|| self.add_object(&o, policy))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(name = "delete_attribute")]
    pub fn delete_attribute_gil(&mut self, namespace: String, name: String) -> Option<Attribute> {
        release_gil(|| self.delete_attribute(namespace, name))
    }

    #[pyo3(name = "set_attribute")]
    pub fn set_attribute_gil(&mut self, attribute: Attribute) -> Option<Attribute> {
        self.set_attribute(attribute)
    }

    #[pyo3(name = "clear_attributes")]
    pub fn clear_attributes_gil(&mut self) {
        release_gil(|| self.clear_attributes())
    }

    #[pyo3(name = "set_draw_label")]
    pub fn set_draw_label_gil(&self, q: &MatchQueryProxy, draw_label: SetDrawLabelKindProxy) {
        release_gil(|| self.set_draw_label(q.inner.deref(), draw_label.inner))
    }

    #[pyo3(name = "get_object")]
    pub fn get_object_gil(&self, id: i64) -> Option<VideoObjectProxy> {
        release_gil(|| self.get_object(id))
    }

    #[pyo3(name = "access_objects")]
    pub fn access_objects_gil(&self, q: &MatchQueryProxy) -> VideoObjectsView {
        release_gil(|| self.access_objects(q.inner.deref()).into())
    }

    #[pyo3(name = "access_objects_by_id")]
    pub fn access_objects_by_id_gil(&self, ids: Vec<i64>) -> VideoObjectsView {
        release_gil(|| self.access_objects_by_id(&ids).into())
    }

    #[pyo3(name = "delete_objects_by_ids")]
    pub fn delete_objects_by_ids_gil(&self, ids: Vec<i64>) -> VideoObjectsView {
        release_gil(|| self.delete_objects_by_ids(&ids).into())
    }

    #[pyo3(name = "delete_objects")]
    pub fn delete_objects_gil(&self, query: &MatchQueryProxy) -> VideoObjectsView {
        release_gil(|| self.delete_objects(&query.inner).into())
    }

    #[pyo3(name = "set_parent")]
    pub fn set_parent_gil(
        &self,
        q: &MatchQueryProxy,
        parent: &VideoObjectProxy,
    ) -> VideoObjectsView {
        release_gil(|| self.set_parent(q.inner.deref(), parent).into())
    }

    #[pyo3(name = "set_parent_by_id")]
    pub fn set_parent_by_id_gil(&self, object_id: i64, parent_id: i64) -> PyResult<()> {
        release_gil(|| self.set_parent_by_id(object_id, parent_id))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(name = "clear_parent")]
    pub fn clear_parent_gil(&self, q: &MatchQueryProxy) -> VideoObjectsView {
        release_gil(|| self.clear_parent(q.inner.deref()).into())
    }

    pub fn clear_objects(&self) {
        let mut frame = self.inner.write();
        frame.resident_objects.clear();
    }

    #[pyo3(name = "make_snapshot")]
    pub fn make_snapshot_gil(&self) {
        release_gil(|| self.make_snapshot())
    }

    #[pyo3(name = "restore_from_snapshot")]
    pub fn restore_from_snapshot_gil(&self) {
        release_gil(|| self.restore_from_snapshot())
    }

    #[pyo3(name = "get_modified_objects")]
    pub fn get_modified_objects_gil(&self) -> VideoObjectsView {
        release_gil(|| self.get_modified_objects().into())
    }

    #[pyo3(name = "get_children")]
    pub fn get_children_gil(&self, id: i64) -> VideoObjectsView {
        release_gil(|| self.get_children(id).into())
    }

    #[pyo3(name = "copy")]
    pub fn copy_gil(&self) -> VideoFrameProxy {
        release_gil(|| self.deep_copy())
    }

    #[pyo3(name = "update_attributes")]
    pub fn update_attributes_gil(&self, other: &VideoFrameUpdate) -> PyResult<()> {
        release_gil(|| self.update_attributes(other))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(name = "update_objects")]
    pub fn update_objects_gil(&self, other: &VideoFrameUpdate) -> PyResult<()> {
        release_gil(|| self.update_objects(other)).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(name = "update")]
    pub fn update_gil(&self, other: &VideoFrameUpdate) -> PyResult<()> {
        release_gil(|| self.update(other)).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn create_objects_from_numpy(&self, namespace: String, boxes: &PyAny) -> PyResult<()> {
        fn check_shape(shape: &[usize]) -> PyResult<Option<usize>> {
            if shape.len() != 2 {
                return Err(PyValueError::new_err("Array must have 2 dimensions"));
            }

            let col_number = shape[1];

            if col_number == 7 {
                Ok(Some(6))
            } else if col_number == 6 {
                Ok(None)
            } else {
                Err(PyValueError::new_err(
                    "Array must have 6 or 7 columns (class_id, conf, xc, yc, width, height [, angle])",
                ))
            }
        }

        let boxes: Vec<_> = if let Ok(arr) = boxes.downcast::<PyArray<f32, IxDyn>>() {
            let shape = arr.shape();
            let angle_col = check_shape(shape)?;
            let ro_binding = arr.readonly();
            let ro_array = ro_binding.as_array();
            ro_array
                .rows()
                .into_iter()
                .map(|r| {
                    let class_id = r[0] as i64;
                    let conf = r[1] as f64;
                    let angle = angle_col.map(|c| r[c] as f64);
                    let bbox =
                        RBBox::new(r[2] as f64, r[3] as f64, r[4] as f64, r[5] as f64, angle);
                    (class_id, conf, bbox)
                })
                .collect()
        } else if let Ok(arr) = boxes.downcast::<PyArray<f64, IxDyn>>() {
            let shape = arr.shape();
            let angle_col = check_shape(shape)?;
            let ro_binding = arr.readonly();
            let ro_array = ro_binding.as_array();
            ro_array
                .rows()
                .into_iter()
                .map(|r| {
                    let class_id = r[0] as i64;
                    let conf = r[1];
                    let angle = angle_col.map(|c| r[c]);
                    let bbox = RBBox::new(r[2], r[3], r[4], r[5], angle);
                    (class_id, conf, bbox)
                })
                .collect()
        } else {
            return Err(PyValueError::new_err("Array must be of type f32 or f64"));
        };

        release_gil(|| {
            let model_id = get_model_id(&namespace).map_err(|e| {
                PyValueError::new_err(format!("Failed to get model id: {}", e.to_string()))
            })?;

            boxes.into_iter().try_for_each(|(cls_id, conf, b)| {
                let label = get_object_label(model_id, cls_id);

                match label {
                    None => Err(PyValueError::new_err(format!(
                        "Failed to get object label for model={} (id={}): cls_id={}",
                        &namespace, model_id, cls_id
                    ))),

                    Some(l) => {
                        let object = VideoObjectProxy::new(
                            0,
                            namespace.clone(),
                            l,
                            b,
                            HashMap::default(),
                            Some(conf),
                            None,
                            None,
                        );
                        self.add_object(&object, IdCollisionResolutionPolicy::GenerateNewId)
                            .map_err(|e| {
                                PyValueError::new_err(format!(
                                    "Failed to add object: {}",
                                    e.to_string()
                                ))
                            })
                    }
                }
            })?;

            Ok(())
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
    use crate::primitives::attribute::AttributeMethods;
    use crate::primitives::message::video::object::VideoObjectBuilder;
    use crate::primitives::message::video::query::match_query::MatchQuery;
    use crate::primitives::message::video::query::{eq, one_of};
    use crate::primitives::{
        IdCollisionResolutionPolicy, RBBox, SetDrawLabelKind, VideoObjectModification,
        VideoObjectProxy,
    };
    use crate::test::utils::{gen_empty_frame, gen_frame, gen_object, s};
    use std::sync::Arc;

    #[test]
    fn test_access_objects_by_id() {
        pyo3::prepare_freethreaded_python();
        let t = gen_frame();
        let objects = t.access_objects_by_id(&vec![0, 1]);
        assert_eq!(objects.len(), 2);
        assert_eq!(objects[0].get_id(), 0);
        assert_eq!(objects[1].get_id(), 1);
    }

    #[test]
    fn test_objects_by_id() {
        pyo3::prepare_freethreaded_python();
        let t = gen_frame();
        let objects = t.access_objects_by_id(&vec![0, 1]);
        assert_eq!(objects.len(), 2);
        assert_eq!(objects[0].get_id(), 0);
        assert_eq!(objects[1].get_id(), 1);
    }

    #[test]
    fn test_get_attribute() {
        pyo3::prepare_freethreaded_python();
        let t = gen_frame();
        let attribute = t.get_attribute("system".to_string(), "test".to_string());
        assert!(attribute.is_some());
        assert_eq!(
            attribute.unwrap().values[0].as_string().unwrap(),
            "1".to_string()
        );
    }

    #[test]
    fn test_find_attributes() {
        pyo3::prepare_freethreaded_python();
        let t = gen_frame();
        let mut attributes = t.find_attributes_gil(Some("system".to_string()), vec![], None);
        attributes.sort();
        assert_eq!(attributes.len(), 2);
        assert_eq!(attributes[0], ("system".to_string(), "test".to_string()));
        assert_eq!(attributes[1], ("system".to_string(), "test2".to_string()));

        let attributes =
            t.find_attributes_gil(Some("system".to_string()), vec!["test".to_string()], None);
        assert_eq!(attributes.len(), 1);
        assert_eq!(attributes[0], ("system".to_string(), "test".to_string()));

        let attributes = t.find_attributes_gil(
            Some("system".to_string()),
            vec!["test".to_string()],
            Some("test".to_string()),
        );
        assert_eq!(attributes.len(), 1);
        assert_eq!(attributes[0], ("system".to_string(), "test".to_string()));

        let mut attributes = t.find_attributes_gil(None, vec![], Some("test".to_string()));
        attributes.sort();
        assert_eq!(attributes.len(), 2);
        assert_eq!(attributes[0], ("system".to_string(), "test".to_string()));
        assert_eq!(attributes[1], ("system".to_string(), "test2".to_string()));
    }

    #[test]
    fn test_delete_objects_by_ids() {
        pyo3::prepare_freethreaded_python();
        let f = gen_frame();
        f.delete_objects_by_ids(&[0, 1]);
        let objects = f.access_objects(&MatchQuery::Idle);
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].get_id(), 2);
    }

    #[test]
    fn test_parent_cleared_when_delete_objects_by_ids() {
        let f = gen_frame();
        f.delete_objects_by_ids(&[0]);
        let o = f.get_object(1).unwrap();
        assert!(o.get_parent().is_none());
        let o = f.get_object(2).unwrap();
        assert!(o.get_parent().is_none());

        let f = gen_frame();
        f.delete_objects_by_ids(&[1]);
        let o = f.get_object(2).unwrap();
        assert!(o.get_parent().is_some());
    }

    #[test]
    fn test_parent_cleared_when_delete_objects_by_query() {
        let f = gen_frame();

        let o = f.get_object(0).unwrap();
        assert!(o.get_frame().is_some());

        let removed = f.delete_objects(&MatchQuery::Id(eq(0)));
        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0].get_id(), 0);
        assert!(removed[0].get_frame().is_none());

        let o = f.get_object(1).unwrap();
        assert!(o.get_parent().is_none());

        let o = f.get_object(2).unwrap();
        assert!(o.get_parent().is_none());
    }

    #[test]
    fn test_delete_all_objects() {
        pyo3::prepare_freethreaded_python();
        let f = gen_frame();
        let objs = f.delete_objects(&MatchQuery::Idle);
        assert_eq!(objs.len(), 3);
        let objects = f.access_objects(&MatchQuery::Idle);
        assert!(objects.is_empty());
    }

    #[test]
    fn test_snapshot_simple() {
        let f = gen_frame();
        f.make_snapshot_gil();
        let o = f.access_objects_by_id(&vec![0]).pop().unwrap();
        o.set_namespace(s("modified"));
        f.restore_from_snapshot_gil();
        let o = f.access_objects_by_id(&vec![0]).pop().unwrap();
        assert_eq!(o.get_namespace(), s("test"));
    }

    #[test]
    fn test_modified_objects() {
        let t = gen_frame();
        let o = t.access_objects_by_id(&vec![0]).pop().unwrap();
        o.set_namespace(s("modified"));
        let mut modified = t.get_modified_objects();
        assert_eq!(modified.len(), 1);
        let modified = modified.pop().unwrap();
        assert_eq!(modified.get_namespace(), s("modified"));

        let mods = modified.take_modifications();
        assert_eq!(mods.len(), 1);
        assert_eq!(mods, vec![VideoObjectModification::Namespace]);

        let modified = t.get_modified_objects();
        assert!(modified.is_empty());
    }

    #[test]
    #[should_panic]
    fn test_panic_snapshot_no_parent_added_to_frame() {
        let parent = VideoObjectProxy::from_video_object(
            VideoObjectBuilder::default()
                .parent_id(None)
                .namespace(s("some-model"))
                .label(s("some-label"))
                .id(155)
                .detection_box(RBBox::new(0.0, 0.0, 0.0, 0.0, None).try_into().unwrap())
                .build()
                .unwrap(),
        );
        let frame = gen_frame();
        let obj = frame.get_object(0).unwrap();
        obj.set_parent(Some(parent.get_id()));
        frame.make_snapshot();
    }

    #[test]
    fn test_snapshot_with_parent_added_to_frame() {
        let parent = VideoObjectProxy::from_video_object(
            VideoObjectBuilder::default()
                .parent_id(None)
                .namespace(s("some-model"))
                .label(s("some-label"))
                .id(155)
                .detection_box(RBBox::new(0.0, 0.0, 0.0, 0.0, None).try_into().unwrap())
                .build()
                .unwrap(),
        );
        let frame = gen_frame();
        frame
            .add_object(&parent, IdCollisionResolutionPolicy::Error)
            .unwrap();
        let obj = frame.get_object(0).unwrap();
        obj.set_parent(Some(parent.get_id()));
        frame.make_snapshot();
        frame.restore_from_snapshot();
        let obj = frame.get_object(0).unwrap();
        assert_eq!(obj.get_parent().unwrap().inner.read_recursive().id, 155);
    }

    #[test]
    fn test_no_children() {
        let frame = gen_frame();
        let obj = frame.get_object(2).unwrap();
        assert!(frame.get_children(obj.get_id()).is_empty());
    }

    #[test]
    fn test_two_children() {
        let frame = gen_frame();
        let obj = frame.get_object(0).unwrap();
        assert_eq!(frame.get_children(obj.get_id()).len(), 2);
    }

    #[test]
    fn set_parent_draw_label() {
        let frame = gen_frame();
        frame.set_draw_label(&MatchQuery::Idle, SetDrawLabelKind::ParentLabel(s("draw")));
        let parent_object = frame.get_object(0).unwrap();
        assert_eq!(parent_object.get_draw_label(), s("draw"));

        let child_object = frame.get_object(1).unwrap();
        assert_ne!(child_object.get_draw_label(), s("draw"));
    }

    #[test]
    fn set_own_draw_label() {
        let frame = gen_frame();
        frame.set_draw_label(&MatchQuery::Idle, SetDrawLabelKind::OwnLabel(s("draw")));
        let parent_object = frame.get_object(0).unwrap();
        assert_eq!(parent_object.get_draw_label(), s("draw"));

        let child_object = frame.get_object(1).unwrap();
        assert_eq!(child_object.get_draw_label(), s("draw"));

        let child_object = frame.get_object(2).unwrap();
        assert_eq!(child_object.get_draw_label(), s("draw"));
    }

    #[test]
    fn test_set_clear_parent_ops() {
        let frame = gen_frame();
        let parent = frame.get_object(0).unwrap();
        frame.clear_parent(&MatchQuery::Id(one_of(&[1, 2])));
        let obj = frame.get_object(1).unwrap();
        assert!(obj.get_parent().is_none());
        let obj = frame.get_object(2).unwrap();
        assert!(obj.get_parent().is_none());

        frame.set_parent(&MatchQuery::Id(one_of(&[1, 2])), &parent);
        let obj = frame.get_object(1).unwrap();
        assert!(obj.get_parent().is_some());

        let obj = frame.get_object(2).unwrap();
        assert!(obj.get_parent().is_some());
    }

    #[test]
    fn retrieve_children() {
        let frame = gen_frame();
        let parent = frame.get_object(0).unwrap();
        let children = frame.get_children(parent.get_id());
        assert_eq!(children.len(), 2);
    }

    #[test]
    #[should_panic]
    fn attach_object_with_detached_parent() {
        pyo3::prepare_freethreaded_python();
        let p = VideoObjectProxy::from_video_object(
            VideoObjectBuilder::default()
                .id(11)
                .namespace(s("random"))
                .label(s("something"))
                .detection_box(RBBox::new(1.0, 2.0, 10.0, 20.0, None).try_into().unwrap())
                .build()
                .unwrap(),
        );

        let o = VideoObjectProxy::from_video_object(
            VideoObjectBuilder::default()
                .id(23)
                .namespace(s("random"))
                .label(s("something"))
                .detection_box(RBBox::new(1.0, 2.0, 10.0, 20.0, None).try_into().unwrap())
                .parent_id(Some(p.get_id()))
                .build()
                .unwrap(),
        );

        let f = gen_frame();
        f.add_object(&o, IdCollisionResolutionPolicy::Error)
            .unwrap();
    }

    #[test]
    #[should_panic]
    fn set_detached_parent_as_parent() {
        let f = gen_frame();
        let o = VideoObjectProxy::from_video_object(
            VideoObjectBuilder::default()
                .id(11)
                .namespace(s("random"))
                .label(s("something"))
                .detection_box(RBBox::new(1.0, 2.0, 10.0, 20.0, None).try_into().unwrap())
                .build()
                .unwrap(),
        );
        f.set_parent(&MatchQuery::Id(eq(0)), &o);
    }

    #[test]
    #[should_panic]
    fn set_wrong_parent_as_parent() {
        let f1 = gen_frame();
        let f2 = gen_frame();
        let f1o = f1.get_object(0).unwrap();
        f2.set_parent(&MatchQuery::Id(eq(1)), &f1o);
    }

    #[test]
    fn normally_transfer_parent() {
        let f1 = gen_frame();
        let f2 = gen_frame();
        let mut o = f1.delete_objects_by_ids(&[0]).pop().unwrap();
        assert!(o.get_frame().is_none());
        _ = o.set_id(33);
        f2.add_object(&o, IdCollisionResolutionPolicy::Error)
            .unwrap();
        o = f2.get_object(33).unwrap();
        f2.set_parent(&MatchQuery::Id(eq(1)), &o);
    }

    #[test]
    fn frame_is_properly_set_after_snapshotting() {
        let frame = gen_frame();
        frame.make_snapshot();
        frame.restore_from_snapshot();
        let o = frame.get_object(0).unwrap();
        let saved_frame = o.get_frame();
        assert!(saved_frame.is_some());
        assert!(Arc::ptr_eq(&frame.inner, &saved_frame.unwrap().inner));
    }

    #[test]
    fn ensure_owned_objects_detached_after_snapshot() {
        let frame = gen_frame();
        frame
            .add_object(&gen_object(111), IdCollisionResolutionPolicy::Error)
            .unwrap();
        frame.make_snapshot();
        let object = frame.get_object(111).unwrap();
        assert!(!object.is_detached(), "Object is expected to be attached");

        frame.restore_from_snapshot();
        assert!(object.is_detached(), "Object is expected to be detached");

        let o = frame.get_object(0).unwrap();
        assert!(!o.is_detached(), "Object is expected to be attached");
    }

    #[test]
    fn ensure_object_spoiled_when_frame_is_dropped() {
        let frame = gen_frame();
        let object = frame.get_object(0).unwrap();
        assert!(
            !object.is_spoiled(),
            "Object is expected to be in a normal state."
        );
        drop(frame);
        assert!(object.is_spoiled(), "Object is expected to be spoiled");
    }

    #[test]
    #[should_panic(expected = "Only detached objects can be attached to a frame.")]
    fn ensure_spoiled_object_cannot_be_added() {
        let frame = gen_frame();
        frame
            .add_object(&gen_object(111), IdCollisionResolutionPolicy::Error)
            .unwrap();
        let old_object = frame.get_object(111).unwrap();
        drop(frame);
        let frame = gen_frame();
        assert!(old_object.is_spoiled(), "Object is expected to be spoiled");
        frame
            .add_object(&old_object, IdCollisionResolutionPolicy::Error)
            .unwrap();
    }

    #[test]
    fn deleted_objects_clean() {
        let frame = gen_frame();
        let removed = frame.delete_objects_by_ids(&[0]).pop().unwrap();
        assert!(removed.is_detached());
        assert!(removed.get_parent().is_none());
    }

    #[test]
    fn deep_copy() {
        let f = gen_frame();
        let new_f = f.deep_copy();

        // check that objects are copied
        let o = f.get_object(0).unwrap();
        let new_o = new_f.get_object(0).unwrap();
        let label = s("new label");
        o.set_label(label.clone());
        assert_ne!(new_o.get_label(), label);

        // check that attributes are copied
        f.clear_attributes();
        assert!(f.get_attributes().is_empty());
        assert!(!new_f.get_attributes().is_empty());
    }

    #[test]
    fn add_objects_test_policy_error() {
        let frame = gen_empty_frame();

        let object = gen_object(0);
        frame
            .add_object(&object, IdCollisionResolutionPolicy::Error)
            .unwrap();

        let object = gen_object(0);
        assert!(frame
            .add_object(&object, IdCollisionResolutionPolicy::Error)
            .is_err());
    }

    #[test]
    fn add_objects_test_policy_generate_new_id() {
        let frame = gen_empty_frame();

        let object = gen_object(0);
        frame
            .add_object(&object, IdCollisionResolutionPolicy::GenerateNewId)
            .unwrap();

        let object = gen_object(0);
        frame
            .add_object(&object, IdCollisionResolutionPolicy::GenerateNewId)
            .unwrap();
        assert_eq!(frame.get_max_object_id(), 1);
        let objs = frame.access_objects(&MatchQuery::Idle);
        assert_eq!(objs.len(), 2);
    }

    #[test]
    fn add_objects_test_policy_overwrite() {
        let frame = gen_empty_frame();

        let object = gen_object(0);
        frame
            .add_object(&object, IdCollisionResolutionPolicy::Overwrite)
            .unwrap();

        let object = gen_object(0);
        assert!(frame
            .add_object(&object, IdCollisionResolutionPolicy::Overwrite)
            .is_ok());

        assert_eq!(frame.get_max_object_id(), 0);
        let objs = frame.access_objects(&MatchQuery::Idle);
        assert_eq!(objs.len(), 1);
    }
}
