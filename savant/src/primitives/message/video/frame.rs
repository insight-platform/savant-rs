use crate::capi::InferenceObjectMeta;
use crate::primitives::attribute::{AttributeMethods, Attributive};
use crate::primitives::message::video::object::vector::VectorView;
use crate::primitives::message::video::object::InnerObject;
use crate::primitives::message::video::query::py::QueryWrapper;
use crate::primitives::message::video::query::{ExecutableQuery, IntExpression, Query};
use crate::primitives::to_json_value::ToSerdeJsonValue;
use crate::primitives::{Attribute, Message, Object, SetDrawLabelKind, SetDrawLabelKindWrapper};
use crate::utils::python::no_gil;
use derive_builder::Builder;
use parking_lot::RwLock;
use pyo3::{pyclass, pymethods, Py, PyAny, PyResult};
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
pub struct PyVideoFrameContent {
    pub(crate) inner: VideoFrameContent,
}

impl PyVideoFrameContent {
    pub fn new(inner: VideoFrameContent) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyVideoFrameContent {
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
        Self {
            inner: VideoFrameContent::External(ExternalFrame::new(method, location)),
        }
    }

    #[staticmethod]
    pub fn internal(data: Vec<u8>) -> Self {
        Self {
            inner: VideoFrameContent::Internal(data),
        }
    }

    #[staticmethod]
    pub fn none() -> Self {
        Self {
            inner: VideoFrameContent::None,
        }
    }

    #[getter]
    pub fn is_external(&self) -> bool {
        matches!(self.inner, VideoFrameContent::External(_))
    }

    #[getter]
    pub fn is_internal(&self) -> bool {
        matches!(self.inner, VideoFrameContent::Internal(_))
    }

    #[getter]
    pub fn is_none(&self) -> bool {
        matches!(self.inner, VideoFrameContent::None)
    }

    pub fn get_data(&self) -> PyResult<Vec<u8>> {
        match &self.inner {
            VideoFrameContent::Internal(data) => Ok(data.clone()),
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "Video data is not stored internally",
            )),
        }
    }

    pub fn get_method(&self) -> PyResult<String> {
        match &self.inner {
            VideoFrameContent::External(data) => Ok(data.method.clone()),
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "Video data is not stored externally",
            )),
        }
    }

    pub fn get_location(&self) -> PyResult<Option<String>> {
        match &self.inner {
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
pub enum VideoTranscodingMethod {
    Copy,
    Encoded,
}

impl ToSerdeJsonValue for VideoTranscodingMethod {
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
pub struct PyFrameTransformation {
    pub(crate) inner: FrameTransformation,
}

impl PyFrameTransformation {
    pub fn new(inner: FrameTransformation) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyFrameTransformation {
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
pub struct InnerVideoFrame {
    pub source_id: String,
    pub framerate: String,
    pub width: i64,
    pub height: i64,
    pub transcoding_method: VideoTranscodingMethod,
    pub codec: Option<String>,
    pub keyframe: Option<bool>,
    pub pts: i64,
    pub dts: Option<i64>,
    pub duration: Option<i64>,
    pub content: VideoFrameContent,
    pub transformations: Vec<FrameTransformation>,
    pub attributes: HashMap<(String, String), Attribute>,
    pub offline_objects: HashMap<i64, InnerObject>,
    #[with(Skip)]
    #[builder(setter(skip))]
    pub(crate) resident_objects: HashMap<i64, Arc<RwLock<InnerObject>>>,
}

impl Default for InnerVideoFrame {
    fn default() -> Self {
        Self {
            source_id: String::new(),
            framerate: String::new(),
            width: 0,
            height: 0,
            transcoding_method: VideoTranscodingMethod::Copy,
            codec: None,
            keyframe: None,
            pts: 0,
            dts: None,
            duration: None,
            content: VideoFrameContent::None,
            transformations: Vec::new(),
            attributes: HashMap::new(),
            offline_objects: HashMap::new(),
            resident_objects: HashMap::new(),
        }
    }
}

impl ToSerdeJsonValue for InnerVideoFrame {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!(
            {
                "type": "VideoFrame",
                "source_id": self.source_id,
                "framerate": self.framerate,
                "width": self.width,
                "height": self.height,
                "transcoding_method": self.transcoding_method.to_serde_json_value(),
                "codec": self.codec,
                "keyframe": self.keyframe,
                "pts": self.pts,
                "dts": self.dts,
                "duration": self.duration,
                "content": self.content.to_serde_json_value(),
                "transformations": self.transformations.iter().map(|t| t.to_serde_json_value()).collect::<Vec<_>>(),
                "attributes": self.attributes.values().map(|v| v.to_serde_json_value()).collect::<Vec<_>>(),
                "objects": self.resident_objects.values().map(|o| o.read_recursive().to_serde_json_value()).collect::<Vec<_>>(),
            }
        )
    }
}

impl Attributive for Box<InnerVideoFrame> {
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

impl InnerVideoFrame {
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

    #[allow(dead_code)]
    pub(crate) fn restore_with_merge(&mut self) {
        let _objects = mem::take(&mut self.resident_objects);
        self.restore();
        todo!("merge objects")
    }
}

#[pyclass]
#[derive(Debug, Clone)]
#[repr(C)]
pub struct VideoFrame {
    pub(crate) inner: Arc<RwLock<Box<InnerVideoFrame>>>,
}

#[pyclass]
#[derive(Clone)]
pub struct BelongingVideoFrame {
    pub(crate) inner: Weak<RwLock<Box<InnerVideoFrame>>>,
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

impl From<VideoFrame> for BelongingVideoFrame {
    fn from(value: VideoFrame) -> Self {
        Self {
            inner: Arc::downgrade(&value.inner),
        }
    }
}

impl From<BelongingVideoFrame> for VideoFrame {
    fn from(value: BelongingVideoFrame) -> Self {
        Self {
            inner: value
                .inner
                .upgrade()
                .expect("Frame is dropped, you cannot use attached objects anymore"),
        }
    }
}

impl From<&BelongingVideoFrame> for VideoFrame {
    fn from(value: &BelongingVideoFrame) -> Self {
        Self {
            inner: value
                .inner
                .upgrade()
                .expect("Frame is dropped, you cannot use attached objects anymore"),
        }
    }
}

impl AttributeMethods for VideoFrame {
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

impl VideoFrame {
    pub fn get_inner(&self) -> Arc<RwLock<Box<InnerVideoFrame>>> {
        self.inner.clone()
    }

    pub(crate) fn update_from_inference_meta(
        &self,
        _meta: &InferenceObjectMeta,
    ) -> anyhow::Result<()> {
        todo!("To implement the function");
    }

    pub(crate) fn from_inner(inner: InnerVideoFrame) -> Self {
        VideoFrame {
            inner: Arc::new(RwLock::new(Box::new(inner))),
        }
    }

    pub fn access_objects(&self, q: &Query) -> Vec<Object> {
        let inner = self.inner.read_recursive();
        let resident_objects = inner.resident_objects.clone();
        drop(inner);

        resident_objects
            .iter()
            .filter_map(|(_id, o)| {
                if q.execute(&Object::from_arced_inner_object(o.clone())) {
                    Some(Object::from_arced_inner_object(o.clone()))
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn get_json(&self) -> String {
        serde_json::to_string(&self.to_serde_json_value()).unwrap()
    }

    pub fn get_json_pretty(&self) -> String {
        serde_json::to_string_pretty(&self.to_serde_json_value()).unwrap()
    }

    pub fn access_objects_by_id(&self, ids: &[i64]) -> Vec<Object> {
        let inner = self.inner.read_recursive();
        let resident_objects = inner.resident_objects.clone();
        drop(inner);

        ids.iter()
            .flat_map(|id| {
                let o = resident_objects
                    .get(id)
                    .map(|o| Object::from_arced_inner_object(o.clone()));
                o
            })
            .collect()
    }

    pub fn delete_objects_by_ids(&self, ids: &[i64]) -> Vec<Object> {
        self.clear_parent(&Query::ParentId(IntExpression::OneOf(ids.to_vec())));
        let mut inner = self.inner.write();
        let objects = mem::take(&mut inner.resident_objects);
        let (retained, removed) = objects.into_iter().partition(|(id, _)| !ids.contains(id));
        inner.resident_objects = retained;
        drop(inner);

        removed
            .into_values()
            .map(|o| {
                let o = Object::from_arced_inner_object(o);
                o.detached_copy()
            })
            .collect()
    }

    pub fn object_exists(&self, id: i64) -> bool {
        let inner = self.inner.read_recursive();
        inner.resident_objects.contains_key(&id)
    }

    pub fn delete_objects(&self, q: &Query) -> Vec<Object> {
        let objs = self.access_objects(q);
        let ids = objs.iter().map(|o| o.get_id()).collect::<Vec<_>>();
        self.delete_objects_by_ids(&ids)
    }

    pub fn get_object(&self, id: i64) -> Option<Object> {
        let inner = self.inner.read_recursive();
        inner
            .resident_objects
            .get(&id)
            .map(|o| Object::from_arced_inner_object(o.clone()))
    }

    pub fn make_snapshot(&self) {
        let mut inner = self.inner.write();
        inner.preserve();
    }

    fn fix_object_owned_frame(&self) {
        self.access_objects(&Query::Idle)
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

    pub fn get_modified_objects(&self) -> Vec<Object> {
        let inner = self.inner.read_recursive();
        let resident_objects = inner.resident_objects.clone();
        drop(inner);

        resident_objects
            .iter()
            .filter(|(_id, o)| !o.read_recursive().modifications.is_empty())
            .map(|(_id, o)| Object::from_arced_inner_object(o.clone()))
            .collect()
    }

    pub fn set_draw_label(&self, q: &Query, label: SetDrawLabelKind) {
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

    pub fn set_parent(&self, q: &Query, parent: &Object) -> Vec<Object> {
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

    pub fn clear_parent(&self, q: &Query) -> Vec<Object> {
        let objects = self.access_objects(q);
        objects.iter().for_each(|o| {
            o.set_parent(None);
        });
        objects
    }

    pub fn get_children(&self, id: i64) -> Vec<Object> {
        self.access_objects(&Query::ParentId(IntExpression::EQ(id)))
    }

    pub fn add_object(&self, object: &Object) {
        let parent_id_opt = object.get_parent_id();
        if let Some(parent_id) = parent_id_opt {
            assert!(
                self.object_exists(parent_id),
                "Parent object with ID {} does not exist in the frame.",
                parent_id
            );
        }

        let mut inner = self.inner.write();
        assert!(
            object.is_detached(),
            "Only detached objects can be attached to a frame."
        );

        let object_id = object.get_id();
        if inner.resident_objects.contains_key(&object_id) {
            panic!("Object with ID {} already exists in the frame.", object_id);
        }

        object.attach_to_video_frame(self.clone());
        inner
            .resident_objects
            .insert(object_id, object.inner.clone());
    }
}

impl ToSerdeJsonValue for VideoFrame {
    fn to_serde_json_value(&self) -> Value {
        let inner = self.inner.read_recursive().clone();
        inner.to_serde_json_value()
    }
}

#[pymethods]
impl VideoFrame {
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
        signature = (source_id, framerate, width, height, content, transcoding_method=VideoTranscodingMethod::Copy, codec=None, keyframe=None, pts=0, dts=None, duration=None)
    )]
    pub fn new(
        source_id: String,
        framerate: String,
        width: i64,
        height: i64,
        content: PyVideoFrameContent,
        transcoding_method: VideoTranscodingMethod,
        codec: Option<String>,
        keyframe: Option<bool>,
        pts: i64,
        dts: Option<i64>,
        duration: Option<i64>,
    ) -> Self {
        VideoFrame::from_inner(InnerVideoFrame {
            source_id,
            pts,
            framerate,
            width,
            height,
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
        no_gil(|| serde_json::to_string(&self.to_serde_json_value()).unwrap())
    }

    #[getter]
    #[pyo3(name = "json_pretty")]
    fn json_pretty_gil(&self) -> String {
        no_gil(|| serde_json::to_string_pretty(&self.to_serde_json_value()).unwrap())
    }

    #[setter]
    pub fn set_source_id(&mut self, source_id: String) {
        let mut inner = self.inner.write();
        inner.source_id = source_id;
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
    pub fn get_transcoding_method(&self) -> VideoTranscodingMethod {
        let inner = self.inner.read_recursive();
        inner.transcoding_method.clone()
    }

    #[setter]
    pub fn set_transcoding_method(&mut self, transcoding_method: VideoTranscodingMethod) {
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

    pub fn add_transformation(&mut self, transformation: PyFrameTransformation) {
        let mut inner = self.inner.write();
        inner.transformations.push(transformation.inner);
    }

    #[getter]
    pub fn get_transformations(&self) -> Vec<PyFrameTransformation> {
        let inner = self.inner.read_recursive();
        inner
            .transformations
            .iter()
            .map(|t| PyFrameTransformation::new(t.clone()))
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
    pub fn get_content(&self) -> PyVideoFrameContent {
        let inner = self.inner.read_recursive();
        PyVideoFrameContent::new(inner.content.clone())
    }

    #[setter]
    pub fn set_content(&mut self, content: PyVideoFrameContent) {
        let mut inner = self.inner.write();
        inner.content = content.inner;
    }

    #[getter]
    #[pyo3(name = "attributes")]
    pub fn attributes_gil(&self) -> Vec<(String, String)> {
        no_gil(|| self.get_attributes())
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

    #[pyo3(name = "get_attribute")]
    pub fn get_attribute_gil(&self, creator: String, name: String) -> Option<Attribute> {
        no_gil(|| self.get_attribute(creator, name))
    }

    #[pyo3(signature = (negated=false, creator=None, names=vec![]))]
    #[pyo3(name = "delete_attributes")]
    pub fn delete_attributes_gil(
        &mut self,
        negated: bool,
        creator: Option<String>,
        names: Vec<String>,
    ) {
        no_gil(|| self.delete_attributes(negated, creator, names))
    }

    #[pyo3(name = "add_object")]
    pub fn add_object_py(&self, o: Object) {
        no_gil(|| self.add_object(&o))
    }

    #[pyo3(name = "delete_attribute")]
    pub fn delete_attribute_gil(&mut self, creator: String, name: String) -> Option<Attribute> {
        no_gil(|| self.delete_attribute(creator, name))
    }

    #[pyo3(name = "set_attribute")]
    pub fn set_attribute_gil(&mut self, attribute: Attribute) -> Option<Attribute> {
        self.set_attribute(attribute)
    }

    #[pyo3(name = "clear_attributes")]
    pub fn clear_attributes_gil(&mut self) {
        no_gil(|| self.clear_attributes())
    }

    #[pyo3(name = "set_draw_label")]
    pub fn set_draw_label_gil(&self, q: QueryWrapper, draw_label: SetDrawLabelKindWrapper) {
        no_gil(|| self.set_draw_label(q.inner.deref(), draw_label.inner))
    }

    #[pyo3(name = "get_object")]
    pub fn get_object_gil(&self, id: i64) -> Option<Object> {
        no_gil(|| self.get_object(id))
    }

    #[pyo3(name = "access_objects")]
    pub fn access_objects_gil(&self, q: QueryWrapper) -> VectorView {
        no_gil(|| self.access_objects(q.inner.deref()).into())
    }

    #[pyo3(name = "access_objects_by_id")]
    pub fn access_objects_by_id_gil(&self, ids: Vec<i64>) -> VectorView {
        no_gil(|| self.access_objects_by_id(&ids).into())
    }

    #[pyo3(name = "delete_objects_by_ids")]
    pub fn delete_objects_by_ids_gil(&self, ids: Vec<i64>) -> VectorView {
        no_gil(|| self.delete_objects_by_ids(&ids).into())
    }

    #[pyo3(name = "delete_objects")]
    pub fn delete_objects_gil(&self, query: QueryWrapper) -> VectorView {
        no_gil(|| self.delete_objects(&query.inner).into())
    }

    #[pyo3(name = "set_parent")]
    pub fn set_parent_gil(&self, q: QueryWrapper, parent: Object) -> VectorView {
        no_gil(|| self.set_parent(q.inner.deref(), &parent).into())
    }

    #[pyo3(name = "clear_parent")]
    pub fn clear_parent_gil(&self, q: QueryWrapper) -> VectorView {
        no_gil(|| self.clear_parent(q.inner.deref()).into())
    }

    pub fn clear_objects(&self) {
        let mut frame = self.inner.write();
        frame.resident_objects.clear();
    }

    #[pyo3(name = "make_snapshot")]
    pub fn make_snapshot_gil(&self) {
        no_gil(|| self.make_snapshot())
    }

    #[pyo3(name = "restore_from_snapshot")]
    pub fn restore_from_snapshot_gil(&self) {
        no_gil(|| self.restore_from_snapshot())
    }

    #[pyo3(name = "get_modified_objects")]
    pub fn get_modified_objects_gil(&self) -> VectorView {
        no_gil(|| self.get_modified_objects().into())
    }

    #[pyo3(name = "get_children")]
    pub fn get_children_gil(&self, id: i64) -> VectorView {
        no_gil(|| self.get_children(id).into())
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::attribute::AttributeMethods;
    use crate::primitives::message::video::object::InnerObjectBuilder;
    use crate::primitives::message::video::query::{eq, one_of, Query};
    use crate::primitives::{Modification, Object, RBBox, SetDrawLabelKind};
    use crate::test::utils::{gen_frame, gen_object, s};
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
        let mut attributes = t.find_attributes_gil(Some("system".to_string()), None, None);
        attributes.sort();
        assert_eq!(attributes.len(), 2);
        assert_eq!(attributes[0], ("system".to_string(), "test".to_string()));
        assert_eq!(attributes[1], ("system".to_string(), "test2".to_string()));

        let attributes =
            t.find_attributes_gil(Some("system".to_string()), Some("test".to_string()), None);
        assert_eq!(attributes.len(), 1);
        assert_eq!(attributes[0], ("system".to_string(), "test".to_string()));

        let attributes = t.find_attributes_gil(
            Some("system".to_string()),
            Some("test".to_string()),
            Some("test".to_string()),
        );
        assert_eq!(attributes.len(), 1);
        assert_eq!(attributes[0], ("system".to_string(), "test".to_string()));

        let mut attributes = t.find_attributes_gil(None, None, Some("test".to_string()));
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
        let objects = f.access_objects(&Query::Idle);
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

        let removed = f.delete_objects(&Query::Id(eq(0)));
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
        let objs = f.delete_objects(&Query::Idle);
        assert_eq!(objs.len(), 3);
        let objects = f.access_objects(&Query::Idle);
        assert!(objects.is_empty());
    }

    #[test]
    fn test_snapshot_simple() {
        let f = gen_frame();
        f.make_snapshot_gil();
        let o = f.access_objects_by_id(&vec![0]).pop().unwrap();
        o.set_creator(s("modified"));
        f.restore_from_snapshot_gil();
        let o = f.access_objects_by_id(&vec![0]).pop().unwrap();
        assert_eq!(o.get_creator(), s("test"));
    }

    #[test]
    fn test_modified_objects() {
        let t = gen_frame();
        let o = t.access_objects_by_id(&vec![0]).pop().unwrap();
        o.set_creator(s("modified"));
        let mut modified = t.get_modified_objects();
        assert_eq!(modified.len(), 1);
        let modified = modified.pop().unwrap();
        assert_eq!(modified.get_creator(), s("modified"));

        let mods = modified.take_modifications();
        assert_eq!(mods.len(), 1);
        assert_eq!(mods, vec![Modification::Creator]);

        let modified = t.get_modified_objects();
        assert!(modified.is_empty());
    }

    #[test]
    #[should_panic]
    fn test_panic_snapshot_no_parent_added_to_frame() {
        let parent = Object::from_inner_object(
            InnerObjectBuilder::default()
                .parent_id(None)
                .creator(s("some-model"))
                .label(s("some-label"))
                .id(155)
                .bbox(RBBox::new(0.0, 0.0, 0.0, 0.0, None))
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
        let parent = Object::from_inner_object(
            InnerObjectBuilder::default()
                .parent_id(None)
                .creator(s("some-model"))
                .label(s("some-label"))
                .id(155)
                .bbox(RBBox::new(0.0, 0.0, 0.0, 0.0, None))
                .build()
                .unwrap(),
        );
        let frame = gen_frame();
        frame.add_object(&parent);
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
        frame.set_draw_label(&Query::Idle, SetDrawLabelKind::ParentLabel(s("draw")));
        let parent_object = frame.get_object(0).unwrap();
        assert_eq!(parent_object.draw_label(), s("draw"));

        let child_object = frame.get_object(1).unwrap();
        assert_ne!(child_object.draw_label(), s("draw"));
    }

    #[test]
    fn set_own_draw_label() {
        let frame = gen_frame();
        frame.set_draw_label(&Query::Idle, SetDrawLabelKind::OwnLabel(s("draw")));
        let parent_object = frame.get_object(0).unwrap();
        assert_eq!(parent_object.draw_label(), s("draw"));

        let child_object = frame.get_object(1).unwrap();
        assert_eq!(child_object.draw_label(), s("draw"));

        let child_object = frame.get_object(2).unwrap();
        assert_eq!(child_object.draw_label(), s("draw"));
    }

    #[test]
    fn test_set_clear_parent_ops() {
        let frame = gen_frame();
        let parent = frame.get_object(0).unwrap();
        frame.clear_parent(&Query::Id(one_of(&[1, 2])));
        let obj = frame.get_object(1).unwrap();
        assert!(obj.get_parent().is_none());
        let obj = frame.get_object(2).unwrap();
        assert!(obj.get_parent().is_none());

        frame.set_parent(&Query::Id(one_of(&[1, 2])), &parent);
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
        let p = Object::from_inner_object(
            InnerObjectBuilder::default()
                .id(11)
                .creator(s("random"))
                .label(s("something"))
                .bbox(RBBox::new(1.0, 2.0, 10.0, 20.0, None))
                .build()
                .unwrap(),
        );

        let o = Object::from_inner_object(
            InnerObjectBuilder::default()
                .id(23)
                .creator(s("random"))
                .label(s("something"))
                .bbox(RBBox::new(1.0, 2.0, 10.0, 20.0, None))
                .parent_id(Some(p.get_id()))
                .build()
                .unwrap(),
        );

        let f = gen_frame();
        f.add_object(&o);
    }

    #[test]
    #[should_panic]
    fn set_detached_parent_as_parent() {
        let f = gen_frame();
        let o = Object::from_inner_object(
            InnerObjectBuilder::default()
                .id(11)
                .creator(s("random"))
                .label(s("something"))
                .bbox(RBBox::new(1.0, 2.0, 10.0, 20.0, None))
                .build()
                .unwrap(),
        );
        f.set_parent(&Query::Id(eq(0)), &o);
    }

    #[test]
    #[should_panic]
    fn set_wrong_parent_as_parent() {
        let f1 = gen_frame();
        let f2 = gen_frame();
        let f1o = f1.get_object(0).unwrap();
        f2.set_parent(&Query::Id(eq(1)), &f1o);
    }

    #[test]
    fn normally_transfer_parent() {
        let f1 = gen_frame();
        let f2 = gen_frame();
        let mut o = f1.delete_objects_by_ids(&[0]).pop().unwrap();
        assert!(o.get_frame().is_none());
        o.set_id(33);
        f2.add_object(&o);
        o = f2.get_object(33).unwrap();
        f2.set_parent(&Query::Id(eq(1)), &o);
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
        frame.add_object(&gen_object(111));
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
        frame.add_object(&gen_object(111));
        let old_object = frame.get_object(111).unwrap();
        drop(frame);
        let frame = gen_frame();
        assert!(old_object.is_spoiled(), "Object is expected to be spoiled");
        frame.add_object(&old_object);
    }

    #[test]
    fn deleted_objects_clean() {
        let frame = gen_frame();
        let removed = frame.delete_objects_by_ids(&[0]).pop().unwrap();
        assert!(removed.is_detached());
        assert!(removed.get_parent().is_none());
    }
}
