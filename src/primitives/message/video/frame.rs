use crate::primitives::attribute::{Attributive, InnerAttributes};
use crate::primitives::message::video::object::query::py::QueryWrapper;
use crate::primitives::message::video::object::query::{ExecutableQuery, IntExpression, Query};
use crate::primitives::message::video::object::vector::VectorView;
use crate::primitives::message::video::object::InnerObject;
use crate::primitives::to_json_value::ToSerdeJsonValue;
use crate::primitives::{
    Attribute, Message, Object, ParentObject, SetDrawLabelKind, SetDrawLabelKindWrapper,
};
use crate::utils::python::no_gil;
use derive_builder::Builder;
use hashbrown::HashSet;
use parking_lot::Mutex;
use pyo3::{pyclass, pymethods, Py, PyAny, PyResult};
use rkyv::{with::Skip, Archive, Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::mem;
use std::ops::Deref;
use std::sync::Arc;

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
    pub offline_objects: Vec<InnerObject>,
    #[with(Skip)]
    #[builder(setter(skip))]
    pub(crate) resident_objects: Vec<Arc<Mutex<InnerObject>>>,
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
            offline_objects: Vec::new(),
            resident_objects: Vec::new(),
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
                "objects": self.resident_objects.iter().map(|o| o.lock().to_serde_json_value()).collect::<Vec<_>>(),
            }
        )
    }
}

impl InnerAttributes for Box<InnerVideoFrame> {
    fn get_attributes_ref(&self) -> &HashMap<(String, String), Attribute> {
        &self.attributes
    }

    fn get_attributes_ref_mut(&mut self) -> &mut HashMap<(String, String), Attribute> {
        &mut self.attributes
    }
}

impl InnerVideoFrame {
    pub(crate) fn preserve(&mut self) {
        let mut ids = HashSet::new();
        for o in self.resident_objects.iter() {
            let mut obj = o.lock();
            ids.insert(obj.id);
            let real_parent_id = obj.parent.as_ref().map(|p| p.inner.lock().id);
            obj.parent_id = real_parent_id;
        }

        self.offline_objects = self
            .resident_objects
            .iter()
            .map(|o| o.lock().clone())
            .collect();

        assert!(self
            .offline_objects
            .iter()
            .all(|x| x.parent_id.map(|id| ids.contains(&id)).unwrap_or(true)));
    }

    pub(crate) fn restore(&mut self) {
        self.resident_objects = self
            .offline_objects
            .iter()
            .map(|o| Arc::new(Mutex::new(o.clone())))
            .collect();

        for (i, o) in self.resident_objects.iter().enumerate() {
            let mut o = o.lock();
            let required_parent_id = o.parent_id;
            if required_parent_id.is_none() {
                continue;
            }

            let required_parent_id = required_parent_id.unwrap();

            for (j, p) in self.resident_objects.iter().enumerate() {
                if i == j {
                    continue;
                }

                let p_inner = p.lock();
                let parent_id = p_inner.id;
                if parent_id == required_parent_id {
                    o.parent = Some(ParentObject::new(Object::from_arc_inner_object(p.clone())));
                    break;
                }
            }

            if o.parent.is_none() {
                panic!("Parent object with id {} not found", required_parent_id);
            }
        }
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
pub struct VideoFrame {
    pub(crate) inner: Arc<Mutex<Box<InnerVideoFrame>>>,
}

impl Attributive<Box<InnerVideoFrame>> for VideoFrame {
    fn get_inner(&self) -> Arc<Mutex<Box<InnerVideoFrame>>> {
        self.inner.clone()
    }
}

impl VideoFrame {
    pub(crate) fn from_inner(object: InnerVideoFrame) -> Self {
        let f = VideoFrame {
            inner: Arc::new(Mutex::new(Box::new(object))),
        };
        let objects = f.access_objects(&Query::Idle);
        objects.iter().for_each(|o| o.attach(f.clone()));
        f
    }

    pub fn access_objects(&self, q: &Query) -> Vec<Object> {
        let frame = self.inner.lock();
        frame
            .resident_objects
            .iter()
            .filter_map(|o| {
                if q.execute(&Object::from_arc_inner_object(o.clone())) {
                    Some(Object::from_arc_inner_object(o.clone()))
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
        let frame = self.inner.lock();
        frame
            .resident_objects
            .iter()
            .filter(|o| ids.contains(&o.lock().id))
            .map(|o| Object::from_arc_inner_object(o.clone()))
            .collect()
    }

    pub fn delete_objects_by_ids(&mut self, ids: &[i64]) -> Vec<Object> {
        self.clear_parent(&Query::ParentId(IntExpression::OneOf(ids.to_vec())));
        let mut frame = self.inner.lock();
        let objects = mem::take(&mut frame.resident_objects);
        let (retained, removed) = objects
            .into_iter()
            .partition(|o| !ids.contains(&o.lock().id));
        frame.resident_objects = retained;
        removed
            .into_iter()
            .map(|o| {
                let o = Object::from_arc_inner_object(o);
                o.detach();
                o
            })
            .collect()
    }

    pub fn delete_objects(&mut self, q: &Query) -> Vec<Object> {
        let objs = self.access_objects(q);
        let ids = objs.iter().map(|o| o.get_id()).collect::<Vec<_>>();
        self.delete_objects_by_ids(&ids)
    }

    pub fn get_object(&self, id: i64) -> Option<Object> {
        let frame = self.inner.lock();
        frame
            .resident_objects
            .iter()
            .find(|o| o.lock().id == id)
            .map(|o| Object::from_arc_inner_object(o.clone()))
    }

    pub fn make_snapshot(&mut self) {
        let mut frame = self.inner.lock();
        frame.preserve();
    }

    pub fn restore_from_snapshot(&mut self) {
        let mut frame = self.inner.lock();
        frame.resident_objects.clear();
        frame.restore();
    }

    pub fn get_modified_objects(&self) -> Vec<Object> {
        let frame = self.inner.lock();
        frame
            .resident_objects
            .iter()
            .filter(|o| !o.lock().modifications.is_empty())
            .map(|o| Object::from_arc_inner_object(o.clone()))
            .collect()
    }

    pub fn set_draw_label(&self, q: &Query, label: SetDrawLabelKind) {
        let objects = self.access_objects(q);
        objects.iter().for_each(|o| match &label {
            SetDrawLabelKind::OwnLabel(l) => {
                o.inner.lock().draw_label = Some(l.clone());
            }
            SetDrawLabelKind::ParentLabel(l) => {
                if let Some(p) = o.inner.lock().parent.as_ref() {
                    p.inner.lock().draw_label = Some(l.clone());
                }
            }
        });
    }

    pub fn set_parent(&self, q: &Query, parent: &Object) {
        let objects = self.access_objects(q);
        assert!(
            parent
                .get_frame()
                .filter(|f| Arc::ptr_eq(&f.inner, &self.inner))
                .is_some(),
            "Parent must be attached to the frame before being assigned to its objects!"
        );
        objects.iter().for_each(|o| {
            let mut inner = o.inner.lock();
            inner.parent = Some(ParentObject::new(parent.clone()));
            inner.parent_id = Some(parent.get_id());
        });
    }

    pub fn clear_parent(&self, q: &Query) {
        let objects = self.access_objects(q);
        objects.iter().for_each(|o| {
            let mut inner = o.inner.lock();
            inner.parent = None;
            inner.parent_id = None;
        });
    }

    pub fn get_children(&self, o: &Object) -> Vec<Object> {
        let frame = self.inner.lock();
        frame
            .resident_objects
            .iter()
            .filter_map(|ch| {
                ch.lock().parent.as_ref().map(|p| {
                    Arc::ptr_eq(&o.inner, &p.inner)
                        .then(|| Object::from_arc_inner_object(ch.clone()))
                })
            })
            .flatten()
            .collect()
    }
}

impl ToSerdeJsonValue for VideoFrame {
    fn to_serde_json_value(&self) -> Value {
        self.inner.lock().to_serde_json_value()
    }
}

#[pymethods]
impl VideoFrame {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:#?}", self.inner.lock())
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
        self.inner.lock().source_id.clone()
    }

    #[getter]
    #[pyo3(name = "json")]
    pub fn json_py(&self) -> String {
        no_gil(|| serde_json::to_string(&self.to_serde_json_value()).unwrap())
    }

    #[getter]
    #[pyo3(name = "json_pretty")]
    fn json_pretty_py(&self) -> String {
        no_gil(|| serde_json::to_string_pretty(&self.to_serde_json_value()).unwrap())
    }

    #[setter]
    pub fn set_source_id(&mut self, source_id: String) {
        let mut frame = self.inner.lock();
        frame.source_id = source_id;
    }

    #[getter]
    pub fn get_pts(&self) -> i64 {
        self.inner.lock().pts
    }

    #[setter]
    pub fn set_pts(&mut self, pts: i64) {
        assert!(pts >= 0, "pts must be greater than or equal to 0");
        let mut frame = self.inner.lock();
        frame.pts = pts;
    }

    #[getter]
    pub fn get_framerate(&self) -> String {
        self.inner.lock().framerate.clone()
    }

    #[setter]
    pub fn set_framerate(&mut self, framerate: String) {
        let mut frame = self.inner.lock();
        frame.framerate = framerate;
    }

    #[getter]
    pub fn get_width(&self) -> i64 {
        self.inner.lock().width
    }

    #[setter]
    pub fn set_width(&mut self, width: i64) {
        assert!(width > 0, "width must be greater than 0");
        let mut frame = self.inner.lock();
        frame.width = width;
    }

    #[getter]
    pub fn get_height(&self) -> i64 {
        self.inner.lock().height
    }

    #[setter]
    pub fn set_height(&mut self, height: i64) {
        assert!(height > 0, "height must be greater than 0");
        let mut frame = self.inner.lock();
        frame.height = height;
    }

    #[getter]
    pub fn get_dts(&self) -> Option<i64> {
        let frame = self.inner.lock();
        frame.dts
    }

    #[setter]
    pub fn set_dts(&mut self, dts: Option<i64>) {
        assert!(
            dts.is_none() || dts.unwrap() >= 0,
            "dts must be greater than or equal to 0"
        );
        let mut frame = self.inner.lock();
        frame.dts = dts;
    }

    #[getter]
    pub fn get_duration(&self) -> Option<i64> {
        let frame = self.inner.lock();
        frame.duration
    }

    #[setter]
    pub fn set_duration(&mut self, duration: Option<i64>) {
        assert!(
            duration.is_none() || duration.unwrap() >= 0,
            "duration must be greater than or equal to 0"
        );
        let mut frame = self.inner.lock();
        frame.duration = duration;
    }

    #[getter]
    pub fn get_transcoding_method(&self) -> VideoTranscodingMethod {
        let frame = self.inner.lock();
        frame.transcoding_method.clone()
    }

    #[setter]
    pub fn set_transcoding_method(&mut self, transcoding_method: VideoTranscodingMethod) {
        let mut frame = self.inner.lock();
        frame.transcoding_method = transcoding_method;
    }

    #[getter]
    pub fn get_codec(&self) -> Option<String> {
        let frame = self.inner.lock();
        frame.codec.clone()
    }

    #[setter]
    pub fn set_codec(&mut self, codec: Option<String>) {
        let mut frame = self.inner.lock();
        frame.codec = codec;
    }

    pub fn clear_transformations(&mut self) {
        let mut frame = self.inner.lock();
        frame.transformations.clear();
    }

    pub fn add_transformation(&mut self, transformation: PyFrameTransformation) {
        let mut frame = self.inner.lock();
        frame.transformations.push(transformation.inner);
    }

    #[getter]
    pub fn get_transformations(&self) -> Vec<PyFrameTransformation> {
        let frame = self.inner.lock();
        frame
            .transformations
            .iter()
            .map(|t| PyFrameTransformation::new(t.clone()))
            .collect()
    }

    #[getter]
    pub fn get_keyframe(&self) -> Option<bool> {
        let frame = self.inner.lock();
        frame.keyframe
    }

    #[setter]
    pub fn set_keyframe(&mut self, keyframe: Option<bool>) {
        let mut frame = self.inner.lock();
        frame.keyframe = keyframe;
    }

    #[getter]
    pub fn get_content(&self) -> PyVideoFrameContent {
        let frame = self.inner.lock();
        PyVideoFrameContent::new(frame.content.clone())
    }

    #[setter]
    pub fn set_content(&mut self, content: PyVideoFrameContent) {
        let mut frame = self.inner.lock();
        frame.content = content.inner;
    }

    #[getter]
    pub fn attributes(&self) -> Vec<(String, String)> {
        no_gil(|| self.get_attributes())
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

    #[pyo3(name = "get_attribute")]
    pub fn get_attribute_py(&self, creator: String, name: String) -> Option<Attribute> {
        no_gil(|| self.get_attribute(creator, name))
    }

    #[pyo3(signature = (negated=false, creator=None, names=vec![]))]
    #[pyo3(name = "delete_attributes")]
    pub fn delete_attributes_py(
        &mut self,
        negated: bool,
        creator: Option<String>,
        names: Vec<String>,
    ) {
        no_gil(|| self.delete_attributes(negated, creator, names))
    }

    #[pyo3(name = "delete_attribute")]
    pub fn delete_attribute_py(&mut self, creator: String, name: String) -> Option<Attribute> {
        self.delete_attribute(creator, name)
    }

    #[pyo3(name = "set_draw_label")]
    pub fn set_draw_label_py(&mut self, q: QueryWrapper, draw_label: SetDrawLabelKindWrapper) {
        no_gil(|| self.set_draw_label(q.inner.deref(), draw_label.inner))
    }

    #[pyo3(name = "set_attribute")]
    pub fn set_attribute_py(&mut self, attribute: Attribute) -> Option<Attribute> {
        self.set_attribute(attribute)
    }

    #[pyo3(name = "clear_attributes")]
    pub fn clear_attributes_py(&mut self) {
        self.clear_attributes()
    }

    #[pyo3(name = "get_object")]
    pub fn get_object_py(&self, id: i64) -> Option<Object> {
        no_gil(|| self.get_object(id))
    }

    #[pyo3(name = "access_objects")]
    pub fn access_objects_py(&self, q: QueryWrapper) -> VectorView {
        no_gil(|| self.access_objects(q.inner.deref()).into())
    }

    #[pyo3(name = "access_objects_by_id")]
    pub fn access_objects_by_id_py(&self, ids: Vec<i64>) -> VectorView {
        no_gil(|| self.access_objects_by_id(&ids).into())
    }

    pub fn add_object(&mut self, object: Object) {
        let mut frame = self.inner.lock();
        let parent = object.get_parent();
        if let Some(parent) = parent.as_ref() {
            let parent_frame = parent.object().get_frame();
            assert!(parent_frame.is_some(), "When a detached object with parent is being attached to a frame, the parent must be attached to the same frame.");
            let parent_frame = parent_frame.as_ref().unwrap();
            assert!(
                Arc::ptr_eq(&parent_frame.inner, &self.inner),
                "Parent must be attached to the frame before its children."
            );
        }
        object.attach(self.clone());
        frame.resident_objects.push(object.inner);
    }

    #[pyo3(name = "delete_objects_by_ids")]
    pub fn delete_objects_by_ids_py(&mut self, ids: Vec<i64>) -> VectorView {
        no_gil(|| self.delete_objects_by_ids(&ids).into())
    }

    #[pyo3(name = "delete_objects")]
    pub fn delete_objects_py(&mut self, query: QueryWrapper) -> VectorView {
        no_gil(|| self.delete_objects(&query.inner).into())
    }

    #[pyo3(name = "set_parent")]
    pub fn set_parent_py(&mut self, q: QueryWrapper, parent: Object) {
        no_gil(|| self.set_parent(q.inner.deref(), &parent))
    }

    #[pyo3(name = "clear_parent")]
    pub fn clear_parent_py(&mut self, q: QueryWrapper) {
        no_gil(|| self.clear_parent(q.inner.deref()))
    }

    pub fn clear_objects(&mut self) {
        let mut frame = self.inner.lock();
        frame.resident_objects.clear();
    }

    #[pyo3(name = "make_snapshot")]
    pub fn make_snapshot_py(&mut self) {
        no_gil(|| self.make_snapshot())
    }

    #[pyo3(name = "restore_from_snapshot")]
    pub fn restore_from_snapshot_py(&mut self) {
        no_gil(|| self.restore_from_snapshot())
    }

    #[pyo3(name = "get_modified_objects")]
    pub fn get_modified_objects_py(&self) -> VectorView {
        no_gil(|| self.get_modified_objects().into())
    }

    #[pyo3(name = "get_children")]
    pub fn get_children_py(&self, o: Object) -> VectorView {
        no_gil(|| self.get_children(&o).into())
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::attribute::Attributive;
    use crate::primitives::message::video::object::query::{eq, one_of, Query};
    use crate::primitives::message::video::object::InnerObjectBuilder;
    use crate::primitives::{Modification, Object, RBBox, SetDrawLabelKind};
    use crate::test::utils::{gen_frame, s};

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
        let mut attributes = t.find_attributes_py(Some("system".to_string()), None, None);
        attributes.sort();
        assert_eq!(attributes.len(), 2);
        assert_eq!(attributes[0], ("system".to_string(), "test".to_string()));
        assert_eq!(attributes[1], ("system".to_string(), "test2".to_string()));

        let attributes =
            t.find_attributes_py(Some("system".to_string()), Some("test".to_string()), None);
        assert_eq!(attributes.len(), 1);
        assert_eq!(attributes[0], ("system".to_string(), "test".to_string()));

        let attributes = t.find_attributes_py(
            Some("system".to_string()),
            Some("test".to_string()),
            Some("test".to_string()),
        );
        assert_eq!(attributes.len(), 1);
        assert_eq!(attributes[0], ("system".to_string(), "test".to_string()));

        let mut attributes = t.find_attributes_py(None, None, Some("test".to_string()));
        attributes.sort();
        assert_eq!(attributes.len(), 2);
        assert_eq!(attributes[0], ("system".to_string(), "test".to_string()));
        assert_eq!(attributes[1], ("system".to_string(), "test2".to_string()));
    }

    #[test]
    fn test_delete_objects_by_ids() {
        pyo3::prepare_freethreaded_python();
        let mut t = gen_frame();
        t.delete_objects_by_ids(&[0, 1]);
        let objects = t.access_objects(&Query::Idle);
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].get_id(), 2);
    }

    #[test]
    fn test_parent_cleared_when_delete_objects_by_ids() {
        let mut f = gen_frame();
        f.delete_objects_by_ids(&[0]);
        let o = f.get_object(1).unwrap();
        assert!(o.get_parent().is_none());
        let o = f.get_object(2).unwrap();
        assert!(o.get_parent().is_none());

        let mut f = gen_frame();
        f.delete_objects_by_ids(&[1]);
        let o = f.get_object(2).unwrap();
        assert!(o.get_parent().is_some());
    }

    #[test]
    fn test_parent_cleared_when_delete_objects_by_query() {
        let mut f = gen_frame();

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
        let mut t = gen_frame();
        let objs = t.delete_objects(&Query::Idle);
        assert_eq!(objs.len(), 3);
        let objects = t.access_objects(&Query::Idle);
        assert!(objects.is_empty());
    }

    #[test]
    fn test_snapshot_simple() {
        let mut t = gen_frame();
        t.make_snapshot_py();
        let mut o = t.access_objects_by_id(&vec![0]).pop().unwrap();
        o.set_id(12);
        assert!(matches!(t.access_objects_by_id(&vec![0]).pop(), None));
        t.restore_from_snapshot_py();
        t.access_objects_by_id(&vec![0]).pop().unwrap();
    }

    #[test]
    fn test_modified_objects() {
        let t = gen_frame();
        let mut o = t.access_objects_by_id(&vec![0]).pop().unwrap();
        o.set_id(12);
        let mut modified = t.get_modified_objects();
        assert_eq!(modified.len(), 1);
        let modified = modified.pop().unwrap();
        assert_eq!(modified.get_id(), 12);

        let mods = modified.take_modifications();
        assert_eq!(mods.len(), 1);
        assert_eq!(mods, vec![Modification::Id]);

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
        let mut frame = gen_frame();
        let mut obj = frame.get_object(0).unwrap();
        obj.set_parent(Some(parent));
        frame.make_snapshot();
    }

    #[test]
    fn test_snapshot_with_parent_added_to_frame() {
        let mut parent = Object::from_inner_object(
            InnerObjectBuilder::default()
                .parent_id(None)
                .creator(s("some-model"))
                .label(s("some-label"))
                .id(155)
                .bbox(RBBox::new(0.0, 0.0, 0.0, 0.0, None))
                .build()
                .unwrap(),
        );
        let mut frame = gen_frame();
        frame.add_object(parent.clone());
        let mut obj = frame.get_object(0).unwrap();
        obj.set_parent(Some(parent.clone()));
        parent.set_id(255);
        frame.make_snapshot();
        frame.restore_from_snapshot();
        let obj = frame.get_object(0).unwrap();
        assert_eq!(obj.get_parent().unwrap().inner.lock().id, 255);
    }

    #[test]
    fn test_no_children() {
        let frame = gen_frame();
        let obj = frame.get_object(2).unwrap();
        assert!(frame.get_children(&obj).is_empty());
    }

    #[test]
    fn test_two_children() {
        let frame = gen_frame();
        let obj = frame.get_object(0).unwrap();
        assert_eq!(frame.get_children(&obj).len(), 2);
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
        let children = frame.get_children(&parent);
        assert_eq!(children.len(), 2);
    }

    #[test]
    #[should_panic]
    fn attach_object_with_detached_parent() {
        todo!("")
    }

    #[test]
    #[should_panic]
    fn set_detached_parent_as_parent() {
        let f = gen_frame();
        let o = Object::from_inner_object(
            InnerObjectBuilder::default()
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
        let mut f1 = gen_frame();
        let mut f2 = gen_frame();
        let mut o = f1.delete_objects_by_ids(&[0]).pop().unwrap();
        assert!(o.get_frame().is_none());
        o.set_id(33);
        f2.add_object(o);
        o = f2.get_object(33).unwrap();
        f2.set_parent(&Query::Id(eq(1)), &o);
    }
}
