use crate::match_query::MatchQuery;
use crate::primitives::attribute::Attribute;
use crate::primitives::bbox::VideoObjectBBoxTransformation;
use crate::primitives::draw::SetDrawLabelKind;
use crate::primitives::frame_update::VideoFrameUpdate;
use crate::primitives::message::Message;
use crate::primitives::object::{IdCollisionResolutionPolicy, VideoObject};
use crate::primitives::objects_view::VideoObjectsView;
use crate::release_gil;
use crate::with_gil;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyBytes;
use pyo3::{pyclass, pymethods, Py, PyAny, PyObject, PyResult};
use savant_core::json_api::ToSerdeJsonValue;
use savant_core::primitives::{rust, AttributeMethods};
use serde_json::Value;
use std::fmt::Debug;
use std::mem;

#[pyclass]
pub struct ExternalFrame(pub(crate) rust::ExternalFrame);

#[pymethods]
impl ExternalFrame {
    #[new]
    pub fn new(method: String, location: Option<String>) -> Self {
        Self(rust::ExternalFrame::new(method, location))
    }

    #[getter]
    pub fn method(&self) -> String {
        self.0.method.clone()
    }

    #[getter]
    pub fn location(&self) -> Option<String> {
        self.0.location.clone()
    }

    #[setter]
    pub fn set_method(&mut self, method: String) {
        self.0.method = method;
    }

    #[setter]
    pub fn set_location(&mut self, location: Option<String>) {
        self.0.location = location;
    }

    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", &self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

impl ToSerdeJsonValue for ExternalFrame {
    fn to_serde_json_value(&self) -> Value {
        self.0.to_serde_json_value()
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct VideoFrameContent(rust::VideoFrameContent);

#[pymethods]
impl VideoFrameContent {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", &self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[staticmethod]
    pub fn external(method: String, location: Option<String>) -> Self {
        Self(rust::VideoFrameContent::External(rust::ExternalFrame {
            method,
            location,
        }))
    }

    #[staticmethod]
    pub fn internal(data: Vec<u8>) -> Self {
        Self(rust::VideoFrameContent::Internal(data))
    }

    #[staticmethod]
    pub fn none() -> Self {
        Self(rust::VideoFrameContent::None)
    }

    pub fn is_external(&self) -> bool {
        matches!(&self.0, rust::VideoFrameContent::External(_))
    }

    pub fn is_internal(&self) -> bool {
        matches!(&self.0, rust::VideoFrameContent::Internal(_))
    }

    pub fn is_none(&self) -> bool {
        matches!(&self.0, rust::VideoFrameContent::None)
    }

    pub fn get_data(&self) -> PyResult<Vec<u8>> {
        match &self.0 {
            rust::VideoFrameContent::Internal(data) => Ok(data.clone()),
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "Video data is not stored internally",
            )),
        }
    }

    pub fn get_data_as_bytes(&self) -> PyResult<PyObject> {
        match &self.0 {
            rust::VideoFrameContent::Internal(data) => Ok({
                with_gil!(|py| {
                    let bytes = PyBytes::new(py, data);
                    PyObject::from(bytes)
                })
            }),
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "Video data is not stored internally",
            )),
        }
    }

    pub fn get_method(&self) -> PyResult<String> {
        match &self.0 {
            rust::VideoFrameContent::External(data) => Ok(data.method.clone()),
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "Video data is not stored externally",
            )),
        }
    }

    pub fn get_location(&self) -> PyResult<Option<String>> {
        match &self.0 {
            rust::VideoFrameContent::External(data) => Ok(data.location.clone()),
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "Video data is not stored externally",
            )),
        }
    }
}

#[pyclass]
#[derive(Copy, Clone, Debug)]
pub enum VideoFrameTranscodingMethod {
    Copy,
    Encoded,
}

impl From<VideoFrameTranscodingMethod> for rust::VideoFrameTranscodingMethod {
    fn from(value: VideoFrameTranscodingMethod) -> Self {
        match value {
            VideoFrameTranscodingMethod::Copy => rust::VideoFrameTranscodingMethod::Copy,
            VideoFrameTranscodingMethod::Encoded => rust::VideoFrameTranscodingMethod::Encoded,
        }
    }
}

impl From<rust::VideoFrameTranscodingMethod> for VideoFrameTranscodingMethod {
    fn from(value: rust::VideoFrameTranscodingMethod) -> Self {
        match value {
            rust::VideoFrameTranscodingMethod::Copy => VideoFrameTranscodingMethod::Copy,
            rust::VideoFrameTranscodingMethod::Encoded => VideoFrameTranscodingMethod::Encoded,
        }
    }
}

impl ToSerdeJsonValue for VideoFrameTranscodingMethod {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!(format!(
            "{:?}",
            rust::VideoFrameTranscodingMethod::from(*self)
        ))
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct VideoFrameTransformation(rust::VideoFrameTransformation);

#[pymethods]
impl VideoFrameTransformation {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", &self.0)
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[staticmethod]
    pub fn initial_size(width: i64, height: i64) -> Self {
        assert!(width > 0 && height > 0);
        Self(rust::VideoFrameTransformation::InitialSize(
            u64::try_from(width).unwrap(),
            u64::try_from(height).unwrap(),
        ))
    }

    #[staticmethod]
    pub fn resulting_size(width: i64, height: i64) -> Self {
        assert!(width > 0 && height > 0);
        Self(rust::VideoFrameTransformation::ResultingSize(
            u64::try_from(width).unwrap(),
            u64::try_from(height).unwrap(),
        ))
    }

    #[staticmethod]
    pub fn scale(width: i64, height: i64) -> Self {
        assert!(width > 0 && height > 0);
        Self(rust::VideoFrameTransformation::Scale(
            u64::try_from(width).unwrap(),
            u64::try_from(height).unwrap(),
        ))
    }

    #[staticmethod]
    pub fn padding(left: i64, top: i64, right: i64, bottom: i64) -> Self {
        assert!(left >= 0 && top >= 0 && right >= 0 && bottom >= 0);
        Self(rust::VideoFrameTransformation::Padding(
            u64::try_from(left).unwrap(),
            u64::try_from(top).unwrap(),
            u64::try_from(right).unwrap(),
            u64::try_from(bottom).unwrap(),
        ))
    }

    #[getter]
    pub fn is_initial_size(&self) -> bool {
        matches!(self.0, rust::VideoFrameTransformation::InitialSize(_, _))
    }

    #[getter]
    pub fn is_scale(&self) -> bool {
        matches!(self.0, rust::VideoFrameTransformation::Scale(_, _))
    }

    #[getter]
    pub fn is_padding(&self) -> bool {
        matches!(self.0, rust::VideoFrameTransformation::Padding(_, _, _, _))
    }

    #[getter]
    pub fn is_resulting_size(&self) -> bool {
        matches!(self.0, rust::VideoFrameTransformation::ResultingSize(_, _))
    }

    #[getter]
    pub fn as_initial_size(&self) -> Option<(u64, u64)> {
        match &self.0 {
            rust::VideoFrameTransformation::InitialSize(w, h) => Some((*w, *h)),
            _ => None,
        }
    }

    #[getter]
    pub fn as_resulting_size(&self) -> Option<(u64, u64)> {
        match &self.0 {
            rust::VideoFrameTransformation::ResultingSize(w, h) => Some((*w, *h)),
            _ => None,
        }
    }

    #[getter]
    pub fn as_scale(&self) -> Option<(u64, u64)> {
        match &self.0 {
            rust::VideoFrameTransformation::Scale(w, h) => Some((*w, *h)),
            _ => None,
        }
    }

    #[getter]
    pub fn as_padding(&self) -> Option<(u64, u64, u64, u64)> {
        match &self.0 {
            rust::VideoFrameTransformation::Padding(l, t, r, b) => Some((*l, *t, *r, *b)),
            _ => None,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct VideoFrame(pub(crate) rust::VideoFrameProxy);

impl ToSerdeJsonValue for VideoFrame {
    fn to_serde_json_value(&self) -> Value {
        self.0.to_serde_json_value()
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct BelongingVideoFrame(rust::BelongingVideoFrame);

impl From<VideoFrame> for BelongingVideoFrame {
    fn from(value: VideoFrame) -> Self {
        BelongingVideoFrame(value.0.into())
    }
}

impl From<BelongingVideoFrame> for VideoFrame {
    fn from(value: BelongingVideoFrame) -> Self {
        VideoFrame(value.0.into())
    }
}

#[pymethods]
impl VideoFrame {
    #[pyo3(name = "transform_geometry")]
    #[pyo3(signature = (ops, no_gil=true))]
    fn transform_geometry_gil(&self, ops: Vec<VideoObjectBBoxTransformation>, no_gil: bool) {
        release_gil!(no_gil, || {
            let ops_ref = ops.iter().map(|op| op.0).collect();
            self.0.transform_geometry(&ops_ref);
        })
    }

    #[getter]
    fn memory_handle(&self) -> usize {
        self.0.memory_handle()
    }

    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{:?}", &self.0)
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
        content: VideoFrameContent,
        transcoding_method: VideoFrameTranscodingMethod,
        codec: Option<String>,
        keyframe: Option<bool>,
        time_base: (i64, i64),
        pts: i64,
        dts: Option<i64>,
        duration: Option<i64>,
    ) -> Self {
        VideoFrame(rust::VideoFrameProxy::new(
            source_id,
            framerate,
            width,
            height,
            content.0,
            transcoding_method.into(),
            codec,
            keyframe,
            time_base,
            pts,
            dts,
            duration,
        ))
    }

    pub fn to_message(&self) -> Message {
        Message::video_frame(self)
    }

    #[getter]
    pub fn get_source_id(&self) -> String {
        self.0.get_source_id()
    }

    #[getter]
    #[pyo3(name = "json")]
    pub fn json_gil(&self) -> String {
        release_gil!(true, || serde_json::to_string(&self.to_serde_json_value())
            .unwrap())
    }

    #[getter]
    #[pyo3(name = "json_pretty")]
    fn json_pretty_gil(&self) -> String {
        release_gil!(true, || serde_json::to_string_pretty(
            &self.to_serde_json_value()
        )
        .unwrap())
    }

    #[setter]
    pub fn set_source_id(&mut self, source_id: String) {
        self.0.set_source_id(source_id)
    }

    #[setter]
    pub fn set_time_base(&mut self, time_base: (i32, i32)) {
        self.0.set_time_base(time_base)
    }

    #[getter]
    pub fn get_time_base(&self) -> (i32, i32) {
        self.0.get_time_base()
    }

    #[getter]
    pub fn get_pts(&self) -> i64 {
        self.0.get_pts()
    }

    #[setter]
    pub fn set_pts(&mut self, pts: i64) {
        self.0.set_pts(pts)
    }

    #[getter]
    pub fn get_framerate(&self) -> String {
        self.0.get_framerate()
    }

    #[setter]
    pub fn set_framerate(&mut self, framerate: String) {
        self.0.set_framerate(framerate)
    }

    #[getter]
    pub fn get_width(&self) -> i64 {
        self.0.get_width()
    }

    #[setter]
    pub fn set_width(&mut self, width: i64) {
        self.0.set_width(width)
    }

    #[getter]
    pub fn get_height(&self) -> i64 {
        self.0.get_height()
    }

    #[setter]
    pub fn set_height(&mut self, height: i64) {
        self.0.set_height(height)
    }

    #[getter]
    pub fn get_dts(&self) -> Option<i64> {
        self.0.get_dts()
    }

    #[setter]
    pub fn set_dts(&mut self, dts: Option<i64>) {
        self.0.set_dts(dts)
    }

    #[getter]
    pub fn get_duration(&self) -> Option<i64> {
        self.0.get_duration()
    }

    #[setter]
    pub fn set_duration(&mut self, duration: Option<i64>) {
        self.0.set_duration(duration)
    }

    #[getter]
    pub fn get_transcoding_method(&self) -> VideoFrameTranscodingMethod {
        self.0.get_transcoding_method().into()
    }

    #[setter]
    pub fn set_transcoding_method(&mut self, transcoding_method: VideoFrameTranscodingMethod) {
        self.0.set_transcoding_method(transcoding_method.into())
    }

    #[getter]
    pub fn get_codec(&self) -> Option<String> {
        self.0.get_codec()
    }

    #[setter]
    pub fn set_codec(&mut self, codec: Option<String>) {
        self.0.set_codec(codec)
    }

    pub fn clear_transformations(&mut self) {
        self.0.clear_transformations()
    }

    pub fn add_transformation(&mut self, transformation: VideoFrameTransformation) {
        self.0.add_transformation(transformation.0)
    }

    #[getter]
    pub fn get_transformations(&self) -> Vec<VideoFrameTransformation> {
        unsafe {
            mem::transmute::<Vec<rust::VideoFrameTransformation>, Vec<VideoFrameTransformation>>(
                self.0.get_transformations(),
            )
        }
    }

    #[getter]
    pub fn get_keyframe(&self) -> Option<bool> {
        self.0.get_keyframe()
    }

    #[setter]
    pub fn set_keyframe(&mut self, keyframe: Option<bool>) {
        self.0.set_keyframe(keyframe)
    }

    #[getter]
    pub fn get_content(&self) -> VideoFrameContent {
        VideoFrameContent(self.0.get_content())
    }

    #[setter]
    pub fn set_content(&mut self, content: VideoFrameContent) {
        self.0.set_content(content.0)
    }

    #[getter]
    pub fn attributes(&self) -> Vec<(String, String)> {
        self.0.get_attributes()
    }

    #[pyo3(signature = (namespace=None, names=vec![], hint=None))]
    pub fn find_attributes(
        &self,
        namespace: Option<String>,
        names: Vec<String>,
        hint: Option<String>,
    ) -> Vec<(String, String)> {
        self.0.find_attributes(namespace, names, hint)
    }

    pub fn get_attribute(&self, namespace: String, name: String) -> Option<Attribute> {
        self.0.get_attribute(namespace, name).map(Attribute)
    }

    #[pyo3(signature = (namespace=None, names=vec![]))]
    pub fn delete_attributes(&mut self, namespace: Option<String>, names: Vec<String>) {
        self.0.delete_attributes(namespace, names)
    }

    pub fn add_object(&self, o: VideoObject, policy: IdCollisionResolutionPolicy) -> PyResult<()> {
        self.0
            .add_object(&o.0, policy.into())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn delete_attribute(&mut self, namespace: String, name: String) -> Option<Attribute> {
        self.0.delete_attribute(namespace, name).map(Attribute)
    }

    pub fn set_attribute(&mut self, attribute: Attribute) -> Option<Attribute> {
        self.0.set_attribute(attribute.0).map(Attribute)
    }

    pub fn clear_attributes(&mut self) {
        self.0.clear_attributes()
    }

    #[pyo3(name = "set_draw_label")]
    #[pyo3(signature = (q, draw_label, no_gil = false))]
    pub fn set_draw_label_gil(&self, q: &MatchQuery, draw_label: SetDrawLabelKind, no_gil: bool) {
        release_gil!(no_gil, || self.0.set_draw_label(&q.0, draw_label.0))
    }

    pub fn get_object(&self, id: i64) -> Option<VideoObject> {
        self.0.get_object(id).map(VideoObject)
    }

    #[pyo3(name = "access_objects")]
    #[pyo3(signature = (q, no_gil = true))]
    pub fn access_objects_gil(&self, q: &MatchQuery, no_gil: bool) -> VideoObjectsView {
        release_gil!(no_gil, || self.0.access_objects(&q.0).into())
    }

    pub fn get_all_objects(&self) -> VideoObjectsView {
        self.0.get_all_objects().into()
    }

    pub fn access_objects_by_id(&self, ids: Vec<i64>) -> VideoObjectsView {
        self.0.access_objects_by_id(&ids).into()
    }

    pub fn delete_objects_by_ids(&self, ids: Vec<i64>) -> VideoObjectsView {
        self.0.delete_objects_by_ids(&ids).into()
    }

    #[pyo3(name = "delete_objects")]
    #[pyo3(signature = (q, no_gil = true))]
    pub fn delete_objects_gil(&self, q: &MatchQuery, no_gil: bool) -> VideoObjectsView {
        release_gil!(no_gil, || self.0.delete_objects(&q.0).into())
    }

    #[pyo3(name = "set_parent")]
    #[pyo3(signature = (q, parent, no_gil = true))]
    pub fn set_parent_gil(
        &self,
        q: &MatchQuery,
        parent: &VideoObject,
        no_gil: bool,
    ) -> VideoObjectsView {
        release_gil!(no_gil, || self.0.set_parent(&q.0, &parent.0).into())
    }

    #[pyo3(name = "set_parent_by_id")]
    pub fn set_parent_by_id_py(&self, object_id: i64, parent_id: i64) -> PyResult<()> {
        self.0
            .set_parent_by_id(object_id, parent_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(name = "clear_parent")]
    #[pyo3(signature = (q, no_gil = true))]
    pub fn clear_parent_gil(&self, q: &MatchQuery, no_gil: bool) -> VideoObjectsView {
        release_gil!(no_gil, || self.0.clear_parent(&q.0).into())
    }

    pub fn clear_objects(&self) {
        self.0.clear_objects()
    }

    #[pyo3(name = "make_snapshot")]
    #[pyo3(signature = (no_gil = true))]
    pub fn make_snapshot_gil(&self, no_gil: bool) {
        release_gil!(no_gil, || self.0.make_snapshot())
    }

    #[pyo3(name = "restore_from_snapshot")]
    #[pyo3(signature = (no_gil = true))]
    pub fn restore_from_snapshot_gil(&self, no_gil: bool) {
        release_gil!(no_gil, || self.0.restore_from_snapshot())
    }

    #[pyo3(name = "get_children")]
    pub fn get_children_py(&self, id: i64) -> VideoObjectsView {
        self.0.get_children(id).into()
    }

    #[pyo3(name = "copy")]
    #[pyo3(signature = (no_gil = true))]
    pub fn copy_gil(&self, no_gil: bool) -> VideoFrame {
        release_gil!(no_gil, || VideoFrame(self.0.deep_copy()))
    }

    #[pyo3(name = "update")]
    #[pyo3(signature = (other, no_gil = true))]
    pub fn update_gil(&self, other: &VideoFrameUpdate, no_gil: bool) -> PyResult<()> {
        release_gil!(no_gil, || self.0.update(&other.0))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}
