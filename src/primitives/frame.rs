use crate::primitives::{Object, ProxyObject, Value};
use pyo3::{pyclass, pymethods, Py, PyAny, PyResult};
use rkyv::{with::Skip, Archive, Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct ExternalFrame {
    #[pyo3(get, set)]
    pub method: String,
    #[pyo3(get, set)]
    pub location: Option<String>,
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

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyVideoFrameContent {
    data: VideoFrameContent,
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
            data: VideoFrameContent::External(ExternalFrame::new(method, location)),
        }
    }

    #[staticmethod]
    pub fn internal(data: Vec<u8>) -> Self {
        Self {
            data: VideoFrameContent::Internal(data),
        }
    }

    pub fn is_external(&self) -> bool {
        matches!(self.data, VideoFrameContent::External(_))
    }

    pub fn is_internal(&self) -> bool {
        matches!(self.data, VideoFrameContent::Internal(_))
    }

    pub fn is_none(&self) -> bool {
        matches!(self.data, VideoFrameContent::None)
    }

    pub fn get_data(&self) -> PyResult<Vec<u8>> {
        match &self.data {
            VideoFrameContent::Internal(data) => Ok(data.clone()),
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "Video data is not stored internally",
            )),
        }
    }

    pub fn get_method(&self) -> PyResult<String> {
        match &self.data {
            VideoFrameContent::External(data) => Ok(data.method.clone()),
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "Video data is not stored externally",
            )),
        }
    }

    pub fn get_location(&self) -> PyResult<Option<String>> {
        match &self.data {
            VideoFrameContent::External(data) => Ok(data.location.clone()),
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "Video data is not stored externally",
            )),
        }
    }
}

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, Clone, derive_builder::Builder)]
#[archive(check_bytes)]
pub struct VideoFrame {
    #[pyo3(get, set)]
    pub source_id: String,
    #[pyo3(get, set)]
    pub pts: i64,
    #[pyo3(get, set)]
    pub framerate: String,
    #[pyo3(get, set)]
    pub width: i64,
    #[pyo3(get, set)]
    pub height: i64,
    #[pyo3(get, set)]
    pub dts: Option<i64>,
    #[pyo3(get, set)]
    pub duration: Option<i64>,
    #[pyo3(get, set)]
    pub codec: Option<String>,
    #[pyo3(get, set)]
    pub keyframe: Option<bool>,
    pub content: VideoFrameContent,
    pub tags: HashMap<String, Value>,
    pub offline_objects: Vec<Object>,
    #[with(Skip)]
    pub resident_objects: Vec<Arc<Mutex<Object>>>,
}

#[pymethods]
impl VideoFrame {
    #[classattr]
    const __hash__: Option<Py<PyAny>> = None;

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    #[allow(clippy::too_many_arguments)]
    #[new]
    pub fn new(
        source_id: String,
        content: PyVideoFrameContent,
        pts: i64,
        framerate: String,
        width: i64,
        height: i64,
        tags: HashMap<String, Value>,
        objects: Vec<Object>,
        dts: Option<i64>,
        duration: Option<i64>,
        codec: Option<String>,
        keyframe: Option<bool>,
    ) -> Self {
        Self {
            source_id,
            pts,
            framerate,
            width,
            height,
            dts,
            duration,
            codec,
            keyframe,
            tags,
            offline_objects: Vec::default(),
            resident_objects: objects
                .into_iter()
                .map(|o| Arc::new(Mutex::new(o)))
                .collect(),
            content: content.data,
        }
    }

    pub fn tags(&self) -> Vec<String> {
        self.tags.keys().cloned().collect()
    }

    pub fn get_tag(&self, key: String) -> Option<Value> {
        self.tags.get(&key).cloned()
    }

    pub fn delete_tag(&mut self, key: String) -> Option<Value> {
        self.tags.remove(&key)
    }

    pub fn set_tag(&mut self, key: String, value: Value) -> Option<Value> {
        self.tags.insert(key, value)
    }

    pub fn clear_tags(&mut self) {
        self.tags.clear();
    }

    pub fn object(&self, id: i64) -> Option<ProxyObject> {
        self.resident_objects
            .iter()
            .find(|o| o.lock().unwrap().id == id)
            .map(|o| ProxyObject::new(o.clone()))
    }

    pub fn access_objects(
        &self,
        negated: bool,
        model_name: Option<String>,
        label: Option<String>,
    ) -> Vec<ProxyObject> {
        self.resident_objects
            .iter()
            .filter(|o| {
                let o = o.lock().unwrap();
                let model_name_match = match &model_name {
                    Some(model_name) => o.model_name == *model_name,
                    None => true,
                };
                let label_match = match &label {
                    Some(label) => o.label == *label,
                    None => true,
                };
                model_name_match && label_match ^ negated
            })
            .map(|o| ProxyObject::new(o.clone()))
            .collect()
    }

    pub fn access_objects_by_id(&self, ids: Vec<i64>) -> Vec<ProxyObject> {
        self.resident_objects
            .iter()
            .filter(|o| ids.contains(&o.lock().unwrap().id))
            .map(|o| ProxyObject::new(o.clone()))
            .collect()
    }

    pub fn add_object(&mut self, object: Object) {
        self.resident_objects.push(Arc::new(Mutex::new(object)));
    }

    pub fn delete_objects_by_ids(&mut self, ids: Vec<i64>) {
        self.resident_objects
            .retain(|o| !ids.contains(&o.lock().unwrap().id));
    }

    pub fn delete_objects(
        &mut self,
        negated: bool,
        model_name: Option<String>,
        label: Option<String>,
    ) {
        self.resident_objects.retain(|o| {
            let o = o.lock().unwrap();
            let model_name_match = match &model_name {
                Some(model_name) => o.model_name == *model_name,
                None => true,
            };
            let label_match = match &label {
                Some(label) => o.label == *label,
                None => true,
            };
            !(model_name_match && label_match ^ negated)
        });
    }

    pub fn clear_objects(&mut self) {
        self.resident_objects.clear();
    }
}
