pub mod registry;

use crate::primitives::{Object, ProxyObject, Value};
use pyo3::{pyclass, pymethods, Py, PyAny, PyResult, Python};
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

    #[staticmethod]
    pub fn none() -> Self {
        Self {
            data: VideoFrameContent::None,
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
        Python::with_gil(|py| {
            py.allow_threads(|| {
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
                        (model_name_match && label_match) ^ negated
                    })
                    .map(|o| ProxyObject::new(o.clone()))
                    .collect()
            })
        })
    }

    pub fn access_objects_by_id(&self, ids: Vec<i64>) -> Vec<ProxyObject> {
        Python::with_gil(|py| {
            py.allow_threads(|| {
                self.resident_objects
                    .iter()
                    .filter(|o| ids.contains(&o.lock().unwrap().id))
                    .map(|o| ProxyObject::new(o.clone()))
                    .collect()
            })
        })
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

#[cfg(test)]
mod tests {
    use crate::primitives::frame::PyVideoFrameContent;
    use crate::primitives::{BBox, ObjectBuilder, VideoFrame, VideoFrameBuilder};
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    fn gen_frame() -> VideoFrame {
        VideoFrameBuilder::default()
            .source_id("test".to_string())
            .pts(0)
            .framerate("test".to_string())
            .width(0)
            .height(0)
            .content(PyVideoFrameContent::none().data)
            .dts(None)
            .duration(None)
            .codec(None)
            .keyframe(None)
            .tags(Default::default())
            .offline_objects(Default::default())
            .resident_objects(
                vec![
                    ObjectBuilder::default()
                        .id(0)
                        .bbox(BBox::new(0.0, 0.0, 0.0, 0.0, None))
                        .parent(None)
                        .attributes(HashMap::default())
                        .confidence(None)
                        .model_name("test".to_string())
                        .label("test2".to_string())
                        .build()
                        .unwrap(),
                    ObjectBuilder::default()
                        .id(1)
                        .bbox(BBox::new(0.0, 0.0, 0.0, 0.0, None))
                        .parent(None)
                        .attributes(HashMap::default())
                        .confidence(None)
                        .model_name("test2".to_string())
                        .label("test".to_string())
                        .build()
                        .unwrap(),
                    ObjectBuilder::default()
                        .id(2)
                        .bbox(BBox::new(0.0, 0.0, 0.0, 0.0, None))
                        .parent(None)
                        .attributes(HashMap::default())
                        .confidence(None)
                        .model_name("test2".to_string())
                        .label("test2".to_string())
                        .build()
                        .unwrap(),
                ]
                .into_iter()
                .map(|o| Arc::new(Mutex::new(o)))
                .collect(),
            )
            .build()
            .unwrap()
    }

    #[test]
    fn test_access_objects_by_id() {
        pyo3::prepare_freethreaded_python();
        let t = gen_frame();
        let objects = t.access_objects_by_id(vec![0, 1]);
        assert_eq!(objects.len(), 2);
        assert_eq!(objects[0].id(), 0);
        assert_eq!(objects[1].id(), 1);
    }

    #[test]
    fn test_access_objects() {
        pyo3::prepare_freethreaded_python();

        let t = gen_frame();
        let objects = t.access_objects(false, None, None);
        assert_eq!(objects.len(), 3);

        let t = gen_frame();
        let objects = t.access_objects(true, None, None);
        assert!(objects.is_empty());

        let t = gen_frame();
        let objects = t.access_objects(false, Some("abc".to_string()), None);
        assert!(objects.is_empty());

        let t = gen_frame();
        let objects = t.access_objects(true, Some("abc".to_string()), None);
        assert_eq!(objects.len(), 3);

        let t = gen_frame();
        let objects = t.access_objects(false, Some("test2".to_string()), None);
        assert_eq!(objects.len(), 2);
        assert_eq!(objects[0].id(), 1);
        assert_eq!(objects[1].id(), 2);

        let t = gen_frame();
        let objects = t.access_objects(true, Some("test2".to_string()), None);
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].id(), 0);

        let t = gen_frame();
        let objects = t.access_objects(false, Some("test2".to_string()), Some("test2".to_string()));
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].id(), 2);

        let t = gen_frame();
        let objects = t.access_objects(true, Some("test2".to_string()), Some("test2".to_string()));
        assert_eq!(objects.len(), 2);
        assert_eq!(objects[0].id(), 0);
        assert_eq!(objects[1].id(), 1);
    }
}
