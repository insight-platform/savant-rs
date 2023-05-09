pub mod proxy;

use crate::primitives::message::video::object::InnerObject;
use crate::primitives::{Attribute, Object};
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

    pub fn is_external(&self) -> bool {
        matches!(self.inner, VideoFrameContent::External(_))
    }

    pub fn is_internal(&self) -> bool {
        matches!(self.inner, VideoFrameContent::Internal(_))
    }

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

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub enum FrameTransformation {
    InitialSize(u64, u64),
    Scale(u64, u64),
    Padding(u64, u64, u64, u64),
    None,
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
        assert!(left > 0 && top > 0 && right > 0 && bottom > 0);
        Self {
            inner: FrameTransformation::Padding(
                u64::try_from(left).unwrap(),
                u64::try_from(top).unwrap(),
                u64::try_from(right).unwrap(),
                u64::try_from(bottom).unwrap(),
            ),
        }
    }

    #[staticmethod]
    pub fn none() -> Self {
        Self {
            inner: FrameTransformation::None,
        }
    }

    pub fn is_initial_size(&self) -> bool {
        matches!(self.inner, FrameTransformation::InitialSize(_, _))
    }

    pub fn is_scale(&self) -> bool {
        matches!(self.inner, FrameTransformation::Scale(_, _))
    }

    pub fn is_padding(&self) -> bool {
        matches!(self.inner, FrameTransformation::Padding(_, _, _, _))
    }

    pub fn is_none(&self) -> bool {
        matches!(self.inner, FrameTransformation::None)
    }

    pub fn get_initial_size(&self) -> Option<(u64, u64)> {
        match &self.inner {
            FrameTransformation::InitialSize(w, h) => Some((*w, *h)),
            _ => None,
        }
    }

    pub fn get_scale(&self) -> Option<(u64, u64)> {
        match &self.inner {
            FrameTransformation::Scale(w, h) => Some((*w, *h)),
            _ => None,
        }
    }

    pub fn get_padding(&self) -> Option<(u64, u64, u64, u64)> {
        match &self.inner {
            FrameTransformation::Padding(l, t, r, b) => Some((*l, *t, *r, *b)),
            _ => None,
        }
    }
}

#[derive(Archive, Deserialize, Serialize, Debug, Clone, derive_builder::Builder)]
#[archive(check_bytes)]
pub(crate) struct VideoFrame {
    pub(crate) source_id: String,
    pub(crate) framerate: String,
    pub(crate) width: i64,
    pub(crate) height: i64,
    pub(crate) codec: Option<String>,
    pub(crate) keyframe: Option<bool>,
    pub(crate) pts: i64,
    pub(crate) dts: Option<i64>,
    pub(crate) duration: Option<i64>,
    pub(crate) content: VideoFrameContent,
    pub(crate) transformations: Vec<FrameTransformation>,
    pub(crate) attributes: HashMap<(String, String), Attribute>,
    pub(crate) offline_objects: Vec<InnerObject>,
    #[with(Skip)]
    pub(crate) resident_objects: Vec<Arc<Mutex<InnerObject>>>,
}

impl VideoFrame {
    pub(crate) fn prepare_before_save(&mut self) {
        self.offline_objects = self
            .resident_objects
            .iter()
            .map(|o| o.lock().unwrap().clone())
            .collect();
    }

    pub(crate) fn prepare_after_load(&mut self) {
        self.resident_objects = self
            .offline_objects
            .iter()
            .map(|o| Arc::new(Mutex::new(o.clone())))
            .collect();
    }

    // #[allow(clippy::too_many_arguments)]
    // pub fn new(
    //     source_id: String,
    //     content: PyVideoFrameContent,
    //     pts: i64,
    //     framerate: String,
    //     width: i64,
    //     height: i64,
    //     attributes: HashMap<(String, String), Attribute>,
    //     objects: Vec<InnerObject>,
    //     dts: Option<i64>,
    //     duration: Option<i64>,
    //     codec: Option<String>,
    //     keyframe: Option<bool>,
    // ) -> Self {
    //     Self {
    //         source_id,
    //         pts,
    //         framerate,
    //         width,
    //         height,
    //         dts,
    //         duration,
    //         codec,
    //         keyframe,
    //         attributes,
    //         transformations: Vec::default(),
    //         offline_objects: Vec::default(),
    //         resident_objects: objects
    //             .into_iter()
    //             .map(|o| Arc::new(Mutex::new(o)))
    //             .collect(),
    //         content: content.inner,
    //     }
    // }

    // proxy
    pub fn clear_transformations(&mut self) {
        self.transformations.clear();
    }

    // proxy
    pub fn get_transformations(&self) -> Vec<FrameTransformation> {
        self.transformations.iter().map(|t| t.clone()).collect()
    }

    // proxy
    pub fn add_transformation(&mut self, transformation: FrameTransformation) {
        self.transformations.push(transformation);
    }

    // proxy
    pub fn find_attributes(
        &self,
        creator: Option<String>,
        name: Option<String>,
        hint: Option<String>,
    ) -> Vec<(String, String)> {
        self.attributes
            .iter()
            .filter(|((_, _), a)| {
                if let Some(creator) = &creator {
                    if a.creator != *creator {
                        return false;
                    }
                }

                if let Some(name) = &name {
                    if a.name != *name {
                        return false;
                    }
                }

                if let Some(hint) = &hint {
                    if a.hint.as_ref() != Some(hint) {
                        return false;
                    }
                }

                true
            })
            .map(|((c, n), _)| (c.clone(), n.clone()))
            .collect()
    }

    // proxy
    pub fn get_attribute(&self, creator: String, name: String) -> Option<Attribute> {
        self.attributes.get(&(creator, name)).cloned()
    }

    // proxy
    pub fn delete_attribute(&mut self, creator: String, name: String) -> Option<Attribute> {
        self.attributes.remove(&(creator, name))
    }

    // proxy
    pub fn set_attribute(&mut self, attribute: Attribute) -> Option<Attribute> {
        self.attributes.insert(
            (attribute.creator.clone(), attribute.name.clone()),
            attribute,
        )
    }

    // proxy
    pub fn clear_attributes(&mut self) {
        self.attributes.clear();
    }

    // proxy
    pub fn get_object(&self, id: i64) -> Option<Object> {
        self.resident_objects
            .iter()
            .find(|o| o.lock().unwrap().id == id)
            .map(|o| Object::from_arc_object(o.clone()))
    }

    // proxy
    pub fn access_objects(
        &self,
        negated: bool,
        creator: Option<String>,
        label: Option<String>,
    ) -> Vec<Object> {
        Python::with_gil(|py| {
            py.allow_threads(|| {
                self.resident_objects
                    .iter()
                    .filter(|o| {
                        let o = o.lock().unwrap();
                        let creator_match = match &creator {
                            Some(creator) => o.creator == *creator,
                            None => true,
                        };
                        let label_match = match &label {
                            Some(label) => o.label == *label,
                            None => true,
                        };
                        (creator_match && label_match) ^ negated
                    })
                    .map(|o| Object::from_arc_object(o.clone()))
                    .collect()
            })
        })
    }

    // proxy
    pub fn access_objects_by_id(&self, ids: Vec<i64>) -> Vec<Object> {
        Python::with_gil(|py| {
            py.allow_threads(|| {
                self.resident_objects
                    .iter()
                    .filter(|o| ids.contains(&o.lock().unwrap().id))
                    .map(|o| Object::from_arc_object(o.clone()))
                    .collect()
            })
        })
    }

    // proxy
    pub fn add_object(&mut self, object: Arc<Mutex<InnerObject>>) {
        self.resident_objects.push(object);
    }

    // proxy
    pub fn delete_objects_by_ids(&mut self, ids: Vec<i64>) {
        self.resident_objects
            .retain(|o| !ids.contains(&o.lock().unwrap().id));
    }

    // proxy
    pub fn delete_objects(
        &mut self,
        negated: bool,
        creator: Option<String>,
        label: Option<String>,
    ) {
        self.resident_objects.retain(|o| {
            let o = o.lock().unwrap();
            let creator_match = match &creator {
                Some(creator) => o.creator == *creator,
                None => true,
            };
            let label_match = match &label {
                Some(label) => o.label == *label,
                None => true,
            };
            !((creator_match && label_match) ^ negated)
        });
    }

    pub fn clear_objects(&mut self) {
        self.resident_objects.clear();
    }
}

#[cfg(test)]
mod tests {
    use crate::test::utils::gen_frame;

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

    #[test]
    fn test_objects_by_id() {
        pyo3::prepare_freethreaded_python();
        let t = gen_frame();
        let objects = t.access_objects_by_id(vec![0, 1]);
        assert_eq!(objects.len(), 2);
        assert_eq!(objects[0].id(), 0);
        assert_eq!(objects[1].id(), 1);
    }

    #[test]
    fn test_get_attribute() {
        pyo3::prepare_freethreaded_python();
        let t = gen_frame();
        let attribute = t.get_attribute("system".to_string(), "test".to_string());
        assert!(attribute.is_some());
        assert_eq!(
            attribute.unwrap().value.as_string().unwrap(),
            "1".to_string()
        );
    }

    #[test]
    fn test_find_attributes() {
        pyo3::prepare_freethreaded_python();
        let t = gen_frame();
        let mut attributes = t.find_attributes(Some("system".to_string()), None, None);
        attributes.sort();
        assert_eq!(attributes.len(), 2);
        assert_eq!(attributes[0], ("system".to_string(), "test".to_string()));
        assert_eq!(attributes[1], ("system".to_string(), "test2".to_string()));

        let attributes =
            t.find_attributes(Some("system".to_string()), Some("test".to_string()), None);
        assert_eq!(attributes.len(), 1);
        assert_eq!(attributes[0], ("system".to_string(), "test".to_string()));

        let attributes = t.find_attributes(
            Some("system".to_string()),
            Some("test".to_string()),
            Some("test".to_string()),
        );
        assert_eq!(attributes.len(), 1);
        assert_eq!(attributes[0], ("system".to_string(), "test".to_string()));

        let mut attributes = t.find_attributes(None, None, Some("test".to_string()));
        attributes.sort();
        assert_eq!(attributes.len(), 2);
        assert_eq!(attributes[0], ("system".to_string(), "test".to_string()));
        assert_eq!(attributes[1], ("system".to_string(), "test2".to_string()));
    }

    #[test]
    fn test_delete_objects_by_ids() {
        pyo3::prepare_freethreaded_python();
        let mut t = gen_frame();
        t.delete_objects_by_ids(vec![0, 1]);
        let objects = t.access_objects(false, None, None);
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].id(), 2);
    }

    #[test]
    fn test_delete_objects() {
        pyo3::prepare_freethreaded_python();
        let mut t = gen_frame();
        t.delete_objects(false, None, None);
        let objects = t.access_objects(false, None, None);
        assert!(objects.is_empty());

        let mut t = gen_frame();
        t.delete_objects(true, None, None);
        let objects = t.access_objects(false, None, None);
        assert_eq!(objects.len(), 3);

        let mut t = gen_frame();
        t.delete_objects(false, Some("test2".to_string()), None);
        let objects = t.access_objects(false, None, None);
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].id(), 0);

        let mut t = gen_frame();
        t.delete_objects(true, Some("test2".to_string()), None);
        let objects = t.access_objects(false, None, None);
        assert_eq!(objects.len(), 2);
        assert_eq!(objects[0].id(), 1);
        assert_eq!(objects[1].id(), 2);

        let mut t = gen_frame();
        t.delete_objects(false, Some("test2".to_string()), Some("test2".to_string()));
        let objects = t.access_objects(false, None, None);
        assert_eq!(objects.len(), 2);
        assert_eq!(objects[0].id(), 0);
        assert_eq!(objects[1].id(), 1);

        let mut t = gen_frame();
        t.delete_objects(true, Some("test2".to_string()), Some("test2".to_string()));
        let objects = t.access_objects(false, None, None);
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].id(), 2);
    }
}
