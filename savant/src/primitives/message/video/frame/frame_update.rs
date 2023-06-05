use crate::primitives::{Attribute, VideoObject};
use pyo3::prelude::*;

#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct VideoFrameUpdate {
    #[pyo3(get, set)]
    pub attributes: Vec<Attribute>,
    #[pyo3(get, set)]
    pub objects: Vec<VideoObject>,
}

#[pymethods]
impl VideoFrameUpdate {
    #[new]
    pub fn new() -> Self {
        Self {
            attributes: Vec::new(),
            objects: Vec::new(),
        }
    }

    pub fn add_attribute(&mut self, attribute: Attribute) {
        self.attributes.push(attribute);
    }

    pub fn add_object(&mut self, object: VideoObject) {
        self.objects.push(object);
    }
}
