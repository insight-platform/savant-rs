use crate::primitives::message::video::object::InnerVideoObject;
use crate::primitives::{Attribute, VideoObject};
use pyo3::prelude::*;
use rkyv::{Archive, Deserialize, Serialize};

/// A video frame update object is used to udpate state of a frame from external source.
///
/// It contains a list of attributes and a list of objects.
///
#[pyclass]
#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[archive(check_bytes)]
pub struct VideoFrameUpdate {
    #[pyo3(get, set)]
    pub(crate) attributes: Vec<Attribute>,
    pub(crate) objects: Vec<InnerVideoObject>,
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

    /// Add an attribute to the frame update.
    ///
    /// Parameters
    /// ----------
    /// attribute: savant_rs.primitives.Attribute
    ///    The attribute to add
    ///
    /// Returns
    /// -------
    /// None
    ///
    pub fn add_attribute(&mut self, attribute: Attribute) {
        self.attributes.push(attribute);
    }

    /// Adds an object to the frame update.
    ///
    /// Parameters
    /// ----------
    /// object: savant_rs.primitives.VideoObject
    ///   The object to add
    ///
    /// Returns
    /// -------
    /// None
    ///
    pub fn add_object(&mut self, object: VideoObject) {
        self.objects.push(object.inner.read().clone());
    }

    /// Returns the list of objects
    ///
    /// Returns
    /// -------
    /// List[savant_rs.primitives.VideoObject]
    ///   The list of objects
    ///
    #[getter]
    pub fn get_objects(&self) -> Vec<VideoObject> {
        self.objects
            .iter()
            .map(|o| VideoObject::from_inner_object(o.clone()))
            .collect()
    }
}
