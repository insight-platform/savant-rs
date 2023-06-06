use crate::primitives::attribute::AttributeUpdateCollisionResolutionPolicy;
use crate::primitives::message::video::object::InnerVideoObject;
use crate::primitives::{Attribute, PyAttributeUpdateCollisionResolutionPolicy, VideoObject};
use pyo3::prelude::*;
use rkyv::{Archive, Deserialize, Serialize};

/// A video frame update object is used to update state of a frame from external source.
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
    pub(crate) attribute_collision_resolution_policy: AttributeUpdateCollisionResolutionPolicy,
}

impl VideoFrameUpdate {
    pub fn set_attribute_collision_resolution_policy(
        &mut self,
        p: AttributeUpdateCollisionResolutionPolicy,
    ) {
        self.attribute_collision_resolution_policy = p;
    }

    pub fn get_attribute_collision_resolution_policy(
        &self,
    ) -> AttributeUpdateCollisionResolutionPolicy {
        self.attribute_collision_resolution_policy.clone()
    }
}

#[pymethods]
impl VideoFrameUpdate {
    #[new]
    pub fn new() -> Self {
        Self {
            attributes: Vec::new(),
            objects: Vec::new(),
            attribute_collision_resolution_policy:
                AttributeUpdateCollisionResolutionPolicy::ErrorWhenDuplicate,
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

    /// Sets collision resolution policy for attributes
    ///
    /// Parameters
    /// ----------
    /// attribute: savant_rs.primitives.AttributeUpdateCollisionResolutionPolicy
    ///    The policy to set
    ///
    /// Returns
    /// -------
    /// None
    ///
    #[setter]
    #[pyo3(name = "attribute_collision_resolution_policy")]
    pub fn set_attribute_collision_resolution_policy_py(
        &mut self,
        p: PyAttributeUpdateCollisionResolutionPolicy,
    ) {
        self.attribute_collision_resolution_policy = p.into();
    }

    /// Gets collision resolution policy for attributes
    ///
    /// Returns
    /// -------
    /// attribute: savant_rs.primitives.AttributeUpdateCollisionResolutionPolicy
    ///
    #[getter]
    #[pyo3(name = "attribute_collision_resolution_policy")]
    pub fn get_attribute_collision_resolution_policy_py(
        &self,
    ) -> PyAttributeUpdateCollisionResolutionPolicy {
        self.attribute_collision_resolution_policy.clone().into()
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
