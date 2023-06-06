use crate::primitives::message::video::object::InnerVideoObject;
use crate::primitives::{Attribute, VideoObject};
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
    pub(crate) object_collision_resolution_policy: ObjectUpdateCollisionResolutionPolicy,
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

    pub fn set_object_collision_resolution_policy(
        &mut self,
        p: ObjectUpdateCollisionResolutionPolicy,
    ) {
        self.object_collision_resolution_policy = p;
    }

    pub fn get_object_collision_resolution_policy(&self) -> ObjectUpdateCollisionResolutionPolicy {
        self.object_collision_resolution_policy.clone()
    }
}

#[pymethods]
impl VideoFrameUpdate {
    #[new]
    pub fn new() -> Self {
        Self {
            attributes: Vec::new(),
            objects: Vec::new(),
            object_collision_resolution_policy:
                ObjectUpdateCollisionResolutionPolicy::ErrorIfLabelsCollide,
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

    /// Sets collision resolution policy for objects
    ///
    /// Parameters
    /// ----------
    /// object: savant_rs.primitives.ObjectUpdateCollisionResolutionPolicy
    ///   The policy to set
    ///
    #[setter]
    #[pyo3(name = "object_collision_resolution_policy")]
    pub fn set_object_collision_resolution_policy_py(
        &mut self,
        p: PyObjectUpdateCollisionResolutionPolicy,
    ) {
        self.object_collision_resolution_policy = p.into();
    }

    /// Gets collision resolution policy for attributes
    ///
    /// Returns
    /// -------
    /// savant_rs.primitives.AttributeUpdateCollisionResolutionPolicy
    ///
    #[getter]
    #[pyo3(name = "attribute_collision_resolution_policy")]
    pub fn get_attribute_collision_resolution_policy_py(
        &self,
    ) -> PyAttributeUpdateCollisionResolutionPolicy {
        self.attribute_collision_resolution_policy.clone().into()
    }

    /// Gets collision resolution policy for objects
    ///
    /// Returns
    /// -------
    /// savant_rs.primitives.ObjectUpdateCollisionResolutionPolicy
    ///
    #[getter]
    #[pyo3(name = "object_collision_resolution_policy")]
    pub fn get_object_collision_resolution_policy_py(
        &self,
    ) -> PyObjectUpdateCollisionResolutionPolicy {
        self.object_collision_resolution_policy.clone().into()
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

#[derive(Debug, Clone, Archive, Deserialize, Serialize)]
#[archive(check_bytes)]
pub enum ObjectUpdateCollisionResolutionPolicy {
    AddForeignObjects,
    ErrorIfLabelsCollide,
    ReplaceSameLabelObjects,
}

#[pyclass]
#[derive(Debug, Clone)]
#[pyo3(name = "ObjectUpdateCollisionResolutionPolicy")]
pub struct PyObjectUpdateCollisionResolutionPolicy {
    pub(crate) inner: ObjectUpdateCollisionResolutionPolicy,
}

#[pymethods]
impl PyObjectUpdateCollisionResolutionPolicy {
    #[staticmethod]
    pub fn add_foreign_objects() -> Self {
        Self {
            inner: ObjectUpdateCollisionResolutionPolicy::AddForeignObjects,
        }
    }

    #[staticmethod]
    pub fn error_if_labels_collide() -> Self {
        Self {
            inner: ObjectUpdateCollisionResolutionPolicy::ErrorIfLabelsCollide,
        }
    }

    #[staticmethod]
    pub fn replace_same_label_objects() -> Self {
        Self {
            inner: ObjectUpdateCollisionResolutionPolicy::ReplaceSameLabelObjects,
        }
    }
}

impl From<ObjectUpdateCollisionResolutionPolicy> for PyObjectUpdateCollisionResolutionPolicy {
    fn from(p: ObjectUpdateCollisionResolutionPolicy) -> Self {
        Self { inner: p }
    }
}

impl From<PyObjectUpdateCollisionResolutionPolicy> for ObjectUpdateCollisionResolutionPolicy {
    fn from(p: PyObjectUpdateCollisionResolutionPolicy) -> Self {
        p.inner
    }
}

#[derive(Debug, Clone, Archive, Deserialize, Serialize)]
#[archive(check_bytes)]
pub enum AttributeUpdateCollisionResolutionPolicy {
    ReplaceWithForeignWhenDuplicate,
    KeepOwnWhenDuplicate,
    ErrorWhenDuplicate,
    PrefixDuplicates(String),
}

#[pyclass]
#[derive(Debug, Clone)]
#[pyo3(name = "AttributeUpdateCollisionResolutionPolicy")]
pub struct PyAttributeUpdateCollisionResolutionPolicy {
    pub(crate) inner: AttributeUpdateCollisionResolutionPolicy,
}

#[pymethods]
impl PyAttributeUpdateCollisionResolutionPolicy {
    #[staticmethod]
    pub fn replace_with_foreign() -> Self {
        Self {
            inner: AttributeUpdateCollisionResolutionPolicy::ReplaceWithForeignWhenDuplicate,
        }
    }

    #[staticmethod]
    pub fn keep_own() -> Self {
        Self {
            inner: AttributeUpdateCollisionResolutionPolicy::KeepOwnWhenDuplicate,
        }
    }

    #[staticmethod]
    pub fn error() -> Self {
        Self {
            inner: AttributeUpdateCollisionResolutionPolicy::ErrorWhenDuplicate,
        }
    }

    #[staticmethod]
    pub fn prefix_duplicates(prefix: String) -> Self {
        Self {
            inner: AttributeUpdateCollisionResolutionPolicy::PrefixDuplicates(prefix),
        }
    }
}

impl From<AttributeUpdateCollisionResolutionPolicy> for PyAttributeUpdateCollisionResolutionPolicy {
    fn from(value: AttributeUpdateCollisionResolutionPolicy) -> Self {
        PyAttributeUpdateCollisionResolutionPolicy { inner: value }
    }
}

impl From<PyAttributeUpdateCollisionResolutionPolicy> for AttributeUpdateCollisionResolutionPolicy {
    fn from(value: PyAttributeUpdateCollisionResolutionPolicy) -> Self {
        value.inner
    }
}
