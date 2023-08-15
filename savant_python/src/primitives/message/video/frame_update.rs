use crate::primitives::message::video::object::VideoObject;
use crate::primitives::Attribute;
use pyo3::prelude::*;
use savant_core::primitives::frame_update as rust;

/// Allows setting the policy for resolving collisions when updating objects in the frame with :class:`VideoFrameUpdate`.
///
/// There are three policies:
///   * the one to just add objects;
///   * the one to error if labels collide;
///   * the one to replace objects with the same label.
///
#[pyclass]
#[derive(Clone, Debug)]
pub enum ObjectUpdatePolicy {
    AddForeignObjects,
    ErrorIfLabelsCollide,
    ReplaceSameLabelObjects,
}

impl From<ObjectUpdatePolicy> for rust::ObjectUpdatePolicy {
    fn from(p: ObjectUpdatePolicy) -> Self {
        match p {
            ObjectUpdatePolicy::AddForeignObjects => rust::ObjectUpdatePolicy::AddForeignObjects,
            ObjectUpdatePolicy::ErrorIfLabelsCollide => {
                rust::ObjectUpdatePolicy::ErrorIfLabelsCollide
            }
            ObjectUpdatePolicy::ReplaceSameLabelObjects => {
                rust::ObjectUpdatePolicy::ReplaceSameLabelObjects
            }
        }
    }
}

impl From<rust::ObjectUpdatePolicy> for ObjectUpdatePolicy {
    fn from(p: rust::ObjectUpdatePolicy) -> Self {
        match p {
            rust::ObjectUpdatePolicy::AddForeignObjects => ObjectUpdatePolicy::AddForeignObjects,
            rust::ObjectUpdatePolicy::ErrorIfLabelsCollide => {
                ObjectUpdatePolicy::ErrorIfLabelsCollide
            }
            rust::ObjectUpdatePolicy::ReplaceSameLabelObjects => {
                ObjectUpdatePolicy::ReplaceSameLabelObjects
            }
        }
    }
}

/// Allows setting the policy for resolving collisions when updating attributes in the frame with :class:`VideoFrameUpdate`.
///
/// There are four policies:
///   * the one to replace with foreign attributes when duplicates are found;
///   * the one to keep own attributes when duplicates are found;
///   * the one to error when duplicates are found;
///   * the one to prefix duplicates with a given string.
///
#[pyclass]
#[derive(Clone, Debug)]
pub enum AttributeUpdatePolicy {
    ReplaceWithForeignWhenDuplicate,
    KeepOwnWhenDuplicate,
    ErrorWhenDuplicate,
}

impl From<AttributeUpdatePolicy> for rust::AttributeUpdatePolicy {
    fn from(p: AttributeUpdatePolicy) -> Self {
        match p {
            AttributeUpdatePolicy::ReplaceWithForeignWhenDuplicate => {
                rust::AttributeUpdatePolicy::ReplaceWithForeign
            }
            AttributeUpdatePolicy::KeepOwnWhenDuplicate => rust::AttributeUpdatePolicy::KeepOwn,
            AttributeUpdatePolicy::ErrorWhenDuplicate => rust::AttributeUpdatePolicy::Error,
        }
    }
}

impl From<rust::AttributeUpdatePolicy> for AttributeUpdatePolicy {
    fn from(p: rust::AttributeUpdatePolicy) -> Self {
        match p {
            rust::AttributeUpdatePolicy::ReplaceWithForeign => {
                AttributeUpdatePolicy::ReplaceWithForeignWhenDuplicate
            }
            rust::AttributeUpdatePolicy::KeepOwn => AttributeUpdatePolicy::KeepOwnWhenDuplicate,
            rust::AttributeUpdatePolicy::Error => AttributeUpdatePolicy::ErrorWhenDuplicate,
        }
    }
}

/// A video frame update object is used to update state of a frame from external source.
///
/// It contains a list of attributes and a list of objects.
///
#[pyclass]
#[derive(Clone, Debug, Default)]
pub struct VideoFrameUpdate(pub(crate) rust::VideoFrameUpdate);

#[pymethods]
impl VideoFrameUpdate {
    #[new]
    pub fn new() -> Self {
        Self::default()
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
        self.0.add_attribute(attribute.0);
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
    #[pyo3(name = "attribute_policy")]
    pub fn set_attribute_policy(&mut self, p: AttributeUpdatePolicy) {
        self.0.set_attribute_policy(p.into());
    }

    /// Sets collision resolution policy for objects
    ///
    /// Parameters
    /// ----------
    /// object: savant_rs.primitives.ObjectUpdateCollisionResolutionPolicy
    ///   The policy to set
    ///
    #[setter]
    #[pyo3(name = "object_policy")]
    pub fn set_object_policy(&mut self, p: ObjectUpdatePolicy) {
        self.0.set_object_policy(p.into());
    }

    /// Gets collision resolution policy for attributes
    ///
    /// Returns
    /// -------
    /// savant_rs.primitives.AttributeUpdateCollisionResolutionPolicy
    ///
    #[getter]
    #[pyo3(name = "attribute_policy")]
    pub fn get_attribute_policy(&self) -> AttributeUpdatePolicy {
        self.0.get_attribute_policy().into()
    }

    /// Gets collision resolution policy for objects
    ///
    /// Returns
    /// -------
    /// savant_rs.primitives.ObjectUpdateCollisionResolutionPolicy
    ///
    #[getter]
    #[pyo3(name = "object_policy")]
    pub fn get_object_policy(&self) -> ObjectUpdatePolicy {
        self.0.get_object_policy().into()
    }

    /// Adds an object to the frame update.
    ///
    /// Parameters
    /// ----------
    /// object: savant_rs.primitives.VideoObject
    ///   The object to add
    /// parent_id: Optional[int]
    ///   The parent object id
    ///
    /// Returns
    /// -------
    /// None
    ///
    #[pyo3(signature = (object, parent_id=None))]
    pub fn add_object(&mut self, object: &VideoObject, parent_id: Option<i64>) {
        self.0.add_object(&object.0, parent_id);
    }

    /// Returns the list of objects
    ///
    /// Returns
    /// -------
    /// List[(savant_rs.primitives.VideoObject, Optional[int])]
    ///   The list of objects and their parents
    ///
    #[getter]
    pub fn get_objects(&self) -> Vec<(VideoObject, Option<i64>)> {
        self.0
            .get_objects()
            .into_iter()
            .map(|(o, p)| (VideoObject(o), p))
            .collect()
    }
}
