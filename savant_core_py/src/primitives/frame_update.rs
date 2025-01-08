use crate::primitives::object::VideoObject;
use crate::primitives::Attribute;
use crate::{release_gil, with_gil};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use savant_core::primitives::frame_update as rust;
use savant_core::protobuf::{from_pb, ToProtobuf};

/// Allows setting the policy for resolving collisions when updating objects in the frame with :class:`VideoFrameUpdate`.
///
/// There are three policies:
///   * the one to just add objects;
///   * the one to error if labels collide;
///   * the one to replace objects with the same label.
///
#[pyclass(eq, eq_int)]
#[derive(Clone, Debug, PartialEq)]
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
#[pyclass(eq, eq_int)]
#[derive(Clone, Debug, PartialEq)]
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
    /// attribute: :py:class:`savant_rs.primitives.Attribute`
    ///    The attribute to add
    ///
    /// Returns
    /// -------
    /// None
    ///
    pub fn add_frame_attribute(&mut self, attribute: Attribute) {
        self.0.add_frame_attribute(attribute.0);
    }

    /// Add an object attribute to the frame update.
    ///
    /// Parameters
    /// ----------
    /// object_id: int
    ///   The object id
    /// attribute: :py:class:`savant_rs.primitives.Attribute`
    ///   The attribute to add
    ///
    pub fn add_object_attribute(&mut self, object_id: i64, attribute: Attribute) {
        self.0.add_object_attribute(object_id, attribute.0);
    }

    /// Gets collision resolution policy for attributes
    ///
    /// Returns
    /// -------
    /// :py:class:`savant_rs.primitives.AttributeUpdateCollisionResolutionPolicy`
    ///
    #[getter]
    pub fn get_frame_attribute_policy(&self) -> AttributeUpdatePolicy {
        self.0.get_frame_attribute_policy().into()
    }

    /// Sets collision resolution policy for attributes
    ///
    /// Parameters
    /// ----------
    /// attribute: :py:class:`savant_rs.primitives.AttributeUpdateCollisionResolutionPolicy`
    ///    The policy to set
    ///
    /// Returns
    /// -------
    /// None
    ///
    #[setter]
    pub fn set_frame_attribute_policy(&mut self, p: AttributeUpdatePolicy) {
        self.0.set_frame_attribute_policy(p.into());
    }

    /// Gets collision resolution policy for attributes updated on objects
    ///
    /// Returns
    /// -------
    /// :py:class:`savant_rs.primitives.AttributeUpdateCollisionResolutionPolicy`
    ///
    #[getter]
    pub fn get_object_attribute_policy(&self) -> AttributeUpdatePolicy {
        self.0.get_object_attribute_policy().into()
    }

    /// Sets collision resolution policy for attributes updated on objects
    ///
    /// Parameters
    /// ----------
    /// attribute: :py:class:`savant_rs.primitives.AttributeUpdateCollisionResolutionPolicy`
    ///
    #[setter]
    pub fn set_object_attribute_policy(&mut self, p: AttributeUpdatePolicy) {
        self.0.set_object_attribute_policy(p.into());
    }

    /// Gets collision resolution policy for objects
    ///
    /// Returns
    /// -------
    /// :py:class:`savant_rs.primitives.ObjectUpdateCollisionResolutionPolicy`
    ///
    #[getter]
    #[pyo3(name = "object_policy")]
    pub fn get_object_policy(&self) -> ObjectUpdatePolicy {
        self.0.get_object_policy().into()
    }

    /// Sets collision resolution policy for objects
    ///
    /// Parameters
    /// ----------
    /// object: :py:class:`savant_rs.primitives.ObjectUpdateCollisionResolutionPolicy`
    ///   The policy to set
    ///
    #[setter]
    #[pyo3(name = "object_policy")]
    pub fn set_object_policy(&mut self, p: ObjectUpdatePolicy) {
        self.0.set_object_policy(p.into());
    }

    /// Adds an object to the frame update.
    ///
    /// Parameters
    /// ----------
    /// object: :py:class:`savant_rs.primitives.VideoObject`
    ///   The object to add
    /// parent_id: Optional[int]
    ///   The parent object id
    ///
    /// Returns
    /// -------
    /// None
    ///
    #[pyo3(signature = (object, parent_id=None))]
    pub fn add_object(&mut self, object: VideoObject, parent_id: Option<i64>) {
        self.0.add_object(object.0, parent_id);
    }

    /// Returns the list of objects
    ///
    /// Returns
    /// -------
    /// List[(savant_rs.primitives.VideoObject, Optional[int])]
    ///   The list of objects and their to-be-assigned parents
    ///
    #[getter]
    pub fn get_objects(&self) -> Vec<(VideoObject, Option<i64>)> {
        self.0
            .get_objects()
            .iter()
            .map(|(o, p)| (VideoObject(o.clone()), *p))
            .collect()
    }

    #[getter]
    pub fn json(&self) -> PyResult<String> {
        release_gil!(true, || self
            .0
            .to_json(false)
            .map_err(|e| PyValueError::new_err(e.to_string())))
    }

    #[getter]
    pub fn json_pretty(&self) -> PyResult<String> {
        release_gil!(true, || self
            .0
            .to_json(true)
            .map_err(|e| PyValueError::new_err(e.to_string())))
    }

    #[pyo3(name = "to_protobuf")]
    #[pyo3(signature = (no_gil = true))]
    fn to_protobuf_gil(&self, no_gil: bool) -> PyResult<PyObject> {
        let bytes = release_gil!(no_gil, || {
            self.0.to_pb().map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "Failed to serialize video frame update to protobuf: {}",
                    e
                ))
            })
        })?;
        with_gil!(|py| {
            let bytes = PyBytes::new_with(py, bytes.len(), |b: &mut [u8]| {
                b.copy_from_slice(&bytes);
                Ok(())
            })?;
            Ok(PyObject::from(bytes))
        })
    }

    #[staticmethod]
    #[pyo3(name = "from_protobuf")]
    #[pyo3(signature = (bytes, no_gil = true))]
    fn from_protobuf_gil(bytes: &Bound<'_, PyBytes>, no_gil: bool) -> PyResult<Self> {
        let bytes = bytes.as_bytes();
        release_gil!(no_gil, || {
            let obj =
                from_pb::<savant_core::protobuf::VideoFrameUpdate, rust::VideoFrameUpdate>(bytes)
                    .map_err(|e| {
                    PyRuntimeError::new_err(format!(
                        "Failed to deserialize video frame update from protobuf: {}",
                        e
                    ))
                })?;
            Ok(Self(obj))
        })
    }
}
