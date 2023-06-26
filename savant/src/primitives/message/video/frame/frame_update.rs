use crate::primitives::message::video::object::VideoObject;
use crate::primitives::{Attribute, VideoObjectProxy};
use pyo3::prelude::*;
use rkyv::{Archive, Deserialize, Serialize};

#[derive(Debug, Clone, Archive, Deserialize, Serialize)]
#[archive(check_bytes)]
pub enum ObjectUpdateCollisionResolutionPolicy {
    AddForeignObjects,
    ErrorIfLabelsCollide,
    ReplaceSameLabelObjects,
}

/// Allows setting the policy for resolving collisions when updating objects in the frame with :class:`VideoFrameUpdate`.
///
/// There are three policies:
///   * the one to just add objects;
///   * the one to error if labels collide;
///   * the one to replace objects with the same label.
///
#[pyclass]
#[derive(Debug, Clone)]
#[pyo3(name = "VideoObjectUpdateCollisionResolutionPolicy")]
pub struct VideoObjectUpdateCollisionResolutionPolicyProxy {
    pub(crate) inner: ObjectUpdateCollisionResolutionPolicy,
}

#[pymethods]
impl VideoObjectUpdateCollisionResolutionPolicyProxy {
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

impl From<ObjectUpdateCollisionResolutionPolicy>
    for VideoObjectUpdateCollisionResolutionPolicyProxy
{
    fn from(p: ObjectUpdateCollisionResolutionPolicy) -> Self {
        Self { inner: p }
    }
}

impl From<VideoObjectUpdateCollisionResolutionPolicyProxy>
    for ObjectUpdateCollisionResolutionPolicy
{
    fn from(p: VideoObjectUpdateCollisionResolutionPolicyProxy) -> Self {
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

/// Allows setting the policy for resolving collisions when updating attributes in the frame with :class:`VideoFrameUpdate`.
///
/// There are four policies:
///   * the one to replace with foreign attributes when duplicates are found;
///   * the one to keep own attributes when duplicates are found;
///   * the one to error when duplicates are found;
///   * the one to prefix duplicates with a given string.
///
#[pyclass]
#[derive(Debug, Clone)]
#[pyo3(name = "AttributeUpdateCollisionResolutionPolicy")]
pub struct AttributeUpdateCollisionResolutionPolicyProxy {
    pub(crate) inner: AttributeUpdateCollisionResolutionPolicy,
}

#[pymethods]
impl AttributeUpdateCollisionResolutionPolicyProxy {
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

impl From<AttributeUpdateCollisionResolutionPolicy>
    for AttributeUpdateCollisionResolutionPolicyProxy
{
    fn from(value: AttributeUpdateCollisionResolutionPolicy) -> Self {
        AttributeUpdateCollisionResolutionPolicyProxy { inner: value }
    }
}

impl From<AttributeUpdateCollisionResolutionPolicyProxy>
    for AttributeUpdateCollisionResolutionPolicy
{
    fn from(value: AttributeUpdateCollisionResolutionPolicyProxy) -> Self {
        value.inner
    }
}

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
    pub(crate) objects: Vec<VideoObject>,
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

impl Default for VideoFrameUpdate {
    fn default() -> Self {
        Self {
            attributes: Vec::new(),
            objects: Vec::new(),
            object_collision_resolution_policy:
                ObjectUpdateCollisionResolutionPolicy::ErrorIfLabelsCollide,
            attribute_collision_resolution_policy:
                AttributeUpdateCollisionResolutionPolicy::ErrorWhenDuplicate,
        }
    }
}

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
    pub fn add_attribute(&mut self, attribute: &Attribute) {
        self.attributes.push(attribute.clone());
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
        p: AttributeUpdateCollisionResolutionPolicyProxy,
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
        p: VideoObjectUpdateCollisionResolutionPolicyProxy,
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
    ) -> AttributeUpdateCollisionResolutionPolicyProxy {
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
    ) -> VideoObjectUpdateCollisionResolutionPolicyProxy {
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
    pub fn add_object(&mut self, object: &VideoObjectProxy) {
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
    pub fn get_objects(&self) -> Vec<VideoObjectProxy> {
        self.objects
            .iter()
            .map(|o| VideoObjectProxy::from_video_object(o.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::attribute::attribute_value::{AttributeValue, AttributeValueVariant};
    use crate::primitives::attribute::AttributeMethods;
    use crate::primitives::message::video::query::match_query::MatchQuery;
    use crate::primitives::{
        Attribute, AttributeBuilder, AttributeUpdateCollisionResolutionPolicy,
        ObjectUpdateCollisionResolutionPolicy, VideoFrameUpdate,
    };
    use crate::test::utils::{gen_frame, gen_object, s};

    #[test]
    fn update_attributes_error_when_dup() {
        let f = gen_frame();
        let (my, _) = get_attributes();
        f.set_attribute(my.clone());

        let mut upd = VideoFrameUpdate::new();
        upd.add_attribute(&my);
        upd.set_attribute_collision_resolution_policy(
            AttributeUpdateCollisionResolutionPolicy::ErrorWhenDuplicate,
        );

        let res = f.update_attributes(&upd);
        assert!(res.is_err());
    }

    fn get_attributes() -> (Attribute, Attribute) {
        (
            AttributeBuilder::default()
                .creator("system".into())
                .name("test".into())
                .hint(None)
                .hint(Some("test".into()))
                .values(vec![AttributeValue::boolean(true, None)])
                .build()
                .unwrap(),
            AttributeBuilder::default()
                .creator("system".into())
                .name("test".into())
                .hint(None)
                .hint(Some("test".into()))
                .values(vec![AttributeValue::integer(10, None)])
                .build()
                .unwrap(),
        )
    }

    #[test]
    fn update_attributes_replace_when_dup() {
        let f = gen_frame();
        let (my, their) = get_attributes();
        f.set_attribute(my);

        let mut upd = VideoFrameUpdate::new();
        upd.add_attribute(&their);
        upd.set_attribute_collision_resolution_policy(
            AttributeUpdateCollisionResolutionPolicy::ReplaceWithForeignWhenDuplicate,
        );

        let res = f.update_attributes(&upd);
        assert!(res.is_ok());
        let attr = f.get_attribute(s("system"), s("test")).unwrap();
        let vals = attr.get_values();
        let v = &vals[0];
        assert!(matches!(v.v, AttributeValueVariant::Integer(10)));
    }

    #[test]
    fn update_attributes_keep_own_when_dup() {
        let f = gen_frame();
        let (my, their) = get_attributes();
        f.set_attribute(my);

        let mut upd = VideoFrameUpdate::new();
        upd.add_attribute(&their);
        upd.set_attribute_collision_resolution_policy(
            AttributeUpdateCollisionResolutionPolicy::KeepOwnWhenDuplicate,
        );

        let res = f.update_attributes(&upd);
        assert!(res.is_ok());
        let attr = f.get_attribute(s("system"), s("test")).unwrap();
        let vals = attr.get_values();
        let v = &vals[0];
        assert!(matches!(v.v, AttributeValueVariant::Boolean(true)));
    }

    #[test]
    fn update_attributes_prefix_when_dup() {
        let f = gen_frame();
        let (my, their) = get_attributes();
        f.set_attribute(my);

        let mut upd = VideoFrameUpdate::new();
        upd.add_attribute(&their);
        upd.set_attribute_collision_resolution_policy(
            AttributeUpdateCollisionResolutionPolicy::PrefixDuplicates(s("conflict_")),
        );

        let res = f.update_attributes(&upd);
        assert!(res.is_ok());

        let attr = f.get_attribute(s("system"), s("test")).unwrap();
        let vals = attr.get_values();
        let v = &vals[0];
        assert!(matches!(v.v, AttributeValueVariant::Boolean(true)));

        let attr = f.get_attribute(s("system"), s("conflict_test")).unwrap();
        let vals = attr.get_values();
        let v = &vals[0];
        assert!(matches!(v.v, AttributeValueVariant::Integer(10)));
    }

    #[test]
    fn test_update_objects_add_foreign_objects() {
        let f = gen_frame();
        let o1 = gen_object(1);
        let o2 = gen_object(2);
        let mut upd = VideoFrameUpdate::new();
        upd.add_object(&o1);
        upd.add_object(&o2);
        upd.set_object_collision_resolution_policy(
            ObjectUpdateCollisionResolutionPolicy::AddForeignObjects,
        );
        let res = f.update_objects(&upd);
        assert!(res.is_ok());
        assert_eq!(f.get_max_object_id(), 4);
        let o = f.get_object(4).unwrap();
        assert_eq!(o.get_creator(), s("peoplenet"));
        assert_eq!(f.access_objects(&MatchQuery::Idle).len(), 5);
    }

    #[test]
    fn test_update_error_labels_collide() {
        let f = gen_frame();
        let o1 = gen_object(1);
        let mut upd = VideoFrameUpdate::new();
        upd.add_object(&o1);
        upd.set_object_collision_resolution_policy(
            ObjectUpdateCollisionResolutionPolicy::ErrorIfLabelsCollide,
        );
        let res = f.update_objects(&upd);
        assert!(res.is_ok());
        assert_eq!(f.access_objects(&MatchQuery::Idle).len(), 4);

        let o2 = gen_object(2);
        let mut upd = VideoFrameUpdate::new();
        upd.add_object(&o2);
        upd.set_object_collision_resolution_policy(
            ObjectUpdateCollisionResolutionPolicy::ErrorIfLabelsCollide,
        );
        let res = f.update_objects(&upd);
        assert!(res.is_err());
        assert_eq!(f.access_objects(&MatchQuery::Idle).len(), 4);
    }

    #[test]
    fn test_update_replace_same_label_objects() {
        let f = gen_frame();
        let o1 = gen_object(1);
        let mut upd = VideoFrameUpdate::new();
        upd.add_object(&o1);
        upd.set_object_collision_resolution_policy(
            ObjectUpdateCollisionResolutionPolicy::ReplaceSameLabelObjects,
        );
        let res = f.update_objects(&upd);
        assert!(res.is_ok());
        assert_eq!(f.get_max_object_id(), 3);
        assert_eq!(f.access_objects(&MatchQuery::Idle).len(), 4);

        let o2 = gen_object(2);
        let mut upd = VideoFrameUpdate::new();
        upd.add_object(&o2);
        upd.set_object_collision_resolution_policy(
            ObjectUpdateCollisionResolutionPolicy::ReplaceSameLabelObjects,
        );
        let res = f.update_objects(&upd);
        assert!(res.is_ok());
        assert_eq!(f.get_max_object_id(), 4);
        assert_eq!(f.access_objects(&MatchQuery::Idle).len(), 4);
    }
}
