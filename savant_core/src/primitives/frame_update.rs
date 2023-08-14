use crate::primitives::object::{VideoObject, VideoObjectProxy};
use crate::primitives::Attribute;
use rkyv::{Archive, Deserialize, Serialize};

#[derive(Debug, Clone, Archive, Deserialize, Serialize)]
#[archive(check_bytes)]
pub enum ObjectUpdateCollisionResolutionPolicy {
    AddForeignObjects,
    ErrorIfLabelsCollide,
    ReplaceSameLabelObjects,
}

#[derive(Debug, Clone, Archive, Deserialize, Serialize)]
#[archive(check_bytes)]
pub enum AttributeUpdateCollisionResolutionPolicy {
    ReplaceWithForeignWhenDuplicate,
    KeepOwnWhenDuplicate,
    ErrorWhenDuplicate,
    PrefixDuplicates(String),
}

/// A video frame update object is used to update state of a frame from external source.
///
/// It contains a list of attributes and a list of objects.
///
#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[archive(check_bytes)]
pub struct VideoFrameUpdate {
    attributes: Vec<Attribute>,
    pub(crate) objects: Vec<(VideoObject, Option<i64>)>,
    pub(crate) attribute_collision_resolution_policy: AttributeUpdateCollisionResolutionPolicy,
    pub(crate) object_collision_resolution_policy: ObjectUpdateCollisionResolutionPolicy,
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

impl VideoFrameUpdate {
    pub(crate) fn get_attributes(&self) -> &Vec<Attribute> {
        &self.attributes
    }
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

    pub fn add_attribute(&mut self, attribute: Attribute) {
        self.attributes.push(attribute);
    }

    pub fn add_object(&mut self, object: &VideoObjectProxy, parent_id: Option<i64>) {
        self.objects.push((object.inner.read().clone(), parent_id));
    }

    pub fn get_objects(&self) -> Vec<(VideoObjectProxy, Option<i64>)> {
        self.objects
            .iter()
            .map(|(o, p)| (VideoObjectProxy::from_video_object(o.clone()), *p))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::match_query::{IntExpression, MatchQuery};
    use crate::primitives::attribute_value::{AttributeValue, AttributeValueVariant};
    use crate::primitives::frame_update::{
        AttributeUpdateCollisionResolutionPolicy, ObjectUpdateCollisionResolutionPolicy,
        VideoFrameUpdate,
    };
    use crate::primitives::{Attribute, AttributeMethods};
    use crate::test::{gen_frame, gen_object, s};

    #[test]
    fn update_attributes_error_when_dup() {
        let f = gen_frame();
        let (my, _) = get_attributes();
        f.set_attribute(my.clone());

        let mut upd = VideoFrameUpdate::default();
        upd.add_attribute(my);
        upd.set_attribute_collision_resolution_policy(
            AttributeUpdateCollisionResolutionPolicy::ErrorWhenDuplicate,
        );

        let res = f.update_attributes(&upd);
        assert!(res.is_err());
    }

    fn get_attributes() -> (Attribute, Attribute) {
        (
            Attribute::persistent(
                "system".into(),
                "test".into(),
                vec![AttributeValue::new(
                    AttributeValueVariant::Boolean(true),
                    None,
                )],
                Some("test".into()),
            ),
            Attribute::persistent(
                "system".into(),
                "test".into(),
                vec![AttributeValue::new(
                    AttributeValueVariant::Integer(10),
                    None,
                )],
                Some("test".into()),
            ),
        )
    }

    #[test]
    fn update_attributes_replace_when_dup() {
        let f = gen_frame();
        let (my, their) = get_attributes();
        f.set_attribute(my);

        let mut upd = VideoFrameUpdate::default();
        upd.add_attribute(their);
        upd.set_attribute_collision_resolution_policy(
            AttributeUpdateCollisionResolutionPolicy::ReplaceWithForeignWhenDuplicate,
        );

        let res = f.update_attributes(&upd);
        assert!(res.is_ok());
        let attr = f.get_attribute(s("system"), s("test")).unwrap();
        let vals = attr.get_values();
        let v = &vals[0];
        assert!(matches!(v.get_value(), AttributeValueVariant::Integer(10)));
    }

    #[test]
    fn update_attributes_keep_own_when_dup() {
        let f = gen_frame();
        let (my, their) = get_attributes();
        f.set_attribute(my);

        let mut upd = VideoFrameUpdate::default();
        upd.add_attribute(their);
        upd.set_attribute_collision_resolution_policy(
            AttributeUpdateCollisionResolutionPolicy::KeepOwnWhenDuplicate,
        );

        let res = f.update_attributes(&upd);
        assert!(res.is_ok());
        let attr = f.get_attribute(s("system"), s("test")).unwrap();
        let vals = attr.get_values();
        let v = &vals[0];
        assert!(matches!(
            v.get_value(),
            AttributeValueVariant::Boolean(true)
        ));
    }

    #[test]
    fn update_attributes_prefix_when_dup() {
        let f = gen_frame();
        let (my, their) = get_attributes();
        f.set_attribute(my);

        let mut upd = VideoFrameUpdate::default();
        upd.add_attribute(their);
        upd.set_attribute_collision_resolution_policy(
            AttributeUpdateCollisionResolutionPolicy::PrefixDuplicates(s("conflict_")),
        );

        let res = f.update_attributes(&upd);
        assert!(res.is_ok());

        let attr = f.get_attribute(s("system"), s("test")).unwrap();
        let vals = attr.get_values();
        let v = &vals[0];
        assert!(matches!(
            v.get_value(),
            AttributeValueVariant::Boolean(true)
        ));

        let attr = f.get_attribute(s("system"), s("conflict_test")).unwrap();
        let vals = attr.get_values();
        let v = &vals[0];
        assert!(matches!(v.get_value(), AttributeValueVariant::Integer(10)));
    }

    #[test]
    fn test_update_objects_add_foreign_objects() {
        let f = gen_frame();
        let o1 = gen_object(1);
        let o2 = gen_object(2);
        let mut upd = VideoFrameUpdate::default();
        upd.add_object(&o1, None);
        upd.add_object(&o2, None);
        upd.set_object_collision_resolution_policy(
            ObjectUpdateCollisionResolutionPolicy::AddForeignObjects,
        );
        let res = f.update_objects(&upd);
        assert!(res.is_ok());
        assert_eq!(f.get_max_object_id(), 4);
        let o = f.get_object(4).unwrap();
        assert_eq!(o.get_namespace(), s("peoplenet"));
        assert_eq!(f.access_objects(&MatchQuery::Idle).len(), 5);
    }

    #[test]
    fn test_update_error_labels_collide() {
        let f = gen_frame();
        let o1 = gen_object(1);
        let mut upd = VideoFrameUpdate::default();
        upd.add_object(&o1, None);
        upd.set_object_collision_resolution_policy(
            ObjectUpdateCollisionResolutionPolicy::ErrorIfLabelsCollide,
        );
        let res = f.update_objects(&upd);
        assert!(res.is_ok());
        assert_eq!(f.access_objects(&MatchQuery::Idle).len(), 4);

        let o2 = gen_object(2);
        let mut upd = VideoFrameUpdate::default();
        upd.add_object(&o2, None);
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
        let mut upd = VideoFrameUpdate::default();
        upd.add_object(&o1, None);
        upd.set_object_collision_resolution_policy(
            ObjectUpdateCollisionResolutionPolicy::ReplaceSameLabelObjects,
        );
        let res = f.update_objects(&upd);
        assert!(res.is_ok());
        assert_eq!(f.get_max_object_id(), 3);
        assert_eq!(f.access_objects(&MatchQuery::Idle).len(), 4);

        let o2 = gen_object(2);
        let mut upd = VideoFrameUpdate::default();
        upd.add_object(&o2, None);
        upd.set_object_collision_resolution_policy(
            ObjectUpdateCollisionResolutionPolicy::ReplaceSameLabelObjects,
        );
        let res = f.update_objects(&upd);
        assert!(res.is_ok());
        assert_eq!(f.get_max_object_id(), 4);
        assert_eq!(f.access_objects(&MatchQuery::Idle).len(), 4);
    }

    #[test]
    fn update_objects_with_parent() {
        let f = gen_frame();
        let p = f.get_object(1).unwrap();
        let o = gen_object(100);
        let mut upd = VideoFrameUpdate::default();
        upd.add_object(&o, Some(p.get_id()));
        upd.set_object_collision_resolution_policy(
            ObjectUpdateCollisionResolutionPolicy::AddForeignObjects,
        );
        let res = f.update_objects(&upd);
        assert!(res.is_ok());

        let o = f.access_objects(&MatchQuery::ParentId(IntExpression::EQ(1)));
        assert_eq!(o[0].get_parent().unwrap().get_id(), 1);
    }
}
