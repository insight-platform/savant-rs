use crate::primitives::object::{VideoObject, VideoObjectProxy};
use crate::primitives::Attribute;
use crate::trace;
use rkyv::{Archive, Deserialize, Serialize};

#[derive(Debug, Clone, Archive, Deserialize, Serialize, serde::Serialize, serde::Deserialize)]
#[archive(check_bytes)]
pub enum ObjectUpdatePolicy {
    AddForeignObjects,
    ErrorIfLabelsCollide,
    ReplaceSameLabelObjects,
}

#[derive(Debug, Clone, Archive, Deserialize, Serialize, serde::Serialize, serde::Deserialize)]
#[archive(check_bytes)]
pub enum AttributeUpdatePolicy {
    ReplaceWithForeign,
    KeepOwn,
    Error,
}

/// A video frame update object is used to update state of a frame from external source.
///
/// It contains a list of attributes and a list of objects.
///
#[derive(Archive, Deserialize, Serialize, Debug, Clone, serde::Serialize, serde::Deserialize)]
#[archive(check_bytes)]
pub struct VideoFrameUpdate {
    frame_attributes: Vec<Attribute>,
    pub(crate) object_attributes: Vec<(i64, Attribute)>,
    pub(crate) objects: Vec<(VideoObject, Option<i64>)>,
    pub(crate) frame_attribute_policy: AttributeUpdatePolicy,
    pub(crate) object_attribute_policy: AttributeUpdatePolicy,
    pub(crate) object_policy: ObjectUpdatePolicy,
}

impl Default for VideoFrameUpdate {
    fn default() -> Self {
        Self {
            frame_attributes: Vec::new(),
            object_attributes: Vec::new(),
            objects: Vec::new(),
            object_policy: ObjectUpdatePolicy::ErrorIfLabelsCollide,
            frame_attribute_policy: AttributeUpdatePolicy::Error,
            object_attribute_policy: AttributeUpdatePolicy::Error,
        }
    }
}

impl VideoFrameUpdate {
    pub(crate) fn get_frame_attributes(&self) -> &Vec<Attribute> {
        &self.frame_attributes
    }

    pub(crate) fn get_object_attributes(&self) -> &Vec<(i64, Attribute)> {
        &self.object_attributes
    }

    pub fn set_frame_attribute_policy(&mut self, p: AttributeUpdatePolicy) {
        self.frame_attribute_policy = p;
    }

    pub fn get_frame_attribute_policy(&self) -> AttributeUpdatePolicy {
        self.frame_attribute_policy.clone()
    }

    pub fn set_object_policy(&mut self, p: ObjectUpdatePolicy) {
        self.object_policy = p;
    }

    pub fn get_object_policy(&self) -> ObjectUpdatePolicy {
        self.object_policy.clone()
    }

    pub fn set_object_attribute_policy(&mut self, p: AttributeUpdatePolicy) {
        self.object_attribute_policy = p;
    }

    pub fn get_object_attribute_policy(&self) -> AttributeUpdatePolicy {
        self.object_attribute_policy.clone()
    }

    pub fn add_frame_attribute(&mut self, attribute: Attribute) {
        self.frame_attributes.push(attribute);
    }

    pub fn add_object_attribute(&mut self, object_id: i64, attribute: Attribute) {
        self.object_attributes.push((object_id, attribute));
    }

    pub fn add_object(&mut self, object: &VideoObjectProxy, parent_id: Option<i64>) {
        self.objects
            .push((trace!(object.inner.read()).clone(), parent_id));
    }

    pub fn get_objects(&self) -> Vec<(VideoObjectProxy, Option<i64>)> {
        self.objects
            .iter()
            .map(|(o, p)| (VideoObjectProxy::from(o.clone()), *p))
            .collect()
    }

    pub fn to_json(&self, pretty: bool) -> anyhow::Result<String> {
        Ok(if pretty {
            serde_json::to_string_pretty(self)?
        } else {
            serde_json::to_string(self)?
        })
    }

    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(json)?)
    }
}

#[cfg(test)]
mod tests {
    use crate::match_query::{IntExpression, MatchQuery};
    use crate::primitives::attribute_value::{AttributeValue, AttributeValueVariant};
    use crate::primitives::frame_update::{
        AttributeUpdatePolicy, ObjectUpdatePolicy, VideoFrameUpdate,
    };
    use crate::primitives::{Attribute, AttributeMethods};
    use crate::test::{gen_frame, gen_object, s};

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
                false,
            ),
            Attribute::persistent(
                "system".into(),
                "test".into(),
                vec![AttributeValue::new(
                    AttributeValueVariant::Integer(10),
                    None,
                )],
                Some("test".into()),
                false,
            ),
        )
    }

    fn get_object_attributes() -> Vec<(i64, Attribute)> {
        let (a1, a2) = get_attributes();
        vec![(1, a1), (1, a2)]
    }

    #[test]
    fn update_attributes_error_when_dup() {
        let f = gen_frame();
        let (my, _) = get_attributes();
        f.set_attribute(my.clone());

        let mut upd = VideoFrameUpdate::default();
        upd.add_frame_attribute(my);
        upd.set_frame_attribute_policy(AttributeUpdatePolicy::Error);

        let res = f.update_frame_attributes(&upd);
        assert!(res.is_err());
    }

    #[test]
    fn update_object_attributes_error_when_dup() {
        let f = gen_frame();
        let attrs = get_object_attributes();

        for (id, mut attr) in attrs.clone() {
            attr.make_temporary();
            let o = f.get_object(id).unwrap();
            o.set_attribute(attr);
        }

        let mut upd = VideoFrameUpdate::default();
        for (id, attr) in attrs.clone() {
            upd.add_object_attribute(id, attr);
        }
        upd.set_object_attribute_policy(AttributeUpdatePolicy::Error);
        let res = f.update(&upd);
        assert!(res.is_err());
    }

    #[test]
    fn update_attributes_replace_when_dup() {
        let f = gen_frame();
        let (my, their) = get_attributes();
        f.set_attribute(my);

        let mut upd = VideoFrameUpdate::default();
        upd.add_frame_attribute(their);
        upd.set_frame_attribute_policy(AttributeUpdatePolicy::ReplaceWithForeign);

        let res = f.update_frame_attributes(&upd);
        assert!(res.is_ok());
        let attr = f.get_attribute(s("system"), s("test")).unwrap();
        let vals = attr.get_values();
        let v = &vals[0];
        assert!(matches!(v.get_value(), AttributeValueVariant::Integer(10)));
    }

    #[test]
    fn update_object_attributes_replace_when_dup() {
        let f = gen_frame();
        let attrs = get_object_attributes();

        for (id, mut attr) in attrs.clone() {
            attr.make_temporary();
            let o = f.get_object(id).unwrap();
            o.set_attribute(attr);
        }

        let mut upd = VideoFrameUpdate::default();
        for (id, attr) in attrs.clone() {
            upd.add_object_attribute(id, attr);
        }
        upd.set_object_attribute_policy(AttributeUpdatePolicy::ReplaceWithForeign);
        f.update(&upd).unwrap();

        for (id, attr) in attrs {
            let o = f.get_object(id).unwrap();
            let attr = o.get_attribute(attr.namespace, attr.name).unwrap();
            assert!(attr.is_persistent);
        }
    }

    #[test]
    fn update_attributes_keep_own_when_dup() {
        let f = gen_frame();
        let (my, their) = get_attributes();
        f.set_attribute(my);

        let mut upd = VideoFrameUpdate::default();
        upd.add_frame_attribute(their);
        upd.set_frame_attribute_policy(AttributeUpdatePolicy::KeepOwn);

        let res = f.update_frame_attributes(&upd);
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
    fn update_object_attributes_keep_own_when_dup() {
        let f = gen_frame();
        let attrs = get_object_attributes();

        for (id, mut attr) in attrs.clone() {
            attr.make_temporary();
            let o = f.get_object(id).unwrap();
            o.set_attribute(attr);
        }

        let mut upd = VideoFrameUpdate::default();
        for (id, attr) in attrs.clone() {
            upd.add_object_attribute(id, attr);
        }
        upd.set_object_attribute_policy(AttributeUpdatePolicy::KeepOwn);
        f.update(&upd).unwrap();

        for (id, attr) in attrs {
            let o = f.get_object(id).unwrap();
            let attr = o.get_attribute(attr.namespace, attr.name).unwrap();
            assert!(!attr.is_persistent);
        }
    }

    #[test]
    fn test_update_objects_add_foreign_objects() {
        let f = gen_frame();
        let o1 = gen_object(1);
        let o2 = gen_object(2);
        let mut upd = VideoFrameUpdate::default();
        upd.add_object(&o1, None);
        upd.add_object(&o2, None);
        upd.set_object_policy(ObjectUpdatePolicy::AddForeignObjects);
        let res = f.update_objects(&upd);
        assert!(res.is_ok());
        assert_eq!(f.get_max_object_id(), 4);
        let o = f.get_object(4).unwrap();
        assert_eq!(o.get_namespace(), s("peoplenet"));
        assert_eq!(f.get_all_objects().len(), 5);
    }

    #[test]
    fn test_update_error_labels_collide() {
        let f = gen_frame();
        let o1 = gen_object(1);
        let mut upd = VideoFrameUpdate::default();
        upd.add_object(&o1, None);
        upd.set_object_policy(ObjectUpdatePolicy::ErrorIfLabelsCollide);
        let res = f.update_objects(&upd);
        assert!(res.is_ok());
        assert_eq!(f.get_all_objects().len(), 4);

        let o2 = gen_object(2);
        let mut upd = VideoFrameUpdate::default();
        upd.add_object(&o2, None);
        upd.set_object_policy(ObjectUpdatePolicy::ErrorIfLabelsCollide);
        let res = f.update_objects(&upd);
        assert!(res.is_err());
        assert_eq!(f.get_all_objects().len(), 4);
    }

    #[test]
    fn test_update_replace_same_label_objects() {
        let f = gen_frame();
        let o1 = gen_object(1);
        let mut upd = VideoFrameUpdate::default();
        upd.add_object(&o1, None);
        upd.set_object_policy(ObjectUpdatePolicy::ReplaceSameLabelObjects);
        let res = f.update_objects(&upd);
        assert!(res.is_ok());
        assert_eq!(f.get_max_object_id(), 3);
        assert_eq!(f.get_all_objects().len(), 4);

        let o2 = gen_object(2);
        let mut upd = VideoFrameUpdate::default();
        upd.add_object(&o2, None);
        upd.set_object_policy(ObjectUpdatePolicy::ReplaceSameLabelObjects);
        let res = f.update_objects(&upd);
        assert!(res.is_ok());
        assert_eq!(f.get_max_object_id(), 4);
        assert_eq!(f.get_all_objects().len(), 4);
    }

    #[test]
    fn update_objects_with_parent() {
        let f = gen_frame();
        let p = f.get_object(1).unwrap();
        let o = gen_object(100);
        let mut upd = VideoFrameUpdate::default();
        upd.add_object(&o, Some(p.get_id()));
        upd.set_object_policy(ObjectUpdatePolicy::AddForeignObjects);
        let res = f.update_objects(&upd);
        assert!(res.is_ok());

        let o = f.access_objects(&MatchQuery::ParentId(IntExpression::EQ(1)));
        assert_eq!(o[0].get_parent().unwrap().get_id(), 1);
    }
}
