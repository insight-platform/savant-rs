use crate::primitives::frame_update::{
    AttributeUpdatePolicy, ObjectUpdatePolicy, VideoFrameUpdate,
};
use crate::primitives::object::VideoObject;
use crate::primitives::Attribute;
use crate::protobuf::{generated, serialize};

impl From<AttributeUpdatePolicy> for generated::AttributeUpdatePolicy {
    fn from(p: AttributeUpdatePolicy) -> Self {
        match p {
            AttributeUpdatePolicy::ReplaceWithForeign => {
                generated::AttributeUpdatePolicy::ReplaceWithForeign
            }
            AttributeUpdatePolicy::KeepOwn => generated::AttributeUpdatePolicy::KeepOwn,
            AttributeUpdatePolicy::Error => generated::AttributeUpdatePolicy::Error,
        }
    }
}

impl From<&generated::AttributeUpdatePolicy> for AttributeUpdatePolicy {
    fn from(p: &generated::AttributeUpdatePolicy) -> Self {
        match p {
            generated::AttributeUpdatePolicy::ReplaceWithForeign => {
                AttributeUpdatePolicy::ReplaceWithForeign
            }
            generated::AttributeUpdatePolicy::KeepOwn => AttributeUpdatePolicy::KeepOwn,
            generated::AttributeUpdatePolicy::Error => AttributeUpdatePolicy::Error,
        }
    }
}

impl From<ObjectUpdatePolicy> for generated::ObjectUpdatePolicy {
    fn from(p: ObjectUpdatePolicy) -> Self {
        match p {
            ObjectUpdatePolicy::AddForeignObjects => {
                generated::ObjectUpdatePolicy::AddForeignObjects
            }
            ObjectUpdatePolicy::ErrorIfLabelsCollide => {
                generated::ObjectUpdatePolicy::ErrorIfLabelsCollide
            }
            ObjectUpdatePolicy::ReplaceSameLabelObjects => {
                generated::ObjectUpdatePolicy::ReplaceSameLabelObjects
            }
        }
    }
}

impl From<&generated::ObjectUpdatePolicy> for ObjectUpdatePolicy {
    fn from(p: &generated::ObjectUpdatePolicy) -> Self {
        match p {
            generated::ObjectUpdatePolicy::AddForeignObjects => {
                ObjectUpdatePolicy::AddForeignObjects
            }
            generated::ObjectUpdatePolicy::ErrorIfLabelsCollide => {
                ObjectUpdatePolicy::ErrorIfLabelsCollide
            }
            generated::ObjectUpdatePolicy::ReplaceSameLabelObjects => {
                ObjectUpdatePolicy::ReplaceSameLabelObjects
            }
        }
    }
}

impl From<&VideoFrameUpdate> for generated::VideoFrameUpdate {
    fn from(vfu: &VideoFrameUpdate) -> Self {
        let frame_attributes = vfu
            .get_frame_attributes()
            .iter()
            .filter(|a| a.is_persistent)
            .map(|a| a.into())
            .collect();

        let object_attributes = vfu
            .get_object_attributes()
            .iter()
            .filter(|oa| oa.1.is_persistent)
            .map(|oa| generated::ObjectAttribute {
                object_id: oa.0,
                attribute: Some(generated::Attribute::from(&oa.1)),
            })
            .collect();

        let objects = vfu.get_objects().iter().map(|o| o.into()).collect();

        let frame_attribute_policy =
            generated::AttributeUpdatePolicy::from(vfu.get_frame_attribute_policy()) as i32;

        let object_attribute_policy =
            generated::AttributeUpdatePolicy::from(vfu.get_object_attribute_policy()) as i32;

        let object_policy = generated::ObjectUpdatePolicy::from(vfu.get_object_policy()) as i32;

        generated::VideoFrameUpdate {
            frame_attributes,
            object_attributes,
            objects,
            frame_attribute_policy,
            object_attribute_policy,
            object_policy,
        }
    }
}

impl TryFrom<&generated::VideoFrameUpdate> for VideoFrameUpdate {
    type Error = serialize::Error;

    fn try_from(value: &generated::VideoFrameUpdate) -> Result<Self, Self::Error> {
        let frame_attribute_policy =
            AttributeUpdatePolicy::from(&value.frame_attribute_policy.try_into()?);

        let object_attribute_policy =
            AttributeUpdatePolicy::from(&value.object_attribute_policy.try_into()?);

        let object_policy = ObjectUpdatePolicy::from(&value.object_policy.try_into()?);

        let object_attributes = value
            .object_attributes
            .iter()
            .map(|oa| {
                Attribute::try_from(oa.attribute.as_ref().unwrap()).map(|a| (oa.object_id, a))
            })
            .collect::<Result<_, _>>()?;

        let frame_attributes = value
            .frame_attributes
            .iter()
            .map(Attribute::try_from)
            .collect::<Result<_, _>>()?;

        let objects = value
            .objects
            .iter()
            .map(|so| VideoObject::try_from(so.object.as_ref().unwrap()).map(|o| (o, so.parent_id)))
            .collect::<Result<_, _>>()?;

        Ok(VideoFrameUpdate {
            frame_attributes,
            object_attributes,
            objects,
            frame_attribute_policy,
            object_attribute_policy,
            object_policy,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::attribute_value::AttributeValue;
    use crate::primitives::frame_update::{
        AttributeUpdatePolicy, ObjectUpdatePolicy, VideoFrameUpdate,
    };
    use crate::primitives::object::ObjectOperations;
    use crate::primitives::Attribute;
    use crate::protobuf::generated;
    use crate::test::gen_object;

    #[test]
    fn test_attribute_update_policy() {
        assert_eq!(
            generated::AttributeUpdatePolicy::ReplaceWithForeign,
            generated::AttributeUpdatePolicy::from(AttributeUpdatePolicy::ReplaceWithForeign)
        );
        assert_eq!(
            generated::AttributeUpdatePolicy::KeepOwn,
            generated::AttributeUpdatePolicy::from(AttributeUpdatePolicy::KeepOwn)
        );
        assert_eq!(
            generated::AttributeUpdatePolicy::Error,
            generated::AttributeUpdatePolicy::from(AttributeUpdatePolicy::Error)
        );
        assert_eq!(
            AttributeUpdatePolicy::ReplaceWithForeign,
            AttributeUpdatePolicy::from(&generated::AttributeUpdatePolicy::ReplaceWithForeign)
        );
        assert_eq!(
            AttributeUpdatePolicy::KeepOwn,
            AttributeUpdatePolicy::from(&generated::AttributeUpdatePolicy::KeepOwn)
        );
        assert_eq!(
            AttributeUpdatePolicy::Error,
            AttributeUpdatePolicy::from(&generated::AttributeUpdatePolicy::Error)
        );
    }

    #[test]
    fn test_object_update_policy() {
        assert_eq!(
            generated::ObjectUpdatePolicy::AddForeignObjects,
            generated::ObjectUpdatePolicy::from(ObjectUpdatePolicy::AddForeignObjects)
        );
        assert_eq!(
            generated::ObjectUpdatePolicy::ErrorIfLabelsCollide,
            generated::ObjectUpdatePolicy::from(ObjectUpdatePolicy::ErrorIfLabelsCollide)
        );
        assert_eq!(
            generated::ObjectUpdatePolicy::ReplaceSameLabelObjects,
            generated::ObjectUpdatePolicy::from(ObjectUpdatePolicy::ReplaceSameLabelObjects)
        );
        assert_eq!(
            ObjectUpdatePolicy::AddForeignObjects,
            ObjectUpdatePolicy::from(&generated::ObjectUpdatePolicy::AddForeignObjects)
        );
        assert_eq!(
            ObjectUpdatePolicy::ErrorIfLabelsCollide,
            ObjectUpdatePolicy::from(&generated::ObjectUpdatePolicy::ErrorIfLabelsCollide)
        );
        assert_eq!(
            ObjectUpdatePolicy::ReplaceSameLabelObjects,
            ObjectUpdatePolicy::from(&generated::ObjectUpdatePolicy::ReplaceSameLabelObjects)
        );
    }

    #[test]
    fn test_video_frame_update() {
        let obj = gen_object(0);
        let frame_attr = Attribute::persistent(
            "system2",
            "test2",
            vec![AttributeValue::string("2", None)],
            &None,
            false,
        );
        let mut update = VideoFrameUpdate::default();
        update.add_frame_attribute(frame_attr.clone());
        update.add_object_attribute(obj.get_id(), frame_attr.clone());
        update.add_object(&obj, None);
        update.set_object_policy(ObjectUpdatePolicy::AddForeignObjects);
        update.set_object_attribute_policy(AttributeUpdatePolicy::KeepOwn);
        update.set_frame_attribute_policy(AttributeUpdatePolicy::Error);
        let generated_update = generated::VideoFrameUpdate::from(&update);
        let restored_update = VideoFrameUpdate::try_from(&generated_update).unwrap();
        assert_eq!(
            update.to_json(false).unwrap(),
            restored_update.to_json(false).unwrap()
        );
    }
}
