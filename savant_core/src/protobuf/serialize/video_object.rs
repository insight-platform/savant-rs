use crate::primitives::object::{ObjectOperations, VideoObject};
use crate::primitives::{Attribute, RBBox, WithAttributes};
use crate::protobuf::serialize;
use savant_protobuf::generated;

impl From<&VideoObject> for generated::VideoObject {
    fn from(vop: &VideoObject) -> Self {
        let attributes = vop
            .get_attributes()
            .iter()
            .map(|(ns, l)| generated::Attribute::from(&vop.get_attribute(ns, l).unwrap()))
            .collect();

        generated::VideoObject {
            id: vop.get_id(),
            parent_id: vop.get_parent_id(),
            namespace: vop.get_namespace(),
            label: vop.get_label(),
            draw_label: vop.get_draw_label(),
            detection_box: Some(generated::BoundingBox::from(&vop.get_detection_box())),
            attributes,
            confidence: vop.get_confidence(),
            track_box: vop
                .get_track_box()
                .as_ref()
                .map(generated::BoundingBox::from),
            track_id: vop.get_track_id(),
        }
    }
}

pub(crate) struct GeneratedVideoObjectWithForeignParent(
    pub generated::VideoObjectWithForeignParent,
);

impl From<&(VideoObject, Option<i64>)> for GeneratedVideoObjectWithForeignParent {
    fn from(p: &(VideoObject, Option<i64>)) -> Self {
        GeneratedVideoObjectWithForeignParent(generated::VideoObjectWithForeignParent {
            object: Some(generated::VideoObject::from(&p.0)),
            parent_id: p.1,
        })
    }
}

impl TryFrom<&generated::VideoObject> for VideoObject {
    type Error = serialize::Error;
    fn try_from(obj: &generated::VideoObject) -> Result<Self, Self::Error> {
        let attributes = obj
            .attributes
            .iter()
            .filter(|a| a.is_persistent)
            .map(Attribute::try_from)
            .collect::<Result<Vec<Attribute>, _>>()?;

        Ok(VideoObject {
            id: obj.id,
            namespace: obj.namespace.clone(),
            label: obj.label.clone(),
            draw_label: obj.draw_label.clone(),
            detection_box: obj.detection_box.as_ref().map(RBBox::from).unwrap(),
            attributes,
            confidence: obj.confidence,
            parent_id: obj.parent_id,
            track_box: obj.track_box.as_ref().map(RBBox::from),
            track_id: obj.track_id,
            namespace_id: None,
            label_id: None,
            frame: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::json_api::ToSerdeJsonValue;
    use crate::primitives::object::VideoObject;
    use crate::primitives::rust::AttributeValue;
    use crate::primitives::{Attribute, WithAttributes};
    use crate::test::gen_object;
    use savant_protobuf::generated;

    #[test]
    fn test_object() {
        let obj = gen_object(1);
        let serialized = generated::VideoObject::from(&obj);
        let deserialized = VideoObject::try_from(&serialized).unwrap();
        assert_eq!(
            obj.to_serde_json_value(),
            deserialized.to_serde_json_value()
        );
    }

    #[test]
    fn test_object_with_tmp_attribute() {
        let mut obj = gen_object(1);
        let tmp_attr = Attribute::temporary(
            "tmp",
            "label",
            vec![AttributeValue::float(1.0, None)],
            &None,
            false,
        );
        let persistent_attr = Attribute::persistent(
            "pers",
            "label",
            vec![AttributeValue::integer(1, None)],
            &None,
            false,
        );
        obj.set_attribute(tmp_attr.clone());
        obj.set_attribute(persistent_attr.clone());
        let serialized = generated::VideoObject::from(&obj);
        let deserialized = VideoObject::try_from(&serialized).unwrap();
        assert!(deserialized.get_attribute("tmp", "label").is_none());
        assert_eq!(
            deserialized.get_attribute("pers", "label").unwrap(),
            persistent_attr
        );
    }
}
