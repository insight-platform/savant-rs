use crate::primitives::object::{VideoObject, VideoObjectProxy};
use crate::primitives::{Attribute, AttributeMethods, RBBox};
use crate::protobuf::{generated, serialize};

impl From<&VideoObjectProxy> for generated::VideoObject {
    fn from(vop: &VideoObjectProxy) -> Self {
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

impl From<&(VideoObjectProxy, Option<i64>)> for generated::VideoObjectWithForeignParent {
    fn from(p: &(VideoObjectProxy, Option<i64>)) -> Self {
        generated::VideoObjectWithForeignParent {
            object: Some(generated::VideoObject::from(&p.0)),
            parent_id: p.1,
        }
    }
}

impl TryFrom<&generated::VideoObject> for VideoObjectProxy {
    type Error = serialize::Error;

    fn try_from(value: &crate::protobuf::VideoObject) -> Result<Self, Self::Error> {
        let vo = VideoObject::try_from(value)?;
        Ok(VideoObjectProxy::from(vo))
    }
}

impl TryFrom<&generated::VideoObject> for VideoObject {
    type Error = serialize::Error;
    fn try_from(obj: &generated::VideoObject) -> Result<Self, Self::Error> {
        let attributes = obj
            .attributes
            .iter()
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
    use crate::primitives::rust::VideoObjectProxy;
    use crate::protobuf::generated;
    use crate::test::gen_object;

    #[test]
    fn test_object() {
        let obj = gen_object(1);
        let serialized = generated::VideoObject::from(&obj);
        let deserialized = VideoObjectProxy::from(VideoObject::try_from(&serialized).unwrap());
        assert_eq!(
            obj.to_serde_json_value(),
            deserialized.to_serde_json_value()
        );
    }
}
