use crate::primitives::object::{VideoObject, VideoObjectProxy};
use crate::primitives::{Attribute, AttributeMethods};
use crate::protobuf::{generated, serialize};
use hashbrown::HashMap;

impl From<&VideoObjectProxy> for generated::VideoObject {
    fn from(vop: &VideoObjectProxy) -> Self {
        generated::VideoObject {
            id: vop.get_id(),
            parent_id: vop.get_parent_id(),
            namespace: vop.get_namespace(),
            label: vop.get_label(),
            draw_label: vop.get_draw_label(),
            detection_box: Some(vop.get_detection_box().into()),
            attributes: vop
                .get_attributes()
                .iter()
                .map(|(ns, l)| {
                    generated::Attribute::from(&vop.get_attribute(ns.clone(), l.clone()).unwrap())
                })
                .collect(),
            confidence: vop.get_confidence(),
            track_box: vop.get_track_box().map(|rbbox| rbbox.into()),
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

impl TryFrom<&generated::VideoObject> for VideoObject {
    type Error = serialize::Error;
    fn try_from(obj: &generated::VideoObject) -> Result<Self, Self::Error> {
        let attributes = obj
            .attributes
            .iter()
            .map(|a| Attribute::try_from(a).map(|a| ((a.namespace.clone(), a.name.clone()), a)))
            .collect::<Result<HashMap<(String, String), Attribute>, _>>()?;

        Ok(VideoObject {
            id: obj.id,
            namespace: obj.namespace.clone(),
            label: obj.label.clone(),
            draw_label: obj.draw_label.clone(),
            detection_box: obj.detection_box.as_ref().unwrap().into(),
            attributes,
            confidence: obj.confidence,
            parent_id: obj.parent_id,
            track_box: obj.track_box.as_ref().map(|rbbox| rbbox.into()),
            track_id: obj.track_id,
            namespace_id: None,
            label_id: None,
            frame: None,
        })
    }
}
