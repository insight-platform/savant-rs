use crate::primitives::attribute_value::{AttributeValue, AttributeValueVariant};
use crate::primitives::frame::{
    VideoFrameBuilder, VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
};
use crate::primitives::object::{
    IdCollisionResolutionPolicy, VideoObject, VideoObjectBuilder, VideoObjectProxy,
};
use crate::primitives::{Attribute, AttributeMethods, RBBox};
use hashbrown::HashMap;
type Variant = AttributeValueVariant;

pub fn gen_empty_frame() -> VideoFrameProxy {
    VideoFrameProxy::from_inner(
        VideoFrameBuilder::default()
            .source_id("test".to_string())
            .pts(0)
            .framerate("test".to_string())
            .width(0)
            .uuid(uuid::Uuid::new_v4().as_u128())
            .height(0)
            .content(VideoFrameContent::None)
            .transcoding_method(VideoFrameTranscodingMethod::Copy)
            .codec(None)
            .keyframe(None)
            .build()
            .unwrap(),
    )
}

pub fn gen_frame() -> VideoFrameProxy {
    let f = VideoFrameProxy::from_inner(
        VideoFrameBuilder::default()
            .source_id("test".to_string())
            .pts(1000000)
            .framerate("test".to_string())
            .width(1280)
            .uuid(uuid::Uuid::new_v4().as_u128())
            .height(720)
            .content(VideoFrameContent::None)
            .transcoding_method(VideoFrameTranscodingMethod::Copy)
            .codec(None)
            .keyframe(None)
            .build()
            .unwrap(),
    );

    let parent_object = VideoObjectProxy::from(
        VideoObjectBuilder::default()
            .id(0)
            .detection_box(RBBox::new(0.0, 0.0, 0.0, 0.0, None).try_into().unwrap())
            .attributes(HashMap::default())
            .confidence(None)
            .namespace("test".to_string())
            .label("test2".to_string())
            .build()
            .unwrap(),
    );

    let c1 = VideoObjectProxy::from(
        VideoObjectBuilder::default()
            .id(1)
            .detection_box(RBBox::new(0.0, 0.0, 0.0, 0.0, None).try_into().unwrap())
            .parent_id(Some(parent_object.get_id()))
            .attributes(HashMap::default())
            .confidence(None)
            .namespace("test2".to_string())
            .label("test".to_string())
            .build()
            .unwrap(),
    );

    let c2 = VideoObjectProxy::from(
        VideoObjectBuilder::default()
            .id(2)
            .detection_box(RBBox::new(0.0, 0.0, 0.0, 0.0, None).try_into().unwrap())
            .parent_id(Some(parent_object.get_id()))
            .attributes(HashMap::default())
            .confidence(None)
            .namespace("test2".to_string())
            .label("test2".to_string())
            .build()
            .unwrap(),
    );

    f.add_object(&parent_object, IdCollisionResolutionPolicy::Error)
        .unwrap();
    f.add_object(&c1, IdCollisionResolutionPolicy::Error)
        .unwrap();
    f.add_object(&c2, IdCollisionResolutionPolicy::Error)
        .unwrap();

    f.set_attribute(Attribute::persistent(
        "system".into(),
        "test".into(),
        vec![AttributeValue::new(
            AttributeValueVariant::String("1".into()),
            None,
        )],
        Some("test".into()),
    ));

    f.set_attribute(Attribute::persistent(
        "system2".into(),
        "test2".into(),
        vec![AttributeValue::new(
            AttributeValueVariant::String("2".into()),
            None,
        )],
        None,
    ));

    f.set_attribute(Attribute::persistent(
        "system".into(),
        "test2".into(),
        vec![AttributeValue::new(
            AttributeValueVariant::String("3".into()),
            None,
        )],
        Some("test".into()),
    ));

    f.set_attribute(Attribute::persistent(
        "test".to_string(),
        "test".to_string(),
        vec![
            AttributeValue::new(Variant::Bytes(vec![8, 3, 8, 8], [0; 192].into()), None),
            AttributeValue::new(Variant::IntegerVector([0, 1, 2, 3, 4, 5].into()), None),
            AttributeValue::new(Variant::String("incoming".to_string()), Some(0.56)),
        ],
        Some("hint".to_string()),
    ));
    f
}

pub fn gen_object(id: i64) -> VideoObjectProxy {
    let o = VideoObjectProxy::from(VideoObject {
        id,
        namespace: s("peoplenet"),
        label: s("face"),
        confidence: Some(0.5),
        detection_box: RBBox::new(1.0, 2.0, 10.0, 20.0, None).try_into().unwrap(),
        track_id: Some(id),
        track_box: RBBox::new(100.0, 200.0, 10.0, 20.0, None).try_into().ok(),
        ..Default::default()
    });

    let attr = Attribute::persistent(
        "some".to_string(),
        "attribute".to_string(),
        vec![],
        Some("hint".to_string()),
    );
    o.set_attribute(attr);
    o
}

#[inline(always)]
pub fn s(a: &str) -> String {
    a.to_string()
}
