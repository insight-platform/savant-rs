pub mod utils {
    use crate::primitives::attribute::attribute_value::AttributeValue;
    use crate::primitives::attribute::AttributeMethods;
    use crate::primitives::message::video::frame::{
        VideoFrameBuilder, VideoFrameTranscodingMethod,
    };
    use crate::primitives::message::video::object::{VideoObject, VideoObjectBuilder};
    use crate::primitives::{
        AttributeBuilder, Intersection, IntersectionKind, Point, VideoFrameContentProxy,
        VideoObjectProxy,
    };
    use crate::primitives::{RBBox, VideoFrameProxy};
    use pyo3::pyfunction;
    use std::collections::HashMap;

    #[pyfunction]
    pub fn gen_frame() -> VideoFrameProxy {
        let f = VideoFrameProxy::from_inner(
            VideoFrameBuilder::default()
                .source_id("test".to_string())
                .pts(0)
                .framerate("test".to_string())
                .width(0)
                .height(0)
                .content(VideoFrameContentProxy::none().inner)
                .transcoding_method(VideoFrameTranscodingMethod::Copy)
                .codec(None)
                .keyframe(None)
                .build()
                .unwrap(),
        );

        let parent_object = VideoObjectProxy::from_video_object(
            VideoObjectBuilder::default()
                .id(0)
                .track_info(None)
                .modifications(Vec::default())
                .bbox(RBBox::new(0.0, 0.0, 0.0, 0.0, None))
                .attributes(HashMap::default())
                .confidence(None)
                .creator("test".to_string())
                .label("test2".to_string())
                .build()
                .unwrap(),
        );

        let c1 = VideoObjectProxy::from_video_object(
            VideoObjectBuilder::default()
                .id(1)
                .track_info(None)
                .modifications(Vec::default())
                .bbox(RBBox::new(0.0, 0.0, 0.0, 0.0, None))
                .parent_id(Some(parent_object.get_id()))
                .attributes(HashMap::default())
                .confidence(None)
                .creator("test2".to_string())
                .label("test".to_string())
                .build()
                .unwrap(),
        );

        let c2 = VideoObjectProxy::from_video_object(
            VideoObjectBuilder::default()
                .id(2)
                .track_info(None)
                .modifications(Vec::default())
                .bbox(RBBox::new(0.0, 0.0, 0.0, 0.0, None))
                .parent_id(Some(parent_object.get_id()))
                .attributes(HashMap::default())
                .confidence(None)
                .creator("test2".to_string())
                .label("test2".to_string())
                .build()
                .unwrap(),
        );

        f.add_object(&parent_object);
        f.add_object(&c1);
        f.add_object(&c2);

        f.set_attribute(
            AttributeBuilder::default()
                .creator("system".into())
                .name("test".into())
                .hint(None)
                .hint(Some("test".into()))
                .values(vec![AttributeValue::string("1".into(), None)])
                .build()
                .unwrap(),
        );

        f.set_attribute(
            AttributeBuilder::default()
                .creator("system2".into())
                .name("test2".into())
                .hint(None)
                .values(vec![AttributeValue::string("2".into(), None)])
                .build()
                .unwrap(),
        );

        f.set_attribute(
            AttributeBuilder::default()
                .creator("system".into())
                .name("test2".into())
                .hint(Some("test".into()))
                .values(vec![AttributeValue::string("3".into(), None)])
                .build()
                .unwrap(),
        );

        f.set_attribute(
            AttributeBuilder::default()
                .creator("test".to_string())
                .name("test".to_string())
                .hint(Some("hint".to_string()))
                .values(vec![
                    AttributeValue::bytes_from_list(vec![8, 3, 8, 8], [0; 192].into(), None),
                    AttributeValue::integers([0, 1, 2, 3, 4, 5].into(), None),
                    AttributeValue::string("incoming".to_string(), Some(0.56)),
                    AttributeValue::strings(vec!["abc".into(), "cde".into()], None),
                    AttributeValue::string("outgoing".to_string(), Some(0.64)),
                    AttributeValue::none(),
                    AttributeValue::bbox(RBBox::new(0.0, 0.0, 0.0, 0.0, None), None),
                    AttributeValue::bboxes(
                        vec![
                            RBBox::new(0.0, 0.0, 0.0, 0.0, None),
                            RBBox::new(0.0, 0.0, 0.0, 0.0, None),
                        ],
                        None,
                    ),
                    AttributeValue::float(0.0, None),
                    AttributeValue::floats(vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0], None),
                    AttributeValue::points(vec![Point::new(0.0, 0.0), Point::new(0.0, 0.0)], None),
                    AttributeValue::intersection(
                        Intersection::new(
                            &IntersectionKind::Enter,
                            vec![(0, Some("x1".to_string())), (1, Some("y1".to_string()))],
                        ),
                        None,
                    ),
                ])
                .build()
                .unwrap(),
        );

        f
    }

    pub fn gen_object(id: i64) -> VideoObjectProxy {
        VideoObjectProxy::from_video_object(VideoObject {
            id,
            creator: s("peoplenet"),
            label: s("face"),
            confidence: Some(0.5),
            bbox: RBBox::new(1.0, 2.0, 10.0, 20.0, None),
            ..Default::default()
        })
    }

    #[inline(always)]
    pub fn s(a: &str) -> String {
        a.to_string()
    }
}
