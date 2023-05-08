pub mod utils {
    use crate::primitives::{AttributeBuilder, PyVideoFrameContent, Value};
    use crate::primitives::{BBox, ObjectBuilder, VideoFrame, VideoFrameBuilder};
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    pub fn gen_frame() -> VideoFrame {
        let mut f = VideoFrameBuilder::default()
            .source_id("test".to_string())
            .pts(0)
            .framerate("test".to_string())
            .width(0)
            .height(0)
            .content(PyVideoFrameContent::none().data)
            .dts(None)
            .duration(None)
            .codec(None)
            .keyframe(None)
            .attributes(HashMap::default())
            .offline_objects(Default::default())
            .resident_objects(
                vec![
                    ObjectBuilder::default()
                        .id(0)
                        .bbox(BBox::new(0.0, 0.0, 0.0, 0.0, None))
                        .parent(None)
                        .attributes(HashMap::default())
                        .confidence(None)
                        .creator("test".to_string())
                        .label("test2".to_string())
                        .build()
                        .unwrap(),
                    ObjectBuilder::default()
                        .id(1)
                        .bbox(BBox::new(0.0, 0.0, 0.0, 0.0, None))
                        .parent(None)
                        .attributes(HashMap::default())
                        .confidence(None)
                        .creator("test2".to_string())
                        .label("test".to_string())
                        .build()
                        .unwrap(),
                    ObjectBuilder::default()
                        .id(2)
                        .bbox(BBox::new(0.0, 0.0, 0.0, 0.0, None))
                        .parent(None)
                        .attributes(HashMap::default())
                        .confidence(None)
                        .creator("test2".to_string())
                        .label("test2".to_string())
                        .build()
                        .unwrap(),
                ]
                .into_iter()
                .map(|o| Arc::new(Mutex::new(o)))
                .collect(),
            )
            .build()
            .unwrap();

        f.set_attribute(
            AttributeBuilder::default()
                .creator("system".into())
                .name("test".into())
                .hint(None)
                .hint(Some("test".into()))
                .value(Value::string("1".into()))
                .confidence(None)
                .build()
                .unwrap(),
        );

        f.set_attribute(
            AttributeBuilder::default()
                .creator("system2".into())
                .name("test2".into())
                .hint(None)
                .value(Value::string("2".into()))
                .confidence(None)
                .build()
                .unwrap(),
        );

        f.set_attribute(
            AttributeBuilder::default()
                .creator("system".into())
                .name("test2".into())
                .hint(Some("test".into()))
                .value(Value::string("3".into()))
                .confidence(None)
                .build()
                .unwrap(),
        );

        f
    }
}
