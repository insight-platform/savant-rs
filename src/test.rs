pub mod utils {
    use crate::primitives::video::frame::PyVideoFrameContent;
    use crate::primitives::{BBox, ObjectBuilder, VideoFrame, VideoFrameBuilder};
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    pub fn gen_frame() -> VideoFrame {
        VideoFrameBuilder::default()
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
            .attributes(Default::default())
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
            .unwrap()
    }
}
