#![feature(test)]

extern crate test;

use savant_core::primitives::object::{VideoObjectBBoxType, VideoObjectBuilder};
use savant_core::primitives::rust::{VideoFrameProxy, VideoObjectProxy};
use savant_core::primitives::RBBox;
use test::Bencher;

fn get_objects() -> Vec<VideoObjectProxy> {
    (0..100)
        .into_iter()
        .map(|i| {
            let o = VideoObjectBuilder::default()
                .namespace(format!("created_by_{i}"))
                .label(format!("label_{i}"))
                .id(i)
                .confidence(Some(0.53))
                .detection_box(RBBox::new(10.0, 10.0, 1.0, 1.0, None).try_into().unwrap())
                .track_id(Some(i))
                .track_box(None)
                .build()
                .unwrap();
            VideoObjectProxy::from(o)
        })
        .collect::<Vec<_>>()
}

#[bench]
fn bench_validation(b: &mut Bencher) {
    let objs = get_objects();
    b.iter(|| {
        VideoFrameProxy::check_frame_fit(&objs, 100.0, 100.0, VideoObjectBBoxType::Detection)
            .unwrap();
    });
}
