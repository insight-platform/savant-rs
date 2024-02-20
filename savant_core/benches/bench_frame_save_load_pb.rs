#![feature(test)]

extern crate test;
use savant_core::primitives::rust::VideoFrameProxy;
use savant_core::protobuf::{from_pb, ToProtobuf};
use savant_core::test::gen_frame;
use test::Bencher;

#[bench]
fn bench_save_load_video_frame_pb(b: &mut Bencher) {
    let frame = gen_frame();
    b.iter(|| {
        let res = frame.to_pb().unwrap();
        let _ = from_pb::<savant_core::protobuf::VideoFrame, VideoFrameProxy>(&res).unwrap();
    });
}
