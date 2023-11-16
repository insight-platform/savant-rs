#![feature(test)]

extern crate test;

use savant_core::match_query::MatchQuery;
use savant_core::message::{load_message, save_message, Message};
use savant_core::primitives::eos::EndOfStream;
use savant_core::primitives::frame_batch::VideoFrameBatch;
use savant_core::primitives::frame_update::VideoFrameUpdate;
use savant_core::primitives::AttributeMethods;
use savant_core::test::gen_frame;
use test::Bencher;

#[bench]
fn bench_save_load_video_frame(b: &mut Bencher) {
    let message = Message::video_frame(&gen_frame());
    b.iter(|| {
        let res = save_message(&message);
        let m = load_message(&res);
        assert!(m.is_video_frame());
    });
}

#[bench]
fn bench_save_load_eos(b: &mut Bencher) {
    let eos = EndOfStream::new("test".to_string());
    let message = Message::end_of_stream(eos);
    b.iter(|| {
        let res = save_message(&message);
        let m = load_message(&res);
        assert!(m.is_end_of_stream());
    });
}

#[bench]
fn bench_save_load_batch(b: &mut Bencher) {
    let mut batch = VideoFrameBatch::new();
    batch.add(1, gen_frame());
    batch.add(2, gen_frame());
    batch.add(3, gen_frame());
    batch.add(4, gen_frame());
    let message = Message::video_frame_batch(&batch);
    b.iter(|| {
        let res = save_message(&message);
        let m = load_message(&res);
        assert!(m.is_video_frame_batch());
    });
}

#[bench]
fn bench_save_load_frame_update(b: &mut Bencher) {
    let f = gen_frame();
    let mut update = VideoFrameUpdate::default();
    for o in f.access_objects(&MatchQuery::Idle) {
        update.add_object(&o, None);
    }
    let attrs = f.get_attributes();
    for (namespace, label) in attrs {
        update.add_frame_attribute(f.get_attribute(namespace, label).unwrap());
    }

    let message = Message::video_frame_update(update);

    b.iter(|| {
        let res = save_message(&message);
        let m = load_message(&res);
        assert!(m.is_video_frame_update());
    });
}
