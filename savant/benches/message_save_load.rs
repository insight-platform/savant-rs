#![feature(test)]

extern crate test;

use savant_rs::primitives::attribute::AttributeMethods;
use savant_rs::primitives::message::video::query::Query;
use savant_rs::primitives::{load_message, save_message, Message, VideoFrameBatch};
use savant_rs::test::utils::gen_frame;
use test::Bencher;

#[bench]
fn bench_save_load_video_frame(b: &mut Bencher) {
    pyo3::prepare_freethreaded_python();
    let message = Message::video_frame(gen_frame());
    b.iter(|| {
        let res = save_message(message.clone());
        let m = load_message(res);
        assert!(m.is_video_frame());
    });
}

#[bench]
fn bench_save_load_eos(b: &mut Bencher) {
    let eos = savant_rs::primitives::EndOfStream::new("test".to_string());
    let message = Message::end_of_stream(eos);
    b.iter(|| {
        let res = save_message(message.clone());
        let m = load_message(res);
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
    let message = Message::video_frame_batch(batch);
    b.iter(|| {
        let res = save_message(message.clone());
        let m = load_message(res);
        assert!(m.is_video_frame_batch());
    });
}

#[bench]
fn bench_save_load_frame_update(b: &mut Bencher) {
    pyo3::prepare_freethreaded_python();
    let f = gen_frame();
    let mut update = savant_rs::primitives::VideoFrameUpdate::new();
    for o in f.access_objects(&Query::Idle) {
        update.add_object(o);
    }
    let attrs = f.get_attributes();
    for (creator, label) in attrs {
        update.add_attribute(f.get_attribute(creator, label).unwrap());
    }

    let message = Message::video_fram_update(update);

    b.iter(|| {
        let res = save_message(message.clone());
        let m = load_message(res);
        assert!(m.is_video_frame_update());
    });
}
