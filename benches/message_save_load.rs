#![feature(test)]

extern crate test;

use savant_rs::primitives::{load_message, save_message, Message, VideoFrameBatch};
use savant_rs::test::utils::gen_frame;
use test::Bencher;

#[bench]
fn bench_video_frame_sync(b: &mut Bencher) {
    pyo3::prepare_freethreaded_python();
    let frame = Message::video_frame(gen_frame());
    b.iter(|| {
        let res = save_message(frame.clone());
        let m = load_message(res);
        assert!(m.is_video_frame());
    });
}

#[bench]
fn bench_eos_sync(b: &mut Bencher) {
    let eos = savant_rs::primitives::EndOfStream::new("test".to_string());
    let frame = Message::end_of_stream(eos);
    b.iter(|| {
        let res = save_message(frame.clone());
        let m = load_message(res);
        assert!(m.is_end_of_stream());
    });
}

#[bench]
fn bench_batch_sync(b: &mut Bencher) {
    let mut batch = VideoFrameBatch::new();
    batch.add(1, gen_frame());
    batch.add(2, gen_frame());
    batch.add(3, gen_frame());
    batch.add(4, gen_frame());
    let m = Message::video_frame_batch(batch);
    b.iter(|| {
        let res = save_message(m.clone());
        let m = load_message(res);
        assert!(m.is_video_frame_batch());
    });
}
