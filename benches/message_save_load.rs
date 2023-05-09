#![feature(test)]

extern crate test;

use savant_rs::primitives::message::loader::load_message;
use savant_rs::primitives::message::saver::save_message;
use savant_rs::primitives::Message;
use savant_rs::test::utils::gen_frame;
use test::Bencher;

#[bench]
fn bench_video_frame_sync(b: &mut Bencher) {
    pyo3::prepare_freethreaded_python();
    let frame = Message::video_frame(gen_frame());
    b.iter(|| {
        let res = save_message(frame.clone());
        let _ = load_message(res);
    });
}

#[bench]
fn bench_eos_sync(b: &mut Bencher) {
    pyo3::prepare_freethreaded_python();
    let eos = savant_rs::primitives::EndOfStream::new("test".to_string());
    let frame = Message::end_of_stream(eos);
    b.iter(|| {
        let res = save_message(frame.clone());
        let _ = load_message(res);
    });
}
