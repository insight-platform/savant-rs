#![feature(test)]

extern crate test;

use savant_rs::primitives::db::loader::Loader;
use savant_rs::primitives::{Frame, Saver};
use savant_rs::test::utils::gen_frame;
use test::Bencher;

#[bench]
fn bench_save_video_frame(b: &mut Bencher) {
    pyo3::prepare_freethreaded_python();
    let frame = Frame::video_frame(gen_frame());
    let saver = Saver::new(1);
    let loader = Loader::new(1);
    b.iter(|| {
        let res = saver.save(frame.clone());
        let res = res.recv().unwrap();
        let res = loader.load(res);
        let _ = res.recv().unwrap();
    });
}

#[bench]
fn bench_save_eos(b: &mut Bencher) {
    pyo3::prepare_freethreaded_python();
    let eos = savant_rs::primitives::EndOfStream::new("test".to_string());
    let frame = Frame::end_of_stream(eos);
    let saver = Saver::new(1);
    let loader = Loader::new(1);
    b.iter(|| {
        let res = saver.save(frame.clone());
        let res = res.recv().unwrap();
        let res = loader.load(res);
        let _ = res.recv().unwrap();
    });
}
