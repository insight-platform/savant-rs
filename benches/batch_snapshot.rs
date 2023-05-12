#![feature(test)]

extern crate test;

use savant_rs::primitives::VideoFrameBatch;
use savant_rs::test::utils::gen_frame;
use test::Bencher;

#[bench]
fn bench_batch(b: &mut Bencher) {
    let mut batch = VideoFrameBatch::new();
    batch.add(1, gen_frame());
    batch.add(2, gen_frame());
    batch.add(3, gen_frame());
    batch.add(4, gen_frame());
    b.iter(|| {
        batch.snapshot();
        batch.restore();
    });
}

#[bench]
fn bench_batch_py(b: &mut Bencher) {
    pyo3::prepare_freethreaded_python();
    let mut batch = VideoFrameBatch::new();
    batch.add(1, gen_frame());
    batch.add(2, gen_frame());
    batch.add(3, gen_frame());
    batch.add(4, gen_frame());
    b.iter(|| {
        batch.snapshot_py();
        batch.restore_py();
    });
}
