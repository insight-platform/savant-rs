#![feature(test)]

extern crate test;

use savant_core::primitives::frame_batch::VideoFrameBatch;
use savant_core::test::gen_frame;
use test::Bencher;

#[bench]
fn bench_batch_snapshot(b: &mut Bencher) {
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
