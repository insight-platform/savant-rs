#![feature(test)]

extern crate test;

use savant_rs::primitives::RBBox;
use test::Bencher;

#[bench]
fn bench_scale_90(b: &mut Bencher) {
    let bb1 = RBBox::new(0.0, 0.0, 10.0, 20.0, Some(0.0));
    b.iter(|| {
        bb1.clone().scale(0.9, 0.7);
    });
}

#[bench]
fn bench_scale_generic(b: &mut Bencher) {
    let bb1 = RBBox::new(0.0, 0.0, 10.0, 20.0, Some(35.0));
    b.iter(|| {
        bb1.clone().scale(0.9, 0.7);
    });
}

#[bench]
fn bench_get_area(b: &mut Bencher) {
    let bb1 = RBBox::new(0.0, 0.0, 10.0, 20.0, Some(35.0));
    b.iter(|| {
        bb1.get_area();
    });
}

#[bench]
fn bench_iou(b: &mut Bencher) {
    let bb1 = RBBox::new(0.0, 0.0, 10.0, 20.0, Some(0.0));
    let bb2 = RBBox::new(0.0, 0.0, 20.0, 10.0, Some(0.0));
    b.iter(|| {
        bb1.iou(&bb2).expect("iou failed");
    });
}

#[bench]
fn bench_ios(b: &mut Bencher) {
    let bb1 = RBBox::new(0.0, 0.0, 10.0, 20.0, Some(0.0));
    let bb2 = RBBox::new(0.0, 0.0, 20.0, 10.0, Some(0.0));
    b.iter(|| {
        bb1.ios(&bb2).expect("ios failed");
    });
}

#[bench]
fn bench_ioo(b: &mut Bencher) {
    let bb1 = RBBox::new(0.0, 0.0, 10.0, 20.0, Some(0.0));
    let bb2 = RBBox::new(0.0, 0.0, 20.0, 10.0, Some(0.0));
    b.iter(|| {
        bb1.ioo(&bb2).expect("ioo failed");
    });
}
