#![feature(test)]

extern crate test;

use rand::Rng;
use savant_core::primitives::utils::solely_owned_areas;
use savant_core::primitives::RBBox;
use test::Bencher;

fn bench_solely_owned_areas(bbox_count: usize, parallel: bool) {
    let pos_x_range = 0.0..1920.0;
    let pos_y_range = 0.0..1080.0;
    let width_range = 50.0..600.0;
    let height_range = 50.0..400.0;
    let mut rng = rand::rng();
    let bboxes: Vec<RBBox> = (0..bbox_count)
        .map(|_| {
            RBBox::new(
                rng.random_range(pos_x_range.clone()),
                rng.random_range(pos_y_range.clone()),
                rng.random_range(width_range.clone()),
                rng.random_range(height_range.clone()),
                Some(0.0),
            )
        })
        .collect();
    let bbox_refs = bboxes.iter().collect::<Vec<_>>();
    solely_owned_areas(&bbox_refs, parallel);
}

#[bench]
fn bench_seq_solely_owned_areas_010(b: &mut Bencher) {
    b.iter(|| {
        bench_solely_owned_areas(10, false);
    });
}

#[bench]
fn bench_seq_solely_owned_areas_020(b: &mut Bencher) {
    b.iter(|| {
        bench_solely_owned_areas(20, false);
    });
}

#[bench]
fn bench_seq_solely_owned_areas_050(b: &mut Bencher) {
    b.iter(|| {
        bench_solely_owned_areas(50, false);
    });
}

#[bench]
fn bench_par_solely_owned_areas_010(b: &mut Bencher) {
    b.iter(|| {
        bench_solely_owned_areas(10, true);
    });
}

#[bench]
fn bench_par_solely_owned_areas_020(b: &mut Bencher) {
    b.iter(|| {
        bench_solely_owned_areas(20, true);
    });
}

#[bench]
fn bench_par_solely_owned_areas_050(b: &mut Bencher) {
    b.iter(|| {
        bench_solely_owned_areas(50, true);
    });
}
