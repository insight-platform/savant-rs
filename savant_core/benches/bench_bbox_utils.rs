use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use savant_core::primitives::utils::solely_owned_areas;
use savant_core::primitives::RBBox;

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

fn bench_solely_owned_areas_criterion(c: &mut Criterion) {
    let mut group = c.benchmark_group("solely_owned_areas");

    for &bbox_count in &[10, 20, 50] {
        group.bench_function(&format!("seq_{:03}", bbox_count), |b| {
            b.iter(|| bench_solely_owned_areas(black_box(bbox_count), black_box(false)))
        });

        group.bench_function(&format!("par_{:03}", bbox_count), |b| {
            b.iter(|| bench_solely_owned_areas(black_box(bbox_count), black_box(true)))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_solely_owned_areas_criterion);
criterion_main!(benches);
