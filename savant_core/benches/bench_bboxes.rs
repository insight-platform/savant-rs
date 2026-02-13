use criterion::{criterion_group, criterion_main, Criterion};
use savant_core::{draw::PaddingDraw, primitives::RBBox};
use std::hint::black_box;

fn bbox_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("bbox_operations");

    // Scale benchmarks
    group.bench_function("scale_90", |b| {
        let bb1 = RBBox::new(0.0, 0.0, 10.0, 20.0, Some(0.0));
        b.iter(|| {
            bb1.clone().scale(black_box(0.9), black_box(0.7));
            black_box(());
        })
    });

    group.bench_function("scale_generic", |b| {
        let bb1 = RBBox::new(0.0, 0.0, 10.0, 20.0, Some(35.0));
        b.iter(|| {
            bb1.clone().scale(black_box(0.9), black_box(0.7));
            black_box(());
        })
    });

    // Area benchmark
    group.bench_function("get_area", |b| {
        let bb1 = RBBox::new(0.0, 0.0, 10.0, 20.0, Some(35.0));
        b.iter(|| {
            black_box(bb1.get_area());
        })
    });

    // Intersection benchmarks
    group.bench_function("iou", |b| {
        let bb1 = RBBox::new(0.0, 0.0, 10.0, 20.0, Some(0.0));
        let bb2 = RBBox::new(0.0, 0.0, 20.0, 10.0, Some(0.0));
        b.iter(|| {
            black_box(bb1.iou(&bb2).expect("iou failed"));
        })
    });

    group.bench_function("ios", |b| {
        let bb1 = RBBox::new(0.0, 0.0, 10.0, 20.0, Some(0.0));
        let bb2 = RBBox::new(0.0, 0.0, 20.0, 10.0, Some(0.0));
        b.iter(|| {
            black_box(bb1.ios(&bb2).expect("ios failed"));
        })
    });

    group.bench_function("ioo", |b| {
        let bb1 = RBBox::new(0.0, 0.0, 10.0, 20.0, Some(0.0));
        let bb2 = RBBox::new(0.0, 0.0, 20.0, 10.0, Some(0.0));
        b.iter(|| {
            black_box(bb1.ioo(&bb2).expect("ioo failed"));
        })
    });

    group.bench_function("visual_box_with_angle", |b| {
        let bb1 = RBBox::new(0.0, 0.0, 10.0, 20.0, Some(10.0));
        b.iter(|| {
            black_box(
                bb1.get_visual_box(
                    black_box(&PaddingDraw::default_padding()),
                    black_box(0),
                    black_box(1920.0),
                    black_box(1080.0),
                )
                .unwrap(),
            );
        })
    });

    group.bench_function("visual_box_without_angle", |b| {
        let bb1 = RBBox::new(0.0, 0.0, 10.0, 20.0, None);
        b.iter(|| {
            black_box(
                bb1.get_visual_box(
                    black_box(&PaddingDraw::default_padding()),
                    black_box(0),
                    black_box(1920.0),
                    black_box(1080.0),
                )
                .unwrap(),
            );
        })
    });
    group.finish();
}

criterion_group!(benches, bbox_benchmarks);
criterion_main!(benches);
