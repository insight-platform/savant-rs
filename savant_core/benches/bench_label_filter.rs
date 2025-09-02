use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn label_filter_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("label_filter");

    group.bench_function("label_filter", |b| {
        use savant_core::message::label_filter::LabelFilterRule::*;
        let rule = Or(vec![
            Set("test".to_string()),
            Not(Box::new(Or(vec![
                Set("test2".to_string()),
                Set("test3".to_string()),
            ]))),
        ]);

        b.iter(|| {
            black_box(rule.matches(&["test".to_string(), "test2".to_string()]));
        })
    });

    group.finish();
}

criterion_group!(benches, label_filter_benchmarks);
criterion_main!(benches);
