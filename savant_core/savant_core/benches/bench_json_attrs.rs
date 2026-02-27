use criterion::{criterion_group, criterion_main, Criterion};
use savant_core::primitives::rust::AttributeValue;
use savant_core::primitives::Attribute;
use std::hint::black_box;

fn json_empty_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("json_empty");

    group.bench_function("to_json", |b| {
        let attr = Attribute::persistent("test", "test", vec![], &None, false);
        b.iter(|| {
            black_box(attr.to_json().unwrap());
        })
    });

    group.bench_function("from_json", |b| {
        let attr = Attribute::persistent("test", "test", vec![], &None, false);
        let s = attr.to_json().unwrap();
        b.iter(|| {
            black_box(Attribute::from_json(&s).unwrap());
        })
    });

    group.finish();
}

fn json_int_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("json_int");

    group.bench_function("to_json", |b| {
        let attr = Attribute::persistent(
            "test",
            "test",
            vec![AttributeValue::integer(1, Some(0.96))],
            &None,
            false,
        );
        b.iter(|| {
            black_box(attr.to_json().unwrap());
        })
    });

    group.bench_function("from_json", |b| {
        let attr = Attribute::persistent(
            "test",
            "test",
            vec![AttributeValue::integer(1, Some(0.96))],
            &None,
            false,
        );
        let s = attr.to_json().unwrap();
        b.iter(|| {
            black_box(Attribute::from_json(&s).unwrap());
        })
    });

    group.finish();
}

fn json_int_vec_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("json_int_vec");

    group.bench_function("to_json", |b| {
        let attr = Attribute::persistent(
            "test",
            "test",
            vec![AttributeValue::integer_vector(
                vec![1, 2, 3, 4, 5],
                Some(0.96),
            )],
            &None,
            false,
        );
        b.iter(|| {
            black_box(attr.to_json().unwrap());
        })
    });

    group.bench_function("from_json", |b| {
        let attr = Attribute::persistent(
            "test",
            "test",
            vec![AttributeValue::integer_vector(
                vec![1, 2, 3, 4, 5],
                Some(0.96),
            )],
            &None,
            false,
        );
        let s = attr.to_json().unwrap();
        b.iter(|| {
            black_box(Attribute::from_json(&s).unwrap());
        })
    });

    group.finish();
}

fn json_int_many_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("json_int_many");

    group.bench_function("to_json", |b| {
        let attr = Attribute::persistent(
            "test",
            "test",
            (0..10)
                .map(|i| AttributeValue::integer(i, Some(0.96)))
                .collect(),
            &None,
            false,
        );
        b.iter(|| {
            black_box(attr.to_json().unwrap());
        })
    });

    group.bench_function("from_json", |b| {
        let attr = Attribute::persistent(
            "test",
            "test",
            (0..10)
                .map(|i| AttributeValue::integer(i, Some(0.96)))
                .collect(),
            &None,
            false,
        );
        let s = attr.to_json().unwrap();
        b.iter(|| {
            black_box(Attribute::from_json(&s).unwrap());
        })
    });

    group.finish();
}

fn json_float_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("json_float");

    group.bench_function("to_json", |b| {
        let attr = Attribute::persistent(
            "test",
            "test",
            vec![AttributeValue::float(1.0, Some(0.96))],
            &None,
            false,
        );
        b.iter(|| {
            black_box(attr.to_json().unwrap());
        })
    });

    group.bench_function("from_json", |b| {
        let attr = Attribute::persistent(
            "test",
            "test",
            vec![AttributeValue::float(1.0, Some(0.96))],
            &None,
            false,
        );
        let s = attr.to_json().unwrap();
        b.iter(|| {
            black_box(Attribute::from_json(&s).unwrap());
        })
    });

    group.finish();
}

fn json_float_many_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("json_float_many");

    group.bench_function("to_json", |b| {
        let attr = Attribute::persistent(
            "test",
            "test",
            (0..10)
                .map(|i| AttributeValue::float(i as f64, Some(0.96)))
                .collect(),
            &None,
            false,
        );
        b.iter(|| {
            black_box(attr.to_json().unwrap());
        })
    });

    group.bench_function("from_json", |b| {
        let attr = Attribute::persistent(
            "test",
            "test",
            (0..10)
                .map(|i| AttributeValue::float(i as f64, Some(0.96)))
                .collect(),
            &None,
            false,
        );
        let s = attr.to_json().unwrap();
        b.iter(|| {
            black_box(Attribute::from_json(&s).unwrap());
        })
    });

    group.finish();
}

fn json_string_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("json_string");

    group.bench_function("to_json", |b| {
        let attr = Attribute::persistent(
            "test",
            "test",
            vec![AttributeValue::string("test", Some(0.96))],
            &None,
            false,
        );
        b.iter(|| {
            black_box(attr.to_json().unwrap());
        })
    });

    group.bench_function("from_json", |b| {
        let attr = Attribute::persistent(
            "test",
            "test",
            vec![AttributeValue::string("test", Some(0.96))],
            &None,
            false,
        );
        let s = attr.to_json().unwrap();
        b.iter(|| {
            black_box(Attribute::from_json(&s).unwrap());
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    json_empty_benchmarks,
    json_int_benchmarks,
    json_int_vec_benchmarks,
    json_int_many_benchmarks,
    json_float_benchmarks,
    json_float_many_benchmarks,
    json_string_benchmarks
);
criterion_main!(benches);
