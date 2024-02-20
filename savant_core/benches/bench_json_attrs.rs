#![feature(test)]

extern crate test;
use savant_core::primitives::rust::AttributeValue;
use savant_core::primitives::Attribute;
use test::Bencher;

#[bench]
fn bench_to_json_empty(b: &mut Bencher) {
    let attr = Attribute::persistent("test", "test", vec![], &None, false);
    b.iter(|| {
        _ = attr.to_json().unwrap();
    });
}

#[bench]
fn bench_from_json_empty(b: &mut Bencher) {
    let attr = Attribute::persistent("test", "test", vec![], &None, false);
    let s = attr.to_json().unwrap();
    b.iter(|| {
        _ = Attribute::from_json(&s).unwrap();
    });
}

#[bench]
fn bench_to_json_int(b: &mut Bencher) {
    let attr = Attribute::persistent(
        "test",
        "test",
        vec![AttributeValue::integer(1, Some(0.96))],
        &None,
        false,
    );
    b.iter(|| {
        _ = attr.to_json().unwrap();
    });
}

#[bench]
fn bench_from_json_int(b: &mut Bencher) {
    let attr = Attribute::persistent(
        "test",
        "test",
        vec![AttributeValue::integer(1, Some(0.96))],
        &None,
        false,
    );
    let s = attr.to_json().unwrap();
    b.iter(|| {
        _ = Attribute::from_json(&s).unwrap();
    });
}

#[bench]
fn bench_to_json_int_vec(b: &mut Bencher) {
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
        _ = attr.to_json().unwrap();
    });
}

#[bench]
fn bench_from_json_int_vec(b: &mut Bencher) {
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
        _ = Attribute::from_json(&s).unwrap();
    });
}

#[bench]
fn bench_to_json_int_many(b: &mut Bencher) {
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
        _ = attr.to_json().unwrap();
    });
}

#[bench]
fn bench_from_json_int_many(b: &mut Bencher) {
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
        _ = Attribute::from_json(&s).unwrap();
    });
}

#[bench]
fn bench_to_json_float(b: &mut Bencher) {
    let attr = Attribute::persistent(
        "test",
        "test",
        vec![AttributeValue::float(1.0, Some(0.96))],
        &None,
        false,
    );
    b.iter(|| {
        _ = attr.to_json().unwrap();
    });
}

#[bench]
fn bench_from_json_float(b: &mut Bencher) {
    let attr = Attribute::persistent(
        "test",
        "test",
        vec![AttributeValue::float(1.0, Some(0.96))],
        &None,
        false,
    );
    let s = attr.to_json().unwrap();
    b.iter(|| {
        _ = Attribute::from_json(&s).unwrap();
    });
}

#[bench]
fn bench_to_json_float_many(b: &mut Bencher) {
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
        _ = attr.to_json().unwrap();
    });
}

#[bench]
fn bench_from_json_float_many(b: &mut Bencher) {
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
        _ = Attribute::from_json(&s).unwrap();
    });
}

#[bench]
fn bench_to_json_string(b: &mut Bencher) {
    let attr = Attribute::persistent(
        "test",
        "test",
        vec![AttributeValue::string("test", Some(0.96))],
        &None,
        false,
    );
    b.iter(|| {
        _ = attr.to_json().unwrap();
    });
}

#[bench]
fn bench_from_json_string(b: &mut Bencher) {
    let attr = Attribute::persistent(
        "test",
        "test",
        vec![AttributeValue::string("test", Some(0.96))],
        &None,
        false,
    );
    let s = attr.to_json().unwrap();
    b.iter(|| {
        _ = Attribute::from_json(&s).unwrap();
    });
}
