#![feature(test)]

extern crate test;

use savant_rs::primitives::message::video::object::query::{
    ExecutableQuery, FloatQ, OptFloatQ, PropertiesQ, StringQ, Q,
};
use savant_rs::primitives::message::video::object::{InnerObject, InnerObjectBuilder};
use savant_rs::primitives::RBBox;
use test::Bencher;

fn get_objects() -> Vec<InnerObject> {
    (0..100)
        .into_iter()
        .map(|i| {
            InnerObjectBuilder::default()
                .creator(format!("created_by_{i}"))
                .label(format!("label_{i}"))
                .id(i)
                .confidence(Some(0.53))
                .bbox(RBBox::new(0.0, 0.0, 1.0, 1.0, None))
                .track_id(Some(i))
                .build()
                .unwrap()
        })
        .collect::<Vec<_>>()
}

#[bench]
fn bench_filtering(b: &mut Bencher) {
    let expr = Q::And(vec![
        Q::Object(PropertiesQ::Or(vec![
            PropertiesQ::Creator(StringQ::Or(vec![
                StringQ::EndsWith("2".to_string()),
                StringQ::EndsWith("4".to_string()),
            ])),
            PropertiesQ::Label(StringQ::Or(vec![
                StringQ::EndsWith("2".to_string()),
                StringQ::EndsWith("4".to_string()),
                StringQ::EndsWith("6".to_string()),
            ])),
        ])),
        Q::Object(PropertiesQ::And(vec![
            PropertiesQ::Confidence(OptFloatQ::DefinedAnd(FloatQ::GE(0.5))),
            PropertiesQ::Confidence(OptFloatQ::DefinedAnd(FloatQ::LE(0.8))),
        ])),
    ]);

    let objs = get_objects();
    b.iter(|| {
        let _ = objs.iter().map(|o| expr.execute(o)).collect::<Vec<_>>();
    });
}

#[bench]
fn bench_empty_filtering(b: &mut Bencher) {
    let expr = Q::Pass;
    let objs = get_objects();
    b.iter(|| {
        let _ = objs.iter().map(|o| expr.execute(o)).collect::<Vec<_>>();
    });
}

#[bench]
fn bench_simple_filtering(b: &mut Bencher) {
    let expr = Q::Object(PropertiesQ::Creator(StringQ::Or(vec![
        StringQ::EQ("created_by_20".to_string()),
        StringQ::EndsWith("created_by_10".to_string()),
    ])));

    let objs = get_objects();
    b.iter(|| {
        let _ = objs.iter().map(|o| expr.execute(o)).collect::<Vec<_>>();
    });
}
