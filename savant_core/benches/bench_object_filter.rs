use criterion::{criterion_group, criterion_main, Criterion};
use savant_core::eval_resolvers::register_utility_resolver;
use savant_core::match_query::MatchQuery::*;
use savant_core::match_query::*;
use savant_core::primitives::attribute_value::AttributeValue;
use savant_core::primitives::object::{
    IdCollisionResolutionPolicy, VideoObject, VideoObjectBuilder,
};
use savant_core::primitives::{Attribute, RBBox, WithAttributes};
use savant_core::test::gen_empty_frame;
use std::hint::black_box;

const COUNT: i64 = 100;

fn get_objects() -> Vec<VideoObject> {
    (0..COUNT)
        .map(|i| {
            let mut o = VideoObjectBuilder::default()
                .namespace(format!("created_by_{i}"))
                .label(format!("label_{i}"))
                .id(i)
                .confidence(Some(0.53))
                .detection_box(RBBox::new(0.0, 0.0, 1.0, 1.0, None))
                .track_id(Some(i))
                .track_box(Some(RBBox::new(10.0, 20.0, 21.0, 231.0, None)))
                .build()
                .unwrap();
            o.set_attribute(Attribute::persistent(
                "test",
                "test",
                vec![AttributeValue::integer(1, None)],
                &Some("hint"),
                false,
            ));
            o
        })
        .collect::<Vec<_>>()
}

fn object_filter_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("object_filter");

    group.bench_function("complex_filtering", |b| {
        let expr = and![
            or![
                Namespace(one_of(&["created_by_2", "created_by_4"])),
                or![
                    Label(ends_with("2")),
                    Label(ends_with("4")),
                    Label(ends_with("6")),
                ]
            ],
            BoxAngleDefined,
            ParentDefined,
            or![Confidence(ge(0.6)), Confidence(le(0.4)),]
        ];

        let objs = get_objects();
        let frame = gen_empty_frame();
        for o in objs {
            frame
                .add_object(o, IdCollisionResolutionPolicy::Error)
                .unwrap();
        }
        b.iter(|| {
            black_box(frame.access_objects(&expr));
        })
    });

    group.bench_function("idle_filtering", |b| {
        let expr = stop_if_false!(Idle);

        let objs = get_objects();
        let frame = gen_empty_frame();
        for o in objs {
            frame
                .add_object(o, IdCollisionResolutionPolicy::Error)
                .unwrap();
        }
        b.iter(|| {
            let r = black_box(frame.access_objects(&expr));
            assert_eq!(r.len(), COUNT as usize);
        })
    });

    group.bench_function("idle_quick_break", |b| {
        let expr = stop_if_true!(Idle);

        let objs = get_objects();
        let frame = gen_empty_frame();
        for o in objs {
            frame
                .add_object(o, IdCollisionResolutionPolicy::Error)
                .unwrap();
        }
        b.iter(|| {
            let r = black_box(frame.access_objects(&expr));
            assert_eq!(r.len(), 1);
        })
    });

    group.bench_function("eval_filtering", |b| {
        register_utility_resolver();

        let expr = EvalExpr(
            r#"
            ((namespace == "created_by_4" || namespace == "created_by_2") ||
             (label == "2" || label == "4" || label == "6")) &&
            !is_empty(parent.id) &&
            !is_empty(bbox.angle) &&
            (confidence > 0.6 || confidence < 0.4)"#
                .to_string(),
        );

        let objs = get_objects();
        let frame = gen_empty_frame();
        for o in objs {
            frame
                .add_object(o, IdCollisionResolutionPolicy::Error)
                .unwrap();
        }
        b.iter(|| {
            black_box(frame.access_objects(&expr));
        })
    });

    group.bench_function("empty_filtering", |b| {
        let expr = Idle;
        let objs = get_objects();
        let frame = gen_empty_frame();
        for o in objs {
            frame
                .add_object(o, IdCollisionResolutionPolicy::Error)
                .unwrap();
        }
        b.iter(|| {
            black_box(frame.access_objects(&expr));
        })
    });

    group.bench_function("simple_filtering", |b| {
        let expr = or![
            Namespace(eq("created_by_20")),
            Namespace(ends_with("created_by_10")),
        ];

        let objs = get_objects();
        let frame = gen_empty_frame();
        for o in objs {
            frame
                .add_object(o, IdCollisionResolutionPolicy::Error)
                .unwrap();
        }
        b.iter(|| {
            black_box(frame.access_objects(&expr));
        })
    });

    group.bench_function("all_objects", |b| {
        let objs = get_objects();
        let frame = gen_empty_frame();
        for o in objs {
            frame
                .add_object(o, IdCollisionResolutionPolicy::Error)
                .unwrap();
        }
        b.iter(|| {
            black_box(frame.get_all_objects());
        })
    });

    group.finish();
}

criterion_group!(benches, object_filter_benchmarks);
criterion_main!(benches);
