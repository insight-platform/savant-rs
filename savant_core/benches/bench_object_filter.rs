#![feature(test)]

extern crate test;

use savant_core::eval_resolvers::register_utility_resolver;
use savant_core::match_query::MatchQuery::*;
use savant_core::match_query::*;
use savant_core::primitives::attribute_value::{AttributeValue, AttributeValueVariant};
use savant_core::primitives::object::{
    IdCollisionResolutionPolicy, VideoObject, VideoObjectBuilder,
};
use savant_core::primitives::rust::VideoObjectProxy;
use savant_core::primitives::{Attribute, Attributive, RBBox};
use savant_core::test::gen_empty_frame;
use test::Bencher;

fn get_objects() -> Vec<VideoObject> {
    (0..100)
        .into_iter()
        .map(|i| {
            let mut o = VideoObjectBuilder::default()
                .namespace(format!("created_by_{i}"))
                .label(format!("label_{i}"))
                .id(i)
                .confidence(Some(0.53))
                .detection_box(RBBox::new(0.0, 0.0, 1.0, 1.0, None).try_into().unwrap())
                .track_id(Some(i))
                .track_box(Some(
                    RBBox::new(10.0, 20.0, 21.0, 231.0, None)
                        .try_into()
                        .unwrap(),
                ))
                .build()
                .unwrap();
            o.set_attribute(Attribute::persistent(
                "test".to_string(),
                "test".to_string(),
                vec![AttributeValue::new(AttributeValueVariant::Integer(1), None)],
                Some("hint".to_string()),
            ));
            VideoObject::from(o)
        })
        .collect::<Vec<_>>()
}

#[bench]
fn bench_filtering(b: &mut Bencher) {
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
            .add_object(
                &VideoObjectProxy::from(o),
                IdCollisionResolutionPolicy::Error,
            )
            .unwrap();
    }
    b.iter(|| {
        let _ = frame.access_objects(&expr);
    });
}

#[bench]
fn bench_filtering_with_eval(b: &mut Bencher) {
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
    let mut frame = gen_empty_frame();
    frame.set_parallelized(true);
    for o in objs {
        frame
            .add_object(
                &VideoObjectProxy::from(o),
                IdCollisionResolutionPolicy::Error,
            )
            .unwrap();
    }
    b.iter(|| {
        let _ = frame.access_objects(&expr);
    });
}

#[bench]
fn bench_empty_filtering(b: &mut Bencher) {
    let expr = MatchQuery::Idle;
    let objs = get_objects();
    let frame = gen_empty_frame();
    for o in objs {
        frame
            .add_object(
                &VideoObjectProxy::from(o),
                IdCollisionResolutionPolicy::Error,
            )
            .unwrap();
    }
    b.iter(|| {
        let _ = frame.access_objects(&expr);
    });
}

#[bench]
fn bench_simple_filtering(b: &mut Bencher) {
    let expr = or![
        Namespace(eq("created_by_20")),
        Namespace(ends_with("created_by_10")),
    ];

    let objs = get_objects();
    let frame = gen_empty_frame();
    for o in objs {
        frame
            .add_object(
                &VideoObjectProxy::from(o),
                IdCollisionResolutionPolicy::Error,
            )
            .unwrap();
    }
    b.iter(|| {
        let _ = frame.access_objects(&expr);
    });
}

#[bench]
fn bench_all_objects(b: &mut Bencher) {
    let objs = get_objects();
    let frame = gen_empty_frame();
    for o in objs {
        frame
            .add_object(
                &VideoObjectProxy::from(o),
                IdCollisionResolutionPolicy::Error,
            )
            .unwrap();
    }
    b.iter(|| {
        let _ = frame.get_all_objects();
    });
}
