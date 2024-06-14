#![feature(test)]

extern crate test;
use test::Bencher;

#[bench]
fn bench_label_filter(b: &mut Bencher) {
    use savant_core::message::label_filter::LabelFilterRule::*;
    let rule = Or(vec![
        Set("test".to_string()),
        Not(Box::new(Or(vec![
            Set("test2".to_string()),
            Set("test3".to_string()),
        ]))),
    ]);

    b.iter(|| {
        rule.matches(&["test".to_string(), "test2".to_string()]);
    });
}
