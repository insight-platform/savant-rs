#![feature(test)]

extern crate test;

use savant_rs::test::utils::gen_object;
use savant_rs::utils::pluggable_udf_api::*;
use test::Bencher;

#[bench]
fn bench_udf(b: &mut Bencher) -> anyhow::Result<()> {
    pyo3::prepare_freethreaded_python();

    register_plugin_function(
        "../target/release/libsample_plugin.so",
        "unary_op_even",
        UserFunctionKind::ObjectPredicate,
        "sample.unary_op_even",
    )?;

    let o = gen_object(12);
    b.iter(|| {
        assert!(call_object_predicate("sample.unary_op_even", &[&o]).unwrap());
    });

    Ok(())
}
