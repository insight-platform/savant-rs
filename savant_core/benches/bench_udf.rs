#![feature(test)]

extern crate test;

use savant_core::pluggable_udf_api::{
    call_object_predicate, register_plugin_function, UserFunctionType,
};
use savant_core::test::gen_object;
use test::Bencher;

#[bench]
fn bench_udf(b: &mut Bencher) -> anyhow::Result<()> {
    register_plugin_function(
        "../target/debug/libsavant_core.so",
        "unary_op_even",
        &UserFunctionType::ObjectPredicate,
        "sample.unary_op_even",
    )?;

    let o = gen_object(12);
    b.iter(|| {
        assert!(call_object_predicate("sample.unary_op_even", &[&o]).unwrap());
    });

    Ok(())
}
