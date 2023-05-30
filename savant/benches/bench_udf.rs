#![feature(test)]

extern crate test;

use pyo3::prelude::*;
use savant_rs::test::utils::gen_object;
use savant_rs::utils::pluggable_udf_api::*;
use test::Bencher;

#[bench]
fn bench_udf(b: &mut Bencher) -> anyhow::Result<()> {
    pyo3::prepare_freethreaded_python();

    let mut p = UserFunctionPluginFactory::new("../target/release/libsample_plugin.so")?;
    p.register_function("unary_op_even", UserFunctionKind::UnaryObjectPredicate)?;
    let p = p.initialize();

    let o = gen_object(12);
    b.iter(|| {
        assert!(p.eval("unary_op_even", &[&o]).unwrap());
    });

    Ok(())
}
