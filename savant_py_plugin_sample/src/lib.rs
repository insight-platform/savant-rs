extern crate savant_core_py;

use pyo3::prelude::*;
use savant_core_py::primitives::frame::VideoFrame;
use savant_core_py::primitives::object::BorrowedVideoObject;
use std::sync::Arc;

#[pyfunction]
pub fn access_frame(f: &VideoFrame) {
    println!("Frame: {:?}", f.get_uuid());
}

#[pyfunction]
pub fn access_object(o: &BorrowedVideoObject) {
    println!("Object: {:?}", o.get_id());
}

#[pymodule]
fn savant_py_plugin_sample(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(access_frame, m)?)?;
    m.add_function(wrap_pyfunction!(access_object, m)?)?;
    Ok(())
}
