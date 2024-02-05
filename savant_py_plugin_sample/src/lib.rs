extern crate savant_core_py;

use pyo3::prelude::*;
use savant_core_py::primitives::frame::VideoFrame;
use savant_core_py::primitives::object::BorrowedVideoObject;
use std::sync::Arc;

fn with_object<T, F, Q>(o: &PyAny, unlock_gil: bool, f: F) -> Q
where
    Q: Send,
    T: Clone + Send + Sync,
    F: FnOnce(&Arc<T>) -> Q + Send,
{
    let hash = o.hash().unwrap() as usize;
    let mut ptr = unsafe {
        let o = &*(hash as *const T);
        Arc::new(o.clone())
    };

    if unlock_gil {
        Python::with_gil(|py| py.allow_threads(|| f(&mut ptr)))
    } else {
        f(&mut ptr)
    }
}

#[pyfunction]
pub fn access_frame(frame: &PyAny, no_gil: bool) {
    with_object(frame, no_gil, |f: &Arc<VideoFrame>| {
        println!("Frame: {:?}", f.get_uuid());
    });
}

#[pyfunction]
pub fn access_object(o: &PyAny, no_gil: bool) {
    with_object(o, no_gil, |o: &Arc<BorrowedVideoObject>| {
        println!("Object: {:?}", o.get_id());
    });
}

#[pymodule]
fn savant_py_plugin_sample(_: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(access_frame, m)?)?;
    m.add_function(wrap_pyfunction!(access_object, m)?)?;
    Ok(())
}
