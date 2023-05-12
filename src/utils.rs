pub mod fps_meter;
use pyo3::prelude::*;

use crate::primitives::message::loader::load_message_py;
use crate::primitives::message::saver::save_message_py;
use crate::test::utils::gen_frame;

pub use fps_meter::FpsMeter;

#[pymodule]
pub fn utils(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(save_message_py, m)?)?;
    m.add_function(wrap_pyfunction!(load_message_py, m)?)?;
    m.add_function(wrap_pyfunction!(gen_frame, m)?)?;

    m.add_class::<FpsMeter>()?;

    Ok(())
}
