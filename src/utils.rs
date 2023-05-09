pub mod fps_meter;
use pyo3::prelude::*;

use crate::primitives::*;
use crate::test::utils::gen_frame;

#[pymodule]
pub fn utils(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(save_message, m)?)?;
    m.add_function(wrap_pyfunction!(load_message, m)?)?;
    m.add_function(wrap_pyfunction!(gen_frame, m)?)?;
    Ok(())
}
