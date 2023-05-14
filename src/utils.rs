pub mod fps_meter;
pub mod symbol_mapper;

use pyo3::prelude::*;

use crate::primitives::message::loader::load_message_py;
use crate::primitives::message::saver::save_message_py;
use crate::test::utils::gen_frame;
use crate::utils::symbol_mapper::SymbolMapper;
use crate::utils::symbol_mapper::{get_object_id, register_model_objects};

pub use fps_meter::FpsMeter;

#[pymodule]
pub fn utils(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(save_message_py, m)?)?;
    m.add_function(wrap_pyfunction!(load_message_py, m)?)?;
    m.add_function(wrap_pyfunction!(gen_frame, m)?)?;

    m.add_function(wrap_pyfunction!(get_object_id, m)?)?;
    m.add_function(wrap_pyfunction!(register_model_objects, m)?)?;

    m.add_class::<FpsMeter>()?;
    m.add_class::<SymbolMapper>()?;

    Ok(())
}
