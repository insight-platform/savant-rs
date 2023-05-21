pub mod bbox;
pub mod conversions;
pub mod fps_meter;
pub mod np;
pub mod python;
pub mod symbol_mapper;

use pyo3::prelude::*;

use crate::primitives::message::loader::load_message_py;
use crate::primitives::message::saver::save_message_py;
use crate::test::utils::gen_frame;
use crate::utils::np::np_nalgebra;
use crate::utils::np::np_ndarray;
use crate::utils::symbol_mapper::RegistrationPolicy;
use crate::utils::symbol_mapper::SymbolMapper;
use crate::utils::symbol_mapper::{
    build_model_object_key, clear_symbol_maps, dump_registry, get_model_id_py, get_model_name,
    get_object_id_py, get_object_ids, get_object_label, get_object_labels, is_model_registered,
    is_object_registered, parse_compound_key, register_model_objects, validate_base_key,
};

pub use bbox::*;
pub use fps_meter::FpsMeter;
pub use np_nalgebra::*;
pub use np_ndarray::*;

#[pyfunction]
#[inline]
pub fn round_2_digits(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

#[pymodule]
pub fn utils(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gen_frame, m)?)?;
    // ser deser
    m.add_function(wrap_pyfunction!(save_message_py, m)?)?;
    m.add_function(wrap_pyfunction!(load_message_py, m)?)?;
    // utility
    m.add_function(wrap_pyfunction!(round_2_digits, m)?)?;

    // bbox batch ops
    m.add_function(wrap_pyfunction!(rotated_bboxes_to_ndarray_py, m)?)?;
    m.add_function(wrap_pyfunction!(bboxes_to_ndarray_py, m)?)?;
    m.add_function(wrap_pyfunction!(ndarray_to_bboxes_py, m)?)?;
    m.add_function(wrap_pyfunction!(ndarray_to_rotated_bboxes_py, m)?)?;

    // numpy utils
    m.add_function(wrap_pyfunction!(np_to_matrix_py, m)?)?;
    m.add_function(wrap_pyfunction!(matrix_to_np_py, m)?)?;
    m.add_function(wrap_pyfunction!(np_to_ndarray_py, m)?)?;
    m.add_function(wrap_pyfunction!(ndarray_to_np_py, m)?)?;

    // model object registry
    m.add_function(wrap_pyfunction!(build_model_object_key, m)?)?;
    m.add_function(wrap_pyfunction!(clear_symbol_maps, m)?)?;
    m.add_function(wrap_pyfunction!(dump_registry, m)?)?;
    m.add_function(wrap_pyfunction!(get_model_id_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_model_name, m)?)?;
    m.add_function(wrap_pyfunction!(get_object_id_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_object_ids, m)?)?;
    m.add_function(wrap_pyfunction!(get_object_label, m)?)?;
    m.add_function(wrap_pyfunction!(get_object_labels, m)?)?;
    m.add_function(wrap_pyfunction!(is_model_registered, m)?)?;
    m.add_function(wrap_pyfunction!(is_object_registered, m)?)?;
    m.add_function(wrap_pyfunction!(parse_compound_key, m)?)?;
    m.add_function(wrap_pyfunction!(register_model_objects, m)?)?;
    m.add_function(wrap_pyfunction!(validate_base_key, m)?)?;

    m.add_class::<FpsMeter>()?;
    m.add_class::<SymbolMapper>()?;
    m.add_class::<RegistrationPolicy>()?;
    m.add_class::<BBoxFormat>()?;
    m.add_class::<NalgebraDMatrix>()?;
    m.add_class::<NDarray>()?;

    Ok(())
}
