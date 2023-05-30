pub mod bbox;
pub mod conversions;
pub mod fps_meter;
pub mod np;
pub mod pluggable_udf_api;
pub mod python;
pub mod symbol_mapper;

use pyo3::prelude::*;

use crate::primitives::message::loader::load_message_gil;
use crate::primitives::message::saver::save_message_gil;
use crate::test::utils::gen_frame;
use crate::utils::np::np_nalgebra;
use crate::utils::np::np_ndarray;
use crate::utils::symbol_mapper::RegistrationPolicy;
use crate::utils::symbol_mapper::SymbolMapper;
use crate::utils::symbol_mapper::{
    build_model_object_key_gil, clear_symbol_maps_gil, dump_registry_gil, get_model_id_gil,
    get_model_name_gil, get_object_id_gil, get_object_ids_gil, get_object_label_gil,
    get_object_labels_gil, is_model_registered_gil, is_object_registered_gil,
    parse_compound_key_gil, register_model_objects_gil, validate_base_key_gil,
};

use crate::utils::pluggable_udf_api::{
    call_object_inplace_modifier_gil, call_object_map_modifier_gil, call_object_predicate_gil,
    is_plugin_function_registered_gil, register_plugin_function_gil, UserFunctionKind,
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
    m.add_function(wrap_pyfunction!(save_message_gil, m)?)?;
    m.add_function(wrap_pyfunction!(load_message_gil, m)?)?;
    // utility
    m.add_function(wrap_pyfunction!(round_2_digits, m)?)?;

    // bbox batch ops
    m.add_function(wrap_pyfunction!(rotated_bboxes_to_ndarray_gil, m)?)?;
    m.add_function(wrap_pyfunction!(bboxes_to_ndarray_gil, m)?)?;
    m.add_function(wrap_pyfunction!(ndarray_to_bboxes_gil, m)?)?;
    m.add_function(wrap_pyfunction!(ndarray_to_rotated_bboxes_gil, m)?)?;

    // numpy utils
    m.add_function(wrap_pyfunction!(np_to_matrix_gil, m)?)?;
    m.add_function(wrap_pyfunction!(matrix_to_np_gil, m)?)?;
    m.add_function(wrap_pyfunction!(np_to_ndarray_gil, m)?)?;
    m.add_function(wrap_pyfunction!(ndarray_to_np_gil, m)?)?;

    // UDF API
    m.add_function(wrap_pyfunction!(register_plugin_function_gil, m)?)?;
    m.add_function(wrap_pyfunction!(is_plugin_function_registered_gil, m)?)?;
    m.add_function(wrap_pyfunction!(call_object_predicate_gil, m)?)?;
    m.add_function(wrap_pyfunction!(call_object_inplace_modifier_gil, m)?)?;
    m.add_function(wrap_pyfunction!(call_object_map_modifier_gil, m)?)?;

    // model object registry
    m.add_function(wrap_pyfunction!(build_model_object_key_gil, m)?)?;
    m.add_function(wrap_pyfunction!(clear_symbol_maps_gil, m)?)?;
    m.add_function(wrap_pyfunction!(dump_registry_gil, m)?)?;
    m.add_function(wrap_pyfunction!(get_model_id_gil, m)?)?;
    m.add_function(wrap_pyfunction!(get_model_name_gil, m)?)?;
    m.add_function(wrap_pyfunction!(get_object_id_gil, m)?)?;
    m.add_function(wrap_pyfunction!(get_object_ids_gil, m)?)?;
    m.add_function(wrap_pyfunction!(get_object_label_gil, m)?)?;
    m.add_function(wrap_pyfunction!(get_object_labels_gil, m)?)?;
    m.add_function(wrap_pyfunction!(is_model_registered_gil, m)?)?;
    m.add_function(wrap_pyfunction!(is_object_registered_gil, m)?)?;
    m.add_function(wrap_pyfunction!(parse_compound_key_gil, m)?)?;
    m.add_function(wrap_pyfunction!(register_model_objects_gil, m)?)?;
    m.add_function(wrap_pyfunction!(validate_base_key_gil, m)?)?;

    m.add_class::<FpsMeter>()?;
    m.add_class::<SymbolMapper>()?;
    m.add_class::<RegistrationPolicy>()?;
    m.add_class::<BBoxFormat>()?;
    m.add_class::<NalgebraDMatrix>()?;
    m.add_class::<NDarray>()?;
    m.add_class::<UserFunctionKind>()?;

    Ok(())
}
