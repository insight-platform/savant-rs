use evalexpr::Value;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub use fps_meter::FpsMeter;

use crate::logging::{log_level_enabled, LogLevel};
use crate::primitives::bbox::{BBoxMetricType, VideoObjectBBoxTransformation};
use crate::primitives::message::loader::*;
use crate::primitives::message::loader::{
    load_message_from_bytebuffer_gil, load_message_from_bytes_gil,
};
use crate::primitives::message::saver::*;
use crate::primitives::message::saver::{
    save_message_to_bytebuffer_gil, save_message_to_bytes_gil,
};
use crate::primitives::message::Message;
use crate::primitives::objects_view::VideoObjectBBoxType;
use crate::test::utils::{gen_empty_frame, gen_frame};
use crate::utils::byte_buffer::ByteBuffer;
use crate::utils::otlp::{MaybeTelemetrySpan, PropagatedContext, TelemetrySpan};
use crate::utils::pluggable_udf_api::{
    call_object_inplace_modifier, call_object_map_modifier, call_object_predicate,
    is_plugin_function_registered, register_plugin_function, UserFunctionType,
};
use crate::utils::symbol_mapper::RegistrationPolicy;
use crate::utils::symbol_mapper::{
    build_model_object_key_py, clear_symbol_maps_py, dump_registry_gil, get_model_id_py,
    get_model_name_py, get_object_id_py, get_object_ids_py, get_object_label_py,
    get_object_labels_py, is_model_registered_py, is_object_registered_py, parse_compound_key_py,
    register_model_objects_py, validate_base_key_py,
};
use crate::{release_gil, with_gil};

pub mod byte_buffer;
pub mod eval_resolvers;
pub mod fps_meter;
pub mod otlp;
pub mod pluggable_udf_api;
pub mod python;
pub mod symbol_mapper;

#[pyfunction]
#[inline]
pub fn round_2_digits(v: f32) -> f32 {
    savant_core::round_2_digits(v)
}

/// When loglevel is set to Trace reports the number of nanoseconds spent waiting for the GIL
/// The report is sent to the current telemetry span
///
#[pyfunction]
pub fn estimate_gil_contention() {
    if log_level_enabled(LogLevel::Trace) {
        with_gil!(|_| {});
    }
}

fn value_to_py(py: Python, v: Value) -> PyResult<PyObject> {
    match v {
        Value::String(v) => Ok(v.to_object(py)),
        Value::Float(v) => Ok(v.to_object(py)),
        Value::Int(v) => Ok(v.to_object(py)),
        Value::Boolean(v) => Ok(v.to_object(py)),
        Value::Tuple(v) => {
            let mut res = Vec::with_capacity(v.len());
            for v in v {
                res.push(value_to_py(py, v)?);
            }
            Ok(res.to_object(py))
        }
        Value::Empty => Ok(None::<()>.to_object(py)),
    }
}

#[pyfunction]
#[pyo3(name = "eval_expr")]
#[pyo3(signature = (query, ttl = 100, no_gil = true))]
pub fn eval_expr(query: &str, ttl: u64, no_gil: bool) -> PyResult<(PyObject, bool)> {
    let (res, cached) = release_gil!(no_gil, || savant_core::eval_cache::eval_expr(query, ttl)
        .map_err(|e| PyValueError::new_err(e.to_string())))?;
    let v = with_gil!(|py| value_to_py(py, res))?;
    Ok((v, cached))
}

/// Enables deadlock detection
///
#[pyfunction]
pub fn enable_dl_detection() {
    savant_core::deadlock_detection::enable_dl_detection();
}

#[pymodule]
pub fn symbol_mapper_module(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_model_object_key_py, m)?)?;
    m.add_function(wrap_pyfunction!(clear_symbol_maps_py, m)?)?;
    m.add_function(wrap_pyfunction!(dump_registry_gil, m)?)?;
    m.add_function(wrap_pyfunction!(get_model_id_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_model_name_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_object_id_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_object_ids_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_object_label_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_object_labels_py, m)?)?;
    m.add_function(wrap_pyfunction!(is_model_registered_py, m)?)?;
    m.add_function(wrap_pyfunction!(is_object_registered_py, m)?)?;
    m.add_function(wrap_pyfunction!(parse_compound_key_py, m)?)?;
    m.add_function(wrap_pyfunction!(register_model_objects_py, m)?)?;
    m.add_function(wrap_pyfunction!(validate_base_key_py, m)?)?;

    m.add_class::<RegistrationPolicy>()?;

    Ok(())
}

#[pymodule]
pub fn udf_api_module(_py: Python, m: &PyModule) -> PyResult<()> {
    // UDF API
    m.add_function(wrap_pyfunction!(register_plugin_function, m)?)?;
    m.add_function(wrap_pyfunction!(is_plugin_function_registered, m)?)?;
    m.add_function(wrap_pyfunction!(call_object_predicate, m)?)?;
    m.add_function(wrap_pyfunction!(call_object_inplace_modifier, m)?)?;
    m.add_function(wrap_pyfunction!(call_object_map_modifier, m)?)?;

    m.add_class::<UserFunctionType>()?;
    Ok(())
}

#[pymodule]
pub fn serialization_module(_py: Python, m: &PyModule) -> PyResult<()> {
    // ser deser
    m.add_function(wrap_pyfunction!(save_message, m)?)?;
    m.add_function(wrap_pyfunction!(save_message_to_bytebuffer_gil, m)?)?;
    m.add_function(wrap_pyfunction!(save_message_to_bytes_gil, m)?)?;

    m.add_function(wrap_pyfunction!(load_message, m)?)?;
    m.add_function(wrap_pyfunction!(load_message_from_bytebuffer_gil, m)?)?;
    m.add_function(wrap_pyfunction!(load_message_from_bytes_gil, m)?)?;

    m.add_class::<Message>()?;
    Ok(())
}

#[pymodule]
pub fn utils(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(eval_expr, m)?)?;
    m.add_function(wrap_pyfunction!(gen_frame, m)?)?;
    m.add_function(wrap_pyfunction!(gen_empty_frame, m)?)?;
    // utility
    m.add_function(wrap_pyfunction!(round_2_digits, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_gil_contention, m)?)?;
    m.add_function(wrap_pyfunction!(enable_dl_detection, m)?)?;

    m.add_class::<PropagatedContext>()?;
    m.add_class::<TelemetrySpan>()?;
    m.add_class::<MaybeTelemetrySpan>()?;
    m.add_class::<FpsMeter>()?;
    m.add_class::<ByteBuffer>()?;
    m.add_class::<VideoObjectBBoxType>()?;
    m.add_class::<VideoObjectBBoxTransformation>()?;
    m.add_class::<BBoxMetricType>()?;

    Ok(())
}
