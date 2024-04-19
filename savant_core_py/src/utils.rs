use evalexpr::Value;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::logging::{log_level_enabled, LogLevel};
use crate::{release_gil, with_gil};

pub mod byte_buffer;
pub mod eval_resolvers;
pub mod otlp;
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

#[pyfunction]
pub fn incremental_uuid_v7() -> String {
    savant_core::utils::uuid_v7::incremental_uuid_v7().to_string()
}
