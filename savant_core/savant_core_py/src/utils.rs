use evalexpr::Value;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use uuid::Uuid;

use crate::logging::{log_level_enabled, LogLevel};
use crate::{attach, detach};

pub mod bigint;
pub mod byte_buffer;
pub mod eval_resolvers;
pub mod otlp;
pub mod python;
pub mod symbol_mapper;

#[pyfunction]
pub fn check_pybound_name(bo: &Bound<'_, PyAny>, expected_name: &str) -> PyResult<()> {
    let name = bo.get_type().name()?.to_string();
    if name.as_str() != expected_name {
        return Err(PyValueError::new_err(format!(
            "Ingress function must be a StageFunction, got {name}"
        )));
    }
    Ok(())
}

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
        attach!(|_| {});
    }
}

fn value_to_py(py: Python, v: Value) -> PyResult<Py<PyAny>> {
    Ok(match v {
        Value::String(v) => v.into_pyobject(py)?.into_any().unbind(),
        Value::Float(v) => v.into_pyobject(py)?.into_any().unbind(),
        Value::Int(v) => v.into_pyobject(py)?.into_any().unbind(),
        Value::Boolean(v) => {
            let v = v.into_pyobject(py)?.to_owned();
            v.into_any().unbind()
        }
        Value::Tuple(v) => {
            let mut res = Vec::with_capacity(v.len());
            for v in v {
                res.push(value_to_py(py, v)?);
            }
            res.into_pyobject(py)?.into_any().unbind()
        }
        Value::Empty => None::<()>.into_pyobject(py)?.into_any().unbind(),
    })
}

#[pyfunction]
#[pyo3(name = "eval_expr")]
#[pyo3(signature = (query, ttl = 100, no_gil = true))]
pub fn eval_expr(query: &str, ttl: u64, no_gil: bool) -> PyResult<(Py<PyAny>, bool)> {
    let (res, cached) = detach!(no_gil, || savant_core::eval_cache::eval_expr(query, ttl)
        .map_err(|e| PyValueError::new_err(e.to_string())))?;
    let v = attach!(|py| value_to_py(py, res))?;
    Ok((v, cached))
}

/// Enables deadlock detection
///
#[pyfunction]
pub fn enable_dl_detection() {
    savant_core::deadlock_detection::enable_dl_detection();
}

/// Returns a new UUID v7
///
#[pyfunction]
pub fn incremental_uuid_v7() -> String {
    savant_core::utils::uuid_v7::incremental_uuid_v7().to_string()
}

/// Returns a new UUID v7 that is offset from the given UUID by the given number of milliseconds
///
/// The offset can be positive or negative
///
/// Parameters
/// ----------
/// uuid : str
///   The UUID to offset
/// offset_millis : int
///   The number of milliseconds to offset the UUID by
///
/// Returns
/// -------
/// str
///   The new UUID
///
#[pyfunction]
pub fn relative_time_uuid_v7(uuid: &str, offset_millis: i64) -> String {
    let uuid = Uuid::parse_str(uuid).unwrap();
    savant_core::utils::uuid_v7::relative_time_uuid_v7(uuid, offset_millis).to_string()
}
