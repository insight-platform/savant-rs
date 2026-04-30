use std::time::{SystemTime, UNIX_EPOCH};

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
pub fn relative_time_uuid_v7(uuid: &str, offset_millis: i64) -> PyResult<String> {
    let uuid = Uuid::parse_str(uuid).map_err(|e| PyValueError::new_err(e.to_string()))?;
    savant_core::utils::uuid_v7::relative_time_uuid_v7(uuid, offset_millis)
        .map(|u| u.to_string())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

fn parse_video_id(s: &str) -> PyResult<savant_core::utils::video_id::VideoId> {
    let uuid = Uuid::parse_str(s).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(savant_core::utils::video_id::VideoId::from_uuid(uuid))
}

fn current_wall_clock_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Mint a new VideoId from the process-global generator and return
/// it as a UUID-formatted string.
///
/// Parameters
/// ----------
/// source_id : str
///   Stream identifier the frame belongs to.
/// pts : int
///   Frame PTS in stream timebase.
/// is_keyframe : bool
///   `True` updates the per-source keyframe anchor and may bump the
///   `epoch` field if a PTS reset is detected.
/// wall_clock_ms : int, optional
///   Demuxer wall clock in milliseconds. Defaults to the current
///   system time. Pass an explicit value when replaying or when
///   coordinating ids across processes.
///
/// Returns
/// -------
/// str
///   The minted id, encoded in UUID hyphenated form.
///
#[pyfunction]
#[pyo3(signature = (source_id, pts, is_keyframe, wall_clock_ms = None))]
pub fn mint_video_id(
    source_id: &str,
    pts: i64,
    is_keyframe: bool,
    wall_clock_ms: Option<u64>,
) -> String {
    let ts = wall_clock_ms.unwrap_or_else(current_wall_clock_ms);
    savant_core::utils::video_id::mint(source_id, pts, is_keyframe, ts)
        .as_uuid()
        .to_string()
}

/// Drop process-global state for `source_id`. A subsequent
/// :func:`mint_video_id` for the same source restarts the
/// keyframe-anchor state machine.
#[pyfunction]
pub fn forget_video_id(source_id: &str) {
    savant_core::utils::video_id::forget(source_id);
}

/// Return a copy of `id` with its `ts_ms` component shifted by
/// `offset_millis`. Migration target for
/// :func:`relative_time_uuid_v7`. Deterministic — identical inputs
/// always produce identical output.
///
/// Parameters
/// ----------
/// id : str
///   VideoId in UUID hyphenated form.
/// offset_millis : int
///   Signed shift applied to `ts_ms`.
///
#[pyfunction]
pub fn relative_time_video_id(id: &str, offset_millis: i64) -> PyResult<String> {
    let video_id = parse_video_id(id)?;
    savant_core::utils::video_id::relative_time_video_id(video_id, offset_millis)
        .map(|v| v.as_uuid().to_string())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Inclusive lower bound for a wall-clock-time range scan over
/// frames of `source_id` at `ts_ms`. Pair with
/// :func:`video_id_upper_bound`.
#[pyfunction]
pub fn video_id_lower_bound(source_id: &str, ts_ms: u64) -> String {
    savant_core::utils::video_id::VideoId::lower_bound(source_id, ts_ms)
        .as_uuid()
        .to_string()
}

/// Inclusive upper bound for a wall-clock-time range scan over
/// frames of `source_id` at `ts_ms`.
#[pyfunction]
pub fn video_id_upper_bound(source_id: &str, ts_ms: u64) -> String {
    savant_core::utils::video_id::VideoId::upper_bound(source_id, ts_ms)
        .as_uuid()
        .to_string()
}
