use crate::utils::otlp::with_current_context;
use crate::utils::python::release_gil;
use colored::Colorize;
use opentelemetry::trace::{TraceContextExt, TraceId};
use opentelemetry::KeyValue;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyclass]
#[derive(Debug, Clone, Copy)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warning,
    Error,
    Off,
}

#[pymethods]
impl LogLevel {
    #[new]
    fn new() -> Self {
        LogLevel::Info
    }

    pub(crate) fn __str__(&self) -> String {
        format!("{:?}", self)
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

impl From<LogLevel> for log::LevelFilter {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Trace => log::LevelFilter::Trace,
            LogLevel::Debug => log::LevelFilter::Debug,
            LogLevel::Info => log::LevelFilter::Info,
            LogLevel::Warning => log::LevelFilter::Warn,
            LogLevel::Error => log::LevelFilter::Error,
            LogLevel::Off => log::LevelFilter::Off,
        }
    }
}

impl From<log::LevelFilter> for LogLevel {
    fn from(level: log::LevelFilter) -> Self {
        match level {
            log::LevelFilter::Trace => LogLevel::Trace,
            log::LevelFilter::Debug => LogLevel::Debug,
            log::LevelFilter::Info => LogLevel::Info,
            log::LevelFilter::Warn => LogLevel::Warning,
            log::LevelFilter::Error => LogLevel::Error,
            log::LevelFilter::Off => LogLevel::Off,
        }
    }
}

/// Set the log level for the logger.
///
/// Params
/// ------
/// level: :py:class:`LogLevel`
///   The log level to set.
///
/// Returns
/// -------
/// :py:class:`LogLevel`
///   The previous log level.
///
#[pyfunction]
pub fn set_log_level(level: LogLevel) -> LogLevel {
    // set environment variable to enable logging
    let last_level = get_log_level();
    log::set_max_level(level.into());
    last_level
}

/// Get the current log level for the logger.
///
/// Returns
/// -------
/// :py:class:`LogLevel`
///   The current log level.
///
#[pyfunction]
pub fn get_log_level() -> LogLevel {
    log::max_level().into()
}

/// Check if the given log level is enabled.
///
/// Params
/// ------
/// level: :py:class:`LogLevel`
///   The log level to check.
///
/// Returns
/// -------
/// bool
///   True if the log level is enabled, False otherwise.
///
#[pyfunction]
pub fn log_level_enabled(level: LogLevel) -> bool {
    log::max_level().ge(&log::LevelFilter::from(level))
}

/// Log a message.
///
/// GIL Management: This function releases the GIL.
///
/// Params
/// ------
/// level: :py:class:`LogLevel`
///   The log level to use.
/// target: str
///   The code initiated the log message. The target is defined as ``a::b::c`
/// message: str
///   The log message.
/// params: dict
///   The log message parameters.
///
#[pyfunction]
#[pyo3(name = "log")]
#[pyo3(signature = (level, target, message, params=None))]
fn log_message_py(level: LogLevel, target: String, message: String, params: Option<&PyDict>) {
    let params: Option<_> = params.map(|params| {
        params
            .iter()
            .map(|(k, v)| KeyValue::new(k.to_string(), v.to_string()))
            .collect::<Vec<_>>()
    });

    release_gil(|| {
        log_message(level, target, message, params);
    });
}

pub fn log_message(
    level: LogLevel,
    target: String,
    message: String,
    params: Option<Vec<KeyValue>>,
) {
    if log_level_enabled(level) {
        with_current_context(|cx| {
            let trace_id = cx.span().span_context().trace_id();
            let mut params_display = Vec::new();

            if trace_id != TraceId::INVALID {
                params_display.push(format!("trace_id={}", trace_id));
            }

            if let Some(p) = &params {
                params_display.extend(
                    p.iter()
                        .map(|p| format!("{}={}", p.key.as_str(), p.value.as_str())),
                );
            }

            let mut params_display = params_display.join(", ");
            if !params_display.is_empty() {
                params_display = format!("[{}] ", params_display);
            }

            let params_str = &params_display.purple();
            match level {
                LogLevel::Trace => {
                    log::trace!(target: &target, "{}{}", params_str, &message)
                }
                LogLevel::Debug => {
                    log::debug!(target: &target, "{}{}", params_str, &message)
                }
                LogLevel::Info => {
                    log::info!(target: &target, "{}{}", params_str, &message)
                }
                LogLevel::Warning => {
                    log::warn!(target: &target, "{}{}", params_str, &message)
                }
                LogLevel::Error => {
                    log::error!(target: &target, "{}{}", params_str, &message)
                }
                LogLevel::Off => {}
            }

            let mut params = params.unwrap_or_default();
            params.extend(vec![
                KeyValue::new("log.level".to_string(), level.__str__()),
                KeyValue::new("log.target".to_string(), target.clone()),
                KeyValue::new("event.name".to_string(), "log-record".to_string()),
                KeyValue::new("event.domain".to_string(), "savant"),
            ]);
            cx.span().add_event(message, params);
        });
    }
}

#[pymodule]
pub(crate) fn logging(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LogLevel>()?;
    m.add_function(wrap_pyfunction!(set_log_level, m)?)?;
    m.add_function(wrap_pyfunction!(get_log_level, m)?)?;
    m.add_function(wrap_pyfunction!(log_level_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(log_message_py, m)?)?;
    Ok(())
}
