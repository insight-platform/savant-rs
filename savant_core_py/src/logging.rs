use crate::release_gil;
use colored::Colorize;
use opentelemetry::trace::{TraceContextExt, TraceId};
use opentelemetry::KeyValue;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use savant_core::otlp::with_current_context;

/// Allows defining a log level for a logging subsystem
///
/// Returns
/// -------
/// :py:class:`LogLevel`
///   The log level. By default, the log level is set to Info.
///
#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq)]
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
        format!("{self:?}")
    }

    fn __repr__(&self) -> String {
        format!("{self:?}")
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
/// Parameters
/// ----------
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
/// Parameters
/// ----------
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

/// Logs a message.
///
/// GIL Management: This function releases the GIL optionally.
///
/// Parameters
/// ----------
/// level: :py:class:`LogLevel`
///   The log level to use.
/// target: str
///   The code initiated the log message. The target is defined as ``a::b::c``
/// message: str
///   The log message.
/// params: dict
///   The log message parameters.
/// no_gil: bool
///   If True, the GIL is released when the function is executed.
///
#[pyfunction]
#[pyo3(name = "log")]
#[pyo3(signature = (level, target, message, params=None, no_gil=true))]
pub fn log_message_gil(
    level: LogLevel,
    target: &str,
    message: &str,
    params: Option<&Bound<'_, PyDict>>,
    no_gil: bool,
) {
    let params: Option<_> = params.map(|params| {
        params
            .iter()
            .map(|(k, v)| KeyValue::new(k.to_string(), v.to_string()))
            .collect::<Vec<_>>()
    });

    release_gil!(no_gil, || {
        log_message(
            level,
            target.to_string().replace('.', "::").as_str(),
            message,
            params,
        );
    });
}

pub fn log_message(level: LogLevel, target: &str, message: &str, params: Option<Vec<KeyValue>>) {
    if log_level_enabled(level) {
        with_current_context(|cx| {
            let trace_id = cx.span().span_context().trace_id();
            let mut params_display = Vec::new();

            if trace_id != TraceId::INVALID {
                params_display.push(format!("trace_id={trace_id}"));
            }

            if let Some(p) = &params {
                params_display.extend(
                    p.iter()
                        .map(|p| format!("{}={}", p.key.as_str(), p.value.as_str())),
                );
            }

            let mut params_display = params_display.join(", ");
            if !params_display.is_empty() {
                params_display = format!("[{params_display}] ");
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
                KeyValue::new("log.target".to_string(), target.to_string()),
                KeyValue::new("event.name".to_string(), "log-record".to_string()),
                KeyValue::new("event.domain".to_string(), "savant"),
            ]);
            cx.span().add_event(message.to_string(), params);
        });
    }
}
