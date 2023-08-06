use crate::utils::otlp::with_current_context;
use opentelemetry::trace::TraceContextExt;
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone, Copy)]
pub(crate) enum LogLevel {
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

#[pyfunction]
pub(crate) fn set_log_level(level: LogLevel) {
    // set environment variable to enable logging
    log::set_max_level(level.into());
}

#[pyfunction]
fn get_log_level() -> LogLevel {
    log::max_level().into()
}

#[pyfunction]
fn log_level_enabled(level: LogLevel) -> bool {
    log::max_level().ge(&log::LevelFilter::from(level))
}

#[pyfunction]
#[pyo3(name = "log")]
fn log_message(level: LogLevel, target: String, message: String) {
    if log_level_enabled(level) {
        match level {
            LogLevel::Trace => log::trace!(target: &target, "{}", &message),
            LogLevel::Debug => log::debug!(target: &target, "{}", &message),
            LogLevel::Info => log::info!(target: &target, "{}", &message),
            LogLevel::Warning => log::warn!(target: &target, "{}", &message),
            LogLevel::Error => log::error!(target: &target, "{}", &message),
            LogLevel::Off => {}
        }

        with_current_context(|cx| {
            cx.span().add_event(
                format!("{} {} {}", level.__str__().to_uppercase(), target, message,),
                vec![],
            );
        });
    }
}

#[pymodule]
pub(crate) fn logging(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<LogLevel>()?;
    m.add_function(wrap_pyfunction!(set_log_level, m)?)?;
    m.add_function(wrap_pyfunction!(get_log_level, m)?)?;
    m.add_function(wrap_pyfunction!(log_level_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(log_message, m)?)?;
    Ok(())
}
