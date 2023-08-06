use crate::utils::otlp::with_current_context;
use opentelemetry::trace::TraceContextExt;
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone, Copy)]
enum LogLevel {
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

    fn __str__(&self) -> String {
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
fn set_log_level(level: LogLevel) {
    log::set_max_level(level.into());
}

#[pyfunction]
fn get_log_level() -> LogLevel {
    log::max_level().into()
}

#[pyfunction]
fn level_enabled(level: LogLevel) -> bool {
    log::max_level().ge(&log::LevelFilter::from(level))
}

#[pyfunction]
#[pyo3(name = "log")]
fn log_message(level: LogLevel, message: String) {
    match level {
        LogLevel::Trace => log::trace!("{}", &message),
        LogLevel::Debug => log::debug!("{}", &message),
        LogLevel::Info => log::info!("{}", &message),
        LogLevel::Warning => log::warn!("{}", &message),
        LogLevel::Error => log::error!("{}", &message),
        LogLevel::Off => {}
    }
    if level_enabled(level) {
        with_current_context(|cx| {
            cx.span().add_event(
                format!("{}: {}", level.__str__().to_uppercase(), message),
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
    m.add_function(wrap_pyfunction!(level_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(log_message, m)?)?;
    Ok(())
}
