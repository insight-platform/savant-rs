use crate::logging::{log_level_enabled, LogLevel};
use crate::utils::otlp::with_current_context;
use opentelemetry::trace::TraceContextExt;
use opentelemetry::KeyValue;
use pyo3::marker::Ungil;
use pyo3::prelude::*;
use std::time::SystemTime;

#[inline(always)]
pub(crate) fn current_nanos_trace() -> u128 {
    if log_level_enabled(LogLevel::Trace) {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    } else {
        0
    }
}

pub(crate) fn report_gil_wait(start_wait: u128, py: Python) {
    let end_wait = current_nanos_trace();
    if log_level_enabled(LogLevel::Trace) {
        py.allow_threads(|| {
            with_current_context(|cx| {
                cx.span().add_event(
                    "gil-wait".to_string(),
                    vec![
                        KeyValue::new("gil-wait.start", start_wait.to_string()),
                        KeyValue::new("gil-wait.end", end_wait.to_string()),
                        KeyValue::new("gil-wait.duration", (end_wait - start_wait).to_string()),
                        KeyValue::new(
                            "gil-wait.thread.id",
                            format!("{:?}", std::thread::current().id()),
                        ),
                        KeyValue::new(
                            "gil-wait.thread.name",
                            format!("{:?}", std::thread::current().name()),
                        ),
                    ],
                )
            })
        });
    }
}
#[inline(always)]
pub fn release_gil<T, F>(f: F) -> T
where
    F: Ungil + FnOnce() -> T,
    T: Ungil,
{
    with_gil(|py| py.allow_threads(f))
}

#[inline(always)]
pub fn with_gil<F, T>(f: F) -> T
where
    F: FnOnce(Python<'_>) -> T,
{
    let start_wait = current_nanos_trace();
    Python::with_gil(|py| {
        report_gil_wait(start_wait, py);
        f(py)
    })
}
