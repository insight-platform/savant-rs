use crate::logging::{log_level_enabled, LogLevel};
use crate::utils::otlp::with_current_context;
use opentelemetry::trace::TraceContextExt;
use opentelemetry::KeyValue;
use pyo3::marker::Ungil;
use pyo3::prelude::*;
use std::time::Instant;

pub(crate) fn report_gil_wait(start: &Instant, py: Python) {
    let duration = start.elapsed();
    if log_level_enabled(LogLevel::Trace) {
        py.allow_threads(|| {
            with_current_context(|cx| {
                cx.span().add_event(
                    "gil-wait".to_string(),
                    vec![
                        KeyValue::new(
                            "gil-wait.duration",
                            i64::try_from(duration.as_nanos()).unwrap_or(-1),
                        ),
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
    if log_level_enabled(LogLevel::Trace) {
        let start_wait = Python::with_gil(|py| py.allow_threads(|| Instant::now()));
        Python::with_gil(|py| report_gil_wait(&start_wait, py));
    }
    Python::with_gil(f)
}
