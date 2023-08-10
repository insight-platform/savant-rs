use crate::utils::otlp::with_current_context;
use opentelemetry::trace::TraceContextExt;
use opentelemetry::KeyValue;
use pyo3::marker::Ungil;
use pyo3::prelude::*;
use std::collections::VecDeque;
use std::time::Instant;

struct GILContentionRecord(u64);

struct GILContentionCollector {
    max_records: usize,
    records: VecDeque<GILContentionRecord>,
    intervals: Vec<u64>,
}

thread_local! {
    static GIL_CONTENTION_STATS: std::cell::RefCell<GILContentionCollector> = std::cell::RefCell::new(
        GILContentionCollector {
            max_records: 0,
            records: VecDeque::new(),
            intervals: Vec::new()
        });
}

pub(crate) fn add_gil_contention_record(wait_time: u64) {
    GIL_CONTENTION_STATS.with(|stats| {
        let mut stats = stats.borrow_mut();
        if stats.max_records > 0 {
            if stats.records.len() >= stats.max_records {
                stats.records.pop_front();
            }
            stats.records.push_back(GILContentionRecord(wait_time));
        }
    });
}

#[pyfunction]
pub(crate) fn gil_contention_report() -> Vec<(u64, usize)> {
    release_gil(|| {
        GIL_CONTENTION_STATS.with(|stats| {
            let stats = stats.borrow();
            let mut result = Vec::new();
            let mut last_interval = 0;
            for interval in &stats.intervals {
                let number = stats
                    .records
                    .iter()
                    .filter(|r| (last_interval..*interval).contains(&r.0))
                    .count();
                result.push((*interval, number));
                last_interval = *interval;
            }
            result
        })
    })
}

#[pyfunction]
pub(crate) fn configure_thread_gil_contention_collector(
    max_records: usize,
    mut intervals: Vec<u64>,
) {
    release_gil(|| {
        GIL_CONTENTION_STATS.with(|stats| {
            intervals.sort();
            intervals.push(u64::MAX);
            let mut stats = stats.borrow_mut();
            stats.max_records = max_records;
            stats.intervals = intervals;
        })
    });
}

pub(crate) fn is_gil_contention_collector_enabled() -> bool {
    GIL_CONTENTION_STATS.with(|stats| {
        let stats = stats.borrow();
        stats.max_records > 0 && !stats.intervals.is_empty()
    })
}

pub(crate) fn report_gil_wait(start: &Instant, py: Python) {
    let duration = i64::try_from(start.elapsed().as_nanos()).unwrap_or(i64::MAX);
    py.allow_threads(|| {
        add_gil_contention_record(duration as u64);
        with_current_context(|cx| {
            cx.span().add_event(
                "gil-wait".to_string(),
                vec![
                    KeyValue::new("gil-wait.duration", duration),
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
    if is_gil_contention_collector_enabled() {
        let start_wait = Python::with_gil(|py| py.allow_threads(Instant::now));
        Python::with_gil(|py| report_gil_wait(&start_wait, py));
    }
    Python::with_gil(f)
}
