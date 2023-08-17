#[macro_export]
macro_rules! with_gil {
    ($expression:expr) => {{
        let start = std::time::Instant::now();
        let res = pyo3::marker::Python::with_gil($expression);
        let elapsed = start.elapsed();
        $crate::logging::log_message(
            $crate::logging::LogLevel::Trace,
            "savant::gil_management::with_gil".to_string(),
            format!(
                "Holding GIL ({}, {}, {})",
                savant_core::function!(),
                file!(),
                line!()
            ),
            Some(vec![opentelemetry::KeyValue::new(
                "duration".to_string(),
                format!(
                    "{:?}",
                    i64::try_from(elapsed.as_nanos()).unwrap_or(i64::MAX)
                ),
            )]),
        );
        res
    }};
}

#[macro_export]
macro_rules! with_trace {
    ($expression:expr) => {{
        let start = std::time::Instant::now();
        #[allow(clippy::redundant_closure_call)]
        let res = $expression();
        let elapsed = start.elapsed();
        $crate::logging::log_message(
            $crate::logging::LogLevel::Trace,
            "savant::trace".to_string(),
            format!(
                "Tracing ({}, {}, {})",
                savant_core::function!(),
                file!(),
                line!()
            ),
            Some(vec![opentelemetry::KeyValue::new(
                "duration".to_string(),
                format!(
                    "{:?}",
                    i64::try_from(elapsed.as_nanos()).unwrap_or(i64::MAX)
                ),
            )]),
        );
        res
    }};
}

#[macro_export]
macro_rules! release_gil {
    ($predicate:expr, $expression:expr) => {{
        if $predicate {
            let (res, elapsed_nogil, elapsed_gil_back) = pyo3::marker::Python::with_gil(|py| {
                let (res, elapsed_nogil, start_gil_back) = py.allow_threads(|| {
                    let start_nogil = std::time::Instant::now();
                    #[allow(clippy::redundant_closure_call)]
                    let res = $expression();
                    let elapsed_nogil = start_nogil.elapsed();
                    let start_gil_back = std::time::Instant::now();
                    (res, elapsed_nogil, start_gil_back)
                });
                let elapsed_gil_back = start_gil_back.elapsed();
                (res, elapsed_nogil, elapsed_gil_back)
            });
            let gf = i64::try_from(elapsed_nogil.as_nanos()).unwrap_or(i64::MAX);
            let gw = i64::try_from(elapsed_gil_back.as_nanos()).unwrap_or(i64::MAX);
            $crate::logging::log_message(
                $crate::logging::LogLevel::Trace,
                "savant::gil_management::with_released_gil".to_string(),
                format!(
                    "{} GIL-free operation ({}, {}, {})",
                    if gf > 10000 { "👌" } else { "💀" },
                    savant_core::function!(),
                    file!(),
                    line!()
                ),
                Some(vec![
                    opentelemetry::KeyValue::new(
                        "duration.gil-free".to_string(),
                        format!("{:?}", gf),
                    ),
                    opentelemetry::KeyValue::new(
                        "duration.gil-wait".to_string(),
                        format!("{:?}", gw),
                    ),
                ]),
            );
            res
        } else {
            $crate::with_trace!($expression)
        }
    }};
}
