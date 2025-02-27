#[macro_export]
macro_rules! function {
    () => {{
        fn f() {}
        fn type_name_of<T>(_: T) -> &'static str {
            std::any::type_name::<T>()
        }
        let name = type_name_of(f);

        // Find and cut the rest of the path
        match &name[..name.len() - 3].rfind(':') {
            Some(pos) => &name[pos + 1..name.len() - 3],
            None => &name[..name.len() - 3],
        }
    }};
}

#[macro_export]
macro_rules! with_gil {
    ($expression:expr) => {{
        let start = std::time::Instant::now();
        let thread_id = std::thread::current().id();
        log::trace!(
            target: "savant::trace::before::gil_acquire",
                "[{:?}] Trace line ({}, {}, {})",
                thread_id,
                $crate::function!(),
                file!(),
                line!()
        );
        let res = pyo3::marker::Python::with_gil($expression);
        log::trace!(
            target: "savant::trace::after::gil_acquire",
                "[{:?}] Trace line ({}, {}, {})",
                thread_id,
                $crate::function!(),
                file!(),
                line!()
        );
        let elapsed = start.elapsed();
        $crate::logging::log_message(
            $crate::logging::LogLevel::Trace,
            "savant::gil_management::with_gil",
            format!(
                "Holding GIL ({}, {}, {})",
                $crate::function!(),
                file!(),
                line!()
            ).as_str(),
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
            "savant::trace",
            format!(
                "Tracing ({}, {}, {})",
                $crate::function!(),
                file!(),
                line!()
            )
            .as_str(),
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
            let thread_id = std::thread::current().id();
            log::trace!(
                target: "savant::trace::before::gil_release",
                    "[{:?}] Trace line ({}, {}, {})",
                    thread_id,
                    $crate::function!(),
                    file!(),
                    line!()
            );
            let (res, elapsed_nogil, elapsed_gil_back) = pyo3::marker::Python::with_gil(|py| {
                log::trace!(
                    target: "savant::trace::after::gil_release",
                        "[{:?}] Trace line ({}, {}, {})",
                        thread_id,
                        $crate::function!(),
                        file!(),
                        line!()
                );
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
                "savant::gil_management::with_released_gil",
                format!(
                    "{} GIL-free operation ({}, {}, {})",
                    if gf > 10000 { "👌" } else { "💀" },
                    $crate::function!(),
                    file!(),
                    line!()
                ).as_str(),
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

#[macro_export]
macro_rules! err_to_pyo3 {
    ($expr:expr, $py_err:ty) => {
        $expr.map_err(|e| <$py_err>::new_err(e.to_string()))
    };
}
