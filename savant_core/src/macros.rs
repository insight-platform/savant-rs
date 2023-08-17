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
macro_rules! trace {
    ($expression:expr) => {{
        let thread_id = std::thread::current().id();
        log::trace!(
            target: "savant::trace::before",
                "[{:?}] Trace line ({}, {}, {})",
                thread_id,
                $crate::function!(),
                file!(),
                line!()
        );
        let result = $expression;
        log::trace!(
            target: "savant::trace::after",
                "[{:?}] Trace line ({}, {}, {})",
                thread_id,
                $crate::function!(),
                file!(),
                line!()
        );
        result
    }};
}
