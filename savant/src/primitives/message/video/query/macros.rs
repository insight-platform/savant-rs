#[macro_export]
macro_rules! query_not {
    ($arg:expr) => {{
        Query::Not(Box::new($arg))
    }};
}

#[macro_export]
macro_rules! query_or {
    ($($x:expr),+ $(,)?) => ( Query::Or(vec![$($x),+]) );
}

#[macro_export]
macro_rules! query_and {
    ($($x:expr),+ $(,)?) => ( Query::And(vec![$($x),+]) );
}
