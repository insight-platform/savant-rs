#[macro_export]
macro_rules! query_not {
    ($arg:expr) => {{
        MatchQuery::Not(Box::new($arg))
    }};
}

#[macro_export]
macro_rules! query_or {
    ($($x:expr),+ $(,)?) => ( MatchQuery::Or(vec![$($x),+]) );
}

#[macro_export]
macro_rules! query_and {
    ($($x:expr),+ $(,)?) => ( MatchQuery::And(vec![$($x),+]) );
}
