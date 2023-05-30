#[macro_export]
macro_rules! query_not {
    ($arg:expr) => {{
        Query::Not(Box::new($arg))
    }};
}

#[macro_export]
macro_rules! query_or {
    ($($args:expr),* $(,)?) => {{
        let mut v: Vec<Query> = Vec::new();
        $(
            v.push($args);
        )*
        Query::Or(v)
    }}
}

#[macro_export]
macro_rules! query_and {
    ($($args:expr),* $(,)?) => {{
        let mut v: Vec<Query> = Vec::new();
        $(
            v.push($args);
        )*
        Query::And(v)
    }}
}
