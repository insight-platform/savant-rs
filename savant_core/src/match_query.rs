use serde::{Deserialize, Serialize};

pub use crate::query_and as and;
pub use crate::query_not as not;
pub use crate::query_or as or;

pub trait ExecutableMatchQuery<T, C> {
    fn execute(&self, o: T, ctx: &mut C) -> bool;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "float")]
pub enum FloatExpression {
    #[serde(rename = "eq")]
    EQ(f32),
    #[serde(rename = "ne")]
    NE(f32),
    #[serde(rename = "lt")]
    LT(f32),
    #[serde(rename = "le")]
    LE(f32),
    #[serde(rename = "gt")]
    GT(f32),
    #[serde(rename = "ge")]
    GE(f32),
    #[serde(rename = "between")]
    Between(f32, f32),
    #[serde(rename = "one_of")]
    OneOf(Vec<f32>),
}

impl ExecutableMatchQuery<&f32, ()> for FloatExpression {
    fn execute(&self, o: &f32, _: &mut ()) -> bool {
        match self {
            FloatExpression::EQ(x) => x == o,
            FloatExpression::NE(x) => x != o,
            FloatExpression::LT(x) => x > o,
            FloatExpression::LE(x) => x >= o,
            FloatExpression::GT(x) => x < o,
            FloatExpression::GE(x) => x <= o,
            FloatExpression::Between(a, b) => a <= o && o <= b,
            FloatExpression::OneOf(v) => v.contains(o),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "int")]
pub enum IntExpression {
    #[serde(rename = "eq")]
    EQ(i64),
    #[serde(rename = "ne")]
    NE(i64),
    #[serde(rename = "lt")]
    LT(i64),
    #[serde(rename = "le")]
    LE(i64),
    #[serde(rename = "gt")]
    GT(i64),
    #[serde(rename = "ge")]
    GE(i64),
    #[serde(rename = "between")]
    Between(i64, i64),
    #[serde(rename = "one_of")]
    OneOf(Vec<i64>),
}

impl ExecutableMatchQuery<&i64, ()> for IntExpression {
    fn execute(&self, o: &i64, _: &mut ()) -> bool {
        match self {
            IntExpression::EQ(x) => x == o,
            IntExpression::NE(x) => x != o,
            IntExpression::LT(x) => x > o,
            IntExpression::LE(x) => x >= o,
            IntExpression::GT(x) => x < o,
            IntExpression::GE(x) => x <= o,
            IntExpression::Between(a, b) => a <= o && o <= b,
            IntExpression::OneOf(v) => v.contains(o),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "str")]
pub enum StringExpression {
    #[serde(rename = "eq")]
    EQ(String),
    #[serde(rename = "ne")]
    NE(String),
    #[serde(rename = "contains")]
    Contains(String),
    #[serde(rename = "not_contains")]
    NotContains(String),
    #[serde(rename = "starts_with")]
    StartsWith(String),
    #[serde(rename = "ends_with")]
    EndsWith(String),
    #[serde(rename = "one_of")]
    OneOf(Vec<String>),
}

impl ExecutableMatchQuery<&String, ()> for StringExpression {
    fn execute(&self, o: &String, _: &mut ()) -> bool {
        match self {
            StringExpression::EQ(x) => x == o,
            StringExpression::NE(x) => x != o,
            StringExpression::Contains(x) => o.contains(x),
            StringExpression::NotContains(x) => !o.contains(x),
            StringExpression::StartsWith(x) => o.starts_with(x),
            StringExpression::EndsWith(x) => o.ends_with(x),
            StringExpression::OneOf(v) => v.contains(o),
        }
    }
}

pub trait EqOps<T: Clone, R> {
    fn eq(v: T) -> R;
    fn ne(v: T) -> R;
    fn one_of(v: &[T]) -> R;
}

impl EqOps<f32, FloatExpression> for FloatExpression {
    fn eq(v: f32) -> FloatExpression {
        FloatExpression::EQ(v)
    }

    fn ne(v: f32) -> FloatExpression {
        FloatExpression::NE(v)
    }

    fn one_of(v: &[f32]) -> FloatExpression {
        FloatExpression::OneOf(v.to_vec())
    }
}

impl EqOps<i64, IntExpression> for IntExpression {
    fn eq(v: i64) -> IntExpression {
        IntExpression::EQ(v)
    }

    fn ne(v: i64) -> IntExpression {
        IntExpression::NE(v)
    }

    fn one_of(v: &[i64]) -> IntExpression {
        IntExpression::OneOf(v.to_vec())
    }
}

impl EqOps<&str, StringExpression> for StringExpression {
    fn eq(v: &str) -> StringExpression {
        StringExpression::EQ(v.to_string())
    }

    fn ne(v: &str) -> StringExpression {
        StringExpression::NE(v.to_string())
    }

    fn one_of(v: &[&str]) -> StringExpression {
        StringExpression::OneOf(v.iter().map(|x| x.to_string()).collect())
    }
}

impl EqOps<String, StringExpression> for StringExpression {
    fn eq(v: String) -> StringExpression {
        StringExpression::EQ(v)
    }

    fn ne(v: String) -> StringExpression {
        StringExpression::NE(v)
    }

    fn one_of(v: &[String]) -> StringExpression {
        StringExpression::OneOf(v.to_vec())
    }
}

pub trait NumberOps<T, R> {
    fn gt(v: T) -> R;
    fn ge(v: T) -> R;
    fn lt(v: T) -> R;
    fn le(v: T) -> R;
    fn between(a: T, b: T) -> R;
}

impl NumberOps<f32, FloatExpression> for FloatExpression {
    fn gt(v: f32) -> FloatExpression {
        FloatExpression::GT(v)
    }

    fn ge(v: f32) -> FloatExpression {
        FloatExpression::GE(v)
    }

    fn lt(v: f32) -> FloatExpression {
        FloatExpression::LT(v)
    }

    fn le(v: f32) -> FloatExpression {
        FloatExpression::LE(v)
    }

    fn between(a: f32, b: f32) -> FloatExpression {
        FloatExpression::Between(a, b)
    }
}

impl NumberOps<i64, IntExpression> for IntExpression {
    fn gt(v: i64) -> IntExpression {
        IntExpression::GT(v)
    }

    fn ge(v: i64) -> IntExpression {
        IntExpression::GE(v)
    }

    fn lt(v: i64) -> IntExpression {
        IntExpression::LT(v)
    }

    fn le(v: i64) -> IntExpression {
        IntExpression::LE(v)
    }

    fn between(a: i64, b: i64) -> IntExpression {
        IntExpression::Between(a, b)
    }
}

pub fn eq<T: Clone, F>(v: T) -> F
where
    F: EqOps<T, F>,
{
    F::eq(v)
}

pub fn ne<T: Clone, F>(v: T) -> F
where
    F: EqOps<T, F>,
{
    F::ne(v)
}

pub fn one_of<T: Clone, F>(v: &[T]) -> F
where
    F: EqOps<T, F>,
{
    F::one_of(v)
}

pub fn gt<T, F>(v: T) -> F
where
    F: NumberOps<T, F>,
{
    F::gt(v)
}

pub fn ge<T, F>(v: T) -> F
where
    F: NumberOps<T, F>,
{
    F::ge(v)
}

pub fn lt<T, F>(v: T) -> F
where
    F: NumberOps<T, F>,
{
    F::lt(v)
}

pub fn le<T, F>(v: T) -> F
where
    F: NumberOps<T, F>,
{
    F::le(v)
}

pub fn between<T, F>(a: T, b: T) -> F
where
    F: NumberOps<T, F>,
{
    F::between(a, b)
}

pub fn contains<T>(v: T) -> StringExpression
where
    T: Into<String>,
{
    StringExpression::Contains(v.into())
}

pub fn not_contains<T>(v: T) -> StringExpression
where
    T: Into<String>,
{
    StringExpression::NotContains(v.into())
}

pub fn starts_with<T>(v: T) -> StringExpression
where
    T: Into<String>,
{
    StringExpression::StartsWith(v.into())
}

pub fn ends_with<T>(v: T) -> StringExpression
where
    T: Into<String>,
{
    StringExpression::EndsWith(v.into())
}

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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_int() {
        use IntExpression as IE;
        let eq_q: IE = eq(1);
        assert!(eq_q.execute(&1, &mut ()));

        let ne_q: IE = ne(1);
        assert!(ne_q.execute(&2, &mut ()));

        let gt_q: IE = gt(1);
        assert!(gt_q.execute(&2, &mut ()));

        let lt_q: IE = lt(1);
        assert!(lt_q.execute(&0, &mut ()));

        let ge_q: IE = ge(1);
        assert!(ge_q.execute(&1, &mut ()));
        assert!(ge_q.execute(&2, &mut ()));

        let le_q: IE = le(1);
        assert!(le_q.execute(&1, &mut ()));
        assert!(le_q.execute(&0, &mut ()));

        let between_q: IE = between(1, 5);
        assert!(between_q.execute(&2, &mut ()));
        assert!(between_q.execute(&1, &mut ()));
        assert!(between_q.execute(&5, &mut ()));
        assert!(!between_q.execute(&6, &mut ()));

        let one_of_q: IE = one_of(&[1, 2, 3]);
        assert!(one_of_q.execute(&2, &mut ()));
        assert!(!one_of_q.execute(&4, &mut ()));
    }

    #[test]
    fn test_float() {
        use FloatExpression as FE;
        let eq_q: FE = eq(1.0);
        assert!(eq_q.execute(&1.0, &mut ()));

        let ne_q: FE = ne(1.0);
        assert!(ne_q.execute(&2.0, &mut ()));

        let gt_q: FE = gt(1.0);
        assert!(gt_q.execute(&2.0, &mut ()));

        let lt_q: FE = lt(1.0);
        assert!(lt_q.execute(&0.0, &mut ()));

        let ge_q: FE = ge(1.0);
        assert!(ge_q.execute(&1.0, &mut ()));
        assert!(ge_q.execute(&2.0, &mut ()));

        let le_q: FE = le(1.0);
        assert!(le_q.execute(&1.0, &mut ()));
        assert!(le_q.execute(&0.0, &mut ()));

        let between_q: FE = between(1.0, 5.0);
        assert!(between_q.execute(&2.0, &mut ()));
        assert!(between_q.execute(&1.0, &mut ()));
        assert!(between_q.execute(&5.0, &mut ()));
        assert!(!between_q.execute(&6.0, &mut ()));

        let one_of_q: FE = one_of(&[1.0, 2.0, 3.0]);
        assert!(one_of_q.execute(&2.0, &mut ()));
        assert!(!one_of_q.execute(&4.0, &mut ()));
    }

    #[test]
    fn test_string() {
        use StringExpression as SE;
        let eq_q: SE = eq("test");
        assert!(eq_q.execute(&"test".to_string(), &mut ()));

        let ne_q: SE = ne("test");
        assert!(ne_q.execute(&"test2".to_string(), &mut ()));

        let contains_q: SE = contains("test");
        assert!(contains_q.execute(&"testimony".to_string(), &mut ()));
        assert!(contains_q.execute(&"supertest".to_string(), &mut ()));

        let not_contains_q: SE = not_contains("test");
        assert!(not_contains_q.execute(&"apple".to_string(), &mut ()));

        let starts_with_q: SE = starts_with("test");
        assert!(starts_with_q.execute(&"testing".to_string(), &mut ()));
        assert!(!starts_with_q.execute(&"tes".to_string(), &mut ()));

        let ends_with_q: SE = ends_with("test");
        assert!(ends_with_q.execute(&"gettest".to_string(), &mut ()));
        assert!(!ends_with_q.execute(&"supertes".to_string(), &mut ()));

        let one_of_q: SE = one_of(&["test", "me", "now"]);
        assert!(one_of_q.execute(&"me".to_string(), &mut ()));
        assert!(one_of_q.execute(&"now".to_string(), &mut ()));
        assert!(!one_of_q.execute(&"random".to_string(), &mut ()));
    }
}
