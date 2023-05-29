use crate::primitives::message::video::object::query::{
    FloatExpression, IntExpression, StringExpression,
};

pub trait EqOps<T: Clone, R> {
    fn eq(v: T) -> R;
    fn ne(v: T) -> R;
    fn one_of(v: &[T]) -> R;
}

impl EqOps<f64, FloatExpression> for FloatExpression {
    fn eq(v: f64) -> FloatExpression {
        FloatExpression::EQ(v)
    }

    fn ne(v: f64) -> FloatExpression {
        FloatExpression::NE(v)
    }

    fn one_of(v: &[f64]) -> FloatExpression {
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

impl NumberOps<f64, FloatExpression> for FloatExpression {
    fn gt(v: f64) -> FloatExpression {
        FloatExpression::GT(v)
    }

    fn ge(v: f64) -> FloatExpression {
        FloatExpression::GE(v)
    }

    fn lt(v: f64) -> FloatExpression {
        FloatExpression::LT(v)
    }

    fn le(v: f64) -> FloatExpression {
        FloatExpression::LE(v)
    }

    fn between(a: f64, b: f64) -> FloatExpression {
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
