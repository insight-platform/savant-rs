use crate::primitives::message::video::object::InnerObject;
use serde::{Deserialize, Serialize};

pub trait ExecutableQuery<T> {
    fn execute(&self, o: T) -> bool;
}

pub trait EqOps<T: Clone, R> {
    fn eq(v: T) -> R;
    fn ne(v: T) -> R;
    fn one_of(v: &[T]) -> R;
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

pub trait NumberOps<T, R> {
    fn gt(v: T) -> R;
    fn ge(v: T) -> R;
    fn lt(v: T) -> R;
    fn le(v: T) -> R;
    fn between(a: T, b: T) -> R;
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "float")]
pub enum Float {
    #[serde(rename = "eq")]
    EQ(f64),
    #[serde(rename = "ne")]
    NE(f64),
    #[serde(rename = "lt")]
    LT(f64),
    #[serde(rename = "le")]
    LE(f64),
    #[serde(rename = "gt")]
    GT(f64),
    #[serde(rename = "ge")]
    GE(f64),
    #[serde(rename = "between")]
    Between(f64, f64),
    #[serde(rename = "one_of")]
    OneOf(Vec<f64>),
}

impl NumberOps<f64, Float> for Float {
    fn gt(v: f64) -> Float {
        Float::GT(v)
    }

    fn ge(v: f64) -> Float {
        Float::GE(v)
    }

    fn lt(v: f64) -> Float {
        Float::LT(v)
    }

    fn le(v: f64) -> Float {
        Float::LE(v)
    }

    fn between(a: f64, b: f64) -> Float {
        Float::Between(a, b)
    }
}

impl EqOps<f64, Float> for Float {
    fn eq(v: f64) -> Float {
        Float::EQ(v)
    }

    fn ne(v: f64) -> Float {
        Float::NE(v)
    }

    fn one_of(v: &[f64]) -> Float {
        Float::OneOf(v.to_vec())
    }
}

impl ExecutableQuery<&f64> for Float {
    fn execute(&self, o: &f64) -> bool {
        match self {
            Float::EQ(x) => x == o,
            Float::NE(x) => x != o,
            Float::LT(x) => x < o,
            Float::LE(x) => x <= o,
            Float::GT(x) => x > o,
            Float::GE(x) => x >= o,
            Float::Between(a, b) => a <= o && o <= b,
            Float::OneOf(v) => v.contains(o),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "int")]
pub enum Int {
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

impl NumberOps<i64, Int> for Int {
    fn gt(v: i64) -> Int {
        Int::GT(v)
    }

    fn ge(v: i64) -> Int {
        Int::GE(v)
    }

    fn lt(v: i64) -> Int {
        Int::LT(v)
    }

    fn le(v: i64) -> Int {
        Int::LE(v)
    }

    fn between(a: i64, b: i64) -> Int {
        Int::Between(a, b)
    }
}

impl EqOps<i64, Int> for Int {
    fn eq(v: i64) -> Int {
        Int::EQ(v)
    }

    fn ne(v: i64) -> Int {
        Int::NE(v)
    }

    fn one_of(v: &[i64]) -> Int {
        Int::OneOf(v.to_vec())
    }
}

impl ExecutableQuery<&i64> for Int {
    fn execute(&self, o: &i64) -> bool {
        match self {
            Int::EQ(x) => x == o,
            Int::NE(x) => x != o,
            Int::LT(x) => x < o,
            Int::LE(x) => x <= o,
            Int::GT(x) => x > o,
            Int::GE(x) => x >= o,
            Int::Between(a, b) => a <= o && o <= b,
            Int::OneOf(v) => v.contains(o),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "str")]
pub enum Str {
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

impl ExecutableQuery<&String> for Str {
    fn execute(&self, o: &String) -> bool {
        match self {
            Str::EQ(x) => x == o,
            Str::NE(x) => x != o,
            Str::Contains(x) => o.contains(x),
            Str::NotContains(x) => !o.contains(x),
            Str::StartsWith(x) => o.starts_with(x),
            Str::EndsWith(x) => o.ends_with(x),
            Str::OneOf(v) => v.contains(o),
        }
    }
}

impl EqOps<&str, Str> for Str {
    fn eq(v: &str) -> Str {
        Str::EQ(v.to_string())
    }

    fn ne(v: &str) -> Str {
        Str::NE(v.to_string())
    }

    fn one_of(v: &[&str]) -> Str {
        Str::OneOf(v.iter().map(|x| x.to_string()).collect())
    }
}

impl EqOps<String, Str> for Str {
    fn eq(v: String) -> Str {
        Str::EQ(v)
    }

    fn ne(v: String) -> Str {
        Str::NE(v)
    }

    fn one_of(v: &[String]) -> Str {
        Str::OneOf(v.to_vec())
    }
}

pub fn contains<T>(v: T) -> Str
where
    T: Into<String>,
{
    Str::Contains(v.into())
}

pub fn not_contains<T>(v: T) -> Str
where
    T: Into<String>,
{
    Str::NotContains(v.into())
}

pub fn starts_with<T>(v: T) -> Str
where
    T: Into<String>,
{
    Str::StartsWith(v.into())
}

pub fn ends_with<T>(v: T) -> Str
where
    T: Into<String>,
{
    Str::EndsWith(v.into())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "query")]
pub enum Query {
    #[serde(rename = "object.id")]
    Id(Int),
    #[serde(rename = "creator")]
    Creator(Str),
    #[serde(rename = "label")]
    Label(Str),
    #[serde(rename = "confidence")]
    Confidence(Float),
    #[serde(rename = "track_id")]
    TrackId(Int),
    // parent
    #[serde(rename = "parent.id")]
    ParentId(Int),
    #[serde(rename = "parent.creator")]
    ParentCreator(Str),
    #[serde(rename = "parent.label")]
    ParentLabel(Str),
    // bbox
    #[serde(rename = "bbox.xc")]
    BoxXCenter(Float),
    #[serde(rename = "bbox.yc")]
    BoxYCenter(Float),
    #[serde(rename = "bbox.width")]
    BoxWidth(Float),
    #[serde(rename = "bbox.height")]
    BoxHeight(Float),
    #[serde(rename = "bbox.area")]
    BoxArea(Float),
    #[serde(rename = "bbox.angle")]
    BoxAngle(Float),
    #[serde(rename = "and")]
    And(Vec<Query>),
    #[serde(rename = "or")]
    Or(Vec<Query>),
    #[serde(rename = "not")]
    Not(Box<Query>),
    #[serde(rename = "pass")]
    Pass,
}

impl ExecutableQuery<&InnerObject> for Query {
    fn execute(&self, o: &InnerObject) -> bool {
        match self {
            // self
            Query::Id(x) => x.execute(&o.id),
            Query::Creator(x) => x.execute(&o.creator),
            Query::Label(x) => x.execute(&o.label),
            Query::Confidence(x) => o.confidence.is_some() && x.execute(&o.confidence.unwrap()),
            Query::TrackId(x) => o.track_id.is_some() && x.execute(&o.track_id.unwrap()),
            // parent
            Query::ParentId(x) => {
                o.parent.is_some() && x.execute(&o.parent.as_ref().map(|x| x.id).unwrap())
            }
            Query::ParentCreator(x) => {
                o.parent.is_some() && x.execute(&o.parent.as_ref().map(|x| &x.creator).unwrap())
            }
            Query::ParentLabel(x) => {
                o.parent.is_some() && x.execute(&o.parent.as_ref().map(|x| &x.label).unwrap())
            }
            // boxes
            Query::BoxWidth(x) => x.execute(&o.bbox.width),
            Query::BoxHeight(x) => x.execute(&o.bbox.height),
            Query::BoxArea(x) => x.execute(&(o.bbox.width * o.bbox.height)),
            Query::BoxXCenter(x) => x.execute(&o.bbox.xc),
            Query::BoxYCenter(x) => x.execute(&o.bbox.yc),
            Query::BoxAngle(x) => o.bbox.angle.is_some() && x.execute(&o.bbox.angle.unwrap()),
            Query::And(v) => v.iter().all(|x| x.execute(o)),
            Query::Or(v) => v.iter().any(|x| x.execute(o)),
            Query::Not(x) => !x.execute(o),
            Query::Pass => true,
        }
    }
}

#[macro_export]
macro_rules! not {
    ($arg:expr) => {{
        Query::Not(Box::new($arg))
    }};
}

#[macro_export]
macro_rules! or {
    ($($args:expr),* $(,)?) => {{
        let mut v: Vec<Query> = Vec::new();
        $(
            v.push($args);
        )*
        Query::Or(v)
    }}
}

#[macro_export]
macro_rules! and {
    ($($args:expr),* $(,)?) => {{
        let mut v: Vec<Query> = Vec::new();
        $(
            v.push($args);
        )*
        Query::And(v)
    }}
}

#[cfg(test)]
mod tests {
    use super::{ExecutableQuery, Query, Query::*};
    use crate::primitives::message::video::object::query::{eq, gt, one_of};
    use crate::test::utils::gen_frame;

    #[test]
    fn query() {
        let expr = and![
            Id(eq(1)),
            Creator(one_of(&["test", "test2"])),
            Confidence(gt(0.5))
        ];

        let f = gen_frame();
        let objs = f.access_objects(false, None, None);
        let _res = objs
            .iter()
            .map(|o| expr.execute(&o.inner.lock().unwrap()))
            .collect::<Vec<_>>();
        let json = serde_json::to_string(&expr).unwrap();
        print!("{}", &json);
        let _q: super::Query = serde_json::from_str(&json).unwrap();
    }
}
