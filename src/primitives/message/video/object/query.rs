pub mod functions;
pub mod macros;
pub mod py;

use crate::primitives::message::video::object::InnerObject;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::primitives::to_json_value::ToSerdeJsonValue;
pub use crate::query_and as and;
pub use crate::query_not as not;
pub use crate::query_or as or;
pub use functions::*;

pub trait ExecutableQuery<T> {
    fn execute(&self, o: T) -> bool;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "float")]
pub enum FloatExpression {
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

impl ExecutableQuery<&f64> for FloatExpression {
    fn execute(&self, o: &f64) -> bool {
        match self {
            FloatExpression::EQ(x) => x == o,
            FloatExpression::NE(x) => x != o,
            FloatExpression::LT(x) => x < o,
            FloatExpression::LE(x) => x <= o,
            FloatExpression::GT(x) => x > o,
            FloatExpression::GE(x) => x >= o,
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

impl ExecutableQuery<&i64> for IntExpression {
    fn execute(&self, o: &i64) -> bool {
        match self {
            IntExpression::EQ(x) => x == o,
            IntExpression::NE(x) => x != o,
            IntExpression::LT(x) => x < o,
            IntExpression::LE(x) => x <= o,
            IntExpression::GT(x) => x > o,
            IntExpression::GE(x) => x >= o,
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

impl ExecutableQuery<&String> for StringExpression {
    fn execute(&self, o: &String) -> bool {
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "query")]
pub enum Query {
    #[serde(rename = "object.id")]
    Id(IntExpression),
    #[serde(rename = "creator")]
    Creator(StringExpression),
    #[serde(rename = "label")]
    Label(StringExpression),
    #[serde(rename = "confidence")]
    Confidence(FloatExpression),
    #[serde(rename = "confidence.defined")]
    ConfidenceDefined,
    #[serde(rename = "track_id")]
    TrackId(IntExpression),
    #[serde(rename = "track_id.defined")]
    TrackIdDefined,
    // parent
    #[serde(rename = "parent.id")]
    ParentId(IntExpression),
    #[serde(rename = "parent.creator")]
    ParentCreator(StringExpression),
    #[serde(rename = "parent.label")]
    ParentLabel(StringExpression),
    #[serde(rename = "parent.defined")]
    ParentDefined,
    // bbox
    #[serde(rename = "bbox.xc")]
    BoxXCenter(FloatExpression),
    #[serde(rename = "bbox.yc")]
    BoxYCenter(FloatExpression),
    #[serde(rename = "bbox.width")]
    BoxWidth(FloatExpression),
    #[serde(rename = "bbox.height")]
    BoxHeight(FloatExpression),
    #[serde(rename = "bbox.area")]
    BoxArea(FloatExpression),
    #[serde(rename = "bbox.angle")]
    BoxAngle(FloatExpression),
    #[serde(rename = "bbox.angle.defined")]
    BoxAngleDefined,
    // Attributes
    #[serde(rename = "attributes.jmes_query")]
    AttributesJMESQuery(String),
    // combinators
    #[serde(rename = "and")]
    And(Vec<Query>),
    #[serde(rename = "or")]
    Or(Vec<Query>),
    #[serde(rename = "not")]
    Not(Box<Query>),
    #[serde(rename = "pass")]
    Idle,
}

lazy_static! {
    static ref COMPILED_JMP_FILTER: Mutex<HashMap<String, Arc<jmespath::Expression<'static>>>> =
        Mutex::new(HashMap::new());
}

fn get_compiled_jmp_filter(query: &str) -> anyhow::Result<Arc<jmespath::Expression>> {
    let mut compiled_jmp_filter = COMPILED_JMP_FILTER.lock().unwrap();
    if let Some(c) = compiled_jmp_filter.get(query) {
        return Ok(c.clone());
    }
    let c = Arc::new(jmespath::compile(query)?);
    compiled_jmp_filter.insert(query.to_string(), c.clone());
    Ok(c)
}

impl ExecutableQuery<&InnerObject> for Query {
    fn execute(&self, o: &InnerObject) -> bool {
        match self {
            // self
            Query::Id(x) => x.execute(&o.id),
            Query::Creator(x) => x.execute(&o.creator),
            Query::Label(x) => x.execute(&o.label),
            Query::Confidence(x) => o.confidence.map(|c| x.execute(&c)).unwrap_or(false),
            Query::TrackId(x) => o.track_id.map(|t| x.execute(&t)).unwrap_or(false),
            // parent
            Query::ParentId(x) => o.parent.as_ref().map(|p| x.execute(&p.id)).unwrap_or(false),
            Query::ParentCreator(x) => o
                .parent
                .as_ref()
                .map(|p| x.execute(&p.creator))
                .unwrap_or(false),
            Query::ParentLabel(x) => o
                .parent
                .as_ref()
                .map(|p| x.execute(&p.label))
                .unwrap_or(false),
            // boxes
            Query::BoxWidth(x) => x.execute(&o.bbox.width),
            Query::BoxHeight(x) => x.execute(&o.bbox.height),
            Query::BoxArea(x) => x.execute(&(o.bbox.width * o.bbox.height)),
            Query::BoxXCenter(x) => x.execute(&o.bbox.xc),
            Query::BoxYCenter(x) => x.execute(&o.bbox.yc),
            Query::BoxAngle(x) => o.bbox.angle.map(|a| x.execute(&a)).unwrap_or(false),
            Query::And(v) => v.iter().all(|x| x.execute(o)),
            Query::Or(v) => v.iter().any(|x| x.execute(o)),
            Query::Not(x) => !x.execute(o),
            Query::ConfidenceDefined => o.confidence.is_some(),
            Query::TrackIdDefined => o.track_id.is_some(),
            Query::ParentDefined => o.parent.is_some(),
            Query::BoxAngleDefined => o.bbox.angle.is_some(),
            Query::AttributesJMESQuery(x) => {
                let filter = get_compiled_jmp_filter(x).unwrap();
                let json = serde_json::to_string(&serde_json::json!(o
                    .attributes
                    .values()
                    .map(|v| v.to_serde_json_value())
                    .collect::<Vec<_>>()))
                .unwrap();
                let jmp_var = jmespath::Variable::from_json(&json).unwrap();
                let res = filter.search(jmp_var).unwrap();
                !(res.is_null()
                    || (res.is_array() && res.as_array().unwrap().is_empty())
                    || (res.is_boolean() && !res.as_boolean().unwrap()))
            }
            Query::Idle => true,
        }
    }
}

impl Query {
    pub fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(self).unwrap()
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }

    pub fn to_yaml(&self) -> String {
        serde_yaml::to_string(&serde_json::to_value(self).unwrap()).unwrap()
    }

    pub fn from_json(json: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(json)?)
    }

    pub fn from_yaml(yaml: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_value(serde_yaml::from_str(yaml)?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::{Query, Query::*};
    use crate::primitives::message::video::object::query::functions::{eq, gt, one_of};
    use crate::query_and;
    use crate::test::utils::gen_frame;

    #[test]
    fn query() {
        let expr = query_and![
            Id(eq(1)),
            Creator(one_of(&["test", "test2"])),
            Confidence(gt(0.5))
        ];

        let f = gen_frame();
        let _objs = f.access_objects(&expr);
        let json = serde_json::to_string(&expr).unwrap();
        let _q: super::Query = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn jmespath() {
        use jmespath;

        let expr = jmespath::compile("foo.bar").unwrap();

        // Parse some JSON data into a JMESPath variable
        let json_str = r#"{"foo": {"bar": true}}"#;
        let data = jmespath::Variable::from_json(json_str).unwrap();

        // Search the data with the compiled expression
        let result = expr.search(data).unwrap();
        dbg!(&result);
        assert!(!result.is_null());
    }
}
