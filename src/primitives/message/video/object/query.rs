pub mod functions;
pub mod macros;
pub mod py;

use lazy_static::lazy_static;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::primitives::to_json_value::ToSerdeJsonValue;
use crate::primitives::Object;
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

impl ExecutableQuery<&i64> for IntExpression {
    fn execute(&self, o: &i64) -> bool {
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
    // children query
    #[serde(rename = "with_children")]
    WithChildren(Box<Query>, IntExpression),
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
    #[serde(rename = "attributes.empty")]
    AttributesEmpty,
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

const MAX_JMES_CACHE_SIZE: usize = 1024;

lazy_static! {
    static ref COMPILED_JMP_FILTER: Mutex<lru::LruCache<String, Arc<jmespath::Expression<'static>>>> =
        Mutex::new(lru::LruCache::new(
            std::num::NonZeroUsize::new(MAX_JMES_CACHE_SIZE).unwrap()
        ));
}

fn get_compiled_jmp_filter(query: &str) -> anyhow::Result<Arc<jmespath::Expression>> {
    let mut compiled_jmp_filter = COMPILED_JMP_FILTER.lock();
    if let Some(c) = compiled_jmp_filter.get(query) {
        return Ok(c.clone());
    }
    let c = Arc::new(jmespath::compile(query)?);
    compiled_jmp_filter.put(query.to_string(), c.clone());
    Ok(c)
}

impl ExecutableQuery<&Object> for Query {
    fn execute(&self, o: &Object) -> bool {
        match self {
            Query::And(v) => v.iter().all(|x| x.execute(o)),
            Query::Or(v) => v.iter().any(|x| x.execute(o)),
            Query::Not(x) => !x.execute(o),
            Query::WithChildren(q, n) => {
                let children = o.get_children();
                let v = filter(&children, q).len() as i64;
                n.execute(&v)
            }
            // self
            Query::Id(x) => x.execute(&o.get_inner_read().id),
            Query::Creator(x) => x.execute(&o.get_inner_read().creator),
            Query::Label(x) => x.execute(&o.get_inner_read().label),
            Query::Confidence(x) => o
                .get_inner_read()
                .confidence
                .map(|c| x.execute(&c))
                .unwrap_or(false),
            Query::TrackId(x) => o
                .get_inner_read()
                .track_id
                .map(|t| x.execute(&t))
                .unwrap_or(false),
            Query::ConfidenceDefined => o.get_inner_read().confidence.is_some(),
            Query::TrackIdDefined => o.get_inner_read().track_id.is_some(),

            // parent
            Query::ParentDefined => o.get_inner_read().parent.is_some(),
            Query::ParentId(x) => o
                .get_inner_read()
                .parent
                .as_ref()
                .map(|p| x.execute(&p.object().get_id()))
                .unwrap_or(false),
            Query::ParentCreator(x) => o
                .get_inner_read()
                .parent
                .as_ref()
                .map(|p| x.execute(&p.object().get_creator()))
                .unwrap_or(false),
            Query::ParentLabel(x) => o
                .get_inner_read()
                .parent
                .as_ref()
                .map(|p| x.execute(&p.object().get_label()))
                .unwrap_or(false),
            // box
            Query::BoxWidth(x) => x.execute(&o.get_inner_read().bbox.width),
            Query::BoxHeight(x) => x.execute(&o.get_inner_read().bbox.height),
            Query::BoxArea(x) => x.execute({
                let inner = o.get_inner_read();
                &(inner.bbox.width * inner.bbox.height)
            }),
            Query::BoxXCenter(x) => x.execute(&o.get_inner_read().bbox.xc),
            Query::BoxYCenter(x) => x.execute(&o.get_inner_read().bbox.yc),
            Query::BoxAngleDefined => o.get_inner_read().bbox.angle.is_some(),
            Query::BoxAngle(x) => o
                .get_inner_read()
                .bbox
                .angle
                .map(|a| x.execute(&a))
                .unwrap_or(false),

            // attributes
            Query::AttributesEmpty => o.get_inner_read().attributes.is_empty(),
            Query::AttributesJMESQuery(x) => {
                let filter = get_compiled_jmp_filter(x).unwrap();
                let json = &serde_json::json!(o
                    .get_inner_read()
                    .attributes
                    .values()
                    .map(|v| v.to_serde_json_value())
                    .collect::<Vec<_>>());
                let res = filter.search(json).unwrap();
                !(res.is_null()
                    || (res.is_array() && res.as_array().unwrap().is_empty())
                    || (res.is_boolean() && !res.as_boolean().unwrap())
                    || (res.is_object()) && res.as_object().unwrap().is_empty())
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

pub fn filter(objs: &[Object], query: &Query) -> Vec<Object> {
    objs.iter()
        .filter_map(|o| {
            if query.execute(o) {
                Some(o.clone())
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::Query::*;
    use super::*;
    use crate::primitives::attribute::Attributive;
    use crate::primitives::message::video::object::InnerObject;
    use crate::primitives::{AttributeBuilder, RBBox, Value};
    use crate::query_and;
    use crate::test::utils::{gen_frame, s};

    #[test]
    fn test_int() {
        use IntExpression as IE;
        let eq_q: IE = eq(1);
        assert!(eq_q.execute(&1));

        let ne_q: IE = ne(1);
        assert!(ne_q.execute(&2));

        let gt_q: IE = gt(1);
        assert!(gt_q.execute(&2));

        let lt_q: IE = lt(1);
        assert!(lt_q.execute(&0));

        let ge_q: IE = ge(1);
        assert!(ge_q.execute(&1));
        assert!(ge_q.execute(&2));

        let le_q: IE = le(1);
        assert!(le_q.execute(&1));
        assert!(le_q.execute(&0));

        let between_q: IE = between(1, 5);
        assert!(between_q.execute(&2));
        assert!(between_q.execute(&1));
        assert!(between_q.execute(&5));
        assert!(!between_q.execute(&6));

        let one_of_q: IE = one_of(&[1, 2, 3]);
        assert!(one_of_q.execute(&2));
        assert!(!one_of_q.execute(&4));
    }

    #[test]
    fn test_float() {
        use FloatExpression as FE;
        let eq_q: FE = eq(1.0);
        assert!(eq_q.execute(&1.0));

        let ne_q: FE = ne(1.0);
        assert!(ne_q.execute(&2.0));

        let gt_q: FE = gt(1.0);
        assert!(gt_q.execute(&2.0));

        let lt_q: FE = lt(1.0);
        assert!(lt_q.execute(&0.0));

        let ge_q: FE = ge(1.0);
        assert!(ge_q.execute(&1.0));
        assert!(ge_q.execute(&2.0));

        let le_q: FE = le(1.0);
        assert!(le_q.execute(&1.0));
        assert!(le_q.execute(&0.0));

        let between_q: FE = between(1.0, 5.0);
        assert!(between_q.execute(&2.0));
        assert!(between_q.execute(&1.0));
        assert!(between_q.execute(&5.0));
        assert!(!between_q.execute(&6.0));

        let one_of_q: FE = one_of(&[1.0, 2.0, 3.0]);
        assert!(one_of_q.execute(&2.0));
        assert!(!one_of_q.execute(&4.0));
    }

    #[test]
    fn test_string() {
        use StringExpression as SE;
        let eq_q: SE = eq("test");
        assert!(eq_q.execute(&"test".to_string()));

        let ne_q: SE = ne("test");
        assert!(ne_q.execute(&"test2".to_string()));

        let contains_q: SE = contains("test");
        assert!(contains_q.execute(&"testimony".to_string()));
        assert!(contains_q.execute(&"supertest".to_string()));

        let not_contains_q: SE = not_contains("test");
        assert!(not_contains_q.execute(&"apple".to_string()));

        let starts_with_q: SE = starts_with("test");
        assert!(starts_with_q.execute(&"testing".to_string()));
        assert!(!starts_with_q.execute(&"tes".to_string()));

        let ends_with_q: SE = ends_with("test");
        assert!(ends_with_q.execute(&"gettest".to_string()));
        assert!(!ends_with_q.execute(&"supertes".to_string()));

        let one_of_q: SE = one_of(&["test", "me", "now"]);
        assert!(one_of_q.execute(&"me".to_string()));
        assert!(one_of_q.execute(&"now".to_string()));
        assert!(!one_of_q.execute(&"random".to_string()));
    }

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

    fn gen_object() -> Object {
        Object::from_inner_object(InnerObject {
            id: 1,
            creator: s("peoplenet"),
            label: s("face"),
            confidence: Some(0.5),
            bbox: RBBox::new(1.0, 2.0, 10.0, 20.0, None),
            ..Default::default()
        })
    }

    #[test]
    fn test_query() {
        let expr = Id(eq(1));
        assert!(expr.execute(&gen_object()));

        let expr = Creator(eq("peoplenet"));
        assert!(expr.execute(&gen_object()));

        let expr = Label(starts_with("face"));
        assert!(expr.execute(&gen_object()));

        let expr = Confidence(gt(0.4));
        assert!(expr.execute(&gen_object()));

        let expr = ConfidenceDefined;
        assert!(expr.execute(&gen_object()));

        let expr = ParentDefined;
        assert!(!expr.execute(&gen_object()));

        let expr = AttributesEmpty;
        assert!(expr.execute(&gen_object()));

        print!("{:?}", "test");
        let object = gen_object();
        let parent_object = Object::from_inner_object(InnerObject {
            id: 13,
            creator: s("peoplenet"),
            label: s("person"),
            confidence: None,
            bbox: RBBox::new(1.0, 2.0, 10.0, 20.0, None),
            ..Default::default()
        });
        object.set_parent(Some(parent_object));
        assert!(expr.execute(&object));
        print!("{:?}", "after test");

        let expr = ParentId(eq(13));
        assert!(expr.execute(&object));

        let expr = ParentCreator(eq("peoplenet"));
        assert!(expr.execute(&object));

        let expr = ParentLabel(eq("person"));
        assert!(expr.execute(&object));

        let expr = BoxXCenter(gt(0.0));
        assert!(expr.execute(&gen_object()));

        let expr = BoxYCenter(gt(1.0));
        assert!(expr.execute(&gen_object()));

        let expr = BoxWidth(gt(5.0));
        assert!(expr.execute(&gen_object()));

        let expr = BoxHeight(gt(10.0));
        assert!(expr.execute(&gen_object()));

        let expr = BoxArea(gt(150.0));
        assert!(expr.execute(&gen_object()));

        let expr = BoxArea(lt(250.0));
        assert!(expr.execute(&gen_object()));

        let expr = BoxAngleDefined;
        assert!(!expr.execute(&gen_object()));

        let mut object = gen_object();
        object.set_bbox(RBBox::new(1.0, 2.0, 10.0, 20.0, Some(30.0)));
        assert!(expr.execute(&object));

        let expr = BoxAngle(gt(20.0));
        assert!(expr.execute(&object));

        let expr = TrackIdDefined;
        assert!(!expr.execute(&gen_object()));

        object.set_track_id(Some(1));
        assert!(expr.execute(&object));

        object.set_attribute(
            AttributeBuilder::default()
                .name(s("age-min-max-avg"))
                .creator(s("classifier"))
                .hint(Some(s("morphological-classifier")))
                .values(vec![
                    Value::float(10.0, Some(0.7)),
                    Value::float(20.0, Some(0.8)),
                    Value::float(15.0, None),
                ])
                .build()
                .unwrap(),
        );

        let expr = AttributesJMESQuery(s(
            "[? (hint == 'morphological-classifier') && (creator == 'classifier')]",
        ));
        assert!(expr.execute(&object));

        let expr = AttributesJMESQuery(s(
            "[? (hint != 'morphological-classifier') && (creator == 'classifier')]",
        ));
        assert!(!expr.execute(&object));
    }

    #[test]
    fn test_logical_functions() {
        let expr = and![Id(eq(1)), Creator(eq("peoplenet")), Confidence(gt(0.4))];
        assert!(expr.execute(&gen_object()));

        let expr = or![Id(eq(10)), Creator(eq("peoplenet")),];
        assert!(expr.execute(&gen_object()));

        let expr = not!(Id(eq(2)));
        assert!(expr.execute(&gen_object()));
    }

    #[test]
    fn test_children_expression() {
        let f = gen_frame();
        let o = f.access_objects(&WithChildren(Box::new(Idle), eq(2)));
        assert_eq!(o.len(), 1);
        assert_eq!(o[0].get_id(), 0);
    }
}
