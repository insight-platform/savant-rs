use crate::primitives::message::video::object::InnerObject;
use crate::primitives::{ParentObject, RBBox};

pub trait ExecutableQuery<T> {
    fn execute(&self, o: T) -> bool;
}

#[derive(Debug, Clone)]
pub enum FloatQ {
    EQ(f64),
    NE(f64),
    LT(f64),
    LE(f64),
    GT(f64),
    GE(f64),
    Between(f64, f64),
    OneOf(Vec<f64>),
    And(Vec<FloatQ>),
    Or(Vec<FloatQ>),
    Not(Box<FloatQ>),
    Pass,
}

impl ExecutableQuery<f64> for FloatQ {
    fn execute(&self, o: f64) -> bool {
        match self {
            FloatQ::EQ(x) => *x == o,
            FloatQ::NE(x) => *x != o,
            FloatQ::LT(x) => *x < o,
            FloatQ::LE(x) => *x <= o,
            FloatQ::GT(x) => *x > o,
            FloatQ::GE(x) => *x >= o,
            FloatQ::Between(a, b) => *a <= o && o <= *b,
            FloatQ::OneOf(v) => v.contains(&o),
            FloatQ::And(v) => v.iter().all(|x| x.execute(o)),
            FloatQ::Or(v) => v.iter().any(|x| x.execute(o)),
            FloatQ::Not(x) => !x.execute(o),
            FloatQ::Pass => true,
        }
    }
}

#[derive(Debug, Clone)]
pub enum IntQ {
    EQ(i64),
    NE(i64),
    LT(i64),
    LE(i64),
    GT(i64),
    GE(i64),
    Between(i64, i64),
    OneOf(Vec<i64>),
    And(Vec<IntQ>),
    Or(Vec<IntQ>),
    Not(Box<IntQ>),
    Pass,
}

impl ExecutableQuery<i64> for IntQ {
    fn execute(&self, o: i64) -> bool {
        match self {
            IntQ::EQ(x) => *x == o,
            IntQ::NE(x) => *x != o,
            IntQ::LT(x) => *x < o,
            IntQ::LE(x) => *x <= o,
            IntQ::GT(x) => *x > o,
            IntQ::GE(x) => *x >= o,
            IntQ::Between(a, b) => *a <= o && o <= *b,
            IntQ::OneOf(v) => v.contains(&o),
            IntQ::And(v) => v.iter().all(|x| x.execute(o)),
            IntQ::Or(v) => v.iter().any(|x| x.execute(o)),
            IntQ::Not(x) => !x.execute(o),
            IntQ::Pass => true,
        }
    }
}

#[derive(Debug, Clone)]
pub enum StringQ {
    EQ(String),
    NE(String),
    Contains(String),
    NotContains(String),
    StartsWith(String),
    EndsWith(String),
    OneOf(Vec<String>),
    And(Vec<StringQ>),
    Or(Vec<StringQ>),
    Not(Box<StringQ>),
    Pass,
}

impl ExecutableQuery<&String> for StringQ {
    fn execute(&self, o: &String) -> bool {
        match self {
            StringQ::EQ(x) => x == o,
            StringQ::NE(x) => x != o,
            StringQ::Contains(x) => o.contains(x),
            StringQ::NotContains(x) => !o.contains(x),
            StringQ::StartsWith(x) => o.starts_with(x),
            StringQ::EndsWith(x) => o.ends_with(x),
            StringQ::OneOf(v) => v.contains(o),
            StringQ::And(v) => v.iter().all(|x| x.execute(o)),
            StringQ::Or(v) => v.iter().any(|x| x.execute(o)),
            StringQ::Not(x) => !x.execute(o),
            StringQ::Pass => true,
        }
    }
}

#[derive(Debug, Clone)]
pub enum OptFloatQ {
    Defined,
    NotDefined,
    IfDefinedThen(FloatQ),
    DefinedAnd(FloatQ),
    And(Vec<OptFloatQ>),
    Or(Vec<OptFloatQ>),
    Not(Box<OptFloatQ>),
    Pass,
}

impl ExecutableQuery<Option<f64>> for OptFloatQ {
    fn execute(&self, o: Option<f64>) -> bool {
        match self {
            OptFloatQ::Defined => o.is_some(),
            OptFloatQ::NotDefined => o.is_none(),
            OptFloatQ::IfDefinedThen(x) => match o {
                Some(o) => x.execute(o),
                None => true,
            },
            OptFloatQ::DefinedAnd(x) => match o {
                Some(o) => x.execute(o),
                None => false,
            },
            OptFloatQ::And(v) => v.iter().all(|x| x.execute(o)),
            OptFloatQ::Or(v) => v.iter().any(|x| x.execute(o)),
            OptFloatQ::Not(x) => !x.execute(o),
            OptFloatQ::Pass => true,
        }
    }
}

#[derive(Debug, Clone)]
pub enum OptIntQ {
    Defined,
    NotDefined,
    IfDefinedThen(IntQ),
    DefinedAnd(IntQ),
    And(Vec<OptIntQ>),
    Or(Vec<OptIntQ>),
    Not(Box<OptIntQ>),
    Pass,
}

impl ExecutableQuery<Option<i64>> for OptIntQ {
    fn execute(&self, o: Option<i64>) -> bool {
        match self {
            OptIntQ::Defined => o.is_some(),
            OptIntQ::NotDefined => o.is_none(),
            OptIntQ::IfDefinedThen(x) => match o {
                Some(o) => x.execute(o),
                None => true,
            },
            OptIntQ::DefinedAnd(x) => match o {
                Some(o) => x.execute(o),
                None => false,
            },
            OptIntQ::And(v) => v.iter().all(|x| x.execute(o)),
            OptIntQ::Or(v) => v.iter().any(|x| x.execute(o)),
            OptIntQ::Not(x) => !x.execute(o),
            OptIntQ::Pass => true,
        }
    }
}

#[derive(Debug, Clone)]
pub enum OptStringQ {
    Defined,
    NotDefined,
    IfDefinedThen(StringQ),
    DefinedAnd(StringQ),
    And(Vec<OptStringQ>),
    Or(Vec<OptStringQ>),
    Not(Box<OptStringQ>),
    Pass,
}

impl ExecutableQuery<&Option<String>> for OptStringQ {
    fn execute(&self, o: &Option<String>) -> bool {
        match self {
            OptStringQ::Defined => o.is_some(),
            OptStringQ::NotDefined => o.is_none(),
            OptStringQ::IfDefinedThen(x) => match o {
                Some(o) => x.execute(o),
                None => true,
            },
            OptStringQ::DefinedAnd(x) => match o {
                Some(o) => x.execute(o),
                None => false,
            },
            OptStringQ::And(v) => v.iter().all(|x| x.execute(o)),
            OptStringQ::Or(v) => v.iter().any(|x| x.execute(o)),
            OptStringQ::Not(x) => !x.execute(o),
            OptStringQ::Pass => true,
        }
    }
}

#[derive(Debug, Clone)]
pub enum BoxQ {
    Width(FloatQ),
    Height(FloatQ),
    XC(FloatQ),
    YC(FloatQ),
    Area(FloatQ),
    Angle(OptFloatQ),
    And(Vec<BoxQ>),
    Or(Vec<BoxQ>),
    Not(Box<BoxQ>),
    Pass,
}

impl ExecutableQuery<&RBBox> for BoxQ {
    fn execute(&self, o: &RBBox) -> bool {
        match self {
            BoxQ::Width(x) => x.execute(o.width),
            BoxQ::Height(x) => x.execute(o.height),
            BoxQ::XC(x) => x.execute(o.xc),
            BoxQ::YC(x) => x.execute(o.yc),
            BoxQ::Area(x) => x.execute(o.width * o.height),
            BoxQ::Angle(x) => x.execute(o.angle),
            BoxQ::And(v) => v.iter().all(|q| q.execute(o)),
            BoxQ::Or(v) => v.iter().any(|q| q.execute(o)),
            BoxQ::Not(q) => !q.execute(o),
            BoxQ::Pass => true,
        }
    }
}

#[derive(Debug, Clone)]
pub enum PropertiesQ {
    Id(IntQ),
    Creator(StringQ),
    Label(StringQ),
    Confidence(OptFloatQ),
    Box(BoxQ),
    TrackId(OptIntQ),
    And(Vec<PropertiesQ>),
    Or(Vec<PropertiesQ>),
    Not(Box<PropertiesQ>),
    Pass,
}

impl ExecutableQuery<&InnerObject> for PropertiesQ {
    fn execute(&self, o: &InnerObject) -> bool {
        match self {
            PropertiesQ::Id(x) => x.execute(o.id),
            PropertiesQ::Creator(x) => x.execute(&o.creator),
            PropertiesQ::Label(x) => x.execute(&o.label),
            PropertiesQ::Confidence(x) => x.execute(o.confidence),
            PropertiesQ::Box(x) => x.execute(&o.bbox),
            PropertiesQ::TrackId(x) => x.execute(o.track_id),
            PropertiesQ::And(v) => v.iter().all(|q| q.execute(o)),
            PropertiesQ::Or(v) => v.iter().any(|q| q.execute(o)),
            PropertiesQ::Not(q) => !q.execute(o),
            PropertiesQ::Pass => true,
        }
    }
}

impl ExecutableQuery<&Option<ParentObject>> for PropertiesQ {
    fn execute(&self, o: &Option<ParentObject>) -> bool {
        match o {
            Some(o) => match self {
                PropertiesQ::Id(x) => x.execute(o.id),
                PropertiesQ::Creator(x) => x.execute(&o.creator),
                PropertiesQ::Label(x) => x.execute(&o.label),
                _ => panic!("Not implemented"),
            },
            None => false,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Q {
    Object(PropertiesQ),
    ParentObject(PropertiesQ),
    And(Vec<Q>),
    Or(Vec<Q>),
    Not(Box<Q>),
    Pass,
}

impl ExecutableQuery<&InnerObject> for Q {
    fn execute(&self, o: &InnerObject) -> bool {
        match self {
            Q::Object(p) => p.execute(o),
            Q::ParentObject(p) => p.execute(&o.parent),
            Q::And(v) => v.iter().all(|x| x.execute(o)),
            Q::Or(v) => v.iter().any(|x| x.execute(o)),
            Q::Not(x) => !x.execute(o),
            Q::Pass => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ExecutableQuery, FloatQ, OptFloatQ, PropertiesQ, StringQ, Q};
    use crate::test::utils::gen_frame;

    #[test]
    fn query() {
        let expr = Q::And(vec![
            Q::Object(PropertiesQ::Creator(StringQ::Or(vec![
                StringQ::EQ("test2".to_string()),
                StringQ::EQ("test".to_string()),
            ]))),
            Q::Object(PropertiesQ::Confidence(OptFloatQ::DefinedAnd(FloatQ::GE(
                0.5,
            )))),
        ]);

        let f = gen_frame();
        let objs = f.access_objects(false, None, None);
        dbg!(&objs);
        let res = objs
            .iter()
            .map(|o| expr.execute(&o.inner.lock().unwrap()))
            .collect::<Vec<_>>();
        dbg!(&res);
    }
}
