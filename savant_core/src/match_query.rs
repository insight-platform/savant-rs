use crate::eval_cache::{get_compiled_eval_expr, get_compiled_jmp_filter};
use crate::eval_context::ObjectContext;
use crate::eval_resolvers::{
    config_resolver_name, env_resolver_name, etcd_resolver_name, utility_resolver_name,
};
use crate::json_api::ToSerdeJsonValue;
use crate::pluggable_udf_api::{
    call_object_inplace_modifier, call_object_map_modifier, call_object_predicate,
    is_plugin_function_registered, register_plugin_function, UserFunctionType,
};
use crate::primitives::frame::{VideoFrameContent, VideoFrameTranscodingMethod};
use crate::primitives::object::{VideoObject, VideoObjectProxy};
use crate::primitives::{AttributeMethods, Attributive, BBoxMetricType, RBBox};
use parking_lot::RwLockReadGuard;
use savant_utils::iter::{
    all_with_control_flow, any_with_control_flow, fiter_map_with_control_flow,
    partition_with_control_flow,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::ControlFlow;

pub use crate::query_and as and;
pub use crate::query_not as not;
pub use crate::query_or as or;
pub use crate::query_stop_if_false as stop_if_false;
pub use crate::query_stop_if_true as stop_if_true;

pub type VideoObjectsProxyBatch = HashMap<i64, Vec<VideoObjectProxy>>;

pub trait ExecutableMatchQuery<T, C> {
    fn execute(&self, o: T, ctx: &mut C) -> ControlFlow<bool, bool>;
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
    fn execute(&self, o: &f32, _: &mut ()) -> ControlFlow<bool, bool> {
        ControlFlow::Continue(match self {
            FloatExpression::EQ(x) => x == o,
            FloatExpression::NE(x) => x != o,
            FloatExpression::LT(x) => x > o,
            FloatExpression::LE(x) => x >= o,
            FloatExpression::GT(x) => x < o,
            FloatExpression::GE(x) => x <= o,
            FloatExpression::Between(a, b) => a <= o && o <= b,
            FloatExpression::OneOf(v) => v.contains(o),
        })
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
    fn execute(&self, o: &i64, _: &mut ()) -> ControlFlow<bool, bool> {
        ControlFlow::Continue(match self {
            IntExpression::EQ(x) => x == o,
            IntExpression::NE(x) => x != o,
            IntExpression::LT(x) => x > o,
            IntExpression::LE(x) => x >= o,
            IntExpression::GT(x) => x < o,
            IntExpression::GE(x) => x <= o,
            IntExpression::Between(a, b) => a <= o && o <= b,
            IntExpression::OneOf(v) => v.contains(o),
        })
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
    fn execute(&self, o: &String, _: &mut ()) -> ControlFlow<bool, bool> {
        ControlFlow::Continue(match self {
            StringExpression::EQ(x) => x == o,
            StringExpression::NE(x) => x != o,
            StringExpression::Contains(x) => o.contains(x),
            StringExpression::NotContains(x) => !o.contains(x),
            StringExpression::StartsWith(x) => o.starts_with(x),
            StringExpression::EndsWith(x) => o.ends_with(x),
            StringExpression::OneOf(v) => v.contains(o),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "match")]
pub enum MatchQuery {
    #[serde(rename = "id")]
    Id(IntExpression),
    #[serde(rename = "namespace")]
    Namespace(StringExpression),
    #[serde(rename = "label")]
    Label(StringExpression),
    #[serde(rename = "confidence.defined")]
    ConfidenceDefined,
    #[serde(rename = "confidence")]
    Confidence(FloatExpression),

    // track ops
    #[serde(rename = "track.defined")]
    TrackDefined,
    #[serde(rename = "track.id")]
    TrackId(IntExpression),
    #[serde(rename = "track.bbox.xc")]
    TrackBoxXCenter(FloatExpression),
    #[serde(rename = "track.bbox.yc")]
    TrackBoxYCenter(FloatExpression),
    #[serde(rename = "track.bbox.width")]
    TrackBoxWidth(FloatExpression),
    #[serde(rename = "track.bbox.height")]
    TrackBoxHeight(FloatExpression),
    #[serde(rename = "track.bbox.area")]
    TrackBoxArea(FloatExpression),
    #[serde(rename = "track.bbox.width_to_height_ratio")]
    TrackBoxWidthToHeightRatio(FloatExpression),
    #[serde(rename = "track.bbox.angle.defined")]
    TrackBoxAngleDefined,
    #[serde(rename = "track.bbox.angle")]
    TrackBoxAngle(FloatExpression),
    #[serde(rename = "track.bbox.metric")]
    TrackBoxMetric {
        other: (f32, f32, f32, f32, Option<f32>),
        metric_type: BBoxMetricType,
        threshold_expr: FloatExpression,
    },

    // parent
    #[serde(rename = "parent.defined")]
    ParentDefined,
    #[serde(rename = "parent.id")]
    ParentId(IntExpression),
    #[serde(rename = "parent.namespace")]
    ParentNamespace(StringExpression),
    #[serde(rename = "parent.label")]
    ParentLabel(StringExpression),

    // children query
    #[serde(rename = "with_children")]
    WithChildren(Box<MatchQuery>, IntExpression),

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
    #[serde(rename = "bbox.width_to_height_ratio")]
    BoxWidthToHeightRatio(FloatExpression),
    #[serde(rename = "bbox.angle.defined")]
    BoxAngleDefined,
    #[serde(rename = "bbox.angle")]
    BoxAngle(FloatExpression),
    #[serde(rename = "bbox.metric")]
    BoxMetric {
        other: (f32, f32, f32, f32, Option<f32>),
        metric_type: BBoxMetricType,
        threshold_expr: FloatExpression,
    },
    // Attributes
    #[serde(rename = "attribute.exists")]
    AttributeExists(String, String),
    #[serde(rename = "attributes.empty")]
    AttributesEmpty,
    #[serde(rename = "attributes.jmes_query")]
    AttributesJMESQuery(String),

    // combinators
    #[serde(rename = "and")]
    And(Vec<MatchQuery>),
    #[serde(rename = "or")]
    Or(Vec<MatchQuery>),
    #[serde(rename = "not")]
    Not(Box<MatchQuery>),
    #[serde(rename = "pass")]
    Idle,
    #[serde(rename = "stop_if_false")]
    StopIfFalse(Box<MatchQuery>),
    #[serde(rename = "stop_if_true")]
    StopIfTrue(Box<MatchQuery>),
    // User-defined plugin function
    #[serde(rename = "user_defined_object_predicate")]
    UserDefinedObjectPredicate(String, String),
    #[serde(rename = "eval")]
    EvalExpr(String),

    // Frame Properties
    #[serde(rename = "frame.source_id")]
    FrameSourceId(StringExpression),
    #[serde(rename = "frame.is_key_frame")]
    FrameIsKeyFrame,
    #[serde(rename = "frame.transcoding.is_copy")]
    FrameTranscodingIsCopy,
    #[serde(rename = "frame.width")]
    FrameWidth(IntExpression),
    #[serde(rename = "frame.height")]
    FrameHeight(IntExpression),
    #[serde(rename = "frame.no_video")]
    FrameNoVideo,

    // Frame Attributes
    #[serde(rename = "frame.attribute.exists")]
    FrameAttributeExists(String, String),
    #[serde(rename = "frame.attributes.empty")]
    FrameAttributesEmpty,
    #[serde(rename = "frame.attributes.jmes_query")]
    FrameAttributesJMESQuery(String),
}

impl ExecutableMatchQuery<&RwLockReadGuard<'_, VideoObject>, ()> for MatchQuery {
    fn execute(&self, o: &RwLockReadGuard<VideoObject>, _: &mut ()) -> ControlFlow<bool, bool> {
        let detection_box = o.detection_box.clone();
        let tracking_box = o.track_box.clone();
        match self {
            MatchQuery::Id(x) => x.execute(&o.id, &mut ()),
            MatchQuery::Namespace(x) => x.execute(&o.namespace, &mut ()),
            MatchQuery::Label(x) => x.execute(&o.label, &mut ()),
            MatchQuery::Confidence(x) => o
                .confidence
                .map(|c| x.execute(&c, &mut ()))
                .unwrap_or(ControlFlow::Continue(false)),
            MatchQuery::ConfidenceDefined => ControlFlow::Continue(o.confidence.is_some()),
            MatchQuery::TrackDefined => ControlFlow::Continue(o.track_id.is_some()),
            MatchQuery::TrackId(x) => o
                .track_id
                .as_ref()
                .map(|id| x.execute(id, &mut ()))
                .unwrap_or(ControlFlow::Continue(false)),
            MatchQuery::TrackBoxXCenter(x) => tracking_box
                .as_ref()
                .map(|t| x.execute(&t.get_xc(), &mut ()))
                .unwrap_or(ControlFlow::Continue(false)),
            MatchQuery::TrackBoxYCenter(x) => tracking_box
                .as_ref()
                .map(|t| x.execute(&t.get_yc(), &mut ()))
                .unwrap_or(ControlFlow::Continue(false)),
            MatchQuery::TrackBoxWidth(x) => tracking_box
                .as_ref()
                .map(|t| x.execute(&t.get_width(), &mut ()))
                .unwrap_or(ControlFlow::Continue(false)),
            MatchQuery::TrackBoxHeight(x) => tracking_box
                .as_ref()
                .map(|t| x.execute(&t.get_height(), &mut ()))
                .unwrap_or(ControlFlow::Continue(false)),
            MatchQuery::TrackBoxWidthToHeightRatio(x) => tracking_box
                .as_ref()
                .map(|t| x.execute(&t.get_width_to_height_ratio(), &mut ()))
                .unwrap_or(ControlFlow::Continue(false)),
            MatchQuery::TrackBoxArea(x) => tracking_box
                .as_ref()
                .map(|t| x.execute(&(t.get_width() * t.get_height()), &mut ()))
                .unwrap_or(ControlFlow::Continue(false)),
            MatchQuery::TrackBoxAngle(x) => tracking_box
                .as_ref()
                .and_then(|t| t.get_angle().map(|a| x.execute(&a, &mut ())))
                .unwrap_or(ControlFlow::Continue(false)),
            MatchQuery::TrackBoxMetric {
                other,
                metric_type,
                threshold_expr,
            } => tracking_box
                .as_ref()
                .map_or(ControlFlow::Continue(false), |t| {
                    let other = RBBox::new(other.0, other.1, other.2, other.3, other.4);
                    let metric = match metric_type {
                        BBoxMetricType::IoU => t.iou(&other).unwrap_or(0.0),
                        BBoxMetricType::IoSelf => t.ios(&other).unwrap_or(0.0),
                        BBoxMetricType::IoOther => t.ioo(&other).unwrap_or(0.0),
                    };
                    threshold_expr.execute(&metric, &mut ())
                }),

            // parent
            MatchQuery::ParentDefined => ControlFlow::Continue(o.parent_id.is_some()),
            // box
            MatchQuery::BoxWidth(x) => x.execute(&detection_box.get_width(), &mut ()),
            MatchQuery::BoxHeight(x) => x.execute(&detection_box.get_height(), &mut ()),
            MatchQuery::BoxXCenter(x) => x.execute(&detection_box.get_xc(), &mut ()),
            MatchQuery::BoxYCenter(x) => x.execute(&detection_box.get_yc(), &mut ()),
            MatchQuery::BoxAngleDefined => {
                ControlFlow::Continue(detection_box.get_angle().is_some())
            }
            MatchQuery::BoxArea(x) => x.execute(
                &(detection_box.get_width() * detection_box.get_height()),
                &mut (),
            ),
            MatchQuery::BoxWidthToHeightRatio(x) => {
                x.execute(&detection_box.get_width_to_height_ratio(), &mut ())
            }
            MatchQuery::BoxAngle(x) => detection_box
                .get_angle()
                .map(|a| x.execute(&a, &mut ()))
                .unwrap_or(ControlFlow::Continue(false)),
            MatchQuery::BoxMetric {
                other: bbox,
                metric_type,
                threshold_expr,
            } => {
                let other = RBBox::new(bbox.0, bbox.1, bbox.2, bbox.3, bbox.4);
                let metric = match metric_type {
                    BBoxMetricType::IoU => detection_box.iou(&other).unwrap_or(0.0),
                    BBoxMetricType::IoSelf => detection_box.ios(&other).unwrap_or(0.0),
                    BBoxMetricType::IoOther => detection_box.ioo(&other).unwrap_or(0.0),
                };
                threshold_expr.execute(&metric, &mut ())
            }

            // attributes
            MatchQuery::AttributeExists(namespace, label) => {
                ControlFlow::Continue(o.contains_attribute(namespace, label))
            }
            MatchQuery::AttributesEmpty => ControlFlow::Continue(o.attributes.is_empty()),
            MatchQuery::AttributesJMESQuery(x) => {
                let filter = get_compiled_jmp_filter(x).unwrap();
                let json = &serde_json::json!(o
                    .attributes
                    .iter()
                    .map(|v| v.to_serde_json_value())
                    .collect::<Vec<_>>());
                let res = filter.search(json).unwrap();
                ControlFlow::Continue(
                    !(res.is_null()
                        || (res.is_array() && res.as_array().unwrap().is_empty())
                        || (res.is_boolean() && !res.as_boolean().unwrap())
                        || (res.is_object()) && res.as_object().unwrap().is_empty()),
                )
            }
            MatchQuery::Idle => ControlFlow::Continue(true),
            _ => panic!("not implemented"),
        }
    }
}

impl ExecutableMatchQuery<&VideoObjectProxy, ObjectContext<'_>> for MatchQuery {
    fn execute(&self, o: &VideoObjectProxy, ctx: &mut ObjectContext) -> ControlFlow<bool, bool> {
        match self {
            MatchQuery::Idle => ControlFlow::Continue(true),
            MatchQuery::And(v) => all_with_control_flow(v.iter(), |x| x.execute(o, ctx)),
            MatchQuery::Or(v) => any_with_control_flow(v.iter(), |x| x.execute(o, ctx)),
            MatchQuery::Not(x) => match x.execute(o, ctx) {
                ControlFlow::Continue(x) => ControlFlow::Continue(!x),
                ControlFlow::Break(x) => ControlFlow::Break(!x),
            },
            MatchQuery::StopIfFalse(x) => match x.execute(o, ctx) {
                ControlFlow::Continue(true) => ControlFlow::Continue(true),
                ControlFlow::Continue(false) => ControlFlow::Break(false),
                ControlFlow::Break(x) => ControlFlow::Break(x),
            },
            MatchQuery::StopIfTrue(x) => match x.execute(o, ctx) {
                ControlFlow::Continue(true) => ControlFlow::Break(true),
                ControlFlow::Continue(false) => ControlFlow::Continue(false),
                ControlFlow::Break(x) => ControlFlow::Break(x),
            },
            MatchQuery::WithChildren(q, n) => {
                let children = o.get_children();
                let v = filter(&children, q).len() as i64;
                n.execute(&v, &mut ())
            }
            MatchQuery::EvalExpr(x) => {
                let expr = get_compiled_eval_expr(x).unwrap();
                ControlFlow::Continue(expr.eval_boolean_with_context_mut(ctx).unwrap())
            }
            MatchQuery::ParentId(x) => o
                .get_parent()
                .as_ref()
                .map(|p| x.execute(&p.get_id(), &mut ()))
                .unwrap_or(ControlFlow::Continue(false)),
            MatchQuery::ParentNamespace(x) => o
                .get_parent()
                .as_ref()
                .map(|p| x.execute(&p.get_namespace(), &mut ()))
                .unwrap_or(ControlFlow::Continue(false)),
            MatchQuery::ParentLabel(x) => o
                .get_parent()
                .as_ref()
                .map(|p| x.execute(&p.get_label(), &mut ()))
                .unwrap_or(ControlFlow::Continue(false)),
            MatchQuery::UserDefinedObjectPredicate(plugin, function) => {
                let udf_name = format!("{}@{}", plugin, function);
                if !is_plugin_function_registered(&udf_name) {
                    register_plugin_function(
                        plugin,
                        function,
                        &UserFunctionType::ObjectPredicate,
                        &udf_name,
                    )
                    .unwrap_or_else(|e| {
                        panic!(
                            "Failed to register '{}' plugin function. Error: {:?}",
                            udf_name, e
                        )
                    });
                }
                ControlFlow::Continue(call_object_predicate(&udf_name, &[o]).unwrap_or_else(|e| {
                    panic!(
                        "Failed to call '{}' plugin function. Error: {:?}",
                        udf_name, e
                    )
                }))
            }
            MatchQuery::FrameSourceId(se) => {
                let parent_frame_opt = o.get_frame();
                if parent_frame_opt.is_none() {
                    return ControlFlow::Continue(false);
                }
                let parent_frame = parent_frame_opt.unwrap();
                se.execute(&parent_frame.get_source_id(), &mut ())
            }
            MatchQuery::FrameIsKeyFrame => {
                let parent_frame_opt = o.get_frame();
                if parent_frame_opt.is_none() {
                    return ControlFlow::Continue(false);
                }
                let parent_frame = parent_frame_opt.unwrap();
                if let Some(kf) = parent_frame.get_keyframe() {
                    ControlFlow::Continue(kf)
                } else {
                    ControlFlow::Continue(false)
                }
            }
            MatchQuery::FrameTranscodingIsCopy => {
                let parent_frame_opt = o.get_frame();
                if parent_frame_opt.is_none() {
                    return ControlFlow::Continue(false);
                }
                let parent_frame = parent_frame_opt.unwrap();
                match parent_frame.get_transcoding_method() {
                    VideoFrameTranscodingMethod::Copy => ControlFlow::Continue(true),
                    VideoFrameTranscodingMethod::Encoded => ControlFlow::Continue(false),
                }
            }
            MatchQuery::FrameWidth(x) => {
                let parent_frame_opt = o.get_frame();
                if parent_frame_opt.is_none() {
                    return ControlFlow::Continue(false);
                }
                let parent_frame = parent_frame_opt.unwrap();
                x.execute(&parent_frame.get_width(), &mut ())
            }
            MatchQuery::FrameHeight(x) => {
                let parent_frame_opt = o.get_frame();
                if parent_frame_opt.is_none() {
                    return ControlFlow::Continue(false);
                }
                let parent_frame = parent_frame_opt.unwrap();
                x.execute(&parent_frame.get_height(), &mut ())
            }

            MatchQuery::FrameNoVideo => {
                let parent_frame_opt = o.get_frame();
                if parent_frame_opt.is_none() {
                    return ControlFlow::Continue(false);
                }
                let parent_frame = parent_frame_opt.unwrap();

                ControlFlow::Continue(matches!(
                    &*parent_frame.get_content(),
                    VideoFrameContent::None
                ))
            }

            MatchQuery::FrameAttributeExists(namespace, label) => {
                let parent_frame_opt = o.get_frame();
                if parent_frame_opt.is_none() {
                    return ControlFlow::Continue(false);
                }
                let parent_frame = parent_frame_opt.unwrap();
                let res = !parent_frame
                    .find_attributes(&Some(namespace), &[label], &None)
                    .is_empty();

                ControlFlow::Continue(res)
            }
            MatchQuery::FrameAttributesEmpty => {
                let parent_frame_opt = o.get_frame();
                if parent_frame_opt.is_none() {
                    return ControlFlow::Continue(false);
                }
                let parent_frame = parent_frame_opt.unwrap();
                let res = parent_frame.get_attributes().is_empty();
                ControlFlow::Continue(res)
            }
            MatchQuery::FrameAttributesJMESQuery(x) => {
                let parent_frame_opt = o.get_frame();
                if parent_frame_opt.is_none() {
                    return ControlFlow::Continue(false);
                }
                let parent_frame = parent_frame_opt.unwrap();

                let filter = get_compiled_jmp_filter(x).unwrap();
                let attributes = parent_frame
                    .get_attributes()
                    .iter()
                    .flat_map(|(ns, l)| parent_frame.get_attribute(ns, l))
                    .collect::<Vec<_>>();

                let json = &serde_json::json!(attributes
                    .iter()
                    .map(|v| v.to_serde_json_value())
                    .collect::<Vec<_>>());
                let json_res = filter.search(json).unwrap();
                let res = !(json_res.is_null()
                    || (json_res.is_array() && json_res.as_array().unwrap().is_empty())
                    || (json_res.is_boolean() && !json_res.as_boolean().unwrap())
                    || (json_res.is_object()) && json_res.as_object().unwrap().is_empty());

                ControlFlow::Continue(res)
            }

            _ => {
                let inner = o.inner_read_lock();
                self.execute(&inner, &mut ())
            }
        }
    }
}

impl MatchQuery {
    pub fn execute_with_new_context(&self, o: &VideoObjectProxy) -> ControlFlow<bool, bool> {
        let mut context = ObjectContext::new(
            o,
            &[
                utility_resolver_name(),
                etcd_resolver_name(),
                config_resolver_name(),
                env_resolver_name(),
            ],
        );
        self.execute(o, &mut context)
    }

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

pub fn filter(objs: &[VideoObjectProxy], query: &MatchQuery) -> Vec<VideoObjectProxy> {
    fiter_map_with_control_flow(objs.iter(), |o| query.execute_with_new_context(o))
        .into_iter()
        .cloned()
        .collect()
}

pub fn partition(
    objs: &[VideoObjectProxy],
    query: &MatchQuery,
) -> (Vec<VideoObjectProxy>, Vec<VideoObjectProxy>) {
    let (a, b) = partition_with_control_flow(objs.iter(), |o| query.execute_with_new_context(o));
    (
        a.into_iter().cloned().collect(),
        b.into_iter().cloned().collect(),
    )
}

pub fn map_udf(objs: &[VideoObjectProxy], udf: &str) -> anyhow::Result<Vec<VideoObjectProxy>> {
    objs.iter()
        .map(|o| call_object_map_modifier(udf, o))
        .collect()
}

pub fn foreach_udf(objs: &[VideoObjectProxy], udf: &str) -> Vec<anyhow::Result<()>> {
    objs.iter()
        .map(|o| call_object_inplace_modifier(udf, &[o]))
        .collect()
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
macro_rules! query_stop_if_false {
    ($arg:expr) => {{
        MatchQuery::StopIfFalse(Box::new($arg))
    }};
}

#[macro_export]
macro_rules! query_stop_if_true {
    ($arg:expr) => {{
        MatchQuery::StopIfTrue(Box::new($arg))
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
    use crate::eval_resolvers::register_env_resolver;
    use crate::match_query::MatchQuery::*;
    use crate::primitives::attribute_value::{AttributeValue, AttributeValueVariant};
    use crate::primitives::object::IdCollisionResolutionPolicy;
    use crate::primitives::{Attribute, AttributeMethods};
    use crate::test::{gen_empty_frame, gen_frame, gen_object, s};

    #[test]
    fn test_stop_false() {
        let expr = query_stop_if_false!(Id(eq(1)));
        assert!(matches!(
            expr.execute_with_new_context(&gen_object(1)),
            ControlFlow::Continue(true)
        ));

        let expr = query_stop_if_false!(Id(eq(2)));
        assert!(matches!(
            expr.execute_with_new_context(&gen_object(1)),
            ControlFlow::Break(false)
        ));
    }

    #[test]
    fn test_stop_true() {
        let expr = query_stop_if_true!(Id(eq(1)));
        assert!(matches!(
            expr.execute_with_new_context(&gen_object(1)),
            ControlFlow::Break(true)
        ));

        let expr = query_stop_if_true!(Id(eq(2)));
        assert!(matches!(
            expr.execute_with_new_context(&gen_object(1)),
            ControlFlow::Continue(false)
        ));
    }

    #[test]
    fn test_int() {
        use IntExpression as IE;
        let eq_q: IE = eq(1);
        assert!(matches!(
            eq_q.execute(&1, &mut ()),
            ControlFlow::Continue(true)
        ));

        let ne_q: IE = ne(1);
        assert!(matches!(
            ne_q.execute(&2, &mut ()),
            ControlFlow::Continue(true)
        ));

        let gt_q: IE = gt(1);
        assert!(matches!(
            gt_q.execute(&2, &mut ()),
            ControlFlow::Continue(true)
        ));

        let lt_q: IE = lt(1);
        assert!(matches!(
            lt_q.execute(&0, &mut ()),
            ControlFlow::Continue(true)
        ));

        let ge_q: IE = ge(1);
        assert!(matches!(
            ge_q.execute(&1, &mut ()),
            ControlFlow::Continue(true)
        ));
        assert!(matches!(
            ge_q.execute(&2, &mut ()),
            ControlFlow::Continue(true)
        ));

        let le_q: IE = le(1);
        assert!(matches!(
            le_q.execute(&1, &mut ()),
            ControlFlow::Continue(true)
        ));
        assert!(matches!(
            le_q.execute(&0, &mut ()),
            ControlFlow::Continue(true)
        ));

        let between_q: IE = between(1, 5);
        assert!(matches!(
            between_q.execute(&2, &mut ()),
            ControlFlow::Continue(true)
        ));
        assert!(matches!(
            between_q.execute(&1, &mut ()),
            ControlFlow::Continue(true)
        ));
        assert!(matches!(
            between_q.execute(&5, &mut ()),
            ControlFlow::Continue(true)
        ));
        assert!(matches!(
            between_q.execute(&6, &mut ()),
            ControlFlow::Continue(false)
        ));

        let one_of_q: IE = one_of(&[1, 2, 3]);
        assert!(matches!(
            one_of_q.execute(&2, &mut ()),
            ControlFlow::Continue(true)
        ));
        assert!(matches!(
            one_of_q.execute(&4, &mut ()),
            ControlFlow::Continue(false)
        ));
    }

    #[test]
    fn test_float() {
        use FloatExpression as FE;
        let eq_q: FE = eq(1.0);
        // replace all tests with matches!
        //assert!(eq_q.execute(&1.0, &mut ()));
        assert!(matches!(
            eq_q.execute(&1.0, &mut ()),
            ControlFlow::Continue(true)
        ));

        let ne_q: FE = ne(1.0);
        //assert!(ne_q.execute(&2.0, &mut ()));
        assert!(matches!(
            ne_q.execute(&2.0, &mut ()),
            ControlFlow::Continue(true)
        ));

        let gt_q: FE = gt(1.0);
        //assert!(gt_q.execute(&2.0, &mut ()));
        assert!(matches!(
            gt_q.execute(&2.0, &mut ()),
            ControlFlow::Continue(true)
        ));

        let lt_q: FE = lt(1.0);
        // assert!(lt_q.execute(&0.0, &mut ()));
        assert!(matches!(
            lt_q.execute(&0.0, &mut ()),
            ControlFlow::Continue(true)
        ));

        let ge_q: FE = ge(1.0);
        // assert!(ge_q.execute(&1.0, &mut ()));
        // assert!(ge_q.execute(&2.0, &mut ()));
        assert!(matches!(
            ge_q.execute(&1.0, &mut ()),
            ControlFlow::Continue(true)
        ));
        assert!(matches!(
            ge_q.execute(&2.0, &mut ()),
            ControlFlow::Continue(true)
        ));

        let le_q: FE = le(1.0);
        // assert!(le_q.execute(&1.0, &mut ()));
        // assert!(le_q.execute(&0.0, &mut ()));
        assert!(matches!(
            le_q.execute(&1.0, &mut ()),
            ControlFlow::Continue(true)
        ));
        assert!(matches!(
            le_q.execute(&0.0, &mut ()),
            ControlFlow::Continue(true)
        ));

        let between_q: FE = between(1.0, 5.0);
        // assert!(between_q.execute(&2.0, &mut ()));
        // assert!(between_q.execute(&1.0, &mut ()));
        // assert!(between_q.execute(&5.0, &mut ()));
        // assert!(!between_q.execute(&6.0, &mut ()));
        assert!(matches!(
            between_q.execute(&2.0, &mut ()),
            ControlFlow::Continue(true)
        ));
        assert!(matches!(
            between_q.execute(&1.0, &mut ()),
            ControlFlow::Continue(true)
        ));
        assert!(matches!(
            between_q.execute(&5.0, &mut ()),
            ControlFlow::Continue(true)
        ));
        assert!(matches!(
            between_q.execute(&6.0, &mut ()),
            ControlFlow::Continue(false)
        ));

        let one_of_q: FE = one_of(&[1.0, 2.0, 3.0]);
        // assert!(one_of_q.execute(&2.0, &mut ()));
        // assert!(!one_of_q.execute(&4.0, &mut ()));
        assert!(matches!(
            one_of_q.execute(&2.0, &mut ()),
            ControlFlow::Continue(true)
        ));
        assert!(matches!(
            one_of_q.execute(&4.0, &mut ()),
            ControlFlow::Continue(false)
        ));
    }

    #[test]
    fn test_string() {
        use StringExpression as SE;
        let eq_q: SE = eq("test");
        // assert!(eq_q.execute(&"test".to_string(), &mut ()));
        assert!(matches!(
            eq_q.execute(&"test".to_string(), &mut ()),
            ControlFlow::Continue(true)
        ));

        let ne_q: SE = ne("test");
        // assert!(ne_q.execute(&"test2".to_string(), &mut ()));
        assert!(matches!(
            ne_q.execute(&"test2".to_string(), &mut ()),
            ControlFlow::Continue(true)
        ));

        let contains_q: SE = contains("test");
        // assert!(contains_q.execute(&"testimony".to_string(), &mut ()));
        // assert!(contains_q.execute(&"supertest".to_string(), &mut ()));
        assert!(matches!(
            contains_q.execute(&"testimony".to_string(), &mut ()),
            ControlFlow::Continue(true)
        ));
        assert!(matches!(
            contains_q.execute(&"supertest".to_string(), &mut ()),
            ControlFlow::Continue(true)
        ));

        let not_contains_q: SE = not_contains("test");
        // assert!(not_contains_q.execute(&"apple".to_string(), &mut ()));
        assert!(matches!(
            not_contains_q.execute(&"apple".to_string(), &mut ()),
            ControlFlow::Continue(true)
        ));

        let starts_with_q: SE = starts_with("test");
        //assert!(starts_with_q.execute(&"testing".to_string(), &mut ()));
        //assert!(!starts_with_q.execute(&"tes".to_string(), &mut ()));
        assert!(matches!(
            starts_with_q.execute(&"testing".to_string(), &mut ()),
            ControlFlow::Continue(true)
        ));
        assert!(matches!(
            starts_with_q.execute(&"tes".to_string(), &mut ()),
            ControlFlow::Continue(false)
        ));

        let ends_with_q: SE = ends_with("test");
        //assert!(ends_with_q.execute(&"gettest".to_string(), &mut ()));
        //assert!(!ends_with_q.execute(&"supertes".to_string(), &mut ()));
        assert!(matches!(
            ends_with_q.execute(&"gettest".to_string(), &mut ()),
            ControlFlow::Continue(true)
        ));
        assert!(matches!(
            ends_with_q.execute(&"supertes".to_string(), &mut ()),
            ControlFlow::Continue(false)
        ));

        let one_of_q: SE = one_of(&["test", "me", "now"]);
        // assert!(one_of_q.execute(&"me".to_string(), &mut ()));
        // assert!(one_of_q.execute(&"now".to_string(), &mut ()));
        // assert!(!one_of_q.execute(&"random".to_string(), &mut ()));
        assert!(matches!(
            one_of_q.execute(&"me".to_string(), &mut ()),
            ControlFlow::Continue(true)
        ));
        assert!(matches!(
            one_of_q.execute(&"now".to_string(), &mut ()),
            ControlFlow::Continue(true)
        ));
        assert!(matches!(
            one_of_q.execute(&"random".to_string(), &mut ()),
            ControlFlow::Continue(false)
        ));
    }

    #[test]
    fn query() {
        let expr = query_and![
            Id(eq(1)),
            Namespace(one_of(&["test", "test2"])),
            Confidence(gt(0.5))
        ];

        let f = gen_frame();
        let _objs = f.access_objects(&expr);
        let json = serde_json::to_string(&expr).unwrap();
        let _q: super::MatchQuery = serde_json::from_str(&json).unwrap();
    }

    #[test]
    fn test_eval() {
        let expr = EvalExpr("id == 1".to_string());
        assert!(matches!(
            expr.execute_with_new_context(&gen_object(1)),
            ControlFlow::Continue(true)
        ));

        let expr = EvalExpr("id == 2".to_string());
        //assert!(!expr.execute_with_new_context(&gen_object(1)));
        assert!(matches!(
            expr.execute_with_new_context(&gen_object(1)),
            ControlFlow::Continue(false)
        ));

        register_env_resolver();
        let expr = EvalExpr("env(\"ABC\", \"X\") == \"X\"".to_string());
        //assert!(expr.execute_with_new_context(&gen_object(1)));
        assert!(matches!(
            expr.execute_with_new_context(&gen_object(1)),
            ControlFlow::Continue(true)
        ));
    }

    #[test]
    fn test_query() {
        let expr = Id(eq(1));
        // assert!(expr.execute_with_new_context(&gen_object(1)));
        assert!(matches!(
            expr.execute_with_new_context(&gen_object(1)),
            ControlFlow::Continue(true)
        ));

        let expr = Namespace(eq("peoplenet"));
        // assert!(expr.execute_with_new_context(&gen_object(1)));
        assert!(matches!(
            expr.execute_with_new_context(&gen_object(1)),
            ControlFlow::Continue(true)
        ));

        let expr = Label(starts_with("face"));
        // assert!(expr.execute_with_new_context(&gen_object(1)));
        assert!(matches!(
            expr.execute_with_new_context(&gen_object(1)),
            ControlFlow::Continue(true)
        ));

        let expr = Confidence(gt(0.4));
        // assert!(expr.execute_with_new_context(&gen_object(1)));
        assert!(matches!(
            expr.execute_with_new_context(&gen_object(1)),
            ControlFlow::Continue(true)
        ));

        let expr = ConfidenceDefined;
        // assert!(expr.execute_with_new_context(&gen_object(1)));
        assert!(matches!(
            expr.execute_with_new_context(&gen_object(1)),
            ControlFlow::Continue(true)
        ));

        let expr = ParentDefined;
        // assert!(!expr.execute_with_new_context(&gen_object(1)));
        assert!(matches!(
            expr.execute_with_new_context(&gen_object(1)),
            ControlFlow::Continue(false)
        ));

        let expr = AttributeExists("some".to_string(), "attribute".to_string());
        let o = gen_object(1);
        // assert!(expr.execute_with_new_context(&o));
        assert!(matches!(
            expr.execute_with_new_context(&o),
            ControlFlow::Continue(true)
        ));

        let expr = AttributesEmpty;
        let o = gen_object(1);
        o.delete_attributes_with_ns("some");
        // assert!(expr.execute_with_new_context(&o));
        assert!(matches!(
            expr.execute_with_new_context(&o),
            ControlFlow::Continue(true)
        ));

        let object = gen_object(1);
        let parent_object = gen_object(13);
        let f = gen_empty_frame();
        f.add_object(&parent_object, IdCollisionResolutionPolicy::Error)
            .unwrap();
        f.add_object(&object, IdCollisionResolutionPolicy::Error)
            .unwrap();
        assert!(object.set_parent(Some(parent_object.get_id())).is_ok());

        let expr = ParentId(eq(13));
        //assert!(expr.execute_with_new_context(&object));
        assert!(matches!(
            expr.execute_with_new_context(&object),
            ControlFlow::Continue(true)
        ));

        let expr = ParentNamespace(eq("peoplenet"));
        // assert!(expr.execute_with_new_context(&object));
        assert!(matches!(
            expr.execute_with_new_context(&object),
            ControlFlow::Continue(true)
        ));

        let expr = ParentLabel(eq("face"));
        // assert!(expr.execute_with_new_context(&object));
        assert!(matches!(
            expr.execute_with_new_context(&object),
            ControlFlow::Continue(true)
        ));

        let expr = BoxXCenter(gt(0.0));
        // assert!(expr.execute_with_new_context(&gen_object(1)));
        assert!(matches!(
            expr.execute_with_new_context(&gen_object(1)),
            ControlFlow::Continue(true)
        ));

        let expr = BoxYCenter(gt(1.0));
        // assert!(expr.execute_with_new_context(&gen_object(1)));
        assert!(matches!(
            expr.execute_with_new_context(&gen_object(1)),
            ControlFlow::Continue(true)
        ));

        let expr = BoxWidth(gt(5.0));
        // assert!(expr.execute_with_new_context(&gen_object(1)));
        assert!(matches!(
            expr.execute_with_new_context(&gen_object(1)),
            ControlFlow::Continue(true)
        ));

        let expr = BoxHeight(gt(10.0));
        // assert!(expr.execute_with_new_context(&gen_object(1)));
        assert!(matches!(
            expr.execute_with_new_context(&gen_object(1)),
            ControlFlow::Continue(true)
        ));

        let expr = BoxArea(gt(150.0));
        // assert!(expr.execute_with_new_context(&gen_object(1)));
        assert!(matches!(
            expr.execute_with_new_context(&gen_object(1)),
            ControlFlow::Continue(true)
        ));

        let expr = BoxArea(lt(250.0));
        // assert!(expr.execute_with_new_context(&gen_object(1)));
        assert!(matches!(
            expr.execute_with_new_context(&gen_object(1)),
            ControlFlow::Continue(true)
        ));

        let expr = BoxAngleDefined;
        // assert!(!expr.execute_with_new_context(&gen_object(1)));
        assert!(matches!(
            expr.execute_with_new_context(&gen_object(1)),
            ControlFlow::Continue(false)
        ));

        let object = gen_object(1);
        object.set_detection_box(RBBox::new(1.0, 2.0, 10.0, 20.0, Some(30.0)));
        // assert!(expr.execute_with_new_context(&object));
        assert!(matches!(
            expr.execute_with_new_context(&object),
            ControlFlow::Continue(true)
        ));

        let expr = BoxAngle(gt(20.0));
        // assert!(expr.execute_with_new_context(&object));
        assert!(matches!(
            expr.execute_with_new_context(&object),
            ControlFlow::Continue(true)
        ));

        let expr = TrackDefined;
        // assert!(expr.execute_with_new_context(&object));
        assert!(matches!(
            expr.execute_with_new_context(&object),
            ControlFlow::Continue(true)
        ));

        object.set_attribute(Attribute::persistent(
            s("classifier"),
            s("age-min-max-avg"),
            vec![
                AttributeValue::new(AttributeValueVariant::Float(10.0), Some(0.7)),
                AttributeValue::new(AttributeValueVariant::Float(20.0), Some(0.8)),
                AttributeValue::new(AttributeValueVariant::Float(15.0), None),
            ],
            Some(s("morphological-classifier")),
            false,
        ));

        let expr = AttributesJMESQuery(s(
            "[? (hint == 'morphological-classifier') && (namespace == 'classifier')]",
        ));
        // assert!(expr.execute_with_new_context(&object));
        assert!(matches!(
            expr.execute_with_new_context(&object),
            ControlFlow::Continue(true)
        ));

        let expr = AttributesJMESQuery(s(
            "[? (hint != 'morphological-classifier') && (namespace == 'classifier')]",
        ));
        // assert!(!expr.execute_with_new_context(&object));
        assert!(matches!(
            expr.execute_with_new_context(&object),
            ControlFlow::Continue(false)
        ));
    }

    #[test]
    fn test_logical_functions() {
        let expr = and![Id(eq(1)), Namespace(eq("peoplenet")), Confidence(gt(0.4))];
        // assert!(expr.execute_with_new_context(&gen_object(1)));
        assert!(matches!(
            expr.execute_with_new_context(&gen_object(1)),
            ControlFlow::Continue(true)
        ));

        let expr = or![Id(eq(10)), Namespace(eq("peoplenet")),];
        // assert!(expr.execute_with_new_context(&gen_object(1)));
        assert!(matches!(
            expr.execute_with_new_context(&gen_object(1)),
            ControlFlow::Continue(true)
        ));

        let expr = not!(Id(eq(2)));
        // assert!(expr.execute_with_new_context(&gen_object(1)));
        assert!(matches!(
            expr.execute_with_new_context(&gen_object(1)),
            ControlFlow::Continue(true)
        ));
    }

    #[test]
    fn test_children_expression() {
        let f = gen_frame();
        let o = f.access_objects(&WithChildren(Box::new(Idle), eq(2)));
        assert_eq!(o.len(), 1);
        assert_eq!(o[0].get_id(), 0);
    }

    #[test]
    fn test_filter() {
        let f = gen_frame();
        let objects = f.access_objects(&Idle);
        let filtered = filter(&objects, &Id(eq(1)));
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_partition() {
        let f = gen_frame();
        let objects = f.access_objects(&Idle);
        let (matching, others) = partition(&objects, &Id(eq(1)));
        assert_eq!(matching.len(), 1);
        assert_eq!(others.len(), 2);
    }

    #[test]
    fn test_udf() {
        let f = gen_frame();
        let objects = f.access_objects(&UserDefinedObjectPredicate(
            "../target/debug/libsavant_core.so".to_string(),
            "unary_op_even".to_string(),
        ));
        assert_eq!(objects.len(), 2, "Only even objects must be returned");
    }

    #[test]
    fn test_map_udf() {
        let f = gen_frame();
        let objects = f.access_objects(&Idle);

        let udf_name = "sample.map_modifier";
        if !is_plugin_function_registered(&udf_name) {
            register_plugin_function(
                "../target/debug/libsavant_core.so",
                "map_modifier",
                &UserFunctionType::ObjectMapModifier,
                udf_name,
            )
            .expect(format!("Failed to register '{}' plugin function", udf_name).as_str());
        }

        let new_objects = map_udf(&objects, "sample.map_modifier").unwrap();
        assert_eq!(new_objects.len(), 3);
        for o in new_objects {
            assert!(
                o.get_label().starts_with("modified"),
                "Label must be modified"
            );
        }
    }

    #[test]
    fn test_foreach_udf() {
        let f = gen_frame();
        let objects = f.access_objects(&Idle);

        let udf_name = "sample.inplace_modifier";
        if !is_plugin_function_registered(&udf_name) {
            register_plugin_function(
                "../target/debug/libsavant_core.so",
                "inplace_modifier",
                &UserFunctionType::ObjectInplaceModifier,
                udf_name,
            )
            .expect(format!("Failed to register '{}' plugin function", udf_name).as_str());
        }

        foreach_udf(&objects, "sample.inplace_modifier");

        for o in objects {
            assert!(
                o.get_label().starts_with("modified"),
                "Label must be modified"
            );
        }
    }

    #[test]
    fn test_bbox_metric_iou() {
        let expr = BoxMetric {
            other: (1.0, 2.0, 10.0, 20.0, None), // matches to the box defined in gen_object(1)
            metric_type: BBoxMetricType::IoU,
            threshold_expr: FloatExpression::GE(0.99),
        };
        //assert!(expr.execute_with_new_context(&gen_object(1)));
        assert!(matches!(
            expr.execute_with_new_context(&gen_object(1)),
            ControlFlow::Continue(true)
        ));
    }

    #[test]
    fn test_bbox_metric_ios() {
        let expr = BoxMetric {
            other: (1.0, 2.0, 20.0, 40.0, None), // matches to the box defined in gen_object(1)
            metric_type: BBoxMetricType::IoSelf,
            threshold_expr: FloatExpression::GE(0.99),
        };
        // assert!(expr.execute_with_new_context(&gen_object(1)));
        assert!(matches!(
            expr.execute_with_new_context(&gen_object(1)),
            ControlFlow::Continue(true)
        ));
    }

    #[test]
    fn test_bbox_metric_ioo() {
        let expr = BoxMetric {
            other: (1.0, 2.0, 100.0, 200.0, None), // matches to the box defined in gen_object(1)
            metric_type: BBoxMetricType::IoOther,
            threshold_expr: FloatExpression::LE(0.05), // < 10 * 20 / (100 * 200)
        };
        // assert!(expr.execute_with_new_context(&gen_object(1)));
        assert!(matches!(
            expr.execute_with_new_context(&gen_object(1)),
            ControlFlow::Continue(true)
        ));
    }

    #[test]
    fn test_track_bbox_metric_iou() {
        let expr = TrackBoxMetric {
            other: (100.0, 200.0, 10.0, 20.0, None), // matches to the tracking box defined in gen_object(1)
            metric_type: BBoxMetricType::IoU,
            threshold_expr: FloatExpression::GE(0.99),
        };
        // assert!(expr.execute_with_new_context(&gen_object(1)));
        assert!(matches!(
            expr.execute_with_new_context(&gen_object(1)),
            ControlFlow::Continue(true)
        ));
    }

    #[test]
    fn test_track_bbox_metric_ios() {
        let expr = TrackBoxMetric {
            other: (100.0, 200.0, 20.0, 40.0, None), // matches to the tracking box defined in gen_object(1)
            metric_type: BBoxMetricType::IoSelf,
            threshold_expr: FloatExpression::GE(0.99),
        };
        //assert!(expr.execute_with_new_context(&gen_object(1)));
        assert!(matches!(
            expr.execute_with_new_context(&gen_object(1)),
            ControlFlow::Continue(true)
        ));
    }

    #[test]
    fn test_track_bbox_metric_ioo() {
        let expr = TrackBoxMetric {
            other: (100.0, 200.0, 100.0, 200.0, None), // matches to the tracking box defined in gen_object(1)
            metric_type: BBoxMetricType::IoOther,
            threshold_expr: FloatExpression::LE(0.05), // < 10 * 20 / (100 * 200)
        };
        //assert!(expr.execute_with_new_context(&gen_object(1)));
        assert!(matches!(
            expr.execute_with_new_context(&gen_object(1)),
            ControlFlow::Continue(true)
        ));
    }

    #[test]
    fn test_frame_ops() {
        let f = gen_frame();
        let objects = f.access_objects(&or![stop_if_false!(FrameWidth(gt(1280))), Idle]);
        assert!(objects.is_empty());

        let objects = f.access_objects(&or![stop_if_false!(FrameWidth(eq(1280))), Idle]);
        assert_eq!(objects.len(), 3);

        let objects = f.access_objects(&or![stop_if_true!(FrameWidth(eq(1280))), Idle]);
        assert_eq!(objects.len(), 1);
    }
}
