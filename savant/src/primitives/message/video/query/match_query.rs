use crate::primitives::message::video::object::context::ObjectContext;
use crate::primitives::message::video::object::VideoObject;
use crate::primitives::message::video::query::VideoObjectsProxyBatch;
use crate::primitives::{RBBox, VideoObjectProxy};
use savant_core::to_json_value::ToSerdeJsonValue;

use crate::utils::eval_resolvers::{
    config_resolver_name, env_resolver_name, etcd_resolver_name, utility_resolver_name,
};
use crate::utils::pluggable_udf_api::{
    call_object_inplace_modifier, call_object_map_modifier, call_object_predicate,
    is_plugin_function_registered, register_plugin_function, UserFunctionType,
};
use parking_lot::RwLockReadGuard;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use savant_core::eval_cache::{get_compiled_eval_expr, get_compiled_jmp_filter};
use savant_core::match_query::{
    ExecutableMatchQuery, FloatExpression, IntExpression, StringExpression,
};
use savant_core::primitives::rust;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename = "match")]
pub enum MatchQuery {
    #[serde(rename = "object.id")]
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
    #[serde(rename = "track.id.defined")]
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
        metric_type: rust::BBoxMetricType,
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
        metric_type: rust::BBoxMetricType,
        threshold_expr: FloatExpression,
    },
    // Attributes
    #[serde(rename = "attribute.defined")]
    AttributeDefined(String, String),
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
    // pass-through
    Idle,
    // User-defined plugin function
    #[serde(rename = "user_defined_object_predicate")]
    UserDefinedObjectPredicate(String, String),
    #[serde(rename = "eval")]
    EvalExpr(String),
}

impl ExecutableMatchQuery<&RwLockReadGuard<'_, VideoObject>, ()> for MatchQuery {
    fn execute(&self, o: &RwLockReadGuard<VideoObject>, _: &mut ()) -> bool {
        let detection_box = RBBox::new_from_data(o.detection_box.clone());
        let tracking_box = o.track_box.clone().map(RBBox::new_from_data);
        match self {
            MatchQuery::Id(x) => x.execute(&o.id, &mut ()),
            MatchQuery::Namespace(x) => x.execute(&o.namespace, &mut ()),
            MatchQuery::Label(x) => x.execute(&o.label, &mut ()),
            MatchQuery::Confidence(x) => o
                .confidence
                .map(|c| x.execute(&c, &mut ()))
                .unwrap_or(false),
            MatchQuery::ConfidenceDefined => o.confidence.is_some(),
            MatchQuery::TrackDefined => o.track_id.is_some(),
            MatchQuery::TrackId(x) => o
                .track_id
                .as_ref()
                .map(|id| x.execute(id, &mut ()))
                .unwrap_or(false),
            MatchQuery::TrackBoxXCenter(x) => tracking_box
                .as_ref()
                .map(|t| x.execute(&t.get_xc(), &mut ()))
                .unwrap_or(false),
            MatchQuery::TrackBoxYCenter(x) => tracking_box
                .as_ref()
                .map(|t| x.execute(&t.get_yc(), &mut ()))
                .unwrap_or(false),
            MatchQuery::TrackBoxWidth(x) => tracking_box
                .as_ref()
                .map(|t| x.execute(&t.get_width(), &mut ()))
                .unwrap_or(false),
            MatchQuery::TrackBoxHeight(x) => tracking_box
                .as_ref()
                .map(|t| x.execute(&t.get_height(), &mut ()))
                .unwrap_or(false),
            MatchQuery::TrackBoxWidthToHeightRatio(x) => tracking_box
                .as_ref()
                .map(|t| x.execute(&t.get_width_to_height_ratio(), &mut ()))
                .unwrap_or(false),
            MatchQuery::TrackBoxArea(x) => tracking_box
                .as_ref()
                .map(|t| x.execute(&(t.get_width() * t.get_height()), &mut ()))
                .unwrap_or(false),
            MatchQuery::TrackBoxAngle(x) => tracking_box
                .as_ref()
                .and_then(|t| t.get_angle().map(|a| x.execute(&a, &mut ())))
                .unwrap_or(false),
            MatchQuery::TrackBoxMetric {
                other,
                metric_type,
                threshold_expr,
            } => tracking_box.as_ref().map_or(false, |t| {
                let other = RBBox::new(other.0, other.1, other.2, other.3, other.4);
                let metric = match metric_type {
                    rust::BBoxMetricType::IoU => t.iou(&other).unwrap_or(0.0),
                    rust::BBoxMetricType::IoSelf => t.ios(&other).unwrap_or(0.0),
                    rust::BBoxMetricType::IoOther => t.ioo(&other).unwrap_or(0.0),
                };
                threshold_expr.execute(&metric, &mut ())
            }),

            // parent
            MatchQuery::ParentDefined => o.parent_id.is_some(),
            // box
            MatchQuery::BoxWidth(x) => x.execute(&detection_box.get_width(), &mut ()),
            MatchQuery::BoxHeight(x) => x.execute(&detection_box.get_height(), &mut ()),
            MatchQuery::BoxXCenter(x) => x.execute(&detection_box.get_xc(), &mut ()),
            MatchQuery::BoxYCenter(x) => x.execute(&detection_box.get_yc(), &mut ()),
            MatchQuery::BoxAngleDefined => detection_box.get_angle().is_some(),
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
                .unwrap_or(false),
            MatchQuery::BoxMetric {
                other: bbox,
                metric_type,
                threshold_expr,
            } => {
                let other = RBBox::new(bbox.0, bbox.1, bbox.2, bbox.3, bbox.4);
                let metric = match metric_type {
                    rust::BBoxMetricType::IoU => detection_box.iou(&other).unwrap_or(0.0),
                    rust::BBoxMetricType::IoSelf => detection_box.ios(&other).unwrap_or(0.0),
                    rust::BBoxMetricType::IoOther => detection_box.ioo(&other).unwrap_or(0.0),
                };
                threshold_expr.execute(&metric, &mut ())
            }

            // attributes
            MatchQuery::AttributeDefined(namespace, label) => o
                .attributes
                .get(&(namespace.to_string(), label.to_string()))
                .is_some(),
            MatchQuery::AttributesEmpty => o.attributes.is_empty(),
            MatchQuery::AttributesJMESQuery(x) => {
                let filter = get_compiled_jmp_filter(x).unwrap();
                let json = &serde_json::json!(o
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
            MatchQuery::Idle => true,
            _ => panic!("not implemented"),
        }
    }
}

impl ExecutableMatchQuery<&VideoObjectProxy, ObjectContext<'_>> for MatchQuery {
    fn execute(&self, o: &VideoObjectProxy, ctx: &mut ObjectContext) -> bool {
        match self {
            MatchQuery::And(v) => v.iter().all(|x| x.execute(o, ctx)),
            MatchQuery::Or(v) => v.iter().any(|x| x.execute(o, ctx)),
            MatchQuery::Not(x) => !x.execute(o, ctx),
            MatchQuery::WithChildren(q, n) => {
                let children = o.get_children();
                let v = filter(&children, q).len() as i64;
                n.execute(&v, &mut ())
            }
            MatchQuery::EvalExpr(x) => {
                let expr = get_compiled_eval_expr(x).unwrap();
                expr.eval_boolean_with_context_mut(ctx).unwrap()
            }
            MatchQuery::ParentId(x) => o
                .get_parent()
                .as_ref()
                .map(|p| x.execute(&p.get_id(), &mut ()))
                .unwrap_or(false),
            MatchQuery::ParentNamespace(x) => o
                .get_parent()
                .as_ref()
                .map(|p| x.execute(&p.get_namespace(), &mut ()))
                .unwrap_or(false),
            MatchQuery::ParentLabel(x) => o
                .get_parent()
                .as_ref()
                .map(|p| x.execute(&p.get_label(), &mut ()))
                .unwrap_or(false),
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
                call_object_predicate(&udf_name, &[o]).unwrap_or_else(|e| {
                    panic!(
                        "Failed to call '{}' plugin function. Error: {:?}",
                        udf_name, e
                    )
                })
            }
            _ => {
                let inner = o.get_inner_read();
                self.execute(&inner, &mut ())
            }
        }
    }
}

impl MatchQuery {
    pub fn execute_with_new_context(&self, o: &VideoObjectProxy) -> bool {
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
    objs.iter()
        .filter_map(|o| {
            if query.execute_with_new_context(o) {
                Some(o.clone())
            } else {
                None
            }
        })
        .collect()
}

pub fn batch_filter(
    batch_objects: &VideoObjectsProxyBatch,
    q: &MatchQuery,
) -> VideoObjectsProxyBatch {
    batch_objects
        .par_iter()
        .map(|(k, v)| (*k, filter(v, q)))
        .filter(|(_, v)| !v.is_empty())
        .collect()
}

pub fn partition(
    objs: &[VideoObjectProxy],
    query: &MatchQuery,
) -> (Vec<VideoObjectProxy>, Vec<VideoObjectProxy>) {
    objs.iter().fold((Vec::new(), Vec::new()), |mut acc, o| {
        if query.execute_with_new_context(o) {
            acc.0.push(o.clone());
        } else {
            acc.1.push(o.clone());
        }
        acc
    })
}

pub fn batch_partition(
    batch_objects: VideoObjectsProxyBatch,
    q: &MatchQuery,
) -> (VideoObjectsProxyBatch, VideoObjectsProxyBatch) {
    let partitions: Vec<(i64, Vec<VideoObjectProxy>, Vec<VideoObjectProxy>)> = batch_objects
        .into_par_iter()
        .map(|(k, v)| {
            let (first, second) = partition(&v, q);
            (k, first, second)
        })
        .collect();

    partitions.into_iter().fold(
        (HashMap::new(), HashMap::new()),
        |mut acc, (k, first, second)| {
            acc.0.insert(k, first);
            acc.1.insert(k, second);
            acc
        },
    )
}

pub fn map_udf(objs: &[&VideoObjectProxy], udf: &str) -> anyhow::Result<Vec<VideoObjectProxy>> {
    objs.iter()
        .map(|o| call_object_map_modifier(udf, o))
        .collect()
}

pub fn batch_map_udf(
    batch_objects: &VideoObjectsProxyBatch,
    udf: &str,
) -> anyhow::Result<VideoObjectsProxyBatch> {
    batch_objects
        .par_iter()
        .map(|(k, v)| {
            let mapped = map_udf(v.iter().collect::<Vec<_>>().as_ref(), udf)?;
            Ok((*k, mapped))
        })
        .collect()
}

pub fn foreach_udf(objs: &[&VideoObjectProxy], udf: &str) -> Vec<anyhow::Result<()>> {
    objs.iter()
        .map(|o| call_object_inplace_modifier(udf, &[o]))
        .collect()
}

pub fn batch_foreach_udf(
    batch_objects: &VideoObjectsProxyBatch,
    udf: &str,
) -> HashMap<i64, Vec<anyhow::Result<()>>> {
    batch_objects
        .par_iter()
        .map(|(k, v)| (*k, foreach_udf(v.iter().collect::<Vec<_>>().as_ref(), udf)))
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::primitives::attribute::AttributeMethods;
    use crate::primitives::attribute_value::AttributeValue;
    use crate::primitives::message::video::query::match_query::MatchQuery::*;
    use crate::primitives::message::video::query::match_query::{
        filter, foreach_udf, map_udf, partition, MatchQuery,
    };
    use crate::primitives::{Attribute, IdCollisionResolutionPolicy, RBBox};
    use crate::test::utils::{gen_empty_frame, gen_frame, gen_object, s};
    use crate::utils::eval_resolvers::register_env_resolver;
    use crate::utils::pluggable_udf_api::{
        is_plugin_function_registered, register_plugin_function, UserFunctionType,
    };
    use savant_core::match_query::{
        and, eq, gt, lt, not, one_of, or, starts_with, FloatExpression,
    };
    use savant_core::primitives::BBoxMetricType;
    use savant_core::query_and;

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
        assert!(expr.execute_with_new_context(&gen_object(1)));

        let expr = EvalExpr("id == 2".to_string());
        assert!(!expr.execute_with_new_context(&gen_object(1)));

        register_env_resolver();
        let expr = EvalExpr("env(\"ABC\", \"X\") == \"X\"".to_string());
        assert!(expr.execute_with_new_context(&gen_object(1)));
    }

    #[test]
    fn test_query() {
        let expr = Id(eq(1));
        assert!(expr.execute_with_new_context(&gen_object(1)));

        let expr = Namespace(eq("peoplenet"));
        assert!(expr.execute_with_new_context(&gen_object(1)));

        let expr = Label(starts_with("face"));
        assert!(expr.execute_with_new_context(&gen_object(1)));

        let expr = Confidence(gt(0.4));
        assert!(expr.execute_with_new_context(&gen_object(1)));

        let expr = ConfidenceDefined;
        assert!(expr.execute_with_new_context(&gen_object(1)));

        let expr = ParentDefined;
        assert!(!expr.execute_with_new_context(&gen_object(1)));

        let expr = AttributeDefined("some".to_string(), "attribute".to_string());
        let o = gen_object(1);
        assert!(expr.execute_with_new_context(&o));

        let expr = AttributesEmpty;
        let o = gen_object(1);
        o.delete_attributes(Some("some".to_string()), vec![]);
        assert!(expr.execute_with_new_context(&o));

        let object = gen_object(1);
        let parent_object = gen_object(13);
        let f = gen_empty_frame();
        f.add_object(&parent_object, IdCollisionResolutionPolicy::Error)
            .unwrap();
        f.add_object(&object, IdCollisionResolutionPolicy::Error)
            .unwrap();
        object.set_parent(Some(parent_object.get_id()));

        let expr = ParentId(eq(13));
        assert!(expr.execute_with_new_context(&object));

        let expr = ParentNamespace(eq("peoplenet"));
        assert!(expr.execute_with_new_context(&object));

        let expr = ParentLabel(eq("face"));
        assert!(expr.execute_with_new_context(&object));

        let expr = BoxXCenter(gt(0.0));
        assert!(expr.execute_with_new_context(&gen_object(1)));

        let expr = BoxYCenter(gt(1.0));
        assert!(expr.execute_with_new_context(&gen_object(1)));

        let expr = BoxWidth(gt(5.0));
        assert!(expr.execute_with_new_context(&gen_object(1)));

        let expr = BoxHeight(gt(10.0));
        assert!(expr.execute_with_new_context(&gen_object(1)));

        let expr = BoxArea(gt(150.0));
        assert!(expr.execute_with_new_context(&gen_object(1)));

        let expr = BoxArea(lt(250.0));
        assert!(expr.execute_with_new_context(&gen_object(1)));

        let expr = BoxAngleDefined;
        assert!(!expr.execute_with_new_context(&gen_object(1)));

        let object = gen_object(1);
        object.set_detection_box(RBBox::new(1.0, 2.0, 10.0, 20.0, Some(30.0)));
        assert!(expr.execute_with_new_context(&object));

        let expr = BoxAngle(gt(20.0));
        assert!(expr.execute_with_new_context(&object));

        let expr = TrackDefined;
        assert!(expr.execute_with_new_context(&object));

        object.set_attribute(Attribute::persistent(
            s("classifier"),
            s("age-min-max-avg"),
            vec![
                AttributeValue::float(10.0, Some(0.7)),
                AttributeValue::float(20.0, Some(0.8)),
                AttributeValue::float(15.0, None),
            ],
            Some(s("morphological-classifier")),
        ));

        let expr = AttributesJMESQuery(s(
            "[? (hint == 'morphological-classifier') && (namespace == 'classifier')]",
        ));
        assert!(expr.execute_with_new_context(&object));

        let expr = AttributesJMESQuery(s(
            "[? (hint != 'morphological-classifier') && (namespace == 'classifier')]",
        ));
        assert!(!expr.execute_with_new_context(&object));
    }

    #[test]
    fn test_logical_functions() {
        let expr = and![Id(eq(1)), Namespace(eq("peoplenet")), Confidence(gt(0.4))];
        assert!(expr.execute_with_new_context(&gen_object(1)));

        let expr = or![Id(eq(10)), Namespace(eq("peoplenet")),];
        assert!(expr.execute_with_new_context(&gen_object(1)));

        let expr = not!(Id(eq(2)));
        assert!(expr.execute_with_new_context(&gen_object(1)));
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
            "../target/debug/libsavant_rs.so".to_string(),
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
                "../target/debug/libsavant_rs.so",
                "map_modifier",
                &UserFunctionType::ObjectMapModifier,
                udf_name,
            )
            .expect(format!("Failed to register '{}' plugin function", udf_name).as_str());
        }

        let new_objects = map_udf(
            &objects.iter().collect::<Vec<_>>().as_slice(),
            "sample.map_modifier",
        )
        .unwrap();
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
                "../target/debug/libsavant_rs.so",
                "inplace_modifier",
                &UserFunctionType::ObjectInplaceModifier,
                udf_name,
            )
            .expect(format!("Failed to register '{}' plugin function", udf_name).as_str());
        }

        foreach_udf(
            &objects.iter().collect::<Vec<_>>().as_slice(),
            "sample.inplace_modifier",
        );

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
        assert!(expr.execute_with_new_context(&gen_object(1)));
    }

    #[test]
    fn test_bbox_metric_ios() {
        let expr = BoxMetric {
            other: (1.0, 2.0, 20.0, 40.0, None), // matches to the box defined in gen_object(1)
            metric_type: BBoxMetricType::IoSelf,
            threshold_expr: FloatExpression::GE(0.99),
        };
        assert!(expr.execute_with_new_context(&gen_object(1)));
    }

    #[test]
    fn test_bbox_metric_ioo() {
        let expr = BoxMetric {
            other: (1.0, 2.0, 100.0, 200.0, None), // matches to the box defined in gen_object(1)
            metric_type: BBoxMetricType::IoOther,
            threshold_expr: FloatExpression::LE(0.05), // < 10 * 20 / (100 * 200)
        };
        assert!(expr.execute_with_new_context(&gen_object(1)));
    }

    #[test]
    fn test_track_bbox_metric_iou() {
        let expr = TrackBoxMetric {
            other: (100.0, 200.0, 10.0, 20.0, None), // matches to the tracking box defined in gen_object(1)
            metric_type: BBoxMetricType::IoU,
            threshold_expr: FloatExpression::GE(0.99),
        };
        assert!(expr.execute_with_new_context(&gen_object(1)));
    }

    #[test]
    fn test_track_bbox_metric_ios() {
        let expr = TrackBoxMetric {
            other: (100.0, 200.0, 20.0, 40.0, None), // matches to the tracking box defined in gen_object(1)
            metric_type: BBoxMetricType::IoSelf,
            threshold_expr: FloatExpression::GE(0.99),
        };
        assert!(expr.execute_with_new_context(&gen_object(1)));
    }

    #[test]
    fn test_track_bbox_metric_ioo() {
        let expr = TrackBoxMetric {
            other: (100.0, 200.0, 100.0, 200.0, None), // matches to the tracking box defined in gen_object(1)
            metric_type: BBoxMetricType::IoOther,
            threshold_expr: FloatExpression::LE(0.05), // < 10 * 20 / (100 * 200)
        };
        assert!(expr.execute_with_new_context(&gen_object(1)));
    }
}
