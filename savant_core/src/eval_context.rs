use crate::primitives::object::{ObjectOperations, VideoObject};
use evalexpr::*;
use hashbrown::HashMap;
use std::cell::OnceCell;

use crate::eval_resolvers::EvalWithResolvers;
use crate::primitives::object::private::{SealedWithFrame, SealedWithParent};

const DEFAULT_GLOBAL_CONTEXT_VAR_NUM: usize = 16;
const DEFAULT_OBJECT_CONTEXT_VAR_NUM: usize = 8;

#[derive(Default)]
pub(crate) struct RBBoxFieldsView {
    pub xc: OnceCell<Value>,
    pub yc: OnceCell<Value>,
    pub width: OnceCell<Value>,
    pub height: OnceCell<Value>,
    pub angle: OnceCell<Value>,
}

#[derive(Default)]
pub(crate) struct FrameFieldsView {
    pub source: OnceCell<Value>,
    pub framerate: OnceCell<Value>,
    pub width: OnceCell<Value>,
    pub height: OnceCell<Value>,
    pub keyframe: OnceCell<Value>,

    pub pts: OnceCell<Value>,
    pub dts: OnceCell<Value>,

    pub time_base_nominator: OnceCell<Value>,
    pub time_base_denominator: OnceCell<Value>,
}

pub(crate) struct GlobalContext {
    pub resolvers: Vec<String>,
    pub temp_vars: HashMap<String, Value>,
}

pub(crate) struct ObjectContext<'a> {
    pub object: &'a VideoObject,
    pub resolvers: Vec<String>,
    pub temp_vars: HashMap<String, Value>,
    pub object_view: OnceCell<ObjectFieldsView>,
}

#[derive(Default)]
pub(crate) struct ObjectFieldsView {
    pub id: OnceCell<Value>,
    pub namespace: OnceCell<Value>,
    pub label: OnceCell<Value>,
    pub confidence: OnceCell<Value>,

    pub tracking_info_id: OnceCell<Value>,
    pub tracking_info_bbox: RBBoxFieldsView,

    pub bbox: RBBoxFieldsView,

    pub parent_id: OnceCell<Value>,
    pub parent_namespace: OnceCell<Value>,
    pub parent_label: OnceCell<Value>,

    pub frame: FrameFieldsView,
}

impl GlobalContext {
    pub fn new(resolvers: &[&str]) -> Self {
        GlobalContext {
            resolvers: resolvers.iter().map(|s| s.to_string()).collect(),
            temp_vars: HashMap::with_capacity(DEFAULT_GLOBAL_CONTEXT_VAR_NUM),
        }
    }
}

impl<'a> ObjectContext<'a> {
    pub fn new(object: &'a VideoObject, resolvers: &[&str]) -> Self {
        ObjectContext {
            object,
            resolvers: resolvers.iter().map(|s| s.to_string()).collect(),
            temp_vars: HashMap::with_capacity(DEFAULT_OBJECT_CONTEXT_VAR_NUM),
            object_view: OnceCell::default(),
        }
    }
}

impl EvalWithResolvers for GlobalContext {
    fn get_resolvers(&self) -> &'_ [String] {
        self.resolvers.as_slice()
    }
}

impl<'a> EvalWithResolvers for ObjectContext<'a> {
    fn get_resolvers(&self) -> &'_ [String] {
        self.resolvers.as_slice()
    }
}

impl Context for GlobalContext {
    fn get_value(&self, identifier: &str) -> Option<&Value> {
        self.temp_vars.get(identifier)
    }

    fn call_function(&self, identifier: &str, argument: &Value) -> EvalexprResult<Value> {
        self.resolve(identifier, argument)
    }

    fn are_builtin_functions_disabled(&self) -> bool {
        false
    }

    fn set_builtin_functions_disabled(&mut self, _: bool) -> EvalexprResult<()> {
        Ok(())
    }
}

impl<'a> Context for ObjectContext<'a> {
    fn get_value(&self, identifier: &str) -> Option<&Value> {
        if let Some(v) = self.temp_vars.get(identifier) {
            return Some(v);
        }

        let object_view = self.object_view.get_or_init(ObjectFieldsView::default);

        match identifier {
            "id" => Some(
                object_view
                    .id
                    .get_or_init(|| Value::from(self.object.get_id())),
            ),
            "namespace" => Some(
                object_view
                    .namespace
                    .get_or_init(|| Value::from(self.object.get_namespace())),
            ),
            "label" => Some(
                object_view
                    .label
                    .get_or_init(|| Value::from(self.object.get_label())),
            ),
            "confidence" => {
                Some(
                    object_view
                        .confidence
                        .get_or_init(|| match self.object.get_confidence() {
                            None => Value::Empty,
                            Some(c) => Value::from(c as f64),
                        }),
                )
            }
            "parent.id" => {
                Some(
                    object_view
                        .parent_id
                        .get_or_init(|| match self.object.get_parent_id() {
                            None => Value::Empty,
                            Some(id) => Value::from(id),
                        }),
                )
            }
            "parent.namespace" => {
                Some(
                    object_view
                        .parent_namespace
                        .get_or_init(|| match &self.object.get_parent() {
                            None => Value::Empty,
                            Some(parent) => Value::from(parent.get_namespace()),
                        }),
                )
            }
            "parent.label" => {
                Some(
                    object_view
                        .parent_label
                        .get_or_init(|| match &self.object.get_parent() {
                            None => Value::Empty,
                            Some(parent) => Value::from(parent.get_label()),
                        }),
                )
            }

            "tracking_info.id" => {
                Some(object_view.tracking_info_id.get_or_init(
                    || match self.object.get_track_id() {
                        None => Value::Empty,
                        Some(id) => Value::from(id),
                    },
                ))
            }

            "tracking_info.bbox.xc" => Some(object_view.tracking_info_bbox.xc.get_or_init(|| {
                match self.object.get_track_box() {
                    None => Value::Empty,
                    Some(info) => Value::from(info.get_xc() as f64),
                }
            })),

            "tracking_info.bbox.yc" => Some(object_view.tracking_info_bbox.yc.get_or_init(|| {
                match self.object.get_track_box() {
                    None => Value::Empty,
                    Some(info) => Value::from(info.get_yc() as f64),
                }
            })),

            "tracking_info.bbox.width" => {
                Some(object_view.tracking_info_bbox.width.get_or_init(|| {
                    match self.object.get_track_box() {
                        None => Value::Empty,
                        Some(info) => Value::from(info.get_width() as f64),
                    }
                }))
            }

            "tracking_info.bbox.height" => {
                Some(object_view.tracking_info_bbox.height.get_or_init(|| {
                    match self.object.get_track_box() {
                        None => Value::Empty,
                        Some(info) => Value::from(info.get_height() as f64),
                    }
                }))
            }

            "tracking_info.bbox.angle" => {
                Some(object_view.tracking_info_bbox.angle.get_or_init(|| {
                    match self.object.get_track_box() {
                        None => Value::Empty,
                        Some(info) => match info.get_angle() {
                            None => Value::Empty,
                            Some(angle) => Value::from(angle as f64),
                        },
                    }
                }))
            }

            "bbox.xc" => Some(
                object_view
                    .bbox
                    .xc
                    .get_or_init(|| Value::from(self.object.get_detection_box().get_xc() as f64)),
            ),

            "bbox.yc" => Some(
                object_view
                    .bbox
                    .yc
                    .get_or_init(|| Value::from(self.object.get_detection_box().get_yc() as f64)),
            ),

            "bbox.width" => {
                Some(object_view.bbox.width.get_or_init(|| {
                    Value::from(self.object.get_detection_box().get_width() as f64)
                }))
            }

            "bbox.height" => {
                Some(object_view.bbox.height.get_or_init(|| {
                    Value::from(self.object.get_detection_box().get_height() as f64)
                }))
            }

            "bbox.angle" => Some(object_view.bbox.angle.get_or_init(|| {
                match self.object.get_detection_box().get_angle() {
                    None => Value::Empty,
                    Some(a) => Value::from(a as f64),
                }
            })),

            "frame.source" => {
                Some(
                    object_view
                        .frame
                        .source
                        .get_or_init(|| match self.object.get_frame() {
                            None => Value::Empty,
                            Some(f) => Value::from(f.get_source_id()),
                        }),
                )
            }

            "frame.rate" => {
                Some(
                    object_view
                        .frame
                        .framerate
                        .get_or_init(|| match self.object.get_frame() {
                            None => Value::Empty,
                            Some(f) => Value::from(f.get_framerate()),
                        }),
                )
            }

            "frame.width" => {
                Some(
                    object_view
                        .frame
                        .width
                        .get_or_init(|| match self.object.get_frame() {
                            None => Value::Empty,
                            Some(f) => Value::from(f.get_width()),
                        }),
                )
            }

            "frame.height" => {
                Some(
                    object_view
                        .frame
                        .height
                        .get_or_init(|| match self.object.get_frame() {
                            None => Value::Empty,
                            Some(f) => Value::from(f.get_height()),
                        }),
                )
            }

            "frame.keyframe" => {
                Some(
                    object_view
                        .frame
                        .keyframe
                        .get_or_init(|| match self.object.get_frame() {
                            None => Value::Empty,
                            Some(f) => match f.get_keyframe() {
                                None => Value::Empty,
                                Some(kf) => Value::from(kf),
                            },
                        }),
                )
            }

            "frame.dts" => Some(object_view.frame.dts.get_or_init(
                || match self.object.get_frame() {
                    None => Value::Empty,
                    Some(f) => match f.get_dts() {
                        None => Value::Empty,
                        Some(dts) => Value::from(dts),
                    },
                },
            )),

            "frame.pts" => Some(object_view.frame.pts.get_or_init(
                || match self.object.get_frame() {
                    None => Value::Empty,
                    Some(f) => Value::from(f.get_pts()),
                },
            )),

            "frame.time_base.nominator" => {
                Some(object_view.frame.time_base_nominator.get_or_init(|| {
                    match self.object.get_frame() {
                        None => Value::Empty,
                        Some(f) => Value::from(f.get_time_base().0 as i64),
                    }
                }))
            }
            "frame.time_base.denominator" => {
                Some(object_view.frame.time_base_denominator.get_or_init(|| {
                    match self.object.get_frame() {
                        None => Value::Empty,
                        Some(f) => Value::from(f.get_time_base().1 as i64),
                    }
                }))
            }
            _ => None,
        }
    }

    fn call_function(&self, identifier: &str, argument: &Value) -> EvalexprResult<Value> {
        self.resolve(identifier, argument)
    }

    fn are_builtin_functions_disabled(&self) -> bool {
        false
    }

    fn set_builtin_functions_disabled(&mut self, _: bool) -> EvalexprResult<()> {
        Ok(())
    }
}

impl ContextWithMutableVariables for GlobalContext {
    fn set_value(&mut self, identifier: String, value: Value) -> EvalexprResult<()> {
        // check type mismatch
        if let Some(v) = self.get_value(&identifier) {
            if std::mem::discriminant(v) != std::mem::discriminant(&value) {
                return Err(EvalexprError::TypeError {
                    expected: vec![ValueType::from(v)],
                    actual: value,
                });
            }
        }
        self.temp_vars.insert(identifier, value);
        Ok(())
    }
}

impl<'a> ContextWithMutableVariables for ObjectContext<'a> {
    fn set_value(&mut self, identifier: String, value: Value) -> EvalexprResult<()> {
        // check type mismatch
        if let Some(v) = self.get_value(&identifier) {
            if std::mem::discriminant(v) != std::mem::discriminant(&value) {
                return Err(EvalexprError::TypeError {
                    expected: vec![ValueType::from(v)],
                    actual: value,
                });
            }
        }
        self.temp_vars.insert(identifier, value);
        Ok(())
    }
}
