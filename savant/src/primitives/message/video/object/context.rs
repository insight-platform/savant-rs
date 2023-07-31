use crate::primitives::bbox::context::RBBoxFieldsView;
use crate::primitives::eval_context::EvalWithResolvers;
use crate::primitives::message::video::frame::context::FrameFieldsView;
use crate::primitives::VideoObjectProxy;
use evalexpr::*;
use hashbrown::HashMap;
use std::cell::OnceCell;

pub(crate) struct ObjectContext<'a> {
    pub object: &'a VideoObjectProxy,
    pub resolvers: Vec<String>,
    pub temp_vars: HashMap<String, Value>,
    pub object_view: ObjectFieldsView,
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

impl<'a> ObjectContext<'a> {
    pub fn new(object: &'a VideoObjectProxy, resolvers: &[String]) -> Self {
        ObjectContext {
            object,
            resolvers: resolvers.to_vec(),
            temp_vars: HashMap::new(),
            object_view: ObjectFieldsView::default(),
        }
    }
}

impl<'a> EvalWithResolvers for ObjectContext<'a> {
    fn get_resolvers(&self) -> &'_ [String] {
        self.resolvers.as_slice()
    }
}

impl<'a> Context for ObjectContext<'a> {
    fn get_value(&self, identifier: &str) -> Option<&Value> {
        if let Some(v) = self.temp_vars.get(identifier) {
            return Some(v);
        }

        match identifier {
            "id" => Some(
                self.object_view
                    .id
                    .get_or_init(|| Value::from(self.object.get_id())),
            ),
            "namespace" => Some(
                self.object_view
                    .namespace
                    .get_or_init(|| Value::from(self.object.get_namespace())),
            ),
            "label" => Some(
                self.object_view
                    .label
                    .get_or_init(|| Value::from(self.object.get_label())),
            ),
            "confidence" => {
                Some(self.object_view.confidence.get_or_init(
                    || match self.object.get_confidence() {
                        None => Value::Empty,
                        Some(c) => Value::from(c),
                    },
                ))
            }
            "parent.id" => {
                Some(
                    self.object_view
                        .parent_id
                        .get_or_init(|| match self.object.get_parent_id() {
                            None => Value::Empty,
                            Some(id) => Value::from(id),
                        }),
                )
            }
            "parent.namespace" => Some(self.object_view.parent_namespace.get_or_init(|| {
                match &self.object.get_parent() {
                    None => Value::Empty,
                    Some(parent) => Value::from(parent.get_namespace()),
                }
            })),
            "parent.label" => {
                Some(self.object_view.parent_label.get_or_init(
                    || match &self.object.get_parent() {
                        None => Value::Empty,
                        Some(parent) => Value::from(parent.get_label()),
                    },
                ))
            }

            "tracking_info.id" => Some(self.object_view.tracking_info_id.get_or_init(|| {
                match self.object.get_track_id() {
                    None => Value::Empty,
                    Some(id) => Value::from(id),
                }
            })),

            "tracking_info.bbox.xc" => {
                Some(self.object_view.tracking_info_bbox.xc.get_or_init(|| {
                    match self.object.get_track_box() {
                        None => Value::Empty,
                        Some(info) => Value::from(info.get_xc()),
                    }
                }))
            }

            "tracking_info.bbox.yc" => {
                Some(self.object_view.tracking_info_bbox.yc.get_or_init(|| {
                    match self.object.get_track_box() {
                        None => Value::Empty,
                        Some(info) => Value::from(info.get_yc()),
                    }
                }))
            }

            "tracking_info.bbox.width" => {
                Some(self.object_view.tracking_info_bbox.width.get_or_init(|| {
                    match self.object.get_track_box() {
                        None => Value::Empty,
                        Some(info) => Value::from(info.get_width()),
                    }
                }))
            }

            "tracking_info.bbox.height" => {
                Some(self.object_view.tracking_info_bbox.height.get_or_init(|| {
                    match self.object.get_track_box() {
                        None => Value::Empty,
                        Some(info) => Value::from(info.get_height()),
                    }
                }))
            }

            "tracking_info.bbox.angle" => {
                Some(self.object_view.tracking_info_bbox.angle.get_or_init(|| {
                    match self.object.get_track_box() {
                        None => Value::Empty,
                        Some(info) => match info.get_angle() {
                            None => Value::Empty,
                            Some(angle) => Value::from(angle),
                        },
                    }
                }))
            }

            "bbox.xc" => Some(
                self.object_view
                    .bbox
                    .xc
                    .get_or_init(|| Value::from(self.object.get_detection_box().get_xc())),
            ),

            "bbox.yc" => Some(
                self.object_view
                    .bbox
                    .yc
                    .get_or_init(|| Value::from(self.object.get_detection_box().get_yc())),
            ),

            "bbox.width" => Some(
                self.object_view
                    .bbox
                    .width
                    .get_or_init(|| Value::from(self.object.get_detection_box().get_width())),
            ),

            "bbox.height" => Some(
                self.object_view
                    .bbox
                    .height
                    .get_or_init(|| Value::from(self.object.get_detection_box().get_height())),
            ),

            "bbox.angle" => Some(self.object_view.bbox.angle.get_or_init(|| {
                match self.object.get_detection_box().get_angle() {
                    None => Value::Empty,
                    Some(a) => Value::from(a),
                }
            })),

            "frame.source" => {
                Some(
                    self.object_view
                        .frame
                        .source
                        .get_or_init(|| match self.object.get_frame() {
                            None => Value::Empty,
                            Some(f) => Value::from(f.get_source_id()),
                        }),
                )
            }

            "frame.rate" => {
                Some(self.object_view.frame.framerate.get_or_init(
                    || match self.object.get_frame() {
                        None => Value::Empty,
                        Some(f) => Value::from(f.get_framerate()),
                    },
                ))
            }

            "frame.width" => {
                Some(
                    self.object_view
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
                    self.object_view
                        .frame
                        .height
                        .get_or_init(|| match self.object.get_frame() {
                            None => Value::Empty,
                            Some(f) => Value::from(f.get_height()),
                        }),
                )
            }

            "frame.keyframe" => {
                Some(self.object_view.frame.keyframe.get_or_init(
                    || match self.object.get_frame() {
                        None => Value::Empty,
                        Some(f) => match f.get_keyframe() {
                            None => Value::Empty,
                            Some(kf) => Value::from(kf),
                        },
                    },
                ))
            }

            "frame.dts" => {
                Some(
                    self.object_view
                        .frame
                        .dts
                        .get_or_init(|| match self.object.get_frame() {
                            None => Value::Empty,
                            Some(f) => match f.get_dts() {
                                None => Value::Empty,
                                Some(dts) => Value::from(dts),
                            },
                        }),
                )
            }

            "frame.pts" => {
                Some(
                    self.object_view
                        .frame
                        .pts
                        .get_or_init(|| match self.object.get_frame() {
                            None => Value::Empty,
                            Some(f) => Value::from(f.get_pts()),
                        }),
                )
            }

            "frame.time_base.nominator" => {
                Some(self.object_view.frame.time_base_nominator.get_or_init(|| {
                    match self.object.get_frame() {
                        None => Value::Empty,
                        Some(f) => Value::from(f.get_time_base().0 as i64),
                    }
                }))
            }
            "frame.time_base.denominator" => Some(
                self.object_view
                    .frame
                    .time_base_denominator
                    .get_or_init(|| match self.object.get_frame() {
                        None => Value::Empty,
                        Some(f) => Value::from(f.get_time_base().1 as i64),
                    }),
            ),
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
