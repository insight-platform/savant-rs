use crate::primitives::VideoObjectProxy;
use crate::utils::eval_resolvers::get_symbol_resolver;
use evalexpr::*;
use hashbrown::HashMap;
use std::cell::OnceCell;

pub struct ObjectContext<'a> {
    object: &'a VideoObjectProxy,
    resolvers: &'a [String],
    temp_vars: HashMap<String, Value>,
    object_view: ObjectFieldsView,
}

#[derive(Default)]
struct ObjectFieldsView {
    id: OnceCell<Value>,
    creator: OnceCell<Value>,
    label: OnceCell<Value>,
    confidence: OnceCell<Value>,

    tracking_id: OnceCell<Value>,
    tracking_bbox_xc: OnceCell<Value>,
    tracking_bbox_yc: OnceCell<Value>,
    tracking_bbox_width: OnceCell<Value>,
    tracking_bbox_height: OnceCell<Value>,
    tracking_bbox_angle: OnceCell<Value>,

    box_xc: OnceCell<Value>,
    box_yc: OnceCell<Value>,
    box_width: OnceCell<Value>,
    box_height: OnceCell<Value>,
    box_angle: OnceCell<Value>,

    parent_id: OnceCell<Value>,
    parent_creator: OnceCell<Value>,
    parent_label: OnceCell<Value>,
}

impl<'a> ObjectContext<'a> {
    pub fn new(object: &'a VideoObjectProxy, resolvers: &'a [String]) -> Self {
        ObjectContext {
            object,
            resolvers,
            temp_vars: HashMap::new(),
            object_view: ObjectFieldsView::default(),
        }
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
            "creator" => Some(
                self.object_view
                    .creator
                    .get_or_init(|| Value::from(self.object.get_creator())),
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
            "parent.creator" => {
                Some(self.object_view.parent_creator.get_or_init(
                    || match &self.object.get_parent() {
                        None => Value::Empty,
                        Some(parent) => Value::from(parent.get_creator()),
                    },
                ))
            }
            "parent.label" => {
                Some(self.object_view.parent_label.get_or_init(
                    || match &self.object.get_parent() {
                        None => Value::Empty,
                        Some(parent) => Value::from(parent.get_label()),
                    },
                ))
            }

            "tracking_info.id" => Some(self.object_view.tracking_id.get_or_init(|| {
                match self.object.get_tracking_data() {
                    None => Value::Empty,
                    Some(info) => Value::from(info.id),
                }
            })),

            "tracking_info.bbox.xc" => Some(self.object_view.tracking_bbox_xc.get_or_init(|| {
                match self.object.get_tracking_data() {
                    None => Value::Empty,
                    Some(info) => Value::from(info.bounding_box.get_xc()),
                }
            })),

            "tracking_info.bbox.yc" => Some(self.object_view.tracking_bbox_yc.get_or_init(|| {
                match self.object.get_tracking_data() {
                    None => Value::Empty,
                    Some(info) => Value::from(info.bounding_box.get_yc()),
                }
            })),

            "tracking_info.bbox.width" => {
                Some(self.object_view.tracking_bbox_width.get_or_init(|| {
                    match self.object.get_tracking_data() {
                        None => Value::Empty,
                        Some(info) => Value::from(info.bounding_box.get_width()),
                    }
                }))
            }

            "tracking_info.bbox.height" => {
                Some(self.object_view.tracking_bbox_height.get_or_init(|| {
                    match self.object.get_tracking_data() {
                        None => Value::Empty,
                        Some(info) => Value::from(info.bounding_box.get_height()),
                    }
                }))
            }

            "tracking_info.bbox.angle" => {
                Some(self.object_view.tracking_bbox_angle.get_or_init(|| {
                    match self.object.get_tracking_data() {
                        None => Value::Empty,
                        Some(info) => match info.bounding_box.get_angle() {
                            None => Value::Empty,
                            Some(angle) => Value::from(angle),
                        },
                    }
                }))
            }

            "bbox.xc" => Some(
                self.object_view
                    .box_xc
                    .get_or_init(|| Value::from(self.object.get_bbox().get_xc())),
            ),

            "bbox.yc" => Some(
                self.object_view
                    .box_yc
                    .get_or_init(|| Value::from(self.object.get_bbox().get_yc())),
            ),

            "bbox.width" => Some(
                self.object_view
                    .box_width
                    .get_or_init(|| Value::from(self.object.get_bbox().get_width())),
            ),

            "bbox.height" => Some(
                self.object_view
                    .box_height
                    .get_or_init(|| Value::from(self.object.get_bbox().get_height())),
            ),

            "bbox.angle" => Some(self.object_view.box_angle.get_or_init(|| {
                match self.object.get_bbox().get_angle() {
                    None => Value::Empty,
                    Some(a) => Value::from(a),
                }
            })),

            _ => None,
        }
    }

    fn call_function(&self, identifier: &str, argument: &Value) -> EvalexprResult<Value> {
        let res = get_symbol_resolver(identifier);
        match res {
            Some((r, executor)) => {
                if self.resolvers.contains(&r) {
                    executor
                        .resolve(identifier, argument)
                        .map_err(|e| EvalexprError::CustomMessage(e.to_string()))
                } else {
                    Err(EvalexprError::FunctionIdentifierNotFound(
                        identifier.to_string(),
                    ))
                }
            }
            None => Err(EvalexprError::FunctionIdentifierNotFound(
                identifier.to_string(),
            )),
        }
    }

    fn are_builtin_functions_disabled(&self) -> bool {
        false
    }

    fn set_builtin_functions_disabled(&mut self, _: bool) -> EvalexprResult<()> {
        Err(EvalexprError::BuiltinFunctionsCannotBeDisabled)
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
        self.temp_vars.insert(identifier.to_string(), value);
        Ok(())
    }
}
