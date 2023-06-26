use evalexpr::Value;
use std::cell::OnceCell;

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
