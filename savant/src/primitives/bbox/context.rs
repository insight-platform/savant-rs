use evalexpr::Value;
use std::cell::OnceCell;

#[derive(Default)]
pub(crate) struct RBBoxFieldsView {
    pub xc: OnceCell<Value>,
    pub yc: OnceCell<Value>,
    pub width: OnceCell<Value>,
    pub height: OnceCell<Value>,
    pub angle: OnceCell<Value>,
}
