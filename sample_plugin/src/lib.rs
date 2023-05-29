use savant_rs::primitives::attribute::Attributive;
use savant_rs::primitives::Object;
use std::sync::Arc;

#[no_mangle]
pub extern "C" fn binary_op_parent(left: &Object, right: &Object) -> bool {
    let left_inner = left.get_inner();
    let right_inner = right.get_inner();
    if Arc::ptr_eq(&left_inner, &right_inner) {
        false
    } else {
        left.get_parent().is_some()
            && left
                .get_parent()
                .map(|p| p.object().get_id() == right.get_id())
                .unwrap_or(false)
    }
}

#[no_mangle]
pub extern "C" fn unary_op_even(o: &Object) -> bool {
    o.get_id() % 2 == 0
}
