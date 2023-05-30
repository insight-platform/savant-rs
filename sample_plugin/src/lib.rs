use savant_rs::primitives::attribute::Attributive;
use savant_rs::primitives::Object;
use std::sync::Arc;

#[no_mangle]
pub fn binary_op_parent(objs: &[&Object]) -> bool {
    assert_eq!(objs.len(), 2, "Expected 2 objects, got {}", objs.len());
    let left = objs[0];
    let right = objs[1];
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
pub fn unary_op_even(objs: &[&Object]) -> bool {
    assert_eq!(objs.len(), 1, "Expected 1 object, got {}", objs.len());
    let o = objs[0];
    o.get_id() % 2 == 0
}

#[no_mangle]
pub fn inplace_modifier(objs: &[&Object]) -> anyhow::Result<()> {
    for obj in objs {
        let label = obj.get_label();
        obj.set_label(format!("modified_{}", label));
    }

    Ok(())
}

#[no_mangle]
pub fn map_modifier(obj: &Object) -> anyhow::Result<Object> {
    let label = obj.get_label();
    let new_obj = obj.clean_copy();
    new_obj.set_label(format!("modified_{}", label));
    Ok(new_obj)
}
