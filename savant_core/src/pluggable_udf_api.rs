use crate::primitives::object::VideoObjectProxy;
use crate::trace;
use hashbrown::HashMap;
use lazy_static::lazy_static;
use libloading::os::unix::Symbol;
use parking_lot::{const_rwlock, RwLock};

pub type ObjectPredicateFunc = fn(o: &[&VideoObjectProxy]) -> bool;
pub type ObjectPredicate = Symbol<ObjectPredicateFunc>;

pub type ObjectInplaceModifierFunc = fn(o: &[&VideoObjectProxy]) -> anyhow::Result<()>;
pub type ObjectInplaceModifier = Symbol<ObjectInplaceModifierFunc>;

pub type ObjectMapModifierFunc = fn(o: &VideoObjectProxy) -> anyhow::Result<VideoObjectProxy>;
pub type ObjectMapModifier = Symbol<ObjectMapModifierFunc>;
#[derive(Clone, Debug)]
pub enum UserFunctionType {
    ObjectPredicate,
    ObjectInplaceModifier,
    ObjectMapModifier,
}

#[derive(Clone, Debug)]
pub enum UserFunction {
    ObjectPredicate(ObjectPredicate),
    ObjectInplaceModifier(ObjectInplaceModifier),
    ObjectMapModifier(ObjectMapModifier),
}

lazy_static! {
    static ref PLUGIN_REGISTRY: RwLock<HashMap<String, UserFunction>> =
        const_rwlock(HashMap::new());
    static ref PLUGIN_LIB_REGISTRY: RwLock<HashMap<String, libloading::Library>> =
        const_rwlock(HashMap::new());
}

pub fn is_plugin_function_registered(alias: &str) -> bool {
    let registry = trace!(PLUGIN_REGISTRY.read());
    registry.contains_key(alias)
}

pub fn register_plugin_function(
    plugin: &str,
    function: &str,
    kind: &UserFunctionType,
    alias: &str,
) -> anyhow::Result<()> {
    let mut registry = trace!(PLUGIN_REGISTRY.write());
    let mut lib_registry = trace!(PLUGIN_LIB_REGISTRY.write());

    if !lib_registry.contains_key(plugin) {
        let lib = unsafe { libloading::Library::new(plugin)? };
        lib_registry.insert(plugin.to_string(), lib);
    }

    let lib = lib_registry.get(plugin).unwrap();
    let byte_name = function.as_bytes();

    let func = match kind {
        UserFunctionType::ObjectPredicate => unsafe {
            let func: libloading::Symbol<ObjectPredicateFunc> = lib.get(byte_name)?;
            UserFunction::ObjectPredicate(func.into_raw())
        },
        UserFunctionType::ObjectInplaceModifier => unsafe {
            let func: libloading::Symbol<ObjectInplaceModifierFunc> = lib.get(byte_name)?;
            UserFunction::ObjectInplaceModifier(func.into_raw())
        },
        UserFunctionType::ObjectMapModifier => unsafe {
            let func: libloading::Symbol<ObjectMapModifierFunc> = lib.get(byte_name)?;
            UserFunction::ObjectMapModifier(func.into_raw())
        },
    };

    registry.insert(alias.to_string(), func);
    Ok(())
}

pub fn call_object_predicate(alias: &str, args: &[&VideoObjectProxy]) -> anyhow::Result<bool> {
    let registry = trace!(PLUGIN_REGISTRY.read());
    let func = match registry.get(alias) {
        Some(func) => func,
        None => panic!("Function {} not found", alias),
    };

    match func {
        UserFunction::ObjectPredicate(f) => Ok(f(args)),
        _ => panic!("Function '{}' is not an object predicate", alias),
    }
}

pub fn call_object_inplace_modifier(alias: &str, args: &[&VideoObjectProxy]) -> anyhow::Result<()> {
    let registry = trace!(PLUGIN_REGISTRY.read());
    let func = match registry.get(alias) {
        Some(func) => func,
        None => panic!("Function {} not found", alias),
    };

    match func {
        UserFunction::ObjectInplaceModifier(f) => Ok(f(args)?),
        _ => panic!("Function '{}' is not an inplace object modifier", alias),
    }
}

pub fn call_object_map_modifier(
    alias: &str,
    arg: &VideoObjectProxy,
) -> anyhow::Result<VideoObjectProxy> {
    let registry = trace!(PLUGIN_REGISTRY.read());
    let func = match registry.get(alias) {
        Some(func) => func,
        None => panic!("Function {} not found", alias),
    };

    match func {
        UserFunction::ObjectMapModifier(f) => Ok(f(arg)?),
        _ => panic!("Function '{}' is not an map object modifier", alias),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::object::IdCollisionResolutionPolicy;
    use crate::test::{gen_empty_frame, gen_object};

    #[test]
    fn pluggable_udf_api() -> anyhow::Result<()> {
        register_plugin_function(
            "../target/debug/libsavant_core.so",
            "unary_op_even",
            &UserFunctionType::ObjectPredicate,
            "sample.unary_op_even",
        )?;
        register_plugin_function(
            "../target/debug/libsavant_core.so",
            "binary_op_parent",
            &UserFunctionType::ObjectPredicate,
            "sample.binary_op_parent",
        )?;

        register_plugin_function(
            "../target/debug/libsavant_core.so",
            "inplace_modifier",
            &UserFunctionType::ObjectInplaceModifier,
            "sample.inplace_modifier",
        )?;

        register_plugin_function(
            "../target/debug/libsavant_core.so",
            "map_modifier",
            &UserFunctionType::ObjectMapModifier,
            "sample.map_modifier",
        )?;

        let o = gen_object(12);
        assert!(call_object_predicate("sample.unary_op_even", &[&o])?);

        let o = gen_object(13);
        assert!(!call_object_predicate("sample.unary_op_even", &[&o])?);

        assert!(!call_object_predicate(
            "sample.binary_op_parent",
            &[&o, &o]
        )?);

        let f = gen_empty_frame();
        let parent = gen_object(12);
        f.add_object(&parent, IdCollisionResolutionPolicy::Error)
            .unwrap();
        f.add_object(&o, IdCollisionResolutionPolicy::Error)
            .unwrap();
        o.set_parent(Some(parent.get_id()));

        assert!(call_object_predicate(
            "sample.binary_op_parent",
            &[&o, &parent]
        )?);

        let o = gen_object(12);
        call_object_inplace_modifier("sample.inplace_modifier", &[&o])?;
        let label = o.get_label();
        assert!(label.starts_with("modified"));

        let o = gen_object(12);
        let o = call_object_map_modifier("sample.map_modifier", &o)?;
        let label = o.get_label();
        assert!(label.starts_with("modified"));

        Ok(())
    }
}
