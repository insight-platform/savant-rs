use crate::primitives::Object;
use crate::utils::python::no_gil;
use hashbrown::HashMap;
use lazy_static::lazy_static;
use libloading::os::unix::Symbol;
use parking_lot::{const_rwlock, RwLock};
use pyo3::prelude::*;

pub type ObjectPredicateFunc = fn(o: &[&Object]) -> bool;
pub type ObjectPredicate = Symbol<ObjectPredicateFunc>;

pub type ObjectInplaceModifierFunc = fn(o: &[&Object]) -> anyhow::Result<()>;
pub type ObjectInplaceModifier = Symbol<ObjectInplaceModifierFunc>;

pub type ObjectMapModifierFunc = fn(o: &Object) -> anyhow::Result<Object>;
pub type ObjectMapModifier = Symbol<ObjectMapModifierFunc>;

#[pyclass]
#[derive(Clone, Debug)]
pub enum UserFunctionKind {
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

#[pyfunction]
#[pyo3(name = "is_plugin_function_registered")]
pub fn is_plugin_function_registered_gil(alias: String) -> bool {
    no_gil(|| is_plugin_function_registered(&alias))
}

pub fn is_plugin_function_registered(alias: &str) -> bool {
    let registry = PLUGIN_REGISTRY.read();
    registry.contains_key(alias)
}

#[pyfunction]
#[pyo3(name = "register_plugin_function")]
pub fn register_plugin_function_gil(
    plugin: String,
    function: String,
    kind: UserFunctionKind,
    alias: String,
) -> PyResult<()> {
    no_gil(|| {
        register_plugin_function(&plugin, &function, kind, &alias)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    })
}

pub fn register_plugin_function(
    plugin: &str,
    function: &str,
    kind: UserFunctionKind,
    alias: &str,
) -> anyhow::Result<()> {
    let mut registry = PLUGIN_REGISTRY.write();
    let mut lib_registry = PLUGIN_LIB_REGISTRY.write();

    if !lib_registry.contains_key(plugin) {
        let lib = unsafe { libloading::Library::new(plugin)? };
        lib_registry.insert(plugin.to_string(), lib);
    }

    let lib = lib_registry.get(plugin).unwrap();
    let byte_name = function.as_bytes();

    let func = match kind {
        UserFunctionKind::ObjectPredicate => unsafe {
            let func: libloading::Symbol<ObjectPredicateFunc> = lib.get(byte_name)?;
            UserFunction::ObjectPredicate(func.into_raw())
        },
        UserFunctionKind::ObjectInplaceModifier => unsafe {
            let func: libloading::Symbol<ObjectInplaceModifierFunc> = lib.get(byte_name)?;
            UserFunction::ObjectInplaceModifier(func.into_raw())
        },
        UserFunctionKind::ObjectMapModifier => unsafe {
            let func: libloading::Symbol<ObjectMapModifierFunc> = lib.get(byte_name)?;
            UserFunction::ObjectMapModifier(func.into_raw())
        },
    };

    registry.insert(alias.to_string(), func);
    Ok(())
}

#[pyfunction]
#[pyo3(name = "call_object_predicate")]
pub fn call_object_predicate_gil(alias: String, args: Vec<Object>) -> PyResult<bool> {
    no_gil(|| {
        call_object_predicate(&alias, args.iter().collect::<Vec<_>>().as_slice())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    })
}

pub fn call_object_predicate(alias: &str, args: &[&Object]) -> anyhow::Result<bool> {
    let registry = PLUGIN_REGISTRY.read();
    let func = match registry.get(alias) {
        Some(func) => func,
        None => panic!("Function {} not found", alias),
    };

    match func {
        UserFunction::ObjectPredicate(f) => Ok(f(args)),
        _ => panic!("Function '{}' is not an object predicate", alias),
    }
}

#[pyfunction]
#[pyo3(name = "call_object_inplace_modifier")]
pub fn call_object_inplace_modifier_gil(alias: String, args: Vec<Object>) -> PyResult<()> {
    no_gil(|| {
        call_object_inplace_modifier(&alias, args.iter().collect::<Vec<_>>().as_slice())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    })
}

pub fn call_object_inplace_modifier(alias: &str, args: &[&Object]) -> anyhow::Result<()> {
    let registry = PLUGIN_REGISTRY.read();
    let func = match registry.get(alias) {
        Some(func) => func,
        None => panic!("Function {} not found", alias),
    };

    match func {
        UserFunction::ObjectInplaceModifier(f) => Ok(f(args)?),
        _ => panic!("Function '{}' is not an inplace object modifier", alias),
    }
}

#[pyfunction]
#[pyo3(name = "call_object_map_modifier")]
pub fn call_object_map_modifier_gil(alias: String, arg: Object) -> PyResult<Object> {
    no_gil(|| {
        call_object_map_modifier(&alias, &arg)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    })
}

pub fn call_object_map_modifier(alias: &str, arg: &Object) -> anyhow::Result<Object> {
    let registry = PLUGIN_REGISTRY.read();
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
    use crate::primitives::message::video::query::Query;
    use crate::test::utils::{gen_frame, gen_object};

    #[test]
    fn pluggable_udf_api() -> anyhow::Result<()> {
        pyo3::prepare_freethreaded_python();
        register_plugin_function(
            "../target/release/libsample_plugin.so",
            "unary_op_even",
            UserFunctionKind::ObjectPredicate,
            "sample.unary_op_even",
        )?;
        register_plugin_function(
            "../target/release/libsample_plugin.so",
            "binary_op_parent",
            UserFunctionKind::ObjectPredicate,
            "sample.binary_op_parent",
        )?;

        register_plugin_function(
            "../target/release/libsample_plugin.so",
            "inplace_modifier",
            UserFunctionKind::ObjectInplaceModifier,
            "sample.inplace_modifier",
        )?;

        register_plugin_function(
            "../target/release/libsample_plugin.so",
            "map_modifier",
            UserFunctionKind::ObjectMapModifier,
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

        let f = gen_frame();
        f.delete_objects(&Query::Idle);
        let parent = gen_object(12);
        f.add_object(&parent);
        f.add_object(&o);
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
