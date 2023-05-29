use crate::primitives::Object;
use crate::utils::python::no_gil;
use hashbrown::HashMap;
use lazy_static::lazy_static;
use libloading::os::unix::Symbol;
use parking_lot::{const_rwlock, RwLock};
use pyo3::prelude::*;

pub type UnaryObjectPredicateFunc = extern "C" fn(o: &Object) -> bool;
pub type UnaryObjectPredicate = Symbol<UnaryObjectPredicateFunc>;

pub type BinaryObjectMatchPredicateFunc = extern "C" fn(left: &Object, right: &Object) -> bool;
pub type BinaryObjectMatchPredicate = Symbol<BinaryObjectMatchPredicateFunc>;

#[pyclass]
#[derive(Clone, Debug)]
pub enum UserFunctionKind {
    UnaryObjectPredicate,
    BinaryObjectMatchPredicate,
}

#[derive(Clone, Debug)]
pub enum UserFunction {
    UnaryObjectPredicate(UnaryObjectPredicate),
    BinaryObjectMatchPredicate(BinaryObjectMatchPredicate),
}

lazy_static! {
    static ref PLUGIN_REGISTRY: RwLock<HashMap<String, UserFunction>> =
        const_rwlock(HashMap::new());
    static ref PLUGIN_LIB_REGISTRY: RwLock<HashMap<String, libloading::Library>> =
        const_rwlock(HashMap::new());
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
        UserFunctionKind::UnaryObjectPredicate => unsafe {
            let func: libloading::Symbol<UnaryObjectPredicateFunc> = lib.get(byte_name)?;
            UserFunction::UnaryObjectPredicate(func.into_raw())
        },
        UserFunctionKind::BinaryObjectMatchPredicate => unsafe {
            let func: libloading::Symbol<BinaryObjectMatchPredicateFunc> = lib.get(byte_name)?;
            UserFunction::BinaryObjectMatchPredicate(func.into_raw())
        },
    };

    registry.insert(alias.to_string(), func);
    Ok(())
}

#[pyfunction]
#[pyo3(name = "call_boolean")]
pub fn call_boolean_gil(alias: String, args: Vec<Object>) -> PyResult<bool> {
    no_gil(|| {
        call_boolean(&alias, args.iter().collect::<Vec<_>>().as_slice())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    })
}

pub fn call_boolean(alias: &str, args: &[&Object]) -> anyhow::Result<bool> {
    let registry = PLUGIN_REGISTRY.read();
    let func = match registry.get(alias) {
        Some(func) => func,
        None => panic!("Function {} not found", alias),
    };

    match func {
        UserFunction::UnaryObjectPredicate(f) => {
            let arg = args.get(0).expect("Unary predicate requires one argument");
            Ok(f(arg))
        }
        UserFunction::BinaryObjectMatchPredicate(f) => {
            if args.len() != 2 {
                panic!("Binary predicate requires two arguments");
            }
            Ok(f(args[0], args[1]))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test::utils::gen_object;

    #[test]
    fn test_plugin_api_2() -> anyhow::Result<()> {
        pyo3::prepare_freethreaded_python();
        register_plugin_function(
            "../target/release/libsample_plugin.so",
            "unary_op_even",
            UserFunctionKind::UnaryObjectPredicate,
            "sample.unary_op_even",
        )?;
        register_plugin_function(
            "../target/release/libsample_plugin.so",
            "binary_op_parent",
            UserFunctionKind::BinaryObjectMatchPredicate,
            "sample.binary_op_parent",
        )?;

        let o = gen_object(12);
        assert!(call_boolean("sample.unary_op_even", &[&o])?);

        let o = gen_object(13);
        assert!(!call_boolean("sample.unary_op_even", &[&o])?);

        assert!(!call_boolean("sample.binary_op_parent", &[&o, &o])?);

        let o2 = gen_object(12);
        o.set_parent(Some(o2.clone()));
        assert!(call_boolean("sample.binary_op_parent", &[&o, &o2])?);

        Ok(())
    }
}
