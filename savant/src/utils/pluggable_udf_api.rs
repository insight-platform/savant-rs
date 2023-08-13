use crate::primitives::VideoObjectProxy;
use crate::release_gil;
use hashbrown::HashMap;
use lazy_static::lazy_static;
use libloading::os::unix::Symbol;
use parking_lot::{const_rwlock, RwLock};
use pyo3::prelude::*;

pub type ObjectPredicateFunc = fn(o: &[&VideoObjectProxy]) -> bool;
pub type ObjectPredicate = Symbol<ObjectPredicateFunc>;

pub type ObjectInplaceModifierFunc = fn(o: &[&VideoObjectProxy]) -> anyhow::Result<()>;
pub type ObjectInplaceModifier = Symbol<ObjectInplaceModifierFunc>;

pub type ObjectMapModifierFunc = fn(o: &VideoObjectProxy) -> anyhow::Result<VideoObjectProxy>;
pub type ObjectMapModifier = Symbol<ObjectMapModifierFunc>;

/// Determines the type of user function.
///
/// ObjectPredicate
///   A function that takes a slice of objects and returns a boolean.
///
/// ObjectInplaceModifier
///   A function that takes a slice of objects and modifies them in place.
///
/// ObjectMapModifier
///   A function that takes an object and returns a new object.
///
#[pyclass]
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

/// Checks if a plugin function is registered for alias.
///
/// GIL Management: This function is GIL-free.
///
/// Parameters
/// ----------
/// alias : str
///   The alias of the plugin function.
///
#[pyfunction]
#[pyo3(name = "is_plugin_function_registered")]
pub fn is_plugin_function_registered_py(alias: String) -> bool {
    is_plugin_function_registered(&alias)
}

pub fn is_plugin_function_registered(alias: &str) -> bool {
    let registry = PLUGIN_REGISTRY.read();
    registry.contains_key(alias)
}

/// Allows registering a plugin function.
///
/// GIL Management: This function is GIL-free.
///
/// Parameters
/// ----------
/// plugin : str
///   The path to the plugin library.
/// function : str
///   The name of the function to register.
/// function_type : UserFunctionType
///   The type of the function to register.
/// alias : str
///   The alias to register the function under.
///
/// Raises
/// ------
/// PyRuntimeError
///   If the function cannot be registered.
///
#[pyfunction]
#[pyo3(name = "register_plugin_function")]
pub fn register_plugin_function_py(
    plugin: String,
    function: String,
    function_type: &UserFunctionType,
    alias: String,
) -> PyResult<()> {
    register_plugin_function(&plugin, &function, function_type, &alias)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

pub fn register_plugin_function(
    plugin: &str,
    function: &str,
    kind: &UserFunctionType,
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

/// Invokes a registered plugin function of the :class:`UserFunctionType.ObjectPredicate` type.
///
/// GIL Management: This function is GIL-free.
///
/// Parameters
/// ----------
/// alias : str
///   The alias of the plugin function.
/// args : List[savant_rs.primitives.Object]
///   The arguments to pass to the plugin function.
///
/// Returns
/// -------
/// bool
///   The result of the plugin function.
///
/// Raises
/// ------
/// PyRuntimeError
///   If the plugin function cannot be invoked.
///
#[pyfunction]
#[pyo3(name = "call_object_predicate")]
#[pyo3(signature = (alias, args, no_gil = true))]
pub fn call_object_predicate_gil(
    alias: String,
    args: Vec<VideoObjectProxy>,
    no_gil: bool,
) -> PyResult<bool> {
    release_gil!(no_gil, || {
        call_object_predicate(&alias, args.iter().collect::<Vec<_>>().as_slice())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    })
}

pub fn call_object_predicate(alias: &str, args: &[&VideoObjectProxy]) -> anyhow::Result<bool> {
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

/// Invokes a registered plugin function of the :class:`UserFunctionType.ObjectInplaceModifier`
///
/// GIL Management: This function is GIL-free.
///
/// Parameters
/// ----------
/// alias : str
///   The alias of the plugin function.
/// args : List[savant_rs.primitives.Object]
///   The arguments to pass to the plugin function.
///
/// Raises
/// ------
/// PyRuntimeError
///   If the plugin function cannot be invoked.
///
#[pyfunction]
#[pyo3(name = "call_object_inplace_modifier")]
#[pyo3(signature = (alias, args, no_gil = true))]
pub fn call_object_inplace_modifier_gil(
    alias: String,
    args: Vec<VideoObjectProxy>,
    no_gil: bool,
) -> PyResult<()> {
    release_gil!(no_gil, || {
        call_object_inplace_modifier(&alias, args.iter().collect::<Vec<_>>().as_slice())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    })
}

pub fn call_object_inplace_modifier(alias: &str, args: &[&VideoObjectProxy]) -> anyhow::Result<()> {
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

/// Invokes a registered plugin function of the :class:`UserFunctionType.ObjectMapModifier`
///
/// GIL Management: This function is GIL-free.
///
/// Parameters
/// ----------
/// alias : str
///   The alias of the plugin function.
/// arg : savant_rs.primitives.Object
///   The argument to pass to the plugin function.
///
/// Returns
/// -------
/// savant_rs.primitives.Object
///   Resulting object
///
/// Raises
/// ------
/// PyRuntimeError
///   If the plugin function cannot be invoked.
///
#[pyfunction]
#[pyo3(name = "call_object_map_modifier")]
#[pyo3(signature = (alias, arg, no_gil = true))]
pub fn call_object_map_modifier_gil(
    alias: String,
    arg: &VideoObjectProxy,
    no_gil: bool,
) -> PyResult<VideoObjectProxy> {
    release_gil!(no_gil, || {
        call_object_map_modifier(&alias, arg)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    })
}

pub fn call_object_map_modifier(
    alias: &str,
    arg: &VideoObjectProxy,
) -> anyhow::Result<VideoObjectProxy> {
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
    use crate::primitives::IdCollisionResolutionPolicy;
    use crate::test::utils::{gen_empty_frame, gen_object};

    #[test]
    fn pluggable_udf_api() -> anyhow::Result<()> {
        pyo3::prepare_freethreaded_python();
        register_plugin_function(
            "../target/debug/libsavant_rs.so",
            "unary_op_even",
            &UserFunctionType::ObjectPredicate,
            "sample.unary_op_even",
        )?;
        register_plugin_function(
            "../target/debug/libsavant_rs.so",
            "binary_op_parent",
            &UserFunctionType::ObjectPredicate,
            "sample.binary_op_parent",
        )?;

        register_plugin_function(
            "../target/debug/libsavant_rs.so",
            "inplace_modifier",
            &UserFunctionType::ObjectInplaceModifier,
            "sample.inplace_modifier",
        )?;

        register_plugin_function(
            "../target/debug/libsavant_rs.so",
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
