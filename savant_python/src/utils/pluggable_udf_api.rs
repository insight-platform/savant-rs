use crate::primitives::object::VideoObject;
use crate::release_gil;
use pyo3::prelude::*;
use savant_core::pluggable_udf_api as rust;

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

impl From<UserFunctionType> for rust::UserFunctionType {
    fn from(py_type: UserFunctionType) -> Self {
        match py_type {
            UserFunctionType::ObjectPredicate => rust::UserFunctionType::ObjectPredicate,
            UserFunctionType::ObjectInplaceModifier => {
                rust::UserFunctionType::ObjectInplaceModifier
            }
            UserFunctionType::ObjectMapModifier => rust::UserFunctionType::ObjectMapModifier,
        }
    }
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
pub fn is_plugin_function_registered(alias: &str) -> bool {
    rust::is_plugin_function_registered(alias)
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
pub fn register_plugin_function(
    plugin: &str,
    function: &str,
    function_type: UserFunctionType,
    alias: &str,
) -> PyResult<()> {
    rust::register_plugin_function(plugin, function, &(function_type.into()), alias)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
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
#[pyo3(signature = (alias, args, no_gil = true))]
pub fn call_object_predicate(alias: &str, args: Vec<VideoObject>, no_gil: bool) -> PyResult<bool> {
    release_gil!(no_gil, || {
        rust::call_object_predicate(
            alias,
            args.iter().map(|o| &o.0).collect::<Vec<_>>().as_slice(),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    })
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
#[pyo3(signature = (alias, args, no_gil = true))]
pub fn call_object_inplace_modifier(
    alias: &str,
    args: Vec<VideoObject>,
    no_gil: bool,
) -> PyResult<()> {
    release_gil!(no_gil, || {
        rust::call_object_inplace_modifier(
            alias,
            args.iter().map(|o| &o.0).collect::<Vec<_>>().as_slice(),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    })
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
#[pyo3(signature = (alias, arg, no_gil = true))]
pub fn call_object_map_modifier(
    alias: &str,
    arg: &VideoObject,
    no_gil: bool,
) -> PyResult<VideoObject> {
    release_gil!(no_gil, || {
        rust::call_object_map_modifier(alias, &arg.0)
            .map(VideoObject)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    })
}
