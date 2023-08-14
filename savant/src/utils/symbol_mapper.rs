use crate::release_gil;
use lazy_static::lazy_static;
use parking_lot::const_mutex;
use parking_lot::Mutex;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use savant_core::rust;
use savant_core::rust::SymbolMapper;
use std::collections::HashMap;

lazy_static! {
    static ref SYMBOL_MAPPER: Mutex<SymbolMapper> = const_mutex(SymbolMapper::default());
}

/// Defines how to act when the key is already registered.
///
/// Override
///   The key will be registered and the previous value will be overwritten.
/// ErrorIfNonUnique
///   The key will not be registered and a error will be triggered.
///
#[pyclass]
#[derive(Debug, Clone)]
pub enum RegistrationPolicy {
    Override,
    ErrorIfNonUnique,
}

impl From<RegistrationPolicy> for rust::RegistrationPolicy {
    fn from(value: RegistrationPolicy) -> Self {
        match value {
            RegistrationPolicy::Override => rust::RegistrationPolicy::Override,
            RegistrationPolicy::ErrorIfNonUnique => rust::RegistrationPolicy::ErrorIfNonUnique,
        }
    }
}

impl From<rust::RegistrationPolicy> for RegistrationPolicy {
    fn from(value: rust::RegistrationPolicy) -> Self {
        match value {
            rust::RegistrationPolicy::Override => RegistrationPolicy::Override,
            rust::RegistrationPolicy::ErrorIfNonUnique => RegistrationPolicy::ErrorIfNonUnique,
        }
    }
}

/// The function is used to fetch designated model id by a model name.
///
/// Parameters
/// ----------
/// model_name : str
///   The name of the model to fetch id for.
///
/// Returns
/// -------
/// int
///   Id of the model (int)
///
/// Raises
/// ------
/// ValueError
///   if the model is not registered
///
#[pyfunction]
#[pyo3(name = "get_model_id")]
pub fn get_model_id_py(model_name: &str) -> PyResult<i64> {
    let mut mapper = SYMBOL_MAPPER.lock();
    mapper
        .get_model_id(model_name)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

pub fn get_model_id(model_name: &str) -> anyhow::Result<i64> {
    let mut mapper = SYMBOL_MAPPER.lock();
    mapper.get_model_id(model_name)
}

/// The function is used to fetch designated object id by a model name and object label.
///
/// Parameters
/// ----------
/// model_name : str
///   The name of the model to fetch id for.
/// object_label : str
///   The label of the object to fetch id for.
///
/// Returns
/// -------
/// (int, int)
///   Id of the model (int) and id of the object (int)
///
/// Raises
/// ------
/// ValueError
///   if the object is not registered
///
#[pyfunction]
#[pyo3(name = "get_object_id")]
pub fn get_object_id_py(model_name: &str, object_label: &str) -> PyResult<(i64, i64)> {
    let mut mapper = SYMBOL_MAPPER.lock();
    mapper
        .get_object_id(model_name, object_label)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

pub fn get_object_id(model_name: &str, object_label: &str) -> anyhow::Result<(i64, i64)> {
    let mut mapper = SYMBOL_MAPPER.lock();
    mapper.get_object_id(model_name, object_label)
}

/// The function is used to register a new model and its object classes.
///
/// Parameters
/// ----------
/// model_name : str
///   The name of the model to register.
/// elements : dict[int, str]
///   The dictionary of objects in the form  id:label to register.
/// policy : :class:`RegistrationPolicy`
///   The policy to use when registering objects.
///
/// Returns
/// -------
/// int
///   Id of the model
///
/// Raises
/// ------
/// ValueError
///   if there are objects with the same IDs or labels are already registered.
///
#[pyfunction]
#[pyo3(name = "register_model_objects")]
pub fn register_model_objects_py(
    model_name: &str,
    elements: HashMap<i64, String>,
    policy: RegistrationPolicy,
) -> PyResult<i64> {
    let mut mapper = SYMBOL_MAPPER.lock();
    mapper
        .register_model_objects(model_name, &elements, &(policy.into()))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

pub fn get_model_name(model_id: i64) -> Option<String> {
    let mapper = SYMBOL_MAPPER.lock();
    mapper.get_model_name(model_id)
}

/// The function allows the fetch a model name by its id.
///
/// Parameters
/// ----------
/// model_id : int
///   The id of the model to fetch name for.
///
/// Returns
/// -------
/// str
///   Name of the model
/// None
///   If the model is not registered
///
#[pyfunction]
#[pyo3(name = "get_model_name")]
pub fn get_model_name_py(model_id: i64) -> Option<String> {
    get_model_name(model_id)
}

pub fn get_object_label(model_id: i64, object_id: i64) -> Option<String> {
    let mapper = SYMBOL_MAPPER.lock();
    mapper.get_object_label(model_id, object_id)
}

/// The function allows getting the object label by its id.
///
/// Parameters
/// ----------
/// model_id : int
///   The id of the model to fetch name for.
/// object_id : int
///   The id of the object to fetch label for.
///
/// Returns
/// -------
/// str
///   Label of the object
/// None
///   If the object is not registered
///
#[pyfunction]
#[pyo3(name = "get_object_label")]
pub fn get_object_label_py(model_id: i64, object_id: i64) -> Option<String> {
    get_object_label(model_id, object_id)
}

/// The function allows getting the object labels by their ids (bulk operation).
///
/// Parameters
/// ----------
/// model_id : int
///   The id of the model to fetch objects for.
/// object_ids : list[int]
///   The ids of the objects to fetch labels for.
///
/// Returns
/// -------
/// list[(int, str)]
///   List of tuples (object_id, object_label), if object_label is None, then the object is not registered.
///
#[pyfunction]
#[pyo3(name = "get_object_labels")]
pub fn get_object_labels_py(model_id: i64, object_ids: Vec<i64>) -> Vec<(i64, Option<String>)> {
    let mapper = SYMBOL_MAPPER.lock();
    object_ids
        .iter()
        .flat_map(|object_id| {
            mapper
                .get_object_label(model_id, *object_id)
                .map(|label| (*object_id, Some(label)))
                .or(Some((*object_id, None)))
        })
        .collect()
}

/// The function allows getting the object ids by their labels (bulk operation).
///
/// Parameters
/// ----------
/// model_name : str
///   The name of the model to fetch objects for.
/// object_labels : list[str]
///   The labels of the objects to fetch ids for.
///
/// Returns
/// -------
///   List of tuples (object_label, object_id), if object_id is None, then the object is not registered.
///
#[pyfunction]
#[pyo3(name = "get_object_ids")]
pub fn get_object_ids_py(
    model_name: &str,
    object_labels: Vec<String>,
) -> Vec<(String, Option<i64>)> {
    let mut mapper = SYMBOL_MAPPER.lock();
    object_labels
        .iter()
        .flat_map(|object_label| {
            mapper
                .get_object_id(model_name, object_label)
                .ok()
                .map(|(_model_id, object_id)| (object_label.clone(), Some(object_id)))
                .or_else(|| Some((object_label.clone(), None)))
        })
        .collect()
}

/// The function clears mapping database.
///
#[pyfunction]
#[pyo3(name = "clear_symbol_maps")]
pub fn clear_symbol_maps_py() {
    let mut mapper = SYMBOL_MAPPER.lock();
    mapper.clear();
}

/// The function allows building a model object key from model name and object label.
///
/// Parameters
/// ----------
/// model_name : str
///   The name of the model.
/// object_label : str
///   The label of the object.
///
/// Returns
/// -------
/// str
///   The model object key.
///
#[pyfunction]
#[pyo3(name = "build_model_object_key")]
pub fn build_model_object_key_py(model_name: &str, object_label: &str) -> String {
    SymbolMapper::build_model_object_key(model_name, object_label)
}

/// The function allows parsing a model object key into model name and object label.
///
/// Parameters
/// ----------
/// key : str
///   The model object key.
///
/// Returns
/// -------
/// (str, str)
///   The model name and object label.
///
/// Raises
/// ------
/// ValueError
///   If the key is not a valid key in format "model.key".
///
#[pyfunction]
#[pyo3(name = "parse_compound_key")]
pub fn parse_compound_key_py(key: &str) -> PyResult<(String, String)> {
    SymbolMapper::parse_compound_key(key).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// The function allows validating a model or object key.
///
/// Parameters
/// ----------
/// key : str
///   The model or object key.
///
/// Returns
/// -------
/// str
///   The key.
///
/// Raises
/// ------
/// ValueError
///   If the key is not a valid key in format "wordwithoutdots".
///
#[pyfunction]
#[pyo3(name = "validate_base_key")]
pub fn validate_base_key_py(key: &str) -> PyResult<String> {
    SymbolMapper::validate_base_key(key).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// The function checks if the model is registered.
///
/// Parameters
/// ----------
/// model_name : str
///   The name of the model.
///
/// Returns
/// -------
/// bool
///   True if the model is registered, False otherwise.
///
#[pyfunction]
#[pyo3(name = "is_model_registered")]
pub fn is_model_registered_py(model_name: &str) -> bool {
    let mapper = SYMBOL_MAPPER.lock();
    mapper.is_model_registered(model_name)
}

/// The function checks if the object is registered.
///
/// Parameters
/// ----------
/// model_name : str
///   The name of the model.
/// object_label : str
///   The label of the object.
///
/// Returns
/// -------
/// bool
///   True if the object is registered, False otherwise.
///
#[pyfunction]
#[pyo3(name = "is_object_registered")]
pub fn is_object_registered_py(model_name: &str, object_label: &str) -> bool {
    let mapper = SYMBOL_MAPPER.lock();
    mapper.is_object_registered(model_name, object_label)
}

/// The function dumps the registry in the form of model or object label, model_id and optional object id.
///
/// GIL management: the function is GIL-free.
///
/// Returns
/// -------
/// list[str]
///   The list of strings in the form of "model_name[.object_label] model_id Option[object_id]".
#[pyfunction]
#[pyo3(name = "dump_registry")]
pub fn dump_registry_gil() -> Vec<String> {
    release_gil!(true, || {
        let mapper = SYMBOL_MAPPER.lock();
        mapper.dump_registry()
    })
}
