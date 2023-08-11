use crate::release_gil;
use hashbrown::HashMap;
use lazy_static::lazy_static;
use parking_lot::const_mutex;
use parking_lot::Mutex;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap as StdHashMap;
use thiserror::Error;

const REGISTRY_KEY_SEPARATOR: char = '.';

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

#[derive(Error, Debug)]
pub enum Errors {
    #[error("The key `{0}` is expected to be a new one, but it already exists.")]
    DuplicateName(String),
    #[error("The key `{0}` is expected to result to (model_id, object_id), not (model_id, None).")]
    UnexpectedModelIdObjectId(String),
    #[error("The key `{0}` is expected to be fully qualified name of the form `model_name.object_label`.")]
    FullyQualifiedObjectNameParseError(String),
    #[error("The key `{0}` is expected to be a base name of the form `some-thing_name` without `.` symbols.")]
    BaseNameParseError(String),
    #[error("For model `{0}({1})` the `{2}({3})` object already exists and policy is set to `ErrorIfNonUnique`.")]
    DuplicateId(String, i64, String, i64),
}

/// The function is used to fetch designated model id by a model name.
///
/// GIL management: the function is GIL-free.
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
pub fn get_model_id_gil(model_name: String) -> PyResult<i64> {
    release_gil!(|| {
        let mut mapper = SYMBOL_MAPPER.lock();
        mapper
            .get_model_id(&model_name)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

pub fn get_model_id(model_name: &String) -> anyhow::Result<i64> {
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
pub fn get_object_id_gil(model_name: String, object_label: String) -> PyResult<(i64, i64)> {
    let mut mapper = SYMBOL_MAPPER.lock();
    mapper
        .get_object_id(&model_name, &object_label)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

pub fn get_object_id(model_name: &String, object_label: &String) -> anyhow::Result<(i64, i64)> {
    let mut mapper = SYMBOL_MAPPER.lock();
    mapper.get_object_id(model_name, object_label)
}

/// The function is used to register a new model and its object classes.
///
/// GIL management: the function is GIL-free.
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
pub fn register_model_objects_gil(
    model_name: String,
    elements: StdHashMap<i64, String>,
    policy: RegistrationPolicy,
) -> PyResult<i64> {
    release_gil!(|| {
        let mut mapper = SYMBOL_MAPPER.lock();
        mapper
            .register_model_objects(&model_name, &elements, &policy)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

pub fn get_model_name(model_id: i64) -> Option<String> {
    let mapper = SYMBOL_MAPPER.lock();
    mapper.get_model_name(model_id)
}

/// The function allows the fetch a model name by its id.
///
/// GIL management: the function is GIL-free.
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
pub fn get_model_name_gil(model_id: i64) -> Option<String> {
    release_gil!(|| get_model_name(model_id))
}

pub fn get_object_label(model_id: i64, object_id: i64) -> Option<String> {
    let mapper = SYMBOL_MAPPER.lock();
    mapper.get_object_label(model_id, object_id)
}

/// The function allows getting the object label by its id.
///
/// GIL management: the function is GIL-free.
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
pub fn get_object_label_gil(model_id: i64, object_id: i64) -> Option<String> {
    release_gil!(|| get_object_label(model_id, object_id))
}

/// The function allows getting the object labels by their ids (bulk operation).
///
/// GIL management: the function is GIL-free.
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
pub fn get_object_labels_gil(model_id: i64, object_ids: Vec<i64>) -> Vec<(i64, Option<String>)> {
    release_gil!(|| {
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
    })
}

/// The function allows getting the object ids by their labels (bulk operation).
///
/// GIL management: the function is GIL-free.
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
pub fn get_object_ids_gil(
    model_name: String,
    object_labels: Vec<String>,
) -> Vec<(String, Option<i64>)> {
    release_gil!(|| {
        let mut mapper = SYMBOL_MAPPER.lock();
        object_labels
            .iter()
            .flat_map(|object_label| {
                mapper
                    .get_object_id(&model_name, object_label)
                    .ok()
                    .map(|(_model_id, object_id)| (object_label.clone(), Some(object_id)))
                    .or_else(|| Some((object_label.clone(), None)))
            })
            .collect()
    })
}

/// The function clears mapping database.
///
/// GIL management: the function is GIL-free.
///
#[pyfunction]
#[pyo3(name = "clear_symbol_maps")]
pub fn clear_symbol_maps_gil() {
    release_gil!(|| {
        let mut mapper = SYMBOL_MAPPER.lock();
        mapper.clear();
    })
}

/// The function allows building a model object key from model name and object label.
///
/// GIL management: the function is GIL-free.
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
pub fn build_model_object_key_gil(model_name: String, object_label: String) -> String {
    release_gil!(|| SymbolMapper::build_model_object_key(&model_name, &object_label))
}

/// The function allows parsing a model object key into model name and object label.
///
/// GIL management: the function is GIL-free.
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
pub fn parse_compound_key_gil(key: String) -> PyResult<(String, String)> {
    release_gil!(|| {
        SymbolMapper::parse_compound_key(&key).map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

/// The function allows validating a model or object key.
///
/// GIL management: the function is GIL-free.
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
pub fn validate_base_key_gil(key: String) -> PyResult<String> {
    release_gil!(|| {
        SymbolMapper::validate_base_key(&key).map_err(|e| PyValueError::new_err(e.to_string()))
    })
}

/// The function checks if the model is registered.
///
/// GIL management: the function is GIL-free.
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
pub fn is_model_registered_gil(model_name: String) -> bool {
    release_gil!(|| {
        let mapper = SYMBOL_MAPPER.lock();
        mapper.is_model_registered(&model_name)
    })
}

/// The function checks if the object is registered.
///
/// GIL management: the function is GIL-free.
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
pub fn is_object_registered_gil(model_name: String, object_label: String) -> bool {
    release_gil!(|| {
        let mapper = SYMBOL_MAPPER.lock();
        mapper.is_object_registered(&model_name, &object_label)
    })
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
    release_gil!(|| {
        let mapper = SYMBOL_MAPPER.lock();
        mapper.dump_registry()
    })
}

/// In case you need it (but you don't) you can use this class to manage symbol mapping.
///
/// It is used internally as a singleton to deliver the functionality provided by the module functions.
///
#[pyclass]
#[derive(Debug, Clone)]
pub struct SymbolMapper {
    registry: HashMap<String, (i64, Option<i64>)>,
    reverse_registry: HashMap<(i64, Option<i64>), String>,
    model_next_id: i64,
    model_object_next_ids: HashMap<String, i64>,
}

impl Default for SymbolMapper {
    fn default() -> Self {
        Self {
            registry: HashMap::new(),
            reverse_registry: HashMap::new(),
            model_object_next_ids: HashMap::new(),
            model_next_id: 0,
        }
    }
}

impl SymbolMapper {
    pub fn clear(&mut self) {
        self.registry.clear();
        self.reverse_registry.clear();
        self.model_object_next_ids.clear();
        self.model_next_id = 0;
    }

    pub fn dump_registry(&self) -> Vec<String> {
        self.registry
            .iter()
            .map(|(key, (model_id, object_id))| {
                format!(
                    "Key={}, ModelId={}, ObjectId={:?}",
                    key, model_id, object_id
                )
            })
            .collect()
    }

    pub fn build_model_object_key(model_name: &String, object_label: &String) -> String {
        format!("{}{}{}", model_name, REGISTRY_KEY_SEPARATOR, object_label)
    }

    pub fn get_model_id(&mut self, model_name: &String) -> anyhow::Result<i64> {
        Self::validate_base_key(model_name)?;
        match self.registry.get(model_name) {
            None => {
                let model_id = self.gen_id();
                self.registry.insert(model_name.clone(), (model_id, None));
                self.reverse_registry
                    .insert((model_id, None), model_name.clone());
                Ok(model_id)
            }
            Some(&(model_id, None)) => Ok(model_id),
            _ => panic!("Model name must return a model id, not model and object ids"),
        }
    }

    pub fn is_model_registered(&self, model: &String) -> bool {
        self.registry.contains_key(model)
    }

    pub fn is_object_registered(&self, model: &String, label: &String) -> bool {
        let key = Self::build_model_object_key(model, label);
        self.registry.contains_key(&key)
    }

    pub fn parse_compound_key(key: &String) -> anyhow::Result<(String, String)> {
        if key.len() < 3 {
            return Err(Errors::FullyQualifiedObjectNameParseError(key.clone()).into());
        }

        let mut parts = key.split(REGISTRY_KEY_SEPARATOR);

        let model_name = parts.next();
        let object_name = parts.next();

        if parts.count() != 0 {
            return Err(Errors::FullyQualifiedObjectNameParseError(key.clone()).into());
        }

        match (model_name, object_name) {
            (Some(m), Some(o)) => {
                if !m.is_empty() && !o.is_empty() {
                    Ok((m.to_string(), o.to_string()))
                } else {
                    Err(Errors::FullyQualifiedObjectNameParseError(key.clone()).into())
                }
            }
            _ => Err(Errors::FullyQualifiedObjectNameParseError(key.clone()).into()),
        }
    }

    pub fn validate_base_key(key: &String) -> anyhow::Result<String> {
        if key.is_empty() {
            return Err(Errors::BaseNameParseError(key.clone()).into());
        }
        let parts = key.split(REGISTRY_KEY_SEPARATOR);
        if parts.count() == 1 {
            Ok(key.clone())
        } else {
            Err(Errors::BaseNameParseError(key.clone()).into())
        }
    }

    pub fn get_object_id(
        &mut self,
        model_name: &String,
        object_label: &String,
    ) -> anyhow::Result<(i64, i64)> {
        let model_id = self.get_model_id(model_name)?;
        Self::validate_base_key(object_label)?;
        let full_key = Self::build_model_object_key(model_name, object_label);

        match self.registry.get(&full_key) {
            Some((model_id, Some(object_id))) => Ok((*model_id, *object_id)),
            Some((_, None)) => Err(Errors::UnexpectedModelIdObjectId(full_key).into()),
            None => {
                let last_object_id = self
                    .model_object_next_ids
                    .get(model_name)
                    .cloned()
                    .unwrap_or(-1);
                let object_id = last_object_id + 1;
                self.registry
                    .insert(full_key.clone(), (model_id, Some(object_id)));
                self.reverse_registry
                    .insert((model_id, Some(object_id)), object_label.clone());
                self.model_object_next_ids
                    .insert(model_name.clone(), object_id);
                Ok((model_id, object_id))
            }
        }
    }

    pub fn register_model_objects(
        &mut self,
        model_name: &String,
        objects: &StdHashMap<i64, String>,
        policy: &RegistrationPolicy,
    ) -> anyhow::Result<i64> {
        let model_id = self.get_model_id(model_name)?;
        let mut last_object_id = self
            .model_object_next_ids
            .get(model_name)
            .cloned()
            .unwrap_or(-1);

        for (label_id, object_label) in objects {
            Self::validate_base_key(object_label)?;
            let key = Self::build_model_object_key(model_name, object_label);
            if matches!(policy, RegistrationPolicy::ErrorIfNonUnique) {
                if self.is_object_registered(model_name, object_label) {
                    return Err(Errors::DuplicateName(object_label.clone()).into());
                }

                if self
                    .reverse_registry
                    .contains_key(&(model_id, Some(*label_id)))
                {
                    return Err(Errors::DuplicateId(
                        model_name.clone(),
                        model_id,
                        object_label.clone(),
                        *label_id,
                    )
                    .into());
                }
            }

            self.registry
                .insert(key.clone(), (model_id, Some(*label_id)));
            self.reverse_registry
                .insert((model_id, Some(*label_id)), object_label.clone());

            if *label_id > last_object_id {
                last_object_id = *label_id;
            }
        }

        self.model_object_next_ids
            .insert(model_name.clone(), last_object_id);

        Ok(model_id)
    }
}

#[pymethods]
impl SymbolMapper {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn gen_id(&mut self) -> i64 {
        let id = self.model_next_id;
        self.model_next_id += 1;
        id
    }

    #[staticmethod]
    pub fn model_object_key_gil(model_name: String, object_label: String) -> String {
        Self::build_model_object_key(&model_name, &object_label)
    }

    #[staticmethod]
    pub fn parse_model_object_key_gil(key: String) -> PyResult<(String, String)> {
        Self::parse_compound_key(&key).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(name = "register_model")]
    pub fn register_model_gil(
        &mut self,
        model_name: String,
        elements: StdHashMap<i64, String>,
        policy: RegistrationPolicy,
    ) -> PyResult<i64> {
        release_gil!(|| self.register_model_objects(&model_name, &elements, &policy))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn get_model_name(&self, id: i64) -> Option<String> {
        self.reverse_registry.get(&(id, None)).cloned()
    }

    pub fn get_object_label(&self, model_id: i64, object_id: i64) -> Option<String> {
        self.reverse_registry
            .get(&(model_id, Some(object_id)))
            .cloned()
    }

    pub fn get_model_id_gil(&mut self, model_name: String) -> PyResult<i64> {
        self.get_model_id(&model_name)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn is_model_object_key_registered_gil(
        &self,
        model_name: String,
        object_label: String,
    ) -> bool {
        self.is_object_registered(&model_name, &object_label)
    }

    #[pyo3(name = "get_object_id")]
    pub fn get_object_id_gil(
        &mut self,
        model_name: String,
        object_label: String,
    ) -> PyResult<(i64, i64)> {
        release_gil!(|| self.get_object_id(&model_name, &object_label))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::SymbolMapper;
    use crate::test::utils::s;
    use crate::utils::symbol_mapper::{
        clear_symbol_maps_gil, get_model_id_gil, get_model_name_gil, get_object_id_gil,
        get_object_label_gil, register_model_objects_gil, RegistrationPolicy,
    };
    use serial_test::serial;

    #[test]
    fn test_validate_base_key() {
        assert!(matches!(SymbolMapper::validate_base_key(&s("")), Err(_)));

        assert!(matches!(
            SymbolMapper::validate_base_key(&s("model")),
            Ok(_)
        ));

        assert!(matches!(
            SymbolMapper::validate_base_key(&s("mo.del")),
            Err(_)
        ));

        assert!(matches!(
            SymbolMapper::validate_base_key(&s(".model")),
            Err(_)
        ));

        assert!(matches!(
            SymbolMapper::validate_base_key(&s("model.")),
            Err(_)
        ));
    }

    #[test]
    fn test_validate_object_key() {
        assert!(matches!(SymbolMapper::parse_compound_key(&s("")), Err(_)));

        assert!(matches!(
            SymbolMapper::parse_compound_key(&s("model")),
            Err(_)
        ));

        assert!(matches!(SymbolMapper::parse_compound_key(&s(".m")), Err(_)));

        assert!(matches!(SymbolMapper::parse_compound_key(&s(".")), Err(_)));

        assert!(matches!(SymbolMapper::parse_compound_key(&s("a.")), Err(_)));

        assert!(matches!(
            SymbolMapper::parse_compound_key(&s("a.b.c")),
            Err(_)
        ));

        assert!(matches!(SymbolMapper::parse_compound_key(&s("a.b")), Ok(_)));

        let (model_name, object_name) =
            SymbolMapper::parse_compound_key(&"a.b".to_string()).unwrap();
        assert_eq!(model_name, s("a"));
        assert_eq!(object_name, s("b"));
    }

    #[test]
    #[serial]
    fn register_incorrect_names() -> anyhow::Result<()> {
        clear_symbol_maps_gil();

        assert!(matches!(
            register_model_objects_gil(
                s("model."),
                [].into_iter().collect(),
                RegistrationPolicy::ErrorIfNonUnique,
            ),
            Err(_)
        ));

        assert!(matches!(
            register_model_objects_gil(
                s("model"),
                [(1, s("obj.ect"))].into_iter().collect(),
                RegistrationPolicy::ErrorIfNonUnique,
            ),
            Err(_)
        ));

        Ok(())
    }

    #[test]
    #[serial]
    fn test_register_duplicate_objects_error_non_unique() -> anyhow::Result<()> {
        clear_symbol_maps_gil();

        assert!(matches!(
            register_model_objects_gil(
                s("model"),
                [(1, s("object")), (2, s("object"))].into_iter().collect(),
                RegistrationPolicy::ErrorIfNonUnique,
            ),
            Err(_)
        ));

        clear_symbol_maps_gil();

        register_model_objects_gil(
            s("model"),
            [(1, s("object"))].into_iter().collect(),
            RegistrationPolicy::ErrorIfNonUnique,
        )?;

        assert!(matches!(
            register_model_objects_gil(
                s("model"),
                [(1, s("object2"))].into_iter().collect(),
                RegistrationPolicy::ErrorIfNonUnique,
            ),
            Err(_)
        ));

        Ok(())
    }

    #[test]
    #[serial]
    fn test_register_duplicate_objects_override() -> anyhow::Result<()> {
        clear_symbol_maps_gil();

        register_model_objects_gil(
            s("model"),
            [(1, s("object"))].into_iter().collect(),
            RegistrationPolicy::Override,
        )?;

        assert!(matches!(
            register_model_objects_gil(
                s("model"),
                [(2, s("object"))].into_iter().collect(),
                RegistrationPolicy::Override,
            ),
            Ok(0)
        ));

        assert!(matches!(
            get_object_id_gil(s("model"), s("object")),
            Ok((0, 2))
        ));

        let label = get_object_label_gil(0, 2).unwrap();
        assert_eq!(label, s("object"));

        clear_symbol_maps_gil();

        register_model_objects_gil(
            s("model"),
            [(1, s("object"))].into_iter().collect(),
            RegistrationPolicy::Override,
        )?;

        assert!(matches!(
            register_model_objects_gil(
                s("model"),
                [(1, s("object2"))].into_iter().collect(),
                RegistrationPolicy::Override,
            ),
            Ok(0)
        ));

        let label = get_object_label_gil(0, 1).unwrap();
        assert_eq!(label, s("object2"));

        Ok(())
    }

    #[test]
    #[serial]
    fn test_get_model_id() {
        clear_symbol_maps_gil();
        let model_id = get_model_id_gil(s("model")).unwrap();
        assert_eq!(model_id, 0);
    }

    #[test]
    #[serial]
    fn test_get_model_name() {
        clear_symbol_maps_gil();
        let model_id = get_model_id_gil(s("model")).unwrap();
        assert_eq!(model_id, 0);

        let model_name = get_model_name_gil(model_id).unwrap();
        assert_eq!(model_name, s("model"));

        let nonexistent_model_name = get_model_name_gil(1);
        assert!(matches!(nonexistent_model_name, None));
    }

    #[test]
    #[serial]
    fn test_get_object_label() {
        clear_symbol_maps_gil();
        let (model_id, object_id) = get_object_id_gil(s("model"), s("object")).unwrap();
        assert_eq!(model_id, 0);
        assert_eq!(object_id, 0);

        let object_label = get_object_label_gil(model_id, object_id).unwrap();
        assert_eq!(object_label, s("object"));

        let nonexistent_object_label = get_object_label_gil(0, 1);
        assert!(matches!(nonexistent_object_label, None));
    }

    #[test]
    #[serial]
    fn get_model_object_ids() {
        clear_symbol_maps_gil();
        let (model_id, object_id) = get_object_id_gil(s("model"), s("object0")).unwrap();
        assert_eq!(model_id, 0);
        assert_eq!(object_id, 0);

        let (model_id, object_id) = get_object_id_gil(s("model"), s("object1")).unwrap();
        assert_eq!(model_id, 0);
        assert_eq!(object_id, 1);

        let (model_id, object_id) = get_object_id_gil(s("model2"), s("object0")).unwrap();
        assert_eq!(model_id, 1);
        assert_eq!(object_id, 0);
    }

    #[test]
    #[serial]
    fn register_and_get_model_object_ids() -> anyhow::Result<()> {
        clear_symbol_maps_gil();
        register_model_objects_gil(
            s("model"),
            [(2, s("object0"))].into_iter().collect(),
            RegistrationPolicy::Override,
        )?;

        let (model_id, object_id) = get_object_id_gil(s("model"), s("object0")).unwrap();
        assert_eq!(model_id, 0);
        assert_eq!(object_id, 2);

        let (model_id, object_id) = get_object_id_gil(s("model"), s("object1")).unwrap();
        assert_eq!(model_id, 0);
        assert_eq!(object_id, 3);

        Ok(())
    }
}
