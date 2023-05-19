use hashbrown::HashMap;
use lazy_static::lazy_static;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap as StdHashMap;
use std::sync::Mutex;
use thiserror::Error;

const REGISTRY_KEY_SEPARATOR: char = '.';

lazy_static! {
    static ref SYMBOL_MAPPER: Mutex<SymbolMapper> = Mutex::new(SymbolMapper::default());
}

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

#[pyfunction]
#[pyo3(name = "get_model_id")]
pub fn get_model_id_py(model_name: String) -> PyResult<i64> {
    Python::with_gil(|py| {
        py.allow_threads(|| {
            let mut mapper = SYMBOL_MAPPER.lock().unwrap();
            mapper
                .get_model_id(&model_name)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    })
}

pub fn get_model_id(model_name: &String) -> anyhow::Result<i64> {
    let mut mapper = SYMBOL_MAPPER.lock().unwrap();
    mapper.get_model_id(model_name)
}

#[pyfunction]
#[pyo3(name = "get_object_id")]
pub fn get_object_id_py(model_name: String, object_label: String) -> PyResult<(i64, i64)> {
    Python::with_gil(|py| {
        py.allow_threads(|| {
            let mut mapper = SYMBOL_MAPPER.lock().unwrap();
            mapper
                .get_object_id(&model_name, &object_label)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    })
}

pub fn get_object_id(model_name: &String, object_label: &String) -> anyhow::Result<(i64, i64)> {
    let mut mapper = SYMBOL_MAPPER.lock().unwrap();
    mapper.get_object_id(model_name, object_label)
}

#[pyfunction]
pub fn register_model_objects(
    model_name: String,
    elements: StdHashMap<i64, String>,
    policy: RegistrationPolicy,
) -> PyResult<i64> {
    Python::with_gil(|py| {
        py.allow_threads(|| {
            let mut mapper = SYMBOL_MAPPER.lock().unwrap();
            mapper
                .register_model_objects(&model_name, &elements, &policy)
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    })
}

#[pyfunction]
pub fn get_model_name(model_id: i64) -> Option<String> {
    Python::with_gil(|py| {
        py.allow_threads(|| {
            let mapper = SYMBOL_MAPPER.lock().unwrap();
            mapper.get_model_name(model_id)
        })
    })
}

#[pyfunction]
pub fn get_object_label(model_id: i64, object_id: i64) -> Option<String> {
    Python::with_gil(|py| {
        py.allow_threads(|| {
            let mapper = SYMBOL_MAPPER.lock().unwrap();
            mapper.get_object_label(model_id, object_id)
        })
    })
}

#[pyfunction]
pub fn get_object_labels(model_id: i64, object_ids: Vec<i64>) -> Vec<(i64, Option<String>)> {
    Python::with_gil(|py| {
        py.allow_threads(|| {
            let mapper = SYMBOL_MAPPER.lock().unwrap();
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
    })
}

#[pyfunction]
pub fn get_object_ids(
    model_name: String,
    object_labels: Vec<String>,
) -> Vec<(String, Option<i64>)> {
    Python::with_gil(|py| {
        py.allow_threads(|| {
            let mut mapper = SYMBOL_MAPPER.lock().unwrap();
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
    })
}

#[pyfunction]
pub fn clear_symbol_maps() {
    Python::with_gil(|py| {
        py.allow_threads(|| {
            let mut mapper = SYMBOL_MAPPER.lock().unwrap();
            mapper.clear();
        })
    })
}

#[pyfunction]
pub fn build_model_object_key(model_name: String, object_label: String) -> String {
    Python::with_gil(|py| {
        py.allow_threads(|| SymbolMapper::build_model_object_key(&model_name, &object_label))
    })
}

#[pyfunction]
pub fn parse_compound_key(key: String) -> PyResult<(String, String)> {
    Python::with_gil(|py| {
        py.allow_threads(|| {
            SymbolMapper::parse_compound_key(&key).map_err(|e| PyValueError::new_err(e.to_string()))
        })
    })
}

#[pyfunction]
pub fn validate_base_key(key: String) -> PyResult<String> {
    Python::with_gil(|py| {
        py.allow_threads(|| {
            SymbolMapper::validate_base_key(&key).map_err(|e| PyValueError::new_err(e.to_string()))
        })
    })
}

#[pyfunction]
pub fn is_model_registered(model_name: String) -> bool {
    Python::with_gil(|py| {
        py.allow_threads(|| {
            let mapper = SYMBOL_MAPPER.lock().unwrap();
            mapper.is_model_registered(&model_name)
        })
    })
}

#[pyfunction]
pub fn is_object_registered(model_name: String, object_label: String) -> bool {
    Python::with_gil(|py| {
        py.allow_threads(|| {
            let mapper = SYMBOL_MAPPER.lock().unwrap();
            mapper.is_object_registered(&model_name, &object_label)
        })
    })
}

#[pyfunction]
pub fn dump_registry() -> Vec<String> {
    Python::with_gil(|py| {
        py.allow_threads(|| {
            let mapper = SYMBOL_MAPPER.lock().unwrap();
            mapper.dump_registry()
        })
    })
}

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
    pub fn model_object_key_py(model_name: String, object_label: String) -> String {
        Self::build_model_object_key(&model_name, &object_label)
    }

    #[staticmethod]
    pub fn parse_model_object_key_py(key: String) -> PyResult<(String, String)> {
        Self::parse_compound_key(&key).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn register_model_py(
        &mut self,
        model_name: String,
        elements: StdHashMap<i64, String>,
        policy: RegistrationPolicy,
    ) -> PyResult<i64> {
        Python::with_gil(|py| {
            py.allow_threads(|| self.register_model_objects(&model_name, &elements, &policy))
        })
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

    pub fn get_model_id_py(&mut self, model_name: String) -> PyResult<i64> {
        self.get_model_id(&model_name)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn is_model_object_key_registered_py(
        &self,
        model_name: String,
        object_label: String,
    ) -> bool {
        self.is_object_registered(&model_name, &object_label)
    }

    #[pyo3(name = "get_object_id")]
    pub fn get_object_id_py(
        &mut self,
        model_name: String,
        object_label: String,
    ) -> PyResult<(i64, i64)> {
        Python::with_gil(|py| py.allow_threads(|| self.get_object_id(&model_name, &object_label)))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::SymbolMapper;
    use crate::test::utils::s;
    use crate::utils::symbol_mapper::{
        clear_symbol_maps, get_model_id_py, get_model_name, get_object_id_py, get_object_label,
        register_model_objects, RegistrationPolicy,
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
        clear_symbol_maps();

        assert!(matches!(
            register_model_objects(
                s("model."),
                [].into_iter().collect(),
                RegistrationPolicy::ErrorIfNonUnique,
            ),
            Err(_)
        ));

        assert!(matches!(
            register_model_objects(
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
        clear_symbol_maps();

        assert!(matches!(
            register_model_objects(
                s("model"),
                [(1, s("object")), (2, s("object"))].into_iter().collect(),
                RegistrationPolicy::ErrorIfNonUnique,
            ),
            Err(_)
        ));

        clear_symbol_maps();

        register_model_objects(
            s("model"),
            [(1, s("object"))].into_iter().collect(),
            RegistrationPolicy::ErrorIfNonUnique,
        )?;

        assert!(matches!(
            register_model_objects(
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
        clear_symbol_maps();

        register_model_objects(
            s("model"),
            [(1, s("object"))].into_iter().collect(),
            RegistrationPolicy::Override,
        )?;

        assert!(matches!(
            register_model_objects(
                s("model"),
                [(2, s("object"))].into_iter().collect(),
                RegistrationPolicy::Override,
            ),
            Ok(0)
        ));

        assert!(matches!(
            get_object_id_py(s("model"), s("object")),
            Ok((0, 2))
        ));

        let label = get_object_label(0, 2).unwrap();
        assert_eq!(label, s("object"));

        clear_symbol_maps();

        register_model_objects(
            s("model"),
            [(1, s("object"))].into_iter().collect(),
            RegistrationPolicy::Override,
        )?;

        assert!(matches!(
            register_model_objects(
                s("model"),
                [(1, s("object2"))].into_iter().collect(),
                RegistrationPolicy::Override,
            ),
            Ok(0)
        ));

        let label = get_object_label(0, 1).unwrap();
        assert_eq!(label, s("object2"));

        Ok(())
    }

    #[test]
    #[serial]
    fn test_get_model_id() {
        clear_symbol_maps();
        let model_id = get_model_id_py(s("model")).unwrap();
        assert_eq!(model_id, 0);
    }

    #[test]
    #[serial]
    fn test_get_model_name() {
        clear_symbol_maps();
        let model_id = get_model_id_py(s("model")).unwrap();
        assert_eq!(model_id, 0);

        let model_name = get_model_name(model_id).unwrap();
        assert_eq!(model_name, s("model"));

        let nonexistent_model_name = get_model_name(1);
        assert!(matches!(nonexistent_model_name, None));
    }

    #[test]
    #[serial]
    fn test_get_object_label() {
        clear_symbol_maps();
        let (model_id, object_id) = get_object_id_py(s("model"), s("object")).unwrap();
        assert_eq!(model_id, 0);
        assert_eq!(object_id, 0);

        let object_label = get_object_label(model_id, object_id).unwrap();
        assert_eq!(object_label, s("object"));

        let nonexistent_object_label = get_object_label(0, 1);
        assert!(matches!(nonexistent_object_label, None));
    }

    #[test]
    #[serial]
    fn get_model_object_ids() {
        clear_symbol_maps();
        let (model_id, object_id) = get_object_id_py(s("model"), s("object0")).unwrap();
        assert_eq!(model_id, 0);
        assert_eq!(object_id, 0);

        let (model_id, object_id) = get_object_id_py(s("model"), s("object1")).unwrap();
        assert_eq!(model_id, 0);
        assert_eq!(object_id, 1);

        let (model_id, object_id) = get_object_id_py(s("model2"), s("object0")).unwrap();
        assert_eq!(model_id, 1);
        assert_eq!(object_id, 0);
    }

    #[test]
    #[serial]
    fn register_and_get_model_object_ids() -> anyhow::Result<()> {
        clear_symbol_maps();
        register_model_objects(
            s("model"),
            [(2, s("object0"))].into_iter().collect(),
            RegistrationPolicy::Override,
        )?;

        let (model_id, object_id) = get_object_id_py(s("model"), s("object0")).unwrap();
        assert_eq!(model_id, 0);
        assert_eq!(object_id, 2);

        let (model_id, object_id) = get_object_id_py(s("model"), s("object1")).unwrap();
        assert_eq!(model_id, 0);
        assert_eq!(object_id, 3);

        Ok(())
    }
}
