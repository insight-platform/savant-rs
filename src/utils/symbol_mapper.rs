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
}

#[pyfunction]
pub fn get_model_id(model_name: String) -> PyResult<i64> {
    let mut mapper = SYMBOL_MAPPER.lock().unwrap();
    mapper
        .get_model_id(&model_name)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
pub fn get_object_id(model_name: String, object_label: String) -> PyResult<(i64, i64)> {
    let mut mapper = SYMBOL_MAPPER.lock().unwrap();
    mapper
        .get_object_id(&model_name, &object_label)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
pub fn register_model_objects(
    model_name: String,
    elements: StdHashMap<i64, String>,
) -> PyResult<i64> {
    let mut mapper = SYMBOL_MAPPER.lock().unwrap();
    mapper
        .register_model_objects(&model_name, &elements)
        .map_err(|e| PyValueError::new_err(e.to_string()))
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
    pub fn model_object_key(model_name: &String, object_label: &String) -> String {
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

    pub fn is_model_object_key_registered(&self, key: &String) -> bool {
        self.registry.contains_key(key)
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
                if m.len() > 0 && o.len() > 0 {
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
        let full_key = Self::model_object_key(model_name, &object_label);

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
                    .insert((model_id, Some(object_id)), full_key);
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
    ) -> anyhow::Result<i64> {
        let model_id = self.get_model_id(model_name)?;
        let mut last_object_id = self
            .model_object_next_ids
            .get(model_name)
            .cloned()
            .unwrap_or(-1);

        for (label_id, object_label) in objects {
            Self::validate_base_key(object_label)?;
            let key = Self::model_object_key(model_name, &object_label);
            if self.is_model_object_key_registered(&key) {
                return Err(Errors::DuplicateName(object_label.clone()).into());
            }

            self.registry
                .insert(key.clone(), (model_id, Some(*label_id)));
            self.reverse_registry
                .insert((model_id, Some(*label_id)), key);

            if *label_id > last_object_id {
                last_object_id = *label_id;
            }
        }

        let v = self.model_object_next_ids.get_mut(model_name).unwrap();
        *v += 1;

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
        Self::model_object_key(&model_name, &object_label)
    }

    #[staticmethod]
    pub fn parse_model_object_key_py(key: String) -> PyResult<(String, String)> {
        Self::parse_compound_key(&key).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn register_model_py(
        &mut self,
        model_name: String,
        elements: StdHashMap<i64, String>,
    ) -> PyResult<i64> {
        Python::with_gil(|py| {
            py.allow_threads(|| self.register_model_objects(&model_name, &elements))
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn get_model_name(&self, id: i64) -> Option<String> {
        self.reverse_registry.get(&(id, None)).cloned()
    }

    pub fn get_object_name(&self, model_id: i64, object_id: i64) -> Option<String> {
        self.reverse_registry
            .get(&(model_id, Some(object_id)))
            .cloned()
    }

    pub fn get_model_id_py(&mut self, model_name: String) -> PyResult<i64> {
        self.get_model_id(&model_name)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    pub fn is_model_object_key_registered_py(&self, key: String) -> bool {
        self.is_model_object_key_registered(&key)
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

    #[test]
    fn test_validate_base_key() {
        assert!(matches!(
            SymbolMapper::validate_base_key(&"".to_string()),
            Err(_)
        ));

        assert!(matches!(
            SymbolMapper::validate_base_key(&"model".to_string()),
            Ok(_)
        ));

        assert!(matches!(
            SymbolMapper::validate_base_key(&"mo.del".to_string()),
            Err(_)
        ));

        assert!(matches!(
            SymbolMapper::validate_base_key(&".model".to_string()),
            Err(_)
        ));

        assert!(matches!(
            SymbolMapper::validate_base_key(&"model.".to_string()),
            Err(_)
        ));
    }

    #[test]
    fn test_validate_object_key() {
        assert!(matches!(
            SymbolMapper::parse_compound_key(&"".to_string()),
            Err(_)
        ));

        assert!(matches!(
            SymbolMapper::parse_compound_key(&"model".to_string()),
            Err(_)
        ));

        assert!(matches!(
            SymbolMapper::parse_compound_key(&".m".to_string()),
            Err(_)
        ));

        assert!(matches!(
            SymbolMapper::parse_compound_key(&".".to_string()),
            Err(_)
        ));

        assert!(matches!(
            SymbolMapper::parse_compound_key(&"a.".to_string()),
            Err(_)
        ));

        assert!(matches!(
            SymbolMapper::parse_compound_key(&"a.b.c".to_string()),
            Err(_)
        ));

        assert!(matches!(
            SymbolMapper::parse_compound_key(&"a.b".to_string()),
            Ok(_)
        ));

        let (model_name, object_name) =
            SymbolMapper::parse_compound_key(&"a.b".to_string()).unwrap();
        assert_eq!(model_name, "a".to_string());
        assert_eq!(object_name, "b".to_string());
    }

    #[test]
    fn get_model_object_ids() {
        // let (model_id, object_id) =
        //     super::get_object_id("model".to_string(), "object".to_string()).unwrap();
        // assert_eq!(model_id, 0);
        // assert_eq!(object_id, 0);
        //
        // let (model_id, object_id) =
        //     super::get_object_id("model".to_string(), "object1".to_string());
        // assert_eq!(model_id, 0);
        // assert_eq!(object_id, 1);
    }
}
