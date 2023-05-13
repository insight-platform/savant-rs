use hashbrown::HashMap;
use itertools::Itertools;
use pyo3::prelude::*;

const REGISTRY_KEY_SEPARATOR: char = '.';

#[pyclass]
#[derive(Debug, Clone)]
pub struct SymbolMapper {
    model_registry: HashMap<String, i64>,
    object_registry: HashMap<String, (i64, i64)>,
    model_object_next_ids: HashMap<String, i64>,
    reverse_registry: HashMap<i64, String>,
    next_id: i64,
}

impl Default for SymbolMapper {
    fn default() -> Self {
        Self {
            model_registry: HashMap::new(),
            object_registry: HashMap::new(),
            reverse_registry: HashMap::new(),
            model_object_next_ids: HashMap::new(),
            next_id: 0,
        }
    }
}

impl SymbolMapper {
    pub fn model_object_key(model_name: &Option<&String>, object_label: &String) -> String {
        match model_name {
            Some(m) => format!("{}{}{}", m, REGISTRY_KEY_SEPARATOR, object_label),
            None => object_label.clone(),
        }
    }

    pub fn get_model_id(&self, model_name: &String) -> Option<i64> {
        self.model_registry.get(model_name).cloned()
    }

    pub fn is_model_object_key_registered(&self, key: &String) -> bool {
        self.object_registry.contains_key(key)
    }

    pub fn parse_model_object_key(key: &String) -> (String, String) {
        let mut parts = key.split(REGISTRY_KEY_SEPARATOR);
        let model_name = parts.next().map(|s| s.to_string());
        match model_name {
            Some(m) => (
                m,
                parts.join(&REGISTRY_KEY_SEPARATOR.to_string()).to_string(),
            ),
            None => ("".to_string(), key.clone()),
        }
    }

    pub fn get_model_object_ids(
        &mut self,
        model_object_key: Option<String>,
        model_name: Option<String>,
        object_label: Option<String>,
    ) -> (i64, i64) {
        let (full_key, model_name) = match (model_object_key, model_name, object_label) {
            (Some(k), None, None) => {
                let (m, _) = Self::parse_model_object_key(&k);
                (k, m)
            },
            (None, Some(m), Some(o)) => {
                (Self::model_object_key(&Some(&m), &o), m)
            },
            _ => panic!("Invalid arguments: either model_object_key or [model_name and object_label] must be provided"),
        };

        let model_id = match self.model_registry.get(&model_name) {
            Some(model_id) => *model_id,
            None => {
                let model_id = self.gen_id();
                self.model_registry.insert(model_name.clone(), model_id);
                self.reverse_registry.insert(model_id, model_name.clone());
                model_id
            }
        };

        match self.object_registry.get(&full_key) {
            Some((model_id, object_id)) => (*model_id, *object_id),
            None => {
                let last_object_id = self
                    .model_object_next_ids
                    .get(&model_name)
                    .cloned()
                    .unwrap_or(-1);
                let object_id = last_object_id + 1;
                self.object_registry
                    .insert(full_key.clone(), (model_id, object_id));
                (model_id, object_id)
            }
        }
    }

    pub fn register_model(
        &mut self,
        model_name: String,
        elements: Option<std::collections::HashMap<i64, String>>,
    ) -> i64 {
        let id = self.gen_id();
        self.model_registry.insert(model_name.clone(), id);
        self.reverse_registry.insert(id, model_name.clone());
        match elements {
            Some(elements) => {
                let mut last_object_id = self
                    .model_object_next_ids
                    .get(&model_name)
                    .cloned()
                    .unwrap_or(-1);

                for (element_id, element_label) in elements {
                    self.object_registry.insert(
                        Self::model_object_key(&Some(&model_name), &element_label),
                        (id, element_id),
                    );
                    if element_id > last_object_id {
                        last_object_id = element_id;
                    }
                }
                self.model_object_next_ids
                    .insert(model_name, last_object_id);
            }
            None => (),
        }
        id
    }
}

#[pymethods]
impl SymbolMapper {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn gen_id(&mut self) -> i64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    #[staticmethod]
    #[pyo3(signature = (model_name=None, object_label="".to_string()))]
    pub fn model_object_key_py(model_name: Option<String>, object_label: String) -> String {
        Self::model_object_key(&model_name.as_ref(), &object_label)
    }

    #[staticmethod]
    pub fn parse_model_object_key_py(key: String) -> (String, String) {
        Self::parse_model_object_key(&key)
    }

    pub fn register_model_py(
        &mut self,
        model_name: String,
        elements: Option<std::collections::HashMap<i64, String>>,
    ) -> i64 {
        Python::with_gil(|py| py.allow_threads(|| self.register_model(model_name, elements)))
    }

    pub fn get_name(&self, id: i64) -> Option<String> {
        self.reverse_registry.get(&id).cloned()
    }

    pub fn get_model_id_py(&self, model_name: String) -> Option<i64> {
        self.get_model_id(&model_name)
    }

    pub fn is_model_object_key_registered_py(&self, key: String) -> bool {
        self.is_model_object_key_registered(&key)
    }

    pub fn get_model_object_ids_py(
        &mut self,
        model_object_key: Option<String>,
        model_name: Option<String>,
        object_label: Option<String>,
    ) -> (i64, i64) {
        Python::with_gil(|py| {
            py.allow_threads(|| {
                self.get_model_object_ids(model_object_key, model_name, object_label)
            })
        })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_parse_object_key() {
        use super::SymbolMapper;

        let (model_name, object_label) = SymbolMapper::parse_model_object_key(&"".to_string());
        assert_eq!(model_name, "".to_string());
        assert_eq!(object_label, "".to_string());

        let (model_name, object_label) = SymbolMapper::parse_model_object_key(&"model".to_string());
        assert_eq!(model_name, "model".to_string());
        assert_eq!(object_label, "".to_string());

        let (model_name, object_label) =
            SymbolMapper::parse_model_object_key(&"model.object".to_string());
        assert_eq!(model_name, "model".to_string());
        assert_eq!(object_label, "object".to_string());

        let (model_name, object_label) =
            SymbolMapper::parse_model_object_key(&"model.object.label".to_string());

        assert_eq!(model_name, "model".to_string());
        assert_eq!(object_label, "object.label".to_string());

        let (model_name, object_label) =
            SymbolMapper::parse_model_object_key(&".object".to_string());
        assert_eq!(model_name, "".to_string());
        assert_eq!(object_label, "object".to_string());

        let (model_name, object_label) =
            SymbolMapper::parse_model_object_key(&"model.".to_string());
        assert_eq!(model_name, "model".to_string());
        assert_eq!(object_label, "".to_string());
    }
}
