use lazy_static::lazy_static;
use parking_lot::{const_mutex, Mutex};
use std::collections::HashMap;
use thiserror::Error;

const REGISTRY_KEY_SEPARATOR: char = '.';

lazy_static! {
    static ref SYMBOL_MAPPER: Mutex<SymbolMapper> = const_mutex(SymbolMapper::default());
}

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

#[derive(Debug, Clone, Default)]
pub struct SymbolMapper {
    registry: HashMap<String, (i64, Option<i64>)>,
    reverse_registry: HashMap<(i64, Option<i64>), String>,
    model_next_id: i64,
    model_object_next_ids: HashMap<String, i64>,
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
                format!("Key={key}, ModelId={model_id}, ObjectId={object_id:?}")
            })
            .collect()
    }

    pub fn build_model_object_key(model_name: &str, object_label: &str) -> String {
        format!("{model_name}{REGISTRY_KEY_SEPARATOR}{object_label}")
    }

    pub fn get_model_id(&mut self, model_name: &str) -> anyhow::Result<i64> {
        Self::validate_base_key(model_name)?;
        match self.registry.get(model_name) {
            None => {
                let model_id = self.gen_id();
                self.registry
                    .insert(model_name.to_string(), (model_id, None));
                self.reverse_registry
                    .insert((model_id, None), model_name.to_string());
                Ok(model_id)
            }
            Some(&(model_id, None)) => Ok(model_id),
            _ => unreachable!("The method must return only a model id, not model and object ids"),
        }
    }

    pub fn is_model_registered(&self, model: &str) -> bool {
        self.registry.contains_key(model)
    }

    pub fn is_object_registered(&self, model: &str, label: &str) -> bool {
        let key = Self::build_model_object_key(model, label);
        self.registry.contains_key(&key)
    }

    pub fn parse_compound_key(key: &str) -> anyhow::Result<(String, String)> {
        if key.len() < 3 {
            return Err(Errors::FullyQualifiedObjectNameParseError(key.to_string()).into());
        }

        let mut parts = key.split(REGISTRY_KEY_SEPARATOR);

        let model_name = parts.next();
        let object_name = parts.next();

        if parts.count() != 0 {
            return Err(Errors::FullyQualifiedObjectNameParseError(key.to_string()).into());
        }

        match (model_name, object_name) {
            (Some(m), Some(o)) => {
                if !m.is_empty() && !o.is_empty() {
                    Ok((m.to_string(), o.to_string()))
                } else {
                    Err(Errors::FullyQualifiedObjectNameParseError(key.to_string()).into())
                }
            }
            _ => Err(Errors::FullyQualifiedObjectNameParseError(key.to_string()).into()),
        }
    }

    pub fn validate_base_key(key: &str) -> anyhow::Result<String> {
        if key.is_empty() {
            return Err(Errors::BaseNameParseError(key.to_string()).into());
        }
        let parts = key.split(REGISTRY_KEY_SEPARATOR);
        if parts.count() == 1 {
            Ok(key.to_string())
        } else {
            Err(Errors::BaseNameParseError(key.to_string()).into())
        }
    }

    pub fn get_object_id(
        &mut self,
        model_name: &str,
        object_label: &str,
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
                    .insert((model_id, Some(object_id)), object_label.to_string());
                self.model_object_next_ids
                    .insert(model_name.to_string(), object_id);
                Ok((model_id, object_id))
            }
        }
    }

    pub fn register_model_objects(
        &mut self,
        model_name: &str,
        objects: &std::collections::HashMap<i64, String>,
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
                        model_name.to_string(),
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
            .insert(model_name.to_string(), last_object_id);

        Ok(model_id)
    }
    pub fn gen_id(&mut self) -> i64 {
        let id = self.model_next_id;
        self.model_next_id += 1;
        id
    }

    pub fn get_model_name(&self, id: i64) -> Option<String> {
        self.reverse_registry.get(&(id, None)).cloned()
    }

    pub fn get_object_label(&self, model_id: i64, object_id: i64) -> Option<String> {
        self.reverse_registry
            .get(&(model_id, Some(object_id)))
            .cloned()
    }
}

pub fn get_model_id(model_name: &str) -> anyhow::Result<i64> {
    let mut mapper = SYMBOL_MAPPER.lock();
    mapper.get_model_id(model_name)
}

pub fn get_object_id(model_name: &str, object_label: &str) -> anyhow::Result<(i64, i64)> {
    let mut mapper = SYMBOL_MAPPER.lock();
    mapper.get_object_id(model_name, object_label)
}

pub fn register_model_objects(
    model_name: &str,
    elements: HashMap<i64, String>,
    policy: RegistrationPolicy,
) -> anyhow::Result<i64> {
    let mut mapper = SYMBOL_MAPPER.lock();
    mapper.register_model_objects(model_name, &elements, &policy)
}

pub fn get_model_name(model_id: i64) -> Option<String> {
    let mapper = SYMBOL_MAPPER.lock();
    mapper.get_model_name(model_id)
}

pub fn get_object_label(model_id: i64, object_id: i64) -> Option<String> {
    let mapper = SYMBOL_MAPPER.lock();
    mapper.get_object_label(model_id, object_id)
}

pub fn get_object_labels(model_id: i64, object_ids: Vec<i64>) -> Vec<(i64, Option<String>)> {
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

pub fn get_object_ids(model_name: &str, object_labels: Vec<String>) -> Vec<(String, Option<i64>)> {
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

pub fn clear_symbol_maps() {
    let mut mapper = SYMBOL_MAPPER.lock();
    mapper.clear();
}

pub fn is_model_registered(model_name: &str) -> bool {
    let mapper = SYMBOL_MAPPER.lock();
    mapper.is_model_registered(model_name)
}

pub fn is_object_registered(model_name: &str, object_label: &str) -> bool {
    let mapper = SYMBOL_MAPPER.lock();
    mapper.is_object_registered(model_name, object_label)
}

pub fn dump_registry() -> Vec<String> {
    let mapper = SYMBOL_MAPPER.lock();
    mapper.dump_registry()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn s(s: &str) -> String {
        s.to_string()
    }

    #[test]
    fn test_validate_base_key() {
        assert!(SymbolMapper::validate_base_key("").is_err());
        assert!(SymbolMapper::validate_base_key("model").is_ok());
        assert!(SymbolMapper::validate_base_key("mo.del").is_err());
        assert!(SymbolMapper::validate_base_key(".model").is_err());
        assert!(SymbolMapper::validate_base_key("model.").is_err());
    }

    #[test]
    fn test_validate_object_key() {
        assert!(SymbolMapper::parse_compound_key("").is_err());
        assert!(SymbolMapper::parse_compound_key("model").is_err());
        assert!(SymbolMapper::parse_compound_key(".m").is_err());
        assert!(SymbolMapper::parse_compound_key(".").is_err());
        assert!(SymbolMapper::parse_compound_key("a.").is_err());
        assert!(SymbolMapper::parse_compound_key("a.b.c").is_err());
        assert!(SymbolMapper::parse_compound_key("a.b").is_ok());
        let (model_name, object_name) = SymbolMapper::parse_compound_key("a.b").unwrap();
        assert_eq!(model_name, "a");
        assert_eq!(object_name, "b");
    }

    #[test]
    fn register_incorrect_names() -> anyhow::Result<()> {
        let mut sm = SymbolMapper::default();

        assert!(sm
            .register_model_objects(
                "model.",
                &std::collections::HashMap::default(),
                &RegistrationPolicy::ErrorIfNonUnique,
            )
            .is_err());

        assert!(sm
            .register_model_objects(
                "model",
                &[(1, s("obj.ect"))].into_iter().collect(),
                &RegistrationPolicy::ErrorIfNonUnique,
            )
            .is_err());

        Ok(())
    }

    #[test]
    fn test_register_duplicate_objects_error_non_unique() -> anyhow::Result<()> {
        let mut sm = SymbolMapper::default();

        assert!(sm
            .register_model_objects(
                "model",
                &[(1, s("object")), (2, s("object"))].into_iter().collect(),
                &RegistrationPolicy::ErrorIfNonUnique,
            )
            .is_err());

        let mut sm = SymbolMapper::default();

        sm.register_model_objects(
            "model",
            &[(1, s("object"))].into_iter().collect(),
            &RegistrationPolicy::ErrorIfNonUnique,
        )?;

        assert!(sm
            .register_model_objects(
                "model",
                &[(1, s("object2"))].into_iter().collect(),
                &RegistrationPolicy::ErrorIfNonUnique,
            )
            .is_err());

        Ok(())
    }

    #[test]
    fn test_register_duplicate_objects_override() -> anyhow::Result<()> {
        let mut sm = SymbolMapper::default();

        sm.register_model_objects(
            "model",
            &[(1, s("object"))].into_iter().collect(),
            &RegistrationPolicy::Override,
        )?;

        assert!(matches!(
            sm.register_model_objects(
                "model",
                &[(2, s("object"))].into_iter().collect(),
                &RegistrationPolicy::Override,
            ),
            Ok(0)
        ));

        assert!(matches!(sm.get_object_id("model", "object"), Ok((0, 2))));

        let label = sm.get_object_label(0, 2).unwrap();
        assert_eq!(label, s("object"));

        let mut sm = SymbolMapper::default();

        sm.register_model_objects(
            "model",
            &[(1, s("object"))].into_iter().collect(),
            &RegistrationPolicy::Override,
        )?;

        assert!(matches!(
            sm.register_model_objects(
                "model",
                &[(1, s("object2"))].into_iter().collect(),
                &RegistrationPolicy::Override,
            ),
            Ok(0)
        ));

        let label = sm.get_object_label(0, 1).unwrap();
        assert_eq!(label, s("object2"));

        Ok(())
    }

    #[test]
    fn test_get_model_id() {
        let mut sm = SymbolMapper::default();
        let model_id = sm.get_model_id("model").unwrap();
        assert_eq!(model_id, 0);
    }

    #[test]
    fn test_get_model_name() {
        let mut sm = SymbolMapper::default();
        let model_id = sm.get_model_id("model").unwrap();
        assert_eq!(model_id, 0);

        let model_name = sm.get_model_name(model_id).unwrap();
        assert_eq!(model_name, s("model"));

        let nonexistent_model_name = sm.get_model_name(1);
        assert!(nonexistent_model_name.is_none());
    }

    #[test]
    fn test_get_object_label() {
        let mut sm = SymbolMapper::default();
        let (model_id, object_id) = sm.get_object_id("model", "object").unwrap();
        assert_eq!(model_id, 0);
        assert_eq!(object_id, 0);

        let object_label = sm.get_object_label(model_id, object_id).unwrap();
        assert_eq!(object_label, s("object"));

        let nonexistent_object_label = sm.get_object_label(0, 1);
        assert!(nonexistent_object_label.is_none());
    }

    #[test]
    fn get_model_object_ids() {
        let mut sm = SymbolMapper::default();
        let (model_id, object_id) = sm.get_object_id("model", "object0").unwrap();
        assert_eq!(model_id, 0);
        assert_eq!(object_id, 0);

        let (model_id, object_id) = sm.get_object_id("model", "object1").unwrap();
        assert_eq!(model_id, 0);
        assert_eq!(object_id, 1);

        let (model_id, object_id) = sm.get_object_id("model2", "object0").unwrap();
        assert_eq!(model_id, 1);
        assert_eq!(object_id, 0);
    }

    #[test]
    fn register_and_get_model_object_ids() -> anyhow::Result<()> {
        let mut sm = SymbolMapper::default();
        sm.register_model_objects(
            "model",
            &[(2, s("object0"))].into_iter().collect(),
            &RegistrationPolicy::Override,
        )?;

        let (model_id, object_id) = sm.get_object_id("model", "object0").unwrap();
        assert_eq!(model_id, 0);
        assert_eq!(object_id, 2);

        let (model_id, object_id) = sm.get_object_id("model", "object1").unwrap();
        assert_eq!(model_id, 0);
        assert_eq!(object_id, 3);

        Ok(())
    }
}
