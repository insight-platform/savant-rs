use crate::json_api::ToSerdeJsonValue;
use crate::primitives::{Attribute, Attributive};
use serde_json::Value;

#[derive(Debug, PartialEq, Clone)]
pub struct UserData {
    pub source_id: String,
    pub attributes: Vec<Attribute>,
}

impl ToSerdeJsonValue for UserData {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!(
        {
            "type": "Telemetry",
            "source_id": self.source_id,
            "attributes": self.attributes.iter().map(|v| v.to_serde_json_value()).collect::<Vec<_>>(),
        })
    }
}

const DEFAULT_ATTRIBUTES_COUNT: usize = 4;

impl UserData {
    pub fn new(source_id: &str) -> Self {
        Self {
            source_id: source_id.to_string(),
            attributes: Vec::with_capacity(DEFAULT_ATTRIBUTES_COUNT),
        }
    }

    pub fn json(&self) -> String {
        serde_json::to_string(&self.to_serde_json_value()).unwrap()
    }

    pub fn json_pretty(&self) -> String {
        serde_json::to_string_pretty(&self.to_serde_json_value()).unwrap()
    }

    pub fn get_source_id(&self) -> &str {
        &self.source_id
    }
}

impl Attributive for UserData {
    fn with_attributes_ref<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Vec<Attribute>) -> R,
    {
        f(&self.attributes)
    }

    fn with_attributes_mut<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Vec<Attribute>) -> R,
    {
        f(&mut self.attributes)
    }
}
