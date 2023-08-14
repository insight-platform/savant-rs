use crate::primitives::{Attribute, Attributive};
use crate::to_json_value::ToSerdeJsonValue;
use rkyv::{Archive, Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::mem;

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct UnspecifiedData {
    pub source_id: String,
    pub attributes: HashMap<(String, String), Attribute>,
}

impl ToSerdeJsonValue for UnspecifiedData {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!(
        {
            "type": "Telemetry",
            "source_id": self.source_id,
            "attributes": self.attributes.values().map(|v| v.to_serde_json_value()).collect::<Vec<_>>(),
        })
    }
}

impl UnspecifiedData {
    pub fn new(source_id: String) -> Self {
        Self {
            source_id,
            attributes: HashMap::new(),
        }
    }

    pub fn json(&self) -> String {
        serde_json::to_string(&self.to_serde_json_value()).unwrap()
    }

    pub fn json_pretty(&self) -> String {
        serde_json::to_string_pretty(&self.to_serde_json_value()).unwrap()
    }
}

impl Attributive for UnspecifiedData {
    fn get_attributes_ref(&self) -> &HashMap<(String, String), Attribute> {
        &self.attributes
    }

    fn get_attributes_ref_mut(&mut self) -> &mut HashMap<(String, String), Attribute> {
        &mut self.attributes
    }

    fn take_attributes(&mut self) -> HashMap<(String, String), Attribute> {
        mem::take(&mut self.attributes)
    }

    fn place_attributes(&mut self, attributes: HashMap<(String, String), Attribute>) {
        self.attributes = attributes;
    }
}
