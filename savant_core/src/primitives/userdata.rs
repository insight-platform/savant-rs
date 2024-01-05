use crate::json_api::ToSerdeJsonValue;
use crate::primitives::{Attribute, Attributive};
use rkyv::{Archive, Deserialize, Serialize};
use serde_json::Value;
use std::mem;

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
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
    pub fn new(source_id: String) -> Self {
        Self {
            source_id,
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
    fn get_attributes_ref(&self) -> &Vec<Attribute> {
        &self.attributes
    }

    fn get_attributes_ref_mut(&mut self) -> &mut Vec<Attribute> {
        &mut self.attributes
    }

    fn take_attributes(&mut self) -> Vec<Attribute> {
        mem::take(&mut self.attributes)
    }

    fn place_attributes(&mut self, mut attributes: Vec<Attribute>) {
        self.attributes.append(&mut attributes);
    }
}
