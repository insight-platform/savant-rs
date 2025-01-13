use crate::json_api::ToSerdeJsonValue;
use crate::primitives::{Attribute, WithAttributes};
use crate::protobuf::from_pb;
use savant_protobuf::generated;
use serde_json::Value;

#[derive(Debug, PartialEq, Clone, serde::Serialize, Default)]
pub struct AttributeSet {
    pub attributes: Vec<Attribute>,
}

impl ToSerdeJsonValue for AttributeSet {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!(self)
    }
}

impl From<Vec<Attribute>> for AttributeSet {
    fn from(attributes: Vec<Attribute>) -> Self {
        Self { attributes }
    }
}

impl AttributeSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn deserialize(bytes: &[u8]) -> anyhow::Result<Vec<Attribute>> {
        let deser = from_pb::<generated::AttributeSet, AttributeSet>(bytes)?;
        Ok(deser.attributes)
    }

    pub fn json(&self) -> String {
        serde_json::to_string(&self.to_serde_json_value()).unwrap()
    }

    pub fn json_pretty(&self) -> String {
        serde_json::to_string_pretty(&self.to_serde_json_value()).unwrap()
    }
}

impl WithAttributes for AttributeSet {
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
