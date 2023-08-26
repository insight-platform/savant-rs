use crate::json_api::ToSerdeJsonValue;
use rkyv::{Archive, Deserialize, Serialize};
use serde_json::Value;

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct Shutdown {
    pub auth: String,
}

impl Shutdown {
    pub fn new(auth: String) -> Self {
        Self { auth }
    }
}

impl ToSerdeJsonValue for Shutdown {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!(
        {
            "type": "Shutdown",
            "auth": self.auth
        })
    }
}
