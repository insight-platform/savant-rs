use crate::json_api::ToSerdeJsonValue;
use serde_json::Value;

#[derive(Debug, PartialEq, Clone)]
pub struct Shutdown(String);

impl Shutdown {
    pub fn new(auth: &str) -> Self {
        Self(auth.to_string())
    }

    pub fn get_auth(&self) -> &str {
        &self.0
    }
}

impl ToSerdeJsonValue for Shutdown {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!(
        {
            "type": "Shutdown",
            "auth": self.get_auth(),
        })
    }
}
