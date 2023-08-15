use crate::to_json_value::ToSerdeJsonValue;
use rkyv::{Archive, Deserialize, Serialize};
use serde_json::Value;
#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct EndOfStream {
    pub source_id: String,
}

impl EndOfStream {
    pub fn new(source_id: String) -> Self {
        Self { source_id }
    }
}

impl ToSerdeJsonValue for EndOfStream {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!(
        {
            "type": "EndOfStream",
            "source_id": self.source_id,
        })
    }
}
