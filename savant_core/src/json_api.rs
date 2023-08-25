pub trait ToSerdeJsonValue {
    fn to_serde_json_value(&self) -> serde_json::Value;
}
