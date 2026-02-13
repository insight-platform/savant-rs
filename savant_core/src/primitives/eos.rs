#[derive(Debug, PartialEq, Clone, serde::Serialize, serde::Deserialize)]
pub struct EndOfStream {
    pub source_id: String,
}

impl EndOfStream {
    pub fn new(source_id: &str) -> Self {
        Self {
            source_id: source_id.to_string(),
        }
    }

    pub fn get_source_id(&self) -> &str {
        &self.source_id
    }
}
