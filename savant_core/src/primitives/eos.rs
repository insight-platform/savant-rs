#[derive(Debug, PartialEq, Clone, serde::Serialize, serde::Deserialize)]
pub struct EndOfStream {
    pub source_id: String,
}

impl EndOfStream {
    pub fn new(source_id: String) -> Self {
        Self { source_id }
    }
}
