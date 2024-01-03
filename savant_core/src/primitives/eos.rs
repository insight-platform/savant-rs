use rkyv::{Archive, Deserialize, Serialize};
#[derive(
    Archive, Deserialize, Serialize, Debug, PartialEq, Clone, serde::Serialize, serde::Deserialize,
)]
#[archive(check_bytes)]
pub struct EndOfStream {
    pub source_id: String,
}

impl EndOfStream {
    pub fn new(source_id: String) -> Self {
        Self { source_id }
    }
}
