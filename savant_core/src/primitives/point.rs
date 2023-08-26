use crate::json_api::ToSerdeJsonValue;
use rkyv::{Archive, Deserialize, Serialize};
use serde_json::Value;

#[derive(
    Archive, Deserialize, Serialize, Debug, PartialEq, Clone, serde::Serialize, serde::Deserialize,
)]
#[archive(check_bytes)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl ToSerdeJsonValue for Point {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!({
            "x": self.x,
            "y": self.y,
        })
    }
}

impl Point {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_point() {
        let p = super::Point::new(1.0, 2.0);
        assert_eq!(p.x, 1.0);
        assert_eq!(p.y, 2.0);
    }
}
