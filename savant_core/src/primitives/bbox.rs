use crate::to_json_value::ToSerdeJsonValue;
use rkyv::{Archive, Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename = "bbox.metric.type")]
pub enum BBoxMetricType {
    IoU,
    IoSelf,
    IoOther,
}

#[derive(Archive, Deserialize, Serialize, Debug, PartialEq, Clone)]
#[archive(check_bytes)]
pub struct OwnedRBBoxData {
    pub xc: f32,
    pub yc: f32,
    pub width: f32,
    pub height: f32,
    pub angle: Option<f32>,
    pub has_modifications: bool,
}

impl Default for OwnedRBBoxData {
    fn default() -> Self {
        Self {
            xc: 0.0,
            yc: 0.0,
            width: 0.0,
            height: 0.0,
            angle: None,
            has_modifications: false,
        }
    }
}

impl ToSerdeJsonValue for OwnedRBBoxData {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!({
            "xc": self.xc,
            "yc": self.yc,
            "width": self.width,
            "height": self.height,
            "angle": self.angle,
        })
    }
}
