//! Tracker misc-track types: shadow, terminated, and past-frame data.
//!
//! These types represent tracker targets that are **not** bound to a regular
//! [`VideoObject`](super::object::VideoObject) in the current frame.  They
//! originate from DeepStream's `NvDsTargetMiscData*` family but are defined
//! here in `savant_core` so that [`VideoFrame`](super::frame::VideoFrame) can
//! carry them without depending on the DeepStream crates.

use crate::json_api::ToSerdeJsonValue;
use crate::primitives::RBBox;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// TrackState
// ---------------------------------------------------------------------------

/// Logical state of a tracker target.
///
/// Mirrors DeepStream's `TRACKER_STATE` enum but lives in `savant_core` so
/// it is available without the DeepStream feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TrackState {
    Empty = 0,
    Active = 1,
    Inactive = 2,
    Tentative = 3,
    Projected = 4,
}

impl ToSerdeJsonValue for TrackState {
    fn to_serde_json_value(&self) -> Value {
        Value::String(format!("{:?}", self))
    }
}

// ---------------------------------------------------------------------------
// MiscTrackCategory
// ---------------------------------------------------------------------------

/// Which tracker output list a [`MiscTrackData`] came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MiscTrackCategory {
    Shadow,
    Terminated,
    PastFrame,
}

impl ToSerdeJsonValue for MiscTrackCategory {
    fn to_serde_json_value(&self) -> Value {
        Value::String(format!("{:?}", self))
    }
}

// ---------------------------------------------------------------------------
// MiscTrackFrame
// ---------------------------------------------------------------------------

/// A single frame's worth of position data inside a misc track history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiscTrackFrame {
    pub frame_num: i64,
    pub bbox_left: f32,
    pub bbox_top: f32,
    pub bbox_width: f32,
    pub bbox_height: f32,
    pub confidence: f32,
    pub age: i64,
    pub state: TrackState,
    pub visibility: f32,
}

impl ToSerdeJsonValue for MiscTrackFrame {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!({
            "frame_num": self.frame_num,
            "bbox_left": self.bbox_left,
            "bbox_top": self.bbox_top,
            "bbox_width": self.bbox_width,
            "bbox_height": self.bbox_height,
            "confidence": self.confidence,
            "age": self.age,
            "state": self.state.to_serde_json_value(),
            "visibility": self.visibility,
        })
    }
}

// ---------------------------------------------------------------------------
// MiscTrackData
// ---------------------------------------------------------------------------

/// One tracker target that is **not** represented as a regular
/// [`VideoObject`](super::object::VideoObject) in the current frame.
///
/// Typical sources are the shadow-track, terminated-track, or past-frame
/// lists produced by DeepStream's nvtracker plugin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiscTrackData {
    pub object_id: u64,
    pub class_id: i64,
    pub label: Option<String>,
    pub source_id: String,
    pub category: MiscTrackCategory,
    pub frames: Vec<MiscTrackFrame>,
}

impl ToSerdeJsonValue for MiscTrackData {
    fn to_serde_json_value(&self) -> Value {
        serde_json::json!({
            "object_id": self.object_id,
            "class_id": self.class_id,
            "label": self.label,
            "source_id": self.source_id,
            "category": self.category.to_serde_json_value(),
            "frames": self.frames.iter().map(|f| f.to_serde_json_value()).collect::<Vec<_>>(),
        })
    }
}

// ---------------------------------------------------------------------------
// TrackUpdate
// ---------------------------------------------------------------------------

/// Describes how to apply tracker output to an existing
/// [`VideoObject`](super::object::VideoObject) on a frame.
///
/// Used by [`VideoFrameProxy::apply_tracking_info`](super::frame::VideoFrameProxy::apply_tracking_info).
#[derive(Debug, Clone)]
pub struct TrackUpdate {
    /// The id of the [`VideoObject`](super::object::VideoObject) on the frame.
    pub object_id: i64,
    /// The tracker-assigned track id.
    pub track_id: i64,
    /// The tracker-assigned bounding box.
    pub track_box: RBBox,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn track_state_json() {
        assert_eq!(
            TrackState::Tentative.to_serde_json_value(),
            Value::String("Tentative".to_string())
        );
    }

    #[test]
    fn misc_track_category_json() {
        assert_eq!(
            MiscTrackCategory::Shadow.to_serde_json_value(),
            Value::String("Shadow".to_string())
        );
    }

    #[test]
    fn misc_track_frame_json() {
        let f = MiscTrackFrame {
            frame_num: 42,
            bbox_left: 1.0,
            bbox_top: 2.0,
            bbox_width: 3.0,
            bbox_height: 4.0,
            confidence: 0.9,
            age: 5,
            state: TrackState::Active,
            visibility: 0.8,
        };
        let v = f.to_serde_json_value();
        assert_eq!(v["frame_num"], 42);
        assert_eq!(v["state"], "Active");
    }

    #[test]
    fn misc_track_data_json() {
        let d = MiscTrackData {
            object_id: 7,
            class_id: 2,
            label: Some("person".to_string()),
            source_id: "cam-1".to_string(),
            category: MiscTrackCategory::Terminated,
            frames: vec![],
        };
        let v = d.to_serde_json_value();
        assert_eq!(v["object_id"], 7);
        assert_eq!(v["category"], "Terminated");
    }

    #[test]
    fn track_state_serde_round_trip() {
        let s = TrackState::Projected;
        let json = serde_json::to_string(&s).unwrap();
        let back: TrackState = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn misc_track_data_serde_round_trip() {
        let d = MiscTrackData {
            object_id: 99,
            class_id: 1,
            label: None,
            source_id: "s".to_string(),
            category: MiscTrackCategory::PastFrame,
            frames: vec![MiscTrackFrame {
                frame_num: 0,
                bbox_left: 0.0,
                bbox_top: 0.0,
                bbox_width: 10.0,
                bbox_height: 10.0,
                confidence: 0.5,
                age: 1,
                state: TrackState::Empty,
                visibility: 1.0,
            }],
        };
        let json = serde_json::to_string(&d).unwrap();
        let back: MiscTrackData = serde_json::from_str(&json).unwrap();
        assert_eq!(back.object_id, 99);
        assert_eq!(back.category, MiscTrackCategory::PastFrame);
        assert_eq!(back.frames.len(), 1);
    }
}
