//! Python wrappers for core misc-track types.

use pyo3::prelude::*;
use savant_core::primitives::misc_track as core;

use crate::primitives::bbox::RBBox;

/// Logical state of a tracker target.
#[pyclass(
    from_py_object,
    name = "TrackState",
    module = "savant_rs.primitives",
    eq,
    eq_int
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackState {
    #[pyo3(name = "EMPTY")]
    Empty = 0,
    #[pyo3(name = "ACTIVE")]
    Active = 1,
    #[pyo3(name = "INACTIVE")]
    Inactive = 2,
    #[pyo3(name = "TENTATIVE")]
    Tentative = 3,
    #[pyo3(name = "PROJECTED")]
    Projected = 4,
}

impl From<core::TrackState> for TrackState {
    fn from(s: core::TrackState) -> Self {
        match s {
            core::TrackState::Empty => TrackState::Empty,
            core::TrackState::Active => TrackState::Active,
            core::TrackState::Inactive => TrackState::Inactive,
            core::TrackState::Tentative => TrackState::Tentative,
            core::TrackState::Projected => TrackState::Projected,
        }
    }
}

impl From<TrackState> for core::TrackState {
    fn from(s: TrackState) -> Self {
        match s {
            TrackState::Empty => core::TrackState::Empty,
            TrackState::Active => core::TrackState::Active,
            TrackState::Inactive => core::TrackState::Inactive,
            TrackState::Tentative => core::TrackState::Tentative,
            TrackState::Projected => core::TrackState::Projected,
        }
    }
}

/// Which tracker output list a misc track came from.
#[pyclass(
    from_py_object,
    name = "MiscTrackCategory",
    module = "savant_rs.primitives",
    eq,
    eq_int
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiscTrackCategory {
    #[pyo3(name = "SHADOW")]
    Shadow = 0,
    #[pyo3(name = "TERMINATED")]
    Terminated = 1,
    #[pyo3(name = "PAST_FRAME")]
    PastFrame = 2,
}

impl From<core::MiscTrackCategory> for MiscTrackCategory {
    fn from(c: core::MiscTrackCategory) -> Self {
        match c {
            core::MiscTrackCategory::Shadow => MiscTrackCategory::Shadow,
            core::MiscTrackCategory::Terminated => MiscTrackCategory::Terminated,
            core::MiscTrackCategory::PastFrame => MiscTrackCategory::PastFrame,
        }
    }
}

impl From<MiscTrackCategory> for core::MiscTrackCategory {
    fn from(c: MiscTrackCategory) -> Self {
        match c {
            MiscTrackCategory::Shadow => core::MiscTrackCategory::Shadow,
            MiscTrackCategory::Terminated => core::MiscTrackCategory::Terminated,
            MiscTrackCategory::PastFrame => core::MiscTrackCategory::PastFrame,
        }
    }
}

/// A single frame's worth of position data inside a misc track history.
#[pyclass(
    from_py_object,
    name = "MiscTrackFrame",
    module = "savant_rs.primitives"
)]
#[derive(Debug, Clone)]
pub struct MiscTrackFrame {
    #[pyo3(get)]
    pub frame_num: u32,
    #[pyo3(get)]
    pub bbox_left: f32,
    #[pyo3(get)]
    pub bbox_top: f32,
    #[pyo3(get)]
    pub bbox_width: f32,
    #[pyo3(get)]
    pub bbox_height: f32,
    #[pyo3(get)]
    pub confidence: f32,
    #[pyo3(get)]
    pub age: u32,
    #[pyo3(get)]
    pub state: TrackState,
    #[pyo3(get)]
    pub visibility: f32,
}

#[pymethods]
impl MiscTrackFrame {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (frame_num, bbox_left, bbox_top, bbox_width, bbox_height, confidence, age, state, visibility))]
    fn new(
        frame_num: u32,
        bbox_left: f32,
        bbox_top: f32,
        bbox_width: f32,
        bbox_height: f32,
        confidence: f32,
        age: u32,
        state: TrackState,
        visibility: f32,
    ) -> Self {
        Self {
            frame_num,
            bbox_left,
            bbox_top,
            bbox_width,
            bbox_height,
            confidence,
            age,
            state,
            visibility,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "MiscTrackFrame(frame_num={}, state={:?}, confidence={:.3})",
            self.frame_num, self.state, self.confidence
        )
    }
}

impl From<core::MiscTrackFrame> for MiscTrackFrame {
    fn from(f: core::MiscTrackFrame) -> Self {
        Self {
            frame_num: f.frame_num,
            bbox_left: f.bbox_left,
            bbox_top: f.bbox_top,
            bbox_width: f.bbox_width,
            bbox_height: f.bbox_height,
            confidence: f.confidence,
            age: f.age,
            state: f.state.into(),
            visibility: f.visibility,
        }
    }
}

impl From<&MiscTrackFrame> for core::MiscTrackFrame {
    fn from(f: &MiscTrackFrame) -> Self {
        Self {
            frame_num: f.frame_num,
            bbox_left: f.bbox_left,
            bbox_top: f.bbox_top,
            bbox_width: f.bbox_width,
            bbox_height: f.bbox_height,
            confidence: f.confidence,
            age: f.age,
            state: f.state.into(),
            visibility: f.visibility,
        }
    }
}

/// A tracker target not bound to a regular VideoObject.
#[pyclass(
    from_py_object,
    name = "MiscTrackData",
    module = "savant_rs.primitives"
)]
#[derive(Debug, Clone)]
pub struct MiscTrackData {
    #[pyo3(get)]
    pub object_id: u64,
    #[pyo3(get)]
    pub class_id: u16,
    #[pyo3(get)]
    pub label: Option<String>,
    #[pyo3(get)]
    pub source_id: String,
    #[pyo3(get)]
    pub category: MiscTrackCategory,
    #[pyo3(get)]
    pub frames: Vec<MiscTrackFrame>,
}

#[pymethods]
impl MiscTrackData {
    #[new]
    #[pyo3(signature = (object_id, class_id, source_id, category, label=None, frames=None))]
    fn new(
        object_id: u64,
        class_id: u16,
        source_id: String,
        category: MiscTrackCategory,
        label: Option<String>,
        frames: Option<Vec<MiscTrackFrame>>,
    ) -> Self {
        Self {
            object_id,
            class_id,
            label,
            source_id,
            category,
            frames: frames.unwrap_or_default(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "MiscTrackData(object_id={}, class_id={}, category={:?}, frames={})",
            self.object_id,
            self.class_id,
            self.category,
            self.frames.len()
        )
    }
}

impl From<core::MiscTrackData> for MiscTrackData {
    fn from(d: core::MiscTrackData) -> Self {
        Self {
            object_id: d.object_id,
            class_id: d.class_id,
            label: d.label,
            source_id: d.source_id,
            category: d.category.into(),
            frames: d.frames.into_iter().map(MiscTrackFrame::from).collect(),
        }
    }
}

impl From<&MiscTrackData> for core::MiscTrackData {
    fn from(d: &MiscTrackData) -> Self {
        Self {
            object_id: d.object_id,
            class_id: d.class_id,
            label: d.label.clone(),
            source_id: d.source_id.clone(),
            category: d.category.into(),
            frames: d.frames.iter().map(core::MiscTrackFrame::from).collect(),
        }
    }
}

/// Describes how to apply tracker output to an existing VideoObject.
#[pyclass(from_py_object, name = "TrackUpdate", module = "savant_rs.primitives")]
#[derive(Debug, Clone)]
pub struct TrackUpdate {
    #[pyo3(get)]
    pub object_id: i64,
    #[pyo3(get)]
    pub track_id: i64,
    #[pyo3(get)]
    pub track_box: RBBox,
}

#[pymethods]
impl TrackUpdate {
    #[new]
    fn new(object_id: i64, track_id: i64, track_box: RBBox) -> Self {
        Self {
            object_id,
            track_id,
            track_box,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TrackUpdate(object_id={}, track_id={})",
            self.object_id, self.track_id
        )
    }
}

impl From<&TrackUpdate> for core::TrackUpdate {
    fn from(u: &TrackUpdate) -> Self {
        Self {
            object_id: u.object_id,
            track_id: u.track_id,
            track_box: u.track_box.0.clone(),
        }
    }
}
