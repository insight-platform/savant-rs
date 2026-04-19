//! Python wrappers for tracker output types.

use super::enums::PyTrackState;
use crate::deepstream::buffer::PySharedBuffer;
use deepstream_buffers::SharedBuffer;
use deepstream_nvtracker::{TrackedObject, TrackerOutput};
use pyo3::prelude::*;
use savant_core::primitives::misc_track::{MiscTrackData, MiscTrackFrame};

#[pyclass(
    name = "TrackedObject",
    module = "savant_rs.nvtracker",
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub struct PyTrackedObject {
    #[pyo3(get)]
    pub object_id: u64,
    #[pyo3(get)]
    pub class_id: i64,
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
    pub tracker_confidence: f32,
    #[pyo3(get)]
    pub label: Option<String>,
    #[pyo3(get)]
    pub slot_number: i64,
    #[pyo3(get)]
    pub source_id: String,
}

impl From<TrackedObject> for PyTrackedObject {
    fn from(o: TrackedObject) -> Self {
        Self {
            object_id: o.object_id,
            class_id: o.class_id,
            bbox_left: o.bbox_left,
            bbox_top: o.bbox_top,
            bbox_width: o.bbox_width,
            bbox_height: o.bbox_height,
            confidence: o.confidence,
            tracker_confidence: o.tracker_confidence,
            label: o.label,
            slot_number: o.slot_number,
            source_id: o.source_id,
        }
    }
}

#[pyclass(
    name = "MiscTrackFrame",
    module = "savant_rs.nvtracker",
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub struct PyMiscTrackFrame {
    #[pyo3(get)]
    pub frame_num: i64,
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
    pub age: i64,
    #[pyo3(get)]
    pub state: PyTrackState,
    #[pyo3(get)]
    pub visibility: f32,
}

impl From<MiscTrackFrame> for PyMiscTrackFrame {
    fn from(f: MiscTrackFrame) -> Self {
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

#[pyclass(
    name = "MiscTrackData",
    module = "savant_rs.nvtracker",
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub struct PyMiscTrackData {
    #[pyo3(get)]
    pub object_id: u64,
    #[pyo3(get)]
    pub class_id: i64,
    #[pyo3(get)]
    pub label: Option<String>,
    #[pyo3(get)]
    pub source_id: String,
    #[pyo3(get)]
    pub frames: Vec<PyMiscTrackFrame>,
}

impl From<MiscTrackData> for PyMiscTrackData {
    fn from(d: MiscTrackData) -> Self {
        Self {
            object_id: d.object_id,
            class_id: d.class_id,
            label: d.label,
            source_id: d.source_id,
            frames: d.frames.into_iter().map(PyMiscTrackFrame::from).collect(),
        }
    }
}

#[pyclass(
    name = "TrackerOutput",
    module = "savant_rs.nvtracker",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyTrackerOutput {
    buffer: SharedBuffer,
    #[pyo3(get)]
    pub current_tracks: Vec<PyTrackedObject>,
    #[pyo3(get)]
    pub shadow_tracks: Vec<PyMiscTrackData>,
    #[pyo3(get)]
    pub terminated_tracks: Vec<PyMiscTrackData>,
    #[pyo3(get)]
    pub past_frame_data: Vec<PyMiscTrackData>,
}

#[pymethods]
impl PyTrackerOutput {
    /// Output batched buffer (wrapped; clones the underlying ``SharedBuffer``).
    fn buffer(&self) -> PySharedBuffer {
        PySharedBuffer::from_rust(self.buffer.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "TrackerOutput(current_tracks={}, shadow={}, terminated={}, past={})",
            self.current_tracks.len(),
            self.shadow_tracks.len(),
            self.terminated_tracks.len(),
            self.past_frame_data.len()
        )
    }
}

impl PyTrackerOutput {
    pub(crate) fn from_rust(o: TrackerOutput) -> Self {
        let (buffer, current_tracks, shadow_tracks, terminated_tracks, past_frame_data) =
            o.into_parts();
        Self {
            buffer,
            current_tracks: current_tracks
                .into_iter()
                .map(PyTrackedObject::from)
                .collect(),
            shadow_tracks: shadow_tracks
                .into_iter()
                .map(PyMiscTrackData::from)
                .collect(),
            terminated_tracks: terminated_tracks
                .into_iter()
                .map(PyMiscTrackData::from)
                .collect(),
            past_frame_data: past_frame_data
                .into_iter()
                .map(PyMiscTrackData::from)
                .collect(),
        }
    }
}
