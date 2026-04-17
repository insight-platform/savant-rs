//! Python enums for nvtracker.

use nvtracker::TrackingIdResetMode;
use pyo3::prelude::*;
use savant_core::primitives::misc_track::TrackState;

/// Maps to DeepStream ``tracking-id-reset-mode`` on ``nvtracker``.
#[pyclass(
    from_py_object,
    name = "TrackingIdResetMode",
    module = "savant_rs.nvtracker",
    eq,
    eq_int
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyTrackingIdResetMode {
    #[pyo3(name = "NONE")]
    None = 0,
    #[pyo3(name = "ON_STREAM_RESET")]
    OnStreamReset = 1,
    #[pyo3(name = "ON_EOS")]
    OnEos = 2,
    #[pyo3(name = "ON_STREAM_RESET_AND_EOS")]
    OnStreamResetAndEos = 3,
}

impl From<PyTrackingIdResetMode> for TrackingIdResetMode {
    fn from(p: PyTrackingIdResetMode) -> Self {
        match p {
            PyTrackingIdResetMode::None => TrackingIdResetMode::None,
            PyTrackingIdResetMode::OnStreamReset => TrackingIdResetMode::OnStreamReset,
            PyTrackingIdResetMode::OnEos => TrackingIdResetMode::OnEos,
            PyTrackingIdResetMode::OnStreamResetAndEos => TrackingIdResetMode::OnStreamResetAndEos,
        }
    }
}

impl From<TrackingIdResetMode> for PyTrackingIdResetMode {
    fn from(p: TrackingIdResetMode) -> Self {
        match p {
            TrackingIdResetMode::None => PyTrackingIdResetMode::None,
            TrackingIdResetMode::OnStreamReset => PyTrackingIdResetMode::OnStreamReset,
            TrackingIdResetMode::OnEos => PyTrackingIdResetMode::OnEos,
            TrackingIdResetMode::OnStreamResetAndEos => PyTrackingIdResetMode::OnStreamResetAndEos,
        }
    }
}

/// Tracker target state in misc (shadow / past-frame) metadata.
#[pyclass(
    from_py_object,
    name = "TrackState",
    module = "savant_rs.nvtracker",
    eq,
    eq_int
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyTrackState {
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

impl From<TrackState> for PyTrackState {
    fn from(s: TrackState) -> Self {
        match s {
            TrackState::Empty => PyTrackState::Empty,
            TrackState::Active => PyTrackState::Active,
            TrackState::Inactive => PyTrackState::Inactive,
            TrackState::Tentative => PyTrackState::Tentative,
            TrackState::Projected => PyTrackState::Projected,
        }
    }
}
