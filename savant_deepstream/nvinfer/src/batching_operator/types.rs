use crate::model_input_scaling::ModelInputScaling;
use crate::roi::RoiKind;
use deepstream_buffers::{SavantIdMetaKind, SharedBuffer};
use parking_lot::Mutex;
use savant_core::primitives::frame::VideoFrame;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use super::output::OperatorOutput;

/// Result returned by the batch formation callback.
pub struct BatchFormationResult {
    /// Per-frame Savant IDs for
    /// [`deepstream_buffers::NonUniformBatch::finalize`].
    pub ids: Vec<SavantIdMetaKind>,
    /// Per-frame ROI specification (index = frame index).
    pub rois: Vec<RoiKind>,
}

/// Callback invoked when a batch is formed, before submission to NvInfer.
///
/// Receives the list of [`VideoFrame`]s in the batch. Must return per-frame
/// ROIs and Savant IDs.
pub type BatchFormationCallback =
    Arc<dyn Fn(&[VideoFrame]) -> BatchFormationResult + Send + Sync>;

/// Callback invoked when inference results for a batch are ready.
pub type OperatorResultCallback = Box<dyn FnMut(OperatorOutput) + Send>;

/// Per-frame pair stored in the pending batch.
pub(super) type FramePair = (VideoFrame, SharedBuffer);

/// A pending batch: original frame/buffer pairs plus the ROI and model
/// configuration needed to construct [`super::scaler::CoordinateScaler`]s when
/// results arrive.
pub(super) struct PendingBatch {
    pub frames: Vec<FramePair>,
    pub rois: Vec<RoiKind>,
    pub model_width: f32,
    pub model_height: f32,
    pub scaling: ModelInputScaling,
    pub submitted_at: Instant,
}

/// Pending-batch map: batch-ID → [`PendingBatch`].
pub(super) type PendingMap = Arc<Mutex<HashMap<u128, PendingBatch>>>;
