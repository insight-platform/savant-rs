use crate::model_input_scaling::ModelInputScaling;
use crate::roi::RoiKind;
use deepstream_buffers::{SavantIdMetaKind, SharedBuffer};
use parking_lot::Mutex;
use savant_core::primitives::frame::VideoFrameProxy;
use std::collections::HashMap;
use std::sync::Arc;

use super::output::OperatorInferenceOutput;

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
/// Receives the list of [`VideoFrameProxy`]s in the batch. Must return per-frame
/// ROIs and Savant IDs.
pub type BatchFormationCallback =
    Arc<dyn Fn(&[VideoFrameProxy]) -> BatchFormationResult + Send + Sync>;

/// Callback invoked when inference results for a batch are ready.
pub type OperatorResultCallback = Box<dyn FnMut(OperatorInferenceOutput) + Send>;

/// Per-frame pair stored in the pending batch.
pub(super) type FramePair = (VideoFrameProxy, SharedBuffer);

/// A pending batch: original frame/buffer pairs plus the ROI and model
/// configuration needed to construct [`super::scaler::CoordinateScaler`]s when
/// results arrive.
pub(super) struct PendingBatch {
    pub frames: Vec<FramePair>,
    pub rois: Vec<RoiKind>,
    pub model_width: f32,
    pub model_height: f32,
    pub scaling: ModelInputScaling,
}

/// Pending-batch map: batch-ID → [`PendingBatch`].
pub(super) type PendingMap = Arc<Mutex<HashMap<i64, PendingBatch>>>;
