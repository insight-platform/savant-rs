use crate::roi::Roi;
use deepstream_buffers::{SavantIdMetaKind, SharedBuffer};
use parking_lot::Mutex;
use savant_core::primitives::frame::VideoFrameProxy;
use std::collections::HashMap;
use std::sync::Arc;

use super::output::TrackerOperatorOutput;

/// Result returned by the batch formation callback.
pub struct TrackerBatchFormationResult {
    /// Per-frame Savant IDs for
    /// [`deepstream_buffers::NonUniformBatch::finalize`].
    pub ids: Vec<SavantIdMetaKind>,
    /// Per-frame ROIs keyed by class id (index = frame index).
    ///
    /// Must have exactly the same length as the input frame slice.
    pub rois: Vec<HashMap<i32, Vec<Roi>>>,
}

/// Callback invoked when a batch is formed, before submission to NvTracker.
///
/// Receives the list of [`VideoFrameProxy`] values in the batch. Must return
/// per-frame ROIs and optional Savant IDs.
pub type TrackerBatchFormationCallback =
    Arc<dyn Fn(&[VideoFrameProxy]) -> TrackerBatchFormationResult + Send + Sync>;

/// Callback invoked when tracking results for a batch are ready.
pub type TrackerOperatorResultCallback = Box<dyn FnMut(TrackerOperatorOutput) + Send>;

/// Per-frame pair stored in the pending batch.
pub(super) type FramePair = (VideoFrameProxy, SharedBuffer);

/// A pending batch: original frame/buffer pairs awaiting callback correlation.
pub(super) struct PendingBatch {
    pub frames: Vec<FramePair>,
    pub frame_nums: Vec<u32>,
}

/// Pending-batch map: batch-ID → [`PendingBatch`].
pub(super) type PendingMap = Arc<Mutex<HashMap<u128, PendingBatch>>>;
