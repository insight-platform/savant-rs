//! Errors for the nvtracker pipeline.

use savant_gstreamer::pipeline::PipelineError;
use thiserror::Error;

/// Result type alias for nvtracker operations.
pub type Result<T> = std::result::Result<T, NvTrackerError>;

/// NvTracker failure modes. Every variant carries enough context to diagnose without a debugger.
#[derive(Debug, Error)]
pub enum NvTrackerError {
    #[error("GStreamer initialization failed: {0}")]
    GstInit(String),

    #[error("failed to create GStreamer element '{element}': {reason}")]
    ElementCreationFailed { element: String, reason: String },

    #[error("pipeline error: {0}")]
    PipelineError(String),

    #[error("failed to link GStreamer elements: {chain}")]
    LinkFailed { chain: String },

    #[error("invalid tracker property '{key}': {reason}")]
    InvalidProperty { key: String, reason: String },

    #[error("configuration error: {0}")]
    ConfigError(String),

    #[error("batch formation failed: {0}")]
    BatchFormationFailed(String),

    #[error("batching operator is shut down")]
    OperatorShutdown,

    #[error("batch metadata error in {operation}: {detail}")]
    BatchMetaFailed { operation: String, detail: String },

    #[error("null pointer from DeepStream FFI call '{function}'")]
    NullPointer { function: String },

    #[error(
        "resolution mismatch for source '{source_id}': slot {slot_a} is {w_a}x{h_a} but slot {slot_b} is {w_b}x{h_b}"
    )]
    ResolutionMismatch {
        source_id: String,
        slot_a: u32,
        w_a: u32,
        h_a: u32,
        slot_b: u32,
        w_b: u32,
        h_b: u32,
    },

    #[error("bounding box conversion failed for ROI {roi_id} in slot {slot}: {reason}")]
    BBoxConversion {
        roi_id: i64,
        slot: u32,
        reason: String,
    },

    #[error("slot index {index} out of bounds (num_filled={num_filled}) during {operation}")]
    SlotIndexOutOfBounds {
        index: u32,
        num_filled: u32,
        operation: String,
    },

    #[error(
        "frame_num counter overflow for source pad_index={pad_index} (source_id='{source_id}')"
    )]
    FrameNumOverflow { pad_index: u32, source_id: String },

    #[error("Pipeline entered failed state (operation timeout exceeded)")]
    PipelineFailed,

    #[error("Batching operator entered failed state (pending batch timeout exceeded)")]
    OperatorFailed,

    #[error("buffer is not writable during {operation}")]
    BufferNotWritable { operation: String },

    #[error(
        "SharedBuffer has outstanding references; cannot take exclusive ownership for {operation}"
    )]
    BufferOwnership { operation: String },

    #[error("DeepStream error: {0}")]
    DeepStream(#[from] deepstream::DeepStreamError),

    #[error("GStreamer pipeline framework error: {0}")]
    FrameworkError(#[from] PipelineError),

    #[error("pipeline channel disconnected")]
    ChannelDisconnected,

    #[error("NvTracker is shutting down; no new input accepted")]
    ShuttingDown,
}

impl NvTrackerError {
    pub(crate) fn batch_meta(operation: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::BatchMetaFailed {
            operation: operation.into(),
            detail: detail.into(),
        }
    }
}
