//! Higher-level batching layer over [`NvTracker`](crate::pipeline::NvTracker).
//!
//! [`NvTrackerBatchingOperator`] accepts `(VideoFrameProxy, SharedBuffer)` pairs
//! one at a time, accumulates them into batches according to configurable
//! policies, constructs [`crate::pipeline::TrackedFrame`] values using the
//! batch-formation callback, and submits to the underlying [`NvTracker`]
//! pipeline. Results are mapped back to the original frame/buffer pairs and
//! delivered via a user-supplied callback.

mod config;
mod operator;
mod output;
mod submit;
mod types;

pub use config::{NvTrackerBatchingOperatorConfig, NvTrackerBatchingOperatorConfigBuilder};
pub use operator::NvTrackerBatchingOperator;
pub use output::{SealedDeliveries, TrackerOperatorFrameOutput, TrackerOperatorOutput};
pub use types::{
    TrackerBatchFormationCallback, TrackerBatchFormationResult, TrackerOperatorResultCallback,
};

#[cfg(test)]
mod tests;
