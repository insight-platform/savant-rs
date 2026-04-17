//! DeepStream nvtracker GStreamer pipeline (`appsrc` → `nvtracker` → `appsink`).
//!
//! Callers supply [`TrackedFrame`]s (individual GPU buffers + class-keyed detections);
//! the crate builds a [`NonUniformBatch`](deepstream_buffers::NonUniformBatch) internally
//! and returns [`TrackerOutput`] with current tracks and optional shadow / terminated /
//! past-frame misc metadata.

pub mod batching_operator;
pub mod config;
pub mod detection_meta;
pub mod error;
pub mod output;
pub mod pipeline;
pub mod roi;

pub use batching_operator::{
    NvTrackerBatchingOperator, NvTrackerBatchingOperatorConfig,
    NvTrackerBatchingOperatorConfigBuilder, SealedDeliveries, TrackerBatchFormationCallback,
    TrackerBatchFormationResult, TrackerOperatorFrameOutput, TrackerOperatorOutput,
    TrackerOperatorResultCallback, TrackerOperatorTrackingOutput,
};
pub use config::{NvTrackerConfig, TrackingIdResetMode};
pub use deepstream_buffers::MetaClearPolicy;
pub use deepstream_buffers::SavantIdMetaKind;
pub use detection_meta::attach_detection_meta;
pub use error::{NvTrackerError, Result};
pub use output::{
    extract_tracker_output, MiscTrackCategory, MiscTrackData, MiscTrackFrame, TrackState,
    TrackedObject, TrackerOutput,
};
pub use pipeline::{default_ll_lib_path, NvTracker, NvTrackerOutput, TrackedFrame};
pub use roi::Roi;
