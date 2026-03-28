//! Safe Rust wrappers for NVIDIA DeepStream metadata structures.
//!
//! This crate provides idiomatic Rust access to the core DeepStream metadata
//! types needed for inference pipelines: batch, frame, object, user, and
//! tensor metadata.

pub mod batch_meta;
pub mod error;
pub mod frame_meta;
pub mod infer_tensor_meta;
pub mod object_meta;
pub mod tracker_meta;
pub mod user_meta;

pub use batch_meta::BatchMeta;
pub use error::DeepStreamError;
pub use frame_meta::FrameMeta;
pub use infer_tensor_meta::{InferDims, InferTensorMeta};
pub use object_meta::ObjectMeta;
pub use tracker_meta::{
    target_misc_batch_from_user_meta, TargetMiscBatch, TargetMiscFrame, TargetMiscObject,
    TargetMiscStream, TrackState,
};
pub use user_meta::UserMeta;

/// Result type for DeepStream operations.
pub type Result<T> = std::result::Result<T, DeepStreamError>;

/// Re-export the raw sys crate for advanced usage.
pub use deepstream_sys as sys;
