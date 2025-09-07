//! Safe Rust API for NVIDIA DeepStream
//!
//! This crate provides safe, idiomatic Rust wrappers around the DeepStream C API,
//! particularly focusing on metadata structures like `NvDsObjectMeta`.
//!
//! # Example
//!
//! ```rust
//! use deepstream::ObjectMeta;
//!
//! // Note: ObjectMeta is a wrapper around existing DeepStream metadata
//! // You would typically get it from a frame or batch, not create it directly
//! // For demonstration purposes, this shows how to work with an existing instance:
//!
//! // Assuming you have a raw pointer to NvDsObjectMeta
//! // let raw_ptr: *mut deepstream_sys::NvDsObjectMeta = /* ... */;
//! // let mut obj_meta = unsafe { ObjectMeta::from_raw(raw_ptr)? };
//!
//! // Set properties
//! // obj_meta.set_class_id(0);
//! // obj_meta.set_object_id(123);
//! // obj_meta.set_confidence(0.95);
//!
//! // Set bounding box
//! // obj_meta.set_bbox(100.0, 200.0, 300.0, 400.0);
//! ```

pub mod batch_meta;
pub mod error;
pub mod frame_meta;
pub mod infer_context;
pub mod infer_tensor_meta;
pub mod object_meta;
pub mod rect_params;
pub mod types;
pub mod user_meta;

pub use batch_meta::BatchMeta;
pub use error::DeepStreamError;
pub use frame_meta::FrameMeta;
pub use infer_context::{
    BatchInput, BatchOutput, DataType, InferContext, InferContextInitParams, InferFormat,
    LayerInfo, LogLevel, NetworkInfo,
};
pub use infer_tensor_meta::InferTensorMeta;
pub use object_meta::ObjectMeta;
pub use rect_params::RectParams;
pub use types::ColorParams;
pub use user_meta::UserMeta;

/// Result type for DeepStream operations
pub type Result<T> = std::result::Result<T, DeepStreamError>;

/// Re-export the raw sys crate for advanced usage
pub use deepstream_sys as sys;
