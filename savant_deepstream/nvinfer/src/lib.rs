//! GStreamer pipeline wrapper for DeepStream nvinfer secondary inference.
//!
//! Builds `appsrc ! queue ! nvinfer ! appsink` (or without queue when depth=0),
//! accepts batched NvBufSurface buffers with ROI lists, and invokes a callback
//! when inference completes with per-ROI output tensors by name.

pub mod batch_meta_builder;
pub mod config;
pub mod error;
pub mod meta_clear_policy;
pub mod nvinfer_types;
pub mod output;
pub mod pipeline;
pub mod roi;

pub use batch_meta_builder::attach_batch_meta_with_rois;
pub use config::NvInferConfig;
pub use deepstream::{InferDims, InferTensorMeta};
pub use error::{NvInferError, Result};
pub use meta_clear_policy::MetaClearPolicy;
pub use nvinfer_types::DataType;
pub use output::{BatchInferenceOutput, ElementOutput, TensorView};
pub use pipeline::NvInfer;
pub use roi::Roi;
