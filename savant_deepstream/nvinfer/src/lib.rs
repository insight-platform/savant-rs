//! GStreamer pipeline wrapper for DeepStream nvinfer inference.
//!
//! Builds `appsrc ! queue ! nvinfer ! appsink` (or without queue when depth=0),
//! accepts batched NvBufSurface buffers with IDs, and invokes a callback when
//! inference completes with per-element output tensors by name.

pub mod batch_meta_builder;
pub mod config;
pub mod error;
pub mod nvinfer_types;
pub mod output;
pub mod pipeline;

pub use batch_meta_builder::attach_batch_meta;
pub use config::NvInferConfig;
pub use deepstream::{InferDims, InferTensorMeta};
pub use error::{NvInferError, Result};
pub use nvinfer_types::DataType;
pub use output::{BatchInferenceOutput, ElementOutput, TensorView};
pub use pipeline::NvInfer;
