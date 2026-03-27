//! Higher-level batching layer over [`NvInfer`](crate::pipeline::NvInfer).
//!
//! [`NvInferBatchingOperator`] accepts `(VideoFrameProxy, SharedBuffer)` pairs
//! one at a time, accumulates them into batches according to configurable
//! policies, forms a [`NonUniformBatch`](deepstream_buffers::NonUniformBatch),
//! and submits to the underlying [`NvInfer`] pipeline.  Results are mapped back
//! to the original frame/buffer pairs and delivered via a user-supplied
//! callback.

mod config;
mod operator;
mod output;
pub mod scaler;
mod state;
mod submit;
mod types;

pub use config::NvInferBatchingOperatorConfig;
pub use operator::NvInferBatchingOperator;
pub use output::{OperatorElement, OperatorFrameOutput, OperatorInferenceOutput};
pub use scaler::CoordinateScaler;
pub use types::{BatchFormationCallback, BatchFormationResult, OperatorResultCallback};

#[cfg(test)]
mod tests;
