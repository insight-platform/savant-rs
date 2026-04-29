//! Higher-level batching layer over [`NvInfer`](crate::pipeline::NvInfer).
//!
//! [`NvInferBatchingOperator`] accepts `(VideoFrame, SharedBuffer)` pairs
//! one at a time, accumulates them into batches according to configurable
//! policies, forms a [`NonUniformBatch`](deepstream_buffers::NonUniformBatch),
//! and submits to the underlying [`NvInfer`] pipeline.  Results are mapped back
//! to the original frame/buffer pairs and delivered via a user-supplied
//! callback.

mod config;
mod operator;
mod output;
pub mod scaler;
mod submit;
mod types;

pub use config::{NvInferBatchingOperatorConfig, NvInferBatchingOperatorConfigBuilder};
pub use operator::NvInferBatchingOperator;
pub use output::{
    OperatorElement, OperatorFrameOutput, OperatorInferenceOutput, OperatorOutput, SealedDeliveries,
};
pub use scaler::CoordinateScaler;
pub use types::{BatchFormationCallback, BatchFormationResult, OperatorResultCallback};

#[cfg(test)]
mod tests;
