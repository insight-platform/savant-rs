use crate::config::NvInferConfig;
use std::time::Duration;

/// Configuration for the [`super::NvInferBatchingOperator`] batching layer.
///
/// Embeds a full [`NvInferConfig`] which is forwarded to the inner
/// [`crate::pipeline::NvInfer`] pipeline.  The GPU device ID for batch
/// construction is taken from [`NvInferConfig::gpu_id`].
#[derive(Debug, Clone)]
pub struct NvInferBatchingOperatorConfig {
    /// Maximum batch size; triggers inference when reached.
    pub max_batch_size: usize,
    /// When `false`, rejects frames whose `source_id` is already present in
    /// the pending batch.
    pub same_source_allowed: bool,
    /// Maximum time to wait before submitting a partial batch.
    pub max_batch_wait: Duration,
    /// Configuration forwarded to the inner [`crate::pipeline::NvInfer`] pipeline.
    pub nvinfer: NvInferConfig,
}
