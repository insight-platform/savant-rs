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
    /// Maximum time to wait before submitting a partial batch.
    pub max_batch_wait: Duration,
    /// Configuration forwarded to the inner [`crate::pipeline::NvInfer`] pipeline.
    pub nvinfer: NvInferConfig,
}

/// Builder for [`NvInferBatchingOperatorConfig`].
pub struct NvInferBatchingOperatorConfigBuilder {
    nvinfer: NvInferConfig,
    max_batch_size: usize,
    max_batch_wait: Duration,
}

impl NvInferBatchingOperatorConfig {
    /// Create a new builder from an [`NvInferConfig`].
    ///
    /// Defaults:
    /// - `max_batch_size`: 1
    /// - `max_batch_wait`: 50ms
    pub fn builder(nvinfer_config: NvInferConfig) -> NvInferBatchingOperatorConfigBuilder {
        NvInferBatchingOperatorConfigBuilder {
            nvinfer: nvinfer_config,
            max_batch_size: 1,
            max_batch_wait: Duration::from_millis(50),
        }
    }
}

impl NvInferBatchingOperatorConfigBuilder {
    /// Set the maximum batch size.
    pub fn max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size;
        self
    }

    /// Set the maximum wait time before submitting a partial batch.
    pub fn max_batch_wait(mut self, wait: Duration) -> Self {
        self.max_batch_wait = wait;
        self
    }

    /// Finish building and return the config.
    pub fn build(self) -> NvInferBatchingOperatorConfig {
        NvInferBatchingOperatorConfig {
            max_batch_size: self.max_batch_size,
            max_batch_wait: self.max_batch_wait,
            nvinfer: self.nvinfer,
        }
    }
}
