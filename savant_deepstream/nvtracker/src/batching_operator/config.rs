use crate::config::NvTrackerConfig;
use std::time::Duration;

/// Configuration for the [`super::NvTrackerBatchingOperator`] batching layer.
#[derive(Debug, Clone)]
pub struct NvTrackerBatchingOperatorConfig {
    /// Maximum batch size; triggers tracking when reached.
    pub max_batch_size: usize,
    /// Maximum time to wait before submitting a partial batch.
    pub max_batch_wait: Duration,
    /// Configuration forwarded to the inner [`crate::pipeline::NvTracker`] pipeline.
    pub nvtracker: NvTrackerConfig,
    /// Maximum time a submitted batch can remain in `pending_batches`
    /// before the operator enters a terminal failed state. Default: 60 s.
    pub pending_batch_timeout: Duration,
}

/// Builder for [`NvTrackerBatchingOperatorConfig`].
pub struct NvTrackerBatchingOperatorConfigBuilder {
    nvtracker: NvTrackerConfig,
    max_batch_size: usize,
    max_batch_wait: Duration,
    pending_batch_timeout: Duration,
}

impl NvTrackerBatchingOperatorConfig {
    /// Create a new builder from an [`NvTrackerConfig`].
    ///
    /// Defaults:
    /// - `max_batch_size`: 1
    /// - `max_batch_wait`: 50ms
    pub fn builder(nvtracker_config: NvTrackerConfig) -> NvTrackerBatchingOperatorConfigBuilder {
        NvTrackerBatchingOperatorConfigBuilder {
            nvtracker: nvtracker_config,
            max_batch_size: 1,
            max_batch_wait: Duration::from_millis(50),
            pending_batch_timeout: Duration::from_secs(60),
        }
    }
}

impl NvTrackerBatchingOperatorConfigBuilder {
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

    /// Set the pending batch timeout.
    pub fn pending_batch_timeout(mut self, timeout: Duration) -> Self {
        self.pending_batch_timeout = timeout;
        self
    }

    /// Finish building and return the config.
    pub fn build(self) -> NvTrackerBatchingOperatorConfig {
        NvTrackerBatchingOperatorConfig {
            max_batch_size: self.max_batch_size,
            max_batch_wait: self.max_batch_wait,
            nvtracker: self.nvtracker,
            pending_batch_timeout: self.pending_batch_timeout,
        }
    }
}
