//! Configuration for [`super::FlexibleDecoder`].

use std::time::Duration;

use super::output::{DEFAULT_DETECT_BUFFER_LIMIT, DEFAULT_IDLE_TIMEOUT_SECS};

/// Configuration for a [`super::FlexibleDecoder`].
///
/// Every field maps 1:1 to an underlying API parameter. Pool dimensions and
/// pixel format (`RGBA`) are derived from the [`VideoFrameProxy`] at decoder
/// activation time.
///
/// [`VideoFrameProxy`]: savant_core::primitives::frame::VideoFrameProxy
#[derive(Debug, Clone)]
pub struct FlexibleDecoderConfig {
    /// Bound source_id; frames with a different source_id are rejected.
    pub source_id: String,
    /// GPU device for `NvDecoder` and the RGBA buffer pool.
    pub gpu_id: u32,
    /// Number of RGBA buffers allocated per internal decoder pool.
    /// Maps to `BufferGenerator::builder().min_buffers(n).max_buffers(n)`.
    pub pool_size: u32,
    /// Max time to wait between consecutive outputs during graceful drain
    /// (parameter change or [`FlexibleDecoder::graceful_shutdown`](super::FlexibleDecoder::graceful_shutdown)).
    /// Passed directly as `NvDecoder::graceful_shutdown(idle_timeout: Some(…), …)`.
    pub idle_timeout: Duration,
    /// Max frames buffered while waiting for an H.264/HEVC random access point.
    pub detect_buffer_limit: usize,
}

impl FlexibleDecoderConfig {
    /// Build with required parameters; optional fields use defaults.
    pub fn new(source_id: impl Into<String>, gpu_id: u32, pool_size: u32) -> Self {
        Self {
            source_id: source_id.into(),
            gpu_id,
            pool_size,
            idle_timeout: Duration::from_secs(DEFAULT_IDLE_TIMEOUT_SECS),
            detect_buffer_limit: DEFAULT_DETECT_BUFFER_LIMIT,
        }
    }

    /// Override the idle timeout for graceful drain.
    pub fn idle_timeout(mut self, d: Duration) -> Self {
        self.idle_timeout = d;
        self
    }

    /// Override the detection buffer limit.
    pub fn detect_buffer_limit(mut self, n: usize) -> Self {
        self.detect_buffer_limit = n;
        self
    }
}
