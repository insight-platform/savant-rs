//! Configuration for [`super::decoder::MultiStreamDecoder`].

use deepstream_buffers::Interpolation;
use std::time::Duration;

/// Defaults for [`MultiStreamDecoderConfig`].
pub mod defaults {
    pub const MAX_DETECTION_BUFFER: usize = 30;
    pub const PER_STREAM_QUEUE_SIZE: usize = 16;
    pub const IDLE_TIMEOUT_SECS: u64 = 30;
}

/// Whether to emit [`DecoderOutput::Eos`](super::error::DecoderOutput::Eos) before
/// [`DecoderOutput::StreamStopped`](super::error::DecoderOutput::StreamStopped) when a **session**
/// is torn down internally (codec / resolution / timestamp discontinuity).
///
/// This is a **semantic** EOS to the consumer. Session resets drain the old `NvDecoder` via
/// `send_eos` before teardown. User [`EndOfStream`](savant_core::primitives::eos::EndOfStream)
/// still performs a full drain and real decoder EOS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionBoundaryEosPolicy {
    pub on_codec_change: bool,
    pub on_resolution_change: bool,
    pub on_timestamp_regress: bool,
}

impl Default for SessionBoundaryEosPolicy {
    fn default() -> Self {
        Self {
            on_codec_change: true,
            on_resolution_change: true,
            on_timestamp_regress: true,
        }
    }
}

/// Configuration shared by all streams managed by a [`MultiStreamDecoder`](super::decoder::MultiStreamDecoder).
///
/// RGBA buffer pool dimensions are **not** global: each `source_id` gets a pool sized from the
/// activating [`VideoFrameProxy`](savant_core::primitives::frame::VideoFrameProxy) width/height
/// (and rebuilt after resolution-change session resets).
#[derive(Debug, Clone)]
pub struct MultiStreamDecoderConfig {
    pub gpu_id: u32,
    pub output_pool_size: u32,
    /// NvBufSurfTransform filter for NVMM→RGBA copy (1:1 into the stream pool).
    pub interpolation: Interpolation,
    pub idle_timeout: Duration,
    pub max_detection_buffer: usize,
    pub per_stream_queue_size: usize,
    pub session_boundary_eos: SessionBoundaryEosPolicy,
}

impl MultiStreamDecoderConfig {
    /// Build with GPU id and per-stream pool buffer count; other fields use defaults.
    pub fn new(gpu_id: u32, output_pool_size: u32) -> Self {
        Self {
            gpu_id,
            output_pool_size,
            interpolation: Interpolation::Bilinear,
            idle_timeout: Duration::from_secs(defaults::IDLE_TIMEOUT_SECS),
            max_detection_buffer: defaults::MAX_DETECTION_BUFFER,
            per_stream_queue_size: defaults::PER_STREAM_QUEUE_SIZE,
            session_boundary_eos: SessionBoundaryEosPolicy::default(),
        }
    }

    /// Override [`Interpolation`] for the per-source NVMM→RGBA transform.
    pub fn interpolation(mut self, i: Interpolation) -> Self {
        self.interpolation = i;
        self
    }

    pub fn session_boundary_eos(mut self, p: SessionBoundaryEosPolicy) -> Self {
        self.session_boundary_eos = p;
        self
    }

    /// Override idle timeout (used by feeder `recv_timeout` and detecting cleanup).
    pub fn idle_timeout(mut self, d: Duration) -> Self {
        self.idle_timeout = d;
        self
    }

    /// Override max buffered frames while waiting for H.264/HEVC RAP.
    pub fn max_detection_buffer(mut self, n: usize) -> Self {
        self.max_detection_buffer = n;
        self
    }

    /// Bounded crossbeam queue depth per active stream.
    pub fn per_stream_queue_size(mut self, n: usize) -> Self {
        self.per_stream_queue_size = n;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepstream_buffers::Interpolation;

    #[test]
    fn new_sets_defaults() {
        let c = MultiStreamDecoderConfig::new(0, 4);
        assert_eq!(c.max_detection_buffer, defaults::MAX_DETECTION_BUFFER);
        assert_eq!(c.per_stream_queue_size, defaults::PER_STREAM_QUEUE_SIZE);
        assert_eq!(c.session_boundary_eos, SessionBoundaryEosPolicy::default());
        assert_eq!(c.interpolation, Interpolation::Bilinear);
    }
}
