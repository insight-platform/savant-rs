//! Configuration for [`super::FlexibleDecoderPool`].

use crate::flexible_decoder::{DecoderConfigCallback, FlexibleDecoderConfig};
use deepstream_decoders::DecoderConfig;
use savant_core::primitives::frame::VideoFrame;
use std::sync::Arc;
use std::time::Duration;

/// Default idle timeout (max gap between outputs during graceful drain).
const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(1);

/// Default max buffered frames during H.264/HEVC detection.
const DEFAULT_DETECT_BUFFER_LIMIT: usize = 30;

/// Decision returned by an eviction callback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionDecision {
    /// Remove the decoder from the pool (graceful drain).
    Evict,
    /// Keep the decoder alive, resetting the TTL.
    Keep,
}

/// Configuration for a [`super::FlexibleDecoderPool`].
///
/// Mirrors [`FlexibleDecoderConfig`] but without `source_id` (derived from
/// each incoming frame) and with an additional `eviction_ttl` that governs
/// how long an idle stream stays in the pool before the eviction callback
/// (or automatic eviction) fires.
#[derive(Clone)]
pub struct FlexibleDecoderPoolConfig {
    /// GPU device for every `FlexibleDecoder` created by the pool.
    pub gpu_id: u32,
    /// Number of RGBA buffers allocated per internal decoder pool.
    pub pool_size: u32,
    /// Max time to wait between consecutive outputs during graceful drain.
    pub idle_timeout: Duration,
    /// Max frames buffered while waiting for an H.264/HEVC random access point.
    pub detect_buffer_limit: usize,
    /// How long a stream can be idle before eviction is considered.
    pub eviction_ttl: Duration,
    /// Optional decoder-config transformation callback, propagated to every
    /// per-stream [`FlexibleDecoderConfig`] built by [`Self::to_flexible_config`].
    pub decoder_config_callback: Option<DecoderConfigCallback>,
}

impl std::fmt::Debug for FlexibleDecoderPoolConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlexibleDecoderPoolConfig")
            .field("gpu_id", &self.gpu_id)
            .field("pool_size", &self.pool_size)
            .field("idle_timeout", &self.idle_timeout)
            .field("detect_buffer_limit", &self.detect_buffer_limit)
            .field("eviction_ttl", &self.eviction_ttl)
            .field(
                "decoder_config_callback",
                &self
                    .decoder_config_callback
                    .as_ref()
                    .map(|_| "<callback>")
                    .unwrap_or("None"),
            )
            .finish()
    }
}

impl FlexibleDecoderPoolConfig {
    /// Build with required parameters; optional fields use defaults.
    pub fn new(gpu_id: u32, pool_size: u32, eviction_ttl: Duration) -> Self {
        Self {
            gpu_id,
            pool_size,
            idle_timeout: DEFAULT_IDLE_TIMEOUT,
            detect_buffer_limit: DEFAULT_DETECT_BUFFER_LIMIT,
            eviction_ttl,
            decoder_config_callback: None,
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

    /// Install a decoder config callback. See
    /// [`FlexibleDecoderConfig::decoder_config_callback`] for semantics.
    /// The callback is propagated to every per-stream [`FlexibleDecoderConfig`]
    /// produced by [`Self::to_flexible_config`].
    pub fn decoder_config_callback<F>(mut self, cb: F) -> Self
    where
        F: Fn(DecoderConfig, &VideoFrame) -> DecoderConfig + Send + Sync + 'static,
    {
        self.decoder_config_callback = Some(Arc::new(cb));
        self
    }

    /// Install a pre-built decoder config callback without re-wrapping it in
    /// an [`Arc`].
    pub fn decoder_config_callback_arc(mut self, cb: DecoderConfigCallback) -> Self {
        self.decoder_config_callback = Some(cb);
        self
    }

    /// Produce a [`FlexibleDecoderConfig`] bound to `source_id`, inheriting
    /// all shared parameters from this pool config (including the optional
    /// decoder-config callback).
    pub fn to_flexible_config(&self, source_id: &str) -> FlexibleDecoderConfig {
        let mut cfg = FlexibleDecoderConfig::new(source_id, self.gpu_id, self.pool_size)
            .idle_timeout(self.idle_timeout)
            .detect_buffer_limit(self.detect_buffer_limit);
        if let Some(cb) = self.decoder_config_callback.as_ref() {
            cfg = cfg.decoder_config_callback_arc(cb.clone());
        }
        cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults() {
        let cfg = FlexibleDecoderPoolConfig::new(0, 4, Duration::from_secs(30));
        assert_eq!(cfg.gpu_id, 0);
        assert_eq!(cfg.pool_size, 4);
        assert_eq!(cfg.eviction_ttl, Duration::from_secs(30));
        assert_eq!(cfg.idle_timeout, Duration::from_secs(1));
        assert_eq!(cfg.detect_buffer_limit, 30);
    }

    #[test]
    fn builder_chaining() {
        let cfg = FlexibleDecoderPoolConfig::new(1, 8, Duration::from_secs(60))
            .idle_timeout(Duration::from_millis(500))
            .detect_buffer_limit(50);
        assert_eq!(cfg.gpu_id, 1);
        assert_eq!(cfg.pool_size, 8);
        assert_eq!(cfg.idle_timeout, Duration::from_millis(500));
        assert_eq!(cfg.detect_buffer_limit, 50);
    }

    #[test]
    fn to_flexible_config_correctness() {
        let pool_cfg = FlexibleDecoderPoolConfig::new(2, 6, Duration::from_secs(10))
            .idle_timeout(Duration::from_millis(750))
            .detect_buffer_limit(20);
        let flex = pool_cfg.to_flexible_config("cam-1");
        assert_eq!(flex.source_id, "cam-1");
        assert_eq!(flex.gpu_id, 2);
        assert_eq!(flex.pool_size, 6);
        assert_eq!(flex.idle_timeout, Duration::from_millis(750));
        assert_eq!(flex.detect_buffer_limit, 20);
        assert!(flex.decoder_config_callback.is_none());
    }

    #[test]
    fn default_callback_is_none() {
        let cfg = FlexibleDecoderPoolConfig::new(0, 4, Duration::from_secs(30));
        assert!(cfg.decoder_config_callback.is_none());
    }

    #[test]
    fn decoder_config_callback_propagates_to_flexible_config() {
        let shared: DecoderConfigCallback = Arc::new(|c, _f| c);
        let pool_cfg = FlexibleDecoderPoolConfig::new(0, 4, Duration::from_secs(30))
            .decoder_config_callback_arc(shared.clone());
        let flex = pool_cfg.to_flexible_config("cam-x");
        let installed = flex
            .decoder_config_callback
            .as_ref()
            .expect("callback propagated");
        assert!(Arc::ptr_eq(&shared, installed));
    }
}
