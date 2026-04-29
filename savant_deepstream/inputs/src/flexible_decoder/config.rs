//! Configuration for [`super::FlexibleDecoder`].

use deepstream_decoders::DecoderConfig;
use savant_core::primitives::frame::VideoFrame;
use std::sync::Arc;
use std::time::Duration;

/// Default idle timeout (max gap between outputs during graceful drain).
const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(1);

/// Default max buffered frames during H.264/HEVC detection.
const DEFAULT_DETECT_BUFFER_LIMIT: usize = 30;

/// Callback signature for user-supplied transformations of a resolved
/// [`DecoderConfig`] right before the underlying decoder is built.
///
/// The callback takes ownership of the [`DecoderConfig`] and returns the
/// (possibly modified) configuration. Receiving the [`VideoFrame`]
/// allows the callback to read per-stream metadata (source id, FPS,
/// attributes, size, …) when deciding how to tune the decoder.
///
/// Fires on each new decoder activation (first submit and each subsequent
/// codec/resolution change). It is not invoked for frames processed by an
/// already-active decoder — the config has already been baked into the
/// running hardware decoder at that point.
pub type DecoderConfigCallback =
    Arc<dyn Fn(DecoderConfig, &VideoFrame) -> DecoderConfig + Send + Sync + 'static>;

/// Configuration for a [`super::FlexibleDecoder`].
///
/// Every field maps 1:1 to an underlying API parameter. Pool dimensions and
/// pixel format (`RGBA`) are derived from the [`VideoFrame`] at decoder
/// activation time.
///
/// [`VideoFrame`]: savant_core::primitives::frame::VideoFrame
#[derive(Clone)]
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
    /// Optional callback invoked right before each new decoder is created
    /// so the caller can override per-codec decoder tunables (e.g.
    /// `low_latency`, `num_extra_surfaces`).
    pub decoder_config_callback: Option<DecoderConfigCallback>,
}

impl std::fmt::Debug for FlexibleDecoderConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlexibleDecoderConfig")
            .field("source_id", &self.source_id)
            .field("gpu_id", &self.gpu_id)
            .field("pool_size", &self.pool_size)
            .field("idle_timeout", &self.idle_timeout)
            .field("detect_buffer_limit", &self.detect_buffer_limit)
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

impl FlexibleDecoderConfig {
    /// Build with required parameters; optional fields use defaults.
    pub fn new(source_id: impl Into<String>, gpu_id: u32, pool_size: u32) -> Self {
        Self {
            source_id: source_id.into(),
            gpu_id,
            pool_size,
            idle_timeout: DEFAULT_IDLE_TIMEOUT,
            detect_buffer_limit: DEFAULT_DETECT_BUFFER_LIMIT,
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

    /// Install a decoder config callback. The callback takes ownership of a
    /// resolved [`DecoderConfig`] and returns a (possibly modified)
    /// [`DecoderConfig`]; it runs inside [`super::FlexibleDecoder::submit`]
    /// for every new decoder activation.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use deepstream_decoders::{DecoderConfig, Vp8DecoderConfig};
    /// use deepstream_inputs::flexible_decoder::FlexibleDecoderConfig;
    ///
    /// let cfg = FlexibleDecoderConfig::new("cam-1", 0, 4)
    ///     .decoder_config_callback(|cfg, _frame| match cfg {
    ///         DecoderConfig::Vp8(c) => DecoderConfig::Vp8(c.num_extra_surfaces(4)),
    ///         other => other,
    ///     });
    /// ```
    pub fn decoder_config_callback<F>(mut self, cb: F) -> Self
    where
        F: Fn(DecoderConfig, &VideoFrame) -> DecoderConfig + Send + Sync + 'static,
    {
        self.decoder_config_callback = Some(Arc::new(cb));
        self
    }

    /// Install a pre-built decoder config callback without re-wrapping it in
    /// an [`Arc`]. Useful when the closure is already owned by a shared
    /// handle (e.g. when re-sharing it across [`super::FlexibleDecoder`]s).
    pub fn decoder_config_callback_arc(mut self, cb: DecoderConfigCallback) -> Self {
        self.decoder_config_callback = Some(cb);
        self
    }

    /// Return the installed callback (if any).
    pub fn decoder_config_callback_ref(&self) -> Option<&DecoderConfigCallback> {
        self.decoder_config_callback.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepstream_decoders::{DecoderConfig, Vp8DecoderConfig};
    use savant_core::primitives::frame::{VideoFrameContent, VideoFrameTranscodingMethod};
    use savant_core::primitives::video_codec::VideoCodec;

    fn make_frame(source_id: &str) -> VideoFrame {
        VideoFrame::new(
            source_id,
            (30, 1),
            64,
            48,
            VideoFrameContent::None,
            VideoFrameTranscodingMethod::Copy,
            Some(VideoCodec::Vp8),
            None,
            (1, 30),
            0,
            None,
            None,
        )
        .expect("test frame")
    }

    #[test]
    fn default_callback_is_none() {
        let cfg = FlexibleDecoderConfig::new("src", 0, 4);
        assert!(cfg.decoder_config_callback.is_none());
    }

    #[test]
    fn fluent_setter_stores_callback() {
        let cfg = FlexibleDecoderConfig::new("src", 0, 4).decoder_config_callback(|c, _f| c);
        assert!(cfg.decoder_config_callback.is_some());
    }

    #[test]
    fn callback_transforms_config() {
        let cfg =
            FlexibleDecoderConfig::new("src", 0, 4).decoder_config_callback(
                |cfg, _frame| match cfg {
                    DecoderConfig::Vp8(c) => DecoderConfig::Vp8(c.num_extra_surfaces(7)),
                    other => other,
                },
            );
        let frame = make_frame("src");
        let input = DecoderConfig::Vp8(Vp8DecoderConfig::default());
        let cb = cfg.decoder_config_callback.as_ref().unwrap();
        let out = cb(input, &frame);
        match out {
            DecoderConfig::Vp8(c) => assert_eq!(c.num_extra_surfaces, Some(7)),
            _ => panic!("expected Vp8"),
        }
    }

    #[test]
    fn callback_arc_sets_same_arc() {
        let arc: DecoderConfigCallback = Arc::new(|c, _f| c);
        let cfg = FlexibleDecoderConfig::new("src", 0, 4).decoder_config_callback_arc(arc.clone());
        let stored = cfg.decoder_config_callback.as_ref().unwrap();
        assert!(Arc::ptr_eq(&arc, stored));
    }
}
