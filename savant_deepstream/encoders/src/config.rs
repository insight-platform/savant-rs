//! Configuration model for [`NvEncoder`](crate::pipeline::NvEncoder).
//!
//! Two layers, mirroring [`deepstream_decoders::config`]:
//!
//! * A per-codec config struct ([`H264EncoderConfig`], [`HevcEncoderConfig`],
//!   [`Av1EncoderConfig`], [`JpegEncoderConfig`], [`PngEncoderConfig`],
//!   [`RawEncoderConfig`]) describing the *bitstream-level* knobs
//!   (resolution, framerate, input format, platform-gated properties).
//! * An aggregate [`EncoderConfig`] enum that dispatches by codec.
//! * A runtime-layer [`NvEncoderConfig`] that wraps the per-codec payload
//!   with framework-level knobs (name, GPU id, channel capacities,
//!   operation timeout).
//!
//! Platform-specific tuning structs (e.g. `H264DgpuProps`,
//! [`H264JetsonProps`]) still live in [`crate::properties`]; the per-codec
//! configs here pick the matching variant at compile time using
//! `#[cfg(target_arch = "aarch64")]`, exactly as the decoder configs do.
//!
//! [`deepstream_decoders::config`]: https://docs.rs/deepstream_decoders

use std::time::Duration;

use deepstream_buffers::{NvBufSurfaceMemType, VideoFormat};
use savant_core::primitives::video_codec::VideoCodec;

#[cfg(not(target_arch = "aarch64"))]
use crate::properties::{Av1DgpuProps, H264DgpuProps, HevcDgpuProps};
#[cfg(target_arch = "aarch64")]
use crate::properties::{Av1JetsonProps, H264JetsonProps, HevcJetsonProps};
use crate::properties::{JpegProps, PngProps, RawProps};

// ─── Per-codec configs ──────────────────────────────────────────────

/// Default framerate numerator (30 fps).
pub const DEFAULT_FPS_NUM: i32 = 30;
/// Default framerate denominator.
pub const DEFAULT_FPS_DEN: i32 = 1;

/// H.264 encoder config (`nvv4l2h264enc`).
#[derive(Debug, Clone)]
pub struct H264EncoderConfig {
    pub width: u32,
    pub height: u32,
    pub format: VideoFormat,
    pub fps_num: i32,
    pub fps_den: i32,
    /// Platform-specific tuning properties. The field type is selected at
    /// compile time (`H264DgpuProps` on dGPU, `H264JetsonProps` on Jetson).
    #[cfg(not(target_arch = "aarch64"))]
    pub props: Option<H264DgpuProps>,
    #[cfg(target_arch = "aarch64")]
    pub props: Option<H264JetsonProps>,
}

impl H264EncoderConfig {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            format: VideoFormat::NV12,
            fps_num: DEFAULT_FPS_NUM,
            fps_den: DEFAULT_FPS_DEN,
            props: None,
        }
    }

    pub fn format(mut self, format: VideoFormat) -> Self {
        self.format = format;
        self
    }

    pub fn fps(mut self, num: i32, den: i32) -> Self {
        assert!(den > 0, "fps denominator must be positive");
        self.fps_num = num;
        self.fps_den = den;
        self
    }

    #[cfg(not(target_arch = "aarch64"))]
    pub fn props(mut self, props: H264DgpuProps) -> Self {
        self.props = Some(props);
        self
    }

    #[cfg(target_arch = "aarch64")]
    pub fn props(mut self, props: H264JetsonProps) -> Self {
        self.props = Some(props);
        self
    }

    pub(crate) fn to_gst_pairs(&self) -> Vec<(&'static str, String)> {
        self.props
            .as_ref()
            .map(|p| p.to_gst_pairs())
            .unwrap_or_default()
    }
}

/// HEVC encoder config (`nvv4l2h265enc`).
#[derive(Debug, Clone)]
pub struct HevcEncoderConfig {
    pub width: u32,
    pub height: u32,
    pub format: VideoFormat,
    pub fps_num: i32,
    pub fps_den: i32,
    #[cfg(not(target_arch = "aarch64"))]
    pub props: Option<HevcDgpuProps>,
    #[cfg(target_arch = "aarch64")]
    pub props: Option<HevcJetsonProps>,
}

impl HevcEncoderConfig {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            format: VideoFormat::NV12,
            fps_num: DEFAULT_FPS_NUM,
            fps_den: DEFAULT_FPS_DEN,
            props: None,
        }
    }

    pub fn format(mut self, format: VideoFormat) -> Self {
        self.format = format;
        self
    }

    pub fn fps(mut self, num: i32, den: i32) -> Self {
        assert!(den > 0, "fps denominator must be positive");
        self.fps_num = num;
        self.fps_den = den;
        self
    }

    #[cfg(not(target_arch = "aarch64"))]
    pub fn props(mut self, props: HevcDgpuProps) -> Self {
        self.props = Some(props);
        self
    }

    #[cfg(target_arch = "aarch64")]
    pub fn props(mut self, props: HevcJetsonProps) -> Self {
        self.props = Some(props);
        self
    }

    pub(crate) fn to_gst_pairs(&self) -> Vec<(&'static str, String)> {
        self.props
            .as_ref()
            .map(|p| p.to_gst_pairs())
            .unwrap_or_default()
    }
}

/// AV1 encoder config (`nvv4l2av1enc`).
#[derive(Debug, Clone)]
pub struct Av1EncoderConfig {
    pub width: u32,
    pub height: u32,
    pub format: VideoFormat,
    pub fps_num: i32,
    pub fps_den: i32,
    #[cfg(not(target_arch = "aarch64"))]
    pub props: Option<Av1DgpuProps>,
    #[cfg(target_arch = "aarch64")]
    pub props: Option<Av1JetsonProps>,
}

impl Av1EncoderConfig {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            format: VideoFormat::NV12,
            fps_num: DEFAULT_FPS_NUM,
            fps_den: DEFAULT_FPS_DEN,
            props: None,
        }
    }

    pub fn format(mut self, format: VideoFormat) -> Self {
        self.format = format;
        self
    }

    pub fn fps(mut self, num: i32, den: i32) -> Self {
        assert!(den > 0, "fps denominator must be positive");
        self.fps_num = num;
        self.fps_den = den;
        self
    }

    #[cfg(not(target_arch = "aarch64"))]
    pub fn props(mut self, props: Av1DgpuProps) -> Self {
        self.props = Some(props);
        self
    }

    #[cfg(target_arch = "aarch64")]
    pub fn props(mut self, props: Av1JetsonProps) -> Self {
        self.props = Some(props);
        self
    }

    pub(crate) fn to_gst_pairs(&self) -> Vec<(&'static str, String)> {
        self.props
            .as_ref()
            .map(|p| p.to_gst_pairs())
            .unwrap_or_default()
    }
}

/// JPEG encoder config (`nvjpegenc`).
#[derive(Debug, Clone)]
pub struct JpegEncoderConfig {
    pub width: u32,
    pub height: u32,
    pub format: VideoFormat,
    pub fps_num: i32,
    pub fps_den: i32,
    pub props: Option<JpegProps>,
}

impl JpegEncoderConfig {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            format: VideoFormat::I420,
            fps_num: DEFAULT_FPS_NUM,
            fps_den: DEFAULT_FPS_DEN,
            props: None,
        }
    }

    pub fn format(mut self, format: VideoFormat) -> Self {
        self.format = format;
        self
    }

    pub fn fps(mut self, num: i32, den: i32) -> Self {
        assert!(den > 0, "fps denominator must be positive");
        self.fps_num = num;
        self.fps_den = den;
        self
    }

    pub fn props(mut self, props: JpegProps) -> Self {
        self.props = Some(props);
        self
    }

    pub(crate) fn to_gst_pairs(&self) -> Vec<(&'static str, String)> {
        self.props
            .as_ref()
            .map(|p| p.to_gst_pairs())
            .unwrap_or_default()
    }
}

/// PNG encoder config (`pngenc`, CPU-based).
#[derive(Debug, Clone)]
pub struct PngEncoderConfig {
    pub width: u32,
    pub height: u32,
    /// Source format. `pngenc` operates on CPU memory; leave at `RGBA`
    /// unless you really know what you are doing.
    pub format: VideoFormat,
    pub fps_num: i32,
    pub fps_den: i32,
    pub props: Option<PngProps>,
}

impl PngEncoderConfig {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            format: VideoFormat::RGBA,
            fps_num: DEFAULT_FPS_NUM,
            fps_den: DEFAULT_FPS_DEN,
            props: None,
        }
    }

    pub fn format(mut self, format: VideoFormat) -> Self {
        self.format = format;
        self
    }

    pub fn fps(mut self, num: i32, den: i32) -> Self {
        assert!(den > 0, "fps denominator must be positive");
        self.fps_num = num;
        self.fps_den = den;
        self
    }

    pub fn props(mut self, props: PngProps) -> Self {
        self.props = Some(props);
        self
    }

    pub(crate) fn to_gst_pairs(&self) -> Vec<(&'static str, String)> {
        self.props
            .as_ref()
            .map(|p| p.to_gst_pairs())
            .unwrap_or_default()
    }
}

/// Raw pseudoencoder config (RGBA / RGB / NV12 GPU→CPU download).
#[derive(Debug, Clone)]
pub struct RawEncoderConfig {
    pub width: u32,
    pub height: u32,
    pub format: VideoFormat,
    pub fps_num: i32,
    pub fps_den: i32,
    pub props: Option<RawProps>,
}

impl RawEncoderConfig {
    pub fn new(width: u32, height: u32, format: VideoFormat) -> Self {
        Self {
            width,
            height,
            format,
            fps_num: DEFAULT_FPS_NUM,
            fps_den: DEFAULT_FPS_DEN,
            props: None,
        }
    }

    pub fn fps(mut self, num: i32, den: i32) -> Self {
        assert!(den > 0, "fps denominator must be positive");
        self.fps_num = num;
        self.fps_den = den;
        self
    }

    pub fn props(mut self, props: RawProps) -> Self {
        self.props = Some(props);
        self
    }

    pub(crate) fn to_gst_pairs(&self) -> Vec<(&'static str, String)> {
        self.props
            .as_ref()
            .map(|p| p.to_gst_pairs())
            .unwrap_or_default()
    }
}

// ─── EncoderConfig enum ─────────────────────────────────────────────

/// Per-codec encoder configuration.
///
/// One variant per supported codec. Selects the correct pipeline element
/// chain and the platform-gated property struct at compile time.
#[derive(Debug, Clone)]
pub enum EncoderConfig {
    H264(H264EncoderConfig),
    Hevc(HevcEncoderConfig),
    Av1(Av1EncoderConfig),
    Jpeg(JpegEncoderConfig),
    Png(PngEncoderConfig),
    RawRgba(RawEncoderConfig),
    RawRgb(RawEncoderConfig),
    RawNv12(RawEncoderConfig),
}

impl EncoderConfig {
    pub fn codec(&self) -> VideoCodec {
        match self {
            Self::H264(_) => VideoCodec::H264,
            Self::Hevc(_) => VideoCodec::Hevc,
            Self::Av1(_) => VideoCodec::Av1,
            Self::Jpeg(_) => VideoCodec::Jpeg,
            Self::Png(_) => VideoCodec::Png,
            Self::RawRgba(_) => VideoCodec::RawRgba,
            Self::RawRgb(_) => VideoCodec::RawRgb,
            Self::RawNv12(_) => VideoCodec::RawNv12,
        }
    }

    pub fn width(&self) -> u32 {
        match self {
            Self::H264(c) => c.width,
            Self::Hevc(c) => c.width,
            Self::Av1(c) => c.width,
            Self::Jpeg(c) => c.width,
            Self::Png(c) => c.width,
            Self::RawRgba(c) | Self::RawRgb(c) | Self::RawNv12(c) => c.width,
        }
    }

    pub fn height(&self) -> u32 {
        match self {
            Self::H264(c) => c.height,
            Self::Hevc(c) => c.height,
            Self::Av1(c) => c.height,
            Self::Jpeg(c) => c.height,
            Self::Png(c) => c.height,
            Self::RawRgba(c) | Self::RawRgb(c) | Self::RawNv12(c) => c.height,
        }
    }

    pub fn format(&self) -> VideoFormat {
        match self {
            Self::H264(c) => c.format,
            Self::Hevc(c) => c.format,
            Self::Av1(c) => c.format,
            Self::Jpeg(c) => c.format,
            Self::Png(c) => c.format,
            Self::RawRgba(c) | Self::RawRgb(c) | Self::RawNv12(c) => c.format,
        }
    }

    pub fn fps(&self) -> (i32, i32) {
        match self {
            Self::H264(c) => (c.fps_num, c.fps_den),
            Self::Hevc(c) => (c.fps_num, c.fps_den),
            Self::Av1(c) => (c.fps_num, c.fps_den),
            Self::Jpeg(c) => (c.fps_num, c.fps_den),
            Self::Png(c) => (c.fps_num, c.fps_den),
            Self::RawRgba(c) | Self::RawRgb(c) | Self::RawNv12(c) => (c.fps_num, c.fps_den),
        }
    }

    /// GStreamer key/value pairs for the encoder element. Raw / PNG
    /// configs return the `pngenc` / pseudoencoder knobs (may be empty).
    pub fn to_gst_pairs(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::H264(c) => c.to_gst_pairs(),
            Self::Hevc(c) => c.to_gst_pairs(),
            Self::Av1(c) => c.to_gst_pairs(),
            Self::Jpeg(c) => c.to_gst_pairs(),
            Self::Png(c) => c.to_gst_pairs(),
            Self::RawRgba(c) | Self::RawRgb(c) | Self::RawNv12(c) => c.to_gst_pairs(),
        }
    }
}

// ─── NvEncoderConfig ────────────────────────────────────────────────

/// Runtime configuration for the channel-based
/// [`NvEncoder`](crate::pipeline::NvEncoder).
///
/// Mirrors [`deepstream_decoders::NvDecoderConfig`](https://docs.rs/deepstream_decoders/).
#[derive(Debug, Clone)]
pub struct NvEncoderConfig {
    pub name: String,
    pub gpu_id: u32,
    pub mem_type: NvBufSurfaceMemType,
    pub encoder: EncoderConfig,
    pub input_channel_capacity: usize,
    pub output_channel_capacity: usize,
    pub operation_timeout: Duration,
    pub drain_poll_interval: Duration,
    /// When `Some`, the inner [`savant_gstreamer::pipeline::GstPipeline`]
    /// auto-invokes
    /// [`flush_idle`](savant_gstreamer::pipeline::GstPipeline::flush_idle)
    /// every interval in a background thread.
    ///
    /// Ensures per-source EOS markers escape the encoder element without
    /// requiring an explicit caller flush or a full graceful_shutdown,
    /// mirroring `deepstream_decoders::NvDecoderConfig::idle_flush_interval`.
    pub idle_flush_interval: Option<Duration>,
}

impl NvEncoderConfig {
    pub fn new(gpu_id: u32, encoder: EncoderConfig) -> Self {
        Self {
            name: String::new(),
            gpu_id,
            mem_type: NvBufSurfaceMemType::Default,
            encoder,
            input_channel_capacity: 16,
            output_channel_capacity: 16,
            operation_timeout: Duration::from_secs(30),
            drain_poll_interval: Duration::from_millis(100),
            idle_flush_interval: Some(Duration::from_millis(10)),
        }
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn mem_type(mut self, mem_type: NvBufSurfaceMemType) -> Self {
        self.mem_type = mem_type;
        self
    }

    pub fn input_channel_capacity(mut self, capacity: usize) -> Self {
        self.input_channel_capacity = capacity;
        self
    }

    pub fn output_channel_capacity(mut self, capacity: usize) -> Self {
        self.output_channel_capacity = capacity;
        self
    }

    pub fn operation_timeout(mut self, timeout: Duration) -> Self {
        self.operation_timeout = timeout;
        self
    }

    pub fn drain_poll_interval(mut self, interval: Duration) -> Self {
        self.drain_poll_interval = interval;
        self
    }

    /// Enable or disable the auto-flush thread for pending custom
    /// downstream events.  See [`NvEncoderConfig::idle_flush_interval`].
    pub fn idle_flush_interval(mut self, interval: Option<Duration>) -> Self {
        self.idle_flush_interval = interval;
        self
    }
}
