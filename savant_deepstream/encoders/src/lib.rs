//! GPU-accelerated video encoders and raw frame downloaders using DeepStream
//! NvBufSurface, built on the [`savant_gstreamer::pipeline`] framework.
//!
//! This crate mirrors the sibling [`deepstream_decoders`] crate: a channel-based
//! [`NvEncoder`] submits buffers and receives [`NvEncoderOutput`] asynchronously,
//! with in-band per-source EOS support via
//! [`NvEncoder::send_source_eos`](crate::pipeline::NvEncoder::send_source_eos).
//!
//! # Design
//!
//! - **No B-frames**: the encoder forcibly disables B-frames on the underlying
//!   GStreamer element regardless of its defaults; the typed property API
//!   offers no B-frame knob.
//! - **Monotonic PTS**: submitted frames must have strictly ascending PTS.
//! - **Integrated buffer management**: the encoder owns an NVMM buffer pool
//!   exposed via [`generator()`](crate::pipeline::NvEncoder::generator).
//! - **Typed config**: [`EncoderConfig`] is an enum with one variant per codec;
//!   each variant carries geometry + fps + a platform-gated property struct.
//! - **Runtime config**: [`NvEncoderConfig`] wraps [`EncoderConfig`] with
//!   framework knobs (channel capacities, operation timeout, GPU id, …).
//!
//! # Example
//!
//! ```rust,no_run
//! use deepstream_encoders::prelude::*;
//! use std::time::Duration;
//!
//! gstreamer::init().unwrap();
//! cuda_init(0).unwrap();
//!
//! # #[cfg(not(target_arch = "aarch64"))]
//! let encoder_cfg = EncoderConfig::H264(
//!     H264EncoderConfig::new(1920, 1080)
//!         .format(VideoFormat::NV12)
//!         .props(H264DgpuProps {
//!             bitrate: Some(8_000_000),
//!             profile: Some(H264Profile::High),
//!             ..Default::default()
//!         }),
//! );
//! # #[cfg(target_arch = "aarch64")]
//! # let encoder_cfg = EncoderConfig::H264(H264EncoderConfig::new(1920, 1080));
//!
//! let cfg = NvEncoderConfig::new(0, encoder_cfg)
//!     .name("demo")
//!     .operation_timeout(Duration::from_secs(5));
//! let encoder = NvEncoder::new(cfg).unwrap();
//!
//! // Submit/receive loop:
//! // for i in 0..10u128 {
//! //     let shared = encoder.generator().lock().acquire(Some(i)).unwrap();
//! //     let buffer = shared.into_buffer().expect("sole owner");
//! //     encoder.submit_frame(buffer, i, i as u64 * 33_333_333, Some(33_333_333)).unwrap();
//! // }
//! // encoder.graceful_shutdown(Some(Duration::from_secs(5)), |_| {}).unwrap();
//! ```

pub mod config;
pub(crate) mod error;
pub mod pipeline;
pub mod prelude;
pub mod properties;

// ─── Public re-exports ──────────────────────────────────────────────

pub use config::{
    Av1EncoderConfig, EncoderConfig, H264EncoderConfig, HevcEncoderConfig, JpegEncoderConfig,
    NvEncoderConfig, PngEncoderConfig, RawEncoderConfig,
};
pub use error::EncoderError;
pub use pipeline::{NvEncoder, NvEncoderExt, NvEncoderOutput};

pub use deepstream_buffers::{
    cuda_init, BufferGenerator, NvBufSurfaceMemType, SharedBuffer, SurfaceView,
    UniformBatchGenerator, VideoFormat,
};
pub use savant_gstreamer::Codec;

// Aggregate property enum kept for generic handling (used by the Python
// bindings and JSON round-tripping).
pub use properties::EncoderProperties;

// ─── EncodedFrame ───────────────────────────────────────────────────

/// A single encoded frame produced by [`NvEncoder`].
#[derive(Debug, Clone)]
pub struct EncodedFrame {
    /// User-defined frame identifier (passed via
    /// [`NvEncoder::submit_frame`]).
    ///
    /// `None` when the encoded buffer could not be matched back to a
    /// submitted frame (e.g. spurious codec-header packets with no
    /// correlatable timestamps).
    pub frame_id: Option<u128>,
    /// Presentation timestamp in nanoseconds.
    pub pts_ns: u64,
    /// Decode timestamp in nanoseconds (if set by the encoder).
    pub dts_ns: Option<u64>,
    /// Duration in nanoseconds (if known).
    pub duration_ns: Option<u64>,
    /// Encoded bitstream data, or tightly-packed raw pixel data for
    /// [`Codec::RawRgba`] / [`Codec::RawRgb`] / [`Codec::RawNv12`].
    pub data: Vec<u8>,
    /// Codec used to encode this frame.
    pub codec: Codec,
    /// `true` when this is an intra-coded (key) frame.
    pub keyframe: bool,
    /// Time base `(num, den)` for interpreting timestamps.  Always
    /// `(1, 1_000_000_000)` — GStreamer operates in nanoseconds.
    pub time_base: (i32, i32),
}
