//! GPU-accelerated video encoders and raw frame downloaders using DeepStream
//! NvBufSurface.
//!
//! This crate provides a high-level API for hardware-accelerated video
//! encoding (H.264, HEVC/H.265, JPEG, AV1) and raw GPU-to-CPU frame
//! download (RGBA, RGB) backed by NVIDIA DeepStream's NvBufSurface buffer
//! pool.
//!
//! # Design
//!
//! - **No B-frames**: The encoder forcibly disables B-frames on the
//!   underlying GStreamer element (regardless of its defaults) and the
//!   typed property API offers no B-frame knob.
//! - **Monotonic PTS**: Each submitted frame's PTS must be strictly greater
//!   than the previous frame's PTS. The encoder raises an error on
//!   reordering.
//! - **Integrated buffer management**: The encoder owns an
//!   [`UniformBatchGenerator`](deepstream_buffers::UniformBatchGenerator)
//!   that provides NVMM GPU buffers for zero-copy rendering.
//! - **Typed encoder properties**: Codec and platform-specific property
//!   structs replace untyped string key-value pairs.  See the
//!   [`prelude`] module for all available types.
//!
//! # Example (Rust)
//!
//! ```rust,no_run
//! use deepstream_encoders::prelude::*;
//!
//! gstreamer::init().unwrap();
//! cuda_init(0).unwrap();
//!
//! let props = EncoderProperties::HevcDgpu(HevcDgpuProps {
//!     bitrate: Some(8_000_000),
//!     profile: Some(HevcProfile::Main),
//!     ..Default::default()
//! });
//!
//! let config = EncoderConfig::new(Codec::Hevc, 1920, 1080)
//!     .properties(props);
//!
//! let mut encoder = NvEncoder::new(&config).unwrap();
//!
//! // Acquire NVMM buffer, render into it, then submit
//! for i in 0..10u128 {
//!     let shared = encoder.generator().acquire(Some(i as i64)).unwrap();
//!     let buffer = shared.into_buffer().expect("sole owner");
//!     let pts_ns = i as u64 * 33_333_333;
//!     encoder.submit_frame(buffer, i, pts_ns, Some(33_333_333)).unwrap();
//! }
//!
//! // Drain remaining frames
//! let remaining = encoder.finish(None).unwrap();
//! ```

pub(crate) mod encoder;
pub(crate) mod error;
pub mod prelude;
pub(crate) mod properties;

pub use encoder::NvEncoder;
pub use error::EncoderError;

// Re-export commonly used items from deepstream_buffers.
pub use deepstream_buffers::{
    cuda_init, BufferGenerator, NvBufSurfaceMemType, SharedBuffer, SurfaceView,
    UniformBatchGenerator, VideoFormat,
};

// Re-export Codec from savant_gstreamer so existing `use deepstream_encoders::Codec` keeps working.
pub use savant_gstreamer::Codec;

// Re-export frequently-used property types at the crate root for convenience.
pub use properties::EncoderProperties;

/// Configuration for creating an [`NvEncoder`].
///
/// Encapsulates all parameters needed to set up the encoding pipeline:
/// codec selection, resolution, framerate, GPU configuration, and
/// encoder-specific properties.
///
/// **Note on buffer pool size**: On discrete GPU, pools use exactly 1
/// buffer so NVENC cannot DMA-read from memory that was already recycled.
/// On Jetson, H.264 / HEVC / AV1 V4L2 encoders use 4 buffers because the
/// driver can hold several input surfaces in flight; a single-buffer pool
/// would block [`BufferGenerator::acquire`](deepstream_buffers::BufferGenerator).
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Video codec to use.
    pub codec: Codec,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Video pixel format (e.g., [`VideoFormat::NV12`], [`VideoFormat::RGBA`]).
    pub format: VideoFormat,
    /// Framerate numerator.
    pub fps_num: i32,
    /// Framerate denominator.
    pub fps_den: i32,
    /// GPU device ID.
    pub gpu_id: u32,
    /// NvBufSurface memory type.
    pub mem_type: NvBufSurfaceMemType,
    /// Typed, codec/platform-specific encoder properties.
    ///
    /// `None` means use the encoder's built-in defaults.  When set, the
    /// variant's codec must match [`codec`](Self::codec).
    pub encoder_params: Option<EncoderProperties>,
}

impl EncoderConfig {
    /// Create a new encoder configuration with sensible defaults.
    ///
    /// Defaults:
    /// - Format: [`VideoFormat::NV12`] (encoder-native, no conversion needed).
    /// - FPS: 30/1.
    /// - GPU ID: 0.
    /// - Memory type: [`NvBufSurfaceMemType::Default`].
    /// - No extra encoder properties.
    pub fn new(codec: Codec, width: u32, height: u32) -> Self {
        Self {
            codec,
            width,
            height,
            format: VideoFormat::NV12,
            fps_num: 30,
            fps_den: 1,
            gpu_id: 0,
            mem_type: NvBufSurfaceMemType::Default,
            encoder_params: None,
        }
    }

    /// Set the video format.
    pub fn format(mut self, format: VideoFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the framerate.
    pub fn fps(mut self, num: i32, den: i32) -> Self {
        assert!(den > 0, "fps denominator must be positive");
        self.fps_num = num;
        self.fps_den = den;
        self
    }

    /// Set the GPU device ID.
    pub fn gpu_id(mut self, gpu_id: u32) -> Self {
        self.gpu_id = gpu_id;
        self
    }

    /// Set the memory type.
    pub fn mem_type(mut self, mem_type: NvBufSurfaceMemType) -> Self {
        self.mem_type = mem_type;
        self
    }

    /// Set typed encoder properties.
    ///
    /// The variant's codec **must** match the [`codec`](Self::codec) of this
    /// config.  A mismatch is detected at [`NvEncoder::new`] time.
    pub fn properties(mut self, props: EncoderProperties) -> Self {
        self.encoder_params = Some(props);
        self
    }
}

/// A single encoded frame pulled from the encoder.
#[derive(Debug, Clone)]
pub struct EncodedFrame {
    /// User-defined frame identifier (passed via [`NvEncoder::submit_frame`]).
    ///
    /// `None` when the encoded buffer could not be matched back to a
    /// submitted frame (e.g. codec header packets emitted by some encoders).
    pub frame_id: Option<u128>,
    /// Presentation timestamp in nanoseconds.
    pub pts_ns: u64,
    /// Decode timestamp in nanoseconds (if set by the encoder).
    ///
    /// For encoders without B-frames (the default in this crate) DTS is
    /// typically equal to PTS or absent.  The field is exposed for sanity
    /// checking and for downstream consumers that need it.
    pub dts_ns: Option<u64>,
    /// Duration in nanoseconds (if known).
    pub duration_ns: Option<u64>,
    /// Encoded bitstream data, or tightly-packed raw pixel data for
    /// [`Codec::RawRgba`] / [`Codec::RawRgb`].
    pub data: Vec<u8>,
    /// Codec used to encode this frame.
    pub codec: Codec,
    /// `true` when this is an intra-coded (key) frame.
    ///
    /// For JPEG and raw pseudoencoders every frame is a keyframe.
    /// For H.264/H.265/AV1 this is derived from the GStreamer buffer flags
    /// (`DELTA_UNIT` absent ⇒ key).
    pub keyframe: bool,
    /// Time base (numerator, denominator) for interpreting timestamps.
    ///
    /// GStreamer operates in nanoseconds, so this is always `(1, 1_000_000_000)`.
    pub time_base: (i32, i32),
}
