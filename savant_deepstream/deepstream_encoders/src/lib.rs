//! GPU-accelerated video encoders using DeepStream NvBufSurface.
//!
//! This crate provides a high-level API for hardware-accelerated video
//! encoding (H.264, HEVC/H.265, JPEG, AV1) backed by NVIDIA DeepStream's
//! NvBufSurface buffer pool and NVENC/NVJPEG encoders.
//!
//! # Design
//!
//! - **No B-frames**: The encoder rejects any property that would enable
//!   B-frames and forcibly disables them on the encoder element.
//! - **Monotonic PTS**: Each submitted frame's PTS must be strictly greater
//!   than the previous frame's PTS. The encoder raises an error on
//!   reordering.
//! - **Integrated buffer management**: The encoder owns an
//!   [`NvBufSurfaceGenerator`](deepstream_nvbufsurface::NvBufSurfaceGenerator)
//!   that provides NVMM GPU buffers for zero-copy rendering.
//!
//! # Example (Rust)
//!
//! ```rust,no_run
//! use deepstream_encoders::{NvEncoder, EncoderConfig, Codec};
//! use deepstream_nvbufsurface::cuda_init;
//!
//! gstreamer::init().unwrap();
//! cuda_init(0).unwrap();
//!
//! let config = EncoderConfig::new(Codec::Hevc, 1920, 1080);
//!
//! let mut encoder = NvEncoder::new(&config).unwrap();
//!
//! // Acquire NVMM buffer, render into it, then submit
//! for i in 0..10 {
//!     let buffer = encoder.generator().acquire_surface(Some(i)).unwrap();
//!     let pts_ns = i as u64 * 33_333_333;
//!     encoder.submit_frame(buffer, i, pts_ns, Some(33_333_333)).unwrap();
//! }
//!
//! // Drain remaining frames
//! let remaining = encoder.finish(None).unwrap();
//! ```

pub mod encoder;
pub mod error;

#[cfg(feature = "python")]
pub mod python;

pub use encoder::NvEncoder;
pub use error::EncoderError;

// Re-export commonly used items from deepstream_nvbufsurface.
pub use deepstream_nvbufsurface::{
    cuda_init, NvBufSurfaceGenerator, NvBufSurfaceMemType, VideoFormat,
};

// Re-export Codec from savant_gstreamer so existing `use deepstream_encoders::Codec` keeps working.
pub use savant_gstreamer::Codec;

/// Configuration for creating an [`NvEncoder`].
///
/// Encapsulates all parameters needed to set up the encoding pipeline:
/// codec selection, resolution, framerate, GPU configuration, and
/// encoder-specific properties.
///
/// **Note on buffer pool size**: The internal buffer pools are always
/// configured with exactly 1 buffer. This is required because the NVENC
/// hardware encoder may continue DMA-reading from a buffer's GPU memory
/// after the GStreamer element has released its reference. A pool size of
/// 1 forces full serialization: each frame must be completely consumed
/// by the hardware encoder before the next frame can be submitted,
/// preventing stale-data artifacts.
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
    /// Encoder-specific GStreamer properties as `(key, value)` string pairs.
    ///
    /// These are passed directly to the encoder element via
    /// `set_property_from_str`. B-frame properties are rejected.
    pub encoder_properties: Vec<(String, String)>,
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
            encoder_properties: Vec::new(),
        }
    }

    /// Set the video format.
    pub fn format(mut self, format: VideoFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the framerate.
    pub fn fps(mut self, num: i32, den: i32) -> Self {
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

    /// Add an encoder property.
    ///
    /// # Errors
    ///
    /// This method validates that the property name does not refer to a
    /// B-frame setting. The validation happens at encoder creation time,
    /// but is also done eagerly here for early error detection.
    pub fn encoder_property(mut self, key: &str, value: &str) -> Result<Self, EncoderError> {
        // Eagerly reject B-frame properties.
        let b_frame_props: &[&str] = &[
            "B-frames",
            "b-frames",
            "bframes",
            "Bframes",
            "num-B-Frames",
            "num-b-frames",
            "num-bframes",
        ];
        for bprop in b_frame_props {
            if key.eq_ignore_ascii_case(bprop) {
                return Err(EncoderError::BFramesNotAllowed(key.to_string()));
            }
        }
        self.encoder_properties
            .push((key.to_string(), value.to_string()));
        Ok(self)
    }
}

/// A single encoded frame pulled from the encoder.
#[derive(Debug, Clone)]
pub struct EncodedFrame {
    /// User-defined frame identifier (passed via [`NvEncoder::submit_frame`]).
    pub frame_id: i64,
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
    /// Encoded bitstream data.
    pub data: Vec<u8>,
    /// Codec used to encode this frame.
    pub codec: Codec,
}
