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
    cuda_init, NvBufSurfaceGenerator, NvBufSurfaceMemType,
};

/// Supported video codecs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Codec {
    /// H.264 / AVC.
    H264,
    /// H.265 / HEVC.
    Hevc,
    /// JPEG (Motion JPEG).
    Jpeg,
    /// AV1.
    Av1,
}

impl Codec {
    /// Return the GStreamer encoder element name for this codec.
    pub fn encoder_element(&self) -> &'static str {
        match self {
            Codec::H264 => "nvv4l2h264enc",
            Codec::Hevc => "nvv4l2h265enc",
            Codec::Jpeg => "nvjpegenc",
            Codec::Av1 => "nvv4l2av1enc",
        }
    }

    /// Return the GStreamer parser element name for this codec.
    pub fn parser_element(&self) -> &'static str {
        match self {
            Codec::H264 => "h264parse",
            Codec::Hevc => "h265parse",
            Codec::Jpeg => "jpegparse",
            Codec::Av1 => "av1parse",
        }
    }

    /// Parse a codec from a string name.
    ///
    /// Accepted names (case-insensitive): `h264`, `hevc`, `h265`, `jpeg`, `av1`.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "h264" => Some(Codec::H264),
            "hevc" | "h265" => Some(Codec::Hevc),
            "jpeg" => Some(Codec::Jpeg),
            "av1" => Some(Codec::Av1),
            _ => None,
        }
    }

    /// Return the canonical name of this codec.
    pub fn name(&self) -> &'static str {
        match self {
            Codec::H264 => "h264",
            Codec::Hevc => "hevc",
            Codec::Jpeg => "jpeg",
            Codec::Av1 => "av1",
        }
    }
}

impl std::fmt::Display for Codec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Configuration for creating an [`NvEncoder`].
///
/// Encapsulates all parameters needed to set up the encoding pipeline:
/// codec selection, resolution, framerate, GPU configuration, and
/// encoder-specific properties.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Video codec to use.
    pub codec: Codec,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Video format string (e.g., `"NV12"`, `"RGBA"`).
    pub format: String,
    /// Framerate numerator.
    pub fps_num: i32,
    /// Framerate denominator.
    pub fps_den: i32,
    /// GPU device ID.
    pub gpu_id: u32,
    /// NvBufSurface memory type (as u32, see [`NvBufSurfaceMemType`]).
    pub mem_type: u32,
    /// Buffer pool size (min and max buffers).
    pub pool_size: u32,
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
    /// - Format: `"NV12"` (encoder-native, no conversion needed).
    /// - FPS: 30/1.
    /// - GPU ID: 0.
    /// - Memory type: 0 (Default).
    /// - Pool size: 4.
    /// - No extra encoder properties.
    pub fn new(codec: Codec, width: u32, height: u32) -> Self {
        Self {
            codec,
            width,
            height,
            format: "NV12".to_string(),
            fps_num: 30,
            fps_den: 1,
            gpu_id: 0,
            mem_type: 0,
            pool_size: 4,
            encoder_properties: Vec::new(),
        }
    }

    /// Set the video format.
    pub fn format(mut self, format: &str) -> Self {
        self.format = format.to_string();
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
    pub fn mem_type(mut self, mem_type: u32) -> Self {
        self.mem_type = mem_type;
        self
    }

    /// Set the buffer pool size.
    pub fn pool_size(mut self, pool_size: u32) -> Self {
        self.pool_size = pool_size;
        self
    }

    /// Add an encoder property.
    ///
    /// # Errors
    ///
    /// This method validates that the property name does not refer to a
    /// B-frame setting. The validation happens at encoder creation time,
    /// but is also done eagerly here for early error detection.
    pub fn encoder_property(
        mut self,
        key: &str,
        value: &str,
    ) -> Result<Self, EncoderError> {
        // Eagerly reject B-frame properties.
        let b_frame_props: &[&str] = &[
            "B-frames", "b-frames", "bframes", "Bframes",
            "num-B-Frames", "num-b-frames", "num-bframes",
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
    /// Duration in nanoseconds (if known).
    pub duration_ns: Option<u64>,
    /// Encoded bitstream data.
    pub data: Vec<u8>,
    /// Codec used to encode this frame.
    pub codec: Codec,
}
