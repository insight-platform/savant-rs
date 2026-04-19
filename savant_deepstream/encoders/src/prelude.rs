//! Convenience re-exports for all public types in the `deepstream_encoders`
//! crate.
//!
//! ```ignore
//! use deepstream_encoders::prelude::*;
//! ```
//!
//! Intentionally excludes [`crate::pipeline::NvEncoderExt`] so that consumers
//! use [`NvEncoder::graceful_shutdown`](crate::pipeline::NvEncoder::graceful_shutdown)
//! instead of the manual `send_eos → recv loop` pattern.

pub use crate::config::{
    Av1EncoderConfig, EncoderConfig, H264EncoderConfig, HevcEncoderConfig, JpegEncoderConfig,
    NvEncoderConfig, PngEncoderConfig, RawEncoderConfig,
};
pub use crate::error::EncoderError;
pub use crate::pipeline::{NvEncoder, NvEncoderOutput};
pub use crate::{EncodedFrame, EncoderProperties};

// Codec (re-exported from savant_gstreamer).
pub use savant_gstreamer::Codec;

// NvBufSurface utilities (re-exported from deepstream_buffers).
pub use deepstream_buffers::{
    cuda_init, BufferGenerator, NvBufSurfaceMemType, SharedBuffer, SurfaceView,
    UniformBatchGenerator, VideoFormat,
};

// Encoder properties — per-codec/platform structs.
pub use crate::properties::{
    Av1DgpuProps, Av1JetsonProps, DgpuPreset, H264DgpuProps, H264JetsonProps, H264Profile,
    HevcDgpuProps, HevcJetsonProps, HevcProfile, JetsonPresetLevel, JpegProps, Platform, PngProps,
    RateControl, RawProps, TuningPreset,
};
