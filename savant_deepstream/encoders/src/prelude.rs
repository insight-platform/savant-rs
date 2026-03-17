//! Convenience re-exports for all public types in the deepstream_encoders
//! crate.
//!
//! ```ignore
//! use deepstream_encoders::prelude::*;
//! ```

// Encoder
pub use crate::encoder::NvEncoder;

// Error
pub use crate::error::EncoderError;

// Config and output
pub use crate::{EncodedFrame, EncoderConfig};

// Codec (re-exported from savant_gstreamer)
pub use savant_gstreamer::Codec;

// NvBufSurface utilities (re-exported from deepstream_buffers)
pub use deepstream_buffers::{
    cuda_init, BufferGenerator, NvBufSurfaceMemType, SharedBuffer, SurfaceView,
    UniformBatchGenerator, VideoFormat,
};

// Encoder properties — aggregate enum and all per-codec/platform structs
pub use crate::properties::{
    Av1DgpuProps, DgpuPreset, EncoderProperties, H264DgpuProps, H264JetsonProps, H264Profile,
    HevcDgpuProps, HevcJetsonProps, HevcProfile, JetsonPresetLevel, JpegProps, Platform, PngProps,
    RateControl, RawProps, TuningPreset,
};
