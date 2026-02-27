use deepstream_encoders::prelude::*;
use deepstream_nvbufsurface::TransformConfig;

/// Describes what to do with each incoming frame for a given source.
#[derive(Debug, Clone)]
pub enum CodecSpec {
    /// Discard the frame entirely.
    Drop,
    /// Pass the frame through without encoding — only transform bboxes back to
    /// initial coordinates.
    Bypass,
    /// GPU-transform the frame to a target resolution, optionally render Skia
    /// overlays, then encode.
    Encode {
        transform: TransformConfig,
        encoder: Box<EncoderConfig>,
    },
}
