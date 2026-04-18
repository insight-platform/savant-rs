use deepstream_buffers::TransformConfig;
use deepstream_encoders::prelude::*;

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
    ///
    /// The `encoder` field carries the full [`NvEncoderConfig`] including
    /// GPU id, channel capacities, memory type, and the codec-specific
    /// sub-config.
    Encode {
        transform: TransformConfig,
        encoder: Box<NvEncoderConfig>,
    },
}
