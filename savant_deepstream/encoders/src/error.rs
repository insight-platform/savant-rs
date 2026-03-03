//! Error types for the encoder module.

/// Errors that can occur during encoder creation or operation.
#[derive(Debug, thiserror::Error)]
pub enum EncoderError {
    /// The requested codec is not supported on this platform.
    #[error("Unsupported codec: {0}")]
    UnsupportedCodec(String),

    /// An encoder property is invalid or rejected.
    #[error("Invalid encoder property '{name}': {reason}")]
    InvalidProperty { name: String, reason: String },

    /// Input PTS reordering was detected (non-monotonic PTS on submit).
    #[error("Input PTS reordering detected: frame {frame_id} has PTS {pts_ns} which is <= previous PTS {prev_pts_ns}")]
    PtsReordered {
        frame_id: u128,
        pts_ns: u64,
        prev_pts_ns: u64,
    },

    /// Output PTS reordering was detected — indicates B-frames were
    /// emitted despite being disabled.
    #[error(
        "Output PTS reordering detected (B-frames?): frame {frame_id} has \
         PTS {pts_ns} < previous output PTS {prev_pts_ns}"
    )]
    OutputPtsReordered {
        frame_id: u128,
        pts_ns: u64,
        prev_pts_ns: u64,
    },

    /// Output DTS exceeds PTS — indicates B-frame reordering.
    #[error(
        "Output DTS > PTS detected (B-frames?): frame {frame_id} has \
         DTS {dts_ns} > PTS {pts_ns}"
    )]
    OutputDtsExceedsPts {
        frame_id: u128,
        dts_ns: u64,
        pts_ns: u64,
    },

    /// GStreamer pipeline error.
    #[error("GStreamer pipeline error: {0}")]
    PipelineError(String),

    /// GStreamer element creation failure.
    #[error("Failed to create GStreamer element '{0}'")]
    ElementCreationFailed(String),

    /// GStreamer element linking failure.
    #[error("Failed to link GStreamer elements: {from} -> {to}")]
    LinkFailed { from: String, to: String },

    /// Buffer acquisition failure.
    #[error("Failed to acquire buffer: {0}")]
    BufferAcquisitionFailed(String),

    /// Encoder has already been finalized (EOS sent).
    #[error("Encoder has been finalized (EOS sent), no more frames can be submitted")]
    AlreadyFinalized,

    /// Upstream NvBufSurface error.
    #[error("NvBufSurface error: {0}")]
    NvBufSurfaceError(#[from] deepstream_nvbufsurface::NvBufSurfaceError),
}
