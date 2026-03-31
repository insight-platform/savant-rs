#[derive(Debug, thiserror::Error)]
pub enum DecoderError {
    #[error("NVDEC hardware not available on GPU {gpu_id} (required for {codec})")]
    NvdecNotAvailable { codec: String, gpu_id: u32 },

    #[error("Invalid decoder property '{name}': {reason}")]
    InvalidProperty { name: String, reason: String },

    #[error("Input PTS reordering: frame {frame_id} PTS {pts_ns} <= previous {prev_pts_ns}")]
    PtsReordered {
        frame_id: u128,
        pts_ns: u64,
        prev_pts_ns: u64,
    },

    #[error("GStreamer pipeline error: {0}")]
    PipelineError(String),

    #[error("Failed to create GStreamer element '{0}'")]
    ElementCreationFailed(String),

    #[error("Failed to link GStreamer elements: {0}")]
    LinkFailed(String),

    #[error("Failed to acquire/map buffer: {0}")]
    BufferError(String),

    #[error("Decoder has been finalized (EOS sent)")]
    AlreadyFinalized,

    #[error("NvBufSurface error: {0}")]
    NvBufSurfaceError(#[from] deepstream_buffers::NvBufSurfaceError),
}
