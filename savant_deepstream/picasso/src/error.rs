/// Errors produced by the Picasso pipeline.
#[derive(Debug, thiserror::Error)]
pub enum PicassoError {
    #[error("Source '{0}' not found")]
    SourceNotFound(String),

    #[error("Worker channel for source '{0}' is disconnected")]
    ChannelDisconnected(String),

    #[error("Encoder error for source '{0}': {1}")]
    Encoder(String, String),

    #[error("Transform error for source '{0}': {1}")]
    Transform(String, String),

    #[error("Skia renderer error for source '{0}': {1}")]
    Renderer(String, String),

    #[error("Invalid transformation chain: {0}")]
    InvalidTransformationChain(String),

    #[error("GPU mismatch for source '{source_id}': buffer on GPU {buffer_gpu}, encoder on GPU {encoder_gpu}")]
    GpuMismatch {
        source_id: String,
        buffer_gpu: u32,
        encoder_gpu: u32,
    },

    #[error("Engine is shut down")]
    Shutdown,
}
