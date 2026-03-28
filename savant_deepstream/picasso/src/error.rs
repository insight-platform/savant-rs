/// Errors produced by the Picasso pipeline.
#[derive(Debug, thiserror::Error)]
pub enum PicassoError {
    #[error("Source '{0}' not found")]
    SourceNotFound(String),

    #[error("Worker channel for source '{0}' is disconnected")]
    ChannelDisconnected(String),

    #[error("Source worker channel closed or disconnected")]
    SourceWorkerSendFailed,

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

    #[error("TransformConfig.cuda_stream must be null; Picasso manages its own CUDA streams")]
    ExternalCudaStream,

    #[error("Failed to create worker CUDA stream: {0}")]
    CudaStreamCreationFailed(String),

    #[error("Invalid letterbox parameters: {0}")]
    InvalidLetterboxParams(String),

    #[error("Engine is shut down")]
    Shutdown,

    #[error("Buffer error: {0}")]
    Buffer(#[from] deepstream_buffers::NvBufSurfaceError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepstream_buffers::NvBufSurfaceError;

    #[test]
    fn buffer_error_into_picasso_error() {
        let be = NvBufSurfaceError::NullPointer("test".to_string());
        let pe: PicassoError = be.into();
        assert!(matches!(pe, PicassoError::Buffer(_)));
    }

    #[test]
    fn buffer_error_via_question_mark() {
        fn inner() -> std::result::Result<(), PicassoError> {
            Err(NvBufSurfaceError::PoolCreationFailed)?
        }
        let err = inner().unwrap_err();
        assert!(matches!(err, PicassoError::Buffer(_)));
    }
}
