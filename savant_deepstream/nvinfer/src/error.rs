//! Error types for the NvInfer pipeline.

use thiserror::Error;

/// Error type for NvInfer operations.
#[derive(Debug, Error)]
pub enum NvInferError {
    #[error("Pipeline error: {0}")]
    PipelineError(String),

    #[error("Element creation failed: {0}")]
    ElementCreationFailed(String),

    #[error("Link failed: {0}")]
    LinkFailed(String),

    #[error("Invalid property: {0}")]
    InvalidProperty(String),

    #[error("Invalid nvinfer config: {0}")]
    InvalidConfig(String),

    #[error("Batch meta attachment failed: {0}")]
    BatchMetaFailed(String),

    #[error("Null pointer: {0}")]
    NullPointer(String),

    #[error("GStreamer initialization failed: {0}")]
    GstInit(String),

    #[error("Batch formation failed: {0}")]
    BatchFormationFailed(String),

    #[error("Operator is shut down")]
    OperatorShutdown,

    #[error("Tensor type mismatch: expected {expected}, got {actual}")]
    TensorTypeMismatch {
        expected: &'static str,
        actual: &'static str,
    },

    #[error("Host tensor data unavailable (host copy disabled or null pointer)")]
    HostDataUnavailable,

    #[error("Buffer error: {0}")]
    Buffer(#[from] deepstream_buffers::NvBufSurfaceError),

    #[error("Pipeline entered failed state (operation timeout exceeded)")]
    PipelineFailed,

    #[error("Batching operator entered failed state (pending batch timeout exceeded)")]
    OperatorFailed,

    #[error("Framework pipeline error: {0}")]
    FrameworkError(#[from] savant_gstreamer::pipeline::PipelineError),

    #[error("Channel disconnected")]
    ChannelDisconnected,
}

/// Result type for NvInfer operations.
pub type Result<T> = std::result::Result<T, NvInferError>;

#[cfg(test)]
mod tests {
    use super::*;
    use deepstream_buffers::NvBufSurfaceError;

    #[test]
    fn buffer_error_into_nvinfer_error() {
        let be = NvBufSurfaceError::NullPointer("test".to_string());
        let ne: NvInferError = be.into();
        assert!(matches!(ne, NvInferError::Buffer(_)));
    }

    #[test]
    fn buffer_error_via_question_mark() {
        fn inner() -> Result<()> {
            Err(NvBufSurfaceError::PoolCreationFailed)?
        }
        let err = inner().unwrap_err();
        assert!(matches!(err, NvInferError::Buffer(_)));
    }
}
