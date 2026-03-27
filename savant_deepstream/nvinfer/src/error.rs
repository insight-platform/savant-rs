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

    #[error("Duplicate source in batch: {0}")]
    DuplicateSource(String),

    #[error("Batch formation failed: {0}")]
    BatchFormationFailed(String),

    #[error("Operator is shut down")]
    OperatorShutdown,
}

/// Result type for NvInfer operations.
pub type Result<T> = std::result::Result<T, NvInferError>;
