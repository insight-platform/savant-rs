//! Error types for the sidecar inference pipeline.

use thiserror::Error;

/// Error type for sidecar nvinfer operations.
#[derive(Debug, Error)]
pub enum SidecarError {
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
}

/// Result type for sidecar operations.
pub type Result<T> = std::result::Result<T, SidecarError>;
