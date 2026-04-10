#[derive(Debug, thiserror::Error, Clone)]
pub enum PipelineError {
    #[error("GStreamer init failed: {0}")]
    InitFailed(String),

    #[error("Failed to create element: {0}")]
    ElementCreationFailed(String),

    #[error("Failed to add element to pipeline: {0}")]
    ElementAddFailed(String),

    #[error("Failed to link elements: {0}")]
    LinkFailed(String),

    #[error("Missing required pad: {0}")]
    MissingPad(String),

    #[error("Failed to change pipeline state: {0}")]
    StateChangeFailed(String),

    #[error("Pipeline runtime error: {0}")]
    RuntimeError(String),
}
