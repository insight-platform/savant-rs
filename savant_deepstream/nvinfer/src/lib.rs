pub mod infer_context;
pub mod infer_tensor_meta;

// Re-export main types for convenience
pub use infer_context::{
    BatchInput, BatchOutput, Context, DataType, InferContextInitParams, InferFormat,
    InferNetworkMode, InferTensorOrder, LayerInfo, LogLevel, NetworkInfo, NetworkType,
};
pub use infer_tensor_meta::{InferDims, InferTensorMeta};

/// Error type for DeepStream nvinfer operations
#[derive(Debug, thiserror::Error)]
pub enum NvInferError {
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
    #[error("Null pointer error in {0}")]
    NullPointer(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("String conversion error: {0}")]
    StringConversion(#[from] std::ffi::NulError),
}

/// Result type for DeepStream nvinfer operations
pub type Result<T> = std::result::Result<T, NvInferError>;
