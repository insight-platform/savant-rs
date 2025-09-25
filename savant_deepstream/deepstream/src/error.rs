use thiserror::Error;

/// Error type for DeepStream operations
#[derive(Error, Debug)]
pub enum DeepStreamError {
    /// Error when a pointer is null
    #[error("Null pointer encountered: {0}")]
    NullPointer(String),

    /// Error when memory allocation fails
    #[error("Memory allocation failed: {0}")]
    AllocationFailed(String),

    /// Error when an operation is invalid
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    /// Error when a parameter is invalid
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Error when a resource is not found
    #[error("Resource not found: {0}")]
    NotFound(String),

    /// Error when an operation times out
    #[error("Operation timed out: {0}")]
    Timeout(String),

    /// Error when a system call fails
    #[error("System error: {0}")]
    SystemError(String),

    /// Error when converting between types
    #[error("Conversion error: {0}")]
    ConversionError(String),

    /// Error when accessing metadata
    #[error("Metadata error: {0}")]
    MetadataError(String),

    /// Error when working with GStreamer
    #[error("GStreamer error: {0}")]
    GStreamerError(String),

    /// Generic error with a message
    #[error("{0}")]
    Generic(String),
}

impl From<std::ffi::NulError> for DeepStreamError {
    fn from(err: std::ffi::NulError) -> Self {
        DeepStreamError::ConversionError(format!("Null byte in string: {}", err))
    }
}

impl From<std::str::Utf8Error> for DeepStreamError {
    fn from(err: std::str::Utf8Error) -> Self {
        DeepStreamError::ConversionError(format!("UTF-8 conversion error: {}", err))
    }
}

impl From<glib::Error> for DeepStreamError {
    fn from(err: glib::Error) -> Self {
        DeepStreamError::GStreamerError(err.to_string())
    }
}



impl DeepStreamError {
    /// Create a null pointer error
    pub fn null_pointer(context: &str) -> Self {
        DeepStreamError::NullPointer(context.to_string())
    }

    /// Create an allocation error
    pub fn allocation_failed(context: &str) -> Self {
        DeepStreamError::AllocationFailed(context.to_string())
    }

    /// Create an invalid operation error
    pub fn invalid_operation(context: &str) -> Self {
        DeepStreamError::InvalidOperation(context.to_string())
    }

    /// Create an invalid parameter error
    pub fn invalid_parameter(context: &str) -> Self {
        DeepStreamError::InvalidParameter(context.to_string())
    }

    /// Create a not found error
    pub fn not_found(context: &str) -> Self {
        DeepStreamError::NotFound(context.to_string())
    }

    /// Create a timeout error
    pub fn timeout(context: &str) -> Self {
        DeepStreamError::Timeout(context.to_string())
    }

    /// Create a system error
    pub fn system_error(context: &str) -> Self {
        DeepStreamError::SystemError(context.to_string())
    }

    /// Create a conversion error
    pub fn conversion_error(context: &str) -> Self {
        DeepStreamError::ConversionError(context.to_string())
    }

    /// Create a metadata error
    pub fn metadata_error(context: &str) -> Self {
        DeepStreamError::MetadataError(context.to_string())
    }

    /// Create a GStreamer error
    pub fn gstreamer_error(context: &str) -> Self {
        DeepStreamError::GStreamerError(context.to_string())
    }

    /// Create a generic error
    pub fn generic(message: &str) -> Self {
        DeepStreamError::Generic(message.to_string())
    }
}
