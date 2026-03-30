# DeepStreamError Reference

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DeepStreamError {
    #[error("Null pointer encountered: {0}")]
    NullPointer(String),

    #[error("Memory allocation failed: {0}")]
    AllocationFailed(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Resource not found: {0}")]
    NotFound(String),

    #[error("Operation timed out: {0}")]
    Timeout(String),

    #[error("System error: {0}")]
    SystemError(String),

    #[error("Conversion error: {0}")]
    ConversionError(String),

    #[error("Metadata error: {0}")]
    MetadataError(String),

    #[error("GStreamer error: {0}")]
    GStreamerError(String),

    #[error("{0}")]
    Generic(String),
}
```

## Helper Constructors

```rust
impl DeepStreamError {
    pub fn null_pointer(context: &str) -> Self;
    pub fn allocation_failed(context: &str) -> Self;
    pub fn invalid_operation(context: &str) -> Self;
    pub fn invalid_parameter(context: &str) -> Self;
    pub fn not_found(context: &str) -> Self;
    pub fn timeout(context: &str) -> Self;
    pub fn system_error(context: &str) -> Self;
    pub fn conversion_error(context: &str) -> Self;
    pub fn metadata_error(context: &str) -> Self;
    pub fn gstreamer_error(context: &str) -> Self;
    pub fn generic(message: &str) -> Self;
}
```

## From Implementations

- `From<std::ffi::NulError>`
- `From<std::str::Utf8Error>`
- `From<glib::Error>`

## When Used in This Crate

| Variant | Typical trigger |
|---|---|
| `NullPointer` | `BatchMeta::from_gst_buffer` when no batch meta in buffer; `from_raw` with null |
| `AllocationFailed` | `ObjectMeta::from_batch` when pool fails |
| `InvalidParameter` | Invalid parameter in metadata operations |
| `ConversionError` | NulError, Utf8Error |
| `GStreamerError` | glib::Error from GStreamer APIs |
