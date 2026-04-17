//! Policy for clearing `NvDsObjectMeta` entries around inference.
//!
//! This type lives in [`deepstream_buffers`] so both nvinfer and nvtracker
//! can share it. Re-exported here for backward compatibility.

pub use deepstream_buffers::MetaClearPolicy;
