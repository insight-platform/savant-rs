//! Configuration for the sidecar inference pipeline.

use std::collections::HashMap;
use std::path::PathBuf;

/// Configuration for the sidecar inference pipeline.
#[derive(Debug, Clone)]
pub struct SidecarConfig {
    /// Path to nvinfer config file (standard .txt format).
    pub config_file_path: PathBuf,
    /// Additional GStreamer element properties (e.g. "unique-id" -> "1").
    pub properties: HashMap<String, String>,
    /// GPU ID (default: 0).
    pub gpu_id: u32,
    /// GStreamer queue element max-size-buffers.
    /// 0 = no queue element (synchronous), >0 = insert queue with this depth.
    pub queue_depth: u32,
    /// Input format for appsrc caps (e.g. "RGBA"). Must match model input.
    pub input_format: String,
    /// Input width for appsrc caps. Must match model input.
    pub input_width: u32,
    /// Input height for appsrc caps. Must match model input.
    pub input_height: u32,
}

impl SidecarConfig {
    /// Create a new config with the given nvinfer config file path and input dimensions.
    pub fn new(
        config_file_path: impl Into<PathBuf>,
        input_format: impl Into<String>,
        input_width: u32,
        input_height: u32,
    ) -> Self {
        Self {
            config_file_path: config_file_path.into(),
            properties: HashMap::new(),
            gpu_id: 0,
            queue_depth: 0,
            input_format: input_format.into(),
            input_width,
            input_height,
        }
    }

    /// Set additional GStreamer element properties.
    pub fn with_properties(mut self, properties: HashMap<String, String>) -> Self {
        self.properties = properties;
        self
    }

    /// Set GPU ID.
    pub fn gpu_id(mut self, gpu_id: u32) -> Self {
        self.gpu_id = gpu_id;
        self
    }

    /// Set queue depth (max-size-buffers). 0 = no queue.
    pub fn queue_depth(mut self, queue_depth: u32) -> Self {
        self.queue_depth = queue_depth;
        self
    }
}
