//! Configuration for the sidecar inference pipeline.

use crate::error::{Result, SidecarError};
use std::collections::HashMap;
use std::io::Write;
use tempfile::NamedTempFile;

/// Configuration for the sidecar inference pipeline.
#[derive(Debug, Clone)]
pub struct SidecarConfig {
    /// NvInfer [property] section keys. Mandatory keys `process-mode` and
    /// `output-tensor-meta` are auto-injected if missing and validated if present.
    pub nvinfer_properties: HashMap<String, String>,
    /// Additional GStreamer element properties (e.g. "unique-id" -> "1").
    pub element_properties: HashMap<String, String>,
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
    /// Create a new config with nvinfer properties map and input dimensions.
    ///
    /// `nvinfer_properties` covers the [property] section. Use absolute paths
    /// for `onnx-file` and `model-engine-file`. Mandatory keys `process-mode`
    /// and `output-tensor-meta` are auto-injected as `1` if missing.
    pub fn new(
        nvinfer_properties: HashMap<String, String>,
        input_format: impl Into<String>,
        input_width: u32,
        input_height: u32,
    ) -> Self {
        Self {
            nvinfer_properties,
            element_properties: HashMap::new(),
            gpu_id: 0,
            queue_depth: 0,
            input_format: input_format.into(),
            input_width,
            input_height,
        }
    }

    /// Set additional GStreamer element properties.
    pub fn with_element_properties(mut self, properties: HashMap<String, String>) -> Self {
        self.element_properties = properties;
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

    /// Validate mandatory keys, inject if missing, and write [property] section
    /// to a temporary file. Caller must keep the returned file alive.
    pub fn validate_and_materialize(&self) -> Result<NamedTempFile> {
        let mut props = self.nvinfer_properties.clone();

        match props.get("process-mode") {
            Some(v) if v.trim() == "1" => {}
            Some(other) => {
                return Err(SidecarError::InvalidConfig(format!(
                    "process-mode={} but sidecar requires process-mode=1 (primary)",
                    other
                )));
            }
            None => {
                props.insert("process-mode".into(), "1".into());
            }
        }

        match props.get("output-tensor-meta") {
            Some(v) if v.trim() == "1" => {}
            Some(other) => {
                return Err(SidecarError::InvalidConfig(format!(
                    "output-tensor-meta={} but sidecar requires output-tensor-meta=1",
                    other
                )));
            }
            None => {
                props.insert("output-tensor-meta".into(), "1".into());
            }
        }

        let mut lines = vec!["[property]".to_string()];
        let mut keys: Vec<_> = props.keys().collect();
        keys.sort();
        for k in keys {
            let v = props.get(k).unwrap();
            lines.push(format!("{}={}", k, v));
        }
        let content = lines.join("\n");

        let mut tmp = NamedTempFile::new()
            .map_err(|e| SidecarError::InvalidConfig(format!("temp file: {}", e)))?;
        tmp.write_all(content.as_bytes())
            .map_err(|e| SidecarError::InvalidConfig(format!("write config: {}", e)))?;
        tmp.flush()
            .map_err(|e| SidecarError::InvalidConfig(format!("flush config: {}", e)))?;

        Ok(tmp)
    }
}
