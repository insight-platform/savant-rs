//! Configuration for the NvInfer pipeline.
//!
//! Keys use dotted notation `section.key` to specify config sections. Keys with
//! the same section are grouped under `[section]`. If the section is `property`
//! or the key has no dot, it is treated as `[property]` (the main nvinfer config).

use crate::error::{NvInferError, Result};
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::io::Write;
use tempfile::NamedTempFile;

/// Parse a key into (section, key). Bare keys go to "property".
fn parse_section_key(k: &str) -> (&str, &str) {
    if let Some(dot) = k.find('.') {
        (&k[..dot], &k[dot + 1..])
    } else {
        ("property", k)
    }
}

/// Configuration for the NvInfer pipeline.
#[derive(Debug, Clone)]
pub struct NvInferConfig {
    /// Optional name for this instance, used internally for logging and
    /// future extensibility.
    pub name: String,
    /// NvInfer config keys. Use dotted notation `section.key` for per-class
    /// sections (e.g. `class-attrs-0.nms-iou-threshold`). Bare keys go to
    /// `[property]`. Mandatory `process-mode` and `output-tensor-meta` are
    /// auto-injected in `[property]` if missing.
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

impl NvInferConfig {
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
            name: String::new(),
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

    /// Set the instance name for logging and future extensibility.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Validate mandatory keys, inject if missing, and write config to a
    /// temporary file. Caller must keep the returned file alive.
    ///
    /// Keys with dotted notation `section.key` are grouped under `[section]`.
    /// Bare keys go to `[property]`.
    pub fn validate_and_materialize(&self) -> Result<NamedTempFile> {
        let mut props = self.nvinfer_properties.clone();

        fn get_prop<'a>(p: &'a HashMap<String, String>, k: &str) -> Option<&'a String> {
            p.get(k).or_else(|| p.get(&format!("property.{}", k)))
        }

        match get_prop(&props, "process-mode") {
            Some(v) if v.trim() == "1" => {}
            Some(other) => {
                return Err(NvInferError::InvalidConfig(format!(
                    "process-mode={} but NvInfer requires process-mode=1 (primary)",
                    other
                )));
            }
            None => {
                props.insert("process-mode".into(), "1".into());
            }
        }

        match get_prop(&props, "output-tensor-meta") {
            Some(v) if v.trim() == "1" => {}
            Some(other) => {
                return Err(NvInferError::InvalidConfig(format!(
                    "output-tensor-meta={} but NvInfer requires output-tensor-meta=1",
                    other
                )));
            }
            None => {
                props.insert("output-tensor-meta".into(), "1".into());
            }
        }

        // Group by section: section -> BTreeMap<key, value>
        let mut by_section: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
        for (full_key, value) in &props {
            let (section, key) = parse_section_key(full_key);
            by_section
                .entry(section.to_string())
                .or_default()
                .insert(key.to_string(), value.clone());
        }

        // Emit [property] first, then other sections alphabetically.
        let mut lines = Vec::new();
        let mut section_order: Vec<_> = by_section.keys().collect();
        section_order.sort_by(|a, b| match (a.as_str(), b.as_str()) {
            ("property", _) => std::cmp::Ordering::Less,
            (_, "property") => std::cmp::Ordering::Greater,
            _ => a.cmp(b),
        });
        for section in section_order {
            lines.push(format!("[{}]", section));
            let keys = by_section.get(section).unwrap();
            for (key, value) in keys {
                lines.push(format!("{}={}", key, value));
            }
        }
        let content = lines.join("\n");

        let mut tmp = NamedTempFile::new()
            .map_err(|e| NvInferError::InvalidConfig(format!("temp file: {}", e)))?;
        tmp.write_all(content.as_bytes())
            .map_err(|e| NvInferError::InvalidConfig(format!("write config: {}", e)))?;
        tmp.flush()
            .map_err(|e| NvInferError::InvalidConfig(format!("flush config: {}", e)))?;

        Ok(tmp)
    }
}
