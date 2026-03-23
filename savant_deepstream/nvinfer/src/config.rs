//! Configuration for the NvInfer pipeline.
//!
//! Keys use dotted notation `section.key` to specify config sections. Keys with
//! the same section are grouped under `[section]`. If the section is `property`
//! or the key has no dot, it is treated as `[property]` (the main nvinfer config).

use crate::error::{NvInferError, Result};
use crate::meta_clear_policy::MetaClearPolicy;
use crate::model_input_scaling::ModelInputScaling;
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
    /// `[property]`. Mandatory keys `process-mode`, `output-tensor-meta`,
    /// `network-type`, and `gie-unique-id` are auto-injected if missing.
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
    /// Input width for appsrc caps.  `Some(w)` constrains the caps to a
    /// fixed width; `None` leaves the width unconstrained (required for
    /// non-uniform / heterogeneous batches).
    pub input_width: Option<u32>,
    /// Input height for appsrc caps.  Same semantics as [`input_width`].
    pub input_height: Option<u32>,
    /// When and whether to clear `NvDsObjectMeta` entries from the batch
    /// buffer. Defaults to [`MetaClearPolicy::Before`].
    pub meta_clear_policy: MetaClearPolicy,
    /// When `true`, nvinfer skips the device-to-host copy of output tensors.
    /// Host pointers in [`TensorView`] will contain stale data;
    /// only device pointers are valid. Default: `false` (copy enabled).
    pub disable_output_host_copy: bool,
    /// How frames are scaled to the model input size. Injected as
    /// `maintain-aspect-ratio` / `symmetric-padding` in the generated config;
    /// those keys must not appear in `nvinfer_properties`. Default: [`ModelInputScaling::Fill`].
    pub scaling: ModelInputScaling,
}

impl NvInferConfig {
    /// Create a new config with nvinfer properties map and fixed input
    /// dimensions.  The width and height are embedded into the appsrc caps
    /// and used as the full-frame ROI fallback when no explicit ROIs are
    /// supplied.
    ///
    /// `nvinfer_properties` covers the [property] section. Use absolute paths
    /// for `onnx-file` and `model-engine-file`. Mandatory keys
    /// `process-mode=2`, `output-tensor-meta=1`, `network-type=100`, and
    /// `gie-unique-id=1` are auto-injected if missing.
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
            input_width: Some(input_width),
            input_height: Some(input_height),
            meta_clear_policy: MetaClearPolicy::default(),
            disable_output_host_copy: false,
            scaling: ModelInputScaling::default(),
        }
    }

    /// Create a config without fixed input dimensions.
    ///
    /// The appsrc caps will not constrain width/height, which is required
    /// for non-uniform (heterogeneous) batches where each frame may have a
    /// different resolution.  Callers **must** provide explicit ROIs for
    /// every slot because the full-frame fallback has no dimensions to use.
    pub fn new_flexible(
        nvinfer_properties: HashMap<String, String>,
        input_format: impl Into<String>,
    ) -> Self {
        Self {
            name: String::new(),
            nvinfer_properties,
            element_properties: HashMap::new(),
            gpu_id: 0,
            queue_depth: 0,
            input_format: input_format.into(),
            input_width: None,
            input_height: None,
            meta_clear_policy: MetaClearPolicy::default(),
            disable_output_host_copy: false,
            scaling: ModelInputScaling::default(),
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

    /// Set the metadata clearing policy.
    pub fn meta_clear_policy(mut self, policy: MetaClearPolicy) -> Self {
        self.meta_clear_policy = policy;
        self
    }

    /// Skip the device-to-host copy of output tensors.
    ///
    /// When `true`, only device (GPU) pointers in [`TensorView`] are valid;
    /// host pointers will contain stale/uninitialized data.
    pub fn disable_output_host_copy(mut self, disable: bool) -> Self {
        self.disable_output_host_copy = disable;
        self
    }

    /// Set how input frames are scaled to the model input dimensions.
    pub fn scaling(mut self, scaling: ModelInputScaling) -> Self {
        self.scaling = scaling;
        self
    }

    /// Validate mandatory keys, inject if missing, and write config to a
    /// temporary file. Caller must keep the returned file alive.
    ///
    /// Keys with dotted notation `section.key` are grouped under `[section]`.
    /// Bare keys go to `[property]`.
    ///
    /// ## Forbidden properties
    ///
    /// The following nvinfer properties are incompatible with NvInfer's
    /// standalone `appsrc → nvinfer → appsink` pipeline and are **rejected**
    /// with an error if the caller supplies them:
    ///
    /// | Key | Reason |
    /// |---|---|
    /// | `operate-on-gie-id` | Filters objects by upstream GIE id; the synthetic ROI sentinels use `unique_component_id = -1` and would be silently skipped. |
    /// | `operate-on-class-ids` | Filters objects by class id; sentinels carry no meaningful class and would be skipped. |
    /// | `secondary-reinfer-interval` | Controls re-inference cadence across frames in a multi-frame pipeline; meaningless in single-shot mode and can silently skip frames. |
    /// | `num-detected-classes` | Only meaningful for detector `network-type` (0); misleading with `network-type=100`. |
    /// | `disable-output-host-copy` | Controlled via [`NvInferConfig::disable_output_host_copy`]; must not be set in `nvinfer_properties`. |
    /// | `maintain-aspect-ratio` | Controlled via [`NvInferConfig::scaling`]; must not be set in `nvinfer_properties`. |
    /// | `symmetric-padding` | Controlled via [`NvInferConfig::scaling`]; must not be set in `nvinfer_properties`. |
    ///
    /// ## Auto-injected properties
    ///
    /// | Key | Value | Reason |
    /// |---|---|---|
    /// | `process-mode` | `2` | Secondary/object mode required for ROI-based inference. |
    /// | `output-tensor-meta` | `1` | Raw tensor output required for post-processing. |
    /// | `network-type` | `100` | Custom network type; avoids DeepStream built-in post-processing. |
    /// | `gie-unique-id` | `1` | Single-GIE pipeline; the value is meaningless but must be set. |
    pub fn validate_and_materialize(&self) -> Result<NamedTempFile> {
        let mut props = self.nvinfer_properties.clone();

        fn get_prop<'a>(p: &'a HashMap<String, String>, k: &str) -> Option<&'a String> {
            p.get(k).or_else(|| p.get(&format!("property.{}", k)))
        }

        const FORBIDDEN_KEYS: &[(&str, &str)] = &[
            (
                "operate-on-gie-id",
                "incompatible with NvInfer's standalone pipeline; \
                 synthetic ROI sentinels use unique_component_id=-1 \
                 and would be silently skipped",
            ),
            (
                "operate-on-class-ids",
                "incompatible with NvInfer's standalone pipeline; \
                 synthetic ROI sentinels carry no class id \
                 and would be silently skipped",
            ),
            (
                "secondary-reinfer-interval",
                "meaningless in NvInfer's single-shot pipeline; \
                 can silently skip frames",
            ),
            (
                "num-detected-classes",
                "only meaningful for detector network-type (0); \
                 NvInfer uses network-type=100 (custom)",
            ),
            (
                "disable-output-host-copy",
                "controlled via NvInferConfig.disable_output_host_copy; \
                 must not be set in nvinfer_properties",
            ),
            (
                "maintain-aspect-ratio",
                "controlled via NvInferConfig.scaling; \
                 must not be set in nvinfer_properties",
            ),
            (
                "symmetric-padding",
                "controlled via NvInferConfig.scaling; \
                 must not be set in nvinfer_properties",
            ),
        ];

        for &(key, reason) in FORBIDDEN_KEYS {
            if get_prop(&props, key).is_some() {
                return Err(NvInferError::InvalidConfig(format!(
                    "'{key}' must not be set: {reason}"
                )));
            }
        }

        match get_prop(&props, "process-mode") {
            Some(v) if v.trim() == "2" => {}
            Some(other) => {
                return Err(NvInferError::InvalidConfig(format!(
                    "process-mode={} but NvInfer requires process-mode=2 (secondary/object)",
                    other
                )));
            }
            None => {
                props.insert("process-mode".into(), "2".into());
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

        match get_prop(&props, "network-type") {
            Some(v) if v.trim() == "100" => {}
            Some(other) => {
                return Err(NvInferError::InvalidConfig(format!(
                    "network-type={other} but NvInfer requires network-type=100 (custom)"
                )));
            }
            None => {
                props.insert("network-type".into(), "100".into());
            }
        }

        match get_prop(&props, "gie-unique-id") {
            Some(v) if v.trim() == "1" => {}
            Some(other) => {
                return Err(NvInferError::InvalidConfig(format!(
                    "gie-unique-id={other} but NvInfer requires gie-unique-id=1"
                )));
            }
            None => {
                props.insert("gie-unique-id".into(), "1".into());
            }
        }

        if self.disable_output_host_copy {
            props.insert("disable-output-host-copy".into(), "1".into());
        }

        match self.scaling {
            ModelInputScaling::Fill => {
                props.insert("maintain-aspect-ratio".into(), "0".into());
            }
            ModelInputScaling::KeepAspectRatio => {
                props.insert("maintain-aspect-ratio".into(), "1".into());
                props.insert("symmetric-padding".into(), "0".into());
            }
            ModelInputScaling::KeepAspectRatioSymmetric => {
                props.insert("maintain-aspect-ratio".into(), "1".into());
                props.insert("symmetric-padding".into(), "1".into());
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ModelInputScaling;

    fn minimal_props() -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("gpu-id".into(), "0".into());
        m.insert("infer-dims".into(), "3;12;12".into());
        m
    }

    #[test]
    fn rejects_operate_on_gie_id() {
        let mut props = minimal_props();
        props.insert("operate-on-gie-id".into(), "0".into());
        let cfg = NvInferConfig::new(props, "RGBA", 12, 12);
        let err = cfg.validate_and_materialize().unwrap_err();
        assert!(
            err.to_string().contains("operate-on-gie-id"),
            "error must mention the key: {err}"
        );
    }

    #[test]
    fn rejects_operate_on_class_ids() {
        let mut props = minimal_props();
        props.insert("operate-on-class-ids".into(), "0".into());
        let cfg = NvInferConfig::new(props, "RGBA", 12, 12);
        let err = cfg.validate_and_materialize().unwrap_err();
        assert!(
            err.to_string().contains("operate-on-class-ids"),
            "error must mention the key: {err}"
        );
    }

    #[test]
    fn rejects_secondary_reinfer_interval() {
        let mut props = minimal_props();
        props.insert("secondary-reinfer-interval".into(), "0".into());
        let cfg = NvInferConfig::new(props, "RGBA", 12, 12);
        let err = cfg.validate_and_materialize().unwrap_err();
        assert!(
            err.to_string().contains("secondary-reinfer-interval"),
            "error must mention the key: {err}"
        );
    }

    #[test]
    fn rejects_num_detected_classes() {
        let mut props = minimal_props();
        props.insert("num-detected-classes".into(), "80".into());
        let cfg = NvInferConfig::new(props, "RGBA", 12, 12);
        let err = cfg.validate_and_materialize().unwrap_err();
        assert!(
            err.to_string().contains("num-detected-classes"),
            "error must mention the key: {err}"
        );
    }

    #[test]
    fn rejects_forbidden_key_with_property_prefix() {
        let mut props = minimal_props();
        props.insert("property.operate-on-gie-id".into(), "0".into());
        let cfg = NvInferConfig::new(props, "RGBA", 12, 12);
        let err = cfg.validate_and_materialize().unwrap_err();
        assert!(
            err.to_string().contains("operate-on-gie-id"),
            "error must catch property-prefixed variant: {err}"
        );
    }

    #[test]
    fn rejects_wrong_network_type() {
        let mut props = minimal_props();
        props.insert("network-type".into(), "0".into());
        let cfg = NvInferConfig::new(props, "RGBA", 12, 12);
        let err = cfg.validate_and_materialize().unwrap_err();
        assert!(
            err.to_string().contains("network-type=0"),
            "error must mention the wrong value: {err}"
        );
    }

    #[test]
    fn rejects_wrong_gie_unique_id() {
        let mut props = minimal_props();
        props.insert("gie-unique-id".into(), "2".into());
        let cfg = NvInferConfig::new(props, "RGBA", 12, 12);
        let err = cfg.validate_and_materialize().unwrap_err();
        assert!(
            err.to_string().contains("gie-unique-id=2"),
            "error must mention the wrong value: {err}"
        );
    }

    #[test]
    fn auto_injects_hardcoded_properties() {
        let cfg = NvInferConfig::new(minimal_props(), "RGBA", 12, 12);
        let tmp = cfg.validate_and_materialize().unwrap();
        let content = std::fs::read_to_string(tmp.path()).unwrap();
        assert!(
            content.contains("network-type=100"),
            "must inject network-type=100"
        );
        assert!(
            content.contains("gie-unique-id=1"),
            "must inject gie-unique-id=1"
        );
        assert!(
            content.contains("process-mode=2"),
            "must inject process-mode=2"
        );
        assert!(
            content.contains("output-tensor-meta=1"),
            "must inject output-tensor-meta=1"
        );
    }

    #[test]
    fn accepts_clean_config() {
        let cfg = NvInferConfig::new(minimal_props(), "RGBA", 12, 12);
        assert!(
            cfg.validate_and_materialize().is_ok(),
            "minimal config without forbidden keys must succeed"
        );
    }

    #[test]
    fn rejects_disable_output_host_copy_in_kv() {
        let mut props = minimal_props();
        props.insert("disable-output-host-copy".into(), "1".into());
        let cfg = NvInferConfig::new(props, "RGBA", 12, 12);
        let err = cfg.validate_and_materialize().unwrap_err();
        assert!(
            err.to_string().contains("disable-output-host-copy"),
            "error must mention the key: {err}"
        );
    }

    #[test]
    fn injects_disable_output_host_copy_when_set() {
        let cfg =
            NvInferConfig::new(minimal_props(), "RGBA", 12, 12).disable_output_host_copy(true);
        let tmp = cfg.validate_and_materialize().unwrap();
        let content = std::fs::read_to_string(tmp.path()).unwrap();
        assert!(
            content.contains("disable-output-host-copy=1"),
            "must inject disable-output-host-copy=1 when enabled"
        );
    }

    #[test]
    fn does_not_inject_disable_output_host_copy_by_default() {
        let cfg = NvInferConfig::new(minimal_props(), "RGBA", 12, 12);
        let tmp = cfg.validate_and_materialize().unwrap();
        let content = std::fs::read_to_string(tmp.path()).unwrap();
        assert!(
            !content.contains("disable-output-host-copy"),
            "must not inject disable-output-host-copy when disabled (default)"
        );
    }

    #[test]
    fn rejects_maintain_aspect_ratio_in_kv() {
        let mut props = minimal_props();
        props.insert("maintain-aspect-ratio".into(), "1".into());
        let cfg = NvInferConfig::new(props, "RGBA", 12, 12);
        let err = cfg.validate_and_materialize().unwrap_err();
        assert!(
            err.to_string().contains("maintain-aspect-ratio"),
            "error must mention the key: {err}"
        );
    }

    #[test]
    fn rejects_symmetric_padding_in_kv() {
        let mut props = minimal_props();
        props.insert("symmetric-padding".into(), "1".into());
        let cfg = NvInferConfig::new(props, "RGBA", 12, 12);
        let err = cfg.validate_and_materialize().unwrap_err();
        assert!(
            err.to_string().contains("symmetric-padding"),
            "error must mention the key: {err}"
        );
    }

    #[test]
    fn injects_scaling_fill() {
        let cfg = NvInferConfig::new(minimal_props(), "RGBA", 12, 12);
        assert_eq!(cfg.scaling, ModelInputScaling::Fill);
        let tmp = cfg.validate_and_materialize().unwrap();
        let content = std::fs::read_to_string(tmp.path()).unwrap();
        assert!(
            content.contains("maintain-aspect-ratio=0"),
            "Fill must inject maintain-aspect-ratio=0: {content}"
        );
        assert!(
            !content.contains("symmetric-padding"),
            "Fill must not inject symmetric-padding: {content}"
        );
    }

    #[test]
    fn injects_scaling_keep_aspect_ratio() {
        let cfg = NvInferConfig::new(minimal_props(), "RGBA", 12, 12)
            .scaling(ModelInputScaling::KeepAspectRatio);
        let tmp = cfg.validate_and_materialize().unwrap();
        let content = std::fs::read_to_string(tmp.path()).unwrap();
        assert!(
            content.contains("maintain-aspect-ratio=1"),
            "must inject maintain-aspect-ratio=1: {content}"
        );
        assert!(
            content.contains("symmetric-padding=0"),
            "must inject symmetric-padding=0: {content}"
        );
    }

    #[test]
    fn injects_scaling_keep_aspect_ratio_symmetric() {
        let cfg = NvInferConfig::new(minimal_props(), "RGBA", 12, 12)
            .scaling(ModelInputScaling::KeepAspectRatioSymmetric);
        let tmp = cfg.validate_and_materialize().unwrap();
        let content = std::fs::read_to_string(tmp.path()).unwrap();
        assert!(
            content.contains("maintain-aspect-ratio=1"),
            "must inject maintain-aspect-ratio=1: {content}"
        );
        assert!(
            content.contains("symmetric-padding=1"),
            "must inject symmetric-padding=1: {content}"
        );
    }
}
