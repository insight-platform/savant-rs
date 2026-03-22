//! PyO3 wrapper for NvInferConfig.

use super::enums::PyMetaClearPolicy;
use nvinfer::NvInferConfig;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Configuration for the NvInfer pipeline.
///
/// Args:
///     nvinfer_properties (Dict[str, str]): NvInfer config keys. Use dotted
///         notation ``section.key`` for per-class sections. Bare keys go to
///         ``[property]``. ``process-mode`` and ``output-tensor-meta`` are
///         auto-injected if missing.
///     input_format (str): Input format for appsrc caps (e.g. ``"RGBA"``).
///     input_width (int): Input width for appsrc caps.
///     input_height (int): Input height for appsrc caps.
///     name (str): Optional instance name for logging.
///     element_properties (Optional[Dict[str, str]]): Additional GStreamer
///         element properties.
///     gpu_id (int): GPU device ID.
///     queue_depth (int): GStreamer queue max-size-buffers (0 = no queue).
///     meta_clear_policy (MetaClearPolicy): When to clear object metadata.
///     disable_output_host_copy (bool): When ``True``, skip device-to-host
///         copy of output tensors. Only device (GPU) pointers will be valid.
///         Default: ``False``.
#[pyclass(
    name = "NvInferConfig",
    module = "savant_rs.nvinfer",
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub struct PyNvInferConfig {
    pub(crate) inner: NvInferConfig,
}

#[pymethods]
impl PyNvInferConfig {
    #[new]
    #[pyo3(signature = (
        nvinfer_properties,
        input_format,
        input_width,
        input_height,
        *,
        name = String::new(),
        element_properties = None,
        gpu_id = 0,
        queue_depth = 0,
        meta_clear_policy = PyMetaClearPolicy::Before,
        disable_output_host_copy = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        nvinfer_properties: HashMap<String, String>,
        input_format: String,
        input_width: u32,
        input_height: u32,
        name: String,
        element_properties: Option<HashMap<String, String>>,
        gpu_id: u32,
        queue_depth: u32,
        meta_clear_policy: PyMetaClearPolicy,
        disable_output_host_copy: bool,
    ) -> Self {
        let mut cfg =
            NvInferConfig::new(nvinfer_properties, input_format, input_width, input_height)
                .gpu_id(gpu_id)
                .queue_depth(queue_depth)
                .meta_clear_policy(meta_clear_policy.into())
                .disable_output_host_copy(disable_output_host_copy);
        if !name.is_empty() {
            cfg = cfg.name(name);
        }
        if let Some(ep) = element_properties {
            cfg = cfg.with_element_properties(ep);
        }
        Self { inner: cfg }
    }

    /// Create a config without fixed input dimensions.
    ///
    /// Required for non-uniform (heterogeneous) batches where each frame may
    /// have a different resolution.  Callers **must** provide explicit ROIs for
    /// every slot.
    ///
    /// Args:
    ///     nvinfer_properties (Dict[str, str]): NvInfer config keys.
    ///     input_format (str): Input format for appsrc caps.
    ///     name (str): Optional instance name for logging.
    ///     element_properties (Optional[Dict[str, str]]): Additional GStreamer
    ///         element properties.
    ///     gpu_id (int): GPU device ID.
    ///     queue_depth (int): GStreamer queue max-size-buffers (0 = no queue).
    ///     meta_clear_policy (MetaClearPolicy): When to clear object metadata.
    #[staticmethod]
    #[pyo3(signature = (
        nvinfer_properties,
        input_format,
        *,
        name = String::new(),
        element_properties = None,
        gpu_id = 0,
        queue_depth = 0,
        meta_clear_policy = PyMetaClearPolicy::Before,
        disable_output_host_copy = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new_flexible(
        nvinfer_properties: HashMap<String, String>,
        input_format: String,
        name: String,
        element_properties: Option<HashMap<String, String>>,
        gpu_id: u32,
        queue_depth: u32,
        meta_clear_policy: PyMetaClearPolicy,
        disable_output_host_copy: bool,
    ) -> Self {
        let mut cfg = NvInferConfig::new_flexible(nvinfer_properties, input_format)
            .gpu_id(gpu_id)
            .queue_depth(queue_depth)
            .meta_clear_policy(meta_clear_policy.into())
            .disable_output_host_copy(disable_output_host_copy);
        if !name.is_empty() {
            cfg = cfg.name(name);
        }
        if let Some(ep) = element_properties {
            cfg = cfg.with_element_properties(ep);
        }
        Self { inner: cfg }
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn gpu_id(&self) -> u32 {
        self.inner.gpu_id
    }

    #[getter]
    fn queue_depth(&self) -> u32 {
        self.inner.queue_depth
    }

    #[getter]
    fn input_format(&self) -> &str {
        &self.inner.input_format
    }

    #[getter]
    fn input_width(&self) -> Option<u32> {
        self.inner.input_width
    }

    #[getter]
    fn input_height(&self) -> Option<u32> {
        self.inner.input_height
    }

    #[getter]
    fn meta_clear_policy(&self) -> PyMetaClearPolicy {
        self.inner.meta_clear_policy.into()
    }

    /// Whether the device-to-host copy of output tensors is disabled.
    #[getter]
    fn disable_output_host_copy(&self) -> bool {
        self.inner.disable_output_host_copy
    }

    fn __repr__(&self) -> String {
        format!(
            "NvInferConfig(name={:?}, gpu_id={}, queue_depth={}, input_format={:?}, \
             input_width={:?}, input_height={:?}, meta_clear_policy={:?}, \
             disable_output_host_copy={})",
            self.inner.name,
            self.inner.gpu_id,
            self.inner.queue_depth,
            self.inner.input_format,
            self.inner.input_width,
            self.inner.input_height,
            PyMetaClearPolicy::from(self.inner.meta_clear_policy),
            self.inner.disable_output_host_copy,
        )
    }
}
