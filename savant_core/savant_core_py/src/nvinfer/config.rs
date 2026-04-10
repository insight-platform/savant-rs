//! PyO3 wrapper for NvInferConfig.

use super::enums::{PyMetaClearPolicy, PyModelColorFormat, PyModelInputScaling};
use crate::deepstream::enums::PyVideoFormat;
use nvinfer::NvInferConfig;
use pyo3::prelude::*;
use savant_gstreamer::VideoFormat;
use std::collections::HashMap;

/// Configuration for the NvInfer pipeline.
///
/// Args:
///     nvinfer_properties (Dict[str, str]): NvInfer config keys. Use dotted
///         notation ``section.key`` for per-class sections. Bare keys go to
///         ``[property]``. ``process-mode`` and ``output-tensor-meta`` are
///         auto-injected if missing.  ``infer-dims`` must **not** be set
///         here; it is auto-injected from ``model_width`` / ``model_height``.
///     input_format (VideoFormat): Pixel format for appsrc caps
///         (e.g. ``VideoFormat.RGBA``).
///     model_width (int): Model input tensor width in pixels.
///     model_height (int): Model input tensor height in pixels.
///     model_color_format (ModelColorFormat): Model input color space.
///         Default: ``ModelColorFormat.RGB``.
///     name (str): Optional instance name for logging.
///     element_properties (Optional[Dict[str, str]]): Additional GStreamer
///         element properties.
///     gpu_id (int): GPU device ID.
///     queue_depth (int): GStreamer queue max-size-buffers (0 = no queue).
///     input_channel_capacity (int): Bounded input channel capacity; when full,
///         ``NvInfer.submit`` blocks. Default: ``16``.
///     output_channel_capacity (int): Bounded output channel capacity. Default: ``16``.
///     drain_poll_interval_ms (int): How often the framework drain thread polls
///         appsink. Default: ``100``.
///     meta_clear_policy (MetaClearPolicy): When to clear object metadata.
///     disable_output_host_copy (bool): When ``True``, skip device-to-host
///         copy of output tensors. Only device (GPU) pointers will be valid.
///         Default: ``False``.
///     scaling (ModelInputScaling): How frames are scaled to model input size.
///         Default: ``ModelInputScaling.FILL``.
///     operation_timeout_ms (int): Maximum time (in milliseconds) for the
///         framework watchdog (in-flight deadline). When exceeded, the pipeline
///         enters a terminal failed state. Default: ``30000``.
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
        model_width,
        model_height,
        *,
        model_color_format = PyModelColorFormat::Rgb,
        name = String::new(),
        element_properties = None,
        gpu_id = 0,
        queue_depth = 0,
        input_channel_capacity = 16usize,
        output_channel_capacity = 16usize,
        drain_poll_interval_ms = 100u64,
        meta_clear_policy = PyMetaClearPolicy::Before,
        disable_output_host_copy = false,
        scaling = PyModelInputScaling::Fill,
        operation_timeout_ms = 30000,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        nvinfer_properties: HashMap<String, String>,
        input_format: PyVideoFormat,
        model_width: u32,
        model_height: u32,
        model_color_format: PyModelColorFormat,
        name: String,
        element_properties: Option<HashMap<String, String>>,
        gpu_id: u32,
        queue_depth: u32,
        input_channel_capacity: usize,
        output_channel_capacity: usize,
        drain_poll_interval_ms: u64,
        meta_clear_policy: PyMetaClearPolicy,
        disable_output_host_copy: bool,
        scaling: PyModelInputScaling,
        operation_timeout_ms: u64,
    ) -> Self {
        let fmt: VideoFormat = input_format.into();
        let mut cfg = NvInferConfig::new(
            nvinfer_properties,
            fmt,
            model_width,
            model_height,
            model_color_format.into(),
        )
        .gpu_id(gpu_id)
        .queue_depth(queue_depth)
        .input_channel_capacity(input_channel_capacity)
        .output_channel_capacity(output_channel_capacity)
        .drain_poll_interval(std::time::Duration::from_millis(drain_poll_interval_ms))
        .meta_clear_policy(meta_clear_policy.into())
        .disable_output_host_copy(disable_output_host_copy)
        .scaling(scaling.into())
        .operation_timeout(std::time::Duration::from_millis(operation_timeout_ms));
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
    fn input_channel_capacity(&self) -> usize {
        self.inner.input_channel_capacity
    }

    #[getter]
    fn output_channel_capacity(&self) -> usize {
        self.inner.output_channel_capacity
    }

    #[getter]
    fn drain_poll_interval_ms(&self) -> u64 {
        self.inner.drain_poll_interval.as_millis() as u64
    }

    /// NvInfer config keys (copy of the map passed at construction).
    #[getter]
    fn nvinfer_properties(&self) -> HashMap<String, String> {
        self.inner.nvinfer_properties.clone()
    }

    /// Additional GStreamer element properties (copy).
    #[getter]
    fn element_properties(&self) -> HashMap<String, String> {
        self.inner.element_properties.clone()
    }

    /// Model input tensor ``(width, height)`` in pixels — same as ``model_width`` and ``model_height``.
    fn model_input_dimensions(&self) -> (u32, u32) {
        self.inner.model_input_dimensions()
    }

    #[getter]
    fn input_format(&self) -> PyVideoFormat {
        self.inner.input_format.into()
    }

    #[getter]
    fn model_width(&self) -> u32 {
        self.inner.model_width
    }

    #[getter]
    fn model_height(&self) -> u32 {
        self.inner.model_height
    }

    #[getter]
    fn model_color_format(&self) -> PyModelColorFormat {
        self.inner.model_color_format.into()
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

    /// How input frames are scaled to the model input size.
    #[getter]
    fn scaling(&self) -> PyModelInputScaling {
        self.inner.scaling.into()
    }

    /// Operation timeout in milliseconds.
    #[getter]
    fn operation_timeout_ms(&self) -> u64 {
        self.inner.operation_timeout.as_millis() as u64
    }

    fn __repr__(&self) -> String {
        format!(
            "NvInferConfig(name={:?}, gpu_id={}, queue_depth={}, \
             in_ch={}, out_ch={}, drain_poll_ms={}, input_format={}, \
             model_width={}, model_height={}, model_color_format={}, \
             meta_clear_policy={:?}, disable_output_host_copy={}, scaling={})",
            self.inner.name,
            self.inner.gpu_id,
            self.inner.queue_depth,
            self.inner.input_channel_capacity,
            self.inner.output_channel_capacity,
            self.inner.drain_poll_interval.as_millis(),
            self.inner.input_format.gst_name(),
            self.inner.model_width,
            self.inner.model_height,
            PyModelColorFormat::from(self.inner.model_color_format).repr_str(),
            PyMetaClearPolicy::from(self.inner.meta_clear_policy),
            self.inner.disable_output_host_copy,
            PyModelInputScaling::from(self.inner.scaling).repr_str(),
        )
    }
}
