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
///     meta_clear_policy (MetaClearPolicy): When to clear object metadata.
///     disable_output_host_copy (bool): When ``True``, skip device-to-host
///         copy of output tensors. Only device (GPU) pointers will be valid.
///         Default: ``False``.
///     scaling (ModelInputScaling): How frames are scaled to model input size.
///         Default: ``ModelInputScaling.FILL``.
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
        meta_clear_policy = PyMetaClearPolicy::Before,
        disable_output_host_copy = false,
        scaling = PyModelInputScaling::Fill,
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
        meta_clear_policy: PyMetaClearPolicy,
        disable_output_host_copy: bool,
        scaling: PyModelInputScaling,
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
        .meta_clear_policy(meta_clear_policy.into())
        .disable_output_host_copy(disable_output_host_copy)
        .scaling(scaling.into());
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

    fn __repr__(&self) -> String {
        format!(
            "NvInferConfig(name={:?}, gpu_id={}, queue_depth={}, input_format={}, \
             model_width={}, model_height={}, model_color_format={}, \
             meta_clear_policy={:?}, disable_output_host_copy={}, scaling={})",
            self.inner.name,
            self.inner.gpu_id,
            self.inner.queue_depth,
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
