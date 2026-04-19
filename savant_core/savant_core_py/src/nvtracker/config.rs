//! PyO3 wrapper for NvTrackerConfig.

use super::enums::PyTrackingIdResetMode;
use crate::deepstream::enums::PyMetaClearPolicy;
use crate::deepstream::enums::PyVideoFormat;
use deepstream_nvtracker::NvTrackerConfig;
use pyo3::prelude::*;
use savant_gstreamer::VideoFormat;
use std::collections::HashMap;

/// Configuration for [`super::pipeline::PyNvTracker`].
///
/// Args:
///     ll_lib_file (str): Path to ``libnvds_nvmultiobjecttracker.so``.
///     ll_config_file (str): Path to low-level tracker YAML.
///     input_format (VideoFormat): Pixel format for appsrc caps.
///     name (str): Optional element name prefix for logging.
///     tracker_width (int): ``tracker-width`` property.
///     tracker_height (int): ``tracker-height`` property.
///     max_batch_size (int): Maximum batch size for the tracker element.
///     gpu_id (int): GPU id.
///     element_properties (Optional[Dict[str, str]]): Extra element properties.
///     tracking_id_reset_mode (TrackingIdResetMode): ID reset behaviour.
///     meta_clear_policy (MetaClearPolicy): When to clear ``NvDsObjectMeta``
///         entries from the batch buffer. ``BEFORE`` (default) removes any
///         stale upstream object metas before attaching the new detection
///         objects; ``AFTER`` clears all objects from the output buffer
///         when the :class:`TrackerOutput` is dropped; ``BOTH`` does both;
///         ``NONE`` disables automatic clearing.
///     operation_timeout_ms (int): Maximum time (in milliseconds) for the
///         framework in-flight watchdog. When exceeded, the pipeline enters a
///         terminal failed state. Default: ``30000``.
///     input_channel_capacity (int): Bounded input channel size. Default: ``16``.
///     output_channel_capacity (int): Bounded output channel size. Default: ``16``.
///     drain_poll_interval_ms (int): Drain thread poll interval when idle. Default: ``100``.
#[pyclass(
    name = "NvTrackerConfig",
    module = "savant_rs.nvtracker",
    skip_from_py_object
)]
#[derive(Debug, Clone)]
pub struct PyNvTrackerConfig {
    pub(crate) inner: NvTrackerConfig,
}

#[pymethods]
impl PyNvTrackerConfig {
    #[new]
    #[pyo3(signature = (
        ll_lib_file,
        ll_config_file,
        input_format,
        *,
        name = String::new(),
        tracker_width = deepstream_nvtracker::config::DEFAULT_TRACKER_WIDTH,
        tracker_height = deepstream_nvtracker::config::DEFAULT_TRACKER_HEIGHT,
        max_batch_size = deepstream_nvtracker::config::DEFAULT_MAX_BATCH_SIZE,
        gpu_id = 0u32,
        element_properties = None,
        tracking_id_reset_mode = PyTrackingIdResetMode::None,
        meta_clear_policy = PyMetaClearPolicy::Before,
        operation_timeout_ms = 30000u64,
        input_channel_capacity = 16usize,
        output_channel_capacity = 16usize,
        drain_poll_interval_ms = 100u64,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        ll_lib_file: String,
        ll_config_file: String,
        input_format: PyVideoFormat,
        name: String,
        tracker_width: u32,
        tracker_height: u32,
        max_batch_size: u32,
        gpu_id: u32,
        element_properties: Option<HashMap<String, String>>,
        tracking_id_reset_mode: PyTrackingIdResetMode,
        meta_clear_policy: PyMetaClearPolicy,
        operation_timeout_ms: u64,
        input_channel_capacity: usize,
        output_channel_capacity: usize,
        drain_poll_interval_ms: u64,
    ) -> Self {
        let fmt: VideoFormat = input_format.into();
        let mut cfg = NvTrackerConfig::new(ll_lib_file, ll_config_file);
        cfg.tracker_width = tracker_width;
        cfg.tracker_height = tracker_height;
        cfg.max_batch_size = max_batch_size;
        cfg.gpu_id = gpu_id;
        cfg.input_format = fmt;
        cfg.tracking_id_reset_mode = tracking_id_reset_mode.into();
        cfg.meta_clear_policy = meta_clear_policy.into();
        cfg.operation_timeout = std::time::Duration::from_millis(operation_timeout_ms);
        cfg.input_channel_capacity = input_channel_capacity;
        cfg.output_channel_capacity = output_channel_capacity;
        cfg.drain_poll_interval = std::time::Duration::from_millis(drain_poll_interval_ms);
        if !name.is_empty() {
            cfg.name = name;
        }
        if let Some(ep) = element_properties {
            cfg.element_properties = ep;
        }
        Self { inner: cfg }
    }

    #[getter]
    fn ll_lib_file(&self) -> String {
        self.inner.ll_lib_file.clone()
    }

    #[getter]
    fn ll_config_file(&self) -> String {
        self.inner.ll_config_file.clone()
    }

    /// Operation timeout in milliseconds.
    #[getter]
    fn operation_timeout_ms(&self) -> u64 {
        self.inner.operation_timeout.as_millis() as u64
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

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    #[getter]
    fn tracker_width(&self) -> u32 {
        self.inner.tracker_width
    }

    #[getter]
    fn tracker_height(&self) -> u32 {
        self.inner.tracker_height
    }

    #[getter]
    fn max_batch_size(&self) -> u32 {
        self.inner.max_batch_size
    }

    #[getter]
    fn gpu_id(&self) -> u32 {
        self.inner.gpu_id
    }

    /// `NvDsObjectMeta` clearing policy.
    #[getter]
    fn meta_clear_policy(&self) -> PyMetaClearPolicy {
        self.inner.meta_clear_policy.into()
    }

    fn __repr__(&self) -> String {
        format!(
            "NvTrackerConfig(name={:?}, gpu_id={}, max_batch_size={}, \
             meta_clear_policy={:?}, operation_timeout_ms={}, \
             input_channel_capacity={}, output_channel_capacity={}, \
             drain_poll_interval_ms={})",
            self.inner.name,
            self.inner.gpu_id,
            self.inner.max_batch_size,
            self.inner.meta_clear_policy,
            self.operation_timeout_ms(),
            self.input_channel_capacity(),
            self.output_channel_capacity(),
            self.drain_poll_interval_ms(),
        )
    }
}
