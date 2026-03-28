//! PyO3 wrapper for NvTrackerConfig.

use super::enums::PyTrackingIdResetMode;
use crate::deepstream::enums::PyVideoFormat;
use nvtracker::NvTrackerConfig;
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
///     queue_depth (int): GStreamer queue ``max-size-buffers`` (0 = no queue).
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
        tracker_width = nvtracker::config::DEFAULT_TRACKER_WIDTH,
        tracker_height = nvtracker::config::DEFAULT_TRACKER_HEIGHT,
        max_batch_size = nvtracker::config::DEFAULT_MAX_BATCH_SIZE,
        gpu_id = 0u32,
        element_properties = None,
        tracking_id_reset_mode = PyTrackingIdResetMode::None,
        queue_depth = 0u32,
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
        queue_depth: u32,
    ) -> Self {
        let fmt: VideoFormat = input_format.into();
        let mut cfg = NvTrackerConfig::new(ll_lib_file, ll_config_file);
        cfg.tracker_width = tracker_width;
        cfg.tracker_height = tracker_height;
        cfg.max_batch_size = max_batch_size;
        cfg.gpu_id = gpu_id;
        cfg.input_format = fmt;
        cfg.tracking_id_reset_mode = tracking_id_reset_mode.into();
        cfg.queue_depth = queue_depth;
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

    #[getter]
    fn queue_depth(&self) -> u32 {
        self.inner.queue_depth
    }
}
