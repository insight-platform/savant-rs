use std::fmt;
use std::time::Duration;

use gstreamer as gst;

/// One-shot hook invoked on `appsrc`'s `src` pad after the pipeline is linked.
pub type AppsrcPadProbe = Box<dyn FnOnce(&gst::Pad) + Send>;

/// Configuration for [`super::runner::GstPipeline::start`].
///
/// `appsrc_probe`, when set, is invoked once after the pipeline is linked and
/// Savant ID meta is bridged across `appsrc` → `appsink`, with the `appsrc`
/// source pad so callers can install pad probes (e.g. DeepStream batch-size
/// queries).
pub struct PipelineConfig {
    pub name: String,
    pub appsrc_caps: gst::Caps,
    pub elements: Vec<gst::Element>,
    pub input_channel_capacity: usize,
    pub output_channel_capacity: usize,
    pub operation_timeout: Option<Duration>,
    pub drain_poll_interval: Duration,
    /// Optional one-shot hook on `appsrc`'s `src` pad (after meta bridge).
    pub appsrc_probe: Option<AppsrcPadProbe>,
}

impl fmt::Debug for PipelineConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PipelineConfig")
            .field("name", &self.name)
            .field("appsrc_caps", &self.appsrc_caps)
            .field("elements_len", &self.elements.len())
            .field("input_channel_capacity", &self.input_channel_capacity)
            .field("output_channel_capacity", &self.output_channel_capacity)
            .field("operation_timeout", &self.operation_timeout)
            .field("drain_poll_interval", &self.drain_poll_interval)
            .field("appsrc_probe", &self.appsrc_probe.as_ref().map(|_| "<fn>"))
            .finish()
    }
}
