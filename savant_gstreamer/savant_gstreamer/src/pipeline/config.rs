use std::fmt;
use std::time::Duration;

use gstreamer as gst;

/// One-shot hook invoked on `appsrc`'s `src` pad after the pipeline is linked.
pub type AppsrcPadProbe = Box<dyn FnOnce(&gst::Pad) + Send>;

/// Timestamp ordering policy enforced by the feeder thread before each buffer
/// is pushed into `appsrc`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtsPolicy {
    /// Every buffer's PTS must be strictly greater than the previous buffer's PTS.
    StrictPts,
    /// Enforce ascending DTS when set on the buffer, ascending PTS otherwise.
    /// Natural constraint for decode pipelines that handle B-frames.
    StrictDecodeOrder,
}

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
    /// When `Some`, spawns a background thread that periodically invokes
    /// [`super::runner::GstPipeline::flush_idle`] every interval.
    ///
    /// `flush_idle` is a cheap no-op when the pipeline is busy (buffers
    /// in flight) or when there are no pending rescue-eligible events,
    /// so a short interval (e.g. 50–200 ms) is reasonable.  Useful for
    /// streams with long quiet periods where trailing custom-downstream
    /// events (e.g. `savant.pipeline.source_eos`) would otherwise sit
    /// inside a `GstVideoDecoder` until full pipeline EOS.
    ///
    /// When `None` (default), callers drive flushing explicitly via
    /// [`super::runner::GstPipeline::flush_idle`] (e.g. from their own
    /// `recv_timeout` branch).
    pub idle_flush_interval: Option<Duration>,
    /// Optional one-shot hook on `appsrc`'s `src` pad (after meta bridge).
    pub appsrc_probe: Option<AppsrcPadProbe>,
    /// Timestamp ordering policy applied in the feeder thread. `None` disables validation.
    pub pts_policy: Option<PtsPolicy>,
    /// When `true`, GObjects (`gst::Pipeline`, `AppSrc`) are leaked with
    /// [`std::mem::forget`] after the pipeline transitions to Null instead of
    /// being dropped (which triggers GObject finalization).
    ///
    /// Enable for elements whose `finalize` is slow or hangs (e.g. DeepStream
    /// `nvtracker`, whose `NvMOT_DeInit` blocks for seconds).  Callers must
    /// ensure essential resource cleanup happens *before* shutdown (e.g. via
    /// `GST_NVEVENT_PAD_DELETED`).
    pub leak_on_finalize: bool,
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
            .field("idle_flush_interval", &self.idle_flush_interval)
            .field("appsrc_probe", &self.appsrc_probe.as_ref().map(|_| "<fn>"))
            .field("pts_policy", &self.pts_policy)
            .field("leak_on_finalize", &self.leak_on_finalize)
            .finish()
    }
}
