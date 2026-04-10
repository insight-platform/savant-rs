use std::time::Duration;

use gstreamer as gst;

#[derive(Debug)]
pub struct PipelineConfig {
    pub name: String,
    pub appsrc_caps: gst::Caps,
    pub elements: Vec<gst::Element>,
    pub input_channel_capacity: usize,
    pub output_channel_capacity: usize,
    pub operation_timeout: Option<Duration>,
    pub drain_poll_interval: Duration,
}
