pub mod bypass;
pub mod encode;

use gstreamer as gst;
use savant_core::primitives::frame::VideoFrameProxy;

/// Per-frame data bundle passed into encode / render pipeline stages.
pub struct FrameInput {
    pub frame: VideoFrameProxy,
    pub buffer: gst::Buffer,
    pub frame_id: u128,
}
