pub(crate) mod bypass;
pub mod encode;

use gstreamer as gst;
use savant_core::primitives::frame::VideoFrameProxy;

/// Per-frame data bundle passed into encode / render pipeline stages.
pub(crate) struct FrameInput {
    pub(crate) frame: VideoFrameProxy,
    pub(crate) buffer: gst::Buffer,
    pub(crate) frame_id: u128,
}
