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

/// Apply pts, dts, and duration from [`VideoFrameProxy`] to a [`gst::Buffer`].
/// Call this at pipeline entry so the buffer carries frame timestamps rather
/// than assuming they were set by the producer.
pub(crate) fn apply_frame_timestamps_to_buffer(
    frame: &VideoFrameProxy,
    buf_ref: &mut gst::BufferRef,
) {
    let pts_ns = frame.get_pts().max(0) as u64;
    buf_ref.set_pts(gst::ClockTime::from_nseconds(pts_ns));
    buf_ref.set_dts(gst::ClockTime::from_nseconds(pts_ns));
    buf_ref.set_duration(
        frame
            .get_duration()
            .map(|d| gst::ClockTime::from_nseconds(d.max(0) as u64)),
    );
}
