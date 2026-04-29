use crate::callbacks::Callbacks;
use crate::message::OutputMessage;
use deepstream_buffers::SurfaceView;
use log::{debug, error};
use savant_core::primitives::frame::{VideoFrame, VideoFrameTranscodingMethod};
use std::sync::Arc;

/// Bypass mode: mark the frame as copy (no re-encoding), transform bboxes
/// from current → initial coordinates, then fire `on_bypass_frame`.
pub(crate) fn process_bypass(
    source_id: &str,
    mut frame: VideoFrame,
    _view: SurfaceView,
    callbacks: &Arc<Callbacks>,
) {
    frame.set_transcoding_method(VideoFrameTranscodingMethod::Copy);

    if let Err(e) = frame.transform_backward() {
        error!("bypass: source={source_id}, transform_backward failed: {e}");
        return;
    }

    debug!("bypass: source={source_id}");

    if let Some(cb) = &callbacks.on_bypass_frame {
        cb.call(OutputMessage::VideoFrame(frame));
    }
    // `view` stays alive (not moved into the callback) until function exit,
    // so the underlying GstBuffer remains valid for the callback's duration.
}
