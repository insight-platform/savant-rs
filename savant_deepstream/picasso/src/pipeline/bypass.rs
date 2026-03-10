use crate::callbacks::Callbacks;
use crate::message::OutputMessage;
use deepstream_nvbufsurface::SurfaceView;
use log::{debug, error};
use savant_core::primitives::frame::{VideoFrameProxy, VideoFrameTranscodingMethod};
use std::sync::Arc;

/// Bypass mode: mark the frame as copy (no re-encoding), transform bboxes
/// from current → initial coordinates, then fire `on_bypass_frame`.
pub(crate) fn process_bypass(
    source_id: &str,
    mut frame: VideoFrameProxy,
    view: SurfaceView,
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
    // `view` (SurfaceView) is dropped here after the callback returns,
    // keeping the underlying GstBuffer alive during callback execution.
    drop(view);
}
