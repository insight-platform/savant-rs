use crate::callbacks::Callbacks;
use crate::message::BypassOutput;
use gstreamer as gst;
use log::{debug, error};
use savant_core::primitives::frame::VideoFrameProxy;
use std::sync::Arc;

/// Bypass mode: transform bboxes from current → initial coordinates,
/// then fire `on_bypass_frame`.
pub fn process_bypass(
    source_id: &str,
    mut frame: VideoFrameProxy,
    buffer: gst::Buffer,
    callbacks: &Arc<Callbacks>,
) {
    if let Err(e) = frame.transform_backward() {
        error!("bypass: source={source_id}, transform_backward failed: {e}");
        return;
    }

    debug!("bypass: source={source_id}");

    if let Some(cb) = &callbacks.on_bypass_frame {
        cb.call(BypassOutput {
            source_id: source_id.to_string(),
            frame,
            buffer,
        });
    }
}
