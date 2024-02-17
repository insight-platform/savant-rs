use savant_core::primitives::frame::VideoFrameProxy;

pub fn process_frame(frame: &mut VideoFrameProxy) {
    frame.get_keyframe().unwrap_or(false);
}
