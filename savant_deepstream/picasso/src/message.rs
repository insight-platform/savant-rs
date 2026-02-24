use crate::spec::SourceSpec;
use gstreamer as gst;
use savant_core::primitives::frame::VideoFrameProxy;

/// Messages sent from the engine to per-source worker threads.
pub enum WorkerMessage {
    /// A new video frame to process.
    Frame(VideoFrameProxy, gst::Buffer),
    /// End-of-stream signal.
    Eos,
    /// Hot-swap the source spec.
    UpdateSpec(Box<SourceSpec>),
    /// Graceful shutdown.
    Shutdown,
}

/// Output produced after encoding a frame (or EOS sentinel).
pub struct EncodedOutput {
    pub source_id: String,
    pub frame: VideoFrameProxy,
    /// `None` for the terminal EOS sentinel.
    pub buffer: Option<gst::Buffer>,
    pub pts: u64,
    pub duration: Option<u64>,
    pub is_keyframe: bool,
    /// `true` only for the terminal EOS sentinel (buffer is `None`).
    pub is_eos: bool,
}

/// Output for bypass mode — frame with bboxes transformed back to initial
/// coordinates.
pub struct BypassOutput {
    pub source_id: String,
    pub frame: VideoFrameProxy,
    pub buffer: gst::Buffer,
}
