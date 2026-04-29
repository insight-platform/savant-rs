use crate::spec::SourceSpec;
use deepstream_buffers::{Rect, SurfaceView};
use savant_core::primitives::eos::EndOfStream;
use savant_core::primitives::frame::VideoFrame;

/// Messages sent from the engine to per-source worker threads.
pub(crate) enum WorkerMessage {
    /// A new video frame to process.
    Frame(VideoFrame, SurfaceView, Option<Rect>),
    /// End-of-stream signal.
    Eos,
    /// Hot-swap the source spec.
    UpdateSpec(Box<SourceSpec>),
    /// Graceful shutdown.
    Shutdown,
}

/// Output produced by the encoding pipeline.
///
/// For [`VideoFrame`](OutputMessage::VideoFrame), the proxy carries all
/// metadata (pts, dts, duration, keyframe, codec, framerate) and the
/// encoded bitstream in [`VideoFrameContent::Internal`].
pub enum OutputMessage {
    /// An encoded video frame with content stored in the proxy.
    VideoFrame(VideoFrame),
    /// End-of-stream signal for a source.
    EndOfStream(EndOfStream),
}
