use crate::spec::SourceSpec;
use deepstream_nvbufsurface::{Rect, SurfaceView};
use savant_core::primitives::eos::EndOfStream;
use savant_core::primitives::frame::VideoFrameProxy;

/// Messages sent from the engine to per-source worker threads.
pub enum WorkerMessage {
    /// A new video frame to process.
    Frame(VideoFrameProxy, SurfaceView, Option<Rect>),
    /// End-of-stream signal.
    Eos,
    /// Hot-swap the source spec.
    UpdateSpec(Box<SourceSpec>),
    /// Graceful shutdown.
    Shutdown,
}

/// Output produced by the encoding pipeline.
///
/// For [`VideoFrame`](EncodedOutput::VideoFrame), the proxy carries all
/// metadata (pts, dts, duration, keyframe, codec, framerate) and the
/// encoded bitstream in [`VideoFrameContent::Internal`].
pub enum EncodedOutput {
    /// An encoded video frame with content stored in the proxy.
    VideoFrame(VideoFrameProxy),
    /// End-of-stream signal for a source.
    EndOfStream(EndOfStream),
}
