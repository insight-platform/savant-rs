//! Callback output types for [`super::FlexibleDecoder`].

use deepstream_decoders::DecoderError;
use gstreamer as gst;
use savant_core::primitives::frame::VideoFrame;
use savant_core::primitives::video_codec::VideoCodec;
use savant_core::utils::release_seal::ReleaseSeal;
use std::sync::Arc;

pub use deepstream_buffers::{Sealed, SealedDelivery};

/// VideoCodec, width and height snapshot for a decoder session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecoderParameters {
    pub codec: VideoCodec,
    pub width: i64,
    pub height: i64,
}

/// Callback payload emitted by [`super::FlexibleDecoder`].
///
/// # Delivery flow (Frame variant)
///
/// 1. Callback receives `FlexibleDecoderOutput::Frame` — reads metadata on
///    [`VideoFrame`] and scalar fields of `DecodedFrame`.
/// 2. Callback calls [`take_delivery`](Self::take_delivery) to get a
///    [`SealedDelivery`] containing the `(frame, buffer)` pair.
/// 3. Callback (or its scope end) drops this struct.  [`Drop`] releases the
///    seal.
/// 4. Downstream calls [`SealedDelivery::unseal`] (or `try_unseal`) to
///    obtain the pair — the call blocks until step 3 completes.
#[derive(Debug)]
pub enum FlexibleDecoderOutput {
    /// Decoded RGBA frame from the underlying `NvDecoder`, paired with the
    /// original [`VideoFrame`] that was submitted.
    ///
    /// The `SharedBuffer` inside `decoded.buffer` can be extracted via
    /// [`take_delivery`](Self::take_delivery).
    Frame {
        frame: VideoFrame,
        decoded: deepstream_decoders::DecodedFrame,
        #[doc(hidden)]
        seal: Arc<ReleaseSeal>,
    },
    /// VideoCodec or resolution changed.  Old decoder has been drained; new one is
    /// active.
    ParameterChange {
        old: DecoderParameters,
        new: DecoderParameters,
    },
    /// Frame was rejected (not submitted to the decoder).
    ///
    /// `data` is the extracted payload when available (`None` for
    /// `SourceIdMismatch` and `NoPayload` where no extraction happened).
    Skipped {
        frame: VideoFrame,
        data: Option<Vec<u8>>,
        reason: SkipReason,
    },
    /// A decoded frame whose `frame_id` had no matching [`VideoFrame`] in
    /// the frame map.  The GPU buffer is still live so the callback can reclaim
    /// or log it.
    OrphanFrame {
        decoded: deepstream_decoders::DecodedFrame,
    },
    /// Logical per-source EOS, ordered with decoded frames.
    SourceEos { source_id: String },
    /// A downstream GStreamer event captured at the pipeline output.
    Event(gst::Event),
    /// Error from the underlying `NvDecoder`.
    Error(DecoderError),
    /// The decoder was torn down (typically because the underlying GStreamer
    /// pipeline's watchdog tripped) and the [`super::FlexibleDecoder`] has
    /// transitioned back to `Idle`.  The next submit re-runs detection /
    /// activation transparently; any frames that were in flight when the
    /// decoder died are surfaced individually as
    /// [`Skipped`](Self::Skipped) with
    /// [`SkipReason::DecoderRestarted`].
    ///
    /// This event is purely informational — the pool has already taken the
    /// recovery action by the time the callback fires.  Hooks may use it to
    /// log, increment metrics, or request a cooperative pipeline stop.
    Restarted {
        /// Source id of the [`super::FlexibleDecoder`] that restarted.
        source_id: String,
        /// Human-readable reason for the restart.  Currently always
        /// `"worker thread exited"` (typically watchdog-induced; the
        /// upstream [`PipelineError`](savant_gstreamer::pipeline::PipelineError)
        /// is reported separately via
        /// [`Error`](Self::Error)`(FrameworkError(...))`).
        reason: String,
        /// Number of in-flight frames that were lost because of the
        /// restart.  Each is also surfaced via
        /// [`Skipped`](Self::Skipped) with
        /// [`SkipReason::DecoderRestarted`].
        lost_frames: usize,
    },
}

impl FlexibleDecoderOutput {
    /// Extract the `(VideoFrame, SharedBuffer)` delivery from a `Frame`
    /// variant as a [`SealedDelivery`].
    ///
    /// Returns `None` for non-`Frame` variants, or if the buffer has already
    /// been taken by a prior call.
    ///
    /// The [`VideoFrame`] is **cloned** (cheap Arc bump) and the
    /// [`deepstream_buffers::SharedBuffer`] is **taken** from the
    /// `DecodedFrame`.  Scalar fields on `decoded` (`pts_ns`, `codec`,
    /// `format`, …) remain readable.
    ///
    /// This mirrors the `take_deliveries()` pattern on the nvinfer / nvtracker
    /// batching operators.  Because `FlexibleDecoder` is single-stream, at most
    /// one delivery is produced per output.
    pub fn take_delivery(&mut self) -> Option<SealedDelivery> {
        match self {
            Self::Frame {
                frame,
                decoded,
                seal,
            } => {
                let buffer = decoded.buffer.take()?;
                Some(Sealed::new((frame.clone(), buffer), Arc::clone(seal)))
            }
            _ => None,
        }
    }
}

impl Drop for FlexibleDecoderOutput {
    fn drop(&mut self) {
        if let Self::Frame { seal, .. } = self {
            seal.release();
        }
    }
}

/// Why a frame was not decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkipReason {
    /// The frame's `source_id` does not match the decoder's bound `source_id`.
    SourceIdMismatch { expected: String, actual: String },
    /// `VideoCodec` not supported or absent.
    UnsupportedCodec(Option<String>),
    /// Frame arrived before the first keyframe (random access point) during
    /// H.264/HEVC stream detection.  Non-decodable without a preceding IDR.
    WaitingForKeyframe,
    /// Too many frames buffered without finding a random access point;
    /// the detection buffer limit was exceeded.
    DetectionBufferOverflow,
    /// While the decoder was still in the `Detecting` state (no keyframe
    /// seen yet), the declared codec or frame dimensions changed, so the
    /// in-progress detection attempt is abandoned and the buffered pre-RAP
    /// packets are surfaced.  Emitted instead of
    /// [`WaitingForKeyframe`](Self::WaitingForKeyframe) on this path so
    /// downstream consumers can distinguish keyframe-pending throughput
    /// stalls from input-parameter renegotiations.
    ParameterChangeDuringDetection {
        /// Whether the codec changed compared to the codec under detection.
        codec_changed: bool,
        /// Whether the width/height changed compared to the dimensions
        /// under detection.
        dims_changed: bool,
    },
    /// Neither the `data` argument nor `frame.get_content()` contained bytes.
    NoPayload,
    /// Payload failed structural validation for the declared codec
    /// (e.g. garbage data labelled as JPEG/PNG).
    InvalidPayload(String),
    /// `NvDecoder::new()` failed when activating.
    DecoderCreationFailed(String),
    /// The decoder was torn down (typically a watchdog-induced restart) while
    /// this frame was in flight; it could not be recovered.  The aggregate
    /// signal is delivered via
    /// [`FlexibleDecoderOutput::Restarted`].
    DecoderRestarted(String),
}
