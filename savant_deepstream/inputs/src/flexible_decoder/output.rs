//! Callback output types for [`super::FlexibleDecoder`].

use deepstream_decoders::DecoderError;
use gstreamer as gst;
use savant_core::primitives::frame::VideoFrameProxy;
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
///    [`VideoFrameProxy`] and scalar fields of [`DecodedFrame`].
/// 2. Callback calls [`take_delivery`](Self::take_delivery) to get a
///    [`SealedDelivery`] containing the `(frame, buffer)` pair.
/// 3. Callback (or its scope end) drops this struct.  [`Drop`] releases the
///    seal.
/// 4. Downstream calls [`SealedDelivery::unseal`] (or `try_unseal`) to
///    obtain the pair — the call blocks until step 3 completes.
#[derive(Debug)]
pub enum FlexibleDecoderOutput {
    /// Decoded RGBA frame from the underlying `NvDecoder`, paired with the
    /// original [`VideoFrameProxy`] that was submitted.
    ///
    /// The [`SharedBuffer`] inside `decoded.buffer` can be extracted via
    /// [`take_delivery`](Self::take_delivery).
    Frame {
        frame: VideoFrameProxy,
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
        frame: VideoFrameProxy,
        data: Option<Vec<u8>>,
        reason: SkipReason,
    },
    /// A decoded frame whose `frame_id` had no matching [`VideoFrameProxy`] in
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
}

impl FlexibleDecoderOutput {
    /// Extract the `(VideoFrameProxy, SharedBuffer)` delivery from a `Frame`
    /// variant as a [`SealedDelivery`].
    ///
    /// Returns `None` for non-`Frame` variants, or if the buffer has already
    /// been taken by a prior call.
    ///
    /// The [`VideoFrameProxy`] is **cloned** (cheap Arc bump) and the
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
    /// Neither the `data` argument nor `frame.get_content()` contained bytes.
    NoPayload,
    /// Payload failed structural validation for the declared codec
    /// (e.g. garbage data labelled as JPEG/PNG).
    InvalidPayload(String),
    /// `NvDecoder::new()` failed when activating.
    DecoderCreationFailed(String),
}
