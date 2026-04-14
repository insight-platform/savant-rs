//! Callback output types for [`super::FlexibleDecoder`].

use deepstream_decoders::DecoderError;
use gstreamer as gst;
use savant_core::primitives::frame::VideoFrameProxy;
use savant_gstreamer::Codec;

/// Codec, width and height snapshot for a decoder session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecoderParameters {
    pub codec: Codec,
    pub width: i64,
    pub height: i64,
}

/// Callback payload emitted by [`super::FlexibleDecoder`].
#[derive(Debug)]
pub enum FlexibleDecoderOutput {
    /// Decoded RGBA frame from the underlying `NvDecoder`, paired with the
    /// original [`VideoFrameProxy`] that was submitted.
    Frame {
        frame: VideoFrameProxy,
        decoded: deepstream_decoders::DecodedFrame,
    },
    /// Codec or resolution changed.  Old decoder has been drained; new one is
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
