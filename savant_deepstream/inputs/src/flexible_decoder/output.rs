//! Callback output types for [`super::FlexibleDecoder`].

use deepstream_decoders::DecoderError;
use savant_core::primitives::frame::VideoFrameProxy;
use savant_gstreamer::Codec;

/// Default idle timeout (max gap between outputs during graceful drain).
pub const DEFAULT_IDLE_TIMEOUT_SECS: u64 = 5;

/// Default max buffered frames during H.264/HEVC detection.
pub const DEFAULT_DETECT_BUFFER_LIMIT: usize = 30;

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
    /// Logical per-source EOS, ordered with decoded frames.
    SourceEos { source_id: String },
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
    /// `NvDecoder::new()` failed when activating.
    DecoderCreationFailed(String),
}
