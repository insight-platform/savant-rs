//! Errors and output events for the multi-stream decoder.

use deepstream_buffers::SharedBuffer;
use deepstream_decoders::DecoderError;
use savant_core::primitives::frame::VideoFrameProxy;
use savant_gstreamer::Codec;
use std::time::Duration;

/// Single callback payload for all decoder outputs.
#[derive(Debug)]
pub enum DecoderOutput {
    /// Decoded RGBA frame in this stream's output pool (see pool sizing on [`super::decoder::MultiStreamDecoder`]).
    Decoded {
        frame: VideoFrameProxy,
        buffer: SharedBuffer,
    },
    /// Frame could not be decoded (or was dropped before decode).
    Undecoded {
        frame: VideoFrameProxy,
        data: Option<Vec<u8>>,
        reason: UndecodedReason,
    },
    /// End-of-stream for `source_id` after the decoder drained.
    Eos { source_id: String },
    /// A per-stream decoder became active.
    StreamStarted { source_id: String, codec: Codec },
    /// Per-stream decoder stopped (EOS, idle, codec change, fatal error).
    StreamStopped {
        source_id: String,
        reason: StopReason,
    },
    /// Underlying `NvDecoder` pipeline restarted after an error.
    PipelineRestarted {
        source_id: String,
        reason: String,
        lost_frame_count: usize,
    },
}

/// Why a frame was not decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UndecodedReason {
    UnsupportedCodec(Option<String>),
    AwaitingKeyframe,
    DetectionFailed(String),
    DecodeError(String),
    StreamEvicted,
    /// Frame was in-flight when a session boundary (codec / resolution / timestamp
    /// regression) was detected; the old decoder was drained but this frame was
    /// never submitted to it (it was still queued behind the drain EOS).
    SessionReset,
    NoPayload,
    ExternalContent,
}

/// Idle eviction verdict from [`MultiStreamDecoder`](super::decoder::MultiStreamDecoder).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvictionVerdict {
    Approve,
    Extend(Duration),
}

/// Why a stream worker stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    Eos,
    IdleEviction,
    CodecChanged,
    /// Width/height changed on the wire for this `source_id`.
    ResolutionChanged,
    /// `DTS` (or `PTS` when no `DTS`) moved backward vs the previous submitted packet.
    TimestampRegressed,
    Error(String),
    Shutdown,
}

/// Errors returned by submit APIs.
#[derive(Debug, thiserror::Error)]
pub enum MultiStreamError {
    #[error("no compressed payload available for frame")]
    NoData,
    #[error("failed to create decoder: {0}")]
    DecoderCreationFailed(String),
    #[error("failed to create buffer pool: {0}")]
    PoolCreationFailed(String),
    #[error("decoder submit failed: {0}")]
    SubmitFailed(#[from] DecoderError),
    #[error("unknown stream: {0}")]
    UnknownStream(String),
    #[error("per-stream queue full for source_id={source_id} (capacity {queue_size})")]
    QueueFull {
        source_id: String,
        queue_size: usize,
    },
    #[error("worker channel disconnected for source_id={0}")]
    ChannelDisconnected(String),
    #[error(
        "VideoFrame width/height must be positive and fit u32 for output pool (got {width}x{height})"
    )]
    InvalidOutputDimensions { width: i64, height: i64 },
    #[error("hardware JPEG requires dimensions aligned to 8 and >= 16 (got {width}x{height})")]
    InvalidJpegDimensions { width: i64, height: i64 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multi_stream_error_display() {
        let e = MultiStreamError::QueueFull {
            source_id: "s1".into(),
            queue_size: 4,
        };
        assert!(e.to_string().contains("s1"));
    }
}
