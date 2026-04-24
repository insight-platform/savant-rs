//! Common demuxer data types shared between `mp4_demuxer` and `uri_demuxer`.

pub mod helpers;

use savant_core::primitives::video_codec::VideoCodec;

/// A single demuxed elementary stream packet.
#[derive(Debug, Clone)]
pub struct DemuxedPacket {
    pub data: Vec<u8>,
    pub pts_ns: u64,
    pub dts_ns: Option<u64>,
    pub duration_ns: Option<u64>,
    pub is_keyframe: bool,
}

/// Video-stream metadata extracted from container caps at demux time.
///
/// Dimensions are the **encoded** width/height — QuickTime display-orientation
/// metadata is NOT applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoInfo {
    pub codec: VideoCodec,
    pub width: u32,
    pub height: u32,
    /// Framerate numerator. `0` if the container does not advertise a rate.
    pub framerate_num: u32,
    /// Framerate denominator. `1` if the container does not advertise a rate.
    pub framerate_den: u32,
}
