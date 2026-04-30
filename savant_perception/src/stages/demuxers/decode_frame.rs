//! Helper for constructing the decoder-facing [`VideoFrame`] that
//! a demuxer hands off as `EncodedMsg::Frame`.
//!
//! Lives under [`stages::demuxers`](super) (rather than under the
//! `decoder` stage) because the function is needed by demuxer
//! `on_packet_as_frame` defaults regardless of whether the
//! `deepstream` feature is enabled.  The `decoder` stage —
//! gated behind `deepstream` — re-exports it for backward
//! compatibility.

use savant_core::primitives::frame::{VideoFrame, VideoFrameContent, VideoFrameTranscodingMethod};
use savant_gstreamer::mp4_demuxer::{DemuxedPacket, VideoInfo};

/// Fallback framerate numerator used when the container does not
/// advertise one (per [`VideoInfo`] contract: `framerate_num == 0`).
pub const FALLBACK_FPS_NUM: i64 = 30;
/// Denominator counterpart to [`FALLBACK_FPS_NUM`].
pub const FALLBACK_FPS_DEN: i64 = 1;

/// Build the per-packet [`VideoFrame`] that an upstream demuxer
/// hands off downstream as `EncodedMsg::Frame`.  The
/// (deepstream-gated) [`Decoder`](super::super::decoder::Decoder)
/// stage consumes the same frame on its
/// [`Handler<FramePayload>`](crate::Handler) path, so demuxer →
/// decoder is wire-compatible regardless of whether the
/// `deepstream` feature is enabled at the demuxer end.
pub fn make_decode_frame(source_id: &str, pkt: &DemuxedPacket, info: &VideoInfo) -> VideoFrame {
    let (fps_num, fps_den) = if info.framerate_num == 0 {
        (FALLBACK_FPS_NUM, FALLBACK_FPS_DEN)
    } else {
        (info.framerate_num as i64, info.framerate_den.max(1) as i64)
    };
    VideoFrame::new(
        source_id,
        (fps_num, fps_den),
        info.width as i64,
        info.height as i64,
        VideoFrameContent::None,
        VideoFrameTranscodingMethod::Copy,
        Some(info.codec),
        Some(pkt.is_keyframe),
        (1, 1_000_000_000),
        pkt.pts_ns as i64,
        pkt.dts_ns.map(|v| v as i64),
        pkt.duration_ns.map(|v| v as i64),
    )
    .expect("VideoFrame::new (decode)")
}
