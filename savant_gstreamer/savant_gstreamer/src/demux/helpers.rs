use gstreamer as gst;
use gstreamer::prelude::*;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex};

use savant_core::primitives::video_codec::VideoCodec;

/// Error building the parser + capsfilter chain.
///
/// Variants are preserved as distinct categories so callers can map them to
/// their own structured error types (e.g. `LinkError` vs `ElementCreation`)
/// without losing the failure class.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ParserChainError {
    /// Failed to create or query a GStreamer element / pad.
    ElementCreation(String),
    /// Failed to add or link an element in the parser chain.
    LinkError(String),
    /// `sync_state_with_parent` (or equivalent) failed.
    StateChangeFailed(String),
}

impl std::fmt::Display for ParserChainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ElementCreation(s) => write!(f, "{s}"),
            Self::LinkError(s) => write!(f, "{s}"),
            Self::StateChangeFailed(s) => write!(f, "{s}"),
        }
    }
}

/// Error converting a GStreamer sample to a demuxed packet; caller maps to its demuxer error type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SampleError {
    pub src: String,
    pub msg: String,
    pub debug: String,
}

/// Notify the condvar that the demuxer is done.
pub(crate) fn signal_done(done_pair: &(Mutex<bool>, Condvar)) {
    let (lock, cvar) = done_pair;
    *lock.lock().unwrap() = true;
    cvar.notify_all();
}

/// Notify the condvar that stream info is known or unavailable because the
/// pipeline has already terminated.
pub(crate) fn signal_info_done(info_pair: &(Mutex<bool>, Condvar)) {
    let (lock, cvar) = info_pair;
    *lock.lock().unwrap() = true;
    cvar.notify_all();
}

/// Build [`super::VideoInfo`] from caps, when enough fields are available.
pub(crate) fn video_info_from_caps(caps: &gst::CapsRef) -> Option<super::VideoInfo> {
    let codec = codec_from_caps(caps)?;
    let s = caps.structure(0)?;
    let width = s
        .get::<i32>("width")
        .ok()
        .and_then(|v| u32::try_from(v).ok())?;
    let height = s
        .get::<i32>("height")
        .ok()
        .and_then(|v| u32::try_from(v).ok())?;
    let (framerate_num, framerate_den) = s
        .get::<gst::Fraction>("framerate")
        .ok()
        .map(|f| (u32::try_from(f.numer()).ok(), u32::try_from(f.denom()).ok()))
        .map(|(num, den)| (num.unwrap_or(0), den.unwrap_or(1)))
        .unwrap_or((0, 1));
    Some(super::VideoInfo {
        codec,
        width,
        height,
        framerate_num,
        framerate_den,
    })
}

/// Try to detect and emit stream info from caps exactly once.
pub(crate) fn maybe_emit_stream_info_from_caps<E>(
    caps: Option<&gstreamer::CapsRef>,
    detected_codec: &Mutex<Option<VideoCodec>>,
    video_info: &Mutex<Option<super::VideoInfo>>,
    stream_info_fired: &AtomicBool,
    info_pair: &(Mutex<bool>, Condvar),
    emit_stream_info: &E,
) where
    E: Fn(super::VideoInfo) + Send + Sync + ?Sized,
{
    let Some(caps) = caps else {
        return;
    };

    let maybe_codec = codec_from_caps(caps);
    if let Some(codec) = maybe_codec {
        let mut detected_codec_guard = detected_codec.lock().unwrap();
        if detected_codec_guard.is_none() {
            *detected_codec_guard = Some(codec);
        }
    }

    let Some(info) = video_info_from_caps(caps) else {
        return;
    };

    // Write `video_info` before flipping `stream_info_fired` so readers that
    // take the fast path (`stream_info_fired.load()` -> `video_info()`) never
    // observe the atomic set without the corresponding `video_info` value.
    {
        let mut guard = video_info.lock().unwrap();
        if guard.is_some() {
            return;
        }
        *guard = Some(info);
    }

    if stream_info_fired
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
    {
        signal_info_done(info_pair);
        emit_stream_info(info);
    }
}

/// Convert a GStreamer sample to a [`super::DemuxedPacket`], updating
/// `detected_codec` on the first sample with caps.
pub(crate) fn sample_to_packet<E>(
    sample: gst::Sample,
    detected_codec: &Mutex<Option<VideoCodec>>,
    video_info: &Mutex<Option<super::VideoInfo>>,
    stream_info_fired: &AtomicBool,
    info_pair: &(Mutex<bool>, Condvar),
    emit_stream_info: &E,
) -> Result<super::DemuxedPacket, SampleError>
where
    E: Fn(super::VideoInfo) + Send + Sync + ?Sized,
{
    maybe_emit_stream_info_from_caps(
        sample.caps(),
        detected_codec,
        video_info,
        stream_info_fired,
        info_pair,
        emit_stream_info,
    );

    let Some(buffer) = sample.buffer() else {
        return Err(SampleError {
            src: "appsink".to_string(),
            msg: "sample has no buffer".to_string(),
            debug: String::new(),
        });
    };
    let map = buffer.map_readable().map_err(|e| SampleError {
        src: "appsink".to_string(),
        msg: format!("unable to map buffer: {e}"),
        debug: String::new(),
    })?;
    let pts_ns = buffer.pts().map(|t| t.nseconds()).unwrap_or(0);
    let dts_ns = buffer.dts().map(|t| t.nseconds());
    let duration_ns = buffer.duration().map(|t| t.nseconds());
    let is_keyframe = !buffer.flags().contains(gst::BufferFlags::DELTA_UNIT);
    Ok(super::DemuxedPacket {
        data: map.as_slice().to_vec(),
        pts_ns,
        dts_ns,
        duration_ns,
        is_keyframe,
    })
}

/// Map GStreamer caps to a [`VideoCodec`] value.
pub(crate) fn codec_from_caps(caps: &gst::CapsRef) -> Option<VideoCodec> {
    let s = caps.structure(0)?;
    match s.name().as_str() {
        "video/x-h264" => Some(VideoCodec::H264),
        "video/x-h265" => Some(VideoCodec::Hevc),
        "video/x-av1" => Some(VideoCodec::Av1),
        "video/x-vp8" => Some(VideoCodec::Vp8),
        "video/x-vp9" => Some(VideoCodec::Vp9),
        "image/jpeg" => Some(VideoCodec::Jpeg),
        "image/png" => Some(VideoCodec::Png),
        _ => None,
    }
}

/// Dynamically insert a codec-specific parser (+ byte-stream capsfilter for
/// H.264/HEVC) between the qtdemux pad and the queue.  Returns the parser's
/// sink pad so the caller can link qtdemux's src_pad to it.
pub(crate) fn build_parser_chain(
    pipeline: &gst::Pipeline,
    codec_name: Option<&str>,
    queue_sink_pad: &gst::Pad,
) -> Result<gst::Pad, ParserChainError> {
    let (factory, needs_byte_stream_caps) = match codec_name {
        Some("video/x-h264") => ("h264parse", true),
        Some("video/x-h265") => ("h265parse", true),
        Some("video/x-av1") => ("av1parse", false),
        Some("video/x-vp8") => ("vp8parse", false),
        Some("video/x-vp9") => ("vp9parse", false),
        Some("image/jpeg") => ("jpegparse", false),
        _ => return Err(ParserChainError::ElementCreation("no parser needed".into())),
    };

    let parser = gst::ElementFactory::make(factory)
        .name("demux-parser")
        .build()
        .map_err(|e| ParserChainError::ElementCreation(format!("{factory}: {e}")))?;

    if needs_byte_stream_caps {
        parser.set_property("config-interval", -1i32);
    }

    pipeline
        .add(&parser)
        .map_err(|e| ParserChainError::LinkError(format!("add parser: {e}")))?;

    if needs_byte_stream_caps {
        let media = if factory == "h264parse" {
            "video/x-h264"
        } else {
            "video/x-h265"
        };
        let caps = gst::Caps::builder(media)
            .field("stream-format", "byte-stream")
            .build();
        let capsfilter = gst::ElementFactory::make("capsfilter")
            .name("demux-parser-capsf")
            .property("caps", &caps)
            .build()
            .map_err(|e| ParserChainError::ElementCreation(format!("capsfilter: {e}")))?;

        pipeline
            .add(&capsfilter)
            .map_err(|e| ParserChainError::LinkError(format!("add capsfilter: {e}")))?;

        parser
            .link(&capsfilter)
            .map_err(|e| ParserChainError::LinkError(format!("parser->capsfilter: {e}")))?;

        let capsf_src = capsfilter
            .static_pad("src")
            .ok_or_else(|| ParserChainError::ElementCreation("capsfilter src pad".into()))?;
        capsf_src
            .link(queue_sink_pad)
            .map_err(|e| ParserChainError::LinkError(format!("capsfilter->queue: {e}")))?;

        capsfilter.sync_state_with_parent().map_err(|_| {
            ParserChainError::StateChangeFailed("capsfilter sync_state_with_parent".into())
        })?;
    } else {
        let parser_src = parser
            .static_pad("src")
            .ok_or_else(|| ParserChainError::ElementCreation("parser src pad".into()))?;
        parser_src
            .link(queue_sink_pad)
            .map_err(|e| ParserChainError::LinkError(format!("parser->queue: {e}")))?;
    }

    parser
        .sync_state_with_parent()
        .map_err(|_| ParserChainError::StateChangeFailed("parser sync_state_with_parent".into()))?;

    parser
        .static_pad("sink")
        .ok_or_else(|| ParserChainError::ElementCreation("parser sink pad missing".into()))
}
