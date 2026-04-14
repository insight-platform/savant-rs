//! Minimal GStreamer MP4 demuxer: `filesrc -> qtdemux -> queue -> appsink`.
//!
//! Reads encoded packets from an MP4 (QuickTime) container and exposes them as
//! elementary stream payloads with timestamps.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use thiserror::Error;

use crate::codec::Codec;

#[derive(Debug, Clone)]
pub struct DemuxedPacket {
    pub data: Vec<u8>,
    pub pts_ns: u64,
    pub dts_ns: Option<u64>,
    pub duration_ns: Option<u64>,
    pub is_keyframe: bool,
}

#[derive(Debug, Error)]
pub enum Mp4DemuxerError {
    #[error("Input file does not exist: {0}")]
    InputNotFound(String),
    #[error("Failed to create GStreamer element: {0}")]
    ElementCreation(String),
    #[error("Failed to link elements: {0}")]
    LinkError(String),
    #[error("Pipeline state change failed")]
    StateChangeFailed,
    #[error("Timed out waiting for demuxed packet")]
    PullTimeout,
    #[error("Pipeline error from {src}: {msg} ({debug})")]
    PipelineError {
        src: String,
        msg: String,
        debug: String,
    },
    #[error("Demuxer already finished")]
    AlreadyFinished,
}

pub struct Mp4Demuxer {
    pipeline: gst::Pipeline,
    appsink: gst_app::AppSink,
    linked_video_pad: Arc<AtomicBool>,
    finished: bool,
    detected_codec: Option<Codec>,
}

impl Mp4Demuxer {
    /// Create a demuxer that outputs raw container-format packets (e.g. AVC
    /// length-prefixed NALUs for H.264 in MP4).
    pub fn new(input_path: &str) -> Result<Self, Mp4DemuxerError> {
        Self::new_inner(input_path, false)
    }

    /// Create a demuxer that inserts codec-specific parsers to convert output
    /// to byte-stream / Annex-B format. Use this when feeding packets to a
    /// standalone decoder (`NvDecoder`) that expects byte-stream input.
    pub fn new_parsed(input_path: &str) -> Result<Self, Mp4DemuxerError> {
        Self::new_inner(input_path, true)
    }

    fn new_inner(input_path: &str, parsed: bool) -> Result<Self, Mp4DemuxerError> {
        let _ = gst::init();
        if !Path::new(input_path).exists() {
            return Err(Mp4DemuxerError::InputNotFound(input_path.to_string()));
        }

        let pipeline = gst::Pipeline::new();

        let src = gst::ElementFactory::make("filesrc")
            .name("demux-src")
            .build()
            .map_err(|e| Mp4DemuxerError::ElementCreation(format!("filesrc: {e}")))?;
        src.set_property("location", input_path);

        let demux = gst::ElementFactory::make("qtdemux")
            .name("demux")
            .build()
            .map_err(|e| Mp4DemuxerError::ElementCreation(format!("qtdemux: {e}")))?;

        let queue = gst::ElementFactory::make("queue")
            .name("demux-queue")
            .build()
            .map_err(|e| Mp4DemuxerError::ElementCreation(format!("queue: {e}")))?;

        let sink_elem = gst::ElementFactory::make("appsink")
            .name("demux-sink")
            .build()
            .map_err(|e| Mp4DemuxerError::ElementCreation(format!("appsink: {e}")))?;
        sink_elem.set_property("sync", false);
        sink_elem.set_property("emit-signals", false);
        sink_elem.set_property("max-buffers", 64u32);
        sink_elem.set_property("drop", false);

        pipeline
            .add_many([&src, &demux, &queue, &sink_elem])
            .map_err(|e| Mp4DemuxerError::LinkError(format!("add_many: {e}")))?;

        src.link(&demux)
            .map_err(|e| Mp4DemuxerError::LinkError(format!("filesrc->qtdemux: {e}")))?;
        queue
            .link(&sink_elem)
            .map_err(|e| Mp4DemuxerError::LinkError(format!("queue->appsink: {e}")))?;

        let queue_sink_pad = queue
            .static_pad("sink")
            .ok_or_else(|| Mp4DemuxerError::ElementCreation("queue sink pad missing".into()))?;

        let linked_video_pad = Arc::new(AtomicBool::new(false));
        let linked_video_pad_closure = linked_video_pad.clone();
        let pipeline_for_pad = pipeline.clone();
        demux.connect_pad_added(move |_demux, src_pad| {
            if linked_video_pad_closure.load(Ordering::SeqCst) {
                return;
            }

            let codec_name = src_pad
                .current_caps()
                .or_else(|| Some(src_pad.query_caps(None)))
                .and_then(|caps| caps.structure(0).map(|s| s.name().to_string()));

            let is_video = codec_name
                .as_deref()
                .is_some_and(|n| n.starts_with("video/") || n.starts_with("image/"));
            if !is_video {
                return;
            }

            if queue_sink_pad.is_linked() {
                return;
            }

            let target_pad = if parsed {
                match build_parser_chain(&pipeline_for_pad, codec_name.as_deref(), &queue_sink_pad)
                {
                    Ok(parser_sink_pad) => parser_sink_pad,
                    Err(_) => queue_sink_pad.clone(),
                }
            } else {
                queue_sink_pad.clone()
            };

            if src_pad.link(&target_pad).is_ok() {
                linked_video_pad_closure.store(true, Ordering::SeqCst);
            }
        });

        let appsink = sink_elem
            .dynamic_cast::<gst_app::AppSink>()
            .map_err(|_| Mp4DemuxerError::ElementCreation("appsink cast failed".into()))?;

        let ret = pipeline.set_state(gst::State::Playing);
        if ret == Err(gst::StateChangeError) {
            return Err(Mp4DemuxerError::StateChangeFailed);
        }

        Ok(Self {
            pipeline,
            appsink,
            linked_video_pad,
            finished: false,
            detected_codec: None,
        })
    }

    /// Pull the next demuxed packet with a 5-second default timeout.
    ///
    /// Returns `Ok(None)` on EOS.
    pub fn pull(&mut self) -> Result<Option<DemuxedPacket>, Mp4DemuxerError> {
        self.pull_timeout(Duration::from_secs(5))
    }

    /// Pull the next demuxed packet with a caller-specified timeout.
    ///
    /// Returns `Ok(None)` on EOS.
    pub fn pull_timeout(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<DemuxedPacket>, Mp4DemuxerError> {
        if self.finished {
            return Err(Mp4DemuxerError::AlreadyFinished);
        }

        if let Some(msg) = self.poll_bus_error_or_eos() {
            return msg;
        }

        if !self.linked_video_pad.load(Ordering::SeqCst) {
            if let Some(msg) = self.poll_bus_error_or_eos() {
                return msg;
            }
        }

        let gst_timeout = gst::ClockTime::from_nseconds(timeout.as_nanos() as u64);
        match self.appsink.try_pull_sample(gst_timeout) {
            Some(sample) => self.sample_to_packet(sample).map(Some),
            None => {
                if let Some(msg) = self.poll_bus_error_or_eos() {
                    msg
                } else if self.appsink.is_eos() {
                    self.finish();
                    Ok(None)
                } else {
                    Err(Mp4DemuxerError::PullTimeout)
                }
            }
        }
    }

    pub fn detected_codec(&self) -> Option<Codec> {
        self.detected_codec
    }

    pub fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        let _ = self.pipeline.set_state(gst::State::Null);
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    fn sample_to_packet(&mut self, sample: gst::Sample) -> Result<DemuxedPacket, Mp4DemuxerError> {
        if self.detected_codec.is_none() {
            self.detected_codec = sample.caps().and_then(Self::codec_from_caps);
        }

        let Some(buffer) = sample.buffer() else {
            return Err(Mp4DemuxerError::PipelineError {
                src: "appsink".to_string(),
                msg: "sample has no buffer".to_string(),
                debug: String::new(),
            });
        };
        let map = buffer
            .map_readable()
            .map_err(|e| Mp4DemuxerError::PipelineError {
                src: "appsink".to_string(),
                msg: format!("unable to map buffer: {e}"),
                debug: String::new(),
            })?;
        let pts_ns = buffer.pts().map(|t| t.nseconds()).unwrap_or(0);
        let dts_ns = buffer.dts().map(|t| t.nseconds());
        let duration_ns = buffer.duration().map(|t| t.nseconds());
        let is_keyframe = !buffer.flags().contains(gst::BufferFlags::DELTA_UNIT);
        Ok(DemuxedPacket {
            data: map.as_slice().to_vec(),
            pts_ns,
            dts_ns,
            duration_ns,
            is_keyframe,
        })
    }

    fn codec_from_caps(caps: &gst::CapsRef) -> Option<Codec> {
        let s = caps.structure(0)?;
        match s.name().as_str() {
            "video/x-h264" => Some(Codec::H264),
            "video/x-h265" => Some(Codec::Hevc),
            "video/x-av1" => Some(Codec::Av1),
            "video/x-vp8" => Some(Codec::Vp8),
            "video/x-vp9" => Some(Codec::Vp9),
            "image/jpeg" => Some(Codec::Jpeg),
            "image/png" => Some(Codec::Png),
            _ => None,
        }
    }

    fn poll_bus_error_or_eos(&mut self) -> Option<Result<Option<DemuxedPacket>, Mp4DemuxerError>> {
        let bus = self.pipeline.bus()?;
        let msg = bus.pop_filtered(&[gst::MessageType::Error, gst::MessageType::Eos])?;
        match msg.view() {
            gst::MessageView::Eos(..) => {
                self.finish();
                Some(Ok(None))
            }
            gst::MessageView::Error(e) => {
                self.finish();
                Some(Err(Mp4DemuxerError::PipelineError {
                    src: e
                        .src()
                        .map(|s| s.path_string().to_string())
                        .unwrap_or_else(|| "<unknown>".to_string()),
                    msg: e.error().to_string(),
                    debug: e.debug().unwrap_or_default().to_string(),
                }))
            }
            _ => None,
        }
    }
}

impl Drop for Mp4Demuxer {
    fn drop(&mut self) {
        self.finish();
    }
}

/// Dynamically insert a codec-specific parser (+ byte-stream capsfilter for
/// H.264/HEVC) between the qtdemux pad and the queue. Returns the parser's
/// sink pad so the caller can link qtdemux's src_pad to it.
fn build_parser_chain(
    pipeline: &gst::Pipeline,
    codec_name: Option<&str>,
    queue_sink_pad: &gst::Pad,
) -> Result<gst::Pad, Mp4DemuxerError> {
    let (factory, needs_byte_stream_caps) = match codec_name {
        Some("video/x-h264") => ("h264parse", true),
        Some("video/x-h265") => ("h265parse", true),
        Some("video/x-av1") => ("av1parse", false),
        Some("video/x-vp8") => ("vp8parse", false),
        Some("video/x-vp9") => ("vp9parse", false),
        Some("image/jpeg") => ("jpegparse", false),
        _ => return Err(Mp4DemuxerError::ElementCreation("no parser needed".into())),
    };

    let parser = gst::ElementFactory::make(factory)
        .name("demux-parser")
        .build()
        .map_err(|e| Mp4DemuxerError::ElementCreation(format!("{factory}: {e}")))?;

    if needs_byte_stream_caps {
        parser.set_property("config-interval", -1i32);
    }

    pipeline
        .add(&parser)
        .map_err(|e| Mp4DemuxerError::LinkError(format!("add parser: {e}")))?;

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
            .map_err(|e| Mp4DemuxerError::ElementCreation(format!("capsfilter: {e}")))?;

        pipeline
            .add(&capsfilter)
            .map_err(|e| Mp4DemuxerError::LinkError(format!("add capsfilter: {e}")))?;

        parser
            .link(&capsfilter)
            .map_err(|e| Mp4DemuxerError::LinkError(format!("parser->capsfilter: {e}")))?;

        let capsf_src = capsfilter
            .static_pad("src")
            .ok_or_else(|| Mp4DemuxerError::ElementCreation("capsfilter src pad".into()))?;
        capsf_src
            .link(queue_sink_pad)
            .map_err(|e| Mp4DemuxerError::LinkError(format!("capsfilter->queue: {e}")))?;

        capsfilter
            .sync_state_with_parent()
            .map_err(|_| Mp4DemuxerError::StateChangeFailed)?;
    } else {
        let parser_src = parser
            .static_pad("src")
            .ok_or_else(|| Mp4DemuxerError::ElementCreation("parser src pad".into()))?;
        parser_src
            .link(queue_sink_pad)
            .map_err(|e| Mp4DemuxerError::LinkError(format!("parser->queue: {e}")))?;
    }

    parser
        .sync_state_with_parent()
        .map_err(|_| Mp4DemuxerError::StateChangeFailed)?;

    parser
        .static_pad("sink")
        .ok_or_else(|| Mp4DemuxerError::ElementCreation("parser sink pad missing".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mp4_muxer::Mp4Muxer;
    use std::str::FromStr;

    #[test]
    fn test_codec_from_caps_mapping() {
        let _ = gst::init();
        let h264 = gst::Caps::from_str("video/x-h264").unwrap();
        let hevc = gst::Caps::from_str("video/x-h265").unwrap();
        let av1 = gst::Caps::from_str("video/x-av1").unwrap();
        let vp8 = gst::Caps::from_str("video/x-vp8").unwrap();
        let vp9 = gst::Caps::from_str("video/x-vp9").unwrap();
        let jpeg = gst::Caps::from_str("image/jpeg").unwrap();

        assert_eq!(Mp4Demuxer::codec_from_caps(&h264), Some(Codec::H264));
        assert_eq!(Mp4Demuxer::codec_from_caps(&hevc), Some(Codec::Hevc));
        assert_eq!(Mp4Demuxer::codec_from_caps(&av1), Some(Codec::Av1));
        assert_eq!(Mp4Demuxer::codec_from_caps(&vp8), Some(Codec::Vp8));
        assert_eq!(Mp4Demuxer::codec_from_caps(&vp9), Some(Codec::Vp9));
        assert_eq!(Mp4Demuxer::codec_from_caps(&jpeg), Some(Codec::Jpeg));
    }

    #[test]
    fn test_demuxer_rejects_missing_file() {
        let result = Mp4Demuxer::new("/tmp/does-not-exist-savant-rs.mp4");
        assert!(matches!(result, Err(Mp4DemuxerError::InputNotFound(_))));
    }

    #[test]
    fn test_demuxer_reports_corrupt_mp4() {
        let _ = gst::init();
        let path = "/tmp/test_demuxer_corrupt.mp4";
        let _ = std::fs::remove_file(path);

        let mut muxer = Mp4Muxer::new(Codec::H264, path, 30, 1).unwrap();
        muxer.finish().unwrap();

        let mut demuxer = Mp4Demuxer::new(path).unwrap();
        let err = demuxer.pull_timeout(Duration::from_secs(2)).unwrap_err();
        assert!(matches!(err, Mp4DemuxerError::PipelineError { .. }));
        let _ = std::fs::remove_file(path);
    }
}
