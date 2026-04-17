//! Callback-based GStreamer MP4 demuxer: `filesrc -> qtdemux -> queue -> appsink`.
//!
//! Reads encoded packets from an MP4 (QuickTime) container and delivers them
//! as elementary stream payloads with timestamps through a user-supplied
//! callback.
//!
//! All output (packets, EOS, errors) is delivered through [`Mp4DemuxerOutput`]
//! via the callback provided at construction.  Callbacks fire on GStreamer's
//! streaming thread.
//!
//! # Threading
//!
//! The callback is invoked from GStreamer's internal streaming thread.
//! **Do not** call [`Mp4Demuxer::finish`] from within the callback — this
//! would deadlock because `finish()` waits for the streaming thread to stop.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use thiserror::Error;

use crate::codec::Codec;

/// A single demuxed elementary stream packet.
#[derive(Debug, Clone)]
pub struct DemuxedPacket {
    pub data: Vec<u8>,
    pub pts_ns: u64,
    pub dts_ns: Option<u64>,
    pub duration_ns: Option<u64>,
    pub is_keyframe: bool,
}

/// Errors that can occur during demuxing.
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
    #[error("Pipeline error from {src}: {msg} ({debug})")]
    PipelineError {
        src: String,
        msg: String,
        debug: String,
    },
    #[error("Demuxer already finished")]
    AlreadyFinished,
}

/// Callback payload delivered by [`Mp4Demuxer`].
#[derive(Debug)]
pub enum Mp4DemuxerOutput {
    /// A demuxed packet from the container.
    Packet(DemuxedPacket),
    /// End of stream — all packets have been delivered.
    Eos,
    /// An error occurred in the pipeline.
    Error(Mp4DemuxerError),
}

/// Callback-based GStreamer MP4 demuxer.
///
/// Reads encoded packets from an MP4 (QuickTime) container and delivers
/// them through the `on_output` callback provided at construction.
///
/// # Threading
///
/// The callback fires on GStreamer's internal streaming thread.  Do **not**
/// call [`finish`](Self::finish) from within the callback.
pub struct Mp4Demuxer {
    pipeline: gst::Pipeline,
    finished: Arc<AtomicBool>,
    detected_codec: Arc<Mutex<Option<Codec>>>,
    done_pair: Arc<(Mutex<bool>, Condvar)>,
}

impl Mp4Demuxer {
    /// Create a demuxer that outputs raw container-format packets (e.g. AVC
    /// length-prefixed NALUs for H.264 in MP4).
    ///
    /// The pipeline starts immediately; packets are delivered through
    /// `on_output`.
    pub fn new<F>(input_path: &str, on_output: F) -> Result<Self, Mp4DemuxerError>
    where
        F: Fn(Mp4DemuxerOutput) + Send + Sync + 'static,
    {
        Self::new_inner(input_path, false, on_output)
    }

    /// Create a demuxer that inserts codec-specific parsers to convert output
    /// to byte-stream / Annex-B format.
    ///
    /// Use this when feeding packets to a standalone decoder (`NvDecoder`)
    /// that expects byte-stream input.
    pub fn new_parsed<F>(input_path: &str, on_output: F) -> Result<Self, Mp4DemuxerError>
    where
        F: Fn(Mp4DemuxerOutput) + Send + Sync + 'static,
    {
        Self::new_inner(input_path, true, on_output)
    }

    fn new_inner<F>(input_path: &str, parsed: bool, on_output: F) -> Result<Self, Mp4DemuxerError>
    where
        F: Fn(Mp4DemuxerOutput) + Send + Sync + 'static,
    {
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

        // Dynamic pad linking (qtdemux -> parser? -> queue)
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

        // Shared state
        let on_output = Arc::new(on_output);
        let detected_codec: Arc<Mutex<Option<Codec>>> = Arc::new(Mutex::new(None));
        let finished = Arc::new(AtomicBool::new(false));
        let done_pair = Arc::new((Mutex::new(false), Condvar::new()));

        // Appsink callbacks (fire on GStreamer streaming thread)
        {
            let on_output_sample = on_output.clone();
            let detected_codec = detected_codec.clone();
            let finished_sample = finished.clone();
            let done_pair_sample = done_pair.clone();

            let on_output_eos = on_output.clone();
            let finished_eos = finished.clone();
            let done_pair_eos = done_pair.clone();

            appsink.set_callbacks(
                gst_app::AppSinkCallbacks::builder()
                    .new_sample(move |sink| {
                        if finished_sample.load(Ordering::SeqCst) {
                            return Err(gst::FlowError::Eos);
                        }
                        let sample = sink.pull_sample().map_err(|_| gst::FlowError::Eos)?;
                        match sample_to_packet(sample, &detected_codec) {
                            Ok(pkt) => {
                                on_output_sample(Mp4DemuxerOutput::Packet(pkt));
                                Ok(gst::FlowSuccess::Ok)
                            }
                            Err(e) => {
                                if !finished_sample.swap(true, Ordering::SeqCst) {
                                    on_output_sample(Mp4DemuxerOutput::Error(e));
                                    signal_done(&done_pair_sample);
                                }
                                Err(gst::FlowError::Error)
                            }
                        }
                    })
                    .eos(move |_| {
                        if !finished_eos.swap(true, Ordering::SeqCst) {
                            on_output_eos(Mp4DemuxerOutput::Eos);
                            signal_done(&done_pair_eos);
                        }
                    })
                    .build(),
            );
        }

        // Bus sync handler for pipeline errors
        if let Some(bus) = pipeline.bus() {
            let on_output_err = on_output.clone();
            let finished_bus = finished.clone();
            let done_pair_bus = done_pair.clone();

            bus.set_sync_handler(move |_, msg| {
                if let gst::MessageView::Error(e) = msg.view() {
                    if !finished_bus.swap(true, Ordering::SeqCst) {
                        on_output_err(Mp4DemuxerOutput::Error(Mp4DemuxerError::PipelineError {
                            src: e
                                .src()
                                .map(|s| s.path_string().to_string())
                                .unwrap_or_else(|| "<unknown>".to_string()),
                            msg: e.error().to_string(),
                            debug: e.debug().unwrap_or_default().to_string(),
                        }));
                        signal_done(&done_pair_bus);
                    }
                    return gst::BusSyncReply::Drop;
                }
                gst::BusSyncReply::Pass
            });
        }

        // Start pipeline
        let ret = pipeline.set_state(gst::State::Playing);
        if ret == Err(gst::StateChangeError) {
            return Err(Mp4DemuxerError::StateChangeFailed);
        }

        Ok(Self {
            pipeline,
            finished,
            detected_codec,
            done_pair,
        })
    }

    /// Block until the demuxer reaches EOS, encounters an error, or
    /// [`finish`](Self::finish) is called.
    pub fn wait(&self) {
        let (lock, cvar) = &*self.done_pair;
        let _guard = cvar
            .wait_while(lock.lock().unwrap(), |done| !*done)
            .unwrap();
    }

    /// Block until the demuxer finishes or the timeout expires.
    ///
    /// Returns `true` if the demuxer finished, `false` on timeout.
    pub fn wait_timeout(&self, timeout: Duration) -> bool {
        let (lock, cvar) = &*self.done_pair;
        let (guard, _) = cvar
            .wait_timeout_while(lock.lock().unwrap(), timeout, |done| !*done)
            .unwrap();
        *guard
    }

    /// Auto-detected video codec from the container, or `None` if no sample
    /// has been processed yet.
    pub fn detected_codec(&self) -> Option<Codec> {
        *self.detected_codec.lock().unwrap()
    }

    /// Stop the pipeline and release resources.
    ///
    /// Safe to call multiple times.  After this call, no more callbacks will
    /// fire.
    ///
    /// # Panics
    ///
    /// Must **not** be called from within the `on_output` callback (deadlock).
    pub fn finish(&mut self) {
        let was_finished = self.finished.swap(true, Ordering::SeqCst);
        let _ = self.pipeline.set_state(gst::State::Null);
        if !was_finished {
            signal_done(&self.done_pair);
        }
    }

    /// Whether the demuxer has been finalized (EOS, error, or explicit
    /// `finish()`).
    pub fn is_finished(&self) -> bool {
        self.finished.load(Ordering::SeqCst)
    }

    /// Convenience: demux all packets from an MP4 file (raw container-format).
    ///
    /// Returns `(packets, detected_codec)`.
    pub fn demux_all(
        input_path: &str,
    ) -> Result<(Vec<DemuxedPacket>, Option<Codec>), Mp4DemuxerError> {
        Self::demux_all_inner(input_path, false)
    }

    /// Convenience: demux all packets from an MP4 file (parsed/byte-stream).
    ///
    /// Returns `(packets, detected_codec)`.
    pub fn demux_all_parsed(
        input_path: &str,
    ) -> Result<(Vec<DemuxedPacket>, Option<Codec>), Mp4DemuxerError> {
        Self::demux_all_inner(input_path, true)
    }

    fn demux_all_inner(
        input_path: &str,
        parsed: bool,
    ) -> Result<(Vec<DemuxedPacket>, Option<Codec>), Mp4DemuxerError> {
        let packets: Arc<Mutex<Vec<DemuxedPacket>>> = Arc::new(Mutex::new(Vec::new()));
        let error: Arc<Mutex<Option<Mp4DemuxerError>>> = Arc::new(Mutex::new(None));
        let p = packets.clone();
        let e = error.clone();

        let demuxer = Self::new_inner(input_path, parsed, move |output| match output {
            Mp4DemuxerOutput::Packet(pkt) => p.lock().unwrap().push(pkt),
            Mp4DemuxerOutput::Eos => {}
            Mp4DemuxerOutput::Error(err) => *e.lock().unwrap() = Some(err),
        })?;

        demuxer.wait();

        if let Some(e) = error.lock().unwrap().take() {
            return Err(e);
        }

        let codec = demuxer.detected_codec();
        let pkts = std::mem::take(&mut *packets.lock().unwrap());
        Ok((pkts, codec))
    }
}

impl Drop for Mp4Demuxer {
    fn drop(&mut self) {
        self.finish();
    }
}

/// Notify the condvar that the demuxer is done.
fn signal_done(done_pair: &(Mutex<bool>, Condvar)) {
    let (lock, cvar) = done_pair;
    *lock.lock().unwrap() = true;
    cvar.notify_all();
}

/// Convert a GStreamer sample to a [`DemuxedPacket`], updating
/// `detected_codec` on the first sample with caps.
fn sample_to_packet(
    sample: gst::Sample,
    detected_codec: &Mutex<Option<Codec>>,
) -> Result<DemuxedPacket, Mp4DemuxerError> {
    {
        let mut codec = detected_codec.lock().unwrap();
        if codec.is_none() {
            *codec = sample.caps().and_then(codec_from_caps);
        }
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

/// Map GStreamer caps to a [`Codec`] value.
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

/// Dynamically insert a codec-specific parser (+ byte-stream capsfilter for
/// H.264/HEVC) between the qtdemux pad and the queue.  Returns the parser's
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

        assert_eq!(codec_from_caps(&h264), Some(Codec::H264));
        assert_eq!(codec_from_caps(&hevc), Some(Codec::Hevc));
        assert_eq!(codec_from_caps(&av1), Some(Codec::Av1));
        assert_eq!(codec_from_caps(&vp8), Some(Codec::Vp8));
        assert_eq!(codec_from_caps(&vp9), Some(Codec::Vp9));
        assert_eq!(codec_from_caps(&jpeg), Some(Codec::Jpeg));
    }

    #[test]
    fn test_demuxer_rejects_missing_file() {
        let result = Mp4Demuxer::new("/tmp/does-not-exist-savant-rs.mp4", |_| {});
        assert!(matches!(result, Err(Mp4DemuxerError::InputNotFound(_))));
    }

    #[test]
    fn test_demuxer_reports_corrupt_mp4() {
        let _ = gst::init();
        let path = "/tmp/test_demuxer_corrupt.mp4";
        let _ = std::fs::remove_file(path);

        let mut muxer = Mp4Muxer::new(Codec::H264, path, 30, 1).unwrap();
        muxer.finish().unwrap();

        let err = Mp4Demuxer::demux_all(path).unwrap_err();
        assert!(matches!(err, Mp4DemuxerError::PipelineError { .. }));
        let _ = std::fs::remove_file(path);
    }
}
