//! Callback-based GStreamer MP4 demuxer: `filesrc -> qtdemux -> queue -> appsink`.
//!
//! Reads encoded packets from an MP4 (QuickTime) container and delivers them
//! as elementary stream payloads with timestamps through a user-supplied
//! callback.
//!
//! What it delivers through [`Mp4DemuxerOutput`]:
//! - [`VideoInfo`] (once, before first packet, when caps are known),
//! - encoded [`DemuxedPacket`] payloads,
//! - EOS / errors.
//!
//! Callbacks fire on GStreamer's streaming thread.
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

use savant_core::primitives::video_codec::VideoCodec;

use crate::demux::helpers::{
    build_parser_chain, sample_to_packet, signal_done, signal_info_done, ParserChainError,
    SampleError,
};
pub use crate::demux::{DemuxedPacket, VideoInfo};

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
    /// Fires exactly once, before the first `Packet`. May be absent if the
    /// pipeline errors before caps are known.
    StreamInfo(VideoInfo),
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
/// Use [`video_info`](Self::video_info) / [`wait_for_video_info`](Self::wait_for_video_info)
/// to read stream metadata discovered from caps.
///
/// # Threading
///
/// The callback fires on GStreamer's internal streaming thread.  Do **not**
/// call [`finish`](Self::finish) from within the callback.
pub struct Mp4Demuxer {
    pipeline: gst::Pipeline,
    finished: Arc<AtomicBool>,
    detected_codec: Arc<Mutex<Option<VideoCodec>>>,
    video_info: Arc<Mutex<Option<VideoInfo>>>,
    info_pair: Arc<(Mutex<bool>, Condvar)>,
    stream_info_fired: Arc<AtomicBool>,
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

        // Shared state
        let on_output = Arc::new(on_output);
        let detected_codec: Arc<Mutex<Option<VideoCodec>>> = Arc::new(Mutex::new(None));
        let video_info: Arc<Mutex<Option<VideoInfo>>> = Arc::new(Mutex::new(None));
        let stream_info_fired = Arc::new(AtomicBool::new(false));
        let finished = Arc::new(AtomicBool::new(false));
        let info_pair = Arc::new((Mutex::new(false), Condvar::new()));
        let done_pair = Arc::new((Mutex::new(false), Condvar::new()));

        // Adapter: the shared helpers emit only `VideoInfo`; the mp4 demuxer wraps
        // that into `Mp4DemuxerOutput::StreamInfo`.
        let emit_stream_info: Arc<dyn Fn(VideoInfo) + Send + Sync> = {
            let on_output = on_output.clone();
            Arc::new(move |info: VideoInfo| on_output(Mp4DemuxerOutput::StreamInfo(info)))
        };

        // Dynamic pad linking (qtdemux -> parser? -> queue)
        let linked_video_pad = Arc::new(AtomicBool::new(false));
        let linked_video_pad_closure = linked_video_pad.clone();
        let pipeline_for_pad = pipeline.clone();
        let detected_codec_pad = detected_codec.clone();
        let video_info_pad = video_info.clone();
        let stream_info_fired_pad = stream_info_fired.clone();
        let info_pair_pad = info_pair.clone();
        let emit_stream_info_pad = emit_stream_info.clone();
        demux.connect_pad_added(move |_demux, src_pad| {
            if linked_video_pad_closure.load(Ordering::SeqCst) {
                return;
            }

            let caps = src_pad
                .current_caps()
                .or_else(|| Some(src_pad.query_caps(None)));
            let codec_name = caps
                .as_ref()
                .and_then(|caps| caps.structure(0).map(|s| s.name().to_string()));

            let is_video = codec_name
                .as_deref()
                .is_some_and(|n| n.starts_with("video/") || n.starts_with("image/"));
            if !is_video {
                return;
            }

            crate::demux::helpers::maybe_emit_stream_info_from_caps(
                caps.as_ref().map(|c| c.as_ref()),
                &detected_codec_pad,
                &video_info_pad,
                &stream_info_fired_pad,
                &info_pair_pad,
                emit_stream_info_pad.as_ref(),
            );

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

        // Appsink callbacks (fire on GStreamer streaming thread)
        {
            let on_output_sample = on_output.clone();
            let detected_codec = detected_codec.clone();
            let video_info_sample = video_info.clone();
            let stream_info_fired_sample = stream_info_fired.clone();
            let finished_sample = finished.clone();
            let info_pair_sample = info_pair.clone();
            let done_pair_sample = done_pair.clone();
            let emit_stream_info_sample = emit_stream_info.clone();

            let on_output_eos = on_output.clone();
            let finished_eos = finished.clone();
            let info_pair_eos = info_pair.clone();
            let done_pair_eos = done_pair.clone();

            appsink.set_callbacks(
                gst_app::AppSinkCallbacks::builder()
                    .new_sample(move |sink| {
                        if finished_sample.load(Ordering::SeqCst) {
                            return Err(gst::FlowError::Eos);
                        }
                        let sample = sink.pull_sample().map_err(|_| gst::FlowError::Eos)?;
                        match sample_to_packet(
                            sample,
                            &detected_codec,
                            &video_info_sample,
                            &stream_info_fired_sample,
                            &info_pair_sample,
                            emit_stream_info_sample.as_ref(),
                        ) {
                            Ok(pkt) => {
                                on_output_sample(Mp4DemuxerOutput::Packet(pkt));
                                Ok(gst::FlowSuccess::Ok)
                            }
                            Err(e) => {
                                if !finished_sample.swap(true, Ordering::SeqCst) {
                                    on_output_sample(Mp4DemuxerOutput::Error(
                                        Mp4DemuxerError::from(e),
                                    ));
                                    signal_info_done(&info_pair_sample);
                                    signal_done(&done_pair_sample);
                                }
                                Err(gst::FlowError::Error)
                            }
                        }
                    })
                    .eos(move |_| {
                        if !finished_eos.swap(true, Ordering::SeqCst) {
                            on_output_eos(Mp4DemuxerOutput::Eos);
                            signal_info_done(&info_pair_eos);
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
            let info_pair_bus = info_pair.clone();
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
                        signal_info_done(&info_pair_bus);
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
            video_info,
            info_pair,
            stream_info_fired,
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
    pub fn detected_codec(&self) -> Option<VideoCodec> {
        *self.detected_codec.lock().unwrap()
    }

    /// Returns `VideoInfo` if it has already been observed on the source pad
    /// caps. Non-blocking.
    ///
    /// ```no_run
    /// use savant_gstreamer::mp4_demuxer::Mp4Demuxer;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let (_packets, info) = Mp4Demuxer::demux_all_parsed("/tmp/input.mp4")?;
    /// if let Some(info) = info {
    ///     println!("codec={:?}, {}x{}", info.codec, info.width, info.height);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn video_info(&self) -> Option<VideoInfo> {
        *self.video_info.lock().unwrap()
    }

    /// Block until `VideoInfo` is known, the pipeline terminates, or the
    /// timeout expires. Returns the info or `None` on timeout / early
    /// termination without caps.
    pub fn wait_for_video_info(&self, timeout: Duration) -> Option<VideoInfo> {
        if self.stream_info_fired.load(Ordering::SeqCst) {
            return self.video_info();
        }
        let (lock, cvar) = &*self.info_pair;
        let (guard, _) = cvar
            .wait_timeout_while(lock.lock().unwrap(), timeout, |known_or_done| {
                !*known_or_done
            })
            .unwrap();
        if !*guard {
            return None;
        }
        drop(guard);
        self.video_info()
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
            signal_info_done(&self.info_pair);
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
    /// Breaking change: returns `(packets, video_info)` instead of codec-only metadata.
    pub fn demux_all(
        input_path: &str,
    ) -> Result<(Vec<DemuxedPacket>, Option<VideoInfo>), Mp4DemuxerError> {
        Self::demux_all_inner(input_path, false)
    }

    /// Convenience: demux all packets from an MP4 file (parsed/byte-stream).
    ///
    /// Breaking change: returns `(packets, video_info)` instead of codec-only metadata.
    pub fn demux_all_parsed(
        input_path: &str,
    ) -> Result<(Vec<DemuxedPacket>, Option<VideoInfo>), Mp4DemuxerError> {
        Self::demux_all_inner(input_path, true)
    }

    fn demux_all_inner(
        input_path: &str,
        parsed: bool,
    ) -> Result<(Vec<DemuxedPacket>, Option<VideoInfo>), Mp4DemuxerError> {
        let packets: Arc<Mutex<Vec<DemuxedPacket>>> = Arc::new(Mutex::new(Vec::new()));
        let error: Arc<Mutex<Option<Mp4DemuxerError>>> = Arc::new(Mutex::new(None));
        let p = packets.clone();
        let e = error.clone();

        let demuxer = Self::new_inner(input_path, parsed, move |output| match output {
            Mp4DemuxerOutput::StreamInfo(_) => {}
            Mp4DemuxerOutput::Packet(pkt) => p.lock().unwrap().push(pkt),
            Mp4DemuxerOutput::Eos => {}
            Mp4DemuxerOutput::Error(err) => *e.lock().unwrap() = Some(err),
        })?;

        demuxer.wait();

        if let Some(e) = error.lock().unwrap().take() {
            return Err(e);
        }

        let info = demuxer.video_info();
        let pkts = std::mem::take(&mut *packets.lock().unwrap());
        Ok((pkts, info))
    }
}

impl Drop for Mp4Demuxer {
    fn drop(&mut self) {
        self.finish();
    }
}

impl From<SampleError> for Mp4DemuxerError {
    fn from(e: SampleError) -> Self {
        Mp4DemuxerError::PipelineError {
            src: e.src,
            msg: e.msg,
            debug: e.debug,
        }
    }
}

impl From<ParserChainError> for Mp4DemuxerError {
    fn from(e: ParserChainError) -> Self {
        Mp4DemuxerError::ElementCreation(e.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mp4_muxer::Mp4Muxer;
    use std::str::FromStr;
    use std::sync::Mutex as StdMutex;

    const H264_SPS_PPS_IDR: [u8; 32] = [
        0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x0A, 0xE9, 0x40, 0x40, 0x04, 0x00, 0x00, 0x00,
        0x02, 0x00, 0x00, 0x00, 0x01, 0x68, 0xCE, 0x38, 0x80, 0x00, 0x00, 0x00, 0x01, 0x65, 0x88,
        0x80, 0x40,
    ];

    fn make_h264_mp4(path: &str, num_frames: usize) {
        let mut muxer = Mp4Muxer::new(VideoCodec::H264, path, 30, 1).unwrap();
        let duration_ns = 33_333_333u64;
        for i in 0..num_frames {
            muxer
                .push(
                    &H264_SPS_PPS_IDR,
                    (i as u64) * duration_ns,
                    None,
                    Some(duration_ns),
                )
                .unwrap();
        }
        muxer.finish().unwrap();
    }

    #[test]
    fn test_codec_from_caps_mapping() {
        use crate::demux::helpers::codec_from_caps;
        let _ = gst::init();
        let h264 = gst::Caps::from_str("video/x-h264").unwrap();
        let hevc = gst::Caps::from_str("video/x-h265").unwrap();
        let av1 = gst::Caps::from_str("video/x-av1").unwrap();
        let vp8 = gst::Caps::from_str("video/x-vp8").unwrap();
        let vp9 = gst::Caps::from_str("video/x-vp9").unwrap();
        let jpeg = gst::Caps::from_str("image/jpeg").unwrap();

        assert_eq!(codec_from_caps(&h264), Some(VideoCodec::H264));
        assert_eq!(codec_from_caps(&hevc), Some(VideoCodec::Hevc));
        assert_eq!(codec_from_caps(&av1), Some(VideoCodec::Av1));
        assert_eq!(codec_from_caps(&vp8), Some(VideoCodec::Vp8));
        assert_eq!(codec_from_caps(&vp9), Some(VideoCodec::Vp9));
        assert_eq!(codec_from_caps(&jpeg), Some(VideoCodec::Jpeg));
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

        let mut muxer = Mp4Muxer::new(VideoCodec::H264, path, 30, 1).unwrap();
        muxer.finish().unwrap();

        let err = Mp4Demuxer::demux_all(path).unwrap_err();
        assert!(matches!(err, Mp4DemuxerError::PipelineError { .. }));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_video_info_available_after_wait() {
        let _ = gst::init();
        let path = "/tmp/test_demuxer_video_info_after_wait.mp4";
        let _ = std::fs::remove_file(path);

        make_h264_mp4(path, 3);
        let (_packets, info) = Mp4Demuxer::demux_all_parsed(path).unwrap();
        let info = info.expect("video info expected");
        assert_eq!(info.codec, VideoCodec::H264);
        assert!(info.width > 0);
        assert!(info.height > 0);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_stream_info_callback_fires_before_first_packet() {
        let _ = gst::init();
        let path = "/tmp/test_demuxer_stream_info_order.mp4";
        let _ = std::fs::remove_file(path);
        make_h264_mp4(path, 3);

        let labels: Arc<StdMutex<Vec<&'static str>>> = Arc::new(StdMutex::new(Vec::new()));
        let labels_cb = labels.clone();
        let demuxer = Mp4Demuxer::new(path, move |output| {
            let mut out = labels_cb.lock().unwrap();
            match output {
                Mp4DemuxerOutput::StreamInfo(_) => out.push("stream_info"),
                Mp4DemuxerOutput::Packet(_) => out.push("packet"),
                Mp4DemuxerOutput::Eos => out.push("eos"),
                Mp4DemuxerOutput::Error(_) => out.push("error"),
            }
        })
        .unwrap();

        demuxer.wait();
        let out = labels.lock().unwrap();
        assert!(!out.is_empty());
        assert_eq!(out[0], "stream_info");
        assert_eq!(
            out.iter().filter(|label| **label == "stream_info").count(),
            1
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_wait_for_video_info_timeout_on_corrupt() {
        let _ = gst::init();
        let path = "/tmp/test_demuxer_video_info_corrupt.mp4";
        let _ = std::fs::remove_file(path);

        let mut muxer = Mp4Muxer::new(VideoCodec::H264, path, 30, 1).unwrap();
        muxer.finish().unwrap();

        let demuxer = Mp4Demuxer::new(path, |_| {}).unwrap();
        let info = demuxer.wait_for_video_info(Duration::from_secs(2));
        assert!(info.is_none());

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_video_info_getter_returns_none_before_caps_and_some_after() {
        let _ = gst::init();
        let path = "/tmp/test_demuxer_video_info_getter.mp4";
        let _ = std::fs::remove_file(path);
        make_h264_mp4(path, 3);

        let demuxer = Mp4Demuxer::new(path, |_| {}).unwrap();
        assert!(demuxer.video_info().is_none());
        let waited = demuxer.wait_for_video_info(Duration::from_secs(5));
        assert!(waited.is_some());
        assert_eq!(demuxer.video_info(), waited);
        demuxer.wait();

        let _ = std::fs::remove_file(path);
    }
}
