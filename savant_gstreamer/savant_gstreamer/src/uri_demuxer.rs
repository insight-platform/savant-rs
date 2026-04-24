//! Callback-based GStreamer URI demuxer: `urisourcebin -> parsebin -> [capsfilter] -> queue -> appsink`.
//!
//! Reads encoded packets from any GStreamer-supported URI (`file://`, `http(s)://`,
//! `rtsp://`, HLS, DASH, MKV, TS, MP4 ...) and delivers them as elementary-stream
//! payloads with timestamps through a user-supplied callback. Does **not**
//! decode.
//!
//! API-compatible with [`crate::mp4_demuxer::Mp4Demuxer`] — it emits the same
//! [`DemuxedPacket`] / [`VideoInfo`] types and exposes the same wait / finish
//! surface.
//!
//! # Pipeline
//!
//! ```text
//! urisourcebin (uri + bin_properties)
//!   -- source-setup --> inner src element (apply source_properties)
//!   -- pad-added    --> parsebin
//!                         -- pad-added --> [byte-stream capsfilter if parsed=true]
//!                                            --> queue --> appsink
//! ```
//!
//! # Threading
//!
//! The callback is invoked from GStreamer's streaming thread. Do **not** call
//! [`UriDemuxer::finish`] from within the callback — that would deadlock.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use gstreamer as gst;
use gstreamer::glib;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use thiserror::Error;

use savant_core::primitives::video_codec::VideoCodec;

use crate::demux::helpers::{
    sample_to_packet, signal_done, signal_info_done, ParserChainError, SampleError,
};
pub use crate::demux::{DemuxedPacket, VideoInfo};

/// Errors that can occur while constructing or running a [`UriDemuxer`].
#[derive(Debug, Error)]
pub enum UriDemuxerError {
    #[error("URI is missing or empty")]
    MissingUri,
    #[error("Invalid URI: {0}")]
    InvalidUri(String),
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
    #[error("Failed to set property '{name}' on '{element}': {error}")]
    PropertyError {
        element: String,
        name: String,
        error: String,
    },
    #[error("Demuxer already finished")]
    AlreadyFinished,
}

impl From<SampleError> for UriDemuxerError {
    fn from(e: SampleError) -> Self {
        UriDemuxerError::PipelineError {
            src: e.src,
            msg: e.msg,
            debug: e.debug,
        }
    }
}

impl From<ParserChainError> for UriDemuxerError {
    fn from(e: ParserChainError) -> Self {
        UriDemuxerError::ElementCreation(e.0)
    }
}

/// Insert a byte-stream (Annex-B) capsfilter between parsebin's H.264/HEVC
/// output and the queue. Returns `Ok(Some(capsfilter_sink_pad))` if the
/// capsfilter was inserted, `Ok(None)` when the codec does not need one
/// (passthrough to the queue), or `Err` on element creation/link failure.
fn insert_byte_stream_capsfilter(
    pipeline: &gst::Pipeline,
    codec_name: Option<&str>,
    queue_sink_pad: &gst::Pad,
) -> Result<Option<gst::Pad>, ParserChainError> {
    let media = match codec_name {
        Some("video/x-h264") => "video/x-h264",
        Some("video/x-h265") => "video/x-h265",
        _ => return Ok(None),
    };

    let caps = gst::Caps::builder(media)
        .field("stream-format", "byte-stream")
        .build();
    let capsfilter = gst::ElementFactory::make("capsfilter")
        .name("uri-byte-stream-capsf")
        .property("caps", &caps)
        .build()
        .map_err(|e| ParserChainError(format!("capsfilter: {e}")))?;

    pipeline
        .add(&capsfilter)
        .map_err(|e| ParserChainError(format!("add capsfilter: {e}")))?;

    let capsf_src = capsfilter
        .static_pad("src")
        .ok_or_else(|| ParserChainError("capsfilter src pad".into()))?;
    capsf_src
        .link(queue_sink_pad)
        .map_err(|e| ParserChainError(format!("capsfilter->queue: {e}")))?;

    capsfilter
        .sync_state_with_parent()
        .map_err(|_| ParserChainError("StateChangeFailed".into()))?;

    let capsf_sink = capsfilter
        .static_pad("sink")
        .ok_or_else(|| ParserChainError("capsfilter sink pad missing".into()))?;
    Ok(Some(capsf_sink))
}

/// Very cheap syntactic URI check: requires a leading scheme and `://`.
///
/// This only rejects obviously malformed values up front; GStreamer does the
/// real validation at state-change time.
fn is_plausible_uri(uri: &str) -> bool {
    let trimmed = uri.trim();
    let Some(scheme_end) = trimmed.find("://") else {
        return false;
    };
    if scheme_end == 0 {
        return false;
    }
    let scheme = &trimmed[..scheme_end];
    scheme
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.')
        && scheme
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic())
}

/// Callback payload delivered by [`UriDemuxer`].
#[derive(Debug)]
pub enum UriDemuxerOutput {
    /// Fires exactly once, before the first `Packet`. May be absent if the
    /// pipeline errors before caps are known.
    StreamInfo(VideoInfo),
    /// A demuxed packet from the container.
    Packet(DemuxedPacket),
    /// End of stream — all packets have been delivered.
    Eos,
    /// An error occurred in the pipeline.
    Error(UriDemuxerError),
}

/// A scalar property value accepted by [`UriDemuxerConfig`].
///
/// GStreamer's GObject type system will auto-convert `I64` / `U64` / `F64`
/// to the target property's concrete type (`u32`, `i32`, etc.). Use [`Bytes`]
/// for binary blobs.
#[derive(Debug, Clone)]
pub enum PropertyValue {
    Bool(bool),
    I64(i64),
    U64(u64),
    F64(f64),
    String(String),
    Bytes(Vec<u8>),
}

impl PropertyValue {
    fn to_glib_value(&self) -> glib::Value {
        match self {
            PropertyValue::Bool(v) => v.to_value(),
            PropertyValue::I64(v) => v.to_value(),
            PropertyValue::U64(v) => v.to_value(),
            PropertyValue::F64(v) => v.to_value(),
            PropertyValue::String(v) => v.to_value(),
            PropertyValue::Bytes(v) => glib::Bytes::from(v.as_slice()).to_value(),
        }
    }
}

/// Apply a [`PropertyValue`] to a GObject property, coercing the numeric
/// GType to match the target property's declared type (`gint`, `guint`,
/// `gint64`, `guint64`, `gfloat`, `gdouble`, ...).
///
/// GStreamer's `set_property_from_value` requires an exact GType match, so
/// a Python `int` wrapped as `gint64` can't be assigned to a `gint`
/// property (e.g. `urisourcebin::buffer-size`). We query the target's
/// `ParamSpec` and use `Value::transform_with_type` to bridge scalar
/// integer / float types. Non-scalar types (strings, boxed bytes) pass
/// through unchanged.
fn apply_property(obj: &glib::Object, name: &str, value: &PropertyValue) -> Result<(), String> {
    let gvalue = value.to_glib_value();
    let target_type = match obj.find_property(name) {
        Some(pspec) => pspec.value_type(),
        None => return Err("no such property".into()),
    };

    // Exact match: no transform needed.
    let coerced = if gvalue.type_() == target_type {
        gvalue
    } else {
        match gvalue.transform_with_type(target_type) {
            Ok(v) => v,
            Err(_) => {
                return Err(format!(
                    "type mismatch: cannot convert {} to {}",
                    gvalue.type_().name(),
                    target_type.name()
                ));
            }
        }
    };

    let obj_clone = obj.clone();
    let name_owned = name.to_string();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        obj_clone.set_property_from_value(&name_owned, &coerced);
    }));
    if result.is_err() {
        return Err("set_property_from_value panicked".into());
    }
    Ok(())
}

impl From<bool> for PropertyValue {
    fn from(v: bool) -> Self {
        PropertyValue::Bool(v)
    }
}
impl From<i32> for PropertyValue {
    fn from(v: i32) -> Self {
        PropertyValue::I64(v as i64)
    }
}
impl From<i64> for PropertyValue {
    fn from(v: i64) -> Self {
        PropertyValue::I64(v)
    }
}
impl From<u32> for PropertyValue {
    fn from(v: u32) -> Self {
        PropertyValue::U64(v as u64)
    }
}
impl From<u64> for PropertyValue {
    fn from(v: u64) -> Self {
        PropertyValue::U64(v)
    }
}
impl From<f32> for PropertyValue {
    fn from(v: f32) -> Self {
        PropertyValue::F64(v as f64)
    }
}
impl From<f64> for PropertyValue {
    fn from(v: f64) -> Self {
        PropertyValue::F64(v)
    }
}
impl From<&str> for PropertyValue {
    fn from(v: &str) -> Self {
        PropertyValue::String(v.to_string())
    }
}
impl From<String> for PropertyValue {
    fn from(v: String) -> Self {
        PropertyValue::String(v)
    }
}
impl From<Vec<u8>> for PropertyValue {
    fn from(v: Vec<u8>) -> Self {
        PropertyValue::Bytes(v)
    }
}

/// Configuration for a [`UriDemuxer`].
///
/// Only [`uri`](Self::uri) is mandatory. Everything else is optional:
///
/// - `parsed` defaults to `true` and inserts a byte-stream (Annex-B) capsfilter
///   after parsebin's H.264/HEVC pad. Set to `false` to pass parsebin output
///   through unchanged.
/// - `bin_properties` are applied to `urisourcebin` itself (e.g. `buffer-size`,
///   `use-buffering`, `connection-speed`).
/// - `source_properties` are applied to the dynamically-created inner source
///   element (e.g. `rtspsrc`'s `latency` / `user-id` / `user-pw` / `protocols`,
///   `souphttpsrc`'s `user-agent` / `extra-headers`). They are applied via the
///   `source-setup` signal; see GStreamer's urisourcebin documentation.
pub struct UriDemuxerConfig {
    pub uri: String,
    pub parsed: bool,
    pub bin_properties: Vec<(String, PropertyValue)>,
    pub source_properties: Vec<(String, PropertyValue)>,
}

impl UriDemuxerConfig {
    /// Create a new config for the given URI.
    pub fn new(uri: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            parsed: true,
            bin_properties: Vec::new(),
            source_properties: Vec::new(),
        }
    }

    /// Set the `parsed` flag.
    pub fn with_parsed(mut self, parsed: bool) -> Self {
        self.parsed = parsed;
        self
    }

    /// Set a property on the `urisourcebin` element.
    pub fn with_bin_property<V: Into<PropertyValue>>(mut self, name: &str, value: V) -> Self {
        self.bin_properties.push((name.to_string(), value.into()));
        self
    }

    /// Set a property on the inner source element (via `source-setup`).
    pub fn with_source_property<V: Into<PropertyValue>>(mut self, name: &str, value: V) -> Self {
        self.source_properties
            .push((name.to_string(), value.into()));
        self
    }
}

/// Callback-based GStreamer URI demuxer.
///
/// Reads encoded packets from any GStreamer-supported URI and delivers them
/// through the `on_output` callback provided at construction.
///
/// # Threading
///
/// The callback fires on GStreamer's internal streaming thread. Do **not**
/// call [`finish`](Self::finish) from within the callback.
pub struct UriDemuxer {
    pipeline: gst::Pipeline,
    finished: Arc<AtomicBool>,
    detected_codec: Arc<Mutex<Option<VideoCodec>>>,
    video_info: Arc<Mutex<Option<VideoInfo>>>,
    info_pair: Arc<(Mutex<bool>, Condvar)>,
    stream_info_fired: Arc<AtomicBool>,
    done_pair: Arc<(Mutex<bool>, Condvar)>,
}

impl UriDemuxer {
    /// Create a new demuxer from a [`UriDemuxerConfig`].
    ///
    /// The pipeline starts immediately; packets are delivered through
    /// `on_output` on GStreamer's streaming thread.
    pub fn new<F>(config: UriDemuxerConfig, on_output: F) -> Result<Self, UriDemuxerError>
    where
        F: Fn(UriDemuxerOutput) + Send + Sync + 'static,
    {
        let _ = gst::init();

        if config.uri.trim().is_empty() {
            return Err(UriDemuxerError::MissingUri);
        }

        // Validate the URI string: require a scheme separator. GStreamer itself
        // validates more strictly during state change, but we reject obviously
        // malformed values up front so the caller gets a typed error.
        if !is_plausible_uri(&config.uri) {
            return Err(UriDemuxerError::InvalidUri(config.uri.clone()));
        }

        let pipeline = gst::Pipeline::new();

        let urisrc = gst::ElementFactory::make("urisourcebin")
            .name("uri-src")
            .build()
            .map_err(|e| UriDemuxerError::ElementCreation(format!("urisourcebin: {e}")))?;
        urisrc.set_property("uri", &config.uri);

        // Apply user-supplied bin properties.
        for (name, value) in &config.bin_properties {
            if let Err(err) = apply_property(urisrc.upcast_ref::<glib::Object>(), name, value) {
                return Err(UriDemuxerError::PropertyError {
                    element: "urisourcebin".into(),
                    name: name.clone(),
                    error: err,
                });
            }
        }

        let parsebin = gst::ElementFactory::make("parsebin")
            .name("uri-parse")
            .build()
            .map_err(|e| UriDemuxerError::ElementCreation(format!("parsebin: {e}")))?;

        // parsebin autoplugs h264parse/h265parse internally with a default
        // `config-interval=-1` that inserts SPS/PPS before every keyframe.
        // We add our own downstream `h264parse`/`h265parse` (same as
        // `Mp4Demuxer`) with `config-interval=-1` to pin byte-stream output.
        // Having both parsers insert SPS/PPS would double the preamble, so
        // set parsebin's internal parser to `config-interval=0` (no in-band
        // insertion) and rely on the downstream parser for single insertion.
        // Match Mp4Demuxer: parsebin's internal h264parse/h265parse handle
        // SPS/PPS insertion (`config-interval=-1`) once per keyframe. We do
        // *not* add another parser downstream to avoid a double insertion.
        if let Some(parse_bin) = parsebin.dynamic_cast_ref::<gst::Bin>() {
            parse_bin.connect_deep_element_added(|_bin, _sub_bin, elem| {
                if let Some(factory) = elem.factory() {
                    let name = factory.name();
                    if (name == "h264parse" || name == "h265parse")
                        && elem.has_property("config-interval")
                    {
                        elem.set_property("config-interval", -1i32);
                    }
                }
            });
        }

        let queue = gst::ElementFactory::make("queue")
            .name("demux-queue")
            .build()
            .map_err(|e| UriDemuxerError::ElementCreation(format!("queue: {e}")))?;

        let sink_elem = gst::ElementFactory::make("appsink")
            .name("demux-sink")
            .build()
            .map_err(|e| UriDemuxerError::ElementCreation(format!("appsink: {e}")))?;
        sink_elem.set_property("sync", false);
        sink_elem.set_property("emit-signals", false);
        sink_elem.set_property("max-buffers", 64u32);
        sink_elem.set_property("drop", false);

        pipeline
            .add_many([&urisrc, &parsebin, &queue, &sink_elem])
            .map_err(|e| UriDemuxerError::LinkError(format!("add_many: {e}")))?;

        queue
            .link(&sink_elem)
            .map_err(|e| UriDemuxerError::LinkError(format!("queue->appsink: {e}")))?;

        let queue_sink_pad = queue
            .static_pad("sink")
            .ok_or_else(|| UriDemuxerError::ElementCreation("queue sink pad missing".into()))?;

        // Shared state
        let on_output = Arc::new(on_output);
        let detected_codec: Arc<Mutex<Option<VideoCodec>>> = Arc::new(Mutex::new(None));
        let video_info: Arc<Mutex<Option<VideoInfo>>> = Arc::new(Mutex::new(None));
        let stream_info_fired = Arc::new(AtomicBool::new(false));
        let finished = Arc::new(AtomicBool::new(false));
        let info_pair = Arc::new((Mutex::new(false), Condvar::new()));
        let done_pair = Arc::new((Mutex::new(false), Condvar::new()));

        // Adapter: the shared helpers emit only `VideoInfo`; wrap into our variant.
        let emit_stream_info: Arc<dyn Fn(VideoInfo) + Send + Sync> = {
            let on_output = on_output.clone();
            Arc::new(move |info: VideoInfo| on_output(UriDemuxerOutput::StreamInfo(info)))
        };

        // source-setup: apply per-source properties once the inner src element is
        // created. This works for any source type urisourcebin autoplugs
        // (filesrc, souphttpsrc, rtspsrc, hlsdemux's underlying http src, ...).
        {
            let source_properties = config.source_properties.clone();
            let on_output_ss = on_output.clone();
            urisrc.connect("source-setup", false, move |args| {
                let source = args
                    .get(1)
                    .and_then(|v| v.get::<gst::Element>().ok())
                    .expect("source-setup provides the source element");
                let factory_name = source
                    .factory()
                    .map(|f| f.name().to_string())
                    .unwrap_or_else(|| "<unknown>".into());
                for (name, value) in &source_properties {
                    if let Err(err) =
                        apply_property(source.upcast_ref::<glib::Object>(), name, value)
                    {
                        on_output_ss(UriDemuxerOutput::Error(UriDemuxerError::PropertyError {
                            element: factory_name.clone(),
                            name: name.clone(),
                            error: err,
                        }));
                    }
                }
                None
            });
        }

        // urisourcebin::pad-added → link src pad to parsebin.sink
        {
            let parsebin_weak = parsebin.downgrade();
            let on_output_uri_pad = on_output.clone();
            urisrc.connect_pad_added(move |_uri_src, src_pad| {
                let Some(parsebin) = parsebin_weak.upgrade() else {
                    return;
                };
                let Some(parse_sink) = parsebin.static_pad("sink") else {
                    return;
                };
                if parse_sink.is_linked() {
                    return;
                }

                // For multi-stream sources (e.g. RTSP), urisourcebin emits
                // separate pads per stream.  Only link non-audio pads so that
                // an audio pad arriving first does not steal the single
                // parsebin sink slot and cause the video stream to be dropped.
                let is_audio = src_pad
                    .current_caps()
                    .or_else(|| Some(src_pad.query_caps(None)))
                    .and_then(|caps| {
                        caps.structure(0).map(|s| {
                            let name = s.name();
                            if name.starts_with("audio/") {
                                return true;
                            }
                            if name == "application/x-rtp" {
                                if let Ok(media) = s.get::<&str>("media") {
                                    return media == "audio";
                                }
                            }
                            false
                        })
                    })
                    .unwrap_or(false);

                if is_audio {
                    log::debug!(
                        "UriDemuxer: skipping audio urisourcebin pad {}",
                        src_pad.name()
                    );
                    return;
                }

                if let Err(e) = src_pad.link(&parse_sink) {
                    on_output_uri_pad(UriDemuxerOutput::Error(UriDemuxerError::LinkError(
                        format!("urisourcebin->parsebin: {e}"),
                    )));
                }
            });
        }

        // parsebin::pad-added → choose first video/image pad, link [capsfilter?] → queue
        {
            let linked_video_pad = Arc::new(AtomicBool::new(false));
            let linked_video_pad_closure = linked_video_pad.clone();
            let pipeline_for_pad = pipeline.clone();
            let queue_sink_pad_for_pad = queue_sink_pad.clone();
            let detected_codec_pad = detected_codec.clone();
            let video_info_pad = video_info.clone();
            let stream_info_fired_pad = stream_info_fired.clone();
            let info_pair_pad = info_pair.clone();
            let emit_stream_info_pad = emit_stream_info.clone();
            let parsed = config.parsed;
            parsebin.connect_pad_added(move |_parsebin, src_pad| {
                if linked_video_pad_closure.load(Ordering::SeqCst) {
                    // Already accepted a video pad; drop additional streams.
                    log::debug!("UriDemuxer: ignoring extra parsebin pad {}", src_pad.name());
                    return;
                }

                let caps = src_pad
                    .current_caps()
                    .or_else(|| Some(src_pad.query_caps(None)));
                let codec_name = caps
                    .as_ref()
                    .and_then(|c| c.structure(0).map(|s| s.name().to_string()));

                let is_video = codec_name
                    .as_deref()
                    .is_some_and(|n| n.starts_with("video/") || n.starts_with("image/"));
                if !is_video {
                    log::debug!(
                        "UriDemuxer: ignoring non-video parsebin pad with caps {:?}",
                        codec_name
                    );
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

                if queue_sink_pad_for_pad.is_linked() {
                    return;
                }

                // parsebin already parses and inserts SPS/PPS at every IDR
                // (via its internal h264parse/h265parse with `config-interval=-1`).
                // For `parsed=true` we only pin byte-stream output via a
                // capsfilter. Adding another parser here would double-insert
                // the SPS/PPS preamble.
                let target_pad = if parsed {
                    match insert_byte_stream_capsfilter(
                        &pipeline_for_pad,
                        codec_name.as_deref(),
                        &queue_sink_pad_for_pad,
                    ) {
                        Ok(Some(capsf_sink)) => capsf_sink,
                        Ok(None) => queue_sink_pad_for_pad.clone(),
                        Err(_) => queue_sink_pad_for_pad.clone(),
                    }
                } else {
                    queue_sink_pad_for_pad.clone()
                };

                if src_pad.link(&target_pad).is_ok() {
                    linked_video_pad_closure.store(true, Ordering::SeqCst);
                }
            });
        }

        let appsink = sink_elem
            .dynamic_cast::<gst_app::AppSink>()
            .map_err(|_| UriDemuxerError::ElementCreation("appsink cast failed".into()))?;

        // Appsink callbacks (fire on GStreamer streaming thread)
        {
            let on_output_sample = on_output.clone();
            let detected_codec_s = detected_codec.clone();
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
                            &detected_codec_s,
                            &video_info_sample,
                            &stream_info_fired_sample,
                            &info_pair_sample,
                            emit_stream_info_sample.as_ref(),
                        ) {
                            Ok(pkt) => {
                                on_output_sample(UriDemuxerOutput::Packet(pkt));
                                Ok(gst::FlowSuccess::Ok)
                            }
                            Err(e) => {
                                if !finished_sample.swap(true, Ordering::SeqCst) {
                                    on_output_sample(UriDemuxerOutput::Error(
                                        UriDemuxerError::from(e),
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
                            on_output_eos(UriDemuxerOutput::Eos);
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
                        on_output_err(UriDemuxerOutput::Error(UriDemuxerError::PipelineError {
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

        let ret = pipeline.set_state(gst::State::Playing);
        if ret == Err(gst::StateChangeError) {
            return Err(UriDemuxerError::StateChangeFailed);
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

    /// Returns [`VideoInfo`] if it has already been observed on the source pad
    /// caps. Non-blocking.
    pub fn video_info(&self) -> Option<VideoInfo> {
        *self.video_info.lock().unwrap()
    }

    /// Block until [`VideoInfo`] is known, the pipeline terminates, or the
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
    /// Safe to call multiple times. After this call, no more callbacks will
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
}

impl Drop for UriDemuxer {
    fn drop(&mut self) {
        self.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mp4_muxer::Mp4Muxer;
    use std::path::Path;

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

    fn file_uri(path: &str) -> String {
        format!("file://{}", Path::new(path).display())
    }

    #[test]
    fn test_rejects_empty_uri() {
        let result = UriDemuxer::new(UriDemuxerConfig::new(""), |_| {});
        assert!(matches!(result, Err(UriDemuxerError::MissingUri)));
    }

    #[test]
    fn test_rejects_malformed_uri() {
        // No scheme separator — not a valid URI.
        let result = UriDemuxer::new(UriDemuxerConfig::new("not a uri at all"), |_| {});
        assert!(matches!(result, Err(UriDemuxerError::InvalidUri(_))));
    }

    #[test]
    fn test_file_uri_happy_path() {
        let _ = gst::init();
        let path = "/tmp/test_uri_demuxer_happy.mp4";
        let _ = std::fs::remove_file(path);
        make_h264_mp4(path, 5);

        let packets: Arc<Mutex<Vec<DemuxedPacket>>> = Arc::new(Mutex::new(Vec::new()));
        let got_eos = Arc::new(AtomicBool::new(false));
        let packets_cb = packets.clone();
        let got_eos_cb = got_eos.clone();

        let demuxer = UriDemuxer::new(
            UriDemuxerConfig::new(file_uri(path)).with_parsed(true),
            move |out| match out {
                UriDemuxerOutput::Packet(p) => packets_cb.lock().unwrap().push(p),
                UriDemuxerOutput::Eos => got_eos_cb.store(true, Ordering::SeqCst),
                UriDemuxerOutput::Error(e) => panic!("unexpected error: {e}"),
                UriDemuxerOutput::StreamInfo(_) => {}
            },
        )
        .unwrap();

        demuxer.wait();
        assert!(!packets.lock().unwrap().is_empty());
        assert!(got_eos.load(Ordering::SeqCst));
        assert_eq!(demuxer.detected_codec(), Some(VideoCodec::H264));
        let info = demuxer.video_info().expect("video info expected");
        assert_eq!(info.codec, VideoCodec::H264);
        assert!(info.width > 0);
        assert!(info.height > 0);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_wait_for_video_info_returns_info() {
        let _ = gst::init();
        let path = "/tmp/test_uri_demuxer_wait_info.mp4";
        let _ = std::fs::remove_file(path);
        make_h264_mp4(path, 3);

        let demuxer = UriDemuxer::new(UriDemuxerConfig::new(file_uri(path)), |_| {}).unwrap();
        let info = demuxer.wait_for_video_info(Duration::from_secs(5));
        let info = info.expect("video info expected");
        assert_eq!(info.codec, VideoCodec::H264);
        assert!(info.width > 0);
        assert!(info.height > 0);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_unknown_bin_property_rejected() {
        let _ = gst::init();
        let path = "/tmp/test_uri_demuxer_bad_bin_prop.mp4";
        let _ = std::fs::remove_file(path);
        make_h264_mp4(path, 1);

        let result = UriDemuxer::new(
            UriDemuxerConfig::new(file_uri(path))
                .with_bin_property("definitely-not-a-real-property", 42i32),
            |_| {},
        );
        assert!(
            matches!(
                result,
                Err(UriDemuxerError::PropertyError { ref name, .. }) if name == "definitely-not-a-real-property"
            ),
            "expected PropertyError, got {:?}",
            result.as_ref().err()
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_unknown_source_property_surfaces_in_callback() {
        let _ = gst::init();
        let path = "/tmp/test_uri_demuxer_bad_src_prop.mp4";
        let _ = std::fs::remove_file(path);
        make_h264_mp4(path, 1);

        let errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let errors_cb = errors.clone();
        let demuxer = UriDemuxer::new(
            UriDemuxerConfig::new(file_uri(path))
                // filesrc has no `latency` property — should surface as
                // PropertyError in the callback (not break the pipeline).
                .with_source_property("definitely-not-a-real-property", 200u32),
            move |out| {
                if let UriDemuxerOutput::Error(UriDemuxerError::PropertyError { name, .. }) = out {
                    errors_cb.lock().unwrap().push(name);
                }
            },
        )
        .unwrap();

        demuxer.wait_timeout(Duration::from_secs(5));
        let errs = errors.lock().unwrap();
        assert!(
            errs.iter().any(|n| n == "definitely-not-a-real-property"),
            "expected PropertyError for unknown source property, got {errs:?}"
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_finish_idempotent() {
        let _ = gst::init();
        let path = "/tmp/test_uri_demuxer_finish.mp4";
        let _ = std::fs::remove_file(path);
        make_h264_mp4(path, 1);

        let mut demuxer = UriDemuxer::new(UriDemuxerConfig::new(file_uri(path)), |_| {}).unwrap();
        demuxer.finish();
        demuxer.finish();
        assert!(demuxer.is_finished());

        let _ = std::fs::remove_file(path);
    }
}
