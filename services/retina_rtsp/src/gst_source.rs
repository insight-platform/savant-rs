use std::collections::VecDeque;
use std::str::FromStr;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use futures::StreamExt;
use glib::prelude::*;
use gstreamer::prelude::*;
use gstreamer::{self as gst, FlowSuccess};
use gstreamer_app as gst_app;
use gstreamer_rtp as gst_rtp;
use hashbrown::HashMap;
use log::{debug, error, info, warn};
use parking_lot::Mutex as ParkingMutex;
use savant_core::primitives::{
    frame::{VideoFrameContent, VideoFrame, VideoFrameTranscodingMethod},
    rust::{ExternalFrame, VideoCodec},
};
use savant_core::utils::rtp_pts_mapper::RtpPtsMapper;
use savant_services_common::job_writer::JobWriter;
use tokio::sync::{mpsc, Mutex};

use crate::configuration::RtspSourceGroup;
use crate::ntp_sync::NtpSync;
use crate::syncer::Syncer;
use crate::utils::{ensure_au_delimiter, ts2epoch_duration, wait_for_shutdown_flag};

const TIME_BASE: (i64, i64) = (1, 1_000_000_000);
const MAX_CHANNEL_CAPACITY: usize = 1_000;

#[derive(Debug, Clone)]
struct SourceInfo {
    source_id: String,
}

#[derive(Debug)]
enum GstEvent {
    VideoFrame {
        source_idx: usize,
        data: Vec<u8>,
        rtp_timestamp: u32,
        is_keyframe: bool,
        encoding: String,
        width: u32,
        height: u32,
        framerate: Option<(u32, u32)>,
    },
    RtcpSr {
        source_idx: usize,
        rtp_time: u32,
        ntp_time: u64,
    },
}

struct PerSourceState {
    clock_rate: Arc<ParkingMutex<u32>>,
}

struct PipelineGuard {
    pipeline: Option<gst::Pipeline>,
    group_name: String,
}

impl PipelineGuard {
    /// Sets the pipeline to Null so streaming stops and the processing loop can exit.
    fn stop(&mut self) {
        if let Some(pipeline) = self.pipeline.take() {
            if let Err(e) = pipeline.set_state(gst::State::Null) {
                error!(
                    "Failed to set pipeline to Null for group '{}': {:?}",
                    self.group_name, e
                );
            } else {
                info!("GStreamer pipeline stopped for group '{}'", self.group_name);
            }
        }
    }
}

impl Drop for PipelineGuard {
    fn drop(&mut self) {
        self.stop();
    }
}

pub async fn run_group(
    group_conf: &RtspSourceGroup,
    group_name: String,
    sink: Arc<Mutex<JobWriter>>,
    eos_on_restart: bool,
    shutdown: Arc<AtomicBool>,
) -> anyhow::Result<()> {
    let sources: Vec<_> = group_conf
        .sources
        .iter()
        .map(|s| SourceInfo {
            source_id: s.source_id.clone(),
        })
        .collect();

    let (tx, mut rx) = mpsc::channel::<GstEvent>(MAX_CHANNEL_CAPACITY);

    // Build GStreamer pipeline
    let pipeline = gst::Pipeline::new();
    let mut per_source_states: Vec<PerSourceState> = Vec::new();

    for (idx, source) in group_conf.sources.iter().enumerate() {
        let rtspsrc = gst::ElementFactory::make("rtspsrc")
            .name(format!("rtspsrc-{}", idx))
            .property("location", &source.url)
            .property_from_str("protocols", "tcp")
            .property("ntp-sync", false)
            .build()
            .with_context(|| format!("Failed to create rtspsrc for source {}", source.source_id))?;

        if let Some(opts) = &source.options {
            rtspsrc.set_property("user-id", &opts.username);
            rtspsrc.set_property("user-pw", &opts.password);
        }

        let appsink = gst_app::AppSink::builder()
            .name(format!("appsink-{}", idx))
            .sync(false)
            .build();

        pipeline.add_many([&rtspsrc, appsink.upcast_ref()])?;

        // Queue of per-frame RTP timestamps, filled by pad probe, consumed by appsink.
        let rtp_queue = Arc::new(ParkingMutex::new((0u32, VecDeque::<u32>::new())));
        let clock_rate_shared = Arc::new(ParkingMutex::new(0u32));

        per_source_states.push(PerSourceState {
            clock_rate: clock_rate_shared.clone(),
        });

        // RTCP SR capture: rtspsrc "new-manager" signal chain
        let tx_rtcp = tx.clone();
        let source_idx = idx;
        rtspsrc.connect("new-manager", false, move |values| {
            let manager = values[1].get::<gst::Element>().unwrap();
            let tx_inner = tx_rtcp.clone();
            let src_idx = source_idx;
            manager.connect("pad-added", false, move |values| {
                let pad = values[1].get::<gst::Pad>().unwrap();
                if !pad.name().starts_with("recv_rtcp_sink") {
                    return None;
                }
                let element = values[0].get::<gst::Element>().unwrap();
                let pad_name = pad.name();
                let session_idx: u32 = pad_name
                    .strip_prefix("recv_rtcp_sink_")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                let session: glib::Object =
                    element.emit_by_name("get-internal-session", &[&session_idx]);
                let tx_sr = tx_inner.clone();
                let si = src_idx;
                session.connect("on-receiving-rtcp", false, move |values| {
                    let buffer = values[1].get::<gst::Buffer>().unwrap();
                    if let Some(events) = parse_rtcp_sr(si, &buffer) {
                        for evt in events {
                            if let Err(e) = tx_sr.try_send(evt) {
                                log::warn!(
                                    "RTCP SR event dropped for source {si}: channel full or closed: {e}"
                                );
                            }
                        }
                    }
                    None
                });
                None
            });
            None
        });

        // pad-added: dynamically link rtspsrc → depay → parse → capsfilter → appsink
        let pipeline_weak = pipeline.downgrade();
        let appsink_name = format!("appsink-{}", idx);
        let source_id_for_pad = source.source_id.clone();
        let rtp_queue_for_pad = rtp_queue.clone();
        let clock_rate_for_pad = clock_rate_shared.clone();
        let tx_frame = tx.clone();
        let pad_source_idx = idx;

        rtspsrc.connect("pad-added", false, move |values| {
            let pad = values[1].get::<gst::Pad>().unwrap();
            if pad.direction() != gst::PadDirection::Src {
                return None;
            }
            let caps = match pad.current_caps() {
                Some(c) => c,
                None => return None,
            };
            let structure = caps.structure(0).unwrap();
            if !structure.name().starts_with("application/x-rtp") {
                return None;
            }

            let media = structure.get::<&str>("media").unwrap_or("unknown");
            if media != "video" {
                info!("Skipping non-video pad: media={}", media);
                return None;
            }

            let clock_rate = structure.get::<i32>("clock-rate").unwrap_or(90000) as u32;
            *clock_rate_for_pad.lock() = clock_rate;

            let encoding_name = structure
                .get::<&str>("encoding-name")
                .unwrap_or("UNKNOWN")
                .to_uppercase();

            info!(
                "Source {}: RTP pad added, codec={}, clock-rate={}",
                source_id_for_pad, encoding_name, clock_rate
            );

            let (depay_factory, parse_factory, caps_str, encoding_lower) =
                match encoding_name.as_str() {
                    "H264" => (
                        "rtph264depay",
                        "h264parse",
                        "video/x-h264,stream-format=byte-stream,alignment=au",
                        "h264",
                    ),
                    "H265" => (
                        "rtph265depay",
                        "h265parse",
                        "video/x-h265,stream-format=byte-stream,alignment=au",
                        "hevc",
                    ),
                    other => {
                        warn!("Unsupported codec: {}", other);
                        return None;
                    }
                };

            let pipeline = match pipeline_weak.upgrade() {
                Some(p) => p,
                None => return None,
            };

            let depay = gst::ElementFactory::make(depay_factory)
                .name(format!("depay-{}", pad_source_idx))
                .build()
                .unwrap();
            let parse = gst::ElementFactory::make(parse_factory)
                .name(format!("parse-{}", pad_source_idx))
                .property("config-interval", -1i32)
                .build()
                .unwrap();
            let capsfilter = gst::ElementFactory::make("capsfilter")
                .name(format!("capsfilter-{}", pad_source_idx))
                .property("caps", gst::Caps::from_str(caps_str).unwrap())
                .build()
                .unwrap();

            pipeline.add_many([&depay, &parse, &capsfilter]).unwrap();
            gst::Element::link_many([&depay, &parse, &capsfilter]).unwrap();
            let appsink_el = pipeline.by_name(&appsink_name).unwrap();
            capsfilter.link(&appsink_el).unwrap();

            for el in [&depay, &parse, &capsfilter] {
                el.sync_state_with_parent().unwrap();
            }

            pad.link(&depay.static_pad("sink").unwrap()).unwrap();

            // Pad probe on the RTP source pad to capture RTP timestamps.
            // Push each unique timestamp into a queue so the appsink callback
            // always reads the timestamp that belongs to *its* frame.
            let rtp_q = rtp_queue_for_pad.clone();
            pad.add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
                if let Some(gst::PadProbeData::Buffer(ref buffer)) = info.data {
                    if let Ok(rtp_buffer) = gst_rtp::RTPBuffer::from_buffer_readable(buffer) {
                        let ts = rtp_buffer.timestamp();
                        let mut guard = rtp_q.lock();
                        if guard.0 != ts || guard.1.is_empty() {
                            guard.0 = ts;
                            guard.1.push_back(ts);
                        }
                    }
                }
                gst::PadProbeReturn::Ok
            });

            // AppSink callbacks
            let tx_f = tx_frame.clone();
            let si = pad_source_idx;
            let enc = encoding_lower.to_string();
            let rtp_q_appsink = rtp_queue_for_pad.clone();

            let appsink = appsink_el.downcast::<gst_app::AppSink>().unwrap();
            appsink.set_callbacks(
                gst_app::AppSinkCallbacks::builder()
                    .new_sample(move |sink| {
                        let sample = sink.pull_sample().map_err(|_| gst::FlowError::Eos)?;
                        let buffer = sample.buffer().ok_or(gst::FlowError::Error)?;

                        let is_keyframe = !buffer.flags().contains(gst::BufferFlags::DELTA_UNIT);

                        let map = buffer.map_readable().map_err(|_| gst::FlowError::Error)?;
                        let data = map.as_slice().to_vec();

                        let rtp_timestamp = {
                            let mut guard = rtp_q_appsink.lock();
                            guard.1.pop_front().unwrap_or(guard.0)
                        };

                        let (width, height, framerate) = extract_video_info(&sample);

                        if let Err(e) = tx_f.try_send(GstEvent::VideoFrame {
                            source_idx: si,
                            data,
                            rtp_timestamp,
                            is_keyframe,
                            encoding: enc.clone(),
                            width,
                            height,
                            framerate,
                        }) {
                            log::warn!(
                                "Video frame event dropped for source {si}: channel full or closed: {e}"
                            );
                        }
                        Ok(FlowSuccess::Ok)
                    })
                    .build(),
            );

            None
        });
    }

    // Bus stream for errors and EOS (async-compatible, no GLib main loop needed)
    let bus = pipeline.bus().unwrap();
    let mut bus_stream = bus.stream_filtered(&[gst::MessageType::Error, gst::MessageType::Eos]);

    pipeline.set_state(gst::State::Playing)?;
    let mut pipeline_guard = PipelineGuard {
        pipeline: Some(pipeline),
        group_name: group_name.clone(),
    };
    info!("GStreamer pipeline started for group '{}'", group_name);

    // Drop the original sender so the channel closes when all callback-held
    // clones are dropped (otherwise rx.recv() can never return None).
    drop(tx);

    // Processing loop — same downstream logic as retina path
    let rtcp_once = group_conf
        .rtcp_sr_sync
        .as_ref()
        .map(|c| c.rtcp_once.unwrap_or(false))
        .unwrap_or(false);
    let mut ntp_sync = if let Some(sync_conf) = &group_conf.rtcp_sr_sync {
        info!(
            "NTP sync enabled for GStreamer group {}, window: {:?}, batch: {:?}",
            group_name, sync_conf.group_window_duration, sync_conf.batch_duration
        );
        Some(NtpSync::new(
            group_name.clone(),
            sync_conf.group_window_duration,
            sync_conf.batch_duration,
            sync_conf.network_skew_correction.unwrap_or(false),
        ))
    } else {
        None
    };

    let mut frame_buffer = Syncer::new();
    let mut active_streams: HashMap<String, ()> = HashMap::new();
    let mut rtp_mappers: HashMap<usize, RtpPtsMapper> = HashMap::new();
    let mut sr_queues: HashMap<usize, VecDeque<(u32, u64)>> = HashMap::new();
    let mut last_rtp_records: HashMap<String, i64> = HashMap::new();

    loop {
        tokio::select! {
            biased;

            _ = wait_for_shutdown_flag(&shutdown) => {
                info!(
                    "Shutdown requested for GStreamer group '{}', stopping pipeline",
                    group_name
                );
                pipeline_guard.stop();
                break;
            }
            event = rx.recv() => {
                let event = match event {
                    Some(e) => e,
                    None => {
                        info!("All GStreamer event senders dropped for group '{}', exiting", group_name);
                        break;
                    }
                };
                match event {
                    GstEvent::RtcpSr {
                        source_idx,
                        rtp_time,
                        ntp_time,
                    } => {
                        let source_id = &sources[source_idx].source_id;

                        info!(
                            "RTCP SR source_id={} rtp={} ntp=0x{:016X}",
                            source_id, rtp_time, ntp_time
                        );

                        // Always queue SR — clock_rate is not needed for queuing
                        sr_queues
                            .entry(source_idx)
                            .or_default()
                            .push_back((rtp_time, ntp_time));

                        let clock_rate = *per_source_states[source_idx].clock_rate.lock();
                        if clock_rate == 0 {
                            warn!(
                                "RTCP SR queued but clock rate unknown for {}, skipping NTP sync",
                                source_id
                            );
                            continue;
                        }

                        if let Some(ntp) = &mut ntp_sync {
                            if ntp.is_ready(source_id) && rtcp_once {
                                debug!("RTCP once: skipping further SR for {}", source_id);
                                continue;
                            }
                            let rtp_elapsed = rtp_time as i64;
                            ntp.add_rtp_mark(
                                source_id,
                                rtp_elapsed,
                                ntp_time,
                                std::num::NonZeroU32::new(clock_rate).unwrap(),
                            );
                        }
                    }
                    GstEvent::VideoFrame {
                        source_idx,
                        data,
                        rtp_timestamp,
                        is_keyframe,
                        encoding,
                        width,
                        height,
                        framerate,
                    } => {
                        let source_id = &sources[source_idx].source_id;

                        if let Some(ntp) = &ntp_sync {
                            if !ntp.is_ready(source_id) {
                                continue;
                            }
                        }

                        // Drain applicable SRs: apply first one as mapper seed,
                        // discard the rest (subsequent SRs already fed to NTP sync).
                        // Only the first SR establishes the PTS base — reseeding would
                        // cause PTS discontinuities that the Syncer rejects.
                        let clock_rate = *per_source_states[source_idx].clock_rate.lock();
                        if let Some(sr_queue) = sr_queues.get_mut(&source_idx) {
                            while let Some(&(sr_rtp, _)) = sr_queue.front() {
                                if rtp_timestamp >= sr_rtp {
                                    let (sr_rtp, sr_ntp) = sr_queue.pop_front().unwrap();
                                    if !rtp_mappers.contains_key(&source_idx) {
                                        let ntp_duration = ts2epoch_duration(sr_ntp, 0);
                                        let mapper = RtpPtsMapper::with_seed(
                                            sr_rtp,
                                            ntp_duration,
                                            (1, clock_rate as i64),
                                            (1, TIME_BASE.1),
                                        )
                                        .expect("valid timebase");
                                        rtp_mappers.insert(source_idx, mapper);
                                    }
                                } else {
                                    break;
                                }
                            }
                        }

                        // Fallback: when NTP sync is not configured and no RTCP SR has
                        // seeded the mapper yet, use the first keyframe's RTP timestamp
                        // as PTS base (matching the retina backend's rtp_bases approach).
                        if !rtp_mappers.contains_key(&source_idx)
                            && ntp_sync.is_none()
                            && is_keyframe
                            && clock_rate > 0
                        {
                            info!(
                                "Source {}: seeding RTP mapper from keyframe (no RTCP SR), rtp={}",
                                source_id, rtp_timestamp
                            );
                            let mapper = RtpPtsMapper::with_seed(
                                rtp_timestamp,
                                Duration::ZERO,
                                (1, clock_rate as i64),
                                (1, TIME_BASE.1),
                            )
                            .expect("valid timebase");
                            rtp_mappers.insert(source_idx, mapper);
                        }

                        let mapper = match rtp_mappers.get_mut(&source_idx) {
                            Some(m) => m,
                            None => {
                                debug!(
                                    "No RTP mapper for {} yet (waiting for RTCP SR or keyframe), skipping",
                                    source_id
                                );
                                continue;
                            }
                        };

                        let mapping = match mapper.map(rtp_timestamp) {
                            Ok(m) => m,
                            Err(e) => {
                                warn!("RTP mapping failed for {}: {}", source_id, e);
                                continue;
                            }
                        };
                        let pts = mapping.pts;
                        let rtp_time = rtp_timestamp as i64;

                        let last_rtp = last_rtp_records
                            .entry(source_id.clone())
                            .or_insert(rtp_time);
                        if rtp_time < *last_rtp {
                            warn!(
                                "RTP time {} < last {} for {}, possible wraparound handled by mapper",
                                rtp_time, last_rtp, source_id
                            );
                        }
                        *last_rtp = rtp_time;

                        if is_keyframe {
                            debug!(
                                target: "retina_rtsp::gst::keyframe",
                                "Source {}: keyframe at RTP {}",
                                source_id, rtp_timestamp
                            );
                            if !active_streams.contains_key(source_id) {
                                if eos_on_restart {
                                    info!("Source {}: sending EOS on restart", source_id);
                                    let _ = sink.lock().await.send_eos(source_id);
                                }
                                active_streams.insert(source_id.clone(), ());
                                info!("Source {}: stream active (GStreamer)", source_id);
                            }
                        }

                        if !active_streams.contains_key(source_id) {
                            debug!("Source {}: not active, waiting for keyframe", source_id);
                            continue;
                        }

                        let fps = framerate
                            .map(|(num, den)| (i64::from(num), i64::from(den.max(1))))
                            .unwrap_or((30, 1));
                        let fps = if fps.0 <= 0 { (30_i64, 1_i64) } else { fps };

                        // Ensure AU delimiter NALU is present (matches retina backend behaviour).
                        // h264parse/h265parse produce Annex B byte-stream but do not
                        // guarantee an AU delimiter; downstream DeepStream decoders need one.
                        let data = ensure_au_delimiter(data, &encoding);

                        let frame = VideoFrame::new(
                            source_id,
                            fps,
                            width as i64,
                            height as i64,
                            VideoFrameContent::External(ExternalFrame::new("zeromq", &None)),
                            VideoFrameTranscodingMethod::Copy,
                            VideoCodec::from_name(&encoding),
                            Some(is_keyframe),
                            TIME_BASE,
                            pts,
                            Some(pts),
                            None,
                        )?;

                        if let Some(ntp) = &mut ntp_sync {
                            ntp.prune_rtp_marks(source_id, rtp_time);
                            ntp.add_frame(rtp_time, frame, &data);
                            let frames = ntp.pull_frames();
                            for (frame, ts, frame_data) in frames {
                                debug!(
                                    "Sending NTP-synced frame ts={:?} for {}",
                                    ts,
                                    frame.get_source_id()
                                );
                                if let Some((frame, data)) = frame_buffer.add_frame(frame, frame_data) {
                                    let message = frame.to_message();
                                    let _ = sink.lock().await.send_message(
                                        &frame.get_source_id(),
                                        &message,
                                        &[&data],
                                    )?;
                                }
                            }
                        } else if let Some((frame, fdata)) = frame_buffer.add_frame(frame, data) {
                            let message = frame.to_message();
                            let _ = sink
                                .lock()
                                .await
                                .send_message(source_id, &message, &[&fdata])?;
                        }
                    }
                }
            }
            msg = bus_stream.next() => {
                match msg {
                    Some(msg) => match msg.view() {
                        gst::MessageView::Error(err) => {
                            error!(
                                "GStreamer pipeline error in group '{}': {} ({:?})",
                                group_name,
                                err.error(),
                                err.debug()
                            );
                            break;
                        }
                        gst::MessageView::Eos(_) => {
                            info!("GStreamer pipeline EOS for group '{}'", group_name);
                            break;
                        }
                        _ => {}
                    }
                    None => {
                        warn!("GStreamer bus stream ended for group '{}'", group_name);
                        break;
                    }
                }
            }
        }
    }

    Ok(())
}

fn parse_rtcp_sr(source_idx: usize, buffer: &gst::Buffer) -> Option<Vec<GstEvent>> {
    let map = buffer.map_readable().ok()?;
    let data = map.as_slice();
    let mut events = Vec::new();
    let mut offset = 0;

    while offset + 4 <= data.len() {
        // RTCP header: V(2) P(1) RC(5) PT(8) length(16)
        let pt = data[offset + 1];
        let length_field = u16::from_be_bytes([data[offset + 2], data[offset + 3]]) as usize;
        let packet_len = (length_field + 1) * 4;

        if pt == 200 && offset + 28 <= data.len() {
            // SR packet type = 200
            // Sender info starts at offset+4 (SSRC) + 4 = offset+8
            let ntp_msw = u32::from_be_bytes([
                data[offset + 8],
                data[offset + 9],
                data[offset + 10],
                data[offset + 11],
            ]);
            let ntp_lsw = u32::from_be_bytes([
                data[offset + 12],
                data[offset + 13],
                data[offset + 14],
                data[offset + 15],
            ]);
            let rtp_ts = u32::from_be_bytes([
                data[offset + 16],
                data[offset + 17],
                data[offset + 18],
                data[offset + 19],
            ]);
            let ntp_time = ((ntp_msw as u64) << 32) | (ntp_lsw as u64);

            events.push(GstEvent::RtcpSr {
                source_idx,
                rtp_time: rtp_ts,
                ntp_time,
            });
        }

        offset += packet_len;
    }

    if events.is_empty() {
        None
    } else {
        Some(events)
    }
}

fn extract_video_info(sample: &gst::Sample) -> (u32, u32, Option<(u32, u32)>) {
    let caps = match sample.caps() {
        Some(c) => c,
        None => return (0, 0, None),
    };
    let structure = match caps.structure(0) {
        Some(s) => s,
        None => return (0, 0, None),
    };
    let width = structure.get::<i32>("width").unwrap_or(0) as u32;
    let height = structure.get::<i32>("height").unwrap_or(0) as u32;
    let framerate = structure
        .get::<gst::Fraction>("framerate")
        .ok()
        .map(|f| (f.numer() as u32, f.denom() as u32));
    (width, height, framerate)
}
