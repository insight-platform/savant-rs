use crate::{
    configuration::{RtspSource, ServiceConfiguration},
    ntp_sync::NtpSync,
    syncer::Syncer,
    utils::{
        check_contains_au_delimiter, convert_to_annexb, is_keyframe, H264_AU_DELIMITER,
        HEVC_AU_DELIMITER,
    },
};
use anyhow::{bail, Context};
use futures::StreamExt;
use hashbrown::HashMap;
use log::{debug, error, info, warn};
use replaydb::job_writer::JobWriter;
use retina::{
    client::{
        Credentials, Demuxed, InitialTimestampPolicy, Session, SessionGroup, SessionOptions,
        SetupOptions, TcpTransportOptions, Transport,
    },
    codec::CodecItem,
};
use savant_core::primitives::{
    frame::{VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod},
    rust::ExternalFrame,
};

use std::{borrow::Cow, num::NonZeroU32, sync::Arc, time::SystemTime};
use tokio::{select, sync::Mutex, task::JoinSet};
use url::Url;

const MAX_JUMP_SECS: u32 = 10;
const MAX_CHANNEL_CAPACITY: usize = 1_000;
const TIME_BASE: (i64, i64) = (1, 1_000_000_000);

#[derive(Debug, Clone)]
pub struct StreamInfo {
    pub source_id: String,
    pub encoding: String,
    pub clock_rate: u32,
    pub pixel_dimensions: (u32, u32),
    pub frame_rate: Option<(u32, u32)>,
}

pub struct VideoStream {
    pub session: Demuxed,
    pub stream_info: Arc<StreamInfo>,
}

pub struct RtspServiceGroup {
    group_name: String,
    conf: Arc<ServiceConfiguration>,
    active_streams: HashMap<String, ()>,
    rtp_bases: HashMap<String, (i64, SystemTime)>,
    last_rtp_records: HashMap<String, i64>,
    ntp_sync: Option<NtpSync>,
    frame_buffer: Syncer,
    rtcp_once: bool,
    eos_on_restart: bool,
}

impl RtspServiceGroup {
    pub async fn init_source(
        source: &RtspSource,
        conf: Arc<ServiceConfiguration>,
        group_name: String,
        session_group: Arc<SessionGroup>,
    ) -> anyhow::Result<VideoStream> {
        let mut stream_infos = HashMap::new();
        let creds = source.options.as_ref().map(|creds| Credentials {
            username: creds.username.clone(),
            password: creds.password.clone(),
        });

        let url = Url::parse(&source.url)?;
        let mut session = Session::describe(
            url,
            SessionOptions::default()
                .creds(creds)
                .session_group(session_group.clone()),
        )
        .await
        .with_context(|| {
            format!(
                "Failed to connect to RTSP source {} ({})",
                source.source_id, source.url
            )
        })?;

        let mut video_stream_positions = Vec::new();
        for (i, stream) in session.streams().iter().enumerate() {
            let encoding_name = match stream.encoding_name() {
                "jpeg" => "jpeg",
                "h264" => "h264",
                "h265" => "hevc",
                _ => "unknown",
            };

            let stream_header = format!(
                "Group: {}, SourceId: {}, Stream #{}: Media: {}, Encoding: {}, Clock rate: 1/{}",
                group_name,
                source.source_id,
                i,
                stream.media(),
                encoding_name,
                stream.clock_rate_hz()
            );

            if let Some(parameters) = stream.parameters() {
                match parameters {
                    retina::codec::ParametersRef::Video(video_parameters) => {
                        let pixel_aspect_ratio = video_parameters.pixel_aspect_ratio();
                        let pixel_dimensions = video_parameters.pixel_dimensions();
                        let frame_rate = video_parameters.frame_rate();
                        let codec = video_parameters.rfc6381_codec();

                        let (support_char, supported) =
                            if !matches!(encoding_name, "jpeg" | "h264" | "hevc") {
                                ('🔴', false)
                            } else {
                                ('🟢', true)
                            };

                        info!(
                            "{} 📹 [{}] RFC6381 Codec: {:?}, Pixel aspect ratio: {:?}, Frame dimensions: {:?}, Frame rate: {:?}",
                            support_char,
                            stream_header,
                            codec,
                            pixel_aspect_ratio.map(|(w, h)| format!("({}x{})", w, h)).unwrap_or("UNKNOWN".to_string()),
                            pixel_dimensions,
                            frame_rate.map(|(num, den)| format!("{}/{}", num, den)).unwrap_or("UNKNOWN".to_string()),
                        );
                        if supported {
                            video_stream_positions.push(i);
                        }
                        stream_infos.insert(
                            i,
                            StreamInfo {
                                source_id: source.source_id.clone(),
                                encoding: encoding_name.to_string(),
                                //rfc6381_codec: codec.to_string(),
                                clock_rate: stream.clock_rate_hz(),
                                //pixel_aspect_ratio: pixel_aspect_ratio,
                                pixel_dimensions,
                                frame_rate,
                            },
                        );
                    }
                    retina::codec::ParametersRef::Audio(audio_parameters) => {
                        let codec = audio_parameters.rfc6381_codec();
                        info!(
                            "    🔉 ❌ RFC6381 Codec: {:?}, Clock rate: 1/{}",
                            codec,
                            audio_parameters.clock_rate(),
                        );
                    }
                    retina::codec::ParametersRef::Message(message_parameters) => {
                        info!("    ✉️ ❌ Message: {:?}", message_parameters);
                    }
                }
            }
        }
        if video_stream_positions.is_empty() {
            bail!(
                "No video streams found for source {} ({})",
                source.source_id,
                source.url
            );
        }

        let video_stream_position = source.stream_position.unwrap_or(video_stream_positions[0]);
        if !video_stream_positions.contains(&video_stream_position) {
            bail!(
                "The requested stream position {} is out of range for source {} ({}), video streams available for indices: {:?}",
                video_stream_position,
                source.source_id,
                source.url,
                video_stream_positions
            );
        }

        session
            .setup(
                video_stream_position,
                SetupOptions::default().transport(Transport::Tcp(TcpTransportOptions::default())),
            )
            .await
            .with_context(|| {
                format!(
                    "Failed to setup session for RTSP source {} ({})",
                    source.source_id, source.url
                )
            })?;

        let demuxed_session = session
            .play(
                retina::client::PlayOptions::default()
                    .enforce_timestamps_with_max_jump_secs(NonZeroU32::new(MAX_JUMP_SECS).unwrap())
                    .initial_timestamp(if conf.rtsp_sources[&group_name].rtcp_sr_sync.is_some() {
                        InitialTimestampPolicy::Require
                    } else {
                        InitialTimestampPolicy::Default
                    }),
            )
            .await?
            .demuxed()?;

        Ok(VideoStream {
            session: demuxed_session,
            stream_info: Arc::new(stream_infos.remove(&video_stream_position).unwrap()),
        })
    }

    pub async fn new(conf: Arc<ServiceConfiguration>, group_name: String) -> anyhow::Result<Self> {
        // log streams
        Ok(Self {
            group_name: group_name.clone(),
            conf: conf.clone(),
            eos_on_restart: conf.eos_on_restart.unwrap_or(true),
            rtcp_once: conf.rtsp_sources[&group_name]
                .rtcp_sr_sync
                .as_ref()
                .map(|c| c.rtcp_once.unwrap_or(false))
                .unwrap_or(false),
            frame_buffer: Syncer::new(),
            rtp_bases: HashMap::new(),
            last_rtp_records: HashMap::new(),
            active_streams: HashMap::new(),
            ntp_sync: if let Some(window_duration) = &conf.rtsp_sources[&group_name].rtcp_sr_sync {
                info!(
                    "NTP sync enabled for group {}, window duration: {:?}, batch duration: {:?}",
                    group_name.clone(),
                    window_duration.group_window_duration,
                    window_duration.batch_duration
                );
                warn!("A stream in group {} will become active when the first RTCP Sender Report will be received.", group_name);
                let skew_correction = window_duration.network_skew_correction.unwrap_or(false);
                if skew_correction {
                    warn!(
                        "Network skew correction is enabled. It relies on the actual network latency to correct unprecise NTP timestamps."
                    );
                }
                Some(NtpSync::new(
                    group_name,
                    window_duration.group_window_duration,
                    window_duration.batch_duration,
                    window_duration.network_skew_correction.unwrap_or(false),
                ))
            } else {
                info!("NTP sync disabled for group {}", group_name);
                None
            },
        })
    }

    pub async fn play(
        &mut self,
        sink: Arc<Mutex<JobWriter>>,
        session_group: Arc<SessionGroup>,
    ) -> anyhow::Result<()> {
        let (tx, mut rx) = tokio::sync::mpsc::channel(MAX_CHANNEL_CAPACITY);
        let mut tasks = JoinSet::new();
        for source in &self.conf.rtsp_sources[&self.group_name].sources {
            let tx = tx.clone();
            let source_id = source.source_id.clone();
            let conf = self.conf.clone();
            let group_name = self.group_name.clone();
            let source = source.clone();
            let session_group = session_group.clone();

            tasks.spawn(async move {
                loop {
                    let stream = Self::init_source(
                        &source,
                        conf.clone(),
                        group_name.clone(),
                        session_group.clone(),
                    ).await;

                    if let Err(e) = stream {
                        error!("Failed to initialize stream for source {}, error: {:?}", source_id, e);
                        tokio::time::sleep(conf.reconnect_interval.unwrap()).await;
                        continue;
                    }

                    let mut stream = stream.unwrap();
                    let mut restarted = true;
                    while let Some(res) = stream.session.next().await {
                        match res {
                            Ok(item) => {
                                
                            tx
                                .send((restarted, item, stream.stream_info.clone()))
                                .await
                                .unwrap_or_else(|e| {
                                    let message = format!(
                                        "Failed to send item to channel for RTSP source {}, error: {:?}",
                                        source_id, e
                                    );
                                    panic!("{}", message);
                                });
                                restarted = false;
                            },
                            Err(e) => {
                                error!(
                                    "Stream {} ended unexpectedly with error: {:?}",
                                    source_id, e
                                );
                                break;
                            }
                        }
                    }
                    error!("Stream {} ended unexpectedly", source_id);
                }
            });
        }

        loop {
            select! {
                err = tasks.join_next() => {
                    if let Some(Err(e)) = err {
                        error!("Task ended with error: {:?}", e);
                    }
                    break;
                }
                Some((restarted,data_item, stream_info)) = rx.recv() => {
                    debug!("Received item from {:?} {:?}", stream_info, data_item);
                    self.process_item(restarted, &stream_info, data_item, sink.clone()).await?;
                }
            }
        }
        Ok(())
    }

    async fn process_item(
        &mut self,
        restarted: bool,
        stream_info: &Arc<StreamInfo>,
        data_item: CodecItem,
        sink: Arc<Mutex<JobWriter>>,
    ) -> anyhow::Result<()> {
        if restarted {
            self.active_streams.remove(&stream_info.source_id);
            self.rtp_bases.remove(&stream_info.source_id);
            self.last_rtp_records.remove(&stream_info.source_id);

            self.ntp_sync.as_mut().map(|s| s.prune(&stream_info.source_id));
            self.frame_buffer.prune(&stream_info.source_id);
        }
        let source_id = &stream_info.source_id;
        match data_item {
            CodecItem::VideoFrame(video_frame) => {
                if let Some(ntp_sync) = &mut self.ntp_sync {
                    if !ntp_sync.is_ready(source_id) {
                        return Ok(());
                    }
                }

                if video_frame.loss() > 0 {
                    warn!(
                        "Detected {} lost RTP frames for source_id={}",
                        video_frame.loss(),
                        source_id
                    );
                }

                let rtp_time = video_frame.timestamp().elapsed();

                let last_rtp_record = self
                    .last_rtp_records
                    .entry(source_id.clone())
                    .or_insert(rtp_time);
                let diff = rtp_time - *last_rtp_record;
                if diff < 0 {
                    warn!(
                        "Received RTP time {} is less than last RTP time {}, this is likely a stream bug.",
                        rtp_time, last_rtp_record
                    );
                }
                *last_rtp_record = rtp_time;

                let size = video_frame.data().len();
                debug!(
                    "Received video frame from {}, stream_id: {}, rtp_time: {}, size: {}",
                    source_id,
                    video_frame.stream_id(),
                    rtp_time,
                    size,
                );

                let frame_rate = stream_info
                    .frame_rate
                    .map(|(num, den)| format!("{}/{}", den / num, 1))
                    .unwrap_or("30/1".to_string());

                let frame_data = if matches!(stream_info.encoding.as_str(), "h264" | "hevc") {
                    Cow::Owned(convert_to_annexb(video_frame)?)
                } else {
                    Cow::Borrowed(video_frame.data())
                };

                let au_delimiter =
                    check_contains_au_delimiter(&frame_data, &source_id, rtp_time, stream_info);

                let frame_data = if !au_delimiter {
                    // add au delimiter
                    let new_data = if matches!(stream_info.encoding.as_str(), "h264") {
                        let mut new_data = H264_AU_DELIMITER.to_vec();
                        new_data.extend_from_slice(&frame_data);
                        Cow::Owned(new_data)
                    } else if matches!(stream_info.encoding.as_str(), "hevc") {
                        let mut new_data = HEVC_AU_DELIMITER.to_vec();
                        new_data.extend_from_slice(&frame_data);
                        Cow::Owned(new_data)
                    } else {
                        frame_data
                    };
                    new_data
                } else {
                    frame_data
                };

                let kf = is_keyframe(&frame_data, &source_id, rtp_time, stream_info);
                if kf {
                    debug!(
                        target: "retina_rtsp::service::keyframe_detector",
                        "Source_id: {}, RTP time: {}, Keyframe arrived",
                        source_id, rtp_time
                    );

                    if !self.active_streams.contains_key(source_id) {
                        if self.eos_on_restart {
                            info!("Source_id: {}, EOS on restart is enabled, sending EOS message to reset remote stream decoder state.", source_id);
                            let _ = sink.lock().await.send_eos(&source_id)?;
                        }
                        self.active_streams.insert(source_id.clone(), ());
                        info!(
                            "Stream_id: {}, RTP time: {}, is active now",
                            source_id, rtp_time
                        );
                        self.rtp_bases
                            .insert(source_id.clone(), (rtp_time, SystemTime::now()));
                        info!("Stream_id: {}, PTS base is set to: {}", source_id, rtp_time);
                    }
                }

                if !self.active_streams.contains_key(source_id) {
                    debug!(
                        "Stream_id: {}, RTP time: {}, is not active, skipping frame before the first keyframe is found",
                        source_id, rtp_time
                    );
                    return Ok(());
                }

                let (pts_base, _base_time) = self.rtp_bases[source_id];
                debug!(target: "retina_rtsp::pts_builder", "Source ID: {}, RTP time: {}, RTP base: {}, Clock rate: {}", source_id, rtp_time, pts_base, stream_info.clock_rate);
                let pts_sec = (rtp_time - pts_base) as f64 / stream_info.clock_rate as f64;
                let pts = pts_sec / TIME_BASE.0 as f64 * TIME_BASE.1 as f64;
                debug!(target: "retina_rtsp::pts_builder", "Source ID: {}, Float PTS: {}", source_id, pts);
                let pts = pts.round() as i64;
                debug!(target: "retina_rtsp::pts_builder", "Source ID: {}, Int PTS: {}", source_id, pts);

                let frame = VideoFrameProxy::new(
                    &source_id,
                    &frame_rate,
                    stream_info.pixel_dimensions.0 as i64,
                    stream_info.pixel_dimensions.1 as i64,
                    VideoFrameContent::External(ExternalFrame::new("zeromq", &None)),
                    VideoFrameTranscodingMethod::Copy,
                    &Some(&stream_info.encoding),
                    Some(kf),
                    TIME_BASE,
                    pts,
                    Some(pts),
                    None,
                );
                debug!(
                    target: "retina_rtsp::service::frame_creator",
                    "Created a new frame: {:?}", 
                    frame);

                if let Some(ntp_sync) = &mut self.ntp_sync {
                    ntp_sync.prune_rtp_marks(&source_id, rtp_time);
                    ntp_sync.add_frame(rtp_time, frame, &frame_data);
                    let frames = ntp_sync.pull_frames();

                    for (frame, ts, frame_data) in frames {
                        debug!(
                            "Sending frame with NTP timestamp {:?} for source_id={}",
                            ts,
                            frame.get_source_id()
                        );
                        if let Some((frame, data)) = self.frame_buffer.add_frame(frame, frame_data)
                        {
                            let message = frame.to_message();
                            let _ = sink.lock().await.send_message(
                                &frame.get_source_id(),
                                &message,
                                &[&data],
                            )?;
                            debug!(
                                target: "retina_rtsp::service::send_frame",
                                "Source_id: {}, Sent video frame with PTS {} to sink",
                                frame.get_source_id(),
                                frame.get_pts()
                            );
                        }
                    }
                } else {
                    if let Some((frame, data)) =
                        self.frame_buffer.add_frame(frame, frame_data.to_vec())
                    {
                        let message = frame.to_message();
                        let _ = sink
                            .lock()
                            .await
                            .send_message(&source_id, &message, &[&data])?;
                        debug!(
                            target: "retina_rtsp::service::send_frame",
                            "Source_id: {}, Sent video frame with PTS {} to sink",
                            frame.get_source_id(),
                            frame.get_pts()
                        );
                    }
                }
            }
            CodecItem::AudioFrame(audio_frame) => {
                info!("Received audio frame from {} {:?}", &source_id, audio_frame);
            }
            CodecItem::MessageFrame(message_frame) => {
                info!(
                    "Received message frame from {} {:?}",
                    &source_id, message_frame
                );
            }
            CodecItem::Rtcp(rtcp) => {
                if let (Some(t), Some(Ok(Some(sr)))) = (
                    rtcp.rtp_timestamp(),
                    rtcp.pkts()
                        .next()
                        .map(retina::rtcp::PacketRef::as_sender_report),
                ) {
                    info!(
                        "RTCP Sender report source_id={} RTP is {} NTP is {}",
                        source_id,
                        t,
                        sr.ntp_timestamp().0
                    );

                    self.ntp_sync.as_mut().map(|s| {
                        if s.is_ready(&source_id) && self.rtcp_once {
                            info!("RTCP is configured to be received only once, skipping further RTCP packets for source_id={}", source_id);
                            return ();
                        }

                        s.add_rtp_mark(
                            &source_id,
                            t.elapsed(),
                            sr.ntp_timestamp(),
                            t.clock_rate().into(),
                        )
                    });
                }
            }
            _ => todo!(),
        }
        Ok(())
    }
}

pub async fn run_group(
    conf: Arc<ServiceConfiguration>,
    group_name: String,
    session_group: Arc<SessionGroup>,
    sink: Arc<Mutex<JobWriter>>,
) -> anyhow::Result<()> {
    let mut service_group = RtspServiceGroup::new(conf.clone(), group_name.clone()).await?;
    service_group.play(sink, session_group).await?;
    Ok(())
}
