use crate::{
    configuration::ServiceConfiguration,
    utils::{convert_to_annexb, ts2epoch_duration},
};
use anyhow::{bail, Context};
use cros_codecs::codec::h264::parser::Nalu as H264Nalu;
use cros_codecs::codec::h265::parser::Nalu as H265Nalu;
use futures::StreamExt;
use hashbrown::HashMap;
use log::{debug, error, info, warn};
use replaydb::job_writer::JobWriter;
use retina::{
    client::{
        Credentials, Demuxed, Session, SessionGroup, SessionOptions, SetupOptions,
        TcpTransportOptions, Transport,
    },
    codec::CodecItem,
    NtpTimestamp,
};
use savant_core::primitives::{
    frame::{VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod},
    rust::{AttributeValue, ExternalFrame},
    Attribute, WithAttributes,
};
use std::{
    borrow::Cow,
    collections::VecDeque,
    io::Cursor,
    num::NonZeroU32,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{select, sync::Mutex, task::JoinSet};
use url::Url;

const MAX_JUMP_SECS: u32 = 10;
const MAX_CHANNEL_CAPACITY: usize = 1_000;
const TIME_BASE: (i64, i64) = (1, 1_000);

#[derive(Debug, Clone)]
pub struct StreamInfo {
    pub encoding: String,
    pub clock_rate: u32,
    pub pixel_dimensions: (u32, u32),
    pub frame_rate: Option<(u32, u32)>,
}

pub struct VideoStream {
    pub position: usize,
    pub session: Demuxed,
}

pub struct NtpSync {
    pub sync_window: VecDeque<(Duration, VideoFrameProxy, Vec<u8>)>,
    pub sync_window_duration: Duration,
    pub batch_duration: Duration,
    pub rtp_marks: HashMap<String, VecDeque<(i64, Duration, NonZeroU32)>>,
    pub batch_id: u64,
}

impl NtpSync {
    pub fn new(duration: Duration, batch_duration: Duration) -> Self {
        Self {
            sync_window: VecDeque::new(),
            sync_window_duration: duration,
            batch_duration,
            rtp_marks: HashMap::new(),
            batch_id: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }

    pub fn add_rtp_mark(
        &mut self,
        source_id: &String,
        rtp_time: i64,
        ntp_time: NtpTimestamp,
        clock_rate: NonZeroU32,
    ) {
        self.rtp_marks
            .entry(source_id.clone())
            .or_insert_with(VecDeque::new)
            .push_back((rtp_time, ts2epoch_duration(ntp_time), clock_rate));
    }

    pub fn is_ready(&self, source_id: &String) -> bool {
        self.rtp_marks.contains_key(source_id)
    }

    pub fn prune_rtp_marks(&mut self, source_id: &String, rtp_time: i64) {
        let marks = self
            .rtp_marks
            .entry(source_id.clone())
            .or_insert_with(VecDeque::new);

        if marks.len() <= 1 {
            return;
        }

        let mut drain_history = 0;
        for (rtp, _, _) in marks.iter() {
            if *rtp <= rtp_time {
                drain_history += 1;
            } else {
                break;
            }
        }

        let drain_history = drain_history.min(marks.len() - 1);
        if drain_history > 1 {
            for _ in 0..drain_history - 1 {
                let elt = marks.pop_front().unwrap();
                info!(
                    "Pruned RTP mark {} for source_id={}, current RTP packet timestamp: {}",
                    elt.0, source_id, rtp_time
                );
            }
            info!(
                "Current RTP mark for source_id={} is {}, packet RTP is {}, Packet RTP - RTP mark = {}",
                source_id,
                marks.front().unwrap().0,
                rtp_time,
                rtp_time - marks.front().unwrap().0
            );
        }
    }

    pub fn current_rtp_mark(&self, source_id: &String) -> Option<(i64, Duration, NonZeroU32)> {
        let marks = self.rtp_marks.get(source_id);
        marks
            .map(|marks| {
                if marks.is_empty() {
                    return None;
                }
                let (rtp, ntp, clock_rate) = marks.front().unwrap();
                Some((*rtp, *ntp, *clock_rate))
            })
            .flatten()
    }

    pub fn add_frame(&mut self, rtp: i64, mut frame: VideoFrameProxy, frame_data: &[u8]) {
        let current_rtp_mark = self.current_rtp_mark(&frame.get_source_id());
        if let Some((rtp_mark, ntp_mark, clock_rate)) = current_rtp_mark {
            let diff = rtp - rtp_mark;
            if diff < 0 {
                error!(
                    "RTP time {} is less than RTP mark {}, diff: {}",
                    rtp, rtp_mark, diff
                );
                return;
            }
            let diff = (diff as f64 / clock_rate.get() as f64 * 1_000_000_000_f64) as u64;
            let diff = Duration::from_nanos(diff);

            let ts = diff + ntp_mark;
            debug!(
                "Adding frame with RTP time {:0>16} and computed NTP timestamp {:?}, source_id={}",
                rtp,
                ts,
                frame.get_source_id()
            );
            frame.set_attribute(Attribute::new(
                "retina-rtsp",
                "ntp-timestamp-ns",
                vec![AttributeValue::string(&ts.as_nanos().to_string(), None)],
                &None,
                true,
                false,
            ));
            self.sync_window.push_back((ts, frame, frame_data.to_vec()));
            self.sync_window
                .make_contiguous()
                .sort_by(|(ts0, _, _), (ts1, _, _)| ts0.cmp(ts1));
        }
    }

    pub fn pull_frames(&mut self) -> Vec<(VideoFrameProxy, Duration, Vec<u8>)> {
        let mut res = HashMap::new();
        if self.sync_window.is_empty() {
            return res.into_values().collect();
        }
        let first = self.sync_window.front().unwrap().0;
        let last = self.sync_window.back().unwrap().0;
        let diff = last - first;
        debug!(
            "Pulling frames, current sync window duration: {:?}, required duration: {:?}, length: {}",
            diff, self.sync_window_duration, self.sync_window.len()
        );
        if diff > self.sync_window_duration {
            // we have window built and full
            // now we must read a batch of frames
            let mut current_duration = first;
            let batch_id = self.batch_id;
            let before_sync_window_len = self.sync_window.len();
            while current_duration - first < self.batch_duration {
                {
                    let (_, frame, _) = self.sync_window.front_mut().unwrap();
                    // no duplicates allowed
                    if res.contains_key(&frame.get_source_id()) {
                        break;
                    }
                }
                let (ts, frame, frame_data) = self.sync_window.pop_front().unwrap();
                res.insert(frame.get_source_id(), (frame, ts, frame_data));
                current_duration = ts;
            }
            let batch_sources = res.keys().cloned().collect::<Vec<_>>();
            let batch_participants = res
                .into_values()
                .map(|(mut frame, duration, frame_data)| {
                    frame.set_attribute(Attribute::new(
                        "retina-rtsp",
                        "batch-id",
                        vec![AttributeValue::string(&batch_id.to_string(), None)],
                        &None,
                        true,
                        false,
                    ));
                    frame.set_attribute(Attribute::new(
                        "retina-rtsp",
                        "batch-sources",
                        vec![AttributeValue::string_vector(
                            batch_sources
                                .as_slice()
                                .iter()
                                .map(|s| s.to_string())
                                .collect(),
                            None,
                        )],
                        &None,
                        true,
                        false,
                    ));
                    (frame, duration, frame_data)
                })
                .collect();
            self.batch_id += 1;
            let after_sync_window_len = self.sync_window.len();
            debug!(
                "Pulled {} frames from sync window, before len: {}, after len: {}",
                before_sync_window_len - after_sync_window_len,
                before_sync_window_len,
                after_sync_window_len
            );
            batch_participants
        } else {
            res.into_values().collect()
        }
    }
}

pub struct RtspServiceGroup {
    streams: HashMap<String, VideoStream>,
    active_streams: HashMap<String, ()>,
    stream_infos: HashMap<(String, usize), StreamInfo>,
    rtp_bases: HashMap<String, i64>,
    last_rtp_records: HashMap<String, i64>,
    ntp_sync: Option<NtpSync>,
}

impl RtspServiceGroup {
    pub async fn new(
        conf: Arc<ServiceConfiguration>,
        group_name: String,
        session_group: Arc<SessionGroup>,
    ) -> anyhow::Result<Self> {
        let mut sessions = HashMap::new();
        let mut stream_infos = HashMap::new();

        for source in &conf.rtsp_sources[&group_name].sources {
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

            info!("[Group: {}, SourceId: {}]", group_name, source.source_id);
            let mut video_stream_positions = Vec::new();
            for (i, stream) in session.streams().iter().enumerate() {
                let encoding_name = match stream.encoding_name() {
                    "jpeg" => "jpeg",
                    "h264" => "h264",
                    "h265" => "hevc",
                    _ => "unknown",
                };
                info!(
                    " ▶ Stream #{}: Media: {}, Encoding: {}, Clock rate: 1/{}",
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
                                "    {} 📹 RFC6381 Codec: {:?}, Pixel aspect ratio: {:?}, Frame dimensions: {:?}, Frame rate: {:?}",
                                support_char,
                                codec,
                                pixel_aspect_ratio.map(|(w, h)| format!("({}x{})", w, h)).unwrap_or("UNKNOWN".to_string()),
                                pixel_dimensions,
                                frame_rate.map(|(num, den)| format!("{}/{}", num, den)).unwrap_or("UNKNOWN".to_string()),
                            );
                            if supported {
                                video_stream_positions.push(i);
                            }
                            stream_infos.insert(
                                (source.source_id.clone(), i),
                                StreamInfo {
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
                    SetupOptions::default()
                        .transport(Transport::Tcp(TcpTransportOptions::default())),
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
                    retina::client::PlayOptions::default().enforce_timestamps_with_max_jump_secs(
                        NonZeroU32::new(MAX_JUMP_SECS).unwrap(),
                    ),
                )
                .await?
                .demuxed()?;

            sessions.insert(
                source.source_id.clone(),
                VideoStream {
                    position: source.stream_position.unwrap_or(0),
                    session: demuxed_session,
                },
            );
        }
        // log streams
        Ok(Self {
            streams: sessions,
            stream_infos,
            rtp_bases: HashMap::new(),
            last_rtp_records: HashMap::new(),
            active_streams: HashMap::new(),
            ntp_sync: if let Some(window_duration) = &conf.rtsp_sources[&group_name].rtcp_sr_sync {
                info!(
                    "NTP sync enabled for group {}, window duration: {:?}, batch duration: {:?}",
                    group_name,
                    window_duration.group_window_duration,
                    window_duration.batch_duration
                );
                warn!("A stream in group {} will become active when the first RTCP Sender Report will be received.", group_name);
                Some(NtpSync::new(
                    window_duration.group_window_duration,
                    window_duration.batch_duration,
                ))
            } else {
                info!("NTP sync disabled for group {}", group_name);
                None
            },
        })
    }

    pub async fn play(&mut self, sink: Arc<Mutex<JobWriter>>) -> anyhow::Result<()> {
        let streams = self.streams.drain().collect::<HashMap<_, _>>();
        let (tx, mut rx) = tokio::sync::mpsc::channel(MAX_CHANNEL_CAPACITY);
        let mut tasks = JoinSet::new();
        for (source_id, mut stream) in streams {
            let tx = tx.clone();
            let task = async move {
                while let Some(res) = stream.session.next().await {
                    match res {
                        Ok(item) => tx
                            .send((stream.position, source_id.clone(), item))
                            .await
                            .unwrap_or_else(|e| {
                                let message = format!(
                                    "Failed to send item to channel for RTSP source {}, error: {:?}",
                                    source_id, e
                                );
                                panic!("{}", message);
                            }),
                        Err(e) => {
                            error!(
                                "Stream {} ended unexpectedly with error: {:?}",
                                source_id, e
                            );
                        }
                    }
                }
                error!("Stream {} ended unexpectedly", source_id);
            };
            tasks.spawn(task);
        }
        loop {
            select! {
                err = tasks.join_next() => {
                    if let Some(Err(e)) = err {
                        error!("Task ended with error: {:?}", e);
                    }
                    break;
                }
                Some((position, source_id, data_item)) = rx.recv() => {
                    debug!("Received item from {} {:?}", source_id, data_item);
                    self.process_item(position, source_id, data_item, sink.clone()).await?;
                }
            }
        }
        Ok(())
    }

    async fn process_item(
        &mut self,
        position: usize,
        source_id: String,
        data_item: CodecItem,
        sink: Arc<Mutex<JobWriter>>,
    ) -> anyhow::Result<()> {
        match data_item {
            CodecItem::VideoFrame(video_frame) => {
                if let Some(ntp_sync) = &mut self.ntp_sync {
                    if !ntp_sync.is_ready(&source_id) {
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

                let rtp_time = video_frame.timestamp().timestamp();

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
                if position != video_frame.stream_id() {
                    debug!(
                        "[SKIPPED] Received video frame from {} {:?}, size: {}, stream_id: {}, expected stream_id: {}",
                        source_id,
                        rtp_time,
                        size,
                        video_frame.stream_id(),
                        position
                    );
                    return Ok(());
                }
                debug!(
                    "Received video frame from {}, stream_id: {}, rtp_time: {}, size: {}",
                    source_id,
                    video_frame.stream_id(),
                    rtp_time,
                    size,
                );
                let stream_info = self
                    .stream_infos
                    .get(&(source_id.clone(), video_frame.stream_id()))
                    .unwrap();
                debug!("Stream info: {:?}", stream_info);

                let frame_rate = stream_info
                    .frame_rate
                    .map(|(num, den)| format!("{}/{}", den, num))
                    .unwrap_or("30/1".to_string());

                let frame_data = if matches!(stream_info.encoding.as_str(), "h264" | "hevc") {
                    Cow::Owned(convert_to_annexb(video_frame)?)
                } else {
                    Cow::Borrowed(video_frame.data())
                };

                let kf = self.is_keyframe(&frame_data, &source_id, rtp_time, stream_info);
                if kf {
                    debug!(
                        target: "retina_rtsp::service::keyframe_detector",
                        "Source_id: {}, RTP time: {}, Keyframe arrived",
                        source_id, rtp_time
                    );
                    self.active_streams
                        .entry(source_id.clone())
                        .or_insert_with(|| {
                            info!(
                                "Stream_id: {}, RTP time: {}, is active now",
                                source_id, rtp_time
                            );
                            ()
                        });
                    self.rtp_bases.entry(source_id.clone()).or_insert_with(|| {
                        info!("Stream_id: {}, PTS base is set to: {}", source_id, rtp_time);
                        rtp_time
                    });
                }

                if !self.active_streams.contains_key(&source_id) {
                    debug!(
                        "Stream_id: {}, RTP time: {}, is not active, skipping frame before the first keyframe is found",
                        source_id, rtp_time
                    );
                    return Ok(());
                }

                let pts_sec =
                    (rtp_time - self.rtp_bases[&source_id]) as f64 / stream_info.clock_rate as f64;
                let pts_millis = (pts_sec * TIME_BASE.1 as f64) as i64;

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
                    pts_millis,
                    Some(pts_millis),
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
                        let message = frame.to_message();
                        let _ = sink.lock().await.send_message(
                            &frame.get_source_id(),
                            &message,
                            &[&frame_data],
                        )?;
                    }
                } else {
                    let message = frame.to_message();
                    let _ = sink
                        .lock()
                        .await
                        .send_message(&source_id, &message, &[&frame_data])?;
                    debug!("Sent video frame to sink");
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
                    self.ntp_sync.as_mut().map(|s| {
                        s.add_rtp_mark(
                            &source_id,
                            t.timestamp(),
                            sr.ntp_timestamp(),
                            t.clock_rate().into(),
                        )
                    });

                    info!(
                        "RTCP Sender report source_id={} RTP is {} NTP is {}",
                        source_id,
                        t,
                        sr.ntp_timestamp().0
                    );
                }
            }
            _ => todo!(),
        }
        Ok(())
    }

    fn is_keyframe(
        &self,
        frame_data: &[u8],
        source_id: &str,
        rtp_time: i64,
        stream_info: &StreamInfo,
    ) -> bool {
        let mut kf = false;
        let mut cursor = Cursor::new(frame_data);
        if matches!(stream_info.encoding.as_str(), "h264") {
            while let Ok(nal) = H264Nalu::next(&mut cursor) {
                debug!(
                    target: "retina_rtsp::service::h264_parser",
                    "Stream_id: {}, RTP time: {}, NAL header: {:?}, offset: {}",
                    source_id, rtp_time, nal.header, nal.offset
                );
                if matches!(
                    nal.header.type_,
                    cros_codecs::codec::h264::parser::NaluType::SliceIdr
                ) {
                    kf = true;
                }
            }
        } else if matches!(stream_info.encoding.as_str(), "hevc") {
            while let Ok(nal) = H265Nalu::next(&mut cursor) {
                debug!(
                    target: "retina_rtsp::service::h265_parser",
                    "Stream_id: {}, RTP time: {}, NAL header: {:?}, offset: {}",
                    source_id, rtp_time, nal.header, nal.offset
                );
                if matches!(
                    nal.header.type_,
                    cros_codecs::codec::h265::parser::NaluType::IdrWRadl
                        | cros_codecs::codec::h265::parser::NaluType::IdrNLp
                        | cros_codecs::codec::h265::parser::NaluType::CraNut
                ) {
                    kf = true;
                }
            }
        } else {
            // MJPEG frames are always keyframes
            kf = true;
        }
        kf
    }
}

pub async fn run_group(
    conf: Arc<ServiceConfiguration>,
    group_name: String,
    session_group: Arc<SessionGroup>,
    sink: Arc<Mutex<JobWriter>>,
) -> anyhow::Result<()> {
    let mut service_group =
        RtspServiceGroup::new(conf.clone(), group_name.clone(), session_group).await?;
    service_group.play(sink).await?;
    Ok(())
}
