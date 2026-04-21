//! Shared helpers for `deepstream_inputs` integration tests (MP4 assets under `decoders/assets`).

#![allow(dead_code)]

use cros_codecs::codec::h264::parser::Nalu as H264Nalu;
use cros_codecs::codec::h264::parser::NaluType as H264NaluType;
use cros_codecs::codec::h265::parser::Nalu as H265Nalu;
use cros_codecs::codec::h265::parser::NaluType as H265NaluType;
use deepstream_decoders::cuda_init;
use deepstream_inputs::flexible_decoder::FlexibleDecoderOutput;
use parking_lot::Mutex;
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
};
use savant_core::primitives::video_codec::VideoCodec;
use savant_gstreamer::mp4_demuxer::{DemuxedPacket, Mp4Demuxer};
use serde::Deserialize;
use std::io::Cursor;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

// ── Initialisation ──────────────────────────────────────────────────

pub fn init() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).expect("CUDA init failed");
}

// ── Asset manifest ──────────────────────────────────────────────────

pub fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../decoders/assets")
}

#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub assets: Vec<AssetEntry>,
}

#[derive(Debug, Deserialize)]
pub struct AssetEntry {
    pub file: String,
    #[serde(default)]
    pub container: Option<String>,
    pub codec: String,
    #[serde(default)]
    pub stream_format: Option<String>,
    pub width: u32,
    pub height: u32,
    pub num_frames: u32,
    pub supported_platforms: Vec<String>,
    #[serde(default)]
    pub colorspace: Option<String>,
    #[serde(default)]
    pub bit_depth: Option<u32>,
    #[serde(default)]
    pub b_frames: Option<bool>,
    #[serde(default)]
    pub profile: Option<String>,
    #[serde(default)]
    pub fps: Option<u32>,
    #[serde(default)]
    pub pipeline: Option<String>,
}

pub fn load_manifest() -> Manifest {
    let path = assets_dir().join("manifest.json");
    let data = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&data).unwrap_or_else(|e| panic!("cannot parse {}: {e}", path.display()))
}

// ── Platform detection ──────────────────────────────────────────────

pub fn current_platform_tag() -> String {
    nvidia_gpu_utils::gpu_platform_tag(0).unwrap_or_else(|_| "unknown".to_string())
}

pub fn manifest_tag_matches_platform(manifest_tag: &str, platform_tag: &str) -> bool {
    match manifest_tag {
        "jetson_orin" => platform_tag.contains("orin"),
        other => platform_tag == other,
    }
}

pub fn asset_supported_on_platform(entry: &AssetEntry, platform_tag: &str) -> bool {
    entry
        .supported_platforms
        .iter()
        .any(|t| manifest_tag_matches_platform(t, platform_tag))
}

// ── VideoCodec mapping ───────────────────────────────────────────────────

pub fn codec_name_to_video_codec(name: &str) -> Option<VideoCodec> {
    VideoCodec::from_name(name)
}

// ── Frame construction ──────────────────────────────────────────────

/// Build a [`VideoFrameProxy`] with nanosecond time base (payload passed separately).
#[allow(clippy::too_many_arguments)]
pub fn make_video_frame_ns(
    source_id: &str,
    codec: VideoCodec,
    width: i64,
    height: i64,
    pts_ns: i64,
    dts_ns: Option<i64>,
    duration_ns: Option<i64>,
    keyframe: Option<bool>,
) -> VideoFrameProxy {
    VideoFrameProxy::new(
        source_id,
        (30, 1),
        width,
        height,
        VideoFrameContent::None,
        VideoFrameTranscodingMethod::Copy,
        Some(codec),
        keyframe,
        (1, 1_000_000_000),
        pts_ns,
        dts_ns,
        duration_ns,
    )
    .expect("make_video_frame_ns")
}

// ── Annex-B NALU parsing (via cros-codecs) ──────────────────────────

pub fn split_annexb_nalus(data: &[u8], codec: VideoCodec) -> Vec<Vec<u8>> {
    let mut cur = Cursor::new(data);
    let mut out = Vec::new();
    match codec {
        VideoCodec::H264 => {
            while let Ok(nalu) = H264Nalu::next(&mut cur) {
                out.push(nalu.data.into_owned());
            }
        }
        VideoCodec::Hevc => {
            while let Ok(nalu) = H265Nalu::next(&mut cur) {
                out.push(nalu.data.into_owned());
            }
        }
        other => panic!("split_annexb_nalus: unsupported codec {other:?}"),
    }
    out
}

pub fn is_vcl_nalu(codec: VideoCodec, nalu: &[u8]) -> bool {
    let mut c = Cursor::new(nalu);
    match codec {
        VideoCodec::H264 => {
            if let Ok(n) = H264Nalu::next(&mut c) {
                matches!(
                    n.header.type_,
                    H264NaluType::Slice
                        | H264NaluType::SliceDpa
                        | H264NaluType::SliceDpb
                        | H264NaluType::SliceDpc
                        | H264NaluType::SliceIdr
                )
            } else {
                false
            }
        }
        VideoCodec::Hevc => {
            if let Ok(n) = H265Nalu::next(&mut c) {
                matches!(
                    n.header.type_,
                    H265NaluType::TrailN
                        | H265NaluType::TrailR
                        | H265NaluType::TsaN
                        | H265NaluType::TsaR
                        | H265NaluType::StsaN
                        | H265NaluType::StsaR
                        | H265NaluType::RadlN
                        | H265NaluType::RadlR
                        | H265NaluType::RaslN
                        | H265NaluType::RaslR
                        | H265NaluType::RsvVclN10
                        | H265NaluType::RsvVclR11
                        | H265NaluType::RsvVclN12
                        | H265NaluType::RsvVclR13
                        | H265NaluType::RsvVclN14
                        | H265NaluType::RsvVclR15
                        | H265NaluType::BlaWLp
                        | H265NaluType::BlaWRadl
                        | H265NaluType::BlaNLp
                        | H265NaluType::IdrWRadl
                        | H265NaluType::IdrNLp
                        | H265NaluType::CraNut
                        | H265NaluType::RsvIrapVcl22
                        | H265NaluType::RsvIrapVcl23
                        | H265NaluType::RsvVcl24
                        | H265NaluType::RsvVcl25
                        | H265NaluType::RsvVcl26
                        | H265NaluType::RsvVcl27
                        | H265NaluType::RsvVcl28
                        | H265NaluType::RsvVcl29
                        | H265NaluType::RsvVcl30
                        | H265NaluType::RsvVcl31
                )
            } else {
                false
            }
        }
        _ => false,
    }
}

pub fn is_aud_nalu(codec: VideoCodec, nalu: &[u8]) -> bool {
    let mut c = Cursor::new(nalu);
    match codec {
        VideoCodec::H264 => H264Nalu::next(&mut c)
            .map(|n| matches!(n.header.type_, H264NaluType::AuDelimiter))
            .unwrap_or(false),
        VideoCodec::Hevc => H265Nalu::next(&mut c)
            .map(|n| matches!(n.header.type_, H265NaluType::AudNut))
            .unwrap_or(false),
        _ => false,
    }
}

pub fn group_nalus_to_access_units(codec: VideoCodec, nalus: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
    let has_aud = nalus.iter().any(|n| is_aud_nalu(codec, n));
    if has_aud {
        let mut out = Vec::new();
        let mut current = Vec::new();
        let mut seen_vcl = false;
        for nalu in nalus {
            if is_aud_nalu(codec, &nalu) && !current.is_empty() {
                if seen_vcl {
                    out.push(current);
                }
                current = Vec::new();
                seen_vcl = false;
            }
            if is_vcl_nalu(codec, &nalu) {
                seen_vcl = true;
            }
            current.extend_from_slice(&nalu);
        }
        if seen_vcl && !current.is_empty() {
            out.push(current);
        }
        return out;
    }

    let mut out = Vec::new();
    let mut current = Vec::new();
    let mut seen_vcl = false;
    for nalu in nalus {
        let nalu_is_vcl = is_vcl_nalu(codec, &nalu);
        if nalu_is_vcl && seen_vcl {
            out.push(current);
            current = Vec::new();
            seen_vcl = false;
        }
        current.extend_from_slice(&nalu);
        if nalu_is_vcl {
            seen_vcl = true;
        }
    }
    if seen_vcl && !current.is_empty() {
        out.push(current);
    }
    out
}

// ── MP4 demux helpers ───────────────────────────────────────────────

/// Uniform access unit produced by demuxing + NALU grouping.
pub struct AccessUnit {
    pub data: Vec<u8>,
    pub pts_ns: u64,
    pub dts_ns: Option<u64>,
    pub duration_ns: Option<u64>,
}

/// Demux an MP4 into Annex-B access units (H.264/HEVC) or raw packets
/// (JPEG, VP8, AV1, etc.) ready for FlexibleDecoder submission.
pub fn demux_mp4_to_access_units(entry: &AssetEntry) -> Vec<AccessUnit> {
    let mp4_path = assets_dir().join(&entry.file);
    let mp4_str = mp4_path.to_str().unwrap();

    let (packets, _info) = Mp4Demuxer::demux_all_parsed(mp4_str)
        .unwrap_or_else(|e| panic!("demuxer failed for {}: {e}", entry.file));

    assert!(
        !packets.is_empty(),
        "{}: demuxer produced 0 packets",
        entry.file
    );

    match entry.codec.as_str() {
        "h264" | "hevc" => demuxed_to_annexb_aus(entry, &packets),
        _ => packets_to_access_units(&packets),
    }
}

fn demuxed_to_annexb_aus(entry: &AssetEntry, packets: &[DemuxedPacket]) -> Vec<AccessUnit> {
    let gst_codec = match entry.codec.as_str() {
        "h264" => VideoCodec::H264,
        "hevc" => VideoCodec::Hevc,
        _ => unreachable!(),
    };
    let mut bytestream = Vec::new();
    for pkt in packets {
        bytestream.extend_from_slice(&pkt.data);
    }
    let nalus = split_annexb_nalus(&bytestream, gst_codec);
    let aus = group_nalus_to_access_units(gst_codec, nalus);

    let dur = 33_333_333u64;
    aus.into_iter()
        .enumerate()
        .map(|(i, data)| {
            let pts = i as u64 * dur;
            AccessUnit {
                data,
                pts_ns: pts,
                dts_ns: Some(pts),
                duration_ns: Some(dur),
            }
        })
        .collect()
}

fn packets_to_access_units(packets: &[DemuxedPacket]) -> Vec<AccessUnit> {
    packets
        .iter()
        .map(|pkt| AccessUnit {
            data: pkt.data.clone(),
            pts_ns: pkt.pts_ns,
            dts_ns: pkt.dts_ns,
            duration_ns: pkt.duration_ns,
        })
        .collect()
}

/// Load a raw Annex-B file and split into access units.
pub fn load_annexb_access_units(entry: &AssetEntry) -> Vec<AccessUnit> {
    let gst_codec = match entry.codec.as_str() {
        "h264" => VideoCodec::H264,
        "hevc" => VideoCodec::Hevc,
        _ => panic!(
            "load_annexb_access_units: unsupported codec {}",
            entry.codec
        ),
    };
    let path = assets_dir().join(&entry.file);
    let bitstream =
        std::fs::read(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let nalus = split_annexb_nalus(&bitstream, gst_codec);
    let aus = group_nalus_to_access_units(gst_codec, nalus);

    let dur = 33_333_333u64;
    aus.into_iter()
        .enumerate()
        .map(|(i, data)| {
            let pts = i as u64 * dur;
            AccessUnit {
                data,
                pts_ns: pts,
                dts_ns: Some(pts),
                duration_ns: Some(dur),
            }
        })
        .collect()
}

// ── Output collector ────────────────────────────────────────────────

/// Lightweight summary of a [`FlexibleDecoderOutput`].
///
/// GPU buffers (`SharedBuffer`) are dropped immediately in the callback
/// so the decoder's buffer pool is never exhausted.
#[derive(Debug, Clone)]
pub enum CollectedOutput {
    Frame {
        proxy_uuid: u128,
        frame_id: Option<u128>,
        pts_ns: u64,
        codec: VideoCodec,
    },
    ParameterChange {
        old_codec: VideoCodec,
        old_w: i64,
        old_h: i64,
        new_codec: VideoCodec,
        new_w: i64,
        new_h: i64,
    },
    Skipped {
        reason: String,
    },
    OrphanFrame {
        frame_id: Option<u128>,
    },
    SourceEos {
        source_id: String,
    },
    Event,
    Error(String),
}

impl CollectedOutput {
    fn from_output(out: &FlexibleDecoderOutput) -> Self {
        match out {
            FlexibleDecoderOutput::Frame {
                frame, decoded: df, ..
            } => {
                let proxy_uuid = frame.get_uuid_u128();
                assert_eq!(
                    Some(proxy_uuid),
                    df.frame_id,
                    "proxy UUID must match decoded frame_id"
                );
                CollectedOutput::Frame {
                    proxy_uuid,
                    frame_id: df.frame_id,
                    pts_ns: df.pts_ns,
                    codec: df.codec,
                }
            }
            FlexibleDecoderOutput::ParameterChange { old, new } => {
                CollectedOutput::ParameterChange {
                    old_codec: old.codec,
                    old_w: old.width,
                    old_h: old.height,
                    new_codec: new.codec,
                    new_w: new.width,
                    new_h: new.height,
                }
            }
            FlexibleDecoderOutput::Skipped { reason, .. } => CollectedOutput::Skipped {
                reason: format!("{reason:?}"),
            },
            FlexibleDecoderOutput::OrphanFrame { decoded: df } => CollectedOutput::OrphanFrame {
                frame_id: df.frame_id,
            },
            FlexibleDecoderOutput::SourceEos { source_id } => CollectedOutput::SourceEos {
                source_id: source_id.clone(),
            },
            FlexibleDecoderOutput::Event(_) => CollectedOutput::Event,
            FlexibleDecoderOutput::Error(e) => CollectedOutput::Error(format!("{e}")),
        }
    }
}

#[derive(Clone)]
pub struct OutputCollector {
    pub outputs: Arc<Mutex<Vec<CollectedOutput>>>,
}

impl OutputCollector {
    pub fn new() -> Self {
        Self {
            outputs: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn callback(&self) -> impl Fn(FlexibleDecoderOutput) + Send + Sync + 'static {
        let outputs = self.outputs.clone();
        move |out| outputs.lock().push(CollectedOutput::from_output(&out))
    }

    pub fn drain(&self) -> Vec<CollectedOutput> {
        std::mem::take(&mut *self.outputs.lock())
    }

    pub fn frame_count(&self) -> usize {
        self.outputs
            .lock()
            .iter()
            .filter(|o| matches!(o, CollectedOutput::Frame { .. }))
            .count()
    }

    pub fn error_count(&self) -> usize {
        self.outputs
            .lock()
            .iter()
            .filter(|o| matches!(o, CollectedOutput::Error(_)))
            .count()
    }

    pub fn skip_count(&self) -> usize {
        self.outputs
            .lock()
            .iter()
            .filter(|o| matches!(o, CollectedOutput::Skipped { .. }))
            .count()
    }

    pub fn frame_uuids(&self) -> Vec<u128> {
        self.outputs
            .lock()
            .iter()
            .filter_map(|o| match o {
                CollectedOutput::Frame { proxy_uuid, .. } => Some(*proxy_uuid),
                _ => None,
            })
            .collect()
    }

    pub fn parameter_change_count(&self) -> usize {
        self.outputs
            .lock()
            .iter()
            .filter(|o| matches!(o, CollectedOutput::ParameterChange { .. }))
            .count()
    }

    pub fn wait_for_frames(&self, count: usize, timeout: Duration) {
        let start = std::time::Instant::now();
        loop {
            let fc = self.frame_count();
            if fc >= count {
                return;
            }
            if start.elapsed() > timeout {
                let all = self.outputs.lock();
                panic!(
                    "timeout waiting for {count} frames (got {fc} after {:?}); collected {} outputs: {:?}",
                    start.elapsed(),
                    all.len(),
                    &*all
                );
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    pub fn wait_for<F>(&self, predicate: F, timeout: Duration)
    where
        F: Fn(&CollectedOutput) -> bool,
    {
        let start = std::time::Instant::now();
        loop {
            if self.outputs.lock().iter().any(&predicate) {
                return;
            }
            if start.elapsed() > timeout {
                let all = self.outputs.lock();
                panic!(
                    "timeout waiting for matching output after {:?}; collected {} outputs: {:?}",
                    start.elapsed(),
                    all.len(),
                    &*all
                );
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Assert that every UUID in `submitted` appears exactly once in Frame
    /// outputs and that there are no extra Frame outputs.
    pub fn assert_frame_uuid_coverage(&self, submitted: &[u128]) {
        let mut output_uuids = self.frame_uuids();
        output_uuids.sort();
        let mut expected = submitted.to_vec();
        expected.sort();
        assert_eq!(
            expected, output_uuids,
            "submitted UUIDs must exactly match Frame output UUIDs"
        );
    }
}
