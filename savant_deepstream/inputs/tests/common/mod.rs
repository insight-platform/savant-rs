//! Shared helpers for `deepstream_inputs` integration tests (MP4 assets under `decoders/assets`).

#![allow(dead_code)]

use cros_codecs::codec::h264::parser::Nalu as H264Nalu;
use cros_codecs::codec::h264::parser::NaluType as H264NaluType;
use cros_codecs::codec::h265::parser::Nalu as H265Nalu;
use cros_codecs::codec::h265::parser::NaluType as H265NaluType;
use deepstream_decoders::prelude::*;
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
};
use savant_core::primitives::video_codec::VideoCodec;
use savant_gstreamer::Codec;
use serde::Deserialize;
use std::io::Cursor;
use std::path::PathBuf;
use std::time::Duration;

/// Initialize logging, GStreamer, and CUDA on GPU 0.
///
/// Integration tests assume that when CUDA initializes successfully, NVDEC
/// (`nvv4l2decoder`) and hardware JPEG (`nvjpegdec`) are available on the
/// target platform — no separate element-factory probes are performed.
pub fn init() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).expect("CUDA init failed");
}

/// MP4 / manifest assets live in the `deepstream_decoders` crate.
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

pub fn codec_to_str(c: Codec) -> &'static str {
    match c {
        Codec::H264 => "h264",
        Codec::Hevc => "hevc",
        Codec::Vp8 => "vp8",
        Codec::Vp9 => "vp9",
        Codec::Av1 => "av1",
        Codec::Jpeg => "jpeg",
        Codec::Png => "png",
        Codec::RawRgba => "raw_rgba",
        Codec::RawRgb => "raw_rgb",
        Codec::RawNv12 => "raw_nv12",
    }
}

/// Build a frame with GStreamer nanosecond time base (payload passed separately to `submit`).
#[allow(clippy::too_many_arguments)]
pub fn make_video_frame_ns(
    source_id: &str,
    codec_str: &str,
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
        Some(
            VideoCodec::from_name(codec_str)
                .unwrap_or_else(|| panic!("make_video_frame_ns: unknown codec {codec_str}")),
        ),
        keyframe,
        (1, 1_000_000_000),
        pts_ns,
        dts_ns,
        duration_ns,
    )
    .expect("make_video_frame_ns")
}

/// Same as [`make_video_frame_ns`] but timestamps are in `time_base` units (non-GStreamer).
#[allow(clippy::too_many_arguments)]
pub fn make_video_frame_scaled(
    source_id: &str,
    codec_str: &str,
    width: i64,
    height: i64,
    time_base: (i64, i64),
    pts: i64,
    dts: Option<i64>,
    duration: Option<i64>,
    keyframe: Option<bool>,
) -> VideoFrameProxy {
    VideoFrameProxy::new(
        source_id,
        (30, 1),
        width,
        height,
        VideoFrameContent::None,
        VideoFrameTranscodingMethod::Copy,
        Some(
            VideoCodec::from_name(codec_str)
                .unwrap_or_else(|| panic!("make_video_frame_scaled: unknown codec {codec_str}")),
        ),
        keyframe,
        time_base,
        pts,
        dts,
        duration,
    )
    .expect("make_video_frame_scaled")
}

pub fn split_annexb_nalus(data: &[u8], codec: &str) -> Vec<Vec<u8>> {
    let mut cur = Cursor::new(data);
    let mut out = Vec::new();
    match codec {
        "h264" => {
            while let Ok(nalu) = H264Nalu::next(&mut cur) {
                out.push(nalu.data.into_owned());
            }
        }
        "hevc" => {
            while let Ok(nalu) = H265Nalu::next(&mut cur) {
                out.push(nalu.data.into_owned());
            }
        }
        _ => panic!("split_annexb_nalus: unsupported codec {codec}"),
    }
    out
}

pub fn nal_payload_offset(nalu: &[u8]) -> usize {
    if nalu.len() >= 4 && nalu[..4] == [0, 0, 0, 1] {
        4
    } else if nalu.len() >= 3 && nalu[..3] == [0, 0, 1] {
        3
    } else {
        0
    }
}

pub fn is_vcl_nalu(codec: &str, nalu: &[u8]) -> bool {
    let mut c = Cursor::new(nalu);
    match codec {
        "h264" => {
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
        "hevc" => {
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

pub fn is_aud_nalu(codec: &str, nalu: &[u8]) -> bool {
    let mut c = Cursor::new(nalu);
    match codec {
        "h264" => H264Nalu::next(&mut c)
            .map(|n| matches!(n.header.type_, H264NaluType::AuDelimiter))
            .unwrap_or(false),
        "hevc" => H265Nalu::next(&mut c)
            .map(|n| matches!(n.header.type_, H265NaluType::AudNut))
            .unwrap_or(false),
        _ => false,
    }
}

pub fn group_nalus_to_access_units(codec: &str, nalus: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
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

pub const SUBMIT_TIMEOUT: Duration = Duration::from_secs(60);
