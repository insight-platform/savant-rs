//! Shared utilities for `deepstream_decoders` integration tests.

#![allow(dead_code)]

use cros_codecs::codec::h264::parser::Nalu as H264Nalu;
use cros_codecs::codec::h264::parser::NaluType as H264NaluType;
use cros_codecs::codec::h265::parser::Nalu as H265Nalu;
use cros_codecs::codec::h265::parser::NaluType as H265NaluType;
use deepstream_decoders::prelude::*;
use serde::Deserialize;
use std::io::Cursor;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;

// ── Initialisation / capability probes ──────────────────────────────

pub fn init() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).expect("CUDA init failed");
}

pub fn has_nvdec() -> bool {
    let _ = gstreamer::init();
    gstreamer::ElementFactory::find("nvv4l2decoder").is_some()
}

pub fn has_nvjpegdec() -> bool {
    let _ = gstreamer::init();
    gstreamer::ElementFactory::find("nvjpegdec").is_some()
}

pub fn has_nvenc() -> bool {
    nvidia_gpu_utils::has_nvenc(0).unwrap_or(false)
}

// ── Asset manifest ──────────────────────────────────────────────────

pub fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets")
}

#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub assets: Vec<AssetEntry>,
}

/// Unified asset entry — all fields present in any manifest variant are
/// covered.  Optional fields use `#[serde(default)]` so both the simple
/// (annexb) and extended (mp4) JSON schemas deserialise cleanly.
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
    // Extra fields present in the manifest that tests don't inspect:
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

// ── Buffer pool / transform helpers ─────────────────────────────────

pub fn make_rgba_pool(w: u32, h: u32) -> BufferGenerator {
    use deepstream_buffers::NvBufSurfaceMemType;
    BufferGenerator::builder(VideoFormat::RGBA, w, h)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(2)
        .max_buffers(4)
        .build()
        .expect("RGBA pool creation failed")
}

pub fn identity_transform_config() -> TransformConfig {
    TransformConfig::default()
}

// ── Decoder event drain ─────────────────────────────────────────────

/// Drain decoder events, calling `on_frame` for each decoded frame.
/// The frame is dropped when `on_frame` returns, releasing pool buffers
/// immediately so the internal pool (size 4) is never exhausted.
pub fn drain_decoder(rx: &mpsc::Receiver<DecoderEvent>, mut on_frame: impl FnMut(DecodedFrame)) {
    loop {
        match rx.recv_timeout(Duration::from_secs(30)) {
            Ok(DecoderEvent::Frame(f)) => on_frame(f),
            Ok(DecoderEvent::Eos) => break,
            Ok(DecoderEvent::Error(e)) => panic!("decoder error: {e}"),
            Ok(DecoderEvent::PipelineRestarted { reason, .. }) => {
                panic!("unexpected restart: {reason}")
            }
            Err(_) => panic!("timeout waiting for decoder events"),
        }
    }
}

// ── Annex-B NALU parsing (via cros-codecs) ──────────────────────────

/// Split a concatenated Annex-B bitstream into individual NAL units,
/// each retaining its start-code prefix.
///
/// `codec` must be `"h264"` or `"hevc"`.
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

/// Byte offset of the first payload byte (after the start-code prefix).
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

/// Group individual Annex-B NAL units into access-unit byte sequences.
/// Each returned `Vec<u8>` is a concatenation of the original NALUs
/// (with start-code prefixes) that form a single AU.
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

// ── Annex-B → AVCC/HVCC conversion ─────────────────────────────────
//
// Uses a GStreamer conversion pipeline to reframe Annex-B access units
// into length-prefixed (AVCC/HVCC) format and extract codec_data.

use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;

/// Result of an Annex-B → AVCC/HVCC conversion.
pub struct AvccConversion {
    /// `AVCDecoderConfigurationRecord` or `HEVCDecoderConfigurationRecord`.
    pub codec_data: Vec<u8>,
    /// One entry per decoded access unit, each containing length-prefixed NALUs.
    pub access_units: Vec<Vec<u8>>,
}

/// Convert Annex-B access units into length-prefixed (AVCC / HVCC) framing.
///
/// Internally runs a small GStreamer pipeline
/// (`appsrc ! h264parse/h265parse ! capsfilter ! appsink`) to let the
/// parser do the heavy lifting of building the configuration record and
/// reframing NALUs.
pub fn convert_annexb_to_avcc(codec: &str, annexb_aus: &[Vec<u8>]) -> AvccConversion {
    assert!(
        matches!(codec, "h264" | "hevc"),
        "AVCC conversion only for h264/hevc"
    );
    let (media_type, parser_name, output_format) = match codec {
        "h264" => ("video/x-h264", "h264parse", "avc"),
        "hevc" => ("video/x-h265", "h265parse", "hvc1"),
        _ => unreachable!(),
    };

    let pipeline = gst::Pipeline::new();

    let appsrc = gst::ElementFactory::make("appsrc")
        .name("avcc-src")
        .build()
        .unwrap()
        .dynamic_cast::<gst_app::AppSrc>()
        .unwrap();
    let parser = gst::ElementFactory::make(parser_name)
        .name("avcc-parse")
        .build()
        .unwrap();
    let capsfilter = gst::ElementFactory::make("capsfilter")
        .name("avcc-caps")
        .build()
        .unwrap();
    let appsink = gst::ElementFactory::make("appsink")
        .name("avcc-sink")
        .build()
        .unwrap()
        .dynamic_cast::<gst_app::AppSink>()
        .unwrap();

    let input_caps = gst::Caps::builder(media_type)
        .field("stream-format", "byte-stream")
        .field("alignment", "au")
        .build();
    appsrc.set_property("caps", &input_caps);
    appsrc.set_property_from_str("format", "time");
    appsrc.set_property_from_str("stream-type", "stream");
    appsrc.set_property("is-live", false);

    let output_caps = gst::Caps::builder(media_type)
        .field("stream-format", output_format)
        .field("alignment", "au")
        .build();
    capsfilter.set_property("caps", &output_caps);

    appsink.set_property("sync", false);
    appsink.set_property("emit-signals", false);

    pipeline
        .add_many([
            appsrc.upcast_ref(),
            &parser,
            &capsfilter,
            appsink.upcast_ref(),
        ])
        .unwrap();
    gst::Element::link_many([
        appsrc.upcast_ref(),
        &parser,
        &capsfilter,
        appsink.upcast_ref(),
    ])
    .unwrap();
    pipeline.set_state(gst::State::Playing).unwrap();

    let dur = 33_333_333u64;
    for (i, au) in annexb_aus.iter().enumerate() {
        let pts = i as u64 * dur;
        let mut buffer = gst::Buffer::from_mut_slice(au.clone());
        {
            let buf = buffer.get_mut().unwrap();
            buf.set_pts(gst::ClockTime::from_nseconds(pts));
            buf.set_dts(gst::ClockTime::from_nseconds(pts));
            buf.set_duration(gst::ClockTime::from_nseconds(dur));
        }
        appsrc.push_buffer(buffer).unwrap();
    }
    appsrc.end_of_stream().unwrap();

    let mut avcc_aus = Vec::new();
    let mut codec_data: Option<Vec<u8>> = None;
    loop {
        match appsink.try_pull_sample(gst::ClockTime::from_seconds(5)) {
            Some(sample) => {
                if codec_data.is_none() {
                    if let Some(caps) = sample.caps() {
                        if let Some(st) = caps.structure(0) {
                            if let Ok(buf) = st.get::<gst::Buffer>("codec_data") {
                                let map = buf.map_readable().unwrap();
                                codec_data = Some(map.as_slice().to_vec());
                            }
                        }
                    }
                }
                let buf = sample.buffer().unwrap();
                let map = buf.map_readable().unwrap();
                avcc_aus.push(map.as_slice().to_vec());
            }
            None if appsink.is_eos() => break,
            None => break,
        }
    }
    pipeline.set_state(gst::State::Null).unwrap();

    AvccConversion {
        codec_data: codec_data.expect("h264parse/h265parse did not produce codec_data"),
        access_units: avcc_aus,
    }
}

// ── Decoder config constructors ─────────────────────────────────────

pub fn decoder_config_annexb(codec: &str) -> DecoderConfig {
    match codec {
        "h264" => DecoderConfig::H264(H264DecoderConfig::new(H264StreamFormat::ByteStream)),
        "hevc" => DecoderConfig::Hevc(HevcDecoderConfig::new(HevcStreamFormat::ByteStream)),
        _ => panic!("unsupported annexb codec: {codec}"),
    }
}

pub fn decoder_config_avcc(codec: &str, codec_data: Vec<u8>) -> DecoderConfig {
    match codec {
        "h264" => DecoderConfig::H264(
            H264DecoderConfig::new(H264StreamFormat::Avc).codec_data(codec_data),
        ),
        "hevc" => DecoderConfig::Hevc(
            HevcDecoderConfig::new(HevcStreamFormat::Hvc1).codec_data(codec_data),
        ),
        _ => panic!("unsupported avcc codec: {codec}"),
    }
}

pub fn decoder_config_for_codec(codec: &str) -> Option<DecoderConfig> {
    match codec {
        "h264" => Some(decoder_config_annexb("h264")),
        "hevc" => Some(decoder_config_annexb("hevc")),
        "vp8" => Some(DecoderConfig::Vp8(Vp8DecoderConfig::new())),
        "vp9" => Some(DecoderConfig::Vp9(Vp9DecoderConfig::new())),
        "av1" => Some(DecoderConfig::Av1(Av1DecoderConfig::new())),
        "jpeg" => Some(DecoderConfig::Jpeg(JpegDecoderConfig::default())),
        _ => None,
    }
}
