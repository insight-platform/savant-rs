//! Parity integration test: `Mp4Demuxer` (ground truth) vs `UriDemuxer`.
//!
//! Both demuxers are run against the same generated MP4 and their outputs
//! compared. Strict equality is asserted for:
//! - packet count,
//! - `pts_ns`, `dts_ns`, `duration_ns`, `is_keyframe` per packet,
//! - `VideoInfo`,
//! - `detected_codec`.
//!
//! H.264/HEVC payload comparison is done on a **normalized** form: leading
//! configuration NAL units (AUD / VPS / SPS / PPS / SEI / filler / EoS / EoB)
//! are stripped via `cros-codecs` so the slice NAL remains. The raw SPS/PPS
//! prefix count differs legitimately between the two paths because
//! `UriDemuxer` uses `parsebin` (whose internal parser chain handles
//! SPS/PPS insertion slightly differently than `Mp4Demuxer`'s direct
//! `qtdemux -> h26{4,5}parse` chain). Downstream decoders handle both
//! formats identically, and the slice bytes (the actual encoded video
//! data) are byte-identical.
//!
//! For codecs that are **not** carried in Annex-B form by `parsed=true`
//! (`Av1`, `Vp8`, `Vp9`, `Jpeg`), payloads are compared byte-for-byte.

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cros_codecs::codec::h264::parser::{Nalu as H264Nalu, NaluType as H264NaluType};
use cros_codecs::codec::h265::parser::{Nalu as H265Nalu, NaluType as H265NaluType};
use gstreamer as gst;
use serde::Deserialize;

use savant_core::primitives::video_codec::VideoCodec;
use savant_gstreamer::demux::{DemuxedPacket, VideoInfo};
use savant_gstreamer::mp4_demuxer::Mp4Demuxer;
use savant_gstreamer::mp4_muxer::Mp4Muxer;
use savant_gstreamer::uri_demuxer::{UriDemuxer, UriDemuxerConfig, UriDemuxerOutput};

const H264_SPS_PPS_IDR: [u8; 32] = [
    0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x0A, 0xE9, 0x40, 0x40, 0x04, 0x00, 0x00, 0x00, 0x02,
    0x00, 0x00, 0x00, 0x01, 0x68, 0xCE, 0x38, 0x80, 0x00, 0x00, 0x00, 0x01, 0x65, 0x88, 0x80, 0x40,
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

fn collect_via_uri_demuxer(
    uri: &str,
) -> (Vec<DemuxedPacket>, Option<VideoInfo>, Option<VideoCodec>) {
    let packets: Arc<Mutex<Vec<DemuxedPacket>>> = Arc::new(Mutex::new(Vec::new()));
    let errored = Arc::new(AtomicBool::new(false));
    let packets_cb = packets.clone();
    let errored_cb = errored.clone();
    let demuxer = UriDemuxer::new(
        UriDemuxerConfig::new(uri).with_parsed(true),
        move |out| match out {
            UriDemuxerOutput::Packet(p) => packets_cb.lock().unwrap().push(p),
            UriDemuxerOutput::Error(e) => {
                errored_cb.store(true, Ordering::SeqCst);
                panic!("UriDemuxer error: {e}");
            }
            _ => {}
        },
    )
    .unwrap();
    assert!(
        demuxer.wait_timeout(Duration::from_secs(10)),
        "UriDemuxer timed out"
    );
    let info = demuxer.video_info();
    let codec = demuxer.detected_codec();
    let pkts = std::mem::take(&mut *packets.lock().unwrap());
    (pkts, info, codec)
}

/// Extract slice (VCL) NAL units from an Annex-B access unit, discarding
/// configuration NALs (parameter sets, SEI, AUD, filler, end-of-sequence /
/// end-of-bitstream markers, ...). Returns one `Vec<u8>` per VCL NALU,
/// including the NALU header byte but **not** the start code.
///
/// This normalisation is what lets us compare payloads from `Mp4Demuxer`
/// (which goes `qtdemux → h264parse/h265parse(-1)`) against `UriDemuxer`
/// (which goes `parsebin(internal h264parse/h265parse(-1)) → byte-stream
/// capsfilter`): both pipelines emit the same slice bytes, but they may
/// differ in how often / where they re-emit SPS/PPS/VPS/AUD/SEI.
fn slice_nals(codec: VideoCodec, au: &[u8]) -> Vec<Vec<u8>> {
    let mut cur = Cursor::new(au);
    let mut slices = Vec::new();
    match codec {
        VideoCodec::H264 => {
            while let Ok(nalu) = H264Nalu::next(&mut cur) {
                match nalu.header.type_ {
                    H264NaluType::AuDelimiter
                    | H264NaluType::Sps
                    | H264NaluType::Pps
                    | H264NaluType::Sei
                    | H264NaluType::FillerData
                    | H264NaluType::SeqEnd
                    | H264NaluType::StreamEnd
                    | H264NaluType::SpsExt
                    | H264NaluType::SubsetSps => {}
                    _ => slices.push(nalu.as_ref().to_vec()),
                }
            }
        }
        VideoCodec::Hevc => {
            while let Ok(nalu) = H265Nalu::next(&mut cur) {
                match nalu.header.type_ {
                    H265NaluType::VpsNut
                    | H265NaluType::SpsNut
                    | H265NaluType::PpsNut
                    | H265NaluType::AudNut
                    | H265NaluType::EosNut
                    | H265NaluType::EobNut
                    | H265NaluType::FdNut
                    | H265NaluType::PrefixSeiNut
                    | H265NaluType::SuffixSeiNut => {}
                    _ => slices.push(nalu.as_ref().to_vec()),
                }
            }
        }
        other => panic!("slice_nals: unsupported codec {other:?}"),
    }
    slices
}

#[test]
fn test_mp4_vs_uri_demuxer_h264_parity() {
    let _ = gst::init();
    let path = "/tmp/test_demuxer_parity_h264.mp4";
    let _ = std::fs::remove_file(path);
    make_h264_mp4(path, 10);

    // Ground truth via Mp4Demuxer.
    let (ref_packets, ref_info) = Mp4Demuxer::demux_all_parsed(path).expect("Mp4Demuxer failed");
    let ref_codec = ref_info.map(|i| i.codec);

    // Candidate via UriDemuxer.
    let (uri_packets, uri_info, uri_codec) = collect_via_uri_demuxer(&file_uri(path));

    // Strict parity on codec + VideoInfo.
    assert_eq!(ref_codec, uri_codec, "detected_codec mismatch");
    assert_eq!(ref_info, uri_info, "VideoInfo mismatch");

    // Strict parity on packet count.
    assert_eq!(
        ref_packets.len(),
        uri_packets.len(),
        "packet count mismatch: mp4={} uri={}",
        ref_packets.len(),
        uri_packets.len()
    );

    // Strict parity on per-packet timing/flags + semantic parity on payload.
    for (idx, (r, u)) in ref_packets.iter().zip(uri_packets.iter()).enumerate() {
        assert_eq!(r.pts_ns, u.pts_ns, "packet[{idx}] pts_ns differ");
        assert_eq!(
            r.is_keyframe, u.is_keyframe,
            "packet[{idx}] is_keyframe differ"
        );
        if let (Some(rd), Some(ud)) = (r.dts_ns, u.dts_ns) {
            assert_eq!(rd, ud, "packet[{idx}] dts_ns differ");
        }
        if let (Some(rd), Some(ud)) = (r.duration_ns, u.duration_ns) {
            assert_eq!(rd, ud, "packet[{idx}] duration_ns differ");
        }

        // Byte-level comparison of slice NAL units (configuration-agnostic).
        let r_slices = slice_nals(VideoCodec::H264, &r.data);
        let u_slices = slice_nals(VideoCodec::H264, &u.data);
        assert_eq!(
            r_slices, u_slices,
            "packet[{idx}] slice NAL units differ\nmp4:  {:?}\nuri:  {:?}",
            r.data, u.data
        );
        assert!(
            !r_slices.is_empty(),
            "packet[{idx}] has no slice NALs (mp4)"
        );
    }

    let _ = std::fs::remove_file(path);
}

// ── Manifest-driven sweep over `decoders/assets/` fixtures ──────────────────

/// Subset of `savant_deepstream/decoders/assets/manifest.json` used by these
/// parity tests. Fields we don't need are omitted; `serde(default)` keeps
/// the schema forward-compatible.
#[derive(Debug, Deserialize)]
struct Manifest {
    assets: Vec<AssetEntry>,
}

#[derive(Debug, Deserialize)]
struct AssetEntry {
    file: String,
    #[serde(default)]
    container: Option<String>,
    codec: String,
    #[serde(default)]
    stream_format: Option<String>,
    width: u32,
    height: u32,
    num_frames: u32,
}

/// Path to the shared fixtures directory:
/// `savant_gstreamer/savant_gstreamer/../../savant_deepstream/decoders/assets`.
fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../savant_deepstream/decoders/assets")
}

/// Load the asset manifest, or return `None` if it is absent (stripped
/// checkout). All other I/O / parse errors are panics so a malformed
/// manifest doesn't silently disable the sweep.
fn load_manifest() -> Option<Manifest> {
    let path = assets_dir().join("manifest.json");
    if !path.exists() {
        eprintln!(
            "demuxer_parity: skipping manifest sweep — {} not found",
            path.display()
        );
        return None;
    }
    let data = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let m: Manifest = serde_json::from_str(&data)
        .unwrap_or_else(|e| panic!("cannot parse {}: {e}", path.display()));
    Some(m)
}

fn manifest_codec_to_video_codec(name: &str) -> VideoCodec {
    match name {
        "h264" => VideoCodec::H264,
        "hevc" => VideoCodec::Hevc,
        "av1" => VideoCodec::Av1,
        "vp8" => VideoCodec::Vp8,
        "vp9" => VideoCodec::Vp9,
        "jpeg" => VideoCodec::Jpeg,
        other => panic!("unknown manifest codec '{other}'"),
    }
}

/// MP4 sweep: for every `container=mp4` fixture, assert `Mp4Demuxer` and
/// `UriDemuxer` produce the same VideoInfo, packet count, per-packet
/// timing/flags, and equivalent payloads.
#[test]
fn test_manifest_mp4_parity() {
    let _ = gst::init();
    let Some(manifest) = load_manifest() else {
        return;
    };

    let assets_dir = assets_dir();
    let mut tested = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for entry in manifest
        .assets
        .iter()
        .filter(|e| e.container.as_deref() == Some("mp4"))
    {
        let asset_path = assets_dir.join(&entry.file);
        if !asset_path.exists() {
            eprintln!(
                "demuxer_parity: SKIP missing fixture {}",
                asset_path.display()
            );
            continue;
        }
        let asset_path_str = asset_path.to_str().expect("non-UTF8 path");

        if let Err(e) = check_mp4_parity(entry, asset_path_str) {
            failures.push(format!("{}: {e}", entry.file));
        }
        tested += 1;
    }

    assert!(
        tested > 0,
        "demuxer_parity: no MP4 fixtures found under {}",
        assets_dir.display()
    );
    assert!(
        failures.is_empty(),
        "MP4 parity failed for {}/{} fixtures:\n  - {}",
        failures.len(),
        tested,
        failures.join("\n  - ")
    );
}

/// Run both demuxers on a single MP4 and assert parity. Returns `Err` with
/// a human-readable reason on mismatch so the caller can aggregate
/// failures across the whole manifest.
fn check_mp4_parity(entry: &AssetEntry, asset_path: &str) -> Result<(), String> {
    let expected_codec = manifest_codec_to_video_codec(&entry.codec);

    let (ref_packets, ref_info) =
        Mp4Demuxer::demux_all_parsed(asset_path).map_err(|e| format!("Mp4Demuxer failed: {e}"))?;
    let (uri_packets, uri_info, uri_codec) = collect_via_uri_demuxer(&file_uri(asset_path));
    let ref_codec = ref_info.map(|i| i.codec);

    if ref_codec != uri_codec {
        return Err(format!(
            "detected_codec mismatch: mp4={ref_codec:?} uri={uri_codec:?}"
        ));
    }
    if ref_info != uri_info {
        return Err(format!(
            "VideoInfo mismatch: mp4={ref_info:?} uri={uri_info:?}"
        ));
    }

    let info = ref_info.ok_or_else(|| "no VideoInfo emitted by Mp4Demuxer".to_string())?;
    if info.codec != expected_codec {
        return Err(format!(
            "codec mismatch vs manifest: got {:?}, expected {:?}",
            info.codec, expected_codec
        ));
    }
    if info.width != entry.width || info.height != entry.height {
        return Err(format!(
            "dimensions mismatch vs manifest: got {}x{}, expected {}x{}",
            info.width, info.height, entry.width, entry.height
        ));
    }

    if ref_packets.len() != uri_packets.len() {
        return Err(format!(
            "packet count mismatch: mp4={} uri={}",
            ref_packets.len(),
            uri_packets.len()
        ));
    }
    if (ref_packets.len() as u32) < entry.num_frames {
        return Err(format!(
            "packet count {} below manifest num_frames {}",
            ref_packets.len(),
            entry.num_frames
        ));
    }

    // Frame timestamps derived from non-integer ns framerates (e.g.
    // 1/30 s = 33_333_333.333… ns) may legitimately round to neighbouring
    // integers between the two parser paths. We tolerate up to 1 ns of
    // drift per accumulated frame on PTS/DTS, and ±1 ns on each per-frame
    // duration; anything larger is a real divergence.
    const NS_PER_FRAME_TOLERANCE: u64 = 1;

    let mut saw_keyframe = false;
    for (idx, (r, u)) in ref_packets.iter().zip(uri_packets.iter()).enumerate() {
        let ts_tol = (idx as u64 + 1) * NS_PER_FRAME_TOLERANCE;
        let pts_diff = r.pts_ns.abs_diff(u.pts_ns);
        if pts_diff > ts_tol {
            return Err(format!(
                "packet[{idx}] pts_ns differ by {pts_diff} ns (tolerance {ts_tol}): mp4={} uri={}",
                r.pts_ns, u.pts_ns
            ));
        }
        if r.is_keyframe != u.is_keyframe {
            return Err(format!(
                "packet[{idx}] is_keyframe differ: mp4={} uri={}",
                r.is_keyframe, u.is_keyframe
            ));
        }
        if r.is_keyframe {
            saw_keyframe = true;
        }
        if let (Some(rd), Some(ud)) = (r.dts_ns, u.dts_ns) {
            let diff = rd.abs_diff(ud);
            if diff > ts_tol {
                return Err(format!(
                    "packet[{idx}] dts_ns differ by {diff} ns (tolerance {ts_tol}): mp4={rd} uri={ud}"
                ));
            }
        }
        if let (Some(rd), Some(ud)) = (r.duration_ns, u.duration_ns) {
            let diff = rd.abs_diff(ud);
            if diff > NS_PER_FRAME_TOLERANCE {
                return Err(format!(
                    "packet[{idx}] duration_ns differ by {diff} ns: mp4={rd} uri={ud}"
                ));
            }
        }

        match info.codec {
            VideoCodec::H264 | VideoCodec::Hevc => {
                let r_slices = slice_nals(info.codec, &r.data);
                let u_slices = slice_nals(info.codec, &u.data);
                if r_slices != u_slices {
                    return Err(format!(
                        "packet[{idx}] slice NAL units differ\n  mp4: {:?}\n  uri: {:?}",
                        r.data, u.data
                    ));
                }
                if r_slices.is_empty() {
                    return Err(format!("packet[{idx}] has no slice NALs (mp4)"));
                }
            }
            VideoCodec::Av1 | VideoCodec::Vp8 | VideoCodec::Vp9 | VideoCodec::Jpeg => {
                if r.data != u.data {
                    return Err(format!(
                        "packet[{idx}] data bytes differ ({} vs {} bytes)",
                        r.data.len(),
                        u.data.len()
                    ));
                }
            }
            other => return Err(format!("unsupported codec for parity check: {other:?}")),
        }
    }

    if !saw_keyframe {
        return Err("no keyframe seen in either demuxer output".into());
    }

    Ok(())
}

/// Raw Annex-B sweep: `Mp4Demuxer` cannot ingest these (no qtdemux step),
/// so we only verify that `UriDemuxer` produces the expected codec /
/// dimensions and emits at least one VCL slice across all packets.
#[test]
fn test_manifest_raw_annexb_uri_demuxer() {
    let _ = gst::init();
    let Some(manifest) = load_manifest() else {
        return;
    };

    let assets_dir = assets_dir();
    let mut tested = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for entry in manifest
        .assets
        .iter()
        .filter(|e| e.container.as_deref() == Some("raw"))
    {
        let asset_path = assets_dir.join(&entry.file);
        if !asset_path.exists() {
            eprintln!(
                "demuxer_parity: SKIP missing fixture {}",
                asset_path.display()
            );
            continue;
        }
        let asset_path_str = asset_path.to_str().expect("non-UTF8 path");
        if let Err(e) = check_raw_annexb(entry, asset_path_str) {
            failures.push(format!("{}: {e}", entry.file));
        }
        tested += 1;
    }

    assert!(
        tested > 0,
        "demuxer_parity: no raw Annex-B fixtures found under {}",
        assets_dir.display()
    );
    assert!(
        failures.is_empty(),
        "raw Annex-B UriDemuxer failed for {}/{} fixtures:\n  - {}",
        failures.len(),
        tested,
        failures.join("\n  - ")
    );
}

fn check_raw_annexb(entry: &AssetEntry, asset_path: &str) -> Result<(), String> {
    let expected_codec = manifest_codec_to_video_codec(&entry.codec);
    if !matches!(expected_codec, VideoCodec::H264 | VideoCodec::Hevc) {
        return Err(format!(
            "raw Annex-B fixtures must be H.264 or HEVC, got {:?}",
            expected_codec
        ));
    }
    if entry.stream_format.as_deref() != Some("byte-stream") {
        return Err(format!(
            "expected stream_format=byte-stream, got {:?}",
            entry.stream_format
        ));
    }

    let (uri_packets, uri_info, uri_codec) = collect_via_uri_demuxer(&file_uri(asset_path));

    if uri_codec != Some(expected_codec) {
        return Err(format!(
            "detected_codec mismatch: got {uri_codec:?}, expected {expected_codec:?}"
        ));
    }
    let info = uri_info.ok_or_else(|| "no VideoInfo emitted by UriDemuxer".to_string())?;
    if info.codec != expected_codec {
        return Err(format!(
            "VideoInfo.codec mismatch: got {:?}, expected {:?}",
            info.codec, expected_codec
        ));
    }
    if info.width != entry.width || info.height != entry.height {
        return Err(format!(
            "dimensions mismatch vs manifest: got {}x{}, expected {}x{}",
            info.width, info.height, entry.width, entry.height
        ));
    }
    if uri_packets.is_empty() {
        return Err("UriDemuxer produced 0 packets".into());
    }

    // Concatenate all packets and assert at least one VCL slice — the
    // byte-stream capsfilter / parsebin chain must keep slice NALs intact.
    let mut bytestream = Vec::new();
    for pkt in &uri_packets {
        bytestream.extend_from_slice(&pkt.data);
    }
    let slices = slice_nals(expected_codec, &bytestream);
    if slices.is_empty() {
        return Err("no VCL slice NAL units recovered from UriDemuxer output".into());
    }

    Ok(())
}
