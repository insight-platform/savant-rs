//! End-to-end tests for [`detect_stream_config`] and [`is_random_access_point`]
//! against real asset files.
//!
//! Covers:
//! - Raw Annex-B `.h264` / `.h265` files → detected as `ByteStream`
//! - MP4-demuxed H264/HEVC packets (AVCC/HVCC via `Mp4Demuxer`) → detected
//!   as `Avc` / `Hvc1` with valid `codec_data`
//! - Raw file prefix detection (first 4 KiB)
//! - Random access points: first Annex-B AU is a RAP; MP4 keyframe packets are RAP;
//!   non-keyframes in B-frame assets are not RAP

mod common;

use common::*;
use deepstream_decoders::prelude::*;
use savant_gstreamer::mp4_demuxer::Mp4Demuxer;
use std::time::Duration;

fn codec_from_str(s: &str) -> Option<Codec> {
    match s {
        "h264" => Some(Codec::H264),
        "hevc" => Some(Codec::Hevc),
        _ => None,
    }
}

// ── Annex-B detection (raw .h264 / .h265 files) ────────────────────

fn run_detect_annexb(entry: &AssetEntry) {
    let file_path = assets_dir().join(&entry.file);
    let bitstream = std::fs::read(&file_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", file_path.display()));

    let nalus = split_annexb_nalus(&bitstream, &entry.codec);
    assert!(
        !nalus.is_empty(),
        "{}: no Annex-B NAL units found",
        entry.file
    );

    let aus = group_nalus_to_access_units(&entry.codec, nalus);
    assert!(!aus.is_empty(), "{}: no access units", entry.file);

    let codec = codec_from_str(&entry.codec).unwrap();
    let first_au = &aus[0];

    let cfg = detect_stream_config(codec, first_au)
        .unwrap_or_else(|| panic!("{}: detect_stream_config returned None", entry.file));

    match (&entry.codec[..], cfg) {
        ("h264", DecoderConfig::H264(c)) => {
            assert_eq!(
                c.stream_format,
                H264StreamFormat::ByteStream,
                "{}: expected ByteStream",
                entry.file
            );
            assert!(
                c.codec_data.is_none(),
                "{}: Annex-B should have no codec_data",
                entry.file
            );
        }
        ("hevc", DecoderConfig::Hevc(c)) => {
            assert_eq!(
                c.stream_format,
                HevcStreamFormat::ByteStream,
                "{}: expected ByteStream",
                entry.file
            );
            assert!(
                c.codec_data.is_none(),
                "{}: Annex-B should have no codec_data",
                entry.file
            );
        }
        (codec_str, other) => panic!(
            "{}: unexpected config variant for codec '{}': {:?}",
            entry.file, codec_str, other
        ),
    }

    eprintln!("  OK detect annexb  {}", entry.file);
}

#[test]
fn test_detect_stream_config_annexb_all_assets() {
    init();
    let manifest = load_manifest();
    let mut tested = 0usize;

    for entry in &manifest.assets {
        if entry.container.as_deref() != Some("raw") {
            continue;
        }
        if !matches!(entry.codec.as_str(), "h264" | "hevc") {
            continue;
        }
        run_detect_annexb(entry);
        tested += 1;
    }

    assert!(tested > 0, "no raw annexb assets were tested");
    eprintln!("detect_stream_config annexb: {tested} assets OK");
}

// ── Random access points (Annex-B first AU) ─────────────────────────

#[test]
fn test_rap_annexb_first_au_all_assets() {
    init();
    let manifest = load_manifest();
    let mut tested = 0usize;

    for entry in &manifest.assets {
        if entry.container.as_deref() != Some("raw") {
            continue;
        }
        if !matches!(entry.codec.as_str(), "h264" | "hevc") {
            continue;
        }
        let file_path = assets_dir().join(&entry.file);
        let bitstream = std::fs::read(&file_path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", file_path.display()));
        let nalus = split_annexb_nalus(&bitstream, &entry.codec);
        let aus = group_nalus_to_access_units(&entry.codec, nalus);
        let codec = codec_from_str(&entry.codec).unwrap();
        assert!(
            is_random_access_point(codec, &aus[0]),
            "{}: first access unit should be a random access point",
            entry.file
        );
        tested += 1;
        eprintln!("  OK rap annexb    {}", entry.file);
    }

    assert!(tested > 0, "no raw annexb assets were tested for RAP");
    eprintln!("is_random_access_point annexb first AU: {tested} assets OK");
}

// ── AVCC / HVCC detection via Mp4Demuxer ────────────────────────────
//
// Uses `Mp4Demuxer::new()` (no parser) to pull raw container-format
// packets from the MP4 files. For H264/HEVC these are length-prefixed
// (AVCC/HVCC), so detect_stream_config should identify them as
// Avc / Hvc1 with codec_data when the AU contains param sets.

fn run_detect_mp4_demuxed(entry: &AssetEntry) {
    let mp4_path = assets_dir().join(&entry.file);
    let mp4_str = mp4_path.to_str().unwrap();

    let mut demuxer = Mp4Demuxer::new(mp4_str)
        .unwrap_or_else(|e| panic!("demuxer failed for {}: {e}", entry.file));

    let mut packets = Vec::new();
    loop {
        match demuxer.pull_timeout(Duration::from_secs(5)) {
            Ok(Some(pkt)) => packets.push(pkt),
            Ok(None) => break,
            Err(e) => panic!("demuxer pull error for {}: {e}", entry.file),
        }
    }
    let detected_codec = demuxer.detected_codec();
    demuxer.finish();

    assert!(
        !packets.is_empty(),
        "{}: demuxer produced 0 packets",
        entry.file
    );

    let codec =
        detected_codec.unwrap_or_else(|| panic!("{}: demuxer did not detect codec", entry.file));

    // Try every packet until we get a successful detection — the first
    // keyframe with in-band parameter sets is what we're after.
    let mut detected = false;
    for (i, pkt) in packets.iter().enumerate() {
        if let Some(cfg) = detect_stream_config(codec, &pkt.data) {
            match (codec, &cfg) {
                (Codec::H264, DecoderConfig::H264(c)) => {
                    assert_eq!(
                        c.stream_format,
                        H264StreamFormat::Avc,
                        "{} pkt#{i}: expected Avc",
                        entry.file
                    );
                    let cd = c.codec_data.as_ref().unwrap_or_else(|| {
                        panic!("{} pkt#{i}: Avc should have codec_data", entry.file)
                    });
                    assert!(
                        cd.len() > 6,
                        "{} pkt#{i}: codec_data too short ({})",
                        entry.file,
                        cd.len()
                    );
                    assert_eq!(
                        cd[0], 1,
                        "{} pkt#{i}: AVCConfigurationRecord version must be 1",
                        entry.file
                    );
                }
                (Codec::Hevc, DecoderConfig::Hevc(c)) => {
                    assert_eq!(
                        c.stream_format,
                        HevcStreamFormat::Hvc1,
                        "{} pkt#{i}: expected Hvc1",
                        entry.file
                    );
                    let cd = c.codec_data.as_ref().unwrap_or_else(|| {
                        panic!("{} pkt#{i}: Hvc1 should have codec_data", entry.file)
                    });
                    assert!(
                        cd.len() > 23,
                        "{} pkt#{i}: codec_data too short ({})",
                        entry.file,
                        cd.len()
                    );
                    assert_eq!(
                        cd[0], 1,
                        "{} pkt#{i}: HEVCConfigurationRecord version must be 1",
                        entry.file
                    );
                }
                _ => panic!(
                    "{} pkt#{i}: mismatched codec {:?} vs config {:?}",
                    entry.file, codec, cfg
                ),
            }
            detected = true;
            eprintln!(
                "  OK detect mp4     {} (pkt#{i}, keyframe={})",
                entry.file, pkt.is_keyframe
            );
            break;
        }
    }

    assert!(
        detected,
        "{}: detect_stream_config returned None for all {} demuxed packets",
        entry.file,
        packets.len()
    );
}

#[test]
fn test_detect_stream_config_mp4_demuxed_all_assets() {
    init();
    let manifest = load_manifest();
    let platform_tag = current_platform_tag();
    let mut tested = 0usize;

    for entry in &manifest.assets {
        if entry.container.as_deref() != Some("mp4") {
            continue;
        }
        if !matches!(entry.codec.as_str(), "h264" | "hevc") {
            continue;
        }
        if !asset_supported_on_platform(entry, &platform_tag) {
            eprintln!("  SKIP (platform) {}", entry.file);
            continue;
        }
        run_detect_mp4_demuxed(entry);
        tested += 1;
    }

    assert!(
        tested > 0,
        "no MP4 h264/hevc assets were tested for stream detection"
    );
    eprintln!("detect_stream_config mp4 demuxed: {tested} assets OK");
}

// ── Random access points (MP4 demuxed packets) ──────────────────────

fn run_rap_mp4_demuxed(entry: &AssetEntry) {
    let mp4_path = assets_dir().join(&entry.file);
    let mp4_str = mp4_path.to_str().unwrap();

    let mut demuxer = Mp4Demuxer::new(mp4_str)
        .unwrap_or_else(|e| panic!("demuxer failed for {}: {e}", entry.file));

    let mut packets = Vec::new();
    loop {
        match demuxer.pull_timeout(Duration::from_secs(5)) {
            Ok(Some(pkt)) => packets.push(pkt),
            Ok(None) => break,
            Err(e) => panic!("demuxer pull error for {}: {e}", entry.file),
        }
    }
    let codec = demuxer
        .detected_codec()
        .unwrap_or_else(|| panic!("{}: no codec", entry.file));
    demuxer.finish();

    let mut first_rap: Option<usize> = None;
    for (i, pkt) in packets.iter().enumerate() {
        if is_random_access_point(codec, &pkt.data) {
            first_rap = Some(i);
            assert!(
                pkt.is_keyframe,
                "{} pkt#{i}: RAP packet should be reported as keyframe",
                entry.file
            );
            break;
        }
    }

    let i = first_rap.unwrap_or_else(|| {
        panic!(
            "{}: no packet was a random access point ({} packets)",
            entry.file,
            packets.len()
        )
    });

    if entry.b_frames == Some(true) && packets.len() > 1 {
        let non_rap = packets
            .iter()
            .enumerate()
            .any(|(j, p)| j != i && !is_random_access_point(codec, &p.data));
        assert!(
            non_rap,
            "{}: expected at least one non-RAP packet in B-frame asset",
            entry.file
        );
    }

    eprintln!(
        "  OK rap mp4       {} (first_rap=pkt#{i}, total={})",
        entry.file,
        packets.len()
    );
}

#[test]
fn test_rap_mp4_demuxed_all_assets() {
    init();
    let manifest = load_manifest();
    let platform_tag = current_platform_tag();
    let mut tested = 0usize;

    for entry in &manifest.assets {
        if entry.container.as_deref() != Some("mp4") {
            continue;
        }
        if !matches!(entry.codec.as_str(), "h264" | "hevc") {
            continue;
        }
        if !asset_supported_on_platform(entry, &platform_tag) {
            eprintln!("  SKIP (platform) {}", entry.file);
            continue;
        }
        run_rap_mp4_demuxed(entry);
        tested += 1;
    }

    assert!(
        tested > 0,
        "no MP4 h264/hevc assets were tested for random access points"
    );
    eprintln!("is_random_access_point mp4 demuxed: {tested} assets OK");
}

// ── Whole-file first-bytes detection ────────────────────────────────
//
// Feed the first N bytes of the raw file directly (no AU splitting)
// to verify the function handles a raw file prefix correctly.

#[test]
fn test_detect_stream_config_raw_file_prefix() {
    init();
    let manifest = load_manifest();
    let mut tested = 0usize;

    for entry in &manifest.assets {
        if entry.container.as_deref() != Some("raw") {
            continue;
        }
        let codec = match codec_from_str(&entry.codec) {
            Some(c) => c,
            None => continue,
        };

        let file_path = assets_dir().join(&entry.file);
        let bitstream = std::fs::read(&file_path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", file_path.display()));

        let prefix_len = bitstream.len().min(4096);
        let cfg = detect_stream_config(codec, &bitstream[..prefix_len]).unwrap_or_else(|| {
            panic!(
                "{}: detect_stream_config returned None for first {} bytes",
                entry.file, prefix_len
            )
        });

        match (&entry.codec[..], cfg) {
            ("h264", DecoderConfig::H264(c)) => {
                assert_eq!(c.stream_format, H264StreamFormat::ByteStream);
            }
            ("hevc", DecoderConfig::Hevc(c)) => {
                assert_eq!(c.stream_format, HevcStreamFormat::ByteStream);
            }
            _ => panic!("{}: unexpected config variant", entry.file),
        }

        tested += 1;
        eprintln!("  OK detect prefix  {}", entry.file);
    }

    assert!(tested > 0, "no raw assets tested for prefix detection");
    eprintln!("detect_stream_config raw prefix: {tested} assets OK");
}
