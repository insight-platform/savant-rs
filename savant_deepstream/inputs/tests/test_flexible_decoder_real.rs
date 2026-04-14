//! Real-codec integration tests for [`FlexibleDecoder`].
//!
//! Exercises the full decode pipeline with actual H.264, HEVC, AV1, and JPEG
//! bitstreams from the shared `decoders/assets` directory.  Tests cover:
//!
//! - Single-codec decode (each codec independently).
//! - Codec changes: H.264↔HEVC, JPEG→H.264, H.264→JPEG, H.264→AV1.
//! - Multi-codec rotation: JPEG → H.264 → HEVC → JPEG.
//! - Source EOS between codec switches.
//! - Graceful shutdown while actively decoding real bitstreams.

mod common;

use common::*;
use deepstream_inputs::flexible_decoder::{
    FlexibleDecoder, FlexibleDecoderConfig, FlexibleDecoderOutput, SealedDelivery,
};
use serial_test::serial;
use std::sync::Arc;
use std::time::Duration;

const SOURCE_ID: &str = "real-codec-test";
const TIMEOUT: Duration = Duration::from_secs(30);

fn default_config() -> FlexibleDecoderConfig {
    FlexibleDecoderConfig::new(SOURCE_ID, 0, 4)
        .idle_timeout(Duration::from_secs(10))
        .detect_buffer_limit(60)
}

/// Find a manifest entry by file name, assert it exists and is supported on this platform.
/// Returns `None` (skip) when the platform doesn't support the entry.
fn find_entry<'a>(manifest: &'a Manifest, file: &str) -> Option<&'a AssetEntry> {
    let entry = manifest
        .assets
        .iter()
        .find(|e| e.file == file)
        .unwrap_or_else(|| panic!("asset '{file}' not found in manifest"));
    let platform = current_platform_tag();
    if asset_supported_on_platform(entry, &platform) {
        Some(entry)
    } else {
        eprintln!("  SKIP (platform '{platform}'): {file}");
        None
    }
}

/// Submit a slice of [`AccessUnit`]s to the decoder, wrapping each in a
/// [`VideoFrameProxy`] with correct metadata.
///
/// Returns the UUIDs of every submitted [`VideoFrameProxy`] (in submission order).
fn submit_access_units(
    dec: &FlexibleDecoder,
    aus: &[AccessUnit],
    entry: &AssetEntry,
    limit: usize,
    pts_offset_ns: u64,
) -> Vec<u128> {
    submit_access_units_with_dims(
        dec,
        aus,
        &entry.codec,
        entry.width as i64,
        entry.height as i64,
        limit,
        pts_offset_ns,
    )
}

/// Like [`submit_access_units`] but with explicit codec name and dimensions,
/// allowing tests to inject mismatched metadata.
///
/// Returns the UUIDs of every submitted [`VideoFrameProxy`] (in submission order).
fn submit_access_units_with_dims(
    dec: &FlexibleDecoder,
    aus: &[AccessUnit],
    codec_name: &str,
    width: i64,
    height: i64,
    limit: usize,
    pts_offset_ns: u64,
) -> Vec<u128> {
    let vc = codec_name_to_video_codec(codec_name)
        .unwrap_or_else(|| panic!("unknown codec: {codec_name}"));
    let mut uuids = Vec::with_capacity(limit);
    for (i, au) in aus.iter().take(limit).enumerate() {
        let pts = pts_offset_ns + au.pts_ns;
        let dts = au.dts_ns.map(|d| (pts_offset_ns + d) as i64);
        let dur = au.duration_ns.map(|d| d as i64);
        let frame = make_video_frame_ns(SOURCE_ID, vc, width, height, pts as i64, dts, dur, None);
        uuids.push(frame.get_uuid_u128());
        dec.submit(&frame, Some(&au.data))
            .unwrap_or_else(|e| panic!("submit failed at AU {i}: {e}"));
    }
    uuids
}

// ═══════════════════════════════════════════════════════════════════
//  Single-codec decode tests
// ═══════════════════════════════════════════════════════════════════

#[test]
#[serial]
fn test_h264_mp4_decode() {
    init();
    let manifest = load_manifest();
    let Some(entry) = find_entry(&manifest, "test_h264_bt709_ip.mp4") else {
        return;
    };

    let collector = OutputCollector::new();
    let mut dec = FlexibleDecoder::new(default_config(), collector.callback());
    let aus = demux_mp4_to_access_units(entry);
    let num = aus.len().min(entry.num_frames as usize);
    let submitted = submit_access_units(&dec, &aus, entry, num, 0);
    assert!(!submitted.is_empty(), "no frames submitted for H.264 MP4");

    collector.wait_for_frames(submitted.len(), TIMEOUT);
    assert_eq!(
        collector.frame_count(),
        submitted.len(),
        "H.264 MP4: decoded != submitted"
    );
    assert_eq!(
        collector.error_count(),
        0,
        "H.264 MP4: unexpected decoder errors"
    );
    collector.assert_frame_uuid_coverage(&submitted);

    dec.graceful_shutdown().unwrap();
    eprintln!("  OK test_h264_mp4_decode: {} frames", submitted.len());
}

#[test]
#[serial]
fn test_h264_annexb_decode() {
    init();
    let manifest = load_manifest();
    let Some(entry) = find_entry(&manifest, "test_h264_annexb_ip.h264") else {
        return;
    };

    let collector = OutputCollector::new();
    let mut dec = FlexibleDecoder::new(default_config(), collector.callback());
    let aus = load_annexb_access_units(entry);
    let num = aus.len().min(entry.num_frames as usize);
    let submitted = submit_access_units(&dec, &aus, entry, num, 0);

    collector.wait_for_frames(submitted.len(), TIMEOUT);
    assert_eq!(collector.frame_count(), submitted.len());
    assert_eq!(collector.error_count(), 0);
    collector.assert_frame_uuid_coverage(&submitted);

    dec.graceful_shutdown().unwrap();
    eprintln!("  OK test_h264_annexb_decode: {} frames", submitted.len());
}

#[test]
#[serial]
fn test_hevc_mp4_decode() {
    init();
    let manifest = load_manifest();
    let Some(entry) = find_entry(&manifest, "test_hevc_bt709_ip.mp4") else {
        return;
    };

    let collector = OutputCollector::new();
    let mut dec = FlexibleDecoder::new(default_config(), collector.callback());
    let aus = demux_mp4_to_access_units(entry);
    let num = aus.len().min(entry.num_frames as usize);
    let submitted = submit_access_units(&dec, &aus, entry, num, 0);

    collector.wait_for_frames(submitted.len(), TIMEOUT);
    assert_eq!(collector.frame_count(), submitted.len());
    assert_eq!(collector.error_count(), 0);
    collector.assert_frame_uuid_coverage(&submitted);

    dec.graceful_shutdown().unwrap();
    eprintln!("  OK test_hevc_mp4_decode: {} frames", submitted.len());
}

#[test]
#[serial]
fn test_hevc_annexb_decode() {
    init();
    let manifest = load_manifest();
    let Some(entry) = find_entry(&manifest, "test_hevc_annexb_ip.h265") else {
        return;
    };

    let collector = OutputCollector::new();
    let mut dec = FlexibleDecoder::new(default_config(), collector.callback());
    let aus = load_annexb_access_units(entry);
    let num = aus.len().min(entry.num_frames as usize);
    let submitted = submit_access_units(&dec, &aus, entry, num, 0);

    collector.wait_for_frames(submitted.len(), TIMEOUT);
    assert_eq!(collector.frame_count(), submitted.len());
    assert_eq!(collector.error_count(), 0);
    collector.assert_frame_uuid_coverage(&submitted);

    dec.graceful_shutdown().unwrap();
    eprintln!("  OK test_hevc_annexb_decode: {} frames", submitted.len());
}

#[test]
#[serial]
fn test_av1_mp4_decode() {
    init();
    let manifest = load_manifest();
    let Some(entry) = find_entry(&manifest, "test_av1_bt709.mp4") else {
        return;
    };

    let collector = OutputCollector::new();
    let mut dec = FlexibleDecoder::new(default_config(), collector.callback());
    let aus = demux_mp4_to_access_units(entry);
    let num = aus.len().min(entry.num_frames as usize);
    let submitted = submit_access_units(&dec, &aus, entry, num, 0);

    collector.wait_for_frames(submitted.len(), TIMEOUT);
    assert_eq!(collector.frame_count(), submitted.len());
    assert_eq!(collector.error_count(), 0);
    collector.assert_frame_uuid_coverage(&submitted);

    dec.graceful_shutdown().unwrap();
    eprintln!("  OK test_av1_mp4_decode: {} frames", submitted.len());
}

#[test]
#[serial]
fn test_jpeg_mp4_decode() {
    init();
    let manifest = load_manifest();
    let Some(entry) = find_entry(&manifest, "test_jpeg_bt709.mp4") else {
        return;
    };

    let collector = OutputCollector::new();
    let mut dec = FlexibleDecoder::new(default_config(), collector.callback());
    let aus = demux_mp4_to_access_units(entry);
    let num = aus.len().min(entry.num_frames as usize);
    let submitted = submit_access_units(&dec, &aus, entry, num, 0);

    collector.wait_for_frames(submitted.len(), TIMEOUT);
    assert_eq!(collector.frame_count(), submitted.len());
    assert_eq!(collector.error_count(), 0);
    collector.assert_frame_uuid_coverage(&submitted);

    dec.graceful_shutdown().unwrap();
    eprintln!("  OK test_jpeg_mp4_decode: {} frames", submitted.len());
}

// ═══════════════════════════════════════════════════════════════════
//  Codec-change tests
// ═══════════════════════════════════════════════════════════════════

#[test]
#[serial]
fn test_h264_to_hevc_codec_change() {
    init();
    let manifest = load_manifest();
    let Some(h264_entry) = find_entry(&manifest, "test_h264_annexb_ip.h264") else {
        return;
    };
    let Some(hevc_entry) = find_entry(&manifest, "test_hevc_annexb_ip.h265") else {
        return;
    };

    let collector = OutputCollector::new();
    let mut dec = FlexibleDecoder::new(default_config(), collector.callback());

    let h264_aus = load_annexb_access_units(h264_entry);
    let h264_count = 4.min(h264_aus.len());
    let mut all_uuids = submit_access_units(&dec, &h264_aus, h264_entry, h264_count, 0);
    collector.wait_for_frames(all_uuids.len(), TIMEOUT);
    eprintln!("  phase 1: {} H.264 frames decoded", all_uuids.len());

    let hevc_aus = load_annexb_access_units(hevc_entry);
    let hevc_count = 4.min(hevc_aus.len());
    let pts_offset = (h264_count as u64) * 33_333_333;
    let hevc_uuids = submit_access_units(&dec, &hevc_aus, hevc_entry, hevc_count, pts_offset);
    all_uuids.extend(&hevc_uuids);

    collector.wait_for(
        |o| matches!(o, CollectedOutput::ParameterChange { .. }),
        TIMEOUT,
    );
    collector.wait_for_frames(all_uuids.len(), TIMEOUT);

    assert_eq!(collector.parameter_change_count(), 1);
    assert_eq!(collector.frame_count(), all_uuids.len());
    assert_eq!(collector.error_count(), 0);
    collector.assert_frame_uuid_coverage(&all_uuids);

    dec.graceful_shutdown().unwrap();
    eprintln!(
        "  OK test_h264_to_hevc: {} frames, 1 param change",
        all_uuids.len()
    );
}

#[test]
#[serial]
fn test_jpeg_to_h264_codec_change() {
    init();
    let manifest = load_manifest();
    let Some(jpeg_entry) = find_entry(&manifest, "test_jpeg_bt709.mp4") else {
        return;
    };
    let Some(h264_entry) = find_entry(&manifest, "test_h264_annexb_ip.h264") else {
        return;
    };

    let collector = OutputCollector::new();
    let mut dec = FlexibleDecoder::new(default_config(), collector.callback());

    let jpeg_aus = demux_mp4_to_access_units(jpeg_entry);
    let jpeg_count = 3.min(jpeg_aus.len());
    let mut all_uuids = submit_access_units(&dec, &jpeg_aus, jpeg_entry, jpeg_count, 0);
    collector.wait_for_frames(all_uuids.len(), TIMEOUT);
    eprintln!("  phase 1: {} JPEG frames decoded", all_uuids.len());

    let h264_aus = load_annexb_access_units(h264_entry);
    let h264_count = 4.min(h264_aus.len());
    let pts_offset = (jpeg_count as u64) * 33_333_333;
    let h264_uuids = submit_access_units(&dec, &h264_aus, h264_entry, h264_count, pts_offset);
    all_uuids.extend(&h264_uuids);

    collector.wait_for(
        |o| matches!(o, CollectedOutput::ParameterChange { .. }),
        TIMEOUT,
    );
    collector.wait_for_frames(all_uuids.len(), TIMEOUT);

    assert_eq!(collector.parameter_change_count(), 1);
    assert_eq!(collector.frame_count(), all_uuids.len());
    assert_eq!(collector.error_count(), 0);
    collector.assert_frame_uuid_coverage(&all_uuids);

    dec.graceful_shutdown().unwrap();
    eprintln!(
        "  OK test_jpeg_to_h264: {} frames, 1 param change",
        all_uuids.len()
    );
}

#[test]
#[serial]
fn test_h264_to_jpeg_codec_change() {
    init();
    let manifest = load_manifest();
    let Some(h264_entry) = find_entry(&manifest, "test_h264_annexb_ip.h264") else {
        return;
    };
    let Some(jpeg_entry) = find_entry(&manifest, "test_jpeg_bt709.mp4") else {
        return;
    };

    let collector = OutputCollector::new();
    let mut dec = FlexibleDecoder::new(default_config(), collector.callback());

    let h264_aus = load_annexb_access_units(h264_entry);
    let h264_count = 4.min(h264_aus.len());
    let mut all_uuids = submit_access_units(&dec, &h264_aus, h264_entry, h264_count, 0);
    collector.wait_for_frames(all_uuids.len(), TIMEOUT);
    eprintln!("  phase 1: {} H.264 frames decoded", all_uuids.len());

    let jpeg_aus = demux_mp4_to_access_units(jpeg_entry);
    let jpeg_count = 3.min(jpeg_aus.len());
    let pts_offset = (h264_count as u64) * 33_333_333;
    let jpeg_uuids = submit_access_units(&dec, &jpeg_aus, jpeg_entry, jpeg_count, pts_offset);
    all_uuids.extend(&jpeg_uuids);

    collector.wait_for(
        |o| matches!(o, CollectedOutput::ParameterChange { .. }),
        TIMEOUT,
    );
    collector.wait_for_frames(all_uuids.len(), TIMEOUT);

    assert_eq!(collector.parameter_change_count(), 1);
    assert_eq!(collector.frame_count(), all_uuids.len());
    assert_eq!(collector.error_count(), 0);
    collector.assert_frame_uuid_coverage(&all_uuids);

    dec.graceful_shutdown().unwrap();
    eprintln!(
        "  OK test_h264_to_jpeg: {} frames, 1 param change",
        all_uuids.len()
    );
}

#[test]
#[serial]
fn test_h264_to_av1_codec_change() {
    init();
    let manifest = load_manifest();
    let Some(h264_entry) = find_entry(&manifest, "test_h264_annexb_ip.h264") else {
        return;
    };
    let Some(av1_entry) = find_entry(&manifest, "test_av1_bt709.mp4") else {
        return;
    };

    let collector = OutputCollector::new();
    let mut dec = FlexibleDecoder::new(default_config(), collector.callback());

    let h264_aus = load_annexb_access_units(h264_entry);
    let h264_count = 4.min(h264_aus.len());
    let mut all_uuids = submit_access_units(&dec, &h264_aus, h264_entry, h264_count, 0);
    collector.wait_for_frames(all_uuids.len(), TIMEOUT);
    eprintln!("  phase 1: {} H.264 frames decoded", all_uuids.len());

    let av1_aus = demux_mp4_to_access_units(av1_entry);
    let av1_count = 4.min(av1_aus.len());
    let pts_offset = (h264_count as u64) * 33_333_333;
    let av1_uuids = submit_access_units(&dec, &av1_aus, av1_entry, av1_count, pts_offset);
    all_uuids.extend(&av1_uuids);

    collector.wait_for(
        |o| matches!(o, CollectedOutput::ParameterChange { .. }),
        TIMEOUT,
    );
    collector.wait_for_frames(all_uuids.len(), TIMEOUT);

    assert_eq!(collector.parameter_change_count(), 1);
    assert_eq!(collector.frame_count(), all_uuids.len());
    assert_eq!(collector.error_count(), 0);
    collector.assert_frame_uuid_coverage(&all_uuids);

    dec.graceful_shutdown().unwrap();
    eprintln!(
        "  OK test_h264_to_av1: {} frames, 1 param change",
        all_uuids.len()
    );
}

// ═══════════════════════════════════════════════════════════════════
//  Multi-codec rotation: JPEG → H.264 → HEVC → JPEG
// ═══════════════════════════════════════════════════════════════════

#[test]
#[serial]
fn test_multi_codec_rotation_jpeg_h264_hevc_jpeg() {
    init();
    let manifest = load_manifest();
    let Some(jpeg_entry) = find_entry(&manifest, "test_jpeg_bt709.mp4") else {
        return;
    };
    let Some(h264_entry) = find_entry(&manifest, "test_h264_annexb_ip.h264") else {
        return;
    };
    let Some(hevc_entry) = find_entry(&manifest, "test_hevc_annexb_ip.h265") else {
        return;
    };

    let collector = OutputCollector::new();
    let mut dec = FlexibleDecoder::new(default_config(), collector.callback());

    let frames_per_phase = 3usize;
    let dur = 33_333_333u64;
    let mut all_uuids = Vec::new();
    let mut expected_changes = 0usize;

    // Phase 1: JPEG
    let jpeg_aus = demux_mp4_to_access_units(jpeg_entry);
    let n = frames_per_phase.min(jpeg_aus.len());
    let uuids = submit_access_units(&dec, &jpeg_aus, jpeg_entry, n, 0);
    all_uuids.extend(&uuids);
    collector.wait_for_frames(all_uuids.len(), TIMEOUT);
    eprintln!(
        "  phase 1 (JPEG): {} frames decoded, total={}",
        uuids.len(),
        all_uuids.len()
    );

    // Phase 2: H.264
    let h264_aus = load_annexb_access_units(h264_entry);
    let n = frames_per_phase.min(h264_aus.len());
    let offset = all_uuids.len() as u64 * dur;
    let uuids = submit_access_units(&dec, &h264_aus, h264_entry, n, offset);
    all_uuids.extend(&uuids);
    expected_changes += 1;
    collector.wait_for_frames(all_uuids.len(), TIMEOUT);
    eprintln!(
        "  phase 2 (H.264): {} frames decoded, total={}",
        uuids.len(),
        all_uuids.len()
    );

    // Phase 3: HEVC
    let hevc_aus = load_annexb_access_units(hevc_entry);
    let n = frames_per_phase.min(hevc_aus.len());
    let offset = all_uuids.len() as u64 * dur;
    let uuids = submit_access_units(&dec, &hevc_aus, hevc_entry, n, offset);
    all_uuids.extend(&uuids);
    expected_changes += 1;
    collector.wait_for_frames(all_uuids.len(), TIMEOUT);
    eprintln!(
        "  phase 3 (HEVC): {} frames decoded, total={}",
        uuids.len(),
        all_uuids.len()
    );

    // Phase 4: JPEG again
    let n = frames_per_phase.min(jpeg_aus.len());
    let offset = all_uuids.len() as u64 * dur;
    let uuids = submit_access_units(&dec, &jpeg_aus, jpeg_entry, n, offset);
    all_uuids.extend(&uuids);
    expected_changes += 1;
    collector.wait_for_frames(all_uuids.len(), TIMEOUT);
    eprintln!(
        "  phase 4 (JPEG again): {} frames decoded, total={}",
        uuids.len(),
        all_uuids.len()
    );

    assert_eq!(
        collector.parameter_change_count(),
        expected_changes,
        "expected {expected_changes} ParameterChange events"
    );
    assert_eq!(collector.frame_count(), all_uuids.len());
    assert_eq!(collector.error_count(), 0);
    collector.assert_frame_uuid_coverage(&all_uuids);

    dec.graceful_shutdown().unwrap();
    eprintln!(
        "  OK test_multi_codec_rotation: {} frames, {expected_changes} changes",
        all_uuids.len()
    );
}

// ═══════════════════════════════════════════════════════════════════
//  Source EOS between codec changes
// ═══════════════════════════════════════════════════════════════════

#[test]
#[serial]
fn test_source_eos_between_codec_changes() {
    init();
    let manifest = load_manifest();
    let Some(h264_entry) = find_entry(&manifest, "test_h264_annexb_ip.h264") else {
        return;
    };
    let Some(hevc_entry) = find_entry(&manifest, "test_hevc_annexb_ip.h265") else {
        return;
    };

    let collector = OutputCollector::new();
    let mut dec = FlexibleDecoder::new(default_config(), collector.callback());

    // Phase 1: H.264
    let h264_aus = load_annexb_access_units(h264_entry);
    let h264_count = 4.min(h264_aus.len());
    let mut all_uuids = submit_access_units(&dec, &h264_aus, h264_entry, h264_count, 0);
    collector.wait_for_frames(all_uuids.len(), TIMEOUT);

    // Source EOS while active; push a flush frame so the custom EOS event
    // propagates through the GStreamer pipeline.
    dec.source_eos(SOURCE_ID).unwrap();
    {
        let vc = codec_name_to_video_codec(&h264_entry.codec).unwrap();
        let flush_au = &h264_aus[0];
        let pts = (h264_count as u64) * 33_333_333;
        let frame = make_video_frame_ns(
            SOURCE_ID,
            vc,
            h264_entry.width as i64,
            h264_entry.height as i64,
            pts as i64,
            Some(pts as i64),
            Some(33_333_333),
            None,
        );
        all_uuids.push(frame.get_uuid_u128());
        dec.submit(&frame, Some(&flush_au.data)).unwrap();
    }
    collector.wait_for(|o| matches!(o, CollectedOutput::SourceEos { .. }), TIMEOUT);
    eprintln!("  phase 1: {} H.264 + SourceEos received", all_uuids.len());

    // Phase 2: switch to HEVC (triggers ParameterChange)
    let hevc_aus = load_annexb_access_units(hevc_entry);
    let hevc_count = 4.min(hevc_aus.len());
    let pts_offset = (h264_count as u64 + 1) * 33_333_333;
    let hevc_uuids = submit_access_units(&dec, &hevc_aus, hevc_entry, hevc_count, pts_offset);
    all_uuids.extend(&hevc_uuids);

    collector.wait_for(
        |o| matches!(o, CollectedOutput::ParameterChange { .. }),
        TIMEOUT,
    );
    collector.wait_for_frames(all_uuids.len(), TIMEOUT);

    assert_eq!(collector.parameter_change_count(), 1);
    assert!(
        collector.outputs.lock().iter().any(|o| matches!(
            o,
            CollectedOutput::SourceEos { source_id } if source_id == SOURCE_ID
        )),
        "expected SourceEos for {SOURCE_ID}"
    );
    assert_eq!(collector.error_count(), 0);
    collector.assert_frame_uuid_coverage(&all_uuids);

    dec.graceful_shutdown().unwrap();
    eprintln!("  OK test_source_eos_between_codec_changes");
}

// ═══════════════════════════════════════════════════════════════════
//  Graceful shutdown during real H.264 decode
// ═══════════════════════════════════════════════════════════════════

#[test]
#[serial]
fn test_graceful_shutdown_during_h264_decode() {
    init();
    let manifest = load_manifest();
    let Some(entry) = find_entry(&manifest, "test_h264_annexb_ip.h264") else {
        return;
    };

    let collector = OutputCollector::new();
    let mut dec = FlexibleDecoder::new(default_config(), collector.callback());

    let aus = load_annexb_access_units(entry);
    let count = 4.min(aus.len());
    let submitted = submit_access_units(&dec, &aus, entry, count, 0);
    collector.wait_for_frames(submitted.len(), TIMEOUT);

    dec.graceful_shutdown().unwrap();

    assert_eq!(collector.frame_count(), submitted.len());
    assert_eq!(collector.error_count(), 0);
    collector.assert_frame_uuid_coverage(&submitted);
    eprintln!(
        "  OK test_graceful_shutdown_during_h264: {} frames drained",
        submitted.len()
    );
}

// ═══════════════════════════════════════════════════════════════════
//  Full manifest-driven single-codec smoke test
// ═══════════════════════════════════════════════════════════════════

#[test]
#[serial]
fn test_all_mp4_assets_single_codec() {
    init();
    let manifest = load_manifest();
    let platform = current_platform_tag();
    eprintln!("Platform: {platform}");

    let mut tested = 0;
    let mut skipped = 0;

    for entry in &manifest.assets {
        if entry.container.as_deref() != Some("mp4") {
            continue;
        }
        if !asset_supported_on_platform(entry, &platform) {
            eprintln!("  SKIP {}", entry.file);
            skipped += 1;
            continue;
        }
        let vc = match codec_name_to_video_codec(&entry.codec) {
            Some(vc) => vc,
            None => {
                eprintln!("  SKIP (unsupported codec {}): {}", entry.codec, entry.file);
                skipped += 1;
                continue;
            }
        };
        // VP9 may not be supported on all platforms for FlexibleDecoder
        // (same limitations as NvDecoder).  Skip gracefully on error.
        let collector = OutputCollector::new();
        let mut dec = FlexibleDecoder::new(
            FlexibleDecoderConfig::new(&entry.file, 0, 4)
                .idle_timeout(Duration::from_secs(10))
                .detect_buffer_limit(60),
            collector.callback(),
        );

        let aus = demux_mp4_to_access_units(entry);
        let limit = aus.len().min(entry.num_frames as usize);
        let w = entry.width as i64;
        let h = entry.height as i64;

        let mut submitted_uuids = Vec::new();
        for au in aus.iter().take(limit) {
            let frame = make_video_frame_ns(
                &entry.file,
                vc,
                w,
                h,
                au.pts_ns as i64,
                au.dts_ns.map(|d| d as i64),
                au.duration_ns.map(|d| d as i64),
                None,
            );
            let uuid = frame.get_uuid_u128();
            if dec.submit(&frame, Some(&au.data)).is_err() {
                break;
            }
            submitted_uuids.push(uuid);
        }

        if submitted_uuids.is_empty() {
            eprintln!("  SKIP (submit failed): {}", entry.file);
            skipped += 1;
            continue;
        }

        // Graceful shutdown flushes B-frame reorder buffers.
        dec.graceful_shutdown().unwrap();

        assert_eq!(
            collector.error_count(),
            0,
            "{}: unexpected errors",
            entry.file
        );
        assert_eq!(
            collector.frame_count(),
            submitted_uuids.len(),
            "{}: decoded ({}) != submitted ({})",
            entry.file,
            collector.frame_count(),
            submitted_uuids.len(),
        );
        collector.assert_frame_uuid_coverage(&submitted_uuids);
        eprintln!("  OK {}: {} frames", entry.file, submitted_uuids.len());
        tested += 1;
    }

    eprintln!("\nSummary: tested={tested}, skipped={skipped}");
    assert!(tested > 0, "no MP4 assets were tested");
}

#[test]
#[serial]
fn test_all_annexb_assets_single_codec() {
    init();
    let manifest = load_manifest();
    let platform = current_platform_tag();

    let mut tested = 0;
    for entry in &manifest.assets {
        if entry.container.as_deref() != Some("raw") {
            continue;
        }
        if !matches!(entry.codec.as_str(), "h264" | "hevc") {
            continue;
        }
        if !asset_supported_on_platform(entry, &platform) {
            continue;
        }

        let collector = OutputCollector::new();
        let mut dec = FlexibleDecoder::new(
            FlexibleDecoderConfig::new(&entry.file, 0, 4)
                .idle_timeout(Duration::from_secs(10))
                .detect_buffer_limit(60),
            collector.callback(),
        );

        let aus = load_annexb_access_units(entry);
        let limit = aus.len().min(entry.num_frames as usize);
        let vc = codec_name_to_video_codec(&entry.codec).unwrap();
        let w = entry.width as i64;
        let h = entry.height as i64;

        let mut submitted_uuids = Vec::new();
        for au in aus.iter().take(limit) {
            let frame = make_video_frame_ns(
                &entry.file,
                vc,
                w,
                h,
                au.pts_ns as i64,
                au.dts_ns.map(|d| d as i64),
                au.duration_ns.map(|d| d as i64),
                None,
            );
            submitted_uuids.push(frame.get_uuid_u128());
            dec.submit(&frame, Some(&au.data)).unwrap();
        }

        // Graceful shutdown flushes B-frame reorder buffers.
        dec.graceful_shutdown().unwrap();

        assert_eq!(collector.error_count(), 0, "{}: errors", entry.file);
        assert_eq!(
            collector.frame_count(),
            submitted_uuids.len(),
            "{}: decoded ({}) != submitted ({})",
            entry.file,
            collector.frame_count(),
            submitted_uuids.len(),
        );
        collector.assert_frame_uuid_coverage(&submitted_uuids);
        eprintln!("  OK {}: {} frames", entry.file, submitted_uuids.len());
        tested += 1;
    }

    assert!(tested > 0, "no Annex-B assets were tested");
}

// ═══════════════════════════════════════════════════════════════════
//  H.264 bt709 → bt2020 (same codec & dims, different profile/bit-depth)
// ═══════════════════════════════════════════════════════════════════

/// Feed bt709 8-bit baseline H.264 followed by bt2020 10-bit high-10 H.264.
///
/// Both streams are 320×240 H.264, so `FlexibleDecoder` sees no codec or
/// dimension change — the existing NvDecoder session stays active.  The test
/// verifies the hardware decoder survives the in-stream SPS change (colour
/// primaries + bit depth) without errors or lost frames.
#[test]
#[serial]
fn test_h264_bt709_then_bt2020_same_session() {
    init();
    let manifest = load_manifest();
    let Some(bt709_entry) = find_entry(&manifest, "test_h264_bt709_ip.mp4") else {
        return;
    };
    let Some(bt2020_entry) = find_entry(&manifest, "test_h264_bt2020_ip.mp4") else {
        return;
    };

    let collector = OutputCollector::new();
    let mut dec = FlexibleDecoder::new(default_config(), collector.callback());

    // Phase 1 — bt709 8-bit
    let bt709_aus = demux_mp4_to_access_units(bt709_entry);
    let bt709_count = bt709_aus.len().min(bt709_entry.num_frames as usize);
    let mut all_uuids = submit_access_units(&dec, &bt709_aus, bt709_entry, bt709_count, 0);
    assert!(!all_uuids.is_empty());
    collector.wait_for_frames(all_uuids.len(), TIMEOUT);
    eprintln!("  phase 1 (bt709): {} frames decoded", all_uuids.len());

    // Phase 2 — bt2020 10-bit  (same codec H264, same 320×240)
    let bt2020_aus = demux_mp4_to_access_units(bt2020_entry);
    let bt2020_count = bt2020_aus.len().min(bt2020_entry.num_frames as usize);
    let pts_offset = (bt709_count as u64) * 33_333_333;
    let bt2020_uuids =
        submit_access_units(&dec, &bt2020_aus, bt2020_entry, bt2020_count, pts_offset);
    assert!(!bt2020_uuids.is_empty());
    all_uuids.extend(&bt2020_uuids);

    collector.wait_for_frames(all_uuids.len(), TIMEOUT);
    eprintln!("  phase 2 (bt2020): {} frames decoded", bt2020_uuids.len());

    // No parameter change expected — same codec and dimensions.
    assert_eq!(
        collector.parameter_change_count(),
        0,
        "no ParameterChange expected for same-codec same-dims switch"
    );
    assert_eq!(collector.frame_count(), all_uuids.len());
    assert_eq!(collector.error_count(), 0, "unexpected decoder errors");
    collector.assert_frame_uuid_coverage(&all_uuids);

    dec.graceful_shutdown().unwrap();
    eprintln!(
        "  OK test_h264_bt709_then_bt2020: {} frames, 0 param changes",
        all_uuids.len()
    );
}

// ═══════════════════════════════════════════════════════════════════
//  Mismatched VideoFrame dimensions vs actual bitstream
// ═══════════════════════════════════════════════════════════════════

/// Submit H.264 frames whose `VideoFrameProxy` width/height (640×480) differs
/// from the actual encoded resolution (320×240 in test_h264_bt709_ip.mp4).
///
/// The NvDecoder pipeline discovers the real dimensions from the bitstream's
/// SPS — the buffer pool was sized for 640×480 (larger), so the decoded
/// 320×240 frames should still fit.  The test checks that all frames decode
/// without errors or panics.
#[test]
#[serial]
fn test_h264_wrong_frame_dimensions() {
    init();
    let manifest = load_manifest();
    let Some(entry) = find_entry(&manifest, "test_h264_bt709_ip.mp4") else {
        return;
    };

    let collector = OutputCollector::new();
    let mut dec = FlexibleDecoder::new(default_config(), collector.callback());

    let aus = demux_mp4_to_access_units(entry);
    let limit = aus.len().min(entry.num_frames as usize);

    // Deliberately wrong dimensions — double the actual 320×240.
    let submitted = submit_access_units_with_dims(&dec, &aus, &entry.codec, 640, 480, limit, 0);
    assert!(!submitted.is_empty(), "no frames submitted");

    collector.wait_for_frames(submitted.len(), TIMEOUT);

    assert_eq!(
        collector.frame_count(),
        submitted.len(),
        "wrong-dims: decoded ({}) != submitted ({})",
        collector.frame_count(),
        submitted.len(),
    );
    assert_eq!(collector.error_count(), 0, "unexpected decoder errors");
    assert_eq!(collector.skip_count(), 0, "unexpected skipped frames");
    collector.assert_frame_uuid_coverage(&submitted);

    dec.graceful_shutdown().unwrap();
    eprintln!(
        "  OK test_h264_wrong_frame_dimensions: {} frames with 640x480 metadata for 320x240 bitstream",
        submitted.len()
    );
}

// ═══════════════════════════════════════════════════════════════════
//  Sealed delivery — cross-thread unseal
// ═══════════════════════════════════════════════════════════════════

/// Decode a real H.264 MP4 and exercise the full `SealedDelivery` lifecycle
/// across threads:
///
/// 1. The callback calls `take_delivery()` on every `Frame` output and
///    sends the `SealedDelivery` to a consumer thread via a channel.
/// 2. The callback returns, dropping the `FlexibleDecoderOutput` which
///    releases the seal.
/// 3. The consumer thread calls `unseal()` (blocks until the seal is
///    released) and verifies the `SharedBuffer` is valid.
#[test]
#[serial]
fn test_sealed_delivery_cross_thread_unseal() {
    init();
    let manifest = load_manifest();
    let Some(entry) = find_entry(&manifest, "test_h264_bt709_ip.mp4") else {
        return;
    };

    let (tx, rx) = std::sync::mpsc::channel::<SealedDelivery>();
    let frame_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let fc = frame_count.clone();
    let error_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let ec = error_count.clone();

    let callback = move |mut out: FlexibleDecoderOutput| {
        match &out {
            FlexibleDecoderOutput::Frame { .. } => {
                if let Some(sealed) = out.take_delivery() {
                    assert!(
                        !sealed.is_released(),
                        "seal must NOT be released while the output is alive"
                    );
                    tx.send(sealed).expect("consumer thread gone");
                }
                fc.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            FlexibleDecoderOutput::Error(_) => {
                ec.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            _ => {}
        }
        // `out` drops here → seal released
    };

    let mut dec = FlexibleDecoder::new(default_config(), callback);
    let aus = demux_mp4_to_access_units(entry);
    let num = aus.len().min(entry.num_frames as usize);
    let submitted = submit_access_units(&dec, &aus, entry, num, 0);
    assert!(!submitted.is_empty(), "no frames submitted");

    // Consumer thread: unseal each delivery and verify the buffer.
    let expected_count = submitted.len();
    let consumer = std::thread::spawn(move || {
        let mut received = 0usize;
        while received < expected_count {
            let sealed = rx
                .recv_timeout(Duration::from_secs(30))
                .unwrap_or_else(|e| {
                    panic!(
                        "consumer: timed out waiting for delivery {}/{}: {e}",
                        received + 1,
                        expected_count
                    )
                });
            let (proxy, buffer) = sealed
                .unseal()
                .expect("SealedDelivery must contain a delivery");
            assert!(proxy.get_uuid_u128() != 0, "proxy UUID must be non-zero");
            let guard = buffer.lock();
            assert!(
                guard.as_ref().size() > 0,
                "SharedBuffer must hold data (frame {})",
                received
            );
            drop(guard);
            received += 1;
        }
        received
    });

    // Wait for all frames to arrive on the decoder side.
    let start = std::time::Instant::now();
    loop {
        let fc = frame_count.load(std::sync::atomic::Ordering::Relaxed);
        if fc >= expected_count {
            break;
        }
        if start.elapsed() > TIMEOUT {
            panic!(
                "timeout waiting for {expected_count} frames (got {fc} after {:?})",
                start.elapsed()
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }

    assert_eq!(
        error_count.load(std::sync::atomic::Ordering::Relaxed),
        0,
        "unexpected decoder errors"
    );

    let unsealed_count = consumer.join().expect("consumer thread panicked");
    assert_eq!(
        unsealed_count, expected_count,
        "consumer must unseal exactly as many deliveries as frames submitted"
    );

    dec.graceful_shutdown().unwrap();
    eprintln!(
        "  OK test_sealed_delivery_cross_thread_unseal: {} frames decoded, sealed, and unsealed",
        expected_count
    );
}
