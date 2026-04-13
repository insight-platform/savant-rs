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
use deepstream_inputs::flexible_decoder::{FlexibleDecoder, FlexibleDecoderConfig};
use serial_test::serial;
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
fn submit_access_units(
    dec: &FlexibleDecoder,
    aus: &[AccessUnit],
    entry: &AssetEntry,
    limit: usize,
    pts_offset_ns: u64,
) -> usize {
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
fn submit_access_units_with_dims(
    dec: &FlexibleDecoder,
    aus: &[AccessUnit],
    codec_name: &str,
    width: i64,
    height: i64,
    limit: usize,
    pts_offset_ns: u64,
) -> usize {
    let vc = codec_name_to_video_codec(codec_name)
        .unwrap_or_else(|| panic!("unknown codec: {codec_name}"));
    let mut submitted = 0;
    for au in aus.iter().take(limit) {
        let pts = pts_offset_ns + au.pts_ns;
        let dts = au.dts_ns.map(|d| (pts_offset_ns + d) as i64);
        let dur = au.duration_ns.map(|d| d as i64);
        let frame = make_video_frame_ns(SOURCE_ID, vc, width, height, pts as i64, dts, dur, None);
        dec.submit(&frame, Some(&au.data))
            .unwrap_or_else(|e| panic!("submit failed at AU {submitted}: {e}"));
        submitted += 1;
    }
    submitted
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
    assert!(submitted > 0, "no frames submitted for H.264 MP4");

    collector.wait_for_frames(submitted, TIMEOUT);
    assert_eq!(
        collector.frame_count(),
        submitted,
        "H.264 MP4: decoded != submitted"
    );
    assert_eq!(
        collector.error_count(),
        0,
        "H.264 MP4: unexpected decoder errors"
    );

    dec.graceful_shutdown().unwrap();
    eprintln!("  OK test_h264_mp4_decode: {submitted} frames");
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

    collector.wait_for_frames(submitted, TIMEOUT);
    assert_eq!(collector.frame_count(), submitted);
    assert_eq!(collector.error_count(), 0);

    dec.graceful_shutdown().unwrap();
    eprintln!("  OK test_h264_annexb_decode: {submitted} frames");
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

    collector.wait_for_frames(submitted, TIMEOUT);
    assert_eq!(collector.frame_count(), submitted);
    assert_eq!(collector.error_count(), 0);

    dec.graceful_shutdown().unwrap();
    eprintln!("  OK test_hevc_mp4_decode: {submitted} frames");
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

    collector.wait_for_frames(submitted, TIMEOUT);
    assert_eq!(collector.frame_count(), submitted);
    assert_eq!(collector.error_count(), 0);

    dec.graceful_shutdown().unwrap();
    eprintln!("  OK test_hevc_annexb_decode: {submitted} frames");
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

    collector.wait_for_frames(submitted, TIMEOUT);
    assert_eq!(collector.frame_count(), submitted);
    assert_eq!(collector.error_count(), 0);

    dec.graceful_shutdown().unwrap();
    eprintln!("  OK test_av1_mp4_decode: {submitted} frames");
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

    collector.wait_for_frames(submitted, TIMEOUT);
    assert_eq!(collector.frame_count(), submitted);
    assert_eq!(collector.error_count(), 0);

    dec.graceful_shutdown().unwrap();
    eprintln!("  OK test_jpeg_mp4_decode: {submitted} frames");
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
    let submitted_h264 = submit_access_units(&dec, &h264_aus, h264_entry, h264_count, 0);
    collector.wait_for_frames(submitted_h264, TIMEOUT);
    eprintln!("  phase 1: {submitted_h264} H.264 frames decoded");

    let hevc_aus = load_annexb_access_units(hevc_entry);
    let hevc_count = 4.min(hevc_aus.len());
    let pts_offset = (h264_count as u64) * 33_333_333;
    let submitted_hevc = submit_access_units(&dec, &hevc_aus, hevc_entry, hevc_count, pts_offset);

    collector.wait_for(
        |o| matches!(o, CollectedOutput::ParameterChange { .. }),
        TIMEOUT,
    );
    collector.wait_for_frames(submitted_h264 + submitted_hevc, TIMEOUT);

    assert_eq!(collector.parameter_change_count(), 1);
    assert_eq!(collector.frame_count(), submitted_h264 + submitted_hevc);
    assert_eq!(collector.error_count(), 0);

    dec.graceful_shutdown().unwrap();
    eprintln!("  OK test_h264_to_hevc: {submitted_h264}+{submitted_hevc} frames, 1 param change");
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
    let submitted_jpeg = submit_access_units(&dec, &jpeg_aus, jpeg_entry, jpeg_count, 0);
    collector.wait_for_frames(submitted_jpeg, TIMEOUT);
    eprintln!("  phase 1: {submitted_jpeg} JPEG frames decoded");

    let h264_aus = load_annexb_access_units(h264_entry);
    let h264_count = 4.min(h264_aus.len());
    let pts_offset = (jpeg_count as u64) * 33_333_333;
    let submitted_h264 = submit_access_units(&dec, &h264_aus, h264_entry, h264_count, pts_offset);

    collector.wait_for(
        |o| matches!(o, CollectedOutput::ParameterChange { .. }),
        TIMEOUT,
    );
    collector.wait_for_frames(submitted_jpeg + submitted_h264, TIMEOUT);

    assert_eq!(collector.parameter_change_count(), 1);
    assert_eq!(collector.frame_count(), submitted_jpeg + submitted_h264);
    assert_eq!(collector.error_count(), 0);

    dec.graceful_shutdown().unwrap();
    eprintln!("  OK test_jpeg_to_h264: {submitted_jpeg}+{submitted_h264} frames, 1 param change");
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
    let submitted_h264 = submit_access_units(&dec, &h264_aus, h264_entry, h264_count, 0);
    collector.wait_for_frames(submitted_h264, TIMEOUT);
    eprintln!("  phase 1: {submitted_h264} H.264 frames decoded");

    let jpeg_aus = demux_mp4_to_access_units(jpeg_entry);
    let jpeg_count = 3.min(jpeg_aus.len());
    let pts_offset = (h264_count as u64) * 33_333_333;
    let submitted_jpeg = submit_access_units(&dec, &jpeg_aus, jpeg_entry, jpeg_count, pts_offset);

    collector.wait_for(
        |o| matches!(o, CollectedOutput::ParameterChange { .. }),
        TIMEOUT,
    );
    collector.wait_for_frames(submitted_h264 + submitted_jpeg, TIMEOUT);

    assert_eq!(collector.parameter_change_count(), 1);
    assert_eq!(collector.frame_count(), submitted_h264 + submitted_jpeg);
    assert_eq!(collector.error_count(), 0);

    dec.graceful_shutdown().unwrap();
    eprintln!("  OK test_h264_to_jpeg: {submitted_h264}+{submitted_jpeg} frames, 1 param change");
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
    let submitted_h264 = submit_access_units(&dec, &h264_aus, h264_entry, h264_count, 0);
    collector.wait_for_frames(submitted_h264, TIMEOUT);
    eprintln!("  phase 1: {submitted_h264} H.264 frames decoded");

    let av1_aus = demux_mp4_to_access_units(av1_entry);
    let av1_count = 4.min(av1_aus.len());
    let pts_offset = (h264_count as u64) * 33_333_333;
    let submitted_av1 = submit_access_units(&dec, &av1_aus, av1_entry, av1_count, pts_offset);

    collector.wait_for(
        |o| matches!(o, CollectedOutput::ParameterChange { .. }),
        TIMEOUT,
    );
    collector.wait_for_frames(submitted_h264 + submitted_av1, TIMEOUT);

    assert_eq!(collector.parameter_change_count(), 1);
    assert_eq!(collector.frame_count(), submitted_h264 + submitted_av1);
    assert_eq!(collector.error_count(), 0);

    dec.graceful_shutdown().unwrap();
    eprintln!("  OK test_h264_to_av1: {submitted_h264}+{submitted_av1} frames, 1 param change");
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
    let mut total_submitted = 0usize;
    let mut expected_changes = 0usize;

    // Phase 1: JPEG
    let jpeg_aus = demux_mp4_to_access_units(jpeg_entry);
    let n = frames_per_phase.min(jpeg_aus.len());
    let s = submit_access_units(&dec, &jpeg_aus, jpeg_entry, n, 0);
    total_submitted += s;
    collector.wait_for_frames(total_submitted, TIMEOUT);
    eprintln!("  phase 1 (JPEG): {s} frames decoded, total={total_submitted}");

    // Phase 2: H.264
    let h264_aus = load_annexb_access_units(h264_entry);
    let n = frames_per_phase.min(h264_aus.len());
    let offset = total_submitted as u64 * dur;
    let s = submit_access_units(&dec, &h264_aus, h264_entry, n, offset);
    total_submitted += s;
    expected_changes += 1;
    collector.wait_for_frames(total_submitted, TIMEOUT);
    eprintln!("  phase 2 (H.264): {s} frames decoded, total={total_submitted}");

    // Phase 3: HEVC
    let hevc_aus = load_annexb_access_units(hevc_entry);
    let n = frames_per_phase.min(hevc_aus.len());
    let offset = total_submitted as u64 * dur;
    let s = submit_access_units(&dec, &hevc_aus, hevc_entry, n, offset);
    total_submitted += s;
    expected_changes += 1;
    collector.wait_for_frames(total_submitted, TIMEOUT);
    eprintln!("  phase 3 (HEVC): {s} frames decoded, total={total_submitted}");

    // Phase 4: JPEG again
    let n = frames_per_phase.min(jpeg_aus.len());
    let offset = total_submitted as u64 * dur;
    let s = submit_access_units(&dec, &jpeg_aus, jpeg_entry, n, offset);
    total_submitted += s;
    expected_changes += 1;
    collector.wait_for_frames(total_submitted, TIMEOUT);
    eprintln!("  phase 4 (JPEG again): {s} frames decoded, total={total_submitted}");

    assert_eq!(
        collector.parameter_change_count(),
        expected_changes,
        "expected {expected_changes} ParameterChange events"
    );
    assert_eq!(collector.frame_count(), total_submitted);
    assert_eq!(collector.error_count(), 0);

    dec.graceful_shutdown().unwrap();
    eprintln!(
        "  OK test_multi_codec_rotation: {total_submitted} frames, {expected_changes} changes"
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
    let submitted_h264 = submit_access_units(&dec, &h264_aus, h264_entry, h264_count, 0);
    collector.wait_for_frames(submitted_h264, TIMEOUT);

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
        dec.submit(&frame, Some(&flush_au.data)).unwrap();
    }
    collector.wait_for(|o| matches!(o, CollectedOutput::SourceEos { .. }), TIMEOUT);
    eprintln!("  phase 1: {submitted_h264} H.264 + SourceEos received");

    // Phase 2: switch to HEVC (triggers ParameterChange)
    let hevc_aus = load_annexb_access_units(hevc_entry);
    let hevc_count = 4.min(hevc_aus.len());
    let pts_offset = (h264_count as u64 + 1) * 33_333_333;
    let submitted_hevc = submit_access_units(&dec, &hevc_aus, hevc_entry, hevc_count, pts_offset);

    collector.wait_for(
        |o| matches!(o, CollectedOutput::ParameterChange { .. }),
        TIMEOUT,
    );
    // +1 for the flush frame
    let total_expected = submitted_h264 + 1 + submitted_hevc;
    collector.wait_for_frames(total_expected, TIMEOUT);

    assert_eq!(collector.parameter_change_count(), 1);
    assert!(
        collector.outputs.lock().iter().any(|o| matches!(
            o,
            CollectedOutput::SourceEos { source_id } if source_id == SOURCE_ID
        )),
        "expected SourceEos for {SOURCE_ID}"
    );
    assert_eq!(collector.error_count(), 0);

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
    collector.wait_for_frames(submitted, TIMEOUT);

    dec.graceful_shutdown().unwrap();

    assert_eq!(collector.frame_count(), submitted);
    assert_eq!(collector.error_count(), 0);
    eprintln!("  OK test_graceful_shutdown_during_h264: {submitted} frames drained");
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

        let mut submitted = 0usize;
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
            if dec.submit(&frame, Some(&au.data)).is_err() {
                break;
            }
            submitted += 1;
        }

        if submitted == 0 {
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
            submitted,
            "{}: decoded ({}) != submitted ({submitted})",
            entry.file,
            collector.frame_count(),
        );
        eprintln!("  OK {}: {submitted} frames", entry.file);
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

        let mut submitted = 0usize;
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
            dec.submit(&frame, Some(&au.data)).unwrap();
            submitted += 1;
        }

        // Graceful shutdown flushes B-frame reorder buffers.
        dec.graceful_shutdown().unwrap();

        assert_eq!(collector.error_count(), 0, "{}: errors", entry.file);
        assert_eq!(
            collector.frame_count(),
            submitted,
            "{}: decoded ({}) != submitted ({submitted})",
            entry.file,
            collector.frame_count(),
        );
        eprintln!("  OK {}: {submitted} frames", entry.file);
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
    let submitted_bt709 = submit_access_units(&dec, &bt709_aus, bt709_entry, bt709_count, 0);
    assert!(submitted_bt709 > 0);
    collector.wait_for_frames(submitted_bt709, TIMEOUT);
    eprintln!("  phase 1 (bt709): {submitted_bt709} frames decoded");

    // Phase 2 — bt2020 10-bit  (same codec H264, same 320×240)
    let bt2020_aus = demux_mp4_to_access_units(bt2020_entry);
    let bt2020_count = bt2020_aus.len().min(bt2020_entry.num_frames as usize);
    let pts_offset = (bt709_count as u64) * 33_333_333;
    let submitted_bt2020 =
        submit_access_units(&dec, &bt2020_aus, bt2020_entry, bt2020_count, pts_offset);
    assert!(submitted_bt2020 > 0);

    let total = submitted_bt709 + submitted_bt2020;
    collector.wait_for_frames(total, TIMEOUT);
    eprintln!("  phase 2 (bt2020): {submitted_bt2020} frames decoded");

    // No parameter change expected — same codec and dimensions.
    assert_eq!(
        collector.parameter_change_count(),
        0,
        "no ParameterChange expected for same-codec same-dims switch"
    );
    assert_eq!(collector.frame_count(), total);
    assert_eq!(collector.error_count(), 0, "unexpected decoder errors");

    dec.graceful_shutdown().unwrap();
    eprintln!(
        "  OK test_h264_bt709_then_bt2020: {submitted_bt709}+{submitted_bt2020} frames, 0 param changes"
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
    assert!(submitted > 0, "no frames submitted");

    collector.wait_for_frames(submitted, TIMEOUT);

    assert_eq!(
        collector.frame_count(),
        submitted,
        "wrong-dims: decoded ({}) != submitted ({submitted})",
        collector.frame_count(),
    );
    assert_eq!(collector.error_count(), 0, "unexpected decoder errors");
    assert_eq!(collector.skip_count(), 0, "unexpected skipped frames");

    dec.graceful_shutdown().unwrap();
    eprintln!(
        "  OK test_h264_wrong_frame_dimensions: {submitted} frames with 640x480 metadata for 320x240 bitstream"
    );
}
