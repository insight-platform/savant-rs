//! Verify that session-boundary resets (codec / resolution change) drain
//! all in-flight frames before tearing down the old decoder.
//!
//! Prior to the drain-before-reset fix, `emit_session_reset` killed the
//! `NvDecoder` immediately, silently discarding any frames that were still
//! queued or being decoded by the GPU.

mod common;

use common::*;
use deepstream_inputs::multistream_decoder::{
    DecoderOutput, EvictionVerdict, MultiStreamDecoder, MultiStreamDecoderConfig,
    SessionBoundaryEosPolicy, StopReason,
};
use savant_core::primitives::eos::EndOfStream;
use savant_gstreamer::mp4_demuxer::Mp4Demuxer;
use serial_test::serial;
use std::sync::mpsc;
use std::time::Duration;

const FRAME_DUR_NS: i64 = 33_333_333;
const RECV_TIMEOUT: Duration = Duration::from_secs(30);
const SUBMIT_TIMEOUT: Duration = Duration::from_secs(5);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_decoder_with_policy(
    eos_policy: SessionBoundaryEosPolicy,
) -> (MultiStreamDecoder, mpsc::Receiver<DecoderOutput>) {
    let (tx, rx) = mpsc::channel();
    let cfg = MultiStreamDecoderConfig::new(0, 16)
        .idle_timeout(Duration::from_secs(600))
        .per_stream_queue_size(32)
        .session_boundary_eos(eos_policy);
    let decoder = MultiStreamDecoder::new(
        cfg,
        move |o| {
            let _ = tx.send(o);
        },
        None::<fn(&str) -> EvictionVerdict>,
    );
    (decoder, rx)
}

fn drain_until_stopped(rx: &mpsc::Receiver<DecoderOutput>) -> Vec<DecoderOutput> {
    let mut events = Vec::new();
    loop {
        match rx.recv_timeout(RECV_TIMEOUT) {
            Ok(ev) => {
                let is_stopped = matches!(&ev, DecoderOutput::StreamStopped { .. });
                events.push(ev);
                if is_stopped {
                    break;
                }
            }
            Err(_) => panic!("timeout — events so far: {events:#?}"),
        }
    }
    events
}

fn count_decoded(events: &[DecoderOutput]) -> usize {
    events
        .iter()
        .filter(|e| matches!(e, DecoderOutput::Decoded { .. }))
        .count()
}

fn count_eos(events: &[DecoderOutput]) -> usize {
    events
        .iter()
        .filter(|e| matches!(e, DecoderOutput::Eos { .. }))
        .count()
}

/// Demux + split Annex-B AUs for an H.26x asset.
fn load_h26x_access_units(file_name: &str, codec_tag: &str) -> Vec<Vec<u8>> {
    let manifest = load_manifest();
    let platform_tag = current_platform_tag();
    let entry = manifest
        .assets
        .iter()
        .find(|a| a.file == file_name)
        .unwrap_or_else(|| panic!("{file_name} not in manifest"));
    assert!(
        asset_supported_on_platform(entry, &platform_tag),
        "{file_name} not supported on this platform"
    );
    let mp4_path = assets_dir().join(&entry.file).to_str().unwrap().to_string();
    let mut demuxer = Mp4Demuxer::new_parsed(&mp4_path).expect("demux");
    let mut packets = Vec::new();
    loop {
        match demuxer.pull_timeout(Duration::from_secs(5)) {
            Ok(Some(p)) => packets.push(p),
            Ok(None) => break,
            Err(e) => panic!("pull {file_name}: {e}"),
        }
    }
    demuxer.finish();

    let mut bytestream = Vec::new();
    for pkt in &packets {
        bytestream.extend_from_slice(&pkt.data);
    }
    let nalus = split_annexb_nalus(&bytestream, codec_tag);
    group_nalus_to_access_units(codec_tag, nalus)
}

/// Submit Annex-B access units from a named MP4 H.26x asset.
///
/// Returns `(submitted_count, entry_width, entry_height)`.
fn submit_h26x_from_asset(
    decoder: &MultiStreamDecoder,
    source_id: &str,
    file_name: &str,
    codec_tag: &str,
    pts_offset_ns: i64,
) -> (u32, u32, u32) {
    let manifest = load_manifest();
    let entry = manifest
        .assets
        .iter()
        .find(|a| a.file == file_name)
        .unwrap_or_else(|| panic!("{file_name} not in manifest"));
    let access_units = load_h26x_access_units(file_name, codec_tag);
    let au_count = access_units.len().min(entry.num_frames as usize);

    for (i, au) in access_units.iter().take(au_count).enumerate() {
        let pts = pts_offset_ns + i as i64 * FRAME_DUR_NS;
        let frame = make_video_frame_ns(
            source_id,
            codec_tag,
            entry.width as i64,
            entry.height as i64,
            pts,
            Some(pts),
            Some(FRAME_DUR_NS),
            None,
        );
        decoder
            .submit(frame, Some(au.as_slice()), SUBMIT_TIMEOUT)
            .unwrap_or_else(|e| panic!("submit {file_name} AU {i}: {e}"));
    }
    (au_count as u32, entry.width, entry.height)
}

/// Same as [`submit_h26x_from_asset`] but uses explicit width/height on every frame.
fn submit_h26x_from_asset_with_dims(
    decoder: &MultiStreamDecoder,
    source_id: &str,
    file_name: &str,
    codec_tag: &str,
    pts_offset_ns: i64,
    width: i64,
    height: i64,
) -> u32 {
    let manifest = load_manifest();
    let entry = manifest
        .assets
        .iter()
        .find(|a| a.file == file_name)
        .unwrap_or_else(|| panic!("{file_name} not in manifest"));
    let access_units = load_h26x_access_units(file_name, codec_tag);
    let au_count = access_units.len().min(entry.num_frames as usize);

    for (i, au) in access_units.iter().take(au_count).enumerate() {
        let pts = pts_offset_ns + i as i64 * FRAME_DUR_NS;
        let frame = make_video_frame_ns(
            source_id,
            codec_tag,
            width,
            height,
            pts,
            Some(pts),
            Some(FRAME_DUR_NS),
            None,
        );
        decoder
            .submit(frame, Some(au.as_slice()), SUBMIT_TIMEOUT)
            .unwrap_or_else(|e| panic!("submit {file_name} AU {i}: {e}"));
    }
    au_count as u32
}

/// Submit `n` monotonic H264 AUs, then one more AU with PTS/DTS 0 (timestamp regression).
fn submit_h264_then_timestamp_regress(
    decoder: &MultiStreamDecoder,
    source_id: &str,
    n_monotonic: usize,
) {
    let manifest = load_manifest();
    let entry = manifest
        .assets
        .iter()
        .find(|a| a.file == "test_h264_bt709_ip.mp4")
        .expect("test_h264_bt709_ip.mp4 in manifest");
    let access_units = load_h26x_access_units("test_h264_bt709_ip.mp4", "h264");
    assert!(
        access_units.len() > n_monotonic,
        "need more AUs than n_monotonic"
    );

    let w = entry.width as i64;
    let h = entry.height as i64;
    for (i, au) in access_units.iter().enumerate().take(n_monotonic) {
        let pts = i as i64 * FRAME_DUR_NS;
        let frame = make_video_frame_ns(
            source_id,
            "h264",
            w,
            h,
            pts,
            Some(pts),
            Some(FRAME_DUR_NS),
            None,
        );
        decoder
            .submit(frame, Some(au.as_slice()), SUBMIT_TIMEOUT)
            .unwrap_or_else(|e| panic!("submit monotonic AU {i}: {e}"));
    }

    let regress_au = &access_units[n_monotonic];
    let frame = make_video_frame_ns(
        source_id,
        "h264",
        w,
        h,
        0,
        Some(0),
        Some(FRAME_DUR_NS),
        None,
    );
    decoder
        .submit(frame, Some(regress_au.as_slice()), SUBMIT_TIMEOUT)
        .expect("submit regressed PTS frame");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const SID: &str = "session_reset_drain";

/// Codec change with `session_boundary_eos = true` (default): submit H264
/// frames, then immediately submit HEVC frames (triggering CodecChanged).
/// All H264 frames must arrive as `Decoded` and there must be an `Eos`
/// before `StreamStopped(CodecChanged)`.
#[test]
#[serial]
fn test_session_reset_drain_codec_change_with_eos() {
    init();

    let (mut decoder, rx) = make_decoder_with_policy(SessionBoundaryEosPolicy::default());

    let (h264_count, _, _) =
        submit_h26x_from_asset(&decoder, SID, "test_h264_bt709_ip.mp4", "h264", 0);
    assert!(h264_count > 0, "no H264 frames submitted");

    let pts_offset = h264_count as i64 * FRAME_DUR_NS;
    let (hevc_count, _, _) =
        submit_h26x_from_asset(&decoder, SID, "test_hevc_bt709_ip.mp4", "hevc", pts_offset);
    assert!(hevc_count > 0, "no HEVC frames submitted");

    // Drain events from the H264 session reset.
    let events = drain_until_stopped(&rx);
    let decoded = count_decoded(&events);
    let eos = count_eos(&events);

    eprintln!(
        "H264 session: submitted={h264_count}, decoded={decoded}, eos={eos}, events={}",
        events.len()
    );

    assert_eq!(
        decoded, h264_count as usize,
        "all H264 frames must be decoded before session reset; events: {events:#?}"
    );
    assert_eq!(
        eos, 1,
        "exactly one Eos event expected; events: {events:#?}"
    );

    let last = events.last().unwrap();
    assert!(
        matches!(
            last,
            DecoderOutput::StreamStopped {
                reason: StopReason::CodecChanged,
                ..
            }
        ),
        "last event must be StreamStopped(CodecChanged); got: {last:#?}"
    );

    // Eos must come after all Decoded events.
    let last_decoded_idx = events
        .iter()
        .rposition(|e| matches!(e, DecoderOutput::Decoded { .. }));
    let eos_idx = events
        .iter()
        .position(|e| matches!(e, DecoderOutput::Eos { .. }));
    if let (Some(ld), Some(ep)) = (last_decoded_idx, eos_idx) {
        assert!(
            ld < ep,
            "Eos must come after all Decoded; events: {events:#?}"
        );
    }

    // Clean up the HEVC session.
    decoder.shutdown();
}

/// Codec change with `session_boundary_eos = false`: same as above but no
/// `Eos` event should be emitted. All H264 frames must still be decoded.
#[test]
#[serial]
fn test_session_reset_drain_codec_change_without_eos() {
    init();

    let policy = SessionBoundaryEosPolicy {
        on_codec_change: false,
        on_resolution_change: false,
        on_timestamp_regress: false,
    };
    let (mut decoder, rx) = make_decoder_with_policy(policy);

    let (h264_count, _, _) =
        submit_h26x_from_asset(&decoder, SID, "test_h264_bt709_ip.mp4", "h264", 0);
    assert!(h264_count > 0);

    let pts_offset = h264_count as i64 * FRAME_DUR_NS;
    submit_h26x_from_asset(&decoder, SID, "test_hevc_bt709_ip.mp4", "hevc", pts_offset);

    let events = drain_until_stopped(&rx);
    let decoded = count_decoded(&events);
    let eos = count_eos(&events);

    eprintln!(
        "H264 session (no-eos): submitted={h264_count}, decoded={decoded}, eos={eos}, events={}",
        events.len()
    );

    assert_eq!(
        decoded, h264_count as usize,
        "all H264 frames must be decoded even with eos policy false; events: {events:#?}"
    );
    assert_eq!(
        eos, 0,
        "no Eos event expected when policy is false; events: {events:#?}"
    );

    let last = events.last().unwrap();
    assert!(
        matches!(
            last,
            DecoderOutput::StreamStopped {
                reason: StopReason::CodecChanged,
                ..
            }
        ),
        "last event must be StreamStopped(CodecChanged); got: {last:#?}"
    );

    decoder.shutdown();
}

/// Resolution change: only `on_resolution_change` affects Eos emission.
#[test]
#[serial]
fn test_session_reset_resolution_change_with_eos() {
    init();

    let policy = SessionBoundaryEosPolicy {
        on_codec_change: false,
        on_resolution_change: true,
        on_timestamp_regress: false,
    };
    let (mut decoder, rx) = make_decoder_with_policy(policy);

    let manifest = load_manifest();
    let entry = manifest
        .assets
        .iter()
        .find(|a| a.file == "test_h264_bt709_ip.mp4")
        .unwrap();
    let w = entry.width as i64;
    let h = entry.height as i64;
    assert!(w > 1, "asset width must allow w-1 session-boundary test");

    let sid = "session_reset_resolution_eos";
    // Use (w-1, h) then (w, h) so session reset is metadata-only; pool still fits the bitstream.
    let n = submit_h26x_from_asset_with_dims(
        &decoder,
        sid,
        "test_h264_bt709_ip.mp4",
        "h264",
        0,
        w - 1,
        h,
    );
    assert!(n > 0);

    let aus = load_h26x_access_units("test_h264_bt709_ip.mp4", "h264");
    let pts = n as i64 * FRAME_DUR_NS;
    let frame = make_video_frame_ns(sid, "h264", w, h, pts, Some(pts), Some(FRAME_DUR_NS), None);
    decoder
        .submit(frame, Some(aus[0].as_slice()), SUBMIT_TIMEOUT)
        .expect("submit resolution-change frame");

    let events = drain_until_stopped(&rx);
    assert_eq!(count_decoded(&events), n as usize, "events: {events:#?}");
    assert_eq!(count_eos(&events), 1);
    let last = events.last().unwrap();
    assert!(
        matches!(
            last,
            DecoderOutput::StreamStopped {
                reason: StopReason::ResolutionChanged,
                ..
            }
        ),
        "got: {last:#?}"
    );

    decoder.shutdown();
}

#[test]
#[serial]
fn test_session_reset_resolution_change_without_eos() {
    init();

    let policy = SessionBoundaryEosPolicy {
        on_codec_change: false,
        on_resolution_change: false,
        on_timestamp_regress: false,
    };
    let (mut decoder, rx) = make_decoder_with_policy(policy);

    let manifest = load_manifest();
    let entry = manifest
        .assets
        .iter()
        .find(|a| a.file == "test_h264_bt709_ip.mp4")
        .unwrap();
    let w = entry.width as i64;
    let h = entry.height as i64;
    assert!(w > 1);

    let sid = "session_reset_resolution_no_eos";
    let n = submit_h26x_from_asset_with_dims(
        &decoder,
        sid,
        "test_h264_bt709_ip.mp4",
        "h264",
        0,
        w - 1,
        h,
    );
    assert!(n > 0);

    let aus = load_h26x_access_units("test_h264_bt709_ip.mp4", "h264");
    let pts = n as i64 * FRAME_DUR_NS;
    let frame = make_video_frame_ns(sid, "h264", w, h, pts, Some(pts), Some(FRAME_DUR_NS), None);
    decoder
        .submit(frame, Some(aus[0].as_slice()), SUBMIT_TIMEOUT)
        .expect("submit resolution-change frame");

    let events = drain_until_stopped(&rx);
    assert_eq!(count_decoded(&events), n as usize, "events: {events:#?}");
    assert_eq!(count_eos(&events), 0);
    let last = events.last().unwrap();
    assert!(
        matches!(
            last,
            DecoderOutput::StreamStopped {
                reason: StopReason::ResolutionChanged,
                ..
            }
        ),
        "got: {last:#?}"
    );

    decoder.shutdown();
}

/// Timestamp regression: only `on_timestamp_regress` affects Eos emission.
#[test]
#[serial]
fn test_session_reset_timestamp_regress_with_eos() {
    init();

    let policy = SessionBoundaryEosPolicy {
        on_codec_change: false,
        on_resolution_change: false,
        on_timestamp_regress: true,
    };
    let (mut decoder, rx) = make_decoder_with_policy(policy);

    let sid = "session_reset_ts_eos";
    let n_monotonic = 3;
    submit_h264_then_timestamp_regress(&decoder, sid, n_monotonic);

    let events = drain_until_stopped(&rx);
    assert_eq!(count_decoded(&events), n_monotonic, "events: {events:#?}");
    assert_eq!(count_eos(&events), 1);
    let last = events.last().unwrap();
    assert!(
        matches!(
            last,
            DecoderOutput::StreamStopped {
                reason: StopReason::TimestampRegressed,
                ..
            }
        ),
        "got: {last:#?}"
    );

    decoder
        .submit_eos(&EndOfStream::new(sid), SUBMIT_TIMEOUT)
        .expect("eos");
    let _ = drain_until_stopped(&rx);
    decoder.shutdown();
}

#[test]
#[serial]
fn test_session_reset_timestamp_regress_without_eos() {
    init();

    let policy = SessionBoundaryEosPolicy {
        on_codec_change: false,
        on_resolution_change: false,
        on_timestamp_regress: false,
    };
    let (mut decoder, rx) = make_decoder_with_policy(policy);

    let sid = "session_reset_ts_no_eos";
    let n_monotonic = 3;
    submit_h264_then_timestamp_regress(&decoder, sid, n_monotonic);

    let events = drain_until_stopped(&rx);
    assert_eq!(count_decoded(&events), n_monotonic, "events: {events:#?}");
    assert_eq!(count_eos(&events), 0);
    let last = events.last().unwrap();
    assert!(
        matches!(
            last,
            DecoderOutput::StreamStopped {
                reason: StopReason::TimestampRegressed,
                ..
            }
        ),
        "got: {last:#?}"
    );

    decoder
        .submit_eos(&EndOfStream::new(sid), SUBMIT_TIMEOUT)
        .expect("eos");
    let _ = drain_until_stopped(&rx);
    decoder.shutdown();
}

/// Multiple codec hops on the same source_id, never sending explicit EOS.
/// Each hop triggers a session reset drain. Verify 100% frame delivery
/// across all segments.
#[test]
#[serial]
fn test_session_reset_drain_multiple_hops() {
    init();

    let policy = SessionBoundaryEosPolicy {
        on_codec_change: true,
        on_resolution_change: true,
        on_timestamp_regress: true,
    };
    let (mut decoder, rx) = make_decoder_with_policy(policy);

    let mut total_submitted = 0u32;
    let mut total_decoded = 0usize;

    // Hop 1: H264
    let (h264_count, _, _) =
        submit_h26x_from_asset(&decoder, SID, "test_h264_bt709_ip.mp4", "h264", 0);
    total_submitted += h264_count;

    // Hop 2: HEVC (triggers CodecChanged on H264 session)
    let pts_offset = total_submitted as i64 * FRAME_DUR_NS;
    let (hevc_count, _, _) =
        submit_h26x_from_asset(&decoder, SID, "test_hevc_bt709_ip.mp4", "hevc", pts_offset);
    total_submitted += hevc_count;

    // Drain H264 session.
    let ev1 = drain_until_stopped(&rx);
    let d1 = count_decoded(&ev1);
    assert_eq!(d1, h264_count as usize, "H264 hop1; events: {ev1:#?}");
    total_decoded += d1;

    // Hop 3: H264 again (triggers CodecChanged on HEVC session)
    let pts_offset2 = total_submitted as i64 * FRAME_DUR_NS;
    let (h264_count2, _, _) =
        submit_h26x_from_asset(&decoder, SID, "test_h264_bt709_ip.mp4", "h264", pts_offset2);
    total_submitted += h264_count2;

    // Drain HEVC session.
    let ev2 = drain_until_stopped(&rx);
    let d2 = count_decoded(&ev2);
    assert_eq!(d2, hevc_count as usize, "HEVC hop2; events: {ev2:#?}");
    total_decoded += d2;

    // Close the final H264 session with explicit EOS to drain cleanly.
    decoder
        .submit_eos(&EndOfStream::new(SID), SUBMIT_TIMEOUT)
        .expect("submit_eos hop3");
    let ev3 = drain_until_stopped(&rx);
    let d3 = count_decoded(&ev3);
    assert_eq!(d3, h264_count2 as usize, "H264 hop3; events: {ev3:#?}");
    total_decoded += d3;

    decoder.shutdown();

    assert_eq!(
        total_decoded, total_submitted as usize,
        "total across all hops"
    );
    eprintln!("OK multi-hop: submitted={total_submitted}, decoded={total_decoded}");
}
