//! Smoke tests: immediate EOS after frame submission must deliver every
//! in-flight decoded frame **and** the `Eos` event before `StreamStopped`.
//!
//! Covers both JPEG (single-frame pipeline) and H264 (multi-frame pipeline
//! with potential internal buffering).

mod common;

use common::*;
use deepstream_encoders::prelude::*;
use deepstream_inputs::multistream_decoder::{
    DecoderOutput, EvictionVerdict, MultiStreamDecoder, MultiStreamDecoderConfig, StopReason,
};
use savant_core::primitives::eos::EndOfStream;
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
};
use savant_core::primitives::video_codec::VideoCodec;
use savant_gstreamer::mp4_demuxer::Mp4Demuxer;
use savant_gstreamer::Codec;
use serial_test::serial;
use std::sync::mpsc;
use std::time::Duration;

const FRAME_DUR_NS: i64 = 33_333_333;
const RECV_TIMEOUT: Duration = Duration::from_secs(30);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_decoder() -> (MultiStreamDecoder, mpsc::Receiver<DecoderOutput>) {
    let (tx, rx) = mpsc::channel();
    let cfg = MultiStreamDecoderConfig::new(0, 8).idle_timeout(Duration::from_secs(600));
    let decoder = MultiStreamDecoder::new(
        cfg,
        move |o| {
            let _ = tx.send(o);
        },
        None::<fn(&str) -> EvictionVerdict>,
    );
    (decoder, rx)
}

/// Drain the output channel until `StreamStopped`, collecting event names.
fn drain_events(rx: &mpsc::Receiver<DecoderOutput>) -> Vec<String> {
    let mut events = Vec::new();
    loop {
        match rx.recv_timeout(RECV_TIMEOUT) {
            Ok(DecoderOutput::Decoded { .. }) => events.push("Decoded".into()),
            Ok(DecoderOutput::StreamStarted { .. }) => events.push("StreamStarted".into()),
            Ok(DecoderOutput::Eos { .. }) => events.push("Eos".into()),
            Ok(DecoderOutput::Undecoded { reason, .. }) => {
                events.push(format!("Undecoded({reason:?})"))
            }
            Ok(DecoderOutput::StreamStopped { reason, .. }) => {
                events.push(format!("StreamStopped({reason:?})"));
                break;
            }
            Ok(DecoderOutput::PipelineRestarted { reason, .. }) => {
                events.push(format!("PipelineRestarted({reason})"));
            }
            Err(_) => panic!("timeout after {RECV_TIMEOUT:?} — events so far: {events:?}"),
        }
    }
    events
}

/// Assert event sequence contains: ≥ `min_decoded` × Decoded, exactly 1 Eos,
/// last event is `StreamStopped(Eos)`, and `Eos` comes after all `Decoded`.
fn assert_full_delivery(events: &[String], min_decoded: usize, ctx: &str) {
    let decoded_count = events.iter().filter(|e| *e == "Decoded").count();
    assert!(
        decoded_count >= min_decoded,
        "{ctx}: expected ≥{min_decoded} Decoded, got {decoded_count}; events: {events:?}"
    );

    assert!(
        events.contains(&"Eos".to_string()),
        "{ctx}: Eos event missing; events: {events:?}"
    );

    assert_eq!(
        events.last().unwrap(),
        &format!("StreamStopped({:?})", StopReason::Eos),
        "{ctx}: last event must be StreamStopped(Eos); events: {events:?}"
    );

    let last_decoded_pos = events.iter().rposition(|e| e == "Decoded");
    let eos_pos = events.iter().position(|e| e == "Eos");
    if let (Some(ld), Some(ep)) = (last_decoded_pos, eos_pos) {
        assert!(
            ld < ep,
            "{ctx}: Eos must come after all Decoded; events: {events:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// JPEG helpers
// ---------------------------------------------------------------------------

fn encode_jpeg_blob(w: u32, h: u32) -> Vec<u8> {
    let config = EncoderConfig::new(Codec::Jpeg, w, h).format(VideoFormat::I420);
    let mut encoder = NvEncoder::new(&config).expect("JPEG NvEncoder");
    let shared = encoder.generator().acquire(Some(0)).expect("acquire");
    let buf = shared.into_buffer().expect("into_buffer");
    encoder
        .submit_frame(buf, 0, 0, Some(FRAME_DUR_NS as u64))
        .expect("submit_frame");
    let frames = encoder.finish(Some(5000)).expect("encoder finish");
    assert!(!frames.is_empty());
    frames.into_iter().next().unwrap().data
}

fn make_jpeg_frame(source_id: &str, w: i64, h: i64, pts_ns: i64) -> VideoFrameProxy {
    VideoFrameProxy::new(
        source_id,
        (30, 1),
        w,
        h,
        VideoFrameContent::None,
        VideoFrameTranscodingMethod::Copy,
        Some(VideoCodec::Jpeg),
        Some(true),
        (1, 1_000_000_000),
        pts_ns,
        None,
        Some(FRAME_DUR_NS),
    )
    .expect("VideoFrameProxy")
}

// ---------------------------------------------------------------------------
// JPEG tests
// ---------------------------------------------------------------------------

/// 1 JPEG + immediate EOS → must get `Decoded`, `Eos`, `StreamStopped(Eos)`.
#[test]
#[serial]
fn test_fast_eos_single_jpeg() {
    init();

    let blob = encode_jpeg_blob(1280, 720);
    let (mut decoder, rx) = make_decoder();

    let frame = make_jpeg_frame("fast_jpeg", 1280, 720, 0);
    decoder
        .submit(frame, Some(&blob), Duration::from_secs(5))
        .expect("submit");
    decoder
        .submit_eos(&EndOfStream::new("fast_jpeg"), Duration::from_secs(5))
        .expect("eos");

    let events = drain_events(&rx);
    decoder.wait_for_pending_teardowns();
    assert_full_delivery(&events, 1, "single JPEG");

    decoder.shutdown();
}

/// Repeat JPEG + EOS 20 times on the same source_id.
#[test]
#[serial]
fn test_fast_eos_repeated_jpeg_cycles() {
    init();

    let blob = encode_jpeg_blob(1280, 720);
    let (mut decoder, rx) = make_decoder();

    for i in 0..20 {
        let pts = i as i64 * FRAME_DUR_NS;
        let frame = make_jpeg_frame("repeated_jpeg", 1280, 720, pts);
        decoder
            .submit(frame, Some(&blob), Duration::from_secs(5))
            .expect("submit");
        decoder
            .submit_eos(&EndOfStream::new("repeated_jpeg"), Duration::from_secs(5))
            .expect("eos");

        let events = drain_events(&rx);
        decoder.wait_for_pending_teardowns();
        assert_full_delivery(&events, 1, &format!("JPEG cycle {i}"));
    }

    decoder.shutdown();
}

// ---------------------------------------------------------------------------
// H264 tests
// ---------------------------------------------------------------------------

/// Submit all H264 IP-only frames + immediate EOS → every frame decoded, Eos delivered.
#[test]
#[serial]
fn test_fast_eos_h264_ip_all_frames() {
    init();

    let manifest = load_manifest();
    let platform = current_platform_tag();
    let entry = manifest_entry_by_file(&manifest, "test_h264_bt709_ip.mp4");
    let entry = match entry {
        Some(e) if asset_supported_on_platform(e, &platform) => e,
        _ => {
            eprintln!("SKIP: test_h264_bt709_ip.mp4 not available on {platform}");
            return;
        }
    };

    let mp4_path = assets_dir().join(&entry.file);
    if !mp4_path.exists() {
        eprintln!("SKIP: asset file missing");
        return;
    }

    let mut demuxer = Mp4Demuxer::new_parsed(mp4_path.to_str().unwrap()).expect("demux");
    let mut packets = Vec::new();
    while let Ok(Some(p)) = demuxer.pull_timeout(Duration::from_secs(5)) {
        packets.push(p);
    }
    demuxer.finish();

    let bytestream: Vec<u8> = packets
        .iter()
        .flat_map(|p| p.data.iter().copied())
        .collect();
    let nalus = split_annexb_nalus(&bytestream, Codec::H264);
    let access_units = group_nalus_to_access_units(Codec::H264, nalus);
    let num_frames = entry.num_frames as usize;
    assert!(access_units.len() >= num_frames);

    let (mut decoder, rx) = make_decoder();
    let source_id = "h264_ip_fast_eos";
    let dur_ns = FRAME_DUR_NS;

    for (i, au) in access_units.iter().take(num_frames).enumerate() {
        let pts = i as i64 * dur_ns;
        let frame = make_video_frame_ns(
            source_id,
            VideoCodec::H264,
            entry.width as i64,
            entry.height as i64,
            pts,
            Some(pts),
            Some(dur_ns),
            None,
        );
        decoder
            .submit(frame, Some(au), SUBMIT_TIMEOUT)
            .unwrap_or_else(|e| panic!("submit AU {i}: {e}"));
    }

    decoder
        .submit_eos(&EndOfStream::new(source_id), Duration::from_secs(5))
        .expect("eos");

    let events = drain_events(&rx);
    decoder.wait_for_pending_teardowns();
    assert_full_delivery(&events, num_frames, "H264 IP all frames + EOS");

    decoder.shutdown();
}

/// Submit only the first H264 access unit (IDR) + immediate EOS → 1 Decoded + Eos.
#[test]
#[serial]
fn test_fast_eos_h264_single_idr() {
    init();

    let manifest = load_manifest();
    let platform = current_platform_tag();
    let entry = manifest_entry_by_file(&manifest, "test_h264_bt709_ip.mp4");
    let entry = match entry {
        Some(e) if asset_supported_on_platform(e, &platform) => e,
        _ => {
            eprintln!("SKIP: test_h264_bt709_ip.mp4 not available on {platform}");
            return;
        }
    };

    let mp4_path = assets_dir().join(&entry.file);
    if !mp4_path.exists() {
        eprintln!("SKIP: asset file missing");
        return;
    }

    let mut demuxer = Mp4Demuxer::new_parsed(mp4_path.to_str().unwrap()).expect("demux");
    let mut packets = Vec::new();
    while let Ok(Some(p)) = demuxer.pull_timeout(Duration::from_secs(5)) {
        packets.push(p);
    }
    demuxer.finish();

    let bytestream: Vec<u8> = packets
        .iter()
        .flat_map(|p| p.data.iter().copied())
        .collect();
    let nalus = split_annexb_nalus(&bytestream, Codec::H264);
    let access_units = group_nalus_to_access_units(Codec::H264, nalus);
    assert!(!access_units.is_empty(), "need at least 1 AU");

    let (mut decoder, rx) = make_decoder();
    let source_id = "h264_single_idr";

    let frame = make_video_frame_ns(
        source_id,
        VideoCodec::H264,
        entry.width as i64,
        entry.height as i64,
        0,
        Some(0),
        Some(FRAME_DUR_NS),
        None,
    );
    decoder
        .submit(frame, Some(&access_units[0]), SUBMIT_TIMEOUT)
        .expect("submit IDR");

    decoder
        .submit_eos(&EndOfStream::new(source_id), Duration::from_secs(5))
        .expect("eos");

    let events = drain_events(&rx);
    decoder.wait_for_pending_teardowns();
    assert_full_delivery(&events, 1, "H264 single IDR + EOS");

    decoder.shutdown();
}

/// Repeat: all H264 IP frames + EOS × 5 cycles on the same source_id.
#[test]
#[serial]
fn test_fast_eos_h264_repeated_cycles() {
    init();

    let manifest = load_manifest();
    let platform = current_platform_tag();
    let entry = manifest_entry_by_file(&manifest, "test_h264_bt709_ip.mp4");
    let entry = match entry {
        Some(e) if asset_supported_on_platform(e, &platform) => e,
        _ => {
            eprintln!("SKIP: test_h264_bt709_ip.mp4 not available on {platform}");
            return;
        }
    };

    let mp4_path = assets_dir().join(&entry.file);
    if !mp4_path.exists() {
        eprintln!("SKIP: asset file missing");
        return;
    }

    let mut demuxer = Mp4Demuxer::new_parsed(mp4_path.to_str().unwrap()).expect("demux");
    let mut packets = Vec::new();
    while let Ok(Some(p)) = demuxer.pull_timeout(Duration::from_secs(5)) {
        packets.push(p);
    }
    demuxer.finish();

    let bytestream: Vec<u8> = packets
        .iter()
        .flat_map(|p| p.data.iter().copied())
        .collect();
    let nalus = split_annexb_nalus(&bytestream, Codec::H264);
    let access_units = group_nalus_to_access_units(Codec::H264, nalus);
    let num_frames = entry.num_frames as usize;
    assert!(access_units.len() >= num_frames);

    let (mut decoder, rx) = make_decoder();
    let source_id = "h264_cycles";
    let dur_ns = FRAME_DUR_NS;

    for cycle in 0..5 {
        let base_pts = cycle as i64 * num_frames as i64 * dur_ns;
        for (i, au) in access_units.iter().take(num_frames).enumerate() {
            let pts = base_pts + i as i64 * dur_ns;
            let frame = make_video_frame_ns(
                source_id,
                VideoCodec::H264,
                entry.width as i64,
                entry.height as i64,
                pts,
                Some(pts),
                Some(dur_ns),
                None,
            );
            decoder
                .submit(frame, Some(au), SUBMIT_TIMEOUT)
                .unwrap_or_else(|e| panic!("cycle {cycle} submit AU {i}: {e}"));
        }

        decoder
            .submit_eos(&EndOfStream::new(source_id), Duration::from_secs(5))
            .expect("eos");

        let events = drain_events(&rx);
        decoder.wait_for_pending_teardowns();
        assert_full_delivery(&events, num_frames, &format!("H264 cycle {cycle}"));
    }

    decoder.shutdown();
}

fn manifest_entry_by_file<'a>(manifest: &'a Manifest, name: &str) -> Option<&'a AssetEntry> {
    manifest.assets.iter().find(|e| e.file == name)
}
