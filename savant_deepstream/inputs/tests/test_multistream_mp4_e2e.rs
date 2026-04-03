//! E2E: [`MultiStreamDecoder`] fed from real MP4 assets + [`Mp4Demuxer`](savant_gstreamer::mp4_demuxer::Mp4Demuxer).

mod common;

use common::*;
use deepstream_inputs::multistream_decoder::{
    DecoderOutput, EvictionVerdict, MultiStreamDecoder, MultiStreamDecoderConfig, StopReason,
    UndecodedReason,
};
use savant_core::primitives::eos::EndOfStream;
use savant_core::primitives::frame::VideoFrameProxy;
use savant_gstreamer::mp4_demuxer::{DemuxedPacket, Mp4Demuxer};
use savant_gstreamer::Codec as GstCodec;
use serial_test::serial;
use std::collections::{HashMap, HashSet};
use std::sync::mpsc;
use std::time::Duration;

const DRAIN_TIMEOUT: Duration = Duration::from_secs(120);

fn drain_until_stopped(rx: &mpsc::Receiver<DecoderOutput>, entry_file: &str) -> (usize, usize) {
    let mut decoded = 0usize;
    let mut undecoded_kf = 0usize;
    loop {
        match rx.recv_timeout(DRAIN_TIMEOUT) {
            Ok(DecoderOutput::StreamStarted { .. }) => {}
            Ok(DecoderOutput::Decoded { frame, .. }) => {
                assert_eq!(
                    frame.get_time_base(),
                    (1, 1_000_000_000),
                    "{entry_file}: decoded frame must use GStreamer ns time_base"
                );
                decoded += 1;
            }
            Ok(DecoderOutput::Undecoded { reason, .. }) => {
                if reason == UndecodedReason::AwaitingKeyframe {
                    undecoded_kf += 1;
                } else {
                    panic!("{entry_file}: unexpected Undecoded {reason:?}");
                }
            }
            Ok(DecoderOutput::Eos { .. }) => {}
            Ok(DecoderOutput::StreamStopped { reason, .. }) => {
                assert_eq!(reason, StopReason::Eos, "{entry_file}");
                break;
            }
            Ok(DecoderOutput::PipelineRestarted { reason, .. }) => {
                panic!("{entry_file}: unexpected PipelineRestarted {reason}");
            }
            Err(_) => panic!("{entry_file}: timeout waiting for decoder output"),
        }
    }
    (decoded, undecoded_kf)
}

/// Pull all video packets; `parsed == true` → Annex-B-style elementary stream (H.264/H.265 parsers).
fn pull_demuxed(mp4_str: &str, parsed: bool) -> (Vec<DemuxedPacket>, GstCodec) {
    let mut demuxer = if parsed {
        Mp4Demuxer::new_parsed(mp4_str)
    } else {
        Mp4Demuxer::new(mp4_str)
    }
    .unwrap_or_else(|e| panic!("demux {mp4_str}: {e}"));
    let mut packets = Vec::new();
    loop {
        match demuxer.pull_timeout(Duration::from_secs(5)) {
            Ok(Some(p)) => packets.push(p),
            Ok(None) => break,
            Err(e) => panic!("pull {mp4_str}: {e}"),
        }
    }
    demuxer.finish();
    let codec = demuxer
        .detected_codec()
        .unwrap_or_else(|| panic!("no codec after demux {mp4_str}"));
    (packets, codec)
}

/// Submit one asset worth of frames for `source_id` without EOS.
///
/// - H.264 / HEVC + `h26x_as_annexb_aus` → concatenate parsed demux payload, split Annex-B AUs.
/// - H.264 / HEVC + `!h26x_as_annexb_aus` → one submit per demuxed sample (MP4 length-prefixed / AVCC HVCC ES).
/// - Other codecs → one submit per parsed demux packet (`demux_parsed` should be true).
fn submit_segment_packets(
    decoder: &MultiStreamDecoder,
    source_id: &str,
    entry: &AssetEntry,
    packets: &[DemuxedPacket],
    codec_str: &str,
    h26x_as_annexb_aus: bool,
) -> u32 {
    if matches!(entry.codec.as_str(), "h264" | "hevc") && h26x_as_annexb_aus {
        let mut bytestream = Vec::new();
        for pkt in packets {
            bytestream.extend_from_slice(&pkt.data);
        }
        let nalus = split_annexb_nalus(&bytestream, &entry.codec);
        let access_units = group_nalus_to_access_units(&entry.codec, nalus);
        assert!(
            access_units.len() >= entry.num_frames as usize,
            "{}: AU count {} < {}",
            entry.file,
            access_units.len(),
            entry.num_frames
        );
        let dur = 33_333_333i64;
        for (i, au) in access_units
            .iter()
            .take(entry.num_frames as usize)
            .enumerate()
        {
            let pts = i as i64 * dur;
            let frame = make_video_frame_ns(
                source_id,
                codec_str,
                entry.width as i64,
                entry.height as i64,
                pts,
                Some(pts),
                Some(dur),
                None,
            );
            decoder
                .submit(frame, Some(au.as_slice()), SUBMIT_TIMEOUT)
                .unwrap_or_else(|e| panic!("submit {} AU {i}: {e}", entry.file));
        }
        entry.num_frames
    } else if matches!(entry.codec.as_str(), "h264" | "hevc") && !h26x_as_annexb_aus {
        assert!(
            packets.len() >= entry.num_frames as usize,
            "{}: ES packet count {} < {}",
            entry.file,
            packets.len(),
            entry.num_frames
        );
        for pkt in packets.iter().take(entry.num_frames as usize) {
            let ordering_ts = pkt.dts_ns.unwrap_or(pkt.pts_ns) as i64;
            let dts = pkt.dts_ns.map(|v| v as i64);
            let dur = pkt.duration_ns.map(|v| v as i64);
            let frame = make_video_frame_ns(
                source_id,
                codec_str,
                entry.width as i64,
                entry.height as i64,
                ordering_ts,
                dts,
                dur,
                Some(pkt.is_keyframe),
            );
            decoder
                .submit(frame, Some(pkt.data.as_slice()), SUBMIT_TIMEOUT)
                .unwrap_or_else(|e| panic!("submit {} ES: {e}", entry.file));
        }
        entry.num_frames
    } else {
        assert!(
            packets.len() >= entry.num_frames as usize,
            "{}: packet count",
            entry.file
        );
        for pkt in packets.iter().take(entry.num_frames as usize) {
            let ordering_ts = pkt.dts_ns.unwrap_or(pkt.pts_ns) as i64;
            let dts = pkt.dts_ns.map(|v| v as i64);
            let dur = pkt.duration_ns.map(|v| v as i64);
            let frame = make_video_frame_ns(
                source_id,
                codec_str,
                entry.width as i64,
                entry.height as i64,
                ordering_ts,
                dts,
                dur,
                Some(pkt.is_keyframe),
            );
            decoder
                .submit(frame, Some(pkt.data.as_slice()), SUBMIT_TIMEOUT)
                .unwrap_or_else(|e| panic!("submit {}: {e}", entry.file));
        }
        entry.num_frames
    }
}

fn manifest_entry_by_file<'a>(manifest: &'a Manifest, name: &str) -> Option<&'a AssetEntry> {
    manifest.assets.iter().find(|e| e.file == name)
}

fn run_multistream_mp4_single(entry: &AssetEntry) {
    let mp4_path = assets_dir().join(&entry.file);
    if !mp4_path.exists() {
        eprintln!("  SKIP (missing file) {}", entry.file);
        return;
    }
    let mp4_str = mp4_path.to_str().unwrap();

    let (demuxed_packets, gst_codec) = pull_demuxed(mp4_str, true);
    let codec_str = codec_to_str(gst_codec);
    let h26x_au = matches!(entry.codec.as_str(), "h264" | "hevc");

    let source_id = entry.file.clone();
    let (tx, rx) = mpsc::channel();
    let cfg = MultiStreamDecoderConfig::new(0, 8).idle_timeout(Duration::from_secs(600));
    let mut decoder = MultiStreamDecoder::new(
        cfg,
        move |o| {
            let _ = tx.send(o);
        },
        None::<fn(&str) -> EvictionVerdict>,
    );

    let submitted = submit_segment_packets(
        &decoder,
        &source_id,
        entry,
        &demuxed_packets,
        codec_str,
        h26x_au,
    );

    let eos = EndOfStream::new(&source_id);
    decoder
        .submit_eos(&eos, SUBMIT_TIMEOUT)
        .unwrap_or_else(|e| panic!("submit_eos {}: {e}", entry.file));

    let (decoded, undecoded_kf) = drain_until_stopped(&rx, &entry.file);
    decoder.shutdown();

    assert_eq!(
        submitted, entry.num_frames,
        "{}: submitted vs manifest",
        entry.file
    );
    assert_eq!(
        decoded + undecoded_kf,
        submitted as usize,
        "{}: decoded({decoded}) + undecoded_kf({undecoded_kf}) != submitted({submitted})",
        entry.file
    );
    assert_eq!(
        decoded, submitted as usize,
        "{}: expected all frames decoded (undecoded_kf={undecoded_kf})",
        entry.file
    );

    eprintln!(
        "  OK {}: submitted={submitted}, decoded={decoded}",
        entry.file
    );
}

#[test]
#[serial]
fn test_multistream_mp4_single_stream_all_codecs() {
    init();

    let platform_tag = current_platform_tag();
    eprintln!("Detected GPU platform tag: {platform_tag}");

    let manifest = load_manifest();
    let mut tested = 0;
    let mut skipped_platform = 0;
    let mut skipped_missing = 0;

    for entry in &manifest.assets {
        if entry.container.as_deref() != Some("mp4") {
            continue;
        }
        if !asset_supported_on_platform(entry, &platform_tag) {
            eprintln!("  SKIP (platform) {}", entry.file);
            skipped_platform += 1;
            continue;
        }

        if !assets_dir().join(&entry.file).exists() {
            skipped_missing += 1;
            continue;
        }

        run_multistream_mp4_single(entry);
        tested += 1;
    }

    eprintln!(
        "\nSummary: tested={tested}, skipped_platform={skipped_platform}, skipped_missing={skipped_missing}"
    );
    let mp4_count = manifest
        .assets
        .iter()
        .filter(|e| e.container.as_deref() == Some("mp4"))
        .count();
    assert!(
        tested > 0 || skipped_platform + skipped_missing == mp4_count,
        "No MP4 assets were tested and not all were expected to be skipped"
    );
}

/// One step in [`test_same_source_sequential_incompatible_mp4_codecs`].
struct SequentialSegment {
    file: &'static str,
    /// `true` → [`Mp4Demuxer::new_parsed`] (byte-stream for H.26x); `false` → [`Mp4Demuxer::new`] (MP4 ES).
    demux_parsed: bool,
    /// For H.264/H.265 only: split Annex-B access units; `false` = one sample per demux packet (AVCC/HVCC).
    h26x_annexb_aus: bool,
}

/// Feed Annex-B then AVCC/HVCC (same H.264 asset), HEVC Annex-B + ES, VP8, VP9 — all on one `source_id` with EOS between clips.
#[test]
#[serial]
fn test_same_source_sequential_incompatible_mp4_codecs() {
    init();

    let platform_tag = current_platform_tag();
    let manifest = load_manifest();

    let plan: &[SequentialSegment] = &[
        SequentialSegment {
            file: "test_h264_bt709_ip.mp4",
            demux_parsed: true,
            h26x_annexb_aus: true,
        },
        SequentialSegment {
            file: "test_h264_bt709_ip.mp4",
            demux_parsed: false,
            h26x_annexb_aus: false,
        },
        SequentialSegment {
            file: "test_hevc_bt709_ip.mp4",
            demux_parsed: true,
            h26x_annexb_aus: true,
        },
        SequentialSegment {
            file: "test_hevc_bt709_ip.mp4",
            demux_parsed: false,
            h26x_annexb_aus: false,
        },
        SequentialSegment {
            file: "test_vp8_bt709.mp4",
            demux_parsed: true,
            h26x_annexb_aus: false,
        },
        SequentialSegment {
            file: "test_vp9_bt709.mp4",
            demux_parsed: true,
            h26x_annexb_aus: false,
        },
    ];

    let mut runnable: Vec<(&SequentialSegment, &AssetEntry)> = Vec::new();
    for seg in plan {
        let Some(entry) = manifest_entry_by_file(&manifest, seg.file) else {
            continue;
        };
        if entry.container.as_deref() != Some("mp4") {
            continue;
        }
        if !asset_supported_on_platform(entry, &platform_tag) {
            continue;
        }
        if !assets_dir().join(seg.file).exists() {
            continue;
        }
        runnable.push((seg, entry));
    }

    if runnable.len() < 4 {
        eprintln!(
            "SKIP sequential incompatible: only {} segments (need >= 4)",
            runnable.len()
        );
        return;
    }

    let expected_total: u32 = runnable.iter().map(|(_, e)| e.num_frames).sum();

    let (tx, rx) = mpsc::channel();
    let cfg = MultiStreamDecoderConfig::new(0, 16)
        .idle_timeout(Duration::from_secs(600))
        .per_stream_queue_size(32);
    let mut decoder = MultiStreamDecoder::new(
        cfg,
        move |o| {
            let _ = tx.send(o);
        },
        None::<fn(&str) -> EvictionVerdict>,
    );

    const SID: &str = "same_source_codec_hops";
    let mut sum_decoded = 0usize;

    for (i, (seg, entry)) in runnable.iter().enumerate() {
        let mp4_str = assets_dir().join(seg.file).to_str().unwrap().to_string();
        let (packets, gst_codec) = pull_demuxed(&mp4_str, seg.demux_parsed);
        let codec_str = codec_to_str(gst_codec);
        let h26x_au = seg.h26x_annexb_aus && matches!(entry.codec.as_str(), "h264" | "hevc");
        let submitted = submit_segment_packets(&decoder, SID, entry, &packets, codec_str, h26x_au);
        assert_eq!(submitted, entry.num_frames, "segment {i} {}", seg.file);

        decoder
            .submit_eos(&EndOfStream::new(SID), SUBMIT_TIMEOUT)
            .unwrap_or_else(|e| panic!("submit_eos segment {i} {}: {e}", seg.file));

        let label = format!("seg{i}_{}", seg.file);
        let (decoded, kf) = drain_until_stopped(&rx, &label);
        assert_eq!(
            decoded + kf,
            entry.num_frames as usize,
            "{label} decoded+kf vs manifest"
        );
        assert_eq!(
            decoded, entry.num_frames as usize,
            "{label}: expect full decode (kf={kf})"
        );
        sum_decoded += decoded;
    }

    assert_eq!(
        sum_decoded, expected_total as usize,
        "total decoded across segments"
    );

    decoder.shutdown();
    eprintln!(
        "OK same_source sequential: {} segments, {} frames decoded",
        runnable.len(),
        sum_decoded
    );
}

/// One compressed packet ready for [`MultiStreamDecoder::submit`].
type PreparedPacket = (VideoFrameProxy, Vec<u8>);

fn prepare_mp4_asset(entry: &AssetEntry) -> Option<Vec<PreparedPacket>> {
    let mp4_path = assets_dir().join(&entry.file);
    if !mp4_path.exists() {
        return None;
    }
    let mp4_str = mp4_path.to_str().unwrap();
    let mut demuxer = Mp4Demuxer::new_parsed(mp4_str).ok()?;
    let mut demuxed_packets = Vec::new();
    loop {
        match demuxer.pull_timeout(Duration::from_secs(5)) {
            Ok(Some(pkt)) => demuxed_packets.push(pkt),
            Ok(None) => break,
            Err(_) => return None,
        }
    }
    demuxer.finish();
    let gst_codec = demuxer.detected_codec()?;
    let codec_str = codec_to_str(gst_codec);
    let source_id = entry.file.clone();

    let mut out = Vec::new();

    if matches!(entry.codec.as_str(), "h264" | "hevc") {
        let mut bytestream = Vec::new();
        for pkt in &demuxed_packets {
            bytestream.extend_from_slice(&pkt.data);
        }
        let nalus = split_annexb_nalus(&bytestream, &entry.codec);
        let access_units = group_nalus_to_access_units(&entry.codec, nalus);
        let dur = 33_333_333i64;
        for (i, au) in access_units
            .iter()
            .take(entry.num_frames as usize)
            .enumerate()
        {
            let pts = i as i64 * dur;
            let frame = make_video_frame_ns(
                &source_id,
                codec_str,
                entry.width as i64,
                entry.height as i64,
                pts,
                Some(pts),
                Some(dur),
                None,
            );
            out.push((frame, au.clone()));
        }
    } else {
        for pkt in demuxed_packets.iter().take(entry.num_frames as usize) {
            let ordering_ts = pkt.dts_ns.unwrap_or(pkt.pts_ns) as i64;
            let dts = pkt.dts_ns.map(|v| v as i64);
            let dur = pkt.duration_ns.map(|v| v as i64);
            let frame = make_video_frame_ns(
                &source_id,
                codec_str,
                entry.width as i64,
                entry.height as i64,
                ordering_ts,
                dts,
                dur,
                Some(pkt.is_keyframe),
            );
            out.push((frame, pkt.data.clone()));
        }
    }

    Some(out)
}

#[test]
#[serial]
fn test_multistream_concurrent_streams() {
    init();

    let platform_tag = current_platform_tag();
    let manifest = load_manifest();

    let want_files = [
        "test_h264_bt709_ip.mp4",
        "test_jpeg_bt709.mp4",
        "test_vp9_bt709.mp4",
    ];
    let mut entries: Vec<&AssetEntry> = Vec::new();
    for name in want_files {
        let Some(e) = manifest.assets.iter().find(|a| a.file == name) else {
            continue;
        };
        if e.container.as_deref() != Some("mp4") {
            continue;
        }
        if !asset_supported_on_platform(e, &platform_tag) {
            continue;
        }
        if !assets_dir().join(&e.file).exists() {
            continue;
        }
        entries.push(e);
    }

    if entries.len() < 2 {
        eprintln!("SKIP concurrent test: not enough assets / HW on this platform");
        return;
    }

    let mut batches: Vec<Vec<PreparedPacket>> = Vec::new();
    for e in &entries {
        let Some(b) = prepare_mp4_asset(e) else {
            panic!("prepare failed for {}", e.file);
        };
        batches.push(b);
    }

    let (tx, rx) = mpsc::channel();
    let cfg = MultiStreamDecoderConfig::new(0, 16)
        .idle_timeout(Duration::from_secs(600))
        .per_stream_queue_size(32);
    let mut decoder = MultiStreamDecoder::new(
        cfg,
        move |o| {
            let _ = tx.send(o);
        },
        None::<fn(&str) -> EvictionVerdict>,
    );

    let max_len = batches.iter().map(|b| b.len()).max().unwrap();
    for round in 0..max_len {
        for batch in &batches {
            if round < batch.len() {
                let (ref frame, ref data) = batch[round];
                decoder
                    .submit(frame.clone(), Some(data.as_slice()), SUBMIT_TIMEOUT)
                    .unwrap_or_else(|e| panic!("concurrent submit round {round}: {e}"));
            }
        }
    }

    for e in &entries {
        let sid = e.file.as_str();
        let eos = EndOfStream::new(sid);
        decoder
            .submit_eos(&eos, SUBMIT_TIMEOUT)
            .unwrap_or_else(|e| panic!("submit_eos {sid}: {e}"));
    }

    let mut per_source_decoded: HashMap<String, usize> = HashMap::new();
    let need: HashSet<String> = entries.iter().map(|e| e.file.clone()).collect();
    let mut stopped: HashSet<String> = HashSet::new();

    while stopped.len() < need.len() {
        match rx.recv_timeout(DRAIN_TIMEOUT) {
            Ok(DecoderOutput::Decoded { frame, .. }) => {
                assert_eq!(frame.get_time_base(), (1, 1_000_000_000));
                *per_source_decoded.entry(frame.get_source_id()).or_insert(0) += 1;
            }
            Ok(DecoderOutput::Undecoded { reason, .. }) => {
                if reason != UndecodedReason::AwaitingKeyframe {
                    panic!("unexpected Undecoded {reason:?}");
                }
            }
            Ok(DecoderOutput::StreamStopped { source_id, reason }) => {
                assert_eq!(reason, StopReason::Eos);
                stopped.insert(source_id);
            }
            Ok(DecoderOutput::StreamStarted { .. }) | Ok(DecoderOutput::Eos { .. }) => {}
            Ok(DecoderOutput::PipelineRestarted { reason, .. }) => {
                panic!("unexpected PipelineRestarted {reason}");
            }
            Err(_) => panic!("timeout concurrent drain"),
        }
    }

    for e in &entries {
        let n = per_source_decoded.get(&e.file).copied().unwrap_or(0);
        assert_eq!(
            n, e.num_frames as usize,
            "{}: decoded {n} expected {}",
            e.file, e.num_frames
        );
    }

    decoder.shutdown();
    eprintln!(
        "OK concurrent streams: {:?}",
        entries.iter().map(|e| &e.file).collect::<Vec<_>>()
    );
}

#[test]
#[serial]
fn test_multistream_foreign_timebase() {
    init();

    let platform_tag = current_platform_tag();
    let manifest = load_manifest();
    let Some(entry) = manifest
        .assets
        .iter()
        .find(|a| a.file == "test_jpeg_bt709.mp4")
    else {
        eprintln!("SKIP: jpeg asset not in manifest");
        return;
    };
    if entry.container.as_deref() != Some("mp4") {
        return;
    }
    if !asset_supported_on_platform(entry, &platform_tag) {
        eprintln!("SKIP (platform)");
        return;
    }
    let mp4_path = assets_dir().join(&entry.file);
    if !mp4_path.exists() {
        eprintln!("SKIP (missing file)");
        return;
    }

    let mp4_str = mp4_path.to_str().unwrap();
    let mut demuxer = Mp4Demuxer::new_parsed(mp4_str).expect("demux");
    let mut packets = Vec::new();
    loop {
        match demuxer.pull_timeout(Duration::from_secs(5)) {
            Ok(Some(p)) => packets.push(p),
            Ok(None) => break,
            Err(e) => panic!("pull {e}"),
        }
    }
    demuxer.finish();
    let gst_codec = demuxer.detected_codec().expect("codec");
    let codec_str = codec_to_str(gst_codec);
    let source_id = "foreign_tb_jpeg";

    let (tx, rx) = mpsc::channel();
    let cfg = MultiStreamDecoderConfig::new(0, 8).idle_timeout(Duration::from_secs(600));
    let mut decoder = MultiStreamDecoder::new(
        cfg,
        move |o| {
            let _ = tx.send(o);
        },
        None::<fn(&str) -> EvictionVerdict>,
    );

    /// PTS after `time_base` (1/90_000) → ns, matching `normalize_frame_to_gst_ns` in
    /// `savant_core::primitives::gstreamer_frame_time`.
    fn ticks_90k_to_ns(ticks: i64) -> i64 {
        let v = (ticks as i128)
            .saturating_mul(1_000_000_000)
            .saturating_mul(1)
            / 90_000;
        v.min(i64::MAX as i128) as i64
    }

    let mut expected_pts_ns: Vec<i64> = Vec::new();
    for (i, pkt) in packets.iter().take(entry.num_frames as usize).enumerate() {
        // Strictly increasing 90 kHz ticks (~30 fps) so `submission_order_ns` never regresses
        // after truncation from the demuxer timeline.
        let pts_90k = (i as i64) * 3_000;
        let dts_90k = Some(pts_90k);
        let dur_90k = Some(3_000i64);

        let frame = make_video_frame_scaled(
            source_id,
            codec_str,
            entry.width as i64,
            entry.height as i64,
            (1, 90_000),
            pts_90k,
            dts_90k,
            dur_90k,
            Some(pkt.is_keyframe),
        );
        // Expect normalization to match rational conversion from the 90 kHz ticks we encoded.
        expected_pts_ns.push(ticks_90k_to_ns(pts_90k));
        decoder
            .submit(frame, Some(pkt.data.as_slice()), SUBMIT_TIMEOUT)
            .unwrap();
    }

    decoder
        .submit_eos(&EndOfStream::new(source_id), SUBMIT_TIMEOUT)
        .unwrap();

    let mut idx = 0usize;
    loop {
        match rx.recv_timeout(DRAIN_TIMEOUT) {
            Ok(DecoderOutput::Decoded { frame, .. }) => {
                assert_eq!(frame.get_time_base(), (1, 1_000_000_000));
                let exp = expected_pts_ns[idx];
                let got = frame.get_pts();
                assert_eq!(got, exp, "PTS mismatch after normalization (idx {idx})");
                idx += 1;
            }
            Ok(DecoderOutput::StreamStopped { reason, .. }) => {
                assert_eq!(reason, StopReason::Eos);
                break;
            }
            Ok(DecoderOutput::Undecoded { reason, .. }) => {
                panic!("unexpected undecoded {reason:?}");
            }
            Ok(_) => {}
            Err(_) => panic!("timeout"),
        }
    }

    assert_eq!(idx, entry.num_frames as usize);
    decoder.shutdown();
}
