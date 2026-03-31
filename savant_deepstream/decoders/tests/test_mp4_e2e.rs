mod common;

use common::*;
use deepstream_decoders::prelude::*;
use savant_gstreamer::mp4_demuxer::Mp4Demuxer;
use serial_test::serial;
use std::sync::mpsc;
use std::time::Duration;

fn run_mp4_e2e(entry: &AssetEntry) {
    let mp4_path = assets_dir().join(&entry.file);
    let mp4_str = mp4_path.to_str().unwrap();

    let config = decoder_config_for_codec(&entry.codec)
        .unwrap_or_else(|| panic!("unsupported codec in manifest: {}", entry.codec));

    let mut demuxer = Mp4Demuxer::new_parsed(mp4_str)
        .unwrap_or_else(|e| panic!("demuxer failed for {}: {e}", entry.file));

    let (tx, rx) = mpsc::channel();
    let mut decoder = NvDecoder::new(
        0,
        &config,
        make_rgba_pool(entry.width, entry.height),
        identity_transform_config(),
        move |ev| {
            let _ = tx.send(ev);
        },
    )
    .unwrap_or_else(|e| panic!("decoder create failed for {}: {e}", entry.file));

    let mut submitted = 0u32;
    let mut demuxed_packets = Vec::new();
    loop {
        match demuxer.pull_timeout(Duration::from_secs(5)) {
            Ok(Some(pkt)) => demuxed_packets.push(pkt),
            Ok(None) => break,
            Err(e) => panic!("demuxer pull error for {}: {e}", entry.file),
        }
    }
    demuxer.finish();

    assert!(
        !demuxed_packets.is_empty(),
        "{}: demuxer produced 0 packets",
        entry.file
    );

    if matches!(entry.codec.as_str(), "h264" | "hevc") {
        let mut bytestream = Vec::new();
        for pkt in &demuxed_packets {
            bytestream.extend_from_slice(&pkt.data);
        }
        let nalus = split_annexb_nalus(&bytestream);
        let access_units = group_nalus_to_access_units(&entry.codec, nalus);
        assert!(
            access_units.len() >= entry.num_frames as usize,
            "{}: reconstructed AU count ({}) < expected ({})",
            entry.file,
            access_units.len(),
            entry.num_frames
        );
        let dur = 33_333_333u64;
        for (i, au) in access_units
            .iter()
            .take(entry.num_frames as usize)
            .enumerate()
        {
            let pts = i as u64 * dur;
            decoder
                .submit_packet(au, i as u128, pts, Some(pts), Some(dur))
                .unwrap_or_else(|e| panic!("submit_packet failed for {} AU {i}: {e}", entry.file));
            submitted += 1;
        }
    } else {
        assert!(
            demuxed_packets.len() >= entry.num_frames as usize,
            "{}: demuxed packet count ({}) < expected ({})",
            entry.file,
            demuxed_packets.len(),
            entry.num_frames
        );
        for pkt in demuxed_packets.iter().take(entry.num_frames as usize) {
            let frame_id = submitted as u128;
            let ordering_ts = pkt.dts_ns.unwrap_or(pkt.pts_ns);
            decoder
                .submit_packet(
                    &pkt.data,
                    frame_id,
                    ordering_ts,
                    pkt.dts_ns,
                    pkt.duration_ns,
                )
                .unwrap_or_else(|e| {
                    panic!(
                        "submit_packet failed for {} frame {submitted}: {e}",
                        entry.file
                    )
                });
            submitted += 1;
        }
    }

    decoder
        .send_eos()
        .unwrap_or_else(|e| panic!("send_eos failed for {}: {e}", entry.file));

    let mut decoded_count = 0usize;
    loop {
        match rx.recv_timeout(Duration::from_secs(30)) {
            Ok(DecoderEvent::Frame(f)) => {
                assert_eq!(
                    f.format,
                    VideoFormat::RGBA,
                    "{}: expected RGBA output",
                    entry.file
                );
                decoded_count += 1;
            }
            Ok(DecoderEvent::Eos) => break,
            Ok(DecoderEvent::Error(e)) => panic!("decoder error for {}: {e}", entry.file),
            Ok(DecoderEvent::PipelineRestarted { reason, .. }) => {
                panic!("unexpected restart for {}: {reason}", entry.file)
            }
            Err(_) => panic!("timeout waiting for decoder events for {}", entry.file),
        }
    }

    assert_eq!(
        submitted, entry.num_frames,
        "{}: submitted packets must match manifest num_frames",
        entry.file
    );
    assert_eq!(
        decoded_count, submitted as usize,
        "{}: decoded ({}) != submitted ({})",
        entry.file, decoded_count, submitted
    );

    eprintln!(
        "  OK {}: submitted={submitted}, decoded={decoded_count}, expected_frames={}",
        entry.file, entry.num_frames
    );
}

#[test]
#[serial]
fn test_mp4_e2e_all_assets() {
    init();

    let platform_tag = current_platform_tag();
    eprintln!("Detected GPU platform tag: {platform_tag}");

    let manifest = load_manifest();
    eprintln!(
        "Manifest: {} assets, filtering for platform '{platform_tag}'",
        manifest.assets.len()
    );

    let mut tested = 0;
    let mut skipped_platform = 0;
    let mut skipped_hw = 0;

    for entry in &manifest.assets {
        if entry.container.as_deref() != Some("mp4") {
            continue;
        }
        if !asset_supported_on_platform(entry, &platform_tag) {
            eprintln!("  SKIP (platform) {}", entry.file);
            skipped_platform += 1;
            continue;
        }

        let needs_nvdec = matches!(
            entry.codec.as_str(),
            "h264" | "hevc" | "vp8" | "vp9" | "av1"
        );
        if needs_nvdec && !has_nvdec() {
            eprintln!("  SKIP (no nvv4l2decoder) {}", entry.file);
            skipped_hw += 1;
            continue;
        }

        if entry.codec == "jpeg" && !has_nvjpegdec() {
            eprintln!("  SKIP (no nvjpegdec) {}", entry.file);
            skipped_hw += 1;
            continue;
        }

        run_mp4_e2e(entry);
        tested += 1;
    }

    eprintln!(
        "\nSummary: tested={tested}, skipped_platform={skipped_platform}, skipped_hw={skipped_hw}"
    );
    let mp4_count = manifest
        .assets
        .iter()
        .filter(|e| e.container.as_deref() == Some("mp4"))
        .count();
    assert!(
        tested > 0 || skipped_platform + skipped_hw == mp4_count,
        "No MP4 assets were tested and not all were expected to be skipped"
    );
}
