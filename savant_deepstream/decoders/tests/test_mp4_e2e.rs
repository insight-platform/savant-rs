mod common;

use common::*;
use deepstream_decoders::prelude::*;
use deepstream_decoders::NvDecoderExt;
use savant_gstreamer::mp4_demuxer::Mp4Demuxer;
use serial_test::serial;
use std::time::Duration;

fn run_mp4_e2e(entry: &AssetEntry) {
    let mp4_path = assets_dir().join(&entry.file);
    let mp4_str = mp4_path.to_str().unwrap();

    let config = decoder_config_for_codec(&entry.codec)
        .unwrap_or_else(|| panic!("unsupported codec in manifest: {}", entry.codec));

    let (demuxed_packets, _codec) = Mp4Demuxer::demux_all_parsed(mp4_str)
        .unwrap_or_else(|e| panic!("demuxer failed for {}: {e}", entry.file));

    let decoder = NvDecoder::new(
        test_decoder_config(0, config),
        make_rgba_pool(entry.width, entry.height),
        identity_transform_config(),
    )
    .unwrap_or_else(|e| panic!("decoder create failed for {}: {e}", entry.file));

    let mut submitted = 0u32;

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
        let nalus = split_annexb_nalus(&bytestream, &entry.codec);
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
        match decoder.recv_timeout(Duration::from_secs(30)) {
            Ok(Some(NvDecoderOutput::Frame(f))) => {
                assert_eq!(
                    f.format,
                    VideoFormat::RGBA,
                    "{}: expected RGBA output",
                    entry.file
                );
                decoded_count += 1;
            }
            Ok(Some(NvDecoderOutput::Eos)) => break,
            Ok(Some(NvDecoderOutput::Error(e))) => panic!("decoder error for {}: {e}", entry.file),
            Ok(Some(NvDecoderOutput::Event(_) | NvDecoderOutput::SourceEos { .. })) => {}
            Ok(None) => panic!("timeout waiting for decoder events for {}", entry.file),
            Err(e) => panic!("recv error for {}: {e}", entry.file),
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

    for entry in &manifest.assets {
        if entry.container.as_deref() != Some("mp4") {
            continue;
        }
        if !asset_supported_on_platform(entry, &platform_tag) {
            eprintln!("  SKIP (platform) {}", entry.file);
            skipped_platform += 1;
            continue;
        }

        run_mp4_e2e(entry);
        tested += 1;
    }

    eprintln!("\nSummary: tested={tested}, skipped_platform={skipped_platform}");
    let mp4_count = manifest
        .assets
        .iter()
        .filter(|e| e.container.as_deref() == Some("mp4"))
        .count();
    assert!(
        tested > 0 || skipped_platform == mp4_count,
        "No MP4 assets were tested and not all were expected to be skipped"
    );
}
