mod common;

use common::*;
use deepstream_decoders::prelude::*;
use serial_test::serial;
use std::sync::mpsc;
use std::time::Duration;

fn run_annexb_e2e(entry: &AssetEntry) {
    let file_path = assets_dir().join(&entry.file);
    let bitstream = std::fs::read(&file_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", file_path.display()));
    let nalus = split_annexb_nalus(&bitstream);
    assert!(
        !nalus.is_empty(),
        "{}: no Annex-B NAL units found",
        entry.file
    );

    let access_units = group_nalus_to_access_units(&entry.codec, nalus);
    assert!(
        access_units.len() >= entry.num_frames as usize,
        "{}: access unit count ({}) < expected ({})",
        entry.file,
        access_units.len(),
        entry.num_frames
    );

    let config = decoder_config_annexb(&entry.codec);
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

    let dur = 33_333_333u64;
    for (i, au) in access_units
        .iter()
        .take(entry.num_frames as usize)
        .enumerate()
    {
        let pts = i as u64 * dur;
        decoder
            .submit_packet(au, i as u128, pts, Some(pts), Some(dur))
            .unwrap_or_else(|e| panic!("submit_packet failed for {} at AU {i}: {e}", entry.file));
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
        decoded_count, entry.num_frames as usize,
        "{}: decoded ({}) != submitted ({})",
        entry.file, decoded_count, entry.num_frames
    );
}

#[test]
#[serial]
fn test_minimal_nvdecoder_annexb() {
    init();
    if !has_nvdec() {
        eprintln!("skip: no nvv4l2decoder");
        return;
    }

    let path = assets_dir().join("test_h264_annexb_ip.h264");
    let data = std::fs::read(&path).unwrap();
    let nalus = split_annexb_nalus(&data);
    let aus = group_nalus_to_access_units("h264", nalus);

    let config = DecoderConfig::H264(H264DecoderConfig::new(H264StreamFormat::ByteStream));
    let (tx, rx) = mpsc::channel();
    let mut decoder = NvDecoder::new(
        0,
        &config,
        make_rgba_pool(320, 240),
        identity_transform_config(),
        move |ev| {
            let _ = tx.send(ev);
        },
    )
    .unwrap();

    let dur = 33_333_333u64;
    for (i, au) in aus.iter().take(8).enumerate() {
        let pts = i as u64 * dur;
        decoder
            .submit_packet(au, i as u128, pts, Some(pts), Some(dur))
            .unwrap();
    }
    decoder.send_eos().unwrap();

    let mut decoded_count = 0usize;
    loop {
        match rx.recv_timeout(Duration::from_secs(10)) {
            Ok(DecoderEvent::Frame(_)) => decoded_count += 1,
            Ok(DecoderEvent::Eos) => break,
            Ok(DecoderEvent::Error(e)) => panic!("error: {e}"),
            Ok(DecoderEvent::PipelineRestarted { reason, .. }) => panic!("restart: {reason}"),
            Err(_) => panic!("TIMEOUT after {decoded_count} frames"),
        }
    }
    assert_eq!(decoded_count, 8);
}

#[test]
#[serial]
fn test_annexb_e2e_all_assets() {
    init();
    if !has_nvdec() {
        eprintln!("skip: no nvv4l2decoder");
        return;
    }

    let platform_tag = current_platform_tag();
    let manifest = load_manifest();

    let mut tested = 0usize;
    for entry in &manifest.assets {
        if entry.container.as_deref() != Some("raw") {
            continue;
        }
        if !matches!(entry.codec.as_str(), "h264" | "hevc") {
            continue;
        }
        if !asset_supported_on_platform(entry, &platform_tag) {
            continue;
        }
        run_annexb_e2e(entry);
        tested += 1;
    }

    assert!(tested > 0, "no raw annexb assets were tested");
}
