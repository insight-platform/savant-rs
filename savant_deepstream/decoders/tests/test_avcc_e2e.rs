//! End-to-end tests for AVCC (H.264) and HVCC (H.265) stream formats.
//!
//! Reads existing Annex-B raw bitstream files, converts them to
//! length-prefixed (AVCC/HVCC) framing at test time using GStreamer's
//! `h264parse`/`h265parse`, then feeds the AVCC packets + codec_data
//! to `NvDecoder` and verifies the decoded output.

mod common;

use common::*;
use deepstream_decoders::prelude::*;
use serial_test::serial;
use std::sync::mpsc;
use std::time::Duration;

fn run_avcc_e2e(entry: &AssetEntry) {
    let file_path = assets_dir().join(&entry.file);
    let bitstream = std::fs::read(&file_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", file_path.display()));
    let nalus = split_annexb_nalus(&bitstream, &entry.codec);
    assert!(
        !nalus.is_empty(),
        "{}: no Annex-B NAL units found",
        entry.file
    );

    let annexb_aus = group_nalus_to_access_units(&entry.codec, nalus);
    assert!(
        annexb_aus.len() >= entry.num_frames as usize,
        "{}: access unit count ({}) < expected ({})",
        entry.file,
        annexb_aus.len(),
        entry.num_frames
    );

    let aus_to_convert: Vec<Vec<u8>> = annexb_aus
        .into_iter()
        .take(entry.num_frames as usize)
        .collect();

    let conversion = convert_annexb_to_avcc(&entry.codec, &aus_to_convert);
    assert!(
        !conversion.codec_data.is_empty(),
        "{}: empty codec_data from conversion",
        entry.file
    );
    assert!(
        conversion.access_units.len() >= entry.num_frames as usize,
        "{}: AVCC AU count ({}) < expected ({})",
        entry.file,
        conversion.access_units.len(),
        entry.num_frames
    );

    let config = decoder_config_avcc(&entry.codec, conversion.codec_data);
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
    .unwrap_or_else(|e| panic!("decoder create failed for {} (AVCC): {e}", entry.file));

    let dur = 33_333_333u64;
    for (i, au) in conversion
        .access_units
        .iter()
        .take(entry.num_frames as usize)
        .enumerate()
    {
        let pts = i as u64 * dur;
        decoder
            .submit_packet(au, i as u128, pts, Some(pts), Some(dur))
            .unwrap_or_else(|e| {
                panic!(
                    "submit_packet failed for {} (AVCC) at AU {i}: {e}",
                    entry.file
                )
            });
    }

    decoder
        .send_eos()
        .unwrap_or_else(|e| panic!("send_eos failed for {} (AVCC): {e}", entry.file));

    let mut decoded_count = 0usize;
    loop {
        match rx.recv_timeout(Duration::from_secs(30)) {
            Ok(DecoderEvent::Frame(f)) => {
                assert_eq!(
                    f.format,
                    VideoFormat::RGBA,
                    "{} (AVCC): expected RGBA output",
                    entry.file
                );
                decoded_count += 1;
            }
            Ok(DecoderEvent::Eos) => break,
            Ok(DecoderEvent::Error(e)) => {
                panic!("decoder error for {} (AVCC): {e}", entry.file)
            }
            Ok(DecoderEvent::PipelineRestarted { reason, .. }) => {
                panic!("unexpected restart for {} (AVCC): {reason}", entry.file)
            }
            Err(_) => panic!(
                "timeout waiting for decoder events for {} (AVCC), decoded so far: {}",
                entry.file, decoded_count
            ),
        }
    }

    assert_eq!(
        decoded_count, entry.num_frames as usize,
        "{} (AVCC): decoded ({}) != submitted ({})",
        entry.file, decoded_count, entry.num_frames
    );

    eprintln!(
        "  OK {} (AVCC): decoded={decoded_count}, expected={}",
        entry.file, entry.num_frames
    );
}

#[test]
#[serial]
fn test_avcc_e2e_all_assets() {
    init();

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
            eprintln!("  SKIP (platform) {} (AVCC)", entry.file);
            continue;
        }
        run_avcc_e2e(entry);
        tested += 1;
    }

    assert!(tested > 0, "no raw AVCC/HVCC assets were tested");
}
