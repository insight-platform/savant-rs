mod common;

use common::*;
use deepstream_decoders::prelude::*;
use deepstream_encoders::{EncoderConfig, NvEncoder};
use serial_test::serial;
use std::collections::HashSet;
use std::sync::mpsc;
use std::time::Duration;

fn encode_test_frames(codec: Codec, w: u32, h: u32, n: usize) -> Vec<(u128, u64, u64, Vec<u8>)> {
    let config = EncoderConfig::new(codec, w, h);
    let mut encoder = NvEncoder::new(&config).expect("encoder creation");
    let dur = 33_333_333u64;
    for i in 0..n {
        let shared = encoder.generator().acquire(Some(i as u128)).unwrap();
        let buf = shared.into_buffer().unwrap();
        encoder
            .submit_frame(buf, i as u128, i as u64 * dur, Some(dur))
            .unwrap();
    }
    encoder
        .finish(Some(5000))
        .unwrap()
        .into_iter()
        .map(|f| {
            (
                f.frame_id.unwrap_or(0),
                f.pts_ns,
                f.duration_ns.unwrap_or(dur),
                f.data,
            )
        })
        .collect()
}

#[test]
#[serial]
fn test_e2e_h264_decode_to_rgba() {
    init();
    if !has_nvenc() || !has_nvdec() {
        eprintln!("skip: no NVENC/NVDEC");
        return;
    }
    let (w, h) = (320, 240);
    let packets = encode_test_frames(Codec::H264, w, h, 5);
    let config = DecoderConfig::H264(H264DecoderConfig::new(H264StreamFormat::ByteStream));
    let (tx, rx) = mpsc::channel();
    let mut decoder = NvDecoder::new(
        0,
        &config,
        make_rgba_pool(w, h),
        identity_transform_config(),
        move |ev| {
            let _ = tx.send(ev);
        },
    )
    .unwrap();
    for (fid, pts, dur, data) in &packets {
        decoder
            .submit_packet(data, *fid, *pts, Some(*pts), Some(*dur))
            .unwrap();
    }
    decoder.send_eos().unwrap();
    let submitted_ids: HashSet<u128> = packets.iter().map(|(fid, _, _, _)| *fid).collect();
    let submitted_pts: HashSet<u64> = packets.iter().map(|(_, pts, _, _)| *pts).collect();
    let mut decoded_ids = HashSet::new();
    let mut decoded_pts = HashSet::new();
    let mut count = 0usize;
    drain_decoder(&rx, |f| {
        assert_eq!(f.format, VideoFormat::RGBA);
        decoded_ids.insert(f.frame_id);
        decoded_pts.insert(f.pts_ns);
        count += 1;
    });
    assert!(count > 0);
    for fid in &submitted_ids {
        assert!(decoded_ids.contains(&Some(*fid)), "frame_id {fid} missing");
    }
    for pts in &submitted_pts {
        assert!(
            decoded_pts.contains(pts),
            "H264: pts {pts} missing from decoded output"
        );
    }
}

#[test]
#[serial]
fn test_e2e_hevc_decode_to_rgba() {
    init();
    if !has_nvenc() || !has_nvdec() {
        eprintln!("skip: no NVENC/NVDEC");
        return;
    }
    let (w, h) = (320, 240);
    let packets = encode_test_frames(Codec::Hevc, w, h, 5);
    let config = DecoderConfig::Hevc(HevcDecoderConfig::new(HevcStreamFormat::ByteStream));
    let (tx, rx) = mpsc::channel();
    let mut decoder = NvDecoder::new(
        0,
        &config,
        make_rgba_pool(w, h),
        identity_transform_config(),
        move |ev| {
            let _ = tx.send(ev);
        },
    )
    .unwrap();
    for (fid, pts, dur, data) in &packets {
        decoder
            .submit_packet(data, *fid, *pts, Some(*pts), Some(*dur))
            .unwrap();
    }
    decoder.send_eos().unwrap();
    let submitted_ids: HashSet<u128> = packets.iter().map(|(fid, _, _, _)| *fid).collect();
    let submitted_pts: HashSet<u64> = packets.iter().map(|(_, pts, _, _)| *pts).collect();
    let mut decoded_ids = HashSet::new();
    let mut decoded_pts = HashSet::new();
    let mut count = 0usize;
    drain_decoder(&rx, |f| {
        assert_eq!(f.format, VideoFormat::RGBA);
        decoded_ids.insert(f.frame_id);
        decoded_pts.insert(f.pts_ns);
        count += 1;
    });
    assert!(count > 0);
    for fid in &submitted_ids {
        assert!(decoded_ids.contains(&Some(*fid)), "frame_id {fid} missing");
    }
    for pts in &submitted_pts {
        assert!(
            decoded_pts.contains(pts),
            "HEVC: pts {pts} missing from decoded output"
        );
    }
}

#[test]
#[serial]
fn test_e2e_jpeg_decode_to_rgba() {
    init();
    if !has_nvjpegdec() {
        eprintln!("skip: no nvjpegdec");
        return;
    }
    let (w, h) = (320, 240);
    let config_enc = EncoderConfig::new(Codec::Jpeg, w, h).format(VideoFormat::I420);
    let mut encoder = NvEncoder::new(&config_enc).expect("JPEG encoder");
    let dur = 33_333_333u64;
    for i in 0..3u128 {
        let shared = encoder.generator().acquire(Some(i)).unwrap();
        let buf = shared.into_buffer().unwrap();
        encoder
            .submit_frame(buf, i, i as u64 * dur, Some(dur))
            .unwrap();
    }
    let enc_frames = encoder.finish(Some(5000)).unwrap();
    let config_dec = DecoderConfig::Jpeg(JpegDecoderConfig::gpu());
    let (tx, rx) = mpsc::channel();
    let mut decoder = NvDecoder::new(
        0,
        &config_dec,
        make_rgba_pool(w, h),
        identity_transform_config(),
        move |ev| {
            let _ = tx.send(ev);
        },
    )
    .unwrap();
    let mut submitted_ids = HashSet::new();
    let mut submitted_pts = HashSet::new();
    for f in &enc_frames {
        let fid = f.frame_id.unwrap_or(0);
        submitted_ids.insert(fid);
        submitted_pts.insert(f.pts_ns);
        decoder
            .submit_packet(&f.data, fid, f.pts_ns, f.dts_ns, f.duration_ns)
            .unwrap();
    }
    decoder.send_eos().unwrap();
    let mut decoded_ids = HashSet::new();
    let mut decoded_pts = HashSet::new();
    let mut count = 0usize;
    drain_decoder(&rx, |f| {
        assert_eq!(f.format, VideoFormat::RGBA);
        decoded_ids.insert(f.frame_id);
        decoded_pts.insert(f.pts_ns);
        count += 1;
    });
    assert!(count > 0);
    for fid in &submitted_ids {
        assert!(
            decoded_ids.contains(&Some(*fid)),
            "JPEG GPU: frame_id {fid} missing"
        );
    }
    for pts in &submitted_pts {
        assert!(
            decoded_pts.contains(pts),
            "JPEG GPU: pts {pts} missing from decoded output"
        );
    }
}

#[test]
#[serial]
fn test_e2e_jpeg_cpu_decode_to_rgba() {
    init();
    let (w, h) = (320, 240);
    let config_enc = EncoderConfig::new(Codec::Jpeg, w, h).format(VideoFormat::I420);
    let mut encoder = NvEncoder::new(&config_enc).expect("JPEG encoder");
    let dur = 33_333_333u64;
    for i in 0..3u128 {
        let shared = encoder.generator().acquire(Some(i)).unwrap();
        let buf = shared.into_buffer().unwrap();
        encoder
            .submit_frame(buf, i, i as u64 * dur, Some(dur))
            .unwrap();
    }
    let enc_frames = encoder.finish(Some(5000)).unwrap();
    let config_dec = DecoderConfig::Jpeg(JpegDecoderConfig::cpu());
    let (tx, rx) = mpsc::channel();
    let mut decoder = NvDecoder::new(
        0,
        &config_dec,
        make_rgba_pool(w, h),
        identity_transform_config(),
        move |ev| {
            let _ = tx.send(ev);
        },
    )
    .unwrap();
    let mut submitted: Vec<(u128, u64)> = Vec::new();
    for f in &enc_frames {
        let fid = f.frame_id.unwrap_or(0);
        submitted.push((fid, f.pts_ns));
        decoder
            .submit_packet(&f.data, fid, f.pts_ns, f.dts_ns, f.duration_ns)
            .unwrap();
    }
    decoder.send_eos().unwrap();
    let mut decoded: Vec<(Option<u128>, u64)> = Vec::new();
    drain_decoder(&rx, |f| {
        assert_eq!(f.format, VideoFormat::RGBA);
        decoded.push((f.frame_id, f.pts_ns));
    });
    assert_eq!(
        decoded.len(),
        submitted.len(),
        "JPEG CPU: expected {} frames, got {}",
        submitted.len(),
        decoded.len()
    );
    for (i, ((exp_fid, exp_pts), (got_fid, got_pts))) in
        submitted.iter().zip(decoded.iter()).enumerate()
    {
        assert_eq!(
            *got_fid,
            Some(*exp_fid),
            "JPEG CPU frame {i}: frame_id mismatch"
        );
        assert_eq!(*got_pts, *exp_pts, "JPEG CPU frame {i}: pts mismatch");
    }
}

#[test]
#[serial]
fn test_e2e_png_decode_to_rgba() {
    init();
    let (w, h) = (64, 48);
    let config_enc = EncoderConfig::new(Codec::Png, w, h).format(VideoFormat::RGBA);
    let mut encoder = NvEncoder::new(&config_enc).expect("PNG encoder");
    let dur = 33_333_333u64;
    for i in 0..2u128 {
        let shared = encoder.generator().acquire(Some(i)).unwrap();
        let buf = shared.into_buffer().unwrap();
        encoder
            .submit_frame(buf, i, i as u64 * dur, Some(dur))
            .unwrap();
    }
    let enc_frames = encoder.finish(Some(5000)).unwrap();
    let (tx, rx) = mpsc::channel();
    let mut decoder = NvDecoder::new(
        0,
        &DecoderConfig::Png(PngDecoderConfig),
        make_rgba_pool(w, h),
        identity_transform_config(),
        move |ev| {
            let _ = tx.send(ev);
        },
    )
    .unwrap();
    let mut submitted: Vec<(u128, u64)> = Vec::new();
    for f in &enc_frames {
        let fid = f.frame_id.unwrap_or(0);
        submitted.push((fid, f.pts_ns));
        decoder
            .submit_packet(&f.data, fid, f.pts_ns, f.dts_ns, f.duration_ns)
            .unwrap();
    }
    decoder.send_eos().unwrap();
    let mut decoded: Vec<(Option<u128>, u64)> = Vec::new();
    drain_decoder(&rx, |f| {
        assert_eq!(f.format, VideoFormat::RGBA);
        decoded.push((f.frame_id, f.pts_ns));
    });
    assert_eq!(
        decoded.len(),
        submitted.len(),
        "PNG: expected {} frames, got {}",
        submitted.len(),
        decoded.len()
    );
    for (i, ((exp_fid, exp_pts), (got_fid, got_pts))) in
        submitted.iter().zip(decoded.iter()).enumerate()
    {
        assert_eq!(
            *got_fid,
            Some(*exp_fid),
            "PNG frame {i}: frame_id mismatch"
        );
        assert_eq!(*got_pts, *exp_pts, "PNG frame {i}: pts mismatch");
    }
}

#[test]
#[serial]
fn test_e2e_raw_rgba_upload_to_rgba() {
    init();
    let (w, h) = (320, 240);
    let pixels = vec![128u8; (w as usize) * (h as usize) * 4];
    let (tx, rx) = mpsc::channel();
    let mut decoder = NvDecoder::new(
        0,
        &DecoderConfig::RawRgba(RawRgbaDecoderConfig::new(w, h)),
        make_rgba_pool(w, h),
        identity_transform_config(),
        move |ev| {
            let _ = tx.send(ev);
        },
    )
    .unwrap();
    decoder
        .submit_packet(&pixels, 42, 1000, Some(0), Some(33_333_333))
        .unwrap();
    decoder.send_eos().unwrap();
    let mut count = 0usize;
    drain_decoder(&rx, |f| {
        assert_eq!(f.format, VideoFormat::RGBA);
        assert_eq!(f.frame_id, Some(42), "raw RGBA: frame_id mismatch");
        assert_eq!(f.pts_ns, 1000, "raw RGBA: pts mismatch");
        count += 1;
    });
    assert_eq!(count, 1);
}

#[test]
#[serial]
fn test_e2e_raw_rgb_upload_to_rgba() {
    init();
    let (w, h) = (320, 240);
    let pixels = vec![77u8; (w as usize) * (h as usize) * 3];
    let (tx, rx) = mpsc::channel();
    let mut decoder = NvDecoder::new(
        0,
        &DecoderConfig::RawRgb(RawRgbDecoderConfig::new(w, h)),
        make_rgba_pool(w, h),
        identity_transform_config(),
        move |ev| {
            let _ = tx.send(ev);
        },
    )
    .unwrap();
    decoder
        .submit_packet(&pixels, 99, 2000, Some(0), Some(33_333_333))
        .unwrap();
    decoder.send_eos().unwrap();
    let mut count = 0usize;
    drain_decoder(&rx, |f| {
        assert_eq!(f.format, VideoFormat::RGBA);
        assert_eq!(f.frame_id, Some(99), "raw RGB: frame_id mismatch");
        assert_eq!(f.pts_ns, 2000, "raw RGB: pts mismatch");
        count += 1;
    });
    assert_eq!(count, 1);
}

#[test]
#[serial]
fn test_dos_garbage_from_start_h264() {
    init();
    if !has_nvdec() {
        eprintln!("skip: no NVDEC");
        return;
    }
    let (tx, rx) = mpsc::channel();
    let mut decoder = NvDecoder::new(
        0,
        &DecoderConfig::H264(H264DecoderConfig::new(H264StreamFormat::ByteStream)),
        make_rgba_pool(320, 240),
        identity_transform_config(),
        move |ev| {
            let _ = tx.send(ev);
        },
    )
    .unwrap();
    let garbage = vec![0xDE, 0xAD, 0xBE, 0xEF];
    decoder
        .submit_packet(&garbage, 1, 1, Some(1), Some(33_333_333))
        .unwrap();

    // Garbage may cause an error/restart, EOS, or simply stall the pipeline.
    // The critical invariant is that no valid frame is produced.
    match rx.recv_timeout(Duration::from_secs(5)) {
        Ok(DecoderEvent::PipelineRestarted { .. }) | Ok(DecoderEvent::Error(_)) => {}
        Ok(DecoderEvent::Frame(_)) => panic!("unexpected frame from garbage"),
        Ok(DecoderEvent::Eos) | Err(_) => {}
    }
}

#[test]
#[serial]
fn test_dos_garbage_mid_stream_h264() {
    init();
    if !has_nvenc() || !has_nvdec() {
        eprintln!("skip");
        return;
    }
    let (w, h) = (320, 240);
    let (tx, rx) = mpsc::channel();
    let mut decoder = NvDecoder::new(
        0,
        &DecoderConfig::H264(H264DecoderConfig::new(H264StreamFormat::ByteStream)),
        make_rgba_pool(w, h),
        identity_transform_config(),
        move |ev| {
            let _ = tx.send(ev);
        },
    )
    .unwrap();
    let valid_packets = encode_test_frames(Codec::H264, w, h, 3);
    for (fid, pts, dur, data) in &valid_packets {
        decoder
            .submit_packet(data, *fid, *pts, Some(*pts), Some(*dur))
            .unwrap();
    }
    let garbage = vec![0u8; 1024];
    decoder
        .submit_packet(&garbage, 999, 999_000_000, Some(999_000_000), None)
        .unwrap();
    let mut got_restart = false;
    loop {
        match rx.recv_timeout(Duration::from_secs(5)) {
            Ok(DecoderEvent::PipelineRestarted { .. }) | Ok(DecoderEvent::Error(_)) => {
                got_restart = true;
                break;
            }
            Ok(DecoderEvent::Frame(_)) => {}
            Ok(DecoderEvent::Eos) => break,
            Err(_) => break,
        }
    }
    if got_restart && has_nvenc() {
        let recovery = encode_test_frames(Codec::H264, w, h, 2);
        for (fid, pts, dur, data) in &recovery {
            decoder
                .submit_packet(data, *fid, *pts, Some(*pts), Some(*dur))
                .unwrap();
        }
        decoder.send_eos().unwrap();
        let mut recovered = 0usize;
        loop {
            match rx.recv_timeout(Duration::from_secs(10)) {
                Ok(DecoderEvent::Frame(_)) => recovered += 1,
                Ok(DecoderEvent::Eos) => break,
                Ok(_) => {}
                Err(_) => break,
            }
        }
        assert!(recovered > 0, "decoder did not recover");
    }
}

#[test]
#[serial]
fn test_dos_garbage_jpeg_png() {
    init();
    if has_nvjpegdec() {
        let (tx, rx) = mpsc::channel();
        let mut jpeg = NvDecoder::new(
            0,
            &DecoderConfig::Jpeg(JpegDecoderConfig::gpu()),
            make_rgba_pool(64, 48),
            identity_transform_config(),
            move |ev| {
                let _ = tx.send(ev);
            },
        )
        .unwrap();
        jpeg.submit_packet(&[0; 128], 1, 1, Some(1), None).unwrap();
        match rx.recv_timeout(Duration::from_secs(3)) {
            Ok(DecoderEvent::Frame(_)) => panic!("unexpected frame from garbage JPEG"),
            Ok(DecoderEvent::Eos)
            | Ok(DecoderEvent::Error(_))
            | Ok(DecoderEvent::PipelineRestarted { .. }) => {}
            Err(_) => {} // pipeline stalled — acceptable for garbage input
        }
    }
    {
        let (tx, _rx) = mpsc::channel();
        let mut png = NvDecoder::new(
            0,
            &DecoderConfig::Png(PngDecoderConfig),
            make_rgba_pool(64, 48),
            identity_transform_config(),
            move |ev| {
                let _ = tx.send(ev);
            },
        )
        .unwrap();
        // PNG uses the `image` crate; garbage data causes an immediate
        // error from submit_packet rather than a deferred pipeline event.
        assert!(png
            .submit_packet(&[0; 128], 1, 1, Some(1), None)
            .is_err());
    }
}

// ── New hardening tests ─────────────────────────────────────────────

#[test]
#[serial]
fn test_submit_after_eos_returns_already_finalized() {
    init();
    let (w, h) = (320, 240);
    let (tx, _rx) = mpsc::channel();
    let mut decoder = NvDecoder::new(
        0,
        &DecoderConfig::RawRgba(RawRgbaDecoderConfig::new(w, h)),
        make_rgba_pool(w, h),
        identity_transform_config(),
        move |ev| {
            let _ = tx.send(ev);
        },
    )
    .unwrap();

    let pixels = vec![128u8; (w as usize) * (h as usize) * 4];
    decoder.submit_packet(&pixels, 1, 0, None, None).unwrap();
    decoder.send_eos().unwrap();

    let result = decoder.submit_packet(&pixels, 2, 1, None, None);
    assert!(
        matches!(result, Err(DecoderError::AlreadyFinalized)),
        "expected AlreadyFinalized, got {result:?}"
    );
}

#[test]
#[serial]
fn test_send_eos_idempotent() {
    init();
    let (w, h) = (64, 48);
    let (tx, _rx) = mpsc::channel();
    let mut decoder = NvDecoder::new(
        0,
        &DecoderConfig::RawRgba(RawRgbaDecoderConfig::new(w, h)),
        make_rgba_pool(w, h),
        identity_transform_config(),
        move |ev| {
            let _ = tx.send(ev);
        },
    )
    .unwrap();

    decoder.send_eos().unwrap();
    decoder.send_eos().unwrap();
}

#[test]
#[serial]
fn test_submit_rejects_non_monotonic_pts() {
    init();
    let (w, h) = (64, 48);
    let (tx, _rx) = mpsc::channel();
    let mut decoder = NvDecoder::new(
        0,
        &DecoderConfig::RawRgba(RawRgbaDecoderConfig::new(w, h)),
        make_rgba_pool(w, h),
        identity_transform_config(),
        move |ev| {
            let _ = tx.send(ev);
        },
    )
    .unwrap();

    let pixels = vec![0u8; (w as usize) * (h as usize) * 4];
    decoder.submit_packet(&pixels, 1, 100, None, None).unwrap();
    let result = decoder.submit_packet(&pixels, 2, 50, None, None);
    assert!(
        matches!(result, Err(DecoderError::PtsReordered { .. })),
        "expected PtsReordered, got {result:?}"
    );
    let result_eq = decoder.submit_packet(&pixels, 3, 100, None, None);
    assert!(
        matches!(result_eq, Err(DecoderError::PtsReordered { .. })),
        "expected PtsReordered for equal PTS, got {result_eq:?}"
    );
}

#[test]
#[serial]
fn test_raw_rgba_rejects_wrong_payload_size() {
    init();
    let (w, h) = (64, 48);
    let (tx, _rx) = mpsc::channel();
    let mut decoder = NvDecoder::new(
        0,
        &DecoderConfig::RawRgba(RawRgbaDecoderConfig::new(w, h)),
        make_rgba_pool(w, h),
        identity_transform_config(),
        move |ev| {
            let _ = tx.send(ev);
        },
    )
    .unwrap();

    let too_short = vec![0u8; (w as usize) * (h as usize) * 4 - 1];
    let result = decoder.submit_packet(&too_short, 1, 0, None, None);
    assert!(
        matches!(result, Err(DecoderError::BufferError(_))),
        "expected BufferError for wrong payload size, got {result:?}"
    );
}

#[test]
#[serial]
fn test_raw_rgb_codec_identity() {
    init();
    let (w, h) = (64, 48);
    let (tx, rx) = mpsc::channel();
    let mut decoder = NvDecoder::new(
        0,
        &DecoderConfig::RawRgb(RawRgbDecoderConfig::new(w, h)),
        make_rgba_pool(w, h),
        identity_transform_config(),
        move |ev| {
            let _ = tx.send(ev);
        },
    )
    .unwrap();

    let pixels = vec![128u8; (w as usize) * (h as usize) * 3];
    decoder.submit_packet(&pixels, 1, 0, None, None).unwrap();
    decoder.send_eos().unwrap();
    let mut count = 0usize;
    drain_decoder(&rx, |f| {
        assert_eq!(
            f.codec,
            Codec::RawRgb,
            "expected RawRgb codec, got {:?}",
            f.codec
        );
        count += 1;
    });
    assert_eq!(count, 1);
}

#[test]
#[serial]
fn test_raw_zero_dimensions_rejected() {
    init();
    let (tx, _rx) = mpsc::channel();
    let result = NvDecoder::new(
        0,
        &DecoderConfig::RawRgba(RawRgbaDecoderConfig::new(0, 0)),
        make_rgba_pool(64, 48),
        identity_transform_config(),
        move |ev| {
            let _ = tx.send(ev);
        },
    );
    assert!(
        matches!(result, Err(DecoderError::InvalidProperty { .. })),
        "expected InvalidProperty for zero dimensions"
    );
}

#[test]
#[serial]
fn test_resolution_mismatch_raw_rgba() {
    init();
    let (raw_w, raw_h) = (320, 240);
    let (pool_w, pool_h) = (640, 480);
    let pixels = vec![128u8; (raw_w as usize) * (raw_h as usize) * 4];
    let (tx, rx) = mpsc::channel();
    let mut decoder = NvDecoder::new(
        0,
        &DecoderConfig::RawRgba(RawRgbaDecoderConfig::new(raw_w, raw_h)),
        make_rgba_pool(pool_w, pool_h),
        identity_transform_config(),
        move |ev| {
            let _ = tx.send(ev);
        },
    )
    .unwrap();
    decoder
        .submit_packet(&pixels, 42, 5000, Some(0), Some(33_333_333))
        .unwrap();
    decoder.send_eos().unwrap();
    let mut count = 0usize;
    drain_decoder(&rx, |f| {
        assert_eq!(f.format, VideoFormat::RGBA);
        assert_eq!(
            f.frame_id,
            Some(42),
            "raw RGBA mismatch: frame_id mismatch"
        );
        assert_eq!(f.pts_ns, 5000, "raw RGBA mismatch: pts mismatch");
        count += 1;
    });
    assert_eq!(
        count, 1,
        "expected 1 frame from raw RGBA with pool mismatch"
    );
}

#[test]
#[serial]
fn test_resolution_mismatch_png() {
    init();
    let (enc_w, enc_h) = (64, 48);
    let (pool_w, pool_h) = (128, 96);
    let config_enc = EncoderConfig::new(Codec::Png, enc_w, enc_h).format(VideoFormat::RGBA);
    let mut encoder = NvEncoder::new(&config_enc).expect("PNG encoder");
    let dur = 33_333_333u64;
    for i in 0..2u128 {
        let shared = encoder.generator().acquire(Some(i)).unwrap();
        let buf = shared.into_buffer().unwrap();
        encoder
            .submit_frame(buf, i, i as u64 * dur, Some(dur))
            .unwrap();
    }
    let enc_frames = encoder.finish(Some(5000)).unwrap();
    let (tx, rx) = mpsc::channel();
    let mut decoder = NvDecoder::new(
        0,
        &DecoderConfig::Png(PngDecoderConfig),
        make_rgba_pool(pool_w, pool_h),
        identity_transform_config(),
        move |ev| {
            let _ = tx.send(ev);
        },
    )
    .unwrap();
    let mut submitted: Vec<(u128, u64)> = Vec::new();
    for f in &enc_frames {
        let fid = f.frame_id.unwrap_or(0);
        submitted.push((fid, f.pts_ns));
        decoder
            .submit_packet(&f.data, fid, f.pts_ns, f.dts_ns, f.duration_ns)
            .unwrap();
    }
    decoder.send_eos().unwrap();
    let mut decoded: Vec<(Option<u128>, u64)> = Vec::new();
    drain_decoder(&rx, |f| {
        assert_eq!(f.format, VideoFormat::RGBA);
        decoded.push((f.frame_id, f.pts_ns));
    });
    assert_eq!(
        decoded.len(),
        submitted.len(),
        "PNG mismatch: expected {} frames, got {}",
        submitted.len(),
        decoded.len()
    );
    for (i, ((exp_fid, exp_pts), (got_fid, got_pts))) in
        submitted.iter().zip(decoded.iter()).enumerate()
    {
        assert_eq!(
            *got_fid,
            Some(*exp_fid),
            "PNG mismatch frame {i}: frame_id mismatch"
        );
        assert_eq!(
            *got_pts, *exp_pts,
            "PNG mismatch frame {i}: pts mismatch"
        );
    }
}
