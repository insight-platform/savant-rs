mod common;

use common::*;
use deepstream_decoders::prelude::*;
use deepstream_decoders::NvDecoderExt;
use deepstream_encoders::{EncoderConfig, NvEncoder};
use serial_test::serial;
use std::collections::HashSet;
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
    if !has_nvenc() {
        eprintln!("skip: no NVENC");
        return;
    }
    let (w, h) = (320, 240);
    let packets = encode_test_frames(Codec::H264, w, h, 5);
    let config = DecoderConfig::H264(H264DecoderConfig::new(H264StreamFormat::ByteStream));
    let decoder = NvDecoder::new(
        test_decoder_config(0, config),
        make_rgba_pool(w, h),
        identity_transform_config(),
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
    drain_decoder(&decoder, |f| {
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
    if !has_nvenc() {
        eprintln!("skip: no NVENC");
        return;
    }
    let (w, h) = (320, 240);
    let packets = encode_test_frames(Codec::Hevc, w, h, 5);
    let config = DecoderConfig::Hevc(HevcDecoderConfig::new(HevcStreamFormat::ByteStream));
    let decoder = NvDecoder::new(
        test_decoder_config(0, config),
        make_rgba_pool(w, h),
        identity_transform_config(),
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
    drain_decoder(&decoder, |f| {
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
    let decoder = NvDecoder::new(
        test_decoder_config(0, config_dec),
        make_rgba_pool(w, h),
        identity_transform_config(),
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
    drain_decoder(&decoder, |f| {
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
    let decoder = NvDecoder::new(
        test_decoder_config(0, config_dec),
        make_rgba_pool(w, h),
        identity_transform_config(),
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
    drain_decoder(&decoder, |f| {
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
    let decoder = NvDecoder::new(
        test_decoder_config(0, DecoderConfig::Png(PngDecoderConfig)),
        make_rgba_pool(w, h),
        identity_transform_config(),
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
    drain_decoder(&decoder, |f| {
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
        assert_eq!(*got_fid, Some(*exp_fid), "PNG frame {i}: frame_id mismatch");
        assert_eq!(*got_pts, *exp_pts, "PNG frame {i}: pts mismatch");
    }
}

#[test]
#[serial]
fn test_e2e_raw_rgba_upload_to_rgba() {
    init();
    let (w, h) = (320, 240);
    let pixels = vec![128u8; (w as usize) * (h as usize) * 4];
    let decoder = NvDecoder::new(
        test_decoder_config(0, DecoderConfig::RawRgba(RawRgbaDecoderConfig::new(w, h))),
        make_rgba_pool(w, h),
        identity_transform_config(),
    )
    .unwrap();
    decoder
        .submit_packet(&pixels, 42, 1000, Some(0), Some(33_333_333))
        .unwrap();
    decoder.send_eos().unwrap();
    let mut count = 0usize;
    drain_decoder(&decoder, |f| {
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
    let decoder = NvDecoder::new(
        test_decoder_config(0, DecoderConfig::RawRgb(RawRgbDecoderConfig::new(w, h))),
        make_rgba_pool(w, h),
        identity_transform_config(),
    )
    .unwrap();
    decoder
        .submit_packet(&pixels, 99, 2000, Some(0), Some(33_333_333))
        .unwrap();
    decoder.send_eos().unwrap();
    let mut count = 0usize;
    drain_decoder(&decoder, |f| {
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
    let decoder = NvDecoder::new(
        test_decoder_config(
            0,
            DecoderConfig::H264(H264DecoderConfig::new(H264StreamFormat::ByteStream)),
        ),
        make_rgba_pool(320, 240),
        identity_transform_config(),
    )
    .unwrap();
    let garbage = vec![0xDE, 0xAD, 0xBE, 0xEF];
    decoder
        .submit_packet(&garbage, 1, 1, Some(1), Some(33_333_333))
        .unwrap();

    match decoder.recv_timeout(Duration::from_secs(2)) {
        Ok(Some(NvDecoderOutput::Error(_))) => {}
        Ok(Some(NvDecoderOutput::Frame(_))) => panic!("unexpected frame from garbage"),
        Ok(Some(NvDecoderOutput::Eos))
        | Ok(Some(NvDecoderOutput::Event(_)))
        | Ok(Some(NvDecoderOutput::SourceEos { .. }))
        | Ok(None)
        | Err(_) => {}
    }
}

#[test]
#[serial]
fn test_dos_garbage_mid_stream_h264() {
    init();
    if !has_nvenc() {
        eprintln!("skip: no NVENC");
        return;
    }
    let (w, h) = (320, 240);
    let decoder = NvDecoder::new(
        test_decoder_config(
            0,
            DecoderConfig::H264(H264DecoderConfig::new(H264StreamFormat::ByteStream)),
        ),
        make_rgba_pool(w, h),
        identity_transform_config(),
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

    // Drain: expect some valid frames, then an error or EOS/timeout.
    // No auto-restart exists, so the decoder will not recover.
    loop {
        match decoder.recv_timeout(Duration::from_secs(2)) {
            Ok(Some(NvDecoderOutput::Frame(_))) => {}
            Ok(Some(NvDecoderOutput::Error(_))) | Ok(Some(NvDecoderOutput::Eos)) => break,
            Ok(Some(NvDecoderOutput::Event(_) | NvDecoderOutput::SourceEos { .. })) => {}
            Ok(None) | Err(_) => break,
        }
    }
}

#[test]
#[serial]
fn test_dos_garbage_jpeg_png() {
    init();
    {
        let jpeg = NvDecoder::new(
            test_decoder_config(0, DecoderConfig::Jpeg(JpegDecoderConfig::gpu())),
            make_rgba_pool(64, 48),
            identity_transform_config(),
        )
        .unwrap();
        jpeg.submit_packet(&[0; 128], 1, 1, Some(1), None).unwrap();
        match jpeg.recv_timeout(Duration::from_secs(1)) {
            Ok(Some(NvDecoderOutput::Frame(_))) => panic!("unexpected frame from garbage JPEG"),
            Ok(Some(NvDecoderOutput::Eos))
            | Ok(Some(NvDecoderOutput::Error(_)))
            | Ok(Some(NvDecoderOutput::Event(_)))
            | Ok(Some(NvDecoderOutput::SourceEos { .. })) => {}
            Ok(None) | Err(_) => {}
        }
    }
    {
        let png = NvDecoder::new(
            test_decoder_config(0, DecoderConfig::Png(PngDecoderConfig)),
            make_rgba_pool(64, 48),
            identity_transform_config(),
        )
        .unwrap();
        assert!(png.submit_packet(&[0; 128], 1, 1, Some(1), None).is_err());
    }
}

// ── New hardening tests ─────────────────────────────────────────────

#[test]
#[serial]
fn test_submit_after_eos_returns_already_finalized() {
    init();
    let (w, h) = (320, 240);
    let decoder = NvDecoder::new(
        test_decoder_config(0, DecoderConfig::RawRgba(RawRgbaDecoderConfig::new(w, h))),
        make_rgba_pool(w, h),
        identity_transform_config(),
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
    let decoder = NvDecoder::new(
        test_decoder_config(0, DecoderConfig::RawRgba(RawRgbaDecoderConfig::new(w, h))),
        make_rgba_pool(w, h),
        identity_transform_config(),
    )
    .unwrap();

    decoder.send_eos().unwrap();
    decoder.send_eos().unwrap();
}

/// RawUpload backend does not go through GstPipeline, so PTS ordering is not
/// enforced at submit time. Callers rely on upstream ordering (e.g. FlexibleDecoder).
#[test]
#[serial]
fn test_raw_upload_accepts_non_monotonic_pts() {
    init();
    let (w, h) = (64, 48);
    let decoder = NvDecoder::new(
        test_decoder_config(0, DecoderConfig::RawRgba(RawRgbaDecoderConfig::new(w, h))),
        make_rgba_pool(w, h),
        identity_transform_config(),
    )
    .unwrap();

    let pixels = vec![0u8; (w as usize) * (h as usize) * 4];
    decoder.submit_packet(&pixels, 1, 100, None, None).unwrap();
    decoder
        .submit_packet(&pixels, 2, 50, None, None)
        .expect("RawUpload has no PTS ordering constraint");
    decoder
        .submit_packet(&pixels, 3, 100, None, None)
        .expect("RawUpload has no PTS ordering constraint");
}

/// Pipeline backend enforces PTS ordering via GstPipeline's StrictDecodeOrder
/// policy. A violation produces an error on the output channel (not from submit).
#[test]
#[serial]
fn test_pipeline_rejects_non_monotonic_pts_async() {
    init();
    let (w, h) = (320, 240);
    let frames = encode_test_frames(Codec::H264, w, h, 3);
    assert!(!frames.is_empty(), "need at least one encoded frame");

    let config = DecoderConfig::H264(H264DecoderConfig::new(H264StreamFormat::ByteStream));
    let decoder = NvDecoder::new(
        test_decoder_config(0, config),
        make_rgba_pool(w, h),
        identity_transform_config(),
    )
    .unwrap();

    let (_, pts0, _, ref data0) = frames[0];
    decoder.submit_packet(data0, 1, pts0, None, None).unwrap();

    // Submit same frame again with a lower PTS to trigger the policy violation.
    let submit_result = decoder.submit_packet(data0, 2, pts0.saturating_sub(1), None, None);
    // submit_packet itself succeeds (async validation).
    assert!(
        submit_result.is_ok(),
        "submit should succeed; validation is async"
    );

    // The violation surfaces as an error on the output channel.
    let mut got_ts_error = false;
    for _ in 0..10 {
        match decoder.recv_timeout(Duration::from_secs(2)) {
            Ok(Some(NvDecoderOutput::Error(DecoderError::FrameworkError(e)))) => {
                let msg = e.to_string();
                if msg.contains("Timestamp order violation") {
                    got_ts_error = true;
                    break;
                }
            }
            Ok(Some(NvDecoderOutput::Frame(_))) => {}
            Ok(Some(NvDecoderOutput::Event(_))) => {}
            _ => break,
        }
    }
    assert!(
        got_ts_error,
        "expected TimestampOrderViolation on the output channel"
    );
}

#[test]
#[serial]
fn test_raw_rgba_rejects_wrong_payload_size() {
    init();
    let (w, h) = (64, 48);
    let decoder = NvDecoder::new(
        test_decoder_config(0, DecoderConfig::RawRgba(RawRgbaDecoderConfig::new(w, h))),
        make_rgba_pool(w, h),
        identity_transform_config(),
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
fn test_raw_rgb_rejects_rgba_payload_size() {
    init();
    let (w, h) = (64, 48);
    let decoder = NvDecoder::new(
        test_decoder_config(0, DecoderConfig::RawRgb(RawRgbDecoderConfig::new(w, h))),
        make_rgba_pool(w, h),
        identity_transform_config(),
    )
    .unwrap();

    let rgba_sized = vec![0u8; (w as usize) * (h as usize) * 4];
    let result = decoder.submit_packet(&rgba_sized, 1, 0, None, None);
    assert!(
        matches!(result, Err(DecoderError::BufferError(_))),
        "expected BufferError for RGBA-sized payload on RawRgb config, got {result:?}"
    );
}

#[test]
#[serial]
fn test_raw_rgb_codec_identity() {
    init();
    let (w, h) = (64, 48);
    let decoder = NvDecoder::new(
        test_decoder_config(0, DecoderConfig::RawRgb(RawRgbDecoderConfig::new(w, h))),
        make_rgba_pool(w, h),
        identity_transform_config(),
    )
    .unwrap();

    let pixels = vec![128u8; (w as usize) * (h as usize) * 3];
    decoder.submit_packet(&pixels, 1, 0, None, None).unwrap();
    decoder.send_eos().unwrap();
    let mut count = 0usize;
    drain_decoder(&decoder, |f| {
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
    let result = NvDecoder::new(
        test_decoder_config(0, DecoderConfig::RawRgba(RawRgbaDecoderConfig::new(0, 0))),
        make_rgba_pool(64, 48),
        identity_transform_config(),
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
    let decoder = NvDecoder::new(
        test_decoder_config(
            0,
            DecoderConfig::RawRgba(RawRgbaDecoderConfig::new(raw_w, raw_h)),
        ),
        make_rgba_pool(pool_w, pool_h),
        identity_transform_config(),
    )
    .unwrap();
    decoder
        .submit_packet(&pixels, 42, 5000, Some(0), Some(33_333_333))
        .unwrap();
    decoder.send_eos().unwrap();
    let mut count = 0usize;
    drain_decoder(&decoder, |f| {
        assert_eq!(f.format, VideoFormat::RGBA);
        assert_eq!(f.frame_id, Some(42), "raw RGBA mismatch: frame_id mismatch");
        assert_eq!(f.pts_ns, 5000, "raw RGBA mismatch: pts mismatch");
        count += 1;
    });
    assert_eq!(
        count, 1,
        "expected 1 frame from raw RGBA with pool mismatch"
    );
}

/// Verify that `graceful_shutdown` returns RGBA buffers with valid,
/// accessible NVMM GPU memory.
///
/// Submits 3 JPEG frames, then calls `graceful_shutdown` (no prior `recv`).
/// For each returned frame the test creates a `SurfaceView` to prove the
/// underlying NVMM memory is still mapped and readable.
#[test]
#[serial]
fn test_graceful_shutdown_returns_valid_rgba() {
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

    let config_dec = DecoderConfig::Jpeg(JpegDecoderConfig::gpu());
    let decoder = NvDecoder::new(
        test_decoder_config(0, config_dec),
        make_rgba_pool(w, h),
        identity_transform_config(),
    )
    .unwrap();

    for f in &enc_frames {
        let fid = f.frame_id.unwrap_or(0);
        decoder
            .submit_packet(&f.data, fid, f.pts_ns, f.dts_ns, f.duration_ns)
            .unwrap();
    }
    // Do NOT call recv — all decoded frames stay in the output channel.
    let mut frame_count = 0usize;
    decoder
        .graceful_shutdown(Some(Duration::from_secs(10)), |item| {
            if let NvDecoderOutput::Frame(f) = &item {
                assert_eq!(f.format, VideoFormat::RGBA, "expected RGBA output");
                let view = SurfaceView::from_buffer(&f.buffer, 0)
                    .expect("SurfaceView::from_buffer must succeed on a valid RGBA buffer");
                assert_eq!(view.width(), w, "surface width mismatch");
                assert_eq!(view.height(), h, "surface height mismatch");
                frame_count += 1;
            }
        })
        .expect("graceful_shutdown failed");

    assert_eq!(
        frame_count,
        enc_frames.len(),
        "graceful_shutdown must return all submitted frames"
    );
}

/// Same as [`test_graceful_shutdown_returns_valid_rgba`] but with H264 (multi-frame,
/// P-frame buffering).  Isolates whether `graceful_shutdown` can flush the
/// `nvv4l2decoder` pipeline correctly.
#[test]
#[serial]
fn test_graceful_shutdown_h264_valid_rgba() {
    init();
    if !has_nvenc() {
        eprintln!("SKIP: no NVENC on this platform");
        return;
    }
    let (w, h) = (320, 240);
    let num_frames = 5;
    let packets = encode_test_frames(Codec::H264, w, h, num_frames);

    let config = DecoderConfig::H264(H264DecoderConfig::new(H264StreamFormat::ByteStream));
    let decoder = NvDecoder::new(
        test_decoder_config(0, config),
        make_rgba_pool(w, h),
        identity_transform_config(),
    )
    .unwrap();

    for (fid, pts, _dur, data) in &packets {
        decoder
            .submit_packet(data, *fid, *pts, Some(*pts), None)
            .unwrap();
    }

    let mut frame_count = 0usize;
    decoder
        .graceful_shutdown(Some(Duration::from_secs(10)), |item| {
            if let NvDecoderOutput::Frame(f) = &item {
                assert_eq!(f.format, VideoFormat::RGBA, "expected RGBA output");
                let view = SurfaceView::from_buffer(&f.buffer, 0)
                    .expect("SurfaceView::from_buffer must succeed on a valid RGBA buffer");
                assert_eq!(view.width(), w, "surface width mismatch");
                assert_eq!(view.height(), h, "surface height mismatch");
                frame_count += 1;
            }
        })
        .expect("graceful_shutdown failed for H264");

    assert!(
        frame_count >= num_frames,
        "graceful_shutdown must return all H264 frames: expected {num_frames}, got {frame_count}"
    );
}

/// Simulates the FlexibleDecoder worker pattern: submit H264 packets,
/// poll with `try_recv` between each (like `pull_ready_outputs`), then call
/// `graceful_shutdown`.  Some frames may have been consumed by `try_recv`
/// before `graceful_shutdown` runs.
#[test]
#[serial]
fn test_graceful_shutdown_h264_after_partial_drain() {
    init();
    if !has_nvenc() {
        eprintln!("SKIP: no NVENC on this platform");
        return;
    }
    let (w, h) = (320, 240);
    let num_frames = 10;
    let packets = encode_test_frames(Codec::H264, w, h, num_frames);

    let config = DecoderConfig::H264(H264DecoderConfig::new(H264StreamFormat::ByteStream));
    let decoder = NvDecoder::new(
        test_decoder_config(0, config),
        make_rgba_pool(w, h),
        identity_transform_config(),
    )
    .unwrap();

    let mut pre_drained = 0usize;
    for (fid, pts, _dur, data) in &packets {
        decoder
            .submit_packet(data, *fid, *pts, Some(*pts), None)
            .unwrap();
        // poll like pull_ready_outputs
        loop {
            match decoder.try_recv() {
                Ok(Some(NvDecoderOutput::Frame(_))) => pre_drained += 1,
                Ok(Some(_)) => {}
                Ok(None) | Err(_) => break,
            }
        }
    }

    eprintln!("pre-drained {pre_drained} frames before graceful_shutdown");

    let mut shutdown_frames = 0usize;
    decoder
        .graceful_shutdown(Some(Duration::from_secs(10)), |item| {
            if matches!(item, NvDecoderOutput::Frame(_)) {
                shutdown_frames += 1;
            }
        })
        .expect("graceful_shutdown failed after partial drain");

    eprintln!("graceful_shutdown returned {shutdown_frames} frames");
    let total = pre_drained + shutdown_frames;
    assert!(
        total >= num_frames,
        "total frames (pre-drained {pre_drained} + shutdown {shutdown_frames} = {total}) \
         must be >= {num_frames}"
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
    let decoder = NvDecoder::new(
        test_decoder_config(0, DecoderConfig::Png(PngDecoderConfig)),
        make_rgba_pool(pool_w, pool_h),
        identity_transform_config(),
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
    drain_decoder(&decoder, |f| {
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
        assert_eq!(*got_pts, *exp_pts, "PNG mismatch frame {i}: pts mismatch");
    }
}
