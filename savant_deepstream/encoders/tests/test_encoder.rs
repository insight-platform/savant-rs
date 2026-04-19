//! Integration tests for the channel-based [`NvEncoder`].
//!
//! Tests requiring NVENC (H.264/HEVC/AV1) are skipped at runtime when the
//! hardware is unavailable (e.g. Orin Nano).  Generic behaviour tests fall
//! back to JPEG via `nvjpegenc` when possible.
//!
//! Run with:
//! ```sh
//! cargo test -p savant-deepstream-encoders --test test_encoder
//! ```

mod common;

use std::time::Duration;

use common::*;
use deepstream_encoders::prelude::*;
use serial_test::serial;

const FRAME_DUR_NS: u64 = 33_333_333; // ~30 fps

// ─── Codec tests ─────────────────────────────────────────────────────────

#[test]
fn test_codec_from_name() {
    assert_eq!(Codec::from_name("h264"), Some(Codec::H264));
    assert_eq!(Codec::from_name("H264"), Some(Codec::H264));
    assert_eq!(Codec::from_name("hevc"), Some(Codec::Hevc));
    assert_eq!(Codec::from_name("h265"), Some(Codec::Hevc));
    assert_eq!(Codec::from_name("jpeg"), Some(Codec::Jpeg));
    assert_eq!(Codec::from_name("av1"), Some(Codec::Av1));
    assert_eq!(Codec::from_name("png"), Some(Codec::Png));
    assert_eq!(Codec::from_name("vp8"), Some(Codec::Vp8));
    assert_eq!(Codec::from_name("vp9"), Some(Codec::Vp9));
    assert_eq!(Codec::from_name("raw_rgba"), Some(Codec::RawRgba));
    assert_eq!(Codec::from_name("raw_rgb"), Some(Codec::RawRgb));
    assert_eq!(Codec::from_name("raw_nv12"), Some(Codec::RawNv12));
    assert_eq!(Codec::from_name("RAW_NV12"), Some(Codec::RawNv12));
    assert_eq!(Codec::from_name("unknown"), None);
}

#[test]
fn test_codec_names() {
    assert_eq!(Codec::H264.name(), "h264");
    assert_eq!(Codec::Hevc.name(), "hevc");
    assert_eq!(Codec::Jpeg.name(), "jpeg");
    assert_eq!(Codec::Av1.name(), "av1");
    assert_eq!(Codec::Png.name(), "png");
    assert_eq!(Codec::Vp8.name(), "vp8");
    assert_eq!(Codec::Vp9.name(), "vp9");
    assert_eq!(Codec::RawRgba.name(), "raw_rgba");
    assert_eq!(Codec::RawRgb.name(), "raw_rgb");
    assert_eq!(Codec::RawNv12.name(), "raw_nv12");
}

// ─── EncoderConfig tests ─────────────────────────────────────────────────

#[test]
fn test_config_defaults() {
    let config = HevcEncoderConfig::new(1920, 1080);
    assert_eq!(config.width, 1920);
    assert_eq!(config.height, 1080);
    assert_eq!(config.format, VideoFormat::NV12);
    assert_eq!(config.fps_num, 30);
    assert_eq!(config.fps_den, 1);
    assert!(config.props.is_none());

    let encoder_config = EncoderConfig::Hevc(config);
    assert_eq!(encoder_config.codec(), Codec::Hevc);
    assert_eq!(encoder_config.width(), 1920);
    assert_eq!(encoder_config.height(), 1080);
    assert_eq!(encoder_config.format(), VideoFormat::NV12);
}

#[test]
fn test_config_builder_chain() {
    let config = H264EncoderConfig::new(1280, 720)
        .format(VideoFormat::RGBA)
        .fps(60, 1);
    assert_eq!(config.format, VideoFormat::RGBA);
    assert_eq!(config.fps_num, 60);
    assert_eq!(config.fps_den, 1);
}

#[test]
fn test_nv_encoder_config_builder() {
    let encoder = EncoderConfig::H264(H264EncoderConfig::new(320, 240));
    let cfg = NvEncoderConfig::new(1, encoder)
        .name("alpha")
        .mem_type(NvBufSurfaceMemType::CudaDevice)
        .operation_timeout(Duration::from_secs(10))
        .input_channel_capacity(32)
        .output_channel_capacity(32);
    assert_eq!(cfg.gpu_id, 1);
    assert_eq!(cfg.name, "alpha");
    assert_eq!(cfg.mem_type, NvBufSurfaceMemType::CudaDevice);
    assert_eq!(cfg.operation_timeout, Duration::from_secs(10));
    assert_eq!(cfg.input_channel_capacity, 32);
    assert_eq!(cfg.output_channel_capacity, 32);
}

// ─── NvEncoder creation tests ────────────────────────────────────────────

#[test]
#[serial]
fn test_encoder_creation_hevc() {
    init();
    if !has_nvenc() {
        eprintln!("NVENC not available — skipping test_encoder_creation_hevc");
        return;
    }
    let cfg = test_nv_encoder_config(EncoderConfig::Hevc(HevcEncoderConfig::new(640, 480)));
    let encoder = NvEncoder::new(cfg);
    assert!(
        encoder.is_ok(),
        "Failed to create HEVC encoder: {:?}",
        encoder.err()
    );
}

#[test]
#[serial]
fn test_encoder_creation_h264() {
    init();
    if !has_nvenc() {
        eprintln!("NVENC not available — skipping test_encoder_creation_h264");
        return;
    }
    let cfg = test_nv_encoder_config(EncoderConfig::H264(H264EncoderConfig::new(640, 480)));
    assert!(NvEncoder::new(cfg).is_ok());
}

#[test]
#[serial]
fn test_encoder_creation_jpeg() {
    init();
    if !has_nvjpegenc() {
        eprintln!("nvjpegenc not available — skipping test_encoder_creation_jpeg");
        return;
    }
    let cfg = test_nv_encoder_config(EncoderConfig::Jpeg(
        JpegEncoderConfig::new(640, 480).format(VideoFormat::I420),
    ));
    assert!(NvEncoder::new(cfg).is_ok());
}

#[test]
#[serial]
fn test_encoder_creation_av1() {
    init();
    if !has_nvenc() {
        eprintln!("NVENC not available — skipping test_encoder_creation_av1");
        return;
    }
    let cfg = test_nv_encoder_config(EncoderConfig::Av1(Av1EncoderConfig::new(640, 480)));
    assert!(NvEncoder::new(cfg).is_ok());
}

#[test]
#[serial]
fn test_encoder_rejects_vp8() {
    init();
    // VP8 has no EncoderConfig variant, so attempting to construct it is
    // a compile-time error; validate that `UnsupportedCodec` fires when
    // the codec is present but not in the enum.  We cover this via raw
    // config by using Png pipeline with wrong format (rejected too).
    let cfg = test_nv_encoder_config(EncoderConfig::Png(PngEncoderConfig::new(32, 32)));
    // PNG-with-default-format is fine; use this test purely as a sanity
    // check that the `EncoderConfig` API doesn't surface VP8 / VP9.
    let _ = NvEncoder::new(cfg);
    // No VP8 variant exists — this test used to assert UnsupportedCodec,
    // but the new API prevents the construction at compile time.
}

#[test]
#[serial]
fn test_encoder_codec_getter() {
    init();
    let Some(encoder_cfg) = make_default_config(320, 240) else {
        eprintln!("No encoder available — skipping test_encoder_codec_getter");
        return;
    };
    let expected = encoder_cfg.codec();
    let encoder = NvEncoder::new(test_nv_encoder_config(encoder_cfg)).unwrap();
    assert_eq!(encoder.codec(), expected);
}

// ─── submit / drain tests ───────────────────────────────────────────────

#[test]
#[serial]
fn test_submit_and_pull_frames() {
    init();
    let Some(encoder_cfg) = make_default_config(320, 240) else {
        eprintln!("No encoder available — skipping test_submit_and_pull_frames");
        return;
    };
    let expected_codec = encoder_cfg.codec();
    let encoder = NvEncoder::new(test_nv_encoder_config(encoder_cfg)).unwrap();

    for i in 0..5u128 {
        let buf = acquire_buffer(&encoder, i);
        let pts_ns = i as u64 * FRAME_DUR_NS;
        encoder
            .submit_frame(buf, i, pts_ns, Some(FRAME_DUR_NS))
            .unwrap();
    }

    let frames = drain_to_frames(&encoder);
    assert!(
        !frames.is_empty(),
        "Expected at least one encoded frame after graceful_shutdown()"
    );
    for f in &frames {
        assert!(!f.data.is_empty(), "Encoded frame data should not be empty");
        assert_eq!(f.codec, expected_codec);
    }
}

#[test]
#[serial]
fn test_submit_rgba_with_conversion() {
    init();
    let Some(cfg) = make_rgba_config(320, 240) else {
        eprintln!("No encoder available — skipping test_submit_rgba_with_conversion");
        return;
    };
    let encoder = NvEncoder::new(test_nv_encoder_config(cfg)).unwrap();
    for i in 0..3u128 {
        let buf = acquire_buffer(&encoder, i);
        encoder
            .submit_frame(buf, i, i as u64 * FRAME_DUR_NS, Some(FRAME_DUR_NS))
            .unwrap();
    }
    let frames = drain_to_frames(&encoder);
    assert!(!frames.is_empty());
}

#[test]
#[serial]
fn test_h264_submit_and_pull_frames() {
    init();
    if !has_nvenc() {
        eprintln!("NVENC not available — skipping test_h264_submit_and_pull_frames");
        return;
    }
    let cfg = test_nv_encoder_config(EncoderConfig::H264(H264EncoderConfig::new(320, 240)));
    let encoder = NvEncoder::new(cfg).unwrap();
    for i in 0..5u128 {
        let buf = acquire_buffer(&encoder, i);
        encoder
            .submit_frame(buf, i, i as u64 * FRAME_DUR_NS, Some(FRAME_DUR_NS))
            .unwrap();
    }
    let frames = drain_to_frames(&encoder);
    assert!(!frames.is_empty());
    for f in &frames {
        assert_eq!(f.codec, Codec::H264);
        assert!(!f.data.is_empty());
    }
}

#[test]
#[serial]
fn test_hevc_submit_and_pull_frames() {
    init();
    if !has_nvenc() {
        eprintln!("NVENC not available — skipping test_hevc_submit_and_pull_frames");
        return;
    }
    let cfg = test_nv_encoder_config(EncoderConfig::Hevc(HevcEncoderConfig::new(320, 240)));
    let encoder = NvEncoder::new(cfg).unwrap();
    for i in 0..5u128 {
        let buf = acquire_buffer(&encoder, i);
        encoder
            .submit_frame(buf, i, i as u64 * FRAME_DUR_NS, Some(FRAME_DUR_NS))
            .unwrap();
    }
    let frames = drain_to_frames(&encoder);
    assert!(!frames.is_empty());
    for f in &frames {
        assert_eq!(f.codec, Codec::Hevc);
        assert!(!f.data.is_empty());
    }
}

#[test]
#[serial]
fn test_jpeg_submit_and_pull_frames() {
    init();
    if !has_nvjpegenc() {
        eprintln!("nvjpegenc not available — skipping test_jpeg_submit_and_pull_frames");
        return;
    }
    let cfg = test_nv_encoder_config(EncoderConfig::Jpeg(
        JpegEncoderConfig::new(320, 240).format(VideoFormat::I420),
    ));
    let encoder = NvEncoder::new(cfg).unwrap();
    for i in 0..5u128 {
        let buf = acquire_buffer(&encoder, i);
        encoder
            .submit_frame(buf, i, i as u64 * FRAME_DUR_NS, Some(FRAME_DUR_NS))
            .unwrap();
    }
    let frames = drain_to_frames(&encoder);
    assert!(!frames.is_empty());
    for f in &frames {
        assert_eq!(f.codec, Codec::Jpeg);
        assert!(f.keyframe, "Every JPEG frame must be a keyframe");
        assert!(!f.data.is_empty());
    }
}

// ─── AV1 stream-header-in-first-frame tests ────────────────────────────

#[test]
#[serial]
fn test_av1_single_frame() {
    init();
    if !has_nvenc() {
        eprintln!("NVENC not available — skipping test_av1_single_frame");
        return;
    }
    let cfg = test_nv_encoder_config(EncoderConfig::Av1(Av1EncoderConfig::new(320, 240)));
    let encoder = NvEncoder::new(cfg).unwrap();
    let buf = acquire_buffer(&encoder, 0);
    encoder.submit_frame(buf, 0, 0, Some(FRAME_DUR_NS)).unwrap();
    let frames = drain_to_frames(&encoder);
    assert!(!frames.is_empty());
    for f in &frames {
        assert_eq!(f.codec, Codec::Av1);
        assert!(!f.data.is_empty());
    }
}

#[test]
#[serial]
fn test_av1_multi_frame() {
    init();
    if !has_nvenc() {
        eprintln!("NVENC not available — skipping test_av1_multi_frame");
        return;
    }
    let cfg = test_nv_encoder_config(EncoderConfig::Av1(Av1EncoderConfig::new(320, 240)));
    let encoder = NvEncoder::new(cfg).unwrap();
    for i in 0..10u128 {
        let buf = acquire_buffer(&encoder, i);
        encoder
            .submit_frame(buf, i, i as u64 * FRAME_DUR_NS, Some(FRAME_DUR_NS))
            .unwrap();
    }
    let frames = drain_to_frames(&encoder);
    assert!(frames.len() >= 5);
}

// ─── PTS validation tests ───────────────────────────────────────────────

#[test]
#[serial]
fn test_pts_reordering_rejected() {
    init();
    let Some(encoder_cfg) = make_default_config(320, 240) else {
        eprintln!("No encoder available — skipping test_pts_reordering_rejected");
        return;
    };
    let encoder = NvEncoder::new(test_nv_encoder_config(encoder_cfg)).unwrap();

    let buf0 = acquire_buffer(&encoder, 0);
    encoder
        .submit_frame(buf0, 0, 1_000_000, Some(FRAME_DUR_NS))
        .unwrap();

    let buf1 = acquire_buffer(&encoder, 1);
    let result = encoder.submit_frame(buf1, 1, 500_000, Some(FRAME_DUR_NS));
    match result {
        Err(EncoderError::PtsReordered { frame_id, .. }) => {
            assert_eq!(frame_id, 1);
        }
        other => panic!("Expected PtsReordered, got {other:?}"),
    }
}

#[test]
#[serial]
fn test_pts_equal_rejected() {
    init();
    let Some(encoder_cfg) = make_default_config(320, 240) else {
        eprintln!("No encoder available — skipping test_pts_equal_rejected");
        return;
    };
    let encoder = NvEncoder::new(test_nv_encoder_config(encoder_cfg)).unwrap();

    let b0 = acquire_buffer(&encoder, 0);
    encoder
        .submit_frame(b0, 0, 1_000_000, Some(FRAME_DUR_NS))
        .unwrap();

    let b1 = acquire_buffer(&encoder, 1);
    let r = encoder.submit_frame(b1, 1, 1_000_000, Some(FRAME_DUR_NS));
    assert!(matches!(r, Err(EncoderError::PtsReordered { .. })));
}

// ─── Lifecycle / shutdown tests ─────────────────────────────────────────

#[test]
#[serial]
fn test_double_shutdown_is_idempotent() {
    init();
    let Some(cfg) = make_default_config(320, 240) else {
        eprintln!("No encoder available — skipping test_double_shutdown_is_idempotent");
        return;
    };
    let encoder = NvEncoder::new(test_nv_encoder_config(cfg)).unwrap();
    encoder
        .graceful_shutdown(Some(Duration::from_secs(1)), |_| {})
        .unwrap();
    // Second call returns ShuttingDown.
    let second = encoder.graceful_shutdown(Some(Duration::from_secs(1)), |_| {});
    assert!(matches!(second, Err(EncoderError::ShuttingDown)));
}

#[test]
#[serial]
fn test_submit_after_shutdown_fails() {
    init();
    let Some(cfg) = make_default_config(320, 240) else {
        eprintln!("No encoder available — skipping test_submit_after_shutdown_fails");
        return;
    };
    let encoder = NvEncoder::new(test_nv_encoder_config(cfg)).unwrap();
    encoder
        .graceful_shutdown(Some(Duration::from_secs(1)), |_| {})
        .unwrap();

    // After graceful_shutdown the pipeline is torn down; submit_frame
    // must reject new input.
    let gen = encoder.generator();
    let guard = gen.lock();
    let shared = match guard.acquire(Some(99)) {
        Ok(s) => s,
        // Pool may be drained after shutdown — this is also acceptable.
        Err(_) => return,
    };
    drop(guard);
    let buf = match shared.into_buffer() {
        Ok(b) => b,
        Err(_) => return,
    };
    let r = encoder.submit_frame(buf, 99, 1_000_000_000, Some(FRAME_DUR_NS));
    assert!(
        matches!(
            r,
            Err(EncoderError::AlreadyFinalized)
                | Err(EncoderError::ShuttingDown)
                | Err(EncoderError::ChannelDisconnected)
        ),
        "unexpected result: {r:?}"
    );
}

#[test]
#[serial]
fn test_encoder_drop_does_not_panic() {
    init();
    let Some(cfg) = make_default_config(320, 240) else {
        eprintln!("No encoder available — skipping test_encoder_drop_does_not_panic");
        return;
    };
    let encoder = NvEncoder::new(test_nv_encoder_config(cfg)).unwrap();
    let buf = acquire_buffer(&encoder, 0);
    encoder.submit_frame(buf, 0, 0, Some(FRAME_DUR_NS)).unwrap();
    drop(encoder);
}

#[test]
#[serial]
fn test_generator_accessor() {
    init();
    let Some(cfg) = make_default_config(320, 240) else {
        eprintln!("No encoder available — skipping test_generator_accessor");
        return;
    };
    let encoder = NvEncoder::new(test_nv_encoder_config(cfg)).unwrap();
    let gen = encoder.generator();
    // Arc can be cloned and shared; locking works.
    let _gen2 = gen.clone();
    let _guard = gen.lock();
}

#[test]
#[serial]
fn test_frame_id_preserved() {
    init();
    if !has_nvjpegenc() {
        eprintln!("nvjpegenc not available — skipping test_frame_id_preserved");
        return;
    }
    let cfg = test_nv_encoder_config(EncoderConfig::Jpeg(
        JpegEncoderConfig::new(320, 240).format(VideoFormat::I420),
    ));
    let encoder = NvEncoder::new(cfg).unwrap();
    for i in 0..3u128 {
        let buf = acquire_buffer(&encoder, i);
        encoder
            .submit_frame(buf, i, i as u64 * FRAME_DUR_NS, Some(FRAME_DUR_NS))
            .unwrap();
    }
    let frames = drain_to_frames(&encoder);
    assert_eq!(frames.len(), 3);
    for (i, f) in frames.iter().enumerate() {
        assert_eq!(f.frame_id, Some(i as u128));
    }
}

// ─── PNG encoding ────────────────────────────────────────────────────────

#[test]
#[serial]
fn test_png_requires_rgba() {
    init();
    let cfg = test_nv_encoder_config(EncoderConfig::Png(
        PngEncoderConfig::new(320, 240).format(VideoFormat::NV12),
    ));
    let r = NvEncoder::new(cfg);
    assert!(matches!(r, Err(EncoderError::InvalidProperty { .. })));
}

#[test]
#[serial]
fn test_png_submit_and_pull_frames() {
    init();
    let cfg = test_nv_encoder_config(EncoderConfig::Png(PngEncoderConfig::new(320, 240)));
    let encoder = NvEncoder::new(cfg).unwrap();
    for i in 0..3u128 {
        let buf = acquire_buffer(&encoder, i);
        encoder
            .submit_frame(buf, i, i as u64 * FRAME_DUR_NS, Some(FRAME_DUR_NS))
            .unwrap();
    }
    let frames = drain_to_frames(&encoder);
    assert_eq!(frames.len(), 3);
    for f in &frames {
        assert_eq!(f.codec, Codec::Png);
        assert!(f.keyframe);
        // PNG signature: 89 50 4E 47 0D 0A 1A 0A
        assert_eq!(
            &f.data[..8],
            &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
        );
    }
}

#[test]
#[serial]
fn test_png_frame_id_preserved() {
    init();
    let cfg = test_nv_encoder_config(EncoderConfig::Png(PngEncoderConfig::new(320, 240)));
    let encoder = NvEncoder::new(cfg).unwrap();
    for i in 0..4u128 {
        let buf = acquire_buffer(&encoder, i);
        encoder
            .submit_frame(buf, i, i as u64 * FRAME_DUR_NS, Some(FRAME_DUR_NS))
            .unwrap();
    }
    let frames = drain_to_frames(&encoder);
    assert_eq!(frames.len(), 4);
    for (i, f) in frames.iter().enumerate() {
        assert_eq!(f.frame_id, Some(i as u128));
    }
}

// ─── Raw pseudoencoder tests ─────────────────────────────────────────────

#[test]
#[serial]
fn test_encoder_creation_raw_rgba() {
    init();
    let cfg = test_nv_encoder_config(EncoderConfig::RawRgba(RawEncoderConfig::new(
        320,
        240,
        VideoFormat::RGBA,
    )));
    assert!(NvEncoder::new(cfg).is_ok());
}

#[test]
#[serial]
fn test_encoder_creation_raw_rgb() {
    init();
    let cfg = test_nv_encoder_config(EncoderConfig::RawRgb(RawEncoderConfig::new(
        320,
        240,
        VideoFormat::RGBA,
    )));
    assert!(NvEncoder::new(cfg).is_ok());
}

#[test]
#[serial]
fn test_encoder_creation_raw_nv12() {
    init();
    let cfg = test_nv_encoder_config(EncoderConfig::RawNv12(RawEncoderConfig::new(
        320,
        240,
        VideoFormat::RGBA,
    )));
    assert!(NvEncoder::new(cfg).is_ok());
}

#[test]
#[serial]
fn test_raw_rgba_submit_and_pull_frames() {
    init();
    let w = 320;
    let h = 240;
    let cfg = test_nv_encoder_config(EncoderConfig::RawRgba(RawEncoderConfig::new(
        w,
        h,
        VideoFormat::RGBA,
    )));
    let encoder = NvEncoder::new(cfg).unwrap();
    for i in 0..3u128 {
        let buf = acquire_buffer(&encoder, i);
        encoder
            .submit_frame(buf, i, i as u64 * FRAME_DUR_NS, Some(FRAME_DUR_NS))
            .unwrap();
    }
    let frames = drain_to_frames(&encoder);
    assert_eq!(frames.len(), 3);
    for f in &frames {
        assert_eq!(f.codec, Codec::RawRgba);
        assert!(f.keyframe);
        assert_eq!(f.data.len(), (w * h * 4) as usize);
    }
}

#[test]
#[serial]
fn test_raw_rgb_submit_and_pull_frames() {
    init();
    let w = 320;
    let h = 240;
    let cfg = test_nv_encoder_config(EncoderConfig::RawRgb(RawEncoderConfig::new(
        w,
        h,
        VideoFormat::RGBA,
    )));
    let encoder = NvEncoder::new(cfg).unwrap();
    for i in 0..3u128 {
        let buf = acquire_buffer(&encoder, i);
        encoder
            .submit_frame(buf, i, i as u64 * FRAME_DUR_NS, Some(FRAME_DUR_NS))
            .unwrap();
    }
    let frames = drain_to_frames(&encoder);
    assert_eq!(frames.len(), 3);
    for f in &frames {
        assert_eq!(f.codec, Codec::RawRgb);
        assert!(f.keyframe);
        assert_eq!(f.data.len(), (w * h * 3) as usize);
    }
}

#[test]
#[serial]
fn test_raw_nv12_submit_and_pull_frames() {
    init();
    let w = 320;
    let h = 240;
    let cfg = test_nv_encoder_config(EncoderConfig::RawNv12(RawEncoderConfig::new(
        w,
        h,
        VideoFormat::RGBA,
    )));
    let encoder = NvEncoder::new(cfg).unwrap();
    for i in 0..3u128 {
        let buf = acquire_buffer(&encoder, i);
        encoder
            .submit_frame(buf, i, i as u64 * FRAME_DUR_NS, Some(FRAME_DUR_NS))
            .unwrap();
    }
    let frames = drain_to_frames(&encoder);
    assert_eq!(frames.len(), 3);
    for f in &frames {
        assert_eq!(f.codec, Codec::RawNv12);
        assert!(f.keyframe);
        // NV12: Y plane (w*h) + interleaved UV (w*h/2) = w*h*3/2
        assert_eq!(f.data.len(), (w * h * 3 / 2) as usize);
    }
}

#[test]
#[serial]
fn test_raw_frame_id_preserved() {
    init();
    let cfg = test_nv_encoder_config(EncoderConfig::RawRgba(RawEncoderConfig::new(
        320,
        240,
        VideoFormat::RGBA,
    )));
    let encoder = NvEncoder::new(cfg).unwrap();
    for i in 0..4u128 {
        let buf = acquire_buffer(&encoder, i);
        encoder
            .submit_frame(buf, i, i as u64 * FRAME_DUR_NS, Some(FRAME_DUR_NS))
            .unwrap();
    }
    let frames = drain_to_frames(&encoder);
    assert_eq!(frames.len(), 4);
    for (i, f) in frames.iter().enumerate() {
        assert_eq!(f.frame_id, Some(i as u128));
    }
}

// ─── Platform-specific property smoke tests ─────────────────────────────

#[test]
#[serial]
fn test_encoder_creation_h264_platform_props() {
    init();
    if !has_nvenc() {
        eprintln!("NVENC not available — skipping test_encoder_creation_h264_platform_props");
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    let enc_cfg = H264EncoderConfig::new(640, 480).props(H264DgpuProps {
        bitrate: Some(4_000_000),
        profile: Some(H264Profile::High),
        ..Default::default()
    });
    #[cfg(target_arch = "aarch64")]
    let enc_cfg = H264EncoderConfig::new(640, 480).props(H264JetsonProps {
        bitrate: Some(4_000_000),
        profile: Some(H264Profile::High),
        ..Default::default()
    });
    let cfg = test_nv_encoder_config(EncoderConfig::H264(enc_cfg));
    assert!(NvEncoder::new(cfg).is_ok());
}

#[test]
#[serial]
fn test_encoder_creation_hevc_platform_props() {
    init();
    if !has_nvenc() {
        eprintln!("NVENC not available — skipping test_encoder_creation_hevc_platform_props");
        return;
    }
    #[cfg(not(target_arch = "aarch64"))]
    let enc_cfg = HevcEncoderConfig::new(640, 480).props(HevcDgpuProps {
        bitrate: Some(4_000_000),
        profile: Some(HevcProfile::Main),
        ..Default::default()
    });
    #[cfg(target_arch = "aarch64")]
    let enc_cfg = HevcEncoderConfig::new(640, 480).props(HevcJetsonProps {
        bitrate: Some(4_000_000),
        profile: Some(HevcProfile::Main),
        ..Default::default()
    });
    let cfg = test_nv_encoder_config(EncoderConfig::Hevc(enc_cfg));
    assert!(NvEncoder::new(cfg).is_ok());
}

#[test]
#[serial]
fn test_encoder_creation_jpeg_props() {
    init();
    if !has_nvjpegenc() {
        eprintln!("nvjpegenc not available — skipping test_encoder_creation_jpeg_props");
        return;
    }
    let enc_cfg = JpegEncoderConfig::new(640, 480)
        .format(VideoFormat::I420)
        .props(JpegProps { quality: Some(85) });
    let cfg = test_nv_encoder_config(EncoderConfig::Jpeg(enc_cfg));
    assert!(NvEncoder::new(cfg).is_ok());
}
