//! Integration tests for the NvEncoder.
//!
//! Tests that require NVENC (H.264/HEVC/AV1) are skipped when the hardware
//! is not available (e.g. Orin Nano).  General behaviour tests fall back to
//! JPEG encoding via `nvjpegenc` when possible.
//!
//! Run with: `cargo test -p deepstream_encoders`

use deepstream_encoders::prelude::*;
use serial_test::serial;

/// Initialize CUDA and GStreamer once.
fn init() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).expect("CUDA init failed — is a GPU available?");
}

fn has_nvenc() -> bool {
    nvidia_gpu_utils::has_nvenc(0).unwrap_or(false)
}

fn has_nvjpegenc() -> bool {
    let _ = gstreamer::init();
    gstreamer::ElementFactory::find("nvjpegenc").is_some()
}

fn is_jetson() -> bool {
    cfg!(target_arch = "aarch64")
}

/// Returns an encoder config using the best available codec, or `None` if
/// neither NVENC nor nvjpegenc is present.
fn make_default_config(w: u32, h: u32) -> Option<EncoderConfig> {
    if has_nvenc() {
        Some(EncoderConfig::new(Codec::Hevc, w, h))
    } else if has_nvjpegenc() {
        Some(EncoderConfig::new(Codec::Jpeg, w, h).format(VideoFormat::I420))
    } else {
        None
    }
}

/// Like [`make_default_config`] but with RGBA user format (tests format
/// conversion path: RGBA -> NV12 for NVENC or RGBA -> I420 for JPEG).
fn make_rgba_config(w: u32, h: u32) -> Option<EncoderConfig> {
    if has_nvenc() {
        Some(EncoderConfig::new(Codec::H264, w, h).format(VideoFormat::RGBA))
    } else if has_nvjpegenc() {
        Some(EncoderConfig::new(Codec::Jpeg, w, h).format(VideoFormat::RGBA))
    } else {
        None
    }
}

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
    assert_eq!(Codec::from_name("raw_rgba"), Some(Codec::RawRgba));
    assert_eq!(Codec::from_name("raw_rgb"), Some(Codec::RawRgb));
    assert_eq!(Codec::from_name("unknown"), None);
}

#[test]
fn test_codec_names() {
    assert_eq!(Codec::H264.name(), "h264");
    assert_eq!(Codec::Hevc.name(), "hevc");
    assert_eq!(Codec::Jpeg.name(), "jpeg");
    assert_eq!(Codec::Av1.name(), "av1");
    assert_eq!(Codec::Png.name(), "png");
    assert_eq!(Codec::RawRgba.name(), "raw_rgba");
    assert_eq!(Codec::RawRgb.name(), "raw_rgb");
}

#[test]
fn test_codec_encoder_elements() {
    assert_eq!(Codec::H264.encoder_element(), "nvv4l2h264enc");
    assert_eq!(Codec::Hevc.encoder_element(), "nvv4l2h265enc");
    assert_eq!(Codec::Jpeg.encoder_element(), "nvjpegenc");
    assert_eq!(Codec::Av1.encoder_element(), "nvv4l2av1enc");
    assert_eq!(Codec::Png.encoder_element(), "pngenc");
    assert_eq!(Codec::RawRgba.encoder_element(), "identity");
    assert_eq!(Codec::RawRgb.encoder_element(), "identity");
}

#[test]
fn test_codec_parser_elements() {
    assert_eq!(Codec::H264.parser_element(), "h264parse");
    assert_eq!(Codec::Hevc.parser_element(), "h265parse");
    assert_eq!(Codec::Jpeg.parser_element(), "jpegparse");
    assert_eq!(Codec::Av1.parser_element(), "av1parse");
    assert_eq!(Codec::Png.parser_element(), "identity");
    assert_eq!(Codec::RawRgba.parser_element(), "identity");
    assert_eq!(Codec::RawRgb.parser_element(), "identity");
}

#[test]
fn test_codec_display() {
    assert_eq!(format!("{}", Codec::H264), "h264");
    assert_eq!(format!("{}", Codec::Hevc), "hevc");
    assert_eq!(format!("{}", Codec::Jpeg), "jpeg");
    assert_eq!(format!("{}", Codec::Av1), "av1");
    assert_eq!(format!("{}", Codec::Png), "png");
    assert_eq!(format!("{}", Codec::RawRgba), "raw_rgba");
    assert_eq!(format!("{}", Codec::RawRgb), "raw_rgb");
}

// ─── EncoderConfig tests ─────────────────────────────────────────────────

#[test]
fn test_config_defaults() {
    let config = EncoderConfig::new(Codec::Hevc, 1920, 1080);
    assert_eq!(config.codec, Codec::Hevc);
    assert_eq!(config.width, 1920);
    assert_eq!(config.height, 1080);
    assert_eq!(config.format, VideoFormat::NV12);
    assert_eq!(config.fps_num, 30);
    assert_eq!(config.fps_den, 1);
    assert_eq!(config.gpu_id, 0);
    assert!(config.encoder_params.is_none());
}

#[test]
fn test_config_builder_chain() {
    let config = EncoderConfig::new(Codec::H264, 1280, 720)
        .format(VideoFormat::RGBA)
        .fps(60, 1)
        .gpu_id(1)
        .mem_type(NvBufSurfaceMemType::CudaDevice);
    assert_eq!(config.format, VideoFormat::RGBA);
    assert_eq!(config.fps_num, 60);
    assert_eq!(config.gpu_id, 1);
    assert_eq!(config.mem_type, NvBufSurfaceMemType::CudaDevice);
}

#[test]
fn test_config_with_typed_properties() {
    let props = EncoderProperties::HevcDgpu(HevcDgpuProps {
        bitrate: Some(4_000_000),
        ..Default::default()
    });
    let config = EncoderConfig::new(Codec::Hevc, 640, 480).properties(props);
    assert!(config.encoder_params.is_some());
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
    let config = EncoderConfig::new(Codec::Hevc, 640, 480);
    let encoder = NvEncoder::new(&config);
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
    let config = EncoderConfig::new(Codec::H264, 640, 480);
    let encoder = NvEncoder::new(&config);
    assert!(
        encoder.is_ok(),
        "Failed to create H264 encoder: {:?}",
        encoder.err()
    );
}

#[test]
#[serial]
fn test_encoder_creation_jpeg() {
    init();
    if !has_nvjpegenc() {
        eprintln!("nvjpegenc not available — skipping test_encoder_creation_jpeg");
        return;
    }
    let config = EncoderConfig::new(Codec::Jpeg, 640, 480).format(VideoFormat::I420);
    let encoder = NvEncoder::new(&config);
    assert!(
        encoder.is_ok(),
        "Failed to create JPEG encoder: {:?}",
        encoder.err()
    );
}

#[test]
#[serial]
fn test_encoder_creation_av1() {
    init();
    if !has_nvenc() {
        eprintln!("NVENC not available — skipping test_encoder_creation_av1");
        return;
    }
    let config = EncoderConfig::new(Codec::Av1, 640, 480);
    let encoder = NvEncoder::new(&config);
    assert!(
        encoder.is_ok(),
        "Failed to create AV1 encoder: {:?}",
        encoder.err()
    );
}

#[test]
#[serial]
fn test_encoder_codec_getter() {
    init();
    let Some(config) = make_default_config(320, 240) else {
        eprintln!("No encoder available — skipping test_encoder_codec_getter");
        return;
    };
    let expected_codec = config.codec;
    let encoder = NvEncoder::new(&config).unwrap();
    assert_eq!(encoder.codec(), expected_codec);
}

// ─── NvEncoder frame submission tests ────────────────────────────────────

#[test]
#[serial]
fn test_submit_and_pull_frames() {
    init();
    let Some(config) = make_default_config(320, 240) else {
        eprintln!("No encoder available — skipping test_submit_and_pull_frames");
        return;
    };
    let expected_codec = config.codec;
    let mut encoder = NvEncoder::new(&config).unwrap();

    let frame_duration_ns = 33_333_333u64; // ~30fps

    for i in 0..5u128 {
        let shared = encoder.generator().acquire(Some(i as i64)).unwrap();
        let buffer = shared.into_buffer().expect("sole owner");
        let pts_ns = i as u64 * frame_duration_ns;
        encoder
            .submit_frame(buffer, i, pts_ns, Some(frame_duration_ns))
            .unwrap();
    }

    let remaining = encoder.finish(Some(3000)).unwrap();

    assert!(
        !remaining.is_empty(),
        "Expected at least one encoded frame after finish()"
    );

    for frame in &remaining {
        assert!(
            !frame.data.is_empty(),
            "Encoded frame data should not be empty"
        );
        assert_eq!(frame.codec, expected_codec);
    }
}

#[test]
#[serial]
fn test_submit_rgba_with_conversion() {
    init();
    let Some(config) = make_rgba_config(320, 240) else {
        eprintln!("No encoder available — skipping test_submit_rgba_with_conversion");
        return;
    };
    let mut encoder = NvEncoder::new(&config).unwrap();

    for i in 0..3u128 {
        let shared = encoder.generator().acquire(Some(i as i64)).unwrap();
        let buffer = shared.into_buffer().expect("sole owner");
        let pts_ns = i as u64 * 33_333_333;
        encoder
            .submit_frame(buffer, i, pts_ns, Some(33_333_333))
            .unwrap();
    }

    let remaining = encoder.finish(Some(3000)).unwrap();
    assert!(
        !remaining.is_empty(),
        "Expected encoded frames from RGBA conversion pipeline"
    );
}

#[test]
#[serial]
fn test_h264_submit_and_pull_frames() {
    init();
    if !has_nvenc() {
        eprintln!("NVENC not available — skipping test_h264_submit_and_pull_frames");
        return;
    }
    let config = EncoderConfig::new(Codec::H264, 320, 240);
    let mut encoder = NvEncoder::new(&config).unwrap();

    let frame_duration_ns = 33_333_333u64;
    for i in 0..5u128 {
        let shared = encoder.generator().acquire(Some(i as i64)).unwrap();
        let buffer = shared.into_buffer().expect("sole owner");
        let pts_ns = i as u64 * frame_duration_ns;
        encoder
            .submit_frame(buffer, i, pts_ns, Some(frame_duration_ns))
            .unwrap();
    }

    let remaining = encoder.finish(Some(3000)).unwrap();
    assert!(
        !remaining.is_empty(),
        "Expected at least one encoded H264 frame after finish()"
    );
    for frame in &remaining {
        assert!(
            !frame.data.is_empty(),
            "Encoded H264 frame data should not be empty"
        );
        assert_eq!(frame.codec, Codec::H264);
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
    let config = EncoderConfig::new(Codec::Hevc, 320, 240);
    let mut encoder = NvEncoder::new(&config).unwrap();

    let frame_duration_ns = 33_333_333u64;
    for i in 0..5u128 {
        let shared = encoder.generator().acquire(Some(i as i64)).unwrap();
        let buffer = shared.into_buffer().expect("sole owner");
        let pts_ns = i as u64 * frame_duration_ns;
        encoder
            .submit_frame(buffer, i, pts_ns, Some(frame_duration_ns))
            .unwrap();
    }

    let remaining = encoder.finish(Some(3000)).unwrap();
    assert!(
        !remaining.is_empty(),
        "Expected at least one encoded HEVC frame after finish()"
    );
    for frame in &remaining {
        assert!(
            !frame.data.is_empty(),
            "Encoded HEVC frame data should not be empty"
        );
        assert_eq!(frame.codec, Codec::Hevc);
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
    let config = EncoderConfig::new(Codec::Jpeg, 320, 240).format(VideoFormat::I420);
    let mut encoder = NvEncoder::new(&config).unwrap();

    let frame_duration_ns = 33_333_333u64;
    for i in 0..5u128 {
        let shared = encoder.generator().acquire(Some(i as i64)).unwrap();
        let buffer = shared.into_buffer().expect("sole owner");
        let pts_ns = i as u64 * frame_duration_ns;
        encoder
            .submit_frame(buffer, i, pts_ns, Some(frame_duration_ns))
            .unwrap();
    }

    let remaining = encoder.finish(Some(3000)).unwrap();
    assert!(
        !remaining.is_empty(),
        "Expected at least one encoded JPEG frame after finish()"
    );
    for frame in &remaining {
        assert!(
            !frame.data.is_empty(),
            "Encoded JPEG frame data should not be empty"
        );
        assert_eq!(frame.codec, Codec::Jpeg);
        assert!(frame.keyframe, "Every JPEG frame must be a keyframe");
    }
}

// ─── AV1 encoding tests ─────────────────────────────────────────────────
//
// AV1 encoders emit a sequence header buffer before the first data frame,
// often with the same PTS.  These tests verify that the output PTS/DTS
// validation does not reject these codec-injected headers.

#[test]
#[serial]
fn test_av1_single_frame() {
    init();
    if !has_nvenc() {
        eprintln!("NVENC not available — skipping test_av1_single_frame");
        return;
    }
    let config = EncoderConfig::new(Codec::Av1, 320, 240);
    let mut encoder = NvEncoder::new(&config).unwrap();

    let shared = encoder.generator().acquire(Some(0)).unwrap();
    let buf = shared.into_buffer().expect("sole owner");
    encoder.submit_frame(buf, 0, 0, Some(33_333_333)).unwrap();

    let frames = encoder.finish(Some(5000)).unwrap();
    assert!(
        !frames.is_empty(),
        "Expected at least one encoded AV1 frame"
    );
    for f in &frames {
        assert!(!f.data.is_empty(), "AV1 frame data should not be empty");
        assert_eq!(f.codec, Codec::Av1);
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
    let config = EncoderConfig::new(Codec::Av1, 320, 240);
    let mut encoder = NvEncoder::new(&config).unwrap();

    let frame_dur = 33_333_333u64;
    for i in 0..10u128 {
        let shared = encoder.generator().acquire(Some(i as i64)).unwrap();
        let buf = shared.into_buffer().expect("sole owner");
        encoder
            .submit_frame(buf, i, i as u64 * frame_dur, Some(frame_dur))
            .unwrap();
    }

    let frames = encoder.finish(Some(5000)).unwrap();
    assert!(
        frames.len() >= 5,
        "Expected at least 5 encoded AV1 frames, got {}",
        frames.len()
    );
}

#[test]
#[serial]
fn test_av1_with_rgba_conversion() {
    init();
    if !has_nvenc() {
        eprintln!("NVENC not available — skipping test_av1_with_rgba_conversion");
        return;
    }
    let config = EncoderConfig::new(Codec::Av1, 320, 240).format(VideoFormat::RGBA);
    let mut encoder = NvEncoder::new(&config).unwrap();

    for i in 0..3u128 {
        let shared = encoder.generator().acquire(Some(i as i64)).unwrap();
        let buf = shared.into_buffer().expect("sole owner");
        encoder
            .submit_frame(buf, i, i as u64 * 33_333_333, Some(33_333_333))
            .unwrap();
    }

    let frames = encoder.finish(Some(5000)).unwrap();
    assert!(
        !frames.is_empty(),
        "Expected encoded frames from RGBA->AV1 pipeline"
    );
}

// ─── PTS validation tests ────────────────────────────────────────────────

#[test]
#[serial]
fn test_pts_reordering_rejected() {
    init();
    let Some(config) = make_default_config(320, 240) else {
        eprintln!("No encoder available — skipping test_pts_reordering_rejected");
        return;
    };
    let mut encoder = NvEncoder::new(&config).unwrap();

    // First frame at PTS=100
    let shared1 = encoder.generator().acquire(Some(0)).unwrap();
    let buf1 = shared1.into_buffer().expect("sole owner");
    encoder.submit_frame(buf1, 0, 100, None).unwrap();

    // Second frame at PTS=50 (reordered — should fail)
    let shared2 = encoder.generator().acquire(Some(1)).unwrap();
    let buf2 = shared2.into_buffer().expect("sole owner");
    let result = encoder.submit_frame(buf2, 1, 50, None);

    assert!(result.is_err());
    match result.unwrap_err() {
        EncoderError::PtsReordered {
            frame_id,
            pts_ns,
            prev_pts_ns,
        } => {
            assert_eq!(frame_id, 1);
            assert_eq!(pts_ns, 50);
            assert_eq!(prev_pts_ns, 100);
        }
        other => panic!("Expected PtsReordered, got {:?}", other),
    }
}

#[test]
#[serial]
fn test_pts_equal_rejected() {
    init();
    let Some(config) = make_default_config(320, 240) else {
        eprintln!("No encoder available — skipping test_pts_equal_rejected");
        return;
    };
    let mut encoder = NvEncoder::new(&config).unwrap();

    let shared1 = encoder.generator().acquire(Some(0)).unwrap();
    let buf1 = shared1.into_buffer().expect("sole owner");
    encoder.submit_frame(buf1, 0, 100, None).unwrap();

    // Same PTS as previous — should be rejected.
    let shared2 = encoder.generator().acquire(Some(1)).unwrap();
    let buf2 = shared2.into_buffer().expect("sole owner");
    let result = encoder.submit_frame(buf2, 1, 100, None);
    assert!(result.is_err());
}

// ─── Finalization tests ──────────────────────────────────────────────────

#[test]
#[serial]
fn test_double_finish_returns_empty() {
    init();
    let Some(config) = make_default_config(320, 240) else {
        eprintln!("No encoder available — skipping test_double_finish_returns_empty");
        return;
    };
    let mut encoder = NvEncoder::new(&config).unwrap();

    let shared = encoder.generator().acquire(Some(0)).unwrap();
    let buf = shared.into_buffer().expect("sole owner");
    encoder.submit_frame(buf, 0, 0, None).unwrap();

    let first = encoder.finish(Some(3000)).unwrap();
    let second = encoder.finish(Some(1000)).unwrap();

    // first finish may or may not have frames — no assertion needed.
    let _ = first;
    assert!(second.is_empty(), "Second finish() should return empty vec");
}

#[test]
#[serial]
fn test_submit_after_finish_fails() {
    init();
    let Some(config) = make_default_config(320, 240) else {
        eprintln!("No encoder available — skipping test_submit_after_finish_fails");
        return;
    };
    let mut encoder = NvEncoder::new(&config).unwrap();

    let _ = encoder.finish(Some(1000));

    let shared = encoder.generator().acquire(Some(0)).unwrap();
    let buf = shared.into_buffer().expect("sole owner");
    let result = encoder.submit_frame(buf, 0, 0, None);
    assert!(result.is_err());
    match result.unwrap_err() {
        EncoderError::AlreadyFinalized => {}
        other => panic!("Expected AlreadyFinalized, got {:?}", other),
    }
}

// ─── Drop behavior test ─────────────────────────────────────────────────

#[test]
#[serial]
fn test_encoder_drop_does_not_panic() {
    init();
    let Some(config) = make_default_config(320, 240) else {
        eprintln!("No encoder available — skipping test_encoder_drop_does_not_panic");
        return;
    };
    let mut encoder = NvEncoder::new(&config).unwrap();

    // Submit a frame but don't call finish — drop should be safe.
    let shared = encoder.generator().acquire(Some(0)).unwrap();
    let buf = shared.into_buffer().expect("sole owner");
    encoder.submit_frame(buf, 0, 0, Some(33_333_333)).unwrap();

    drop(encoder); // Should not panic.
}

// ─── Generator accessor test ─────────────────────────────────────────────

#[test]
#[serial]
fn test_generator_accessor() {
    init();
    let Some(config) = make_default_config(640, 480) else {
        eprintln!("No encoder available — skipping test_generator_accessor");
        return;
    };
    let expected_format = config.format;
    let encoder = NvEncoder::new(&config).unwrap();

    let gen = encoder.generator();
    assert_eq!(gen.width(), 640);
    assert_eq!(gen.height(), 480);
    assert_eq!(gen.format(), expected_format);
}

// ─── Frame ID tracking test ─────────────────────────────────────────────

#[test]
#[serial]
fn test_frame_id_preserved() {
    init();
    let Some(config) = make_default_config(320, 240) else {
        eprintln!("No encoder available — skipping test_frame_id_preserved");
        return;
    };
    let mut encoder = NvEncoder::new(&config).unwrap();

    let frame_ids: Vec<u128> = vec![100, 200, 300, 400, 500];
    let frame_duration_ns = 33_333_333u64;

    for (i, &fid) in frame_ids.iter().enumerate() {
        let shared = encoder.generator().acquire(Some(fid as i64)).unwrap();
        let buffer = shared.into_buffer().expect("sole owner");
        let pts_ns = i as u64 * frame_duration_ns;
        encoder
            .submit_frame(buffer, fid, pts_ns, Some(frame_duration_ns))
            .unwrap();
    }

    let remaining = encoder.finish(Some(3000)).unwrap();

    for frame in &remaining {
        let fid = frame
            .frame_id
            .expect("frame_id should be Some for user-submitted frames");
        assert!(
            frame_ids.contains(&fid),
            "Unexpected frame_id {} not in {:?}",
            fid,
            frame_ids,
        );
    }
}

// ─── PNG encoder tests ──────────────────────────────────────────────────
//
// PNG uses a CPU-based pipeline (appsrc -> nvvideoconvert -> pngenc ->
// appsink) and works on every platform — no NVENC or nvjpegenc required.

#[test]
#[serial]
fn test_encoder_creation_png() {
    init();
    let config = EncoderConfig::new(Codec::Png, 640, 480).format(VideoFormat::RGBA);
    let encoder = NvEncoder::new(&config);
    assert!(
        encoder.is_ok(),
        "Failed to create PNG encoder: {:?}",
        encoder.err()
    );
}

#[test]
#[serial]
fn test_png_requires_rgba() {
    init();
    let config = EncoderConfig::new(Codec::Png, 320, 240).format(VideoFormat::NV12);
    match NvEncoder::new(&config) {
        Err(EncoderError::InvalidProperty { name, .. }) => {
            assert_eq!(name, "format");
        }
        Err(other) => panic!("Expected InvalidProperty, got {:?}", other),
        Ok(_) => panic!("Expected error for PNG with NV12 format"),
    }
}

#[test]
#[serial]
fn test_png_submit_and_pull_frames() {
    init();
    let config = EncoderConfig::new(Codec::Png, 320, 240).format(VideoFormat::RGBA);
    let mut encoder = NvEncoder::new(&config).unwrap();

    let frame_duration_ns = 33_333_333u64;

    for i in 0..5u128 {
        let shared = encoder.generator().acquire(Some(i as i64)).unwrap();
        let buffer = shared.into_buffer().expect("sole owner");
        let pts_ns = i as u64 * frame_duration_ns;
        encoder
            .submit_frame(buffer, i, pts_ns, Some(frame_duration_ns))
            .unwrap();
    }

    let remaining = encoder.finish(Some(5000)).unwrap();

    assert!(
        !remaining.is_empty(),
        "Expected at least one encoded PNG frame after finish()"
    );

    for frame in &remaining {
        assert!(
            !frame.data.is_empty(),
            "Encoded PNG frame data should not be empty"
        );
        assert_eq!(frame.codec, Codec::Png);
        assert!(frame.keyframe, "Every PNG frame must be a keyframe");
    }
}

#[test]
#[serial]
fn test_png_with_compression_level() {
    init();
    let props = EncoderProperties::Png(PngProps {
        compression_level: Some(1),
    });
    let config = EncoderConfig::new(Codec::Png, 320, 240)
        .format(VideoFormat::RGBA)
        .properties(props);
    let mut encoder = NvEncoder::new(&config).unwrap();

    let shared = encoder.generator().acquire(Some(0)).unwrap();
    let buffer = shared.into_buffer().expect("sole owner");
    encoder
        .submit_frame(buffer, 0, 0, Some(33_333_333))
        .unwrap();

    let frames = encoder.finish(Some(5000)).unwrap();
    assert!(
        !frames.is_empty(),
        "Expected at least one encoded PNG frame"
    );
    assert!(!frames[0].data.is_empty());
}

#[test]
#[serial]
fn test_png_frame_id_preserved() {
    init();
    let config = EncoderConfig::new(Codec::Png, 320, 240).format(VideoFormat::RGBA);
    let mut encoder = NvEncoder::new(&config).unwrap();

    let frame_ids: Vec<u128> = vec![10, 20, 30];
    let frame_duration_ns = 33_333_333u64;

    for (i, &fid) in frame_ids.iter().enumerate() {
        let shared = encoder.generator().acquire(Some(fid as i64)).unwrap();
        let buffer = shared.into_buffer().expect("sole owner");
        let pts_ns = i as u64 * frame_duration_ns;
        encoder
            .submit_frame(buffer, fid, pts_ns, Some(frame_duration_ns))
            .unwrap();
    }

    let remaining = encoder.finish(Some(5000)).unwrap();

    for frame in &remaining {
        let fid = frame
            .frame_id
            .expect("frame_id should be Some for user-submitted frames");
        assert!(
            frame_ids.contains(&fid),
            "Unexpected frame_id {} not in {:?}",
            fid,
            frame_ids,
        );
    }
}

// ─── Raw frame download tests ────────────────────────────────────────────
//
// Raw pseudoencoders download GPU frames to CPU memory as tightly-packed
// pixel data. They always work — no NVENC or nvjpegenc required.

#[test]
#[serial]
fn test_encoder_creation_raw_rgba() {
    init();
    let config = EncoderConfig::new(Codec::RawRgba, 640, 480).format(VideoFormat::RGBA);
    let encoder = NvEncoder::new(&config);
    assert!(
        encoder.is_ok(),
        "Failed to create RawRgba encoder: {:?}",
        encoder.err()
    );
}

#[test]
#[serial]
fn test_encoder_creation_raw_rgb() {
    init();
    let config = EncoderConfig::new(Codec::RawRgb, 640, 480).format(VideoFormat::RGBA);
    let encoder = NvEncoder::new(&config);
    assert!(
        encoder.is_ok(),
        "Failed to create RawRgb encoder: {:?}",
        encoder.err()
    );
}

#[test]
#[serial]
fn test_raw_rgba_submit_and_pull_frames() {
    init();
    let config = EncoderConfig::new(Codec::RawRgba, 320, 240).format(VideoFormat::RGBA);
    let mut encoder = NvEncoder::new(&config).unwrap();

    let frame_duration_ns = 33_333_333u64;

    for i in 0..5u128 {
        let shared = encoder.generator().acquire(Some(i as i64)).unwrap();
        let buffer = shared.into_buffer().expect("sole owner");
        let pts_ns = i as u64 * frame_duration_ns;
        encoder
            .submit_frame(buffer, i, pts_ns, Some(frame_duration_ns))
            .unwrap();
    }

    let remaining = encoder.finish(Some(5000)).unwrap();

    assert!(
        !remaining.is_empty(),
        "Expected at least one raw RGBA frame after finish()"
    );

    let expected_size = 320 * 240 * 4; // W * H * 4 bytes per pixel
    for frame in &remaining {
        assert_eq!(
            frame.data.len(),
            expected_size,
            "Raw RGBA frame data should be {} bytes, got {}",
            expected_size,
            frame.data.len()
        );
        assert_eq!(frame.codec, Codec::RawRgba);
        assert!(frame.keyframe, "Every raw frame must be a keyframe");
    }
}

#[test]
#[serial]
fn test_raw_rgb_submit_and_pull_frames() {
    init();
    let config = EncoderConfig::new(Codec::RawRgb, 320, 240).format(VideoFormat::RGBA);
    let mut encoder = NvEncoder::new(&config).unwrap();

    let frame_duration_ns = 33_333_333u64;

    for i in 0..5u128 {
        let shared = encoder.generator().acquire(Some(i as i64)).unwrap();
        let buffer = shared.into_buffer().expect("sole owner");
        let pts_ns = i as u64 * frame_duration_ns;
        encoder
            .submit_frame(buffer, i, pts_ns, Some(frame_duration_ns))
            .unwrap();
    }

    let remaining = encoder.finish(Some(5000)).unwrap();

    assert!(
        !remaining.is_empty(),
        "Expected at least one raw RGB frame after finish()"
    );

    let expected_size = 320 * 240 * 3; // W * H * 3 bytes per pixel
    for frame in &remaining {
        assert_eq!(
            frame.data.len(),
            expected_size,
            "Raw RGB frame data should be {} bytes, got {}",
            expected_size,
            frame.data.len()
        );
        assert_eq!(frame.codec, Codec::RawRgb);
        assert!(frame.keyframe, "Every raw frame must be a keyframe");
    }
}

#[test]
#[serial]
fn test_raw_frame_id_preserved() {
    init();
    let config = EncoderConfig::new(Codec::RawRgba, 320, 240).format(VideoFormat::RGBA);
    let mut encoder = NvEncoder::new(&config).unwrap();

    let frame_ids: Vec<u128> = vec![10, 20, 30];
    let frame_duration_ns = 33_333_333u64;

    for (i, &fid) in frame_ids.iter().enumerate() {
        let shared = encoder.generator().acquire(Some(fid as i64)).unwrap();
        let buffer = shared.into_buffer().expect("sole owner");
        let pts_ns = i as u64 * frame_duration_ns;
        encoder
            .submit_frame(buffer, fid, pts_ns, Some(frame_duration_ns))
            .unwrap();
    }

    let remaining = encoder.finish(Some(5000)).unwrap();

    for frame in &remaining {
        let fid = frame
            .frame_id
            .expect("frame_id should be Some for user-submitted frames");
        assert!(
            frame_ids.contains(&fid),
            "Unexpected frame_id {} not in {:?}",
            fid,
            frame_ids,
        );
    }
}

#[test]
#[serial]
fn test_raw_rgba_from_nv12_input() {
    init();
    let config = EncoderConfig::new(Codec::RawRgba, 320, 240).format(VideoFormat::NV12);
    let mut encoder = NvEncoder::new(&config).unwrap();

    let shared = encoder.generator().acquire(Some(0)).unwrap();
    let buffer = shared.into_buffer().expect("sole owner");
    encoder
        .submit_frame(buffer, 0, 0, Some(33_333_333))
        .unwrap();

    let frames = encoder.finish(Some(5000)).unwrap();
    assert!(
        !frames.is_empty(),
        "Expected at least one frame from NV12->RGBA raw download"
    );
    let expected_size = 320 * 240 * 4;
    assert_eq!(frames[0].data.len(), expected_size);
    assert_eq!(frames[0].codec, Codec::RawRgba);
}

#[test]
#[serial]
fn test_raw_rgba_pixel_data_round_trip() {
    init();
    let w: u32 = 64;
    let h: u32 = 48;
    let bpp: usize = 4; // RGBA

    let config = EncoderConfig::new(Codec::RawRgba, w, h).format(VideoFormat::RGBA);
    let mut encoder = NvEncoder::new(&config).unwrap();

    // Generate a deterministic gradient pattern: each pixel's RGBA
    // values depend on its position.
    let mut input_pixels = vec![0u8; (w as usize) * (h as usize) * bpp];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let offset = (y * w as usize + x) * bpp;
            input_pixels[offset] = (x & 0xFF) as u8; // R
            input_pixels[offset + 1] = (y & 0xFF) as u8; // G
            input_pixels[offset + 2] = ((x + y) & 0xFF) as u8; // B
            input_pixels[offset + 3] = 255; // A
        }
    }

    // Upload the pattern to the GPU surface.
    let shared = encoder.generator().acquire(Some(0)).unwrap();
    let view = deepstream_buffers::SurfaceView::from_buffer(&shared, 0).unwrap();
    view.upload(&input_pixels, w, h, 4)
        .expect("upload_to_surface failed");
    drop(view);
    let buffer = shared.into_buffer().expect("sole owner");

    encoder
        .submit_frame(buffer, 42, 0, Some(33_333_333))
        .unwrap();

    let frames = encoder.finish(Some(5000)).unwrap();
    assert!(
        !frames.is_empty(),
        "Expected at least one frame from pixel round-trip"
    );

    let frame = &frames[0];
    assert_eq!(frame.frame_id, Some(42));
    assert_eq!(frame.data.len(), input_pixels.len());
    assert_eq!(
        frame.data, input_pixels,
        "Output pixels differ from input — raw RGBA round-trip is not lossless"
    );
}

#[test]
#[serial]
fn test_encoder_creation_h264_jetson_props() {
    init();
    if !is_jetson() || !has_nvenc() {
        eprintln!("Skipping — requires Jetson + NVENC");
        return;
    }
    let config = EncoderConfig::new(Codec::H264, 640, 480)
        .format(VideoFormat::RGBA)
        .properties(EncoderProperties::H264Jetson(H264JetsonProps {
            preset_level: Some(JetsonPresetLevel::UltraFast),
            maxperf_enable: Some(true),
            ..Default::default()
        }));
    let encoder = NvEncoder::new(&config);
    assert!(
        encoder.is_ok(),
        "Failed to create H264 Jetson encoder: {:?}",
        encoder.err()
    );
}

#[test]
#[serial]
fn test_encoder_creation_hevc_jetson_props() {
    init();
    if !is_jetson() || !has_nvenc() {
        eprintln!("Skipping — requires Jetson + NVENC");
        return;
    }
    let config = EncoderConfig::new(Codec::Hevc, 640, 480)
        .format(VideoFormat::RGBA)
        .properties(EncoderProperties::HevcJetson(HevcJetsonProps {
            preset_level: Some(JetsonPresetLevel::UltraFast),
            maxperf_enable: Some(true),
            ..Default::default()
        }));
    let encoder = NvEncoder::new(&config);
    assert!(
        encoder.is_ok(),
        "Failed to create HEVC Jetson encoder: {:?}",
        encoder.err()
    );
}

#[test]
#[serial]
fn test_encoder_creation_h264_dgpu_props() {
    init();
    if is_jetson() || !has_nvenc() {
        eprintln!("Skipping — requires dGPU + NVENC");
        return;
    }
    let config = EncoderConfig::new(Codec::H264, 640, 480)
        .format(VideoFormat::RGBA)
        .properties(EncoderProperties::H264Dgpu(H264DgpuProps {
            preset: Some(DgpuPreset::P1),
            tuning_info: Some(TuningPreset::LowLatency),
            ..Default::default()
        }));
    let encoder = NvEncoder::new(&config);
    assert!(
        encoder.is_ok(),
        "Failed to create H264 dGPU encoder: {:?}",
        encoder.err()
    );
}

#[test]
#[serial]
fn test_encoder_creation_hevc_dgpu_props() {
    init();
    if is_jetson() || !has_nvenc() {
        eprintln!("Skipping — requires dGPU + NVENC");
        return;
    }
    let config = EncoderConfig::new(Codec::Hevc, 640, 480)
        .format(VideoFormat::RGBA)
        .properties(EncoderProperties::HevcDgpu(HevcDgpuProps {
            preset: Some(DgpuPreset::P1),
            tuning_info: Some(TuningPreset::LowLatency),
            ..Default::default()
        }));
    let encoder = NvEncoder::new(&config);
    assert!(
        encoder.is_ok(),
        "Failed to create HEVC dGPU encoder: {:?}",
        encoder.err()
    );
}

#[test]
#[serial]
fn test_encoder_creation_av1_dgpu_props() {
    init();
    if is_jetson() || !has_nvenc() {
        eprintln!("Skipping — requires dGPU + NVENC");
        return;
    }
    let config = EncoderConfig::new(Codec::Av1, 640, 480)
        .format(VideoFormat::RGBA)
        .properties(EncoderProperties::Av1Dgpu(Av1DgpuProps {
            preset: Some(DgpuPreset::P1),
            tuning_info: Some(TuningPreset::LowLatency),
            ..Default::default()
        }));
    let encoder = NvEncoder::new(&config);
    assert!(
        encoder.is_ok(),
        "Failed to create AV1 dGPU encoder: {:?}",
        encoder.err()
    );
}

#[test]
#[serial]
fn test_encoder_creation_jpeg_props() {
    init();
    if !has_nvjpegenc() {
        eprintln!("nvjpegenc not available — skipping test_encoder_creation_jpeg_props");
        return;
    }
    let config = EncoderConfig::new(Codec::Jpeg, 640, 480)
        .format(VideoFormat::I420)
        .properties(EncoderProperties::Jpeg(JpegProps { quality: Some(90) }));
    let encoder = NvEncoder::new(&config);
    assert!(
        encoder.is_ok(),
        "Failed to create JPEG encoder with props: {:?}",
        encoder.err()
    );
}
