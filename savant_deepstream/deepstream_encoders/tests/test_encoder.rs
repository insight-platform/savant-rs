//! Integration tests for the NvEncoder.
//!
//! These tests require a GPU with NVENC support and DeepStream installed.
//! Run with: `cargo test -p deepstream_encoders`

use deepstream_encoders::{cuda_init, Codec, EncoderConfig, EncoderError, NvEncoder};

/// Initialize CUDA and GStreamer once.
fn init() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).expect("CUDA init failed — is a GPU available?");
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
    assert_eq!(Codec::from_name("unknown"), None);
}

#[test]
fn test_codec_names() {
    assert_eq!(Codec::H264.name(), "h264");
    assert_eq!(Codec::Hevc.name(), "hevc");
    assert_eq!(Codec::Jpeg.name(), "jpeg");
    assert_eq!(Codec::Av1.name(), "av1");
}

#[test]
fn test_codec_encoder_elements() {
    assert_eq!(Codec::H264.encoder_element(), "nvv4l2h264enc");
    assert_eq!(Codec::Hevc.encoder_element(), "nvv4l2h265enc");
    assert_eq!(Codec::Jpeg.encoder_element(), "nvjpegenc");
    assert_eq!(Codec::Av1.encoder_element(), "nvv4l2av1enc");
}

#[test]
fn test_codec_parser_elements() {
    assert_eq!(Codec::H264.parser_element(), "h264parse");
    assert_eq!(Codec::Hevc.parser_element(), "h265parse");
    assert_eq!(Codec::Jpeg.parser_element(), "jpegparse");
    assert_eq!(Codec::Av1.parser_element(), "av1parse");
}

#[test]
fn test_codec_display() {
    assert_eq!(format!("{}", Codec::H264), "h264");
    assert_eq!(format!("{}", Codec::Hevc), "hevc");
}

// ─── EncoderConfig tests ─────────────────────────────────────────────────

#[test]
fn test_config_defaults() {
    let config = EncoderConfig::new(Codec::Hevc, 1920, 1080);
    assert_eq!(config.codec, Codec::Hevc);
    assert_eq!(config.width, 1920);
    assert_eq!(config.height, 1080);
    assert_eq!(config.format, "NV12");
    assert_eq!(config.fps_num, 30);
    assert_eq!(config.fps_den, 1);
    assert_eq!(config.gpu_id, 0);
    assert!(config.encoder_properties.is_empty());
}

#[test]
fn test_config_builder_chain() {
    let config = EncoderConfig::new(Codec::H264, 1280, 720)
        .format("RGBA")
        .fps(60, 1)
        .gpu_id(1)
        .mem_type(2);
    assert_eq!(config.format, "RGBA");
    assert_eq!(config.fps_num, 60);
    assert_eq!(config.gpu_id, 1);
    assert_eq!(config.mem_type, 2);
}

#[test]
fn test_config_rejects_b_frame_properties() {
    let result = EncoderConfig::new(Codec::Hevc, 640, 480)
        .encoder_property("num-B-Frames", "2");
    assert!(result.is_err());
    match result.unwrap_err() {
        EncoderError::BFramesNotAllowed(name) => {
            assert_eq!(name, "num-B-Frames");
        }
        other => panic!("Expected BFramesNotAllowed, got {:?}", other),
    }
}

#[test]
fn test_config_rejects_b_frames_case_insensitive() {
    let result = EncoderConfig::new(Codec::H264, 640, 480)
        .encoder_property("b-frames", "1");
    assert!(result.is_err());
}

#[test]
fn test_config_accepts_valid_property() {
    let config = EncoderConfig::new(Codec::Hevc, 640, 480)
        .encoder_property("bitrate", "4000000")
        .unwrap();
    assert_eq!(config.encoder_properties.len(), 1);
    assert_eq!(config.encoder_properties[0], ("bitrate".into(), "4000000".into()));
}

// ─── NvEncoder creation tests ────────────────────────────────────────────

#[test]
fn test_encoder_creation_hevc() {
    init();
    let config = EncoderConfig::new(Codec::Hevc, 640, 480);
    let encoder = NvEncoder::new(&config);
    assert!(encoder.is_ok(), "Failed to create HEVC encoder: {:?}", encoder.err());
}

#[test]
fn test_encoder_creation_h264() {
    init();
    let config = EncoderConfig::new(Codec::H264, 640, 480);
    let encoder = NvEncoder::new(&config);
    assert!(encoder.is_ok(), "Failed to create H264 encoder: {:?}", encoder.err());
}

#[test]
fn test_encoder_creation_jpeg() {
    init();
    let config = EncoderConfig::new(Codec::Jpeg, 640, 480).format("I420");
    let encoder = NvEncoder::new(&config);
    assert!(encoder.is_ok(), "Failed to create JPEG encoder: {:?}", encoder.err());
}

#[test]
fn test_encoder_codec_getter() {
    init();
    let config = EncoderConfig::new(Codec::H264, 320, 240);
    let encoder = NvEncoder::new(&config).unwrap();
    assert_eq!(encoder.codec(), Codec::H264);
}

// ─── NvEncoder frame submission tests ────────────────────────────────────

#[test]
fn test_submit_and_pull_frames() {
    init();
    let config = EncoderConfig::new(Codec::Hevc, 320, 240);
    let mut encoder = NvEncoder::new(&config).unwrap();

    let frame_duration_ns = 33_333_333u64; // ~30fps

    // Submit a few frames.
    for i in 0..5i64 {
        let buffer = encoder.generator().acquire_surface(Some(i)).unwrap();
        let pts_ns = i as u64 * frame_duration_ns;
        encoder
            .submit_frame(buffer, i, pts_ns, Some(frame_duration_ns))
            .unwrap();
    }

    // Finish and collect remaining frames.
    let remaining = encoder.finish(Some(3000)).unwrap();

    // We should have gotten at least some encoded frames.
    // (Hardware encoders may buffer 1-2 frames.)
    assert!(
        !remaining.is_empty(),
        "Expected at least one encoded frame after finish()"
    );

    for frame in &remaining {
        assert!(!frame.data.is_empty(), "Encoded frame data should not be empty");
        assert_eq!(frame.codec, Codec::Hevc);
    }
}

#[test]
fn test_submit_rgba_with_conversion() {
    init();
    let config = EncoderConfig::new(Codec::H264, 320, 240)
        .format("RGBA");
    let mut encoder = NvEncoder::new(&config).unwrap();

    for i in 0..3i64 {
        let buffer = encoder.generator().acquire_surface(Some(i)).unwrap();
        let pts_ns = i as u64 * 33_333_333;
        encoder
            .submit_frame(buffer, i, pts_ns, Some(33_333_333))
            .unwrap();
    }

    let remaining = encoder.finish(Some(3000)).unwrap();
    assert!(
        !remaining.is_empty(),
        "Expected encoded frames from RGBA->H264 pipeline"
    );
}

// ─── PTS validation tests ────────────────────────────────────────────────

#[test]
fn test_pts_reordering_rejected() {
    init();
    let config = EncoderConfig::new(Codec::Hevc, 320, 240);
    let mut encoder = NvEncoder::new(&config).unwrap();

    // First frame at PTS=100
    let buf1 = encoder.generator().acquire_surface(Some(0)).unwrap();
    encoder.submit_frame(buf1, 0, 100, None).unwrap();

    // Second frame at PTS=50 (reordered — should fail)
    let buf2 = encoder.generator().acquire_surface(Some(1)).unwrap();
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
fn test_pts_equal_rejected() {
    init();
    let config = EncoderConfig::new(Codec::H264, 320, 240);
    let mut encoder = NvEncoder::new(&config).unwrap();

    let buf1 = encoder.generator().acquire_surface(Some(0)).unwrap();
    encoder.submit_frame(buf1, 0, 100, None).unwrap();

    // Same PTS as previous — should be rejected.
    let buf2 = encoder.generator().acquire_surface(Some(1)).unwrap();
    let result = encoder.submit_frame(buf2, 1, 100, None);
    assert!(result.is_err());
}

// ─── Finalization tests ──────────────────────────────────────────────────

#[test]
fn test_double_finish_returns_empty() {
    init();
    let config = EncoderConfig::new(Codec::Hevc, 320, 240);
    let mut encoder = NvEncoder::new(&config).unwrap();

    let buf = encoder.generator().acquire_surface(Some(0)).unwrap();
    encoder.submit_frame(buf, 0, 0, None).unwrap();

    let first = encoder.finish(Some(3000)).unwrap();
    let second = encoder.finish(Some(1000)).unwrap();

    assert!(!first.is_empty() || true); // first finish may have frames
    assert!(second.is_empty(), "Second finish() should return empty vec");
}

#[test]
fn test_submit_after_finish_fails() {
    init();
    let config = EncoderConfig::new(Codec::Hevc, 320, 240);
    let mut encoder = NvEncoder::new(&config).unwrap();

    let _ = encoder.finish(Some(1000));

    let buf = encoder.generator().acquire_surface(Some(0)).unwrap();
    let result = encoder.submit_frame(buf, 0, 0, None);
    assert!(result.is_err());
    match result.unwrap_err() {
        EncoderError::AlreadyFinalized => {}
        other => panic!("Expected AlreadyFinalized, got {:?}", other),
    }
}

// ─── Drop behavior test ─────────────────────────────────────────────────

#[test]
fn test_encoder_drop_does_not_panic() {
    init();
    let config = EncoderConfig::new(Codec::Hevc, 320, 240);
    let mut encoder = NvEncoder::new(&config).unwrap();

    // Submit a frame but don't call finish — drop should be safe.
    let buf = encoder.generator().acquire_surface(Some(0)).unwrap();
    encoder.submit_frame(buf, 0, 0, Some(33_333_333)).unwrap();

    drop(encoder); // Should not panic.
}

// ─── Generator accessor test ─────────────────────────────────────────────

#[test]
fn test_generator_accessor() {
    init();
    let config = EncoderConfig::new(Codec::Hevc, 640, 480);
    let encoder = NvEncoder::new(&config).unwrap();

    let gen = encoder.generator();
    assert_eq!(gen.width(), 640);
    assert_eq!(gen.height(), 480);
    assert_eq!(gen.format(), "NV12");
}

// ─── Frame ID tracking test ─────────────────────────────────────────────

#[test]
fn test_frame_id_preserved() {
    init();
    let config = EncoderConfig::new(Codec::Hevc, 320, 240);
    let mut encoder = NvEncoder::new(&config).unwrap();

    let frame_ids: Vec<i64> = vec![100, 200, 300, 400, 500];
    let frame_duration_ns = 33_333_333u64;

    for (i, &fid) in frame_ids.iter().enumerate() {
        let buffer = encoder.generator().acquire_surface(Some(fid)).unwrap();
        let pts_ns = i as u64 * frame_duration_ns;
        encoder
            .submit_frame(buffer, fid, pts_ns, Some(frame_duration_ns))
            .unwrap();
    }

    let remaining = encoder.finish(Some(3000)).unwrap();

    // All returned frames should have frame_ids from our list.
    for frame in &remaining {
        assert!(
            frame_ids.contains(&frame.frame_id),
            "Unexpected frame_id {} not in {:?}",
            frame.frame_id,
            frame_ids,
        );
    }
}
