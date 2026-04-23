//! Integration tests for [`FlexibleDecoder`].
//!
//! These tests require a GPU with CUDA and nvjpegdec support.

use deepstream_decoders::cuda_init;
use deepstream_inputs::flexible_decoder::{
    FlexibleDecoder, FlexibleDecoderConfig, FlexibleDecoderError, FlexibleDecoderOutput, SkipReason,
};
use parking_lot::Mutex;
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
};
use savant_core::primitives::video_codec::VideoCodec;
use serial_test::serial;
use std::sync::Arc;
use std::time::Duration;

fn init() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).expect("CUDA init failed");
}

/// Build a synthetic JPEG image at the given dimensions.
fn make_jpeg(width: u32, height: u32) -> Vec<u8> {
    let img = image::RgbaImage::from_pixel(width, height, image::Rgba([128, 64, 32, 255]));
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Jpeg).unwrap();
    buf.into_inner()
}

/// Build a synthetic PNG image at the given dimensions.
fn make_png(width: u32, height: u32) -> Vec<u8> {
    let img = image::RgbaImage::from_pixel(width, height, image::Rgba([64, 128, 255, 255]));
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
    buf.into_inner()
}

/// Build a [`VideoFrameProxy`] for testing.
fn make_frame(
    source_id: &str,
    width: i64,
    height: i64,
    codec: Option<VideoCodec>,
    content: VideoFrameContent,
    pts: i64,
) -> VideoFrameProxy {
    VideoFrameProxy::new(
        source_id,
        (30, 1),
        width,
        height,
        content,
        VideoFrameTranscodingMethod::Copy,
        codec,
        None,
        (1, 1_000_000_000),
        pts,
        None,
        None,
    )
    .unwrap()
}

/// Collector for callback outputs.
#[derive(Clone)]
struct OutputCollector {
    outputs: Arc<Mutex<Vec<FlexibleDecoderOutput>>>,
}

impl OutputCollector {
    fn new() -> Self {
        Self {
            outputs: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn callback(&self) -> impl Fn(FlexibleDecoderOutput) + Send + Sync + 'static {
        let outputs = self.outputs.clone();
        move |out| outputs.lock().push(out)
    }

    fn drain(&self) -> Vec<FlexibleDecoderOutput> {
        std::mem::take(&mut *self.outputs.lock())
    }

    fn wait_for_frames(&self, count: usize, timeout: Duration) {
        let start = std::time::Instant::now();
        loop {
            let frame_count = self.frame_count();
            if frame_count >= count {
                return;
            }
            if start.elapsed() > timeout {
                panic!(
                    "timeout waiting for {count} frames (got {frame_count} after {:?})",
                    start.elapsed()
                );
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn wait_for<F>(&self, predicate: F, timeout: Duration)
    where
        F: Fn(&FlexibleDecoderOutput) -> bool,
    {
        let start = std::time::Instant::now();
        loop {
            if self.outputs.lock().iter().any(&predicate) {
                return;
            }
            if start.elapsed() > timeout {
                let outputs = self.drain();
                panic!(
                    "timeout waiting for matching output after {:?}; collected: {outputs:?}",
                    start.elapsed()
                );
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn frame_count(&self) -> usize {
        self.outputs
            .lock()
            .iter()
            .filter(|o| matches!(o, FlexibleDecoderOutput::Frame { .. }))
            .count()
    }

    fn assert_frame_uuid_coverage(&self, submitted: &[u128]) {
        let outputs = self.outputs.lock();
        let mut output_uuids: Vec<u128> = outputs
            .iter()
            .filter_map(|o| match o {
                FlexibleDecoderOutput::Frame { frame, decoded, .. } => {
                    assert_eq!(
                        Some(frame.get_uuid_u128()),
                        decoded.frame_id,
                        "proxy UUID must match decoded frame_id"
                    );
                    Some(frame.get_uuid_u128())
                }
                _ => None,
            })
            .collect();
        output_uuids.sort();
        let mut expected = submitted.to_vec();
        expected.sort();
        assert_eq!(
            expected, output_uuids,
            "submitted UUIDs must exactly match Frame output UUIDs"
        );
    }
}

fn default_config(source_id: &str) -> FlexibleDecoderConfig {
    FlexibleDecoderConfig::new(source_id, 0, 4)
        .idle_timeout(Duration::from_secs(2))
        .detect_buffer_limit(30)
}

// ── Tests ──────────────────────────────────────────────────────────────

#[test]
#[serial]
fn test_source_id_mismatch() {
    init();
    let collector = OutputCollector::new();
    let dec = FlexibleDecoder::new(default_config("cam-1"), collector.callback());

    let jpeg_data = make_jpeg(320, 240);
    let frame = make_frame(
        "cam-2",
        320,
        240,
        Some(VideoCodec::Jpeg),
        VideoFrameContent::Internal(jpeg_data),
        0,
    );
    dec.submit(&frame, None).unwrap();

    let outputs = collector.drain();
    assert_eq!(outputs.len(), 1);
    match &outputs[0] {
        FlexibleDecoderOutput::Skipped { reason, .. } => {
            assert!(matches!(reason, SkipReason::SourceIdMismatch { .. }));
        }
        other => panic!("expected SourceIdMismatch, got: {other:?}"),
    }
}

#[test]
#[serial]
fn test_no_payload() {
    init();
    let collector = OutputCollector::new();
    let dec = FlexibleDecoder::new(default_config("cam-1"), collector.callback());

    let frame = make_frame(
        "cam-1",
        320,
        240,
        Some(VideoCodec::Jpeg),
        VideoFrameContent::None,
        0,
    );
    dec.submit(&frame, None).unwrap();

    let outputs = collector.drain();
    assert_eq!(outputs.len(), 1);
    match &outputs[0] {
        FlexibleDecoderOutput::Skipped { reason, .. } => {
            assert!(matches!(reason, SkipReason::NoPayload));
        }
        other => panic!("expected NoPayload, got: {other:?}"),
    }
}

#[test]
#[serial]
fn test_unsupported_codec() {
    init();
    let collector = OutputCollector::new();
    let dec = FlexibleDecoder::new(default_config("cam-1"), collector.callback());

    let frame = make_frame("cam-1", 320, 240, None, VideoFrameContent::None, 0);
    dec.submit(&frame, Some(&[1, 2, 3])).unwrap();

    let outputs = collector.drain();
    assert_eq!(outputs.len(), 1);
    match &outputs[0] {
        FlexibleDecoderOutput::Skipped { reason, .. } => {
            assert!(matches!(reason, SkipReason::UnsupportedCodec(None)));
        }
        other => panic!("expected UnsupportedCodec, got: {other:?}"),
    }
}

#[test]
#[serial]
fn test_jpeg_normal_decode() {
    init();
    let collector = OutputCollector::new();
    let dec = FlexibleDecoder::new(default_config("cam-1"), collector.callback());

    let jpeg_data = make_jpeg(320, 240);
    let num_frames = 3;
    let mut submitted_uuids = Vec::new();
    for i in 0..num_frames {
        let frame = make_frame(
            "cam-1",
            320,
            240,
            Some(VideoCodec::Jpeg),
            VideoFrameContent::None,
            i as i64 * 33_333_333,
        );
        submitted_uuids.push(frame.get_uuid_u128());
        dec.submit(&frame, Some(&jpeg_data)).unwrap();
    }

    collector.wait_for_frames(num_frames, Duration::from_secs(10));
    assert_eq!(collector.frame_count(), num_frames);
    collector.assert_frame_uuid_coverage(&submitted_uuids);
}

#[test]
#[serial]
fn test_source_eos_idle() {
    init();
    let collector = OutputCollector::new();
    let dec = FlexibleDecoder::new(default_config("cam-1"), collector.callback());

    dec.source_eos("cam-1").unwrap();

    let outputs = collector.drain();
    assert_eq!(outputs.len(), 1);
    match &outputs[0] {
        FlexibleDecoderOutput::SourceEos { source_id } => {
            assert_eq!(source_id, "cam-1");
        }
        other => panic!("expected SourceEos, got: {other:?}"),
    }
}

#[test]
#[serial]
fn test_source_eos_active() {
    init();
    let collector = OutputCollector::new();
    let dec = FlexibleDecoder::new(default_config("cam-1"), collector.callback());

    let jpeg_data = make_jpeg(320, 240);
    let frame1 = make_frame(
        "cam-1",
        320,
        240,
        Some(VideoCodec::Jpeg),
        VideoFrameContent::None,
        0,
    );
    let mut submitted_uuids = vec![frame1.get_uuid_u128()];
    dec.submit(&frame1, Some(&jpeg_data)).unwrap();
    collector.wait_for_frames(1, Duration::from_secs(5));

    dec.source_eos("cam-1").unwrap();

    // Submit another frame to push the custom EOS event through the
    // GStreamer pipeline (some elements buffer events until the next
    // buffer arrives).
    let frame2 = make_frame(
        "cam-1",
        320,
        240,
        Some(VideoCodec::Jpeg),
        VideoFrameContent::None,
        33_333_333,
    );
    submitted_uuids.push(frame2.get_uuid_u128());
    dec.submit(&frame2, Some(&jpeg_data)).unwrap();

    collector.wait_for(
        |o| matches!(o, FlexibleDecoderOutput::SourceEos { .. }),
        Duration::from_secs(5),
    );
    collector.wait_for_frames(2, Duration::from_secs(5));
    collector.assert_frame_uuid_coverage(&submitted_uuids);
}

#[test]
#[serial]
fn test_graceful_shutdown() {
    init();
    let collector = OutputCollector::new();
    let dec = FlexibleDecoder::new(default_config("cam-1"), collector.callback());

    let jpeg_data = make_jpeg(320, 240);
    let mut submitted_uuids = Vec::new();
    for i in 0..3 {
        let frame = make_frame(
            "cam-1",
            320,
            240,
            Some(VideoCodec::Jpeg),
            VideoFrameContent::None,
            i * 33_333_333,
        );
        submitted_uuids.push(frame.get_uuid_u128());
        dec.submit(&frame, Some(&jpeg_data)).unwrap();
    }

    collector.wait_for_frames(3, Duration::from_secs(10));
    collector.assert_frame_uuid_coverage(&submitted_uuids);
    dec.graceful_shutdown().unwrap();

    let result = dec.submit(
        &make_frame(
            "cam-1",
            320,
            240,
            Some(VideoCodec::Jpeg),
            VideoFrameContent::None,
            100,
        ),
        Some(&jpeg_data),
    );
    assert!(matches!(result, Err(FlexibleDecoderError::ShutDown)));
}

#[test]
#[serial]
fn test_shutdown_immediate() {
    init();
    let collector = OutputCollector::new();
    let dec = FlexibleDecoder::new(default_config("cam-1"), collector.callback());

    let jpeg_data = make_jpeg(320, 240);
    let frame = make_frame(
        "cam-1",
        320,
        240,
        Some(VideoCodec::Jpeg),
        VideoFrameContent::None,
        0,
    );
    dec.submit(&frame, Some(&jpeg_data)).unwrap();

    dec.shutdown();

    let result = dec.source_eos("cam-1");
    assert!(matches!(result, Err(FlexibleDecoderError::ShutDown)));
}

#[test]
#[serial]
fn test_resolution_change() {
    init();
    let collector = OutputCollector::new();
    let dec = FlexibleDecoder::new(default_config("cam-1"), collector.callback());

    let jpeg_320 = make_jpeg(320, 240);
    let mut submitted_uuids = Vec::new();
    for i in 0..2 {
        let frame = make_frame(
            "cam-1",
            320,
            240,
            Some(VideoCodec::Jpeg),
            VideoFrameContent::None,
            i * 33_333_333,
        );
        submitted_uuids.push(frame.get_uuid_u128());
        dec.submit(&frame, Some(&jpeg_320)).unwrap();
    }
    collector.wait_for_frames(2, Duration::from_secs(10));

    let jpeg_640 = make_jpeg(640, 480);
    let frame_big = make_frame(
        "cam-1",
        640,
        480,
        Some(VideoCodec::Jpeg),
        VideoFrameContent::None,
        2 * 33_333_333,
    );
    submitted_uuids.push(frame_big.get_uuid_u128());
    // ParameterChange is emitted synchronously during this submit call.
    dec.submit(&frame_big, Some(&jpeg_640)).unwrap();

    // Wait for the ParameterChange event.
    collector.wait_for(
        |o| matches!(o, FlexibleDecoderOutput::ParameterChange { .. }),
        Duration::from_secs(10),
    );

    // Also wait for the new decoder to produce a frame.
    collector.wait_for_frames(3, Duration::from_secs(10));
    collector.assert_frame_uuid_coverage(&submitted_uuids);
}

#[test]
#[serial]
fn test_codec_change_jpeg_to_png() {
    init();
    let collector = OutputCollector::new();
    let dec = FlexibleDecoder::new(default_config("cam-1"), collector.callback());

    let jpeg_data = make_jpeg(320, 240);
    let frame_jpeg = make_frame(
        "cam-1",
        320,
        240,
        Some(VideoCodec::Jpeg),
        VideoFrameContent::None,
        0,
    );
    let mut submitted_uuids = vec![frame_jpeg.get_uuid_u128()];
    dec.submit(&frame_jpeg, Some(&jpeg_data)).unwrap();
    collector.wait_for_frames(1, Duration::from_secs(10));

    let img = image::RgbaImage::from_pixel(320, 240, image::Rgba([64, 128, 255, 255]));
    let mut png_buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut png_buf, image::ImageFormat::Png).unwrap();
    let png_data = png_buf.into_inner();

    let frame_png = make_frame(
        "cam-1",
        320,
        240,
        Some(VideoCodec::Png),
        VideoFrameContent::None,
        33_333_333,
    );
    submitted_uuids.push(frame_png.get_uuid_u128());
    // ParameterChange emitted synchronously during this submit call.
    dec.submit(&frame_png, Some(&png_data)).unwrap();

    collector.wait_for(
        |o| matches!(o, FlexibleDecoderOutput::ParameterChange { .. }),
        Duration::from_secs(10),
    );

    collector.wait_for_frames(2, Duration::from_secs(10));
    collector.assert_frame_uuid_coverage(&submitted_uuids);
}

#[test]
#[serial]
fn test_detection_buffer_overflow() {
    init();
    let collector = OutputCollector::new();
    let dec = FlexibleDecoder::new(
        default_config("cam-1").detect_buffer_limit(3),
        collector.callback(),
    );

    let dummy_data = vec![0u8; 100];
    for i in 0..5 {
        let frame = make_frame(
            "cam-1",
            320,
            240,
            Some(VideoCodec::H264),
            VideoFrameContent::None,
            i * 33_333_333,
        );
        dec.submit(&frame, Some(&dummy_data)).unwrap();
    }

    let outputs = collector.drain();
    let overflow_count = outputs
        .iter()
        .filter(|o| {
            matches!(
                o,
                FlexibleDecoderOutput::Skipped {
                    reason: SkipReason::DetectionBufferOverflow,
                    ..
                }
            )
        })
        .count();
    assert!(
        overflow_count > 0,
        "expected DetectionBufferOverflow skips, got: {outputs:?}"
    );
}

#[test]
#[serial]
fn test_payload_from_internal_content() {
    init();
    let collector = OutputCollector::new();
    let dec = FlexibleDecoder::new(default_config("cam-1"), collector.callback());

    let jpeg_data = make_jpeg(320, 240);
    let frame = make_frame(
        "cam-1",
        320,
        240,
        Some(VideoCodec::Jpeg),
        VideoFrameContent::Internal(jpeg_data),
        0,
    );
    let submitted_uuids = vec![frame.get_uuid_u128()];
    dec.submit(&frame, None).unwrap();

    collector.wait_for_frames(1, Duration::from_secs(10));
    assert_eq!(collector.frame_count(), 1);
    collector.assert_frame_uuid_coverage(&submitted_uuids);
}

#[test]
#[serial]
fn test_graceful_shutdown_detecting() {
    init();
    let collector = OutputCollector::new();
    let dec = FlexibleDecoder::new(default_config("cam-1"), collector.callback());

    let dummy_data = vec![0u8; 100];
    let frame = make_frame(
        "cam-1",
        320,
        240,
        Some(VideoCodec::H264),
        VideoFrameContent::None,
        0,
    );
    dec.submit(&frame, Some(&dummy_data)).unwrap();

    dec.graceful_shutdown().unwrap();

    let outputs = collector.drain();
    let overflow_count = outputs
        .iter()
        .filter(|o| {
            matches!(
                o,
                FlexibleDecoderOutput::Skipped {
                    reason: SkipReason::WaitingForKeyframe,
                    ..
                }
            )
        })
        .count();
    assert_eq!(
        overflow_count, 1,
        "graceful_shutdown during detection should flush buffered as WaitingForKeyframe"
    );
}

#[test]
#[serial]
fn test_frame_output_carries_video_frame_proxy() {
    init();
    let collector = OutputCollector::new();
    let dec = FlexibleDecoder::new(default_config("cam-1"), collector.callback());

    let jpeg_data = make_jpeg(320, 240);
    let mut submitted_uuids = Vec::new();
    for i in 0..3 {
        let frame = make_frame(
            "cam-1",
            320,
            240,
            Some(VideoCodec::Jpeg),
            VideoFrameContent::None,
            i as i64 * 33_333_333,
        );
        submitted_uuids.push(frame.get_uuid_u128());
        dec.submit(&frame, Some(&jpeg_data)).unwrap();
    }

    collector.wait_for_frames(3, Duration::from_secs(10));

    let outputs = collector.drain();
    let mut returned_uuids: Vec<u128> = outputs
        .iter()
        .filter_map(|o| match o {
            FlexibleDecoderOutput::Frame { frame, decoded, .. } => {
                assert_eq!(frame.get_source_id(), "cam-1");
                assert_eq!(frame.get_uuid_u128(), decoded.frame_id.unwrap());
                Some(frame.get_uuid_u128())
            }
            _ => None,
        })
        .collect();
    returned_uuids.sort();
    submitted_uuids.sort();
    assert_eq!(
        submitted_uuids, returned_uuids,
        "every submitted VideoFrameProxy UUID must appear in Frame outputs"
    );
}

#[test]
#[serial]
fn test_skipped_data_none_for_source_id_mismatch() {
    init();
    let collector = OutputCollector::new();
    let dec = FlexibleDecoder::new(default_config("cam-1"), collector.callback());

    let jpeg_data = make_jpeg(320, 240);
    let frame = make_frame(
        "cam-2",
        320,
        240,
        Some(VideoCodec::Jpeg),
        VideoFrameContent::Internal(jpeg_data),
        0,
    );
    dec.submit(&frame, None).unwrap();

    let outputs = collector.drain();
    assert_eq!(outputs.len(), 1);
    match &outputs[0] {
        FlexibleDecoderOutput::Skipped {
            frame: skipped_frame,
            data,
            reason,
        } => {
            assert!(matches!(reason, SkipReason::SourceIdMismatch { .. }));
            assert!(
                data.is_none(),
                "data must be None before payload extraction"
            );
            assert_eq!(skipped_frame.get_uuid_u128(), frame.get_uuid_u128());
        }
        other => panic!("expected Skipped(SourceIdMismatch), got: {other:?}"),
    }
}

#[test]
#[serial]
fn test_skipped_data_none_for_no_payload() {
    init();
    let collector = OutputCollector::new();
    let dec = FlexibleDecoder::new(default_config("cam-1"), collector.callback());

    let frame = make_frame(
        "cam-1",
        320,
        240,
        Some(VideoCodec::Jpeg),
        VideoFrameContent::None,
        0,
    );
    dec.submit(&frame, None).unwrap();

    let outputs = collector.drain();
    assert_eq!(outputs.len(), 1);
    match &outputs[0] {
        FlexibleDecoderOutput::Skipped {
            frame: skipped_frame,
            data,
            reason,
        } => {
            assert!(matches!(reason, SkipReason::NoPayload));
            assert!(data.is_none(), "data must be None when no payload exists");
            assert_eq!(skipped_frame.get_uuid_u128(), frame.get_uuid_u128());
        }
        other => panic!("expected Skipped(NoPayload), got: {other:?}"),
    }
}

#[test]
#[serial]
fn test_skipped_data_some_for_unsupported_codec() {
    init();
    let collector = OutputCollector::new();
    let dec = FlexibleDecoder::new(default_config("cam-1"), collector.callback());

    let payload = vec![10, 20, 30];
    let frame = make_frame("cam-1", 320, 240, None, VideoFrameContent::None, 0);
    dec.submit(&frame, Some(&payload)).unwrap();

    let outputs = collector.drain();
    assert_eq!(outputs.len(), 1);
    match &outputs[0] {
        FlexibleDecoderOutput::Skipped {
            frame: skipped_frame,
            data,
            reason,
        } => {
            assert!(matches!(reason, SkipReason::UnsupportedCodec(None)));
            assert_eq!(
                data.as_deref(),
                Some(payload.as_slice()),
                "data must carry the exact submitted payload"
            );
            assert_eq!(skipped_frame.get_uuid_u128(), frame.get_uuid_u128());
        }
        other => panic!("expected Skipped(UnsupportedCodec), got: {other:?}"),
    }
}

#[test]
#[serial]
fn test_waiting_for_keyframe_on_codec_change() {
    init();
    let collector = OutputCollector::new();
    let dec = FlexibleDecoder::new(default_config("cam-1"), collector.callback());

    let dummy_h264 = vec![0u8; 64];
    let mut buffered_uuids = Vec::new();
    for i in 0..3 {
        let frame = make_frame(
            "cam-1",
            320,
            240,
            Some(VideoCodec::H264),
            VideoFrameContent::None,
            i * 33_333_333,
        );
        buffered_uuids.push(frame.get_uuid_u128());
        dec.submit(&frame, Some(&dummy_h264)).unwrap();
    }

    // All 3 frames are buffered in Detecting state (no keyframe found).
    // Now switch codec → triggers WaitingForKeyframe for the buffered packets.
    let jpeg_data = make_jpeg(320, 240);
    let jpeg_frame = make_frame(
        "cam-1",
        320,
        240,
        Some(VideoCodec::Jpeg),
        VideoFrameContent::None,
        3 * 33_333_333,
    );
    dec.submit(&jpeg_frame, Some(&jpeg_data)).unwrap();

    collector.wait_for_frames(1, Duration::from_secs(10));
    let outputs = collector.drain();

    let skipped: Vec<_> = outputs
        .iter()
        .filter_map(|o| match o {
            FlexibleDecoderOutput::Skipped {
                frame,
                data,
                reason,
            } => Some((frame, data, reason)),
            _ => None,
        })
        .collect();

    assert_eq!(
        skipped.len(),
        3,
        "all 3 buffered H.264 frames should be skipped, got: {outputs:?}"
    );
    for (skipped_frame, data, reason) in &skipped {
        assert_eq!(
            **reason,
            SkipReason::WaitingForKeyframe,
            "reason must be WaitingForKeyframe, not DetectionBufferOverflow"
        );
        assert!(
            buffered_uuids.contains(&skipped_frame.get_uuid_u128()),
            "skipped frame UUID must match a buffered H.264 frame"
        );
        assert_eq!(
            data.as_deref(),
            Some(dummy_h264.as_slice()),
            "data must carry the buffered payload"
        );
    }

    let frame_outputs: Vec<_> = outputs
        .iter()
        .filter(|o| matches!(o, FlexibleDecoderOutput::Frame { .. }))
        .collect();
    assert!(
        !frame_outputs.is_empty(),
        "the JPEG frame after codec switch must decode"
    );
}

/// Submit two garbage payloads pretending to be JPEG, then a valid JPEG.
///
/// Pre-submission payload validation rejects the garbage immediately as
/// `Skipped { reason: InvalidPayload(..) }`.  The decoder never enters
/// `Active` for the garbage, so the valid JPEG starts a clean session and
/// its UUID maps correctly on output.
#[test]
#[serial]
fn test_garbage_jpeg_then_valid_jpeg() {
    init();
    let collector = OutputCollector::new();
    let dec = FlexibleDecoder::new(default_config("cam-1"), collector.callback());

    let garbage = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0xFF, 0x42, 0x13];

    let frame1 = make_frame(
        "cam-1",
        320,
        240,
        Some(VideoCodec::Jpeg),
        VideoFrameContent::None,
        0,
    );
    dec.submit(&frame1, Some(&garbage))
        .expect("first garbage submit must not panic or fail");

    let frame2 = make_frame(
        "cam-1",
        320,
        240,
        Some(VideoCodec::Jpeg),
        VideoFrameContent::None,
        33_333_333,
    );
    dec.submit(&frame2, Some(&garbage))
        .expect("second garbage submit must not panic or fail");

    // Both garbage frames must be rejected synchronously as Skipped.
    let outputs = collector.drain();
    let skipped: Vec<_> = outputs
        .iter()
        .filter_map(|o| match o {
            FlexibleDecoderOutput::Skipped { frame, reason, .. } => Some((frame, reason)),
            _ => None,
        })
        .collect();
    assert_eq!(skipped.len(), 2, "both garbage frames must be Skipped");
    for (_, reason) in &skipped {
        assert!(
            matches!(reason, SkipReason::InvalidPayload(_)),
            "reason must be InvalidPayload, got: {reason:?}"
        );
    }

    // Submit a valid JPEG — decoder activates from a clean Idle state.
    let real_jpeg = make_jpeg(320, 240);
    let frame3 = make_frame(
        "cam-1",
        320,
        240,
        Some(VideoCodec::Jpeg),
        VideoFrameContent::None,
        2 * 33_333_333,
    );
    let valid_uuid = frame3.get_uuid_u128();
    dec.submit(&frame3, Some(&real_jpeg))
        .expect("valid JPEG submit must succeed");

    collector.wait_for_frames(1, Duration::from_secs(10));

    let outputs = collector.drain();
    let frame_uuids: Vec<u128> = outputs
        .iter()
        .filter_map(|o| match o {
            FlexibleDecoderOutput::Frame { frame, decoded, .. } => {
                assert_eq!(
                    Some(frame.get_uuid_u128()),
                    decoded.frame_id,
                    "proxy UUID must match decoded frame_id"
                );
                Some(frame.get_uuid_u128())
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        frame_uuids,
        vec![valid_uuid],
        "only the valid JPEG must produce a Frame output with the correct UUID"
    );
}

/// Submit two garbage payloads pretending to be PNG, then a valid PNG.
///
/// Pre-submission validation rejects the garbage (bad signature) as
/// `Skipped { reason: InvalidPayload(..) }`.  The valid PNG decodes normally.
#[test]
#[serial]
fn test_garbage_png_then_valid_png() {
    init();
    let collector = OutputCollector::new();
    let dec = FlexibleDecoder::new(default_config("cam-1"), collector.callback());

    let garbage = vec![
        0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0xFF, 0x42, 0x13, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
        0x07, 0x08, 0x09, 0x0A, 0x0B,
    ];

    let frame1 = make_frame(
        "cam-1",
        320,
        240,
        Some(VideoCodec::Png),
        VideoFrameContent::None,
        0,
    );
    dec.submit(&frame1, Some(&garbage))
        .expect("first garbage submit must not panic or fail");

    let frame2 = make_frame(
        "cam-1",
        320,
        240,
        Some(VideoCodec::Png),
        VideoFrameContent::None,
        33_333_333,
    );
    dec.submit(&frame2, Some(&garbage))
        .expect("second garbage submit must not panic or fail");

    let outputs = collector.drain();
    let skipped: Vec<_> = outputs
        .iter()
        .filter_map(|o| match o {
            FlexibleDecoderOutput::Skipped { reason, .. } => Some(reason),
            _ => None,
        })
        .collect();
    assert_eq!(skipped.len(), 2, "both garbage frames must be Skipped");
    for reason in &skipped {
        assert!(
            matches!(reason, SkipReason::InvalidPayload(_)),
            "reason must be InvalidPayload, got: {reason:?}"
        );
    }

    // Valid PNG must decode successfully.
    let real_png = make_png(320, 240);
    let frame3 = make_frame(
        "cam-1",
        320,
        240,
        Some(VideoCodec::Png),
        VideoFrameContent::None,
        2 * 33_333_333,
    );
    let valid_uuid = frame3.get_uuid_u128();
    dec.submit(&frame3, Some(&real_png))
        .expect("valid PNG submit must succeed");

    collector.wait_for_frames(1, Duration::from_secs(10));

    let outputs = collector.drain();
    let frame_uuids: Vec<u128> = outputs
        .iter()
        .filter_map(|o| match o {
            FlexibleDecoderOutput::Frame { frame, decoded, .. } => {
                assert_eq!(
                    Some(frame.get_uuid_u128()),
                    decoded.frame_id,
                    "proxy UUID must match decoded frame_id"
                );
                Some(frame.get_uuid_u128())
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        frame_uuids,
        vec![valid_uuid],
        "only the valid PNG must produce a Frame output with the correct UUID"
    );
}

/// Verify `take_delivery()` extracts the sealed `(VideoFrameProxy, SharedBuffer)`
/// pair from a `Frame` output and returns `None` for non-frame variants.
///
/// The seal blocks `unseal()` until the parent `FlexibleDecoderOutput` is dropped.
#[test]
#[serial]
fn test_take_delivery() {
    init();
    let collector = OutputCollector::new();
    let dec = FlexibleDecoder::new(default_config("cam-1"), collector.callback());

    let jpeg_data = make_jpeg(320, 240);
    let frame = make_frame(
        "cam-1",
        320,
        240,
        Some(VideoCodec::Jpeg),
        VideoFrameContent::None,
        0,
    );
    let submitted_uuid = frame.get_uuid_u128();
    dec.submit(&frame, Some(&jpeg_data)).unwrap();

    collector.wait_for_frames(1, Duration::from_secs(10));

    let outputs = collector.drain();
    let mut frame_outputs = Vec::new();
    let mut other_outputs = Vec::new();
    for o in outputs {
        if matches!(o, FlexibleDecoderOutput::Frame { .. }) {
            frame_outputs.push(o);
        } else {
            other_outputs.push(o);
        }
    }

    assert_eq!(frame_outputs.len(), 1, "exactly one Frame expected");

    let mut frame_out = frame_outputs.into_iter().next().unwrap();

    // take_delivery returns a sealed delivery (buffer is taken from decoded).
    let sealed = frame_out
        .take_delivery()
        .expect("take_delivery must return Some for Frame");

    // Seal is not yet released — the output is still alive.
    assert!(!sealed.is_released());

    // Second call returns None — buffer already taken.
    assert!(
        frame_out.take_delivery().is_none(),
        "second take_delivery must return None (buffer already taken)"
    );

    // Drop the output → releases the seal.
    drop(frame_out);
    assert!(sealed.is_released());

    // Unseal and verify.
    let (proxy, buffer) = sealed.unseal();
    assert_eq!(proxy.get_uuid_u128(), submitted_uuid);
    let guard = buffer.lock();
    assert!(guard.as_ref().size() > 0, "SharedBuffer must hold data");
    drop(guard);

    for mut o in other_outputs {
        assert!(
            o.take_delivery().is_none(),
            "take_delivery must return None for non-Frame variants"
        );
    }
}
