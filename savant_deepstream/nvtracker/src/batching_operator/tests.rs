use super::*;
use crate::config::TrackingIdResetMode;
use crate::{MiscTrackData, MiscTrackFrame, TrackState, TrackedObject};
use deepstream_buffers::BatchState;
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
};
use std::collections::HashMap as StdHashMap;
use std::time::Duration;

type FramePair = (VideoFrameProxy, deepstream_buffers::SharedBuffer);

fn make_frame(source_id: &str) -> VideoFrameProxy {
    VideoFrameProxy::new(
        source_id,
        (30, 1),
        320,
        240,
        VideoFrameContent::None,
        VideoFrameTranscodingMethod::Copy,
        None,
        None,
        (1, 1_000_000_000),
        0,
        None,
        None,
    )
    .expect("failed to build test frame")
}

fn make_shared_buffer() -> deepstream_buffers::SharedBuffer {
    gstreamer::init().expect("gst init");
    deepstream_buffers::SharedBuffer::from(gstreamer::Buffer::new())
}

fn make_nvtracker_config() -> crate::config::NvTrackerConfig {
    crate::config::NvTrackerConfig {
        name: String::new(),
        tracker_width: crate::config::DEFAULT_TRACKER_WIDTH,
        tracker_height: crate::config::DEFAULT_TRACKER_HEIGHT,
        max_batch_size: crate::config::DEFAULT_MAX_BATCH_SIZE,
        ll_lib_file: "/tmp/libnvds_nvmultiobjecttracker.so".to_string(),
        ll_config_file: "/tmp/config_tracker_IOU.yml".to_string(),
        gpu_id: 0,
        input_format: deepstream_buffers::VideoFormat::RGBA,
        element_properties: StdHashMap::new(),
        tracking_id_reset_mode: TrackingIdResetMode::None,
        meta_clear_policy: deepstream_buffers::MetaClearPolicy::Before,
        operation_timeout: std::time::Duration::from_secs(30),
        input_channel_capacity: 16,
        output_channel_capacity: 16,
        drain_poll_interval: std::time::Duration::from_millis(100),
        idle_flush_interval: None,
        stale_source_after: None,
    }
}

/// Helper to build a test [`TrackerOperatorOutput::Tracking`] with N frames.
fn make_test_output(n: usize) -> output::TrackerOperatorOutput {
    let mut frames = Vec::with_capacity(n);
    let mut deliveries = Vec::with_capacity(n);
    for i in 0..n {
        let src = format!("cam{i}");
        let frame = make_frame(&src);
        deliveries.push((frame.clone(), make_shared_buffer()));
        frames.push(output::TrackerOperatorFrameOutput {
            frame,
            tracked_objects: vec![],
            shadow_tracks: vec![],
            terminated_tracks: vec![],
            past_frame_data: vec![],
        });
    }
    output::TrackerOperatorOutput::Tracking(output::TrackerOperatorTrackingOutput::new(
        frames, deliveries,
    ))
}

// ── BatchState ──────────────────────────────────────────────────────

#[test]
fn batch_state_new_is_empty() {
    let state: BatchState<FramePair> = BatchState::new();
    assert!(state.is_empty());
    assert!(state.deadline.is_none());
    assert!(state.frames.is_empty());
}

#[test]
fn batch_state_take_returns_frames_and_resets() {
    let mut state = BatchState::new();
    state.frames.push((make_frame("s1"), make_shared_buffer()));
    state.frames.push((make_frame("s2"), make_shared_buffer()));
    state.deadline = Some(std::time::Instant::now());

    let taken = state.take();
    assert_eq!(taken.len(), 2);
    assert_eq!(taken[0].0.get_source_id(), "s1");
    assert_eq!(taken[1].0.get_source_id(), "s2");
    assert!(state.is_empty());
    assert!(state.deadline.is_none());
}

// ── NvTrackerBatchingOperatorConfig ────────────────────────────────

#[test]
fn operator_config_builder_defaults() {
    let nvtracker_config = make_nvtracker_config();
    let config = NvTrackerBatchingOperatorConfig::builder(nvtracker_config).build();
    assert_eq!(config.max_batch_size, 1);
    assert_eq!(config.max_batch_wait, Duration::from_millis(50));
}

#[test]
fn operator_config_builder_overrides() {
    let nvtracker_config = make_nvtracker_config();
    let config = NvTrackerBatchingOperatorConfig::builder(nvtracker_config)
        .max_batch_size(16)
        .max_batch_wait(Duration::from_millis(100))
        .build();
    assert_eq!(config.max_batch_size, 16);
    assert_eq!(config.max_batch_wait, Duration::from_millis(100));
}

#[test]
fn operator_config_preserves_nvtracker_meta_clear_policy() {
    let mut nvtracker_config = make_nvtracker_config();
    nvtracker_config.meta_clear_policy = deepstream_buffers::MetaClearPolicy::Both;
    let config = NvTrackerBatchingOperatorConfig::builder(nvtracker_config).build();
    assert_eq!(
        config.nvtracker.meta_clear_policy,
        deepstream_buffers::MetaClearPolicy::Both
    );
}

// ── TrackerBatchFormationResult ────────────────────────────────────

#[test]
fn batch_formation_result_empty() {
    let result = TrackerBatchFormationResult {
        ids: vec![],
        rois: vec![],
    };
    assert!(result.ids.is_empty());
    assert!(result.rois.is_empty());
}

#[test]
fn batch_formation_result_with_entries() {
    let result = TrackerBatchFormationResult {
        ids: vec![deepstream_buffers::SavantIdMetaKind::Frame(1)],
        rois: vec![StdHashMap::new()],
    };
    assert_eq!(result.ids.len(), 1);
    assert_eq!(result.rois.len(), 1);
}

// ── Error variants ──────────────────────────────────────────────────

#[test]
fn error_operator_shutdown_display() {
    let err = crate::error::NvTrackerError::OperatorShutdown;
    let msg = err.to_string();
    assert!(msg.contains("shut down"), "expected 'shut down' in: {msg}");
}

#[test]
fn error_batch_formation_failed_display() {
    let err = crate::error::NvTrackerError::BatchFormationFailed("invalid rois".into());
    let msg = err.to_string();
    assert!(
        msg.contains("invalid rois"),
        "expected 'invalid rois' in: {msg}"
    );
}

// ── TrackerOperatorOutput ───────────────────────────────────────────

#[test]
fn tracker_operator_output_debug_format() {
    let output = make_test_output(0);
    let dbg = format!("{output:?}");
    assert!(dbg.contains("Tracking"));
    assert!(dbg.contains("num_frames"));
}

#[test]
fn tracker_operator_output_accessors() {
    let output = make_test_output(2);
    let tracking = output.as_tracking().expect("tracking variant");
    assert_eq!(tracking.frames().len(), 2);
    assert_eq!(tracking.frames()[0].frame.get_source_id(), "cam0");
}

#[test]
fn frame_output_contains_per_frame_groups() {
    let frame = make_frame("cam-a");
    let frame_output = output::TrackerOperatorFrameOutput {
        frame,
        tracked_objects: vec![TrackedObject {
            object_id: 1,
            class_id: 0,
            bbox_left: 0.0,
            bbox_top: 0.0,
            bbox_width: 10.0,
            bbox_height: 10.0,
            confidence: 0.9,
            tracker_confidence: 0.8,
            label: Some("car".to_string()),
            slot_number: 0,
            source_id: "cam-a".to_string(),
            misc_obj_info: [0; 4],
        }],
        shadow_tracks: vec![MiscTrackData {
            object_id: 1,
            class_id: 0,
            label: Some("car".to_string()),
            source_id: "cam-a".to_string(),
            category: crate::MiscTrackCategory::Shadow,
            frames: vec![MiscTrackFrame {
                frame_num: 1,
                bbox_left: 0.0,
                bbox_top: 0.0,
                bbox_width: 1.0,
                bbox_height: 1.0,
                confidence: 0.5,
                age: 1,
                state: TrackState::Active,
                visibility: 1.0,
            }],
        }],
        terminated_tracks: vec![],
        past_frame_data: vec![],
    };

    assert_eq!(frame_output.tracked_objects.len(), 1);
    assert_eq!(frame_output.shadow_tracks.len(), 1);
}

// ── SealedDeliveries ────────────────────────────────────────────────

#[test]
fn take_deliveries_returns_correct_count() {
    let mut output = make_test_output(3);
    let sealed = output
        .as_tracking_mut()
        .expect("tracking")
        .take_deliveries();
    assert!(sealed.is_some());
    let sealed = sealed.unwrap();
    assert_eq!(sealed.len(), 3);
    assert!(!sealed.is_empty());
}

#[test]
fn take_deliveries_twice_returns_none() {
    let mut output = make_test_output(2);
    let tracking = output.as_tracking_mut().expect("tracking");
    let first = tracking.take_deliveries();
    assert!(first.is_some());
    let second = tracking.take_deliveries();
    assert!(second.is_none());
}

#[test]
fn try_unseal_succeeds_after_drop() {
    let mut output = make_test_output(2);
    let sealed = output
        .as_tracking_mut()
        .expect("tracking")
        .take_deliveries()
        .unwrap();
    drop(output);
    assert!(sealed.is_released());
    let pairs = sealed.try_unseal().expect("should succeed after drop");
    assert_eq!(pairs.len(), 2);
}

#[test]
fn unseal_blocks_then_returns() {
    let mut output = make_test_output(2);
    let sealed = output
        .as_tracking_mut()
        .expect("tracking")
        .take_deliveries()
        .unwrap();

    let handle = std::thread::spawn(move || {
        let pairs = sealed.unseal();
        assert_eq!(pairs.len(), 2);
        pairs
    });

    std::thread::sleep(Duration::from_millis(50));
    drop(output);

    let pairs = handle.join().expect("thread panicked");
    assert_eq!(pairs.len(), 2);
}

// ── New error variant tests ────────────────────────────────────────

#[test]
fn error_pipeline_failed_display() {
    let err = crate::error::NvTrackerError::PipelineFailed;
    let msg = err.to_string();
    assert!(
        msg.contains("failed state"),
        "expected 'failed state' in: {msg}"
    );
}

#[test]
fn error_operator_failed_display() {
    let err = crate::error::NvTrackerError::OperatorFailed;
    let msg = err.to_string();
    assert!(
        msg.contains("failed state"),
        "expected 'failed state' in: {msg}"
    );
}

// ── Config defaults for new fields ─────────────────────────────────

#[test]
fn operator_config_builder_pending_batch_timeout_default() {
    let config = NvTrackerBatchingOperatorConfig::builder(make_nvtracker_config()).build();
    assert_eq!(config.pending_batch_timeout, Duration::from_secs(60));
}

#[test]
fn operator_config_builder_pending_batch_timeout_override() {
    let config = NvTrackerBatchingOperatorConfig::builder(make_nvtracker_config())
        .pending_batch_timeout(Duration::from_secs(120))
        .build();
    assert_eq!(config.pending_batch_timeout, Duration::from_secs(120));
}

#[test]
fn nvtracker_config_operation_timeout_default() {
    let cfg = crate::config::NvTrackerConfig::new("/tmp/lib.so", "/tmp/cfg.yml");
    assert_eq!(cfg.operation_timeout, Duration::from_secs(30));
}
