use super::*;
use crate::meta_clear_policy::MetaClearPolicy;
use crate::model_input_scaling::ModelInputScaling;
use crate::roi::{Roi, RoiKind};
use deepstream_buffers::SavantIdMetaKind;
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
};
use savant_core::primitives::RBBox;
use std::collections::HashMap as StdHashMap;
use std::time::Duration;

fn make_frame(source_id: &str) -> VideoFrameProxy {
    VideoFrameProxy::new(
        source_id,
        "30/1",
        320,
        240,
        VideoFrameContent::None,
        VideoFrameTranscodingMethod::Copy,
        &None,
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

fn make_nvinfer_config() -> crate::config::NvInferConfig {
    crate::config::NvInferConfig {
        name: String::new(),
        nvinfer_properties: StdHashMap::new(),
        element_properties: StdHashMap::new(),
        gpu_id: 0,
        queue_depth: 0,
        input_format: deepstream_buffers::VideoFormat::RGBA,
        model_width: 640,
        model_height: 640,
        model_color_format: crate::model_color_format::ModelColorFormat::RGB,
        meta_clear_policy: MetaClearPolicy::Before,
        disable_output_host_copy: false,
        scaling: ModelInputScaling::Fill,
    }
}

/// Helper to build a test `OperatorInferenceOutput` with N frames.
fn make_test_output(n: usize) -> output::OperatorInferenceOutput {
    let mut frames = Vec::with_capacity(n);
    let mut deliveries = Vec::with_capacity(n);
    for i in 0..n {
        let src = format!("cam{i}");
        let frame = make_frame(&src);
        deliveries.push((frame.clone(), make_shared_buffer()));
        frames.push(output::OperatorFrameOutput {
            frame,
            elements: vec![],
        });
    }
    output::OperatorInferenceOutput::new(frames, deliveries, true, make_shared_buffer())
}

// ── BatchState ──────────────────────────────────────────────────────

#[test]
fn batch_state_new_is_empty() {
    let state = state::BatchState::new();
    assert!(state.is_empty());
    assert!(state.deadline.is_none());
    assert!(state.frames.is_empty());
}

#[test]
fn batch_state_deadline_roundtrip() {
    let mut state = state::BatchState::new();
    assert!(state.deadline.is_none());

    let dl = std::time::Instant::now() + Duration::from_millis(500);
    state.deadline = Some(dl);
    assert_eq!(state.deadline, Some(dl));
}

#[test]
fn batch_state_take_on_empty_returns_empty() {
    let mut state = state::BatchState::new();
    state.deadline = Some(std::time::Instant::now() + Duration::from_secs(10));

    let taken = state.take();
    assert!(taken.is_empty(), "no frames were pushed");
    assert!(state.is_empty());
    assert!(state.deadline.is_none());
}

#[test]
fn batch_state_is_empty_reflects_frames() {
    let mut state = state::BatchState::new();
    assert!(state.is_empty());

    state.frames.push((make_frame("src"), make_shared_buffer()));
    assert!(!state.is_empty());
}

#[test]
fn batch_state_take_returns_frames_and_resets() {
    let mut state = state::BatchState::new();
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

#[test]
fn batch_state_take_is_idempotent() {
    let mut state = state::BatchState::new();
    state.frames.push((make_frame("x"), make_shared_buffer()));
    state.deadline = Some(std::time::Instant::now());

    let first = state.take();
    assert_eq!(first.len(), 1);

    let second = state.take();
    assert!(second.is_empty());
    assert!(state.is_empty());
}

// ── NvInferBatchingOperatorConfig ────────────────────────────────────────

#[test]
fn operator_config_fields() {
    let nvinfer = make_nvinfer_config();
    let config = NvInferBatchingOperatorConfig {
        max_batch_size: 8,
        max_batch_wait: Duration::from_millis(100),
        nvinfer,
    };
    assert_eq!(config.max_batch_size, 8);
    assert_eq!(config.max_batch_wait, Duration::from_millis(100));
    assert_eq!(config.nvinfer.gpu_id, 0);
}

#[test]
fn operator_config_clone() {
    let mut nvinfer = make_nvinfer_config();
    nvinfer.gpu_id = 1;
    let config = NvInferBatchingOperatorConfig {
        max_batch_size: 4,
        max_batch_wait: Duration::from_millis(50),
        nvinfer,
    };
    let cloned = config.clone();
    assert_eq!(cloned.max_batch_size, config.max_batch_size);
    assert_eq!(cloned.max_batch_wait, config.max_batch_wait);
    assert_eq!(cloned.nvinfer.gpu_id, config.nvinfer.gpu_id);
}

#[test]
fn operator_config_debug_format() {
    let config = NvInferBatchingOperatorConfig {
        max_batch_size: 2,
        max_batch_wait: Duration::from_millis(10),
        nvinfer: make_nvinfer_config(),
    };
    let dbg = format!("{config:?}");
    assert!(dbg.contains("max_batch_size"));
    assert!(dbg.contains("max_batch_wait"));
    assert!(dbg.contains("nvinfer"));
}

// ── BatchFormationResult ────────────────────────────────────────────

#[test]
fn batch_formation_result_empty() {
    let result = BatchFormationResult {
        ids: vec![],
        rois: vec![],
    };
    assert!(result.ids.is_empty());
    assert!(result.rois.is_empty());
}

#[test]
fn batch_formation_result_with_entries() {
    let roi = Roi {
        id: 42,
        bbox: RBBox::new(100.0, 200.0, 50.0, 60.0, None),
    };
    let result = BatchFormationResult {
        ids: vec![SavantIdMetaKind::Frame(1), SavantIdMetaKind::Frame(2)],
        rois: vec![RoiKind::Rois(vec![roi.clone()]), RoiKind::FullFrame],
    };
    assert_eq!(result.ids.len(), 2);
    assert_eq!(result.rois.len(), 2);
    match &result.rois[0] {
        RoiKind::Rois(v) => {
            assert_eq!(v.len(), 1);
            assert_eq!(v[0].id, 42);
        }
        RoiKind::FullFrame => panic!("Expected Rois variant"),
    }
    assert!(matches!(result.rois[1], RoiKind::FullFrame));
}

// ── Error variants ──────────────────────────────────────────────────

#[test]
fn error_operator_shutdown_display() {
    let err = crate::error::NvInferError::OperatorShutdown;
    let msg = err.to_string();
    assert!(msg.contains("shut down"), "expected 'shut down' in: {msg}");
}

#[test]
fn error_batch_formation_failed_display() {
    let err = crate::error::NvInferError::BatchFormationFailed("buffer alloc".into());
    let msg = err.to_string();
    assert!(
        msg.contains("buffer alloc"),
        "expected 'buffer alloc' in: {msg}"
    );
}

#[test]
fn error_pipeline_error_display() {
    let err = crate::error::NvInferError::PipelineError("init failed".into());
    let msg = err.to_string();
    assert!(
        msg.contains("init failed"),
        "expected 'init failed' in: {msg}"
    );
}

// ── OperatorInferenceOutput ──────────────────────────────────────────

#[test]
fn operator_inference_output_debug_format() {
    let output = make_test_output(0);
    let dbg = format!("{output:?}");
    assert!(dbg.contains("OperatorInferenceOutput"));
    assert!(dbg.contains("num_frames"));
    assert!(dbg.contains("host_copy_enabled"));
}

#[test]
fn operator_inference_output_accessors() {
    let output = make_test_output(0);
    assert!(output.frames().is_empty());
    assert!(output.host_copy_enabled());
}

#[test]
fn operator_inference_output_with_frames() {
    let output = make_test_output(2);
    assert_eq!(output.frames().len(), 2);
    assert!(output.host_copy_enabled());
    assert_eq!(output.frames()[0].frame.get_source_id(), "cam0");
    assert_eq!(output.frames()[1].frame.get_source_id(), "cam1");
}

// ── VideoFrameProxy helper ──────────────────────────────────────────

#[test]
fn make_frame_returns_correct_source_id() {
    let frame = make_frame("test_source");
    assert_eq!(frame.get_source_id(), "test_source");
}

#[test]
fn make_frame_different_sources_are_distinct() {
    let f1 = make_frame("cam1");
    let f2 = make_frame("cam2");
    assert_ne!(f1.get_source_id(), f2.get_source_id());
}

// ── OperatorElement ────────────────────────────────────────────────────

use crate::output::ElementOutput;
use output::OperatorElement;

fn make_element_output(roi_id: Option<i64>, slot: u32) -> ElementOutput {
    ElementOutput {
        roi_id,
        slot_number: slot,
        tensors: vec![],
    }
}

#[test]
fn operator_element_deref_works() {
    let elem = OperatorElement::new(
        make_element_output(Some(42), 0),
        0.0,
        0.0,
        100.0,
        100.0,
        100.0,
        100.0,
        ModelInputScaling::Fill,
    );
    assert_eq!(elem.roi_id, Some(42));
    assert_eq!(elem.slot_number, 0);
    assert!(elem.tensors.is_empty());
}

#[test]
fn operator_element_lazy_scaler_init() {
    let elem = OperatorElement::new(
        make_element_output(None, 0),
        10.0,
        20.0,
        200.0,
        400.0,
        100.0,
        200.0,
        ModelInputScaling::Fill,
    );
    let s1 = elem.coordinate_scaler();
    let s2 = elem.coordinate_scaler();

    let (x1, y1) = s1.scale_point(50.0, 100.0);
    let (x2, y2) = s2.scale_point(50.0, 100.0);
    assert!((x1 - x2).abs() < 1e-6);
    assert!((y1 - y2).abs() < 1e-6);
}

#[test]
fn operator_element_scale_points_fill() {
    // ROI at (10, 20), 200x400, model 100x200 → scale_x=2, scale_y=2
    let elem = OperatorElement::new(
        make_element_output(Some(1), 0),
        10.0,
        20.0,
        200.0,
        400.0,
        100.0,
        200.0,
        ModelInputScaling::Fill,
    );
    let result = elem.scale_points(&[(0.0, 0.0), (50.0, 100.0), (100.0, 200.0)]);
    assert_eq!(result.len(), 3);
    let eps = 1e-4;
    assert!((result[0].0 - 10.0).abs() < eps);
    assert!((result[0].1 - 20.0).abs() < eps);
    assert!((result[1].0 - 110.0).abs() < eps);
    assert!((result[1].1 - 220.0).abs() < eps);
    assert!((result[2].0 - 210.0).abs() < eps);
    assert!((result[2].1 - 420.0).abs() < eps);
}

#[test]
fn operator_element_scale_ltwh_and_ltrb_consistency() {
    let elem = OperatorElement::new(
        make_element_output(None, 0),
        10.0,
        20.0,
        300.0,
        150.0,
        64.0,
        64.0,
        ModelInputScaling::Fill,
    );
    let ltwh = elem.scale_ltwh(&[[5.0, 10.0, 20.0, 30.0]]);
    let ltrb = elem.scale_ltrb(&[[5.0, 10.0, 25.0, 40.0]]);
    let eps = 1e-4;
    assert!((ltwh[0][0] - ltrb[0][0]).abs() < eps, "left");
    assert!((ltwh[0][1] - ltrb[0][1]).abs() < eps, "top");
    assert!((ltwh[0][0] + ltwh[0][2] - ltrb[0][2]).abs() < eps, "right");
    assert!((ltwh[0][1] + ltwh[0][3] - ltrb[0][3]).abs() < eps, "bottom");
}

#[test]
fn operator_element_scale_rbboxes() {
    let elem = OperatorElement::new(
        make_element_output(Some(5), 1),
        0.0,
        0.0,
        200.0,
        200.0,
        100.0,
        100.0,
        ModelInputScaling::KeepAspectRatio,
    );
    let boxes = vec![RBBox::new(50.0, 50.0, 20.0, 30.0, None)];
    let result = elem.scale_rbboxes(&boxes);
    assert_eq!(result.len(), 1);
    let eps = 1e-4;
    assert!((result[0].get_xc() - 100.0).abs() < eps, "xc");
    assert!((result[0].get_yc() - 100.0).abs() < eps, "yc");
    assert!((result[0].get_width() - 40.0).abs() < eps, "w");
    assert!((result[0].get_height() - 60.0).abs() < eps, "h");
    assert_eq!(result[0].get_angle(), None);
}

#[test]
fn operator_element_debug_format() {
    let elem = OperatorElement::new(
        make_element_output(Some(7), 3),
        0.0,
        0.0,
        100.0,
        100.0,
        100.0,
        100.0,
        ModelInputScaling::Fill,
    );
    let dbg = format!("{elem:?}");
    assert!(dbg.contains("OperatorElement"));
    assert!(dbg.contains("roi_id"));
    assert!(dbg.contains("scaler_initialized"));
}

// ── SealedDeliveries ────────────────────────────────────────────────

#[test]
fn take_deliveries_returns_correct_count() {
    let mut output = make_test_output(3);
    let sealed = output.take_deliveries();
    assert!(sealed.is_some());
    let sealed = sealed.unwrap();
    assert_eq!(sealed.len(), 3);
    assert!(!sealed.is_empty());
}

#[test]
fn take_deliveries_twice_returns_none() {
    let mut output = make_test_output(2);
    let first = output.take_deliveries();
    assert!(first.is_some());
    let second = output.take_deliveries();
    assert!(second.is_none());
}

#[test]
fn sealed_deliveries_is_empty_on_zero_frames() {
    let mut output = make_test_output(0);
    let sealed = output.take_deliveries().unwrap();
    assert!(sealed.is_empty());
    assert_eq!(sealed.len(), 0);
}

#[test]
fn try_unseal_fails_while_sealed() {
    let mut output = make_test_output(1);
    let sealed = output.take_deliveries().unwrap();
    assert!(!sealed.is_released());
    let result = sealed.try_unseal();
    assert!(result.is_err());
    let _sealed_back = result.unwrap_err();
    drop(output);
}

#[test]
fn try_unseal_succeeds_after_drop() {
    let mut output = make_test_output(2);
    let sealed = output.take_deliveries().unwrap();
    drop(output);
    assert!(sealed.is_released());
    let pairs = sealed.try_unseal().expect("should succeed after drop");
    assert_eq!(pairs.len(), 2);
}

#[test]
fn unseal_blocks_then_returns() {
    let mut output = make_test_output(2);
    let sealed = output.take_deliveries().unwrap();

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

#[test]
fn is_released_reflects_state() {
    let mut output = make_test_output(1);
    let sealed = output.take_deliveries().unwrap();
    assert!(!sealed.is_released());
    drop(output);
    assert!(sealed.is_released());
}

#[test]
fn sealed_deliveries_debug_format() {
    let mut output = make_test_output(2);
    let sealed = output.take_deliveries().unwrap();
    let dbg = format!("{sealed:?}");
    assert!(dbg.contains("SealedDeliveries"));
    assert!(dbg.contains("len"));
    assert!(dbg.contains("released"));
    drop(output);
}

// ── Early-drop safety tests ─────────────────────────────────────────

#[test]
fn drop_sealed_without_unseal() {
    let mut output = make_test_output(2);
    let sealed = output.take_deliveries().unwrap();
    drop(sealed);
    drop(output);
}

#[test]
fn drop_sealed_before_output() {
    let mut output = make_test_output(2);
    let sealed = output.take_deliveries().unwrap();
    drop(sealed);
    drop(output);
}

#[test]
fn drop_output_before_sealed() {
    let mut output = make_test_output(2);
    let sealed = output.take_deliveries().unwrap();
    drop(output);
    assert!(sealed.is_released());
    drop(sealed);
}

#[test]
fn never_call_take_deliveries() {
    let output = make_test_output(3);
    drop(output);
}

#[test]
fn unseal_after_output_already_dropped_returns_immediately() {
    let mut output = make_test_output(1);
    let sealed = output.take_deliveries().unwrap();
    drop(output);
    let pairs = sealed.unseal();
    assert_eq!(pairs.len(), 1);
}
