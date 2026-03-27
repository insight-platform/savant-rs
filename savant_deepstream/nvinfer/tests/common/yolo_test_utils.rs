//! Helpers for YOLO integration tests (tensor IO, properties, GT matching).
//!
//! This module is only used by `test_yolo_detection`; other integration test
//! crates also `mod common` and would otherwise see unused-item warnings.

#![allow(dead_code)]

use std::collections::HashMap;

use nvinfer::TensorView;
use savant_core::primitives::RBBox;

/// Converts a `TensorView` to `Vec<f32>`, handling fp16 and fp32.
pub fn tensor_to_f32_vec(tv: &TensorView) -> Vec<f32> {
    match tv.data_type {
        nvinfer::DataType::Half => {
            let raw: &[half::f16] = unsafe { tv.as_slice() };
            raw.iter().map(|v| v.to_f32()).collect()
        }
        nvinfer::DataType::Float => {
            let raw: &[f32] = unsafe { tv.as_slice() };
            raw.to_vec()
        }
        other => panic!("unsupported tensor dtype: {other:?}"),
    }
}

/// Gets shape as `Vec<usize>` from a TensorView.
pub fn tensor_shape(tv: &TensorView) -> Vec<usize> {
    tv.dims.dimensions.iter().map(|&d| d as usize).collect()
}

/// nvinfer properties HashMap for yolo11n.
pub fn yolo11n_properties() -> HashMap<String, String> {
    let dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets");
    let engines_dir = super::engines_dir();
    let mut m = HashMap::new();
    m.insert(
        "onnx-file".into(),
        dir.join("yolo11n.onnx").to_string_lossy().into(),
    );
    m.insert(
        "model-engine-file".into(),
        engines_dir
            .join("yolo11n.onnx_b1_gpu0_fp16.engine")
            .to_string_lossy()
            .into(),
    );
    m.insert("gpu-id".into(), "0".into());
    m.insert("network-mode".into(), "2".into()); // fp16
    m.insert("workspace-size".into(), "2048".into());
    m.insert("batch-size".into(), "1".into());
    m.insert("net-scale-factor".into(), "0.003921569790691137".into());
    m.insert("offsets".into(), "0.0;0.0;0.0".into());
    m.insert("output-blob-names".into(), "output0".into());
    // Jetson: force GPU compute for scaling
    if cfg!(target_arch = "aarch64") {
        m.insert("scaling-compute-hw".into(), "1".into());
    }
    m
}

#[derive(Debug, serde::Deserialize)]
pub struct GtDetection {
    pub class_id: usize,
    pub confidence: f32,
    pub xc: f32,
    pub yc: f32,
    pub w: f32,
    pub h: f32,
}

pub fn load_ground_truth() -> HashMap<String, Vec<GtDetection>> {
    let path =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/yolo/ground_truth.json");
    let text = std::fs::read_to_string(&path).expect("read ground_truth.json");
    serde_json::from_str(&text).expect("parse ground_truth.json")
}

/// Matches predicted detections against GT. For each GT detection (above min_gt_conf),
/// finds the best IoU match in predictions with the same class_id.
/// Returns (matched_count, total_gt_count, mismatches_description).
pub fn match_detections(
    predicted: &HashMap<usize, Vec<(f32, RBBox)>>,
    gt: &[GtDetection],
    iou_threshold: f32,
    conf_tolerance: f32,
    min_gt_conf: f32,
) -> (usize, usize, Vec<String>) {
    let _ = conf_tolerance;
    let mut matched = 0usize;
    let mut total = 0usize;
    let mut mismatches = Vec::new();

    for g in gt {
        if g.confidence < min_gt_conf {
            continue;
        }
        total += 1;

        let preds = match predicted.get(&g.class_id) {
            Some(p) => p,
            None => {
                mismatches.push(format!("class {} not found in predictions", g.class_id));
                continue;
            }
        };

        // Find best IoU match
        let mut best_iou = 0.0f32;
        for (_conf, bbox) in preds {
            let iou = axis_aligned_iou(
                g.xc,
                g.yc,
                g.w,
                g.h,
                bbox.get_xc(),
                bbox.get_yc(),
                bbox.get_width(),
                bbox.get_height(),
            );
            if iou > best_iou {
                best_iou = iou;
            }
        }

        if best_iou >= iou_threshold {
            matched += 1;
        } else {
            mismatches.push(format!(
                "class {} at ({:.1},{:.1} {:.1}x{:.1}) conf={:.3}: best IoU={:.3}",
                g.class_id, g.xc, g.yc, g.w, g.h, g.confidence, best_iou
            ));
        }
    }

    (matched, total, mismatches)
}

#[allow(clippy::too_many_arguments)]
fn axis_aligned_iou(
    xc1: f32,
    yc1: f32,
    w1: f32,
    h1: f32,
    xc2: f32,
    yc2: f32,
    w2: f32,
    h2: f32,
) -> f32 {
    let a_x1 = xc1 - w1 * 0.5;
    let a_y1 = yc1 - h1 * 0.5;
    let a_x2 = xc1 + w1 * 0.5;
    let a_y2 = yc1 + h1 * 0.5;
    let b_x1 = xc2 - w2 * 0.5;
    let b_y1 = yc2 - h2 * 0.5;
    let b_x2 = xc2 + w2 * 0.5;
    let b_y2 = yc2 + h2 * 0.5;
    let inter_w = (a_x2.min(b_x2) - a_x1.max(b_x1)).max(0.0);
    let inter_h = (a_y2.min(b_y2) - a_y1.max(b_y1)).max(0.0);
    let inter = inter_w * inter_h;
    let union = w1 * h1 + w2 * h2 - inter;
    if union > 0.0 {
        inter / union
    } else {
        0.0
    }
}
