//! Greedy non-maximum suppression for axis-aligned boxes in `(xc, yc, w, h)` form.

use anyhow::{ensure, Result};
use std::collections::HashMap;

/// Computes intersection-over-union for two boxes given as `[xc, yc, w, h]`.
pub fn iou_xcycwh(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let a_x1 = a[0] - a[2] * 0.5;
    let a_y1 = a[1] - a[3] * 0.5;
    let a_x2 = a[0] + a[2] * 0.5;
    let a_y2 = a[1] + a[3] * 0.5;

    let b_x1 = b[0] - b[2] * 0.5;
    let b_y1 = b[1] - b[3] * 0.5;
    let b_x2 = b[0] + b[2] * 0.5;
    let b_y2 = b[1] + b[3] * 0.5;

    let inter_w = (a_x2.min(b_x2) - a_x1.max(b_x1)).max(0.0);
    let inter_h = (a_y2.min(b_y2) - a_y1.max(b_y1)).max(0.0);
    let inter = inter_w * inter_h;
    let union = a[2] * a[3] + b[2] * b[3] - inter;
    if union > 0.0 {
        inter / union
    } else {
        0.0
    }
}

/// Greedy NMS over all classes: lower-confidence boxes overlapping a kept box are suppressed.
pub fn nms_class_agnostic(
    boxes: &[[f32; 4]],
    confidences: &[f32],
    iou_threshold: f32,
    top_k: usize,
) -> Result<Vec<usize>> {
    let n = boxes.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    ensure!(
        confidences.len() == n,
        "nms_class_agnostic: confidences.len() ({}) != boxes.len() ({})",
        confidences.len(),
        n,
    );

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        confidences[j]
            .partial_cmp(&confidences[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut suppressed = vec![false; n];
    let mut kept = Vec::new();

    for pos in 0..order.len() {
        let i = order[pos];
        if suppressed[i] {
            continue;
        }
        kept.push(i);
        if kept.len() >= top_k {
            break;
        }
        for &j in order.iter().skip(pos + 1) {
            if suppressed[j] {
                continue;
            }
            if iou_xcycwh(&boxes[i], &boxes[j]) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    Ok(kept)
}

/// Greedy NMS applied independently per class; results merged and truncated to `top_k`.
pub fn nms_class_aware(
    boxes: &[[f32; 4]],
    confidences: &[f32],
    class_ids: &[usize],
    iou_threshold: f32,
    top_k: usize,
) -> Result<Vec<usize>> {
    let n = boxes.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    ensure!(
        confidences.len() == n,
        "nms_class_aware: confidences.len() ({}) != boxes.len() ({})",
        confidences.len(),
        n,
    );
    ensure!(
        class_ids.len() == n,
        "nms_class_aware: class_ids.len() ({}) != boxes.len() ({})",
        class_ids.len(),
        n,
    );

    let mut by_class: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &cid) in class_ids.iter().enumerate() {
        by_class.entry(cid).or_default().push(i);
    }

    let mut merged: Vec<usize> = Vec::new();
    for indices in by_class.values() {
        let m = indices.len();
        let mut sub_boxes = Vec::with_capacity(m);
        let mut sub_conf = Vec::with_capacity(m);
        for &idx in indices {
            sub_boxes.push(boxes[idx]);
            sub_conf.push(confidences[idx]);
        }
        let kept_local = nms_class_agnostic(&sub_boxes, &sub_conf, iou_threshold, usize::MAX)?;
        for k in kept_local {
            merged.push(indices[k]);
        }
    }

    merged.sort_by(|&i, &j| {
        confidences[j]
            .partial_cmp(&confidences[i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    merged.truncate(top_k);
    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input() {
        let boxes: [[f32; 4]; 0] = [];
        let conf: [f32; 0] = [];
        assert!(nms_class_agnostic(&boxes, &conf, 0.5, 10)
            .unwrap()
            .is_empty());
        let cls: [usize; 0] = [];
        assert!(nms_class_aware(&boxes, &conf, &cls, 0.5, 10)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn single_box() {
        let boxes = [[10.0_f32, 10.0, 4.0, 4.0]];
        let conf = [0.9_f32];
        let kept = nms_class_agnostic(&boxes, &conf, 0.5, 10).unwrap();
        assert_eq!(kept, vec![0]);
    }

    #[test]
    fn no_overlap() {
        let boxes = [
            [0.0_f32, 0.0, 2.0, 2.0],
            [100.0, 100.0, 2.0, 2.0],
            [200.0, 200.0, 2.0, 2.0],
        ];
        let conf = [0.9_f32, 0.8, 0.7];
        let kept = nms_class_agnostic(&boxes, &conf, 0.5, 10).unwrap();
        assert_eq!(kept.len(), 3);
    }

    #[test]
    fn full_overlap() {
        let boxes = [[10.0_f32, 10.0, 4.0, 4.0], [10.0, 10.0, 4.0, 4.0]];
        let conf = [0.9_f32, 0.7];
        let kept = nms_class_agnostic(&boxes, &conf, 0.5, 10).unwrap();
        assert_eq!(kept, vec![0]);
    }

    #[test]
    fn partial_overlap_above_threshold() {
        let boxes = [[10.0_f32, 10.0, 4.0, 4.0], [11.0, 10.0, 4.0, 4.0]];
        let conf = [0.9_f32, 0.8];
        let iou = iou_xcycwh(&boxes[0], &boxes[1]);
        assert!(iou > 0.3);
        let kept = nms_class_agnostic(&boxes, &conf, 0.3, 10).unwrap();
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn partial_overlap_below_threshold() {
        let boxes = [[0.0_f32, 0.0, 2.0, 2.0], [20.0, 20.0, 2.0, 2.0]];
        let conf = [0.9_f32, 0.8];
        let iou = iou_xcycwh(&boxes[0], &boxes[1]);
        assert!(iou < 0.1);
        let kept = nms_class_agnostic(&boxes, &conf, 0.1, 10).unwrap();
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn top_k_limiting() {
        let boxes = [
            [0.0_f32, 0.0, 2.0, 2.0],
            [50.0, 0.0, 2.0, 2.0],
            [100.0, 0.0, 2.0, 2.0],
            [150.0, 0.0, 2.0, 2.0],
            [200.0, 0.0, 2.0, 2.0],
        ];
        let conf = [0.5_f32, 0.6, 0.7, 0.8, 0.9];
        let kept = nms_class_agnostic(&boxes, &conf, 0.1, 3).unwrap();
        assert_eq!(kept.len(), 3);
        assert_eq!(kept, vec![4, 3, 2]);
    }

    #[test]
    fn class_aware_no_cross_suppression() {
        let boxes = [[10.0_f32, 10.0, 4.0, 4.0], [11.0, 10.0, 4.0, 4.0]];
        let conf = [0.9_f32, 0.8];
        let class_ids = [0_usize, 1];
        let kept_aware = nms_class_aware(&boxes, &conf, &class_ids, 0.3, 10).unwrap();
        assert_eq!(kept_aware.len(), 2);

        let kept_agnostic = nms_class_agnostic(&boxes, &conf, 0.3, 10).unwrap();
        assert_eq!(kept_agnostic.len(), 1);
    }

    #[test]
    fn length_mismatch_agnostic() {
        let boxes = [[10.0_f32, 10.0, 4.0, 4.0]];
        let conf = [0.9_f32, 0.8]; // 2 vs 1 box
        assert!(nms_class_agnostic(&boxes, &conf, 0.5, 10).is_err());
    }

    #[test]
    fn length_mismatch_aware() {
        let boxes = [[10.0_f32, 10.0, 4.0, 4.0]];
        let conf = [0.9_f32];
        let cls = [0_usize, 1]; // 2 vs 1 box
        assert!(nms_class_aware(&boxes, &conf, &cls, 0.5, 10).is_err());
    }

    #[test]
    fn iou_xcycwh_known_pairs() {
        let a = [0.0_f32, 0.0, 2.0, 2.0];
        let b = [0.0_f32, 0.0, 2.0, 2.0];
        assert!((iou_xcycwh(&a, &b) - 1.0).abs() < 1e-5);

        let c = [0.0_f32, 0.0, 2.0, 2.0];
        let d = [100.0_f32, 100.0, 2.0, 2.0];
        assert!(iou_xcycwh(&c, &d) < 1e-5);
    }
}
