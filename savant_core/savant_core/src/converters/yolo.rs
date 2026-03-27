//! YOLO tensor decoding into [`RBBox`](crate::primitives::RBBox) detections.

use std::collections::{HashMap, HashSet};

use crate::primitives::RBBox;

use super::nms::{nms_class_agnostic, nms_class_aware};
use super::{ConverterError, NmsKind, YoloFormat};

struct RawDetection {
    xc: f32,
    yc: f32,
    w: f32,
    h: f32,
    confidence: f32,
    class_id: usize,
}

/// Decodes raw YOLO inference tensors into per-class bounding boxes.
pub struct YoloDetectionConverter {
    /// Tensor layout and variant-specific parameters.
    pub format: YoloFormat,
    /// Minimum confidence when no per-class override applies.
    pub confidence_threshold: f32,
    /// Per-class confidence thresholds; overrides `confidence_threshold` when present.
    pub per_class_conf_threshold: HashMap<usize, f32>,
    /// When set, only detections whose class id is in this set are kept.
    pub class_filter: Option<HashSet<usize>>,
    /// Non-maximum suppression strategy applied after filtering.
    pub nms: NmsKind,
}

impl YoloDetectionConverter {
    /// Creates a converter with the given decoding and filtering configuration.
    pub fn new(
        format: YoloFormat,
        confidence_threshold: f32,
        per_class_conf_threshold: HashMap<usize, f32>,
        class_filter: Option<HashSet<usize>>,
        nms: NmsKind,
    ) -> Self {
        Self {
            format,
            confidence_threshold,
            per_class_conf_threshold,
            class_filter,
            nms,
        }
    }

    fn effective_threshold(&self, class_id: usize) -> f32 {
        self.per_class_conf_threshold
            .get(&class_id)
            .copied()
            .unwrap_or(self.confidence_threshold)
    }

    fn ensure_nonempty(tensors: &[(&[f32], &[usize])]) -> Result<(), ConverterError> {
        for (data, _) in tensors {
            if data.is_empty() {
                return Err(ConverterError::EmptyTensor);
            }
        }
        Ok(())
    }

    fn parse_v8(
        data: &[f32],
        shape: &[usize],
        num_classes: usize,
    ) -> Result<Vec<RawDetection>, ConverterError> {
        let rows = num_classes + 4;
        if shape.len() != 2 || shape[0] != rows {
            return Err(ConverterError::ShapeMismatch {
                tensor_index: 0,
                expected: format!("[{}, N]", rows),
                got: shape.to_vec(),
            });
        }
        let n = shape[1];
        if data.len() != rows * n {
            return Err(ConverterError::ShapeMismatch {
                tensor_index: 0,
                expected: format!("flat length {} * N", rows),
                got: shape.to_vec(),
            });
        }
        let mut out = Vec::with_capacity(n);
        for j in 0..n {
            let xc = data[j];
            let yc = data[n + j];
            let w = data[2 * n + j];
            let h = data[3 * n + j];
            let mut best_score = f32::NEG_INFINITY;
            let mut best_class = 0usize;
            for c in 0..num_classes {
                let s = data[(4 + c) * n + j];
                if s > best_score {
                    best_score = s;
                    best_class = c;
                }
            }
            out.push(RawDetection {
                xc,
                yc,
                w,
                h,
                confidence: best_score,
                class_id: best_class,
            });
        }
        Ok(out)
    }

    fn parse_v5(
        data: &[f32],
        shape: &[usize],
        num_classes: usize,
    ) -> Result<Vec<RawDetection>, ConverterError> {
        let stride = num_classes + 5;
        if shape.len() != 2 || shape[1] != stride {
            return Err(ConverterError::ShapeMismatch {
                tensor_index: 0,
                expected: format!("[N, {}]", stride),
                got: shape.to_vec(),
            });
        }
        let n = shape[0];
        if data.len() != n * stride {
            return Err(ConverterError::ShapeMismatch {
                tensor_index: 0,
                expected: format!("flat length N * {}", stride),
                got: shape.to_vec(),
            });
        }
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let base = i * stride;
            let xc = data[base];
            let yc = data[base + 1];
            let w = data[base + 2];
            let h = data[base + 3];
            let obj_conf = data[base + 4];
            let mut best_score = f32::NEG_INFINITY;
            let mut best_class = 0usize;
            for c in 0..num_classes {
                let s = data[base + 5 + c];
                if s > best_score {
                    best_score = s;
                    best_class = c;
                }
            }
            out.push(RawDetection {
                xc,
                yc,
                w,
                h,
                confidence: obj_conf * best_score,
                class_id: best_class,
            });
        }
        Ok(out)
    }

    fn parse_v4(
        data0: &[f32],
        shape0: &[usize],
        data1: &[f32],
        shape1: &[usize],
        model_width: f32,
        model_height: f32,
    ) -> Result<Vec<RawDetection>, ConverterError> {
        let n = match shape0.len() {
            2 if shape0[1] == 4 => shape0[0],
            3 if shape0[1] == 1 && shape0[2] == 4 => shape0[0],
            _ => {
                return Err(ConverterError::ShapeMismatch {
                    tensor_index: 0,
                    expected: "[N, 4] or [N, 1, 4]".to_string(),
                    got: shape0.to_vec(),
                });
            }
        };
        if data0.len() != n * 4 {
            return Err(ConverterError::ShapeMismatch {
                tensor_index: 0,
                expected: "N * 4 elements".to_string(),
                got: shape0.to_vec(),
            });
        }
        if shape1.len() != 2 || shape1[0] != n {
            return Err(ConverterError::ShapeMismatch {
                tensor_index: 1,
                expected: "[N, K] with matching N".to_string(),
                got: shape1.to_vec(),
            });
        }
        let k = shape1[1];
        if data1.len() != n * k {
            return Err(ConverterError::ShapeMismatch {
                tensor_index: 1,
                expected: "N * K elements".to_string(),
                got: shape1.to_vec(),
            });
        }
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let left = data0[i * 4] * model_width;
            let top = data0[i * 4 + 1] * model_height;
            let right = data0[i * 4 + 2] * model_width;
            let bottom = data0[i * 4 + 3] * model_height;
            let w = right - left;
            let h = bottom - top;
            let xc = left + w * 0.5;
            let yc = top + h * 0.5;
            let mut best_score = f32::NEG_INFINITY;
            let mut best_class = 0usize;
            for c in 0..k {
                let s = data1[i * k + c];
                if s > best_score {
                    best_score = s;
                    best_class = c;
                }
            }
            out.push(RawDetection {
                xc,
                yc,
                w,
                h,
                confidence: best_score,
                class_id: best_class,
            });
        }
        Ok(out)
    }

    fn parse_v3_raw(
        data0: &[f32],
        shape0: &[usize],
        data1: &[f32],
        shape1: &[usize],
        data2: &[f32],
        shape2: &[usize],
    ) -> Result<Vec<RawDetection>, ConverterError> {
        if shape0.len() != 2 || shape0[1] != 4 {
            return Err(ConverterError::ShapeMismatch {
                tensor_index: 0,
                expected: "[N, 4]".to_string(),
                got: shape0.to_vec(),
            });
        }
        let n = shape0[0];
        if data0.len() != n * 4 {
            return Err(ConverterError::ShapeMismatch {
                tensor_index: 0,
                expected: "N * 4 elements".to_string(),
                got: shape0.to_vec(),
            });
        }
        if shape1.len() != 2 || shape1[0] != n {
            return Err(ConverterError::ShapeMismatch {
                tensor_index: 1,
                expected: "[N, K]".to_string(),
                got: shape1.to_vec(),
            });
        }
        let k = shape1[1];
        if data1.len() != n * k {
            return Err(ConverterError::ShapeMismatch {
                tensor_index: 1,
                expected: "N * K elements".to_string(),
                got: shape1.to_vec(),
            });
        }
        if shape2.len() != 1 || shape2[0] != n {
            return Err(ConverterError::ShapeMismatch {
                tensor_index: 2,
                expected: "[N]".to_string(),
                got: shape2.to_vec(),
            });
        }
        if data2.len() != n {
            return Err(ConverterError::ShapeMismatch {
                tensor_index: 2,
                expected: "N elements".to_string(),
                got: shape2.to_vec(),
            });
        }
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let xc = data0[i * 4];
            let yc = data0[i * 4 + 1];
            let w = data0[i * 4 + 2];
            let h = data0[i * 4 + 3];
            let mut best_score = f32::NEG_INFINITY;
            for c in 0..k {
                let s = data1[i * k + c];
                if s > best_score {
                    best_score = s;
                }
            }
            let class_id = data2[i] as usize;
            out.push(RawDetection {
                xc,
                yc,
                w,
                h,
                confidence: best_score,
                class_id,
            });
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn parse_v4_post_nms(
        data0: &[f32],
        shape0: &[usize],
        data1: &[f32],
        shape1: &[usize],
        data2: &[f32],
        shape2: &[usize],
        data3: &[f32],
        shape3: &[usize],
        model_width: f32,
        model_height: f32,
    ) -> Result<Vec<RawDetection>, ConverterError> {
        if shape0 != [1] || data0.len() != 1 {
            return Err(ConverterError::ShapeMismatch {
                tensor_index: 0,
                expected: "[1]".to_string(),
                got: shape0.to_vec(),
            });
        }
        let num_dets = data0[0] as usize;
        if shape1.len() != 2 || shape1[1] != 4 {
            return Err(ConverterError::ShapeMismatch {
                tensor_index: 1,
                expected: "[N, 4]".to_string(),
                got: shape1.to_vec(),
            });
        }
        let n_boxes = shape1[0];
        if data1.len() != n_boxes * 4 {
            return Err(ConverterError::ShapeMismatch {
                tensor_index: 1,
                expected: "N * 4 elements".to_string(),
                got: shape1.to_vec(),
            });
        }
        if shape2.len() != 1 || shape2[0] != n_boxes {
            return Err(ConverterError::ShapeMismatch {
                tensor_index: 2,
                expected: "[N] matching boxes".to_string(),
                got: shape2.to_vec(),
            });
        }
        if data2.len() != n_boxes {
            return Err(ConverterError::ShapeMismatch {
                tensor_index: 2,
                expected: "N elements".to_string(),
                got: shape2.to_vec(),
            });
        }
        if shape3.len() != 1 || shape3[0] != n_boxes {
            return Err(ConverterError::ShapeMismatch {
                tensor_index: 3,
                expected: "[N] matching boxes".to_string(),
                got: shape3.to_vec(),
            });
        }
        if data3.len() != n_boxes {
            return Err(ConverterError::ShapeMismatch {
                tensor_index: 3,
                expected: "N elements".to_string(),
                got: shape3.to_vec(),
            });
        }
        let n = num_dets.min(data2.len()).min(n_boxes);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let left = data1[i * 4] * model_width;
            let top = data1[i * 4 + 1] * model_height;
            let right = data1[i * 4 + 2] * model_width;
            let bottom = data1[i * 4 + 3] * model_height;
            let w = right - left;
            let h = bottom - top;
            let xc = left + w * 0.5;
            let yc = top + h * 0.5;
            out.push(RawDetection {
                xc,
                yc,
                w,
                h,
                confidence: data2[i],
                class_id: data3[i] as usize,
            });
        }
        Ok(out)
    }

    fn parse(
        format: &YoloFormat,
        tensors: &[(&[f32], &[usize])],
    ) -> Result<Vec<RawDetection>, ConverterError> {
        Self::ensure_nonempty(tensors)?;
        match format {
            YoloFormat::V8 { num_classes } => {
                if tensors.len() != 1 {
                    return Err(ConverterError::UnexpectedTensorCount {
                        expected: "1",
                        got: tensors.len(),
                    });
                }
                let (data, shape) = tensors[0];
                Self::parse_v8(data, shape, *num_classes)
            }
            YoloFormat::V5 { num_classes } => {
                if tensors.len() != 1 {
                    return Err(ConverterError::UnexpectedTensorCount {
                        expected: "1",
                        got: tensors.len(),
                    });
                }
                let (data, shape) = tensors[0];
                Self::parse_v5(data, shape, *num_classes)
            }
            YoloFormat::V4 {
                model_width,
                model_height,
            } => {
                if tensors.len() != 2 {
                    return Err(ConverterError::UnexpectedTensorCount {
                        expected: "2",
                        got: tensors.len(),
                    });
                }
                let (d0, s0) = tensors[0];
                let (d1, s1) = tensors[1];
                Self::parse_v4(d0, s0, d1, s1, *model_width, *model_height)
            }
            YoloFormat::V3Raw => {
                if tensors.len() != 3 {
                    return Err(ConverterError::UnexpectedTensorCount {
                        expected: "3",
                        got: tensors.len(),
                    });
                }
                let (d0, s0) = tensors[0];
                let (d1, s1) = tensors[1];
                let (d2, s2) = tensors[2];
                Self::parse_v3_raw(d0, s0, d1, s1, d2, s2)
            }
            YoloFormat::V4PostNms {
                model_width,
                model_height,
            } => {
                if tensors.len() != 4 {
                    return Err(ConverterError::UnexpectedTensorCount {
                        expected: "4",
                        got: tensors.len(),
                    });
                }
                let (d0, s0) = tensors[0];
                let (d1, s1) = tensors[1];
                let (d2, s2) = tensors[2];
                let (d3, s3) = tensors[3];
                Self::parse_v4_post_nms(d0, s0, d1, s1, d2, s2, d3, s3, *model_width, *model_height)
            }
        }
    }

    /// Decodes tensors into detections grouped by class id.
    pub fn decode(
        &self,
        tensors: &[(&[f32], &[usize])],
    ) -> Result<HashMap<usize, Vec<(f32, RBBox)>>, ConverterError> {
        let mut raw = Self::parse(&self.format, tensors)?;

        raw.retain(|d| d.confidence > self.effective_threshold(d.class_id));
        if let Some(ref allowed) = self.class_filter {
            raw.retain(|d| allowed.contains(&d.class_id));
        }

        let n = raw.len();
        if n == 0 {
            return Ok(HashMap::new());
        }

        let mut boxes: Vec<[f32; 4]> = Vec::with_capacity(n);
        let mut confidences: Vec<f32> = Vec::with_capacity(n);
        let mut class_ids: Vec<usize> = Vec::with_capacity(n);
        for d in &raw {
            boxes.push([d.xc, d.yc, d.w, d.h]);
            confidences.push(d.confidence);
            class_ids.push(d.class_id);
        }

        let kept_indices: Vec<usize> = match &self.nms {
            NmsKind::None { top_k } => {
                let mut order: Vec<usize> = (0..n).collect();
                order.sort_by(|&i, &j| {
                    confidences[j]
                        .partial_cmp(&confidences[i])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                order.truncate(*top_k);
                order
            }
            NmsKind::ClassAgnostic {
                iou_threshold,
                top_k,
            } => nms_class_agnostic(&boxes, &confidences, *iou_threshold, *top_k)
                .map_err(|e| ConverterError::NmsInputError(e.to_string()))?,
            NmsKind::ClassAware {
                iou_threshold,
                top_k,
            } => nms_class_aware(&boxes, &confidences, &class_ids, *iou_threshold, *top_k)
                .map_err(|e| ConverterError::NmsInputError(e.to_string()))?,
        };

        let mut by_class: HashMap<usize, Vec<(f32, RBBox)>> = HashMap::new();
        for idx in kept_indices {
            let d = &raw[idx];
            let bbox = RBBox::new(d.xc, d.yc, d.w, d.h, None);
            by_class
                .entry(d.class_id)
                .or_default()
                .push((d.confidence, bbox));
        }
        Ok(by_class)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_converter(format: YoloFormat) -> YoloDetectionConverter {
        YoloDetectionConverter::new(
            format,
            0.01,
            HashMap::new(),
            None,
            NmsKind::None { top_k: 100 },
        )
    }

    #[test]
    fn v8_basic() {
        let num_classes = 2usize;
        let n = 3usize;
        let rows = num_classes + 4;
        let mut data = vec![0.0_f32; rows * n];
        for j in 0..n {
            data[0 * n + j] = 0.1 * (j + 1) as f32;
            data[1 * n + j] = 0.2 * (j + 1) as f32;
            data[2 * n + j] = 0.3;
            data[3 * n + j] = 0.4;
            data[4 * n + j] = 0.9 - j as f32 * 0.1;
            data[5 * n + j] = 0.1 + j as f32 * 0.1;
        }
        let shape = [rows, n];
        let tensors = [(&data[..], &shape[..])];
        let conv = default_converter(YoloFormat::V8 { num_classes });
        let out = conv.decode(&tensors).unwrap();
        let class0 = out.get(&0).expect("class 0");
        assert_eq!(class0.len(), n);
        for j in 0..n {
            let exp_conf = data[4 * n + j].max(data[5 * n + j]);
            let found = class0.iter().any(|(c, b)| {
                (c - exp_conf).abs() < 1e-5
                    && (b.get_xc() - data[j]).abs() < 1e-5
                    && (b.get_yc() - data[1 * n + j]).abs() < 1e-5
            });
            assert!(found, "column j={j}");
        }
    }

    #[test]
    fn v5_basic() {
        let num_classes = 2usize;
        let stride = num_classes + 5;
        let n = 2usize;
        let mut data = vec![0.0_f32; n * stride];
        data[0] = 1.0;
        data[1] = 2.0;
        data[2] = 3.0;
        data[3] = 4.0;
        data[4] = 0.5;
        data[5] = 0.8;
        data[6] = 0.2;
        data[stride] = 10.0;
        data[stride + 1] = 20.0;
        data[stride + 2] = 1.0;
        data[stride + 3] = 1.0;
        data[stride + 4] = 1.0;
        data[stride + 5] = 0.1;
        data[stride + 6] = 0.95;
        let shape = [n, stride];
        let tensors = [(&data[..], &shape[..])];
        let conv = default_converter(YoloFormat::V5 { num_classes });
        let out = conv.decode(&tensors).unwrap();
        let c0 = out.get(&0).expect("class 0");
        assert!((c0[0].0 - 0.4).abs() < 1e-5);
        let c1 = out.get(&1).expect("class 1");
        assert!((c1[0].0 - 0.95).abs() < 1e-5);
    }

    #[test]
    fn v4_basic() {
        let mw = 640.0_f32;
        let mh = 480.0_f32;
        let n = 2usize;
        let k = 3usize;
        let mut d0 = vec![0.0_f32; n * 4];
        d0[0] = 0.0;
        d0[1] = 0.0;
        d0[2] = 0.5;
        d0[3] = 0.5;
        d0[4] = 0.25;
        d0[5] = 0.25;
        d0[6] = 0.75;
        d0[7] = 0.75;
        let mut d1 = vec![0.0_f32; n * k];
        d1[0] = 0.1;
        d1[1] = 0.2;
        d1[2] = 0.7;
        d1[k] = 0.9;
        d1[k + 1] = 0.05;
        d1[k + 2] = 0.05;
        let s0 = [n, 4];
        let s1 = [n, k];
        let tensors = [(&d0[..], &s0[..]), (&d1[..], &s1[..])];
        let conv = default_converter(YoloFormat::V4 {
            model_width: mw,
            model_height: mh,
        });
        let out = conv.decode(&tensors).unwrap();
        let det1 = out.get(&2).unwrap().first().unwrap();
        assert!((det1.0 - 0.7).abs() < 1e-4);
        let left = 0.0_f32 * mw;
        let top = 0.0 * mh;
        let right = 0.5 * mw;
        let bottom = 0.5 * mh;
        let w = right - left;
        let h = bottom - top;
        let xc = left + w * 0.5;
        let yc = top + h * 0.5;
        assert!((det1.1.get_xc() - xc).abs() < 1e-3);
        assert!((det1.1.get_yc() - yc).abs() < 1e-3);
    }

    #[test]
    fn v3_raw_basic() {
        let n = 2usize;
        let k = 2usize;
        let mut d0 = vec![0.0_f32; n * 4];
        d0[0] = 10.0;
        d0[1] = 20.0;
        d0[2] = 4.0;
        d0[3] = 6.0;
        d0[4] = 1.0;
        d0[5] = 2.0;
        d0[6] = 3.0;
        d0[7] = 3.0;
        let mut d1 = vec![0.0_f32; n * k];
        d1[0] = 0.2;
        d1[1] = 0.6;
        d1[k] = 0.9;
        d1[k + 1] = 0.1;
        let mut d2 = vec![0.0_f32; n];
        d2[0] = 1.0;
        d2[1] = 0.0;
        let s0 = [n, 4];
        let s1 = [n, k];
        let s2 = [n];
        let tensors = [(&d0[..], &s0[..]), (&d1[..], &s1[..]), (&d2[..], &s2[..])];
        let conv = default_converter(YoloFormat::V3Raw);
        let out = conv.decode(&tensors).unwrap();
        assert!(out.get(&1).is_some());
        assert!(out.get(&0).is_some());
    }

    #[test]
    fn v4_post_nms_basic() {
        let mw = 100.0_f32;
        let mh = 100.0_f32;
        let n = 4usize;
        let num_dets = 2.0_f32;
        let d0 = vec![num_dets];
        let s0 = [1usize];
        let mut d1 = vec![0.0_f32; n * 4];
        d1[0] = 0.0;
        d1[1] = 0.0;
        d1[2] = 1.0;
        d1[3] = 1.0;
        d1[4] = 0.0;
        d1[5] = 0.0;
        d1[6] = 0.5;
        d1[7] = 0.5;
        let mut d2 = vec![0.0_f32; n];
        d2[0] = 0.95;
        d2[1] = 0.85;
        let mut d3 = vec![0.0_f32; n];
        d3[0] = 3.0;
        d3[1] = 7.0;
        let s1 = [n, 4];
        let s2 = [n];
        let s3 = [n];
        let tensors = [
            (&d0[..], &s0[..]),
            (&d1[..], &s1[..]),
            (&d2[..], &s2[..]),
            (&d3[..], &s3[..]),
        ];
        let conv = default_converter(YoloFormat::V4PostNms {
            model_width: mw,
            model_height: mh,
        });
        let out = conv.decode(&tensors).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out.contains_key(&3));
        assert!(out.contains_key(&7));
    }

    #[test]
    fn confidence_filter() {
        let num_classes = 1usize;
        let n = 2usize;
        let rows = num_classes + 4;
        let mut data = vec![0.0_f32; rows * n];
        for j in 0..n {
            data[0 * n + j] = 10.0;
            data[1 * n + j] = 10.0;
            data[2 * n + j] = 2.0;
            data[3 * n + j] = 2.0;
            data[4 * n + j] = if j == 0 { 0.9 } else { 0.3 };
        }
        let shape = [rows, n];
        let tensors = [(&data[..], &shape[..])];
        let conv = YoloDetectionConverter::new(
            YoloFormat::V8 { num_classes },
            0.5,
            HashMap::new(),
            None,
            NmsKind::None { top_k: 10 },
        );
        let out = conv.decode(&tensors).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out.contains_key(&0));
    }

    #[test]
    fn per_class_confidence() {
        let num_classes = 2usize;
        let n = 2usize;
        let rows = num_classes + 4;
        let mut data = vec![0.0_f32; rows * n];
        for j in 0..n {
            data[0 * n + j] = 0.0;
            data[1 * n + j] = 0.0;
            data[2 * n + j] = 1.0;
            data[3 * n + j] = 1.0;
            if j == 0 {
                data[4 * n + j] = 0.85;
                data[5 * n + j] = 0.1;
            } else {
                data[4 * n + j] = 0.1;
                data[5 * n + j] = 0.85;
            }
        }
        let shape = [rows, n];
        let tensors = [(&data[..], &shape[..])];
        let mut per = HashMap::new();
        per.insert(0, 0.9_f32);
        let conv = YoloDetectionConverter::new(
            YoloFormat::V8 { num_classes },
            0.1,
            per,
            None,
            NmsKind::None { top_k: 10 },
        );
        let out = conv.decode(&tensors).unwrap();
        assert!(!out.contains_key(&0));
        assert!(out.contains_key(&1));
    }

    #[test]
    fn class_filter_whitelist() {
        let num_classes = 3usize;
        let n = 3usize;
        let rows = num_classes + 4;
        let mut data = vec![0.0_f32; rows * n];
        for j in 0..n {
            data[0 * n + j] = 0.0;
            data[1 * n + j] = 0.0;
            data[2 * n + j] = 1.0;
            data[3 * n + j] = 1.0;
            for c in 0..num_classes {
                data[(4 + c) * n + j] = if c == j { 0.99 } else { 0.01 };
            }
        }
        let shape = [rows, n];
        let tensors = [(&data[..], &shape[..])];
        let mut whitelist = HashSet::new();
        whitelist.insert(0);
        whitelist.insert(2);
        let conv = YoloDetectionConverter::new(
            YoloFormat::V8 { num_classes },
            0.5,
            HashMap::new(),
            Some(whitelist),
            NmsKind::None { top_k: 10 },
        );
        let out = conv.decode(&tensors).unwrap();
        assert_eq!(out.len(), 2);
        assert!(!out.contains_key(&1));
    }

    #[test]
    fn nms_applied() {
        let num_classes = 1usize;
        let n = 2usize;
        let rows = num_classes + 4;
        let mut data = vec![0.0_f32; rows * n];
        for j in 0..n {
            data[0 * n + j] = 10.0;
            data[1 * n + j] = 10.0;
            data[2 * n + j] = 4.0;
            data[3 * n + j] = 4.0;
            data[4 * n + j] = 0.99;
        }
        let shape = [rows, n];
        let tensors = [(&data[..], &shape[..])];
        let conv = YoloDetectionConverter::new(
            YoloFormat::V8 { num_classes },
            0.1,
            HashMap::new(),
            None,
            NmsKind::ClassAgnostic {
                iou_threshold: 0.5,
                top_k: 10,
            },
        );
        let out = conv.decode(&tensors).unwrap();
        let total: usize = out.values().map(|v| v.len()).sum();
        assert_eq!(total, 1);
    }

    #[test]
    fn decode_wrong_tensor_count() {
        let data = [1.0_f32];
        let shape = [1usize, 1usize];
        let tensors = [(&data[..], &shape[..]), (&data[..], &shape[..])];
        let conv = default_converter(YoloFormat::V8 { num_classes: 1 });
        let err = conv.decode(&tensors).unwrap_err();
        match err {
            ConverterError::UnexpectedTensorCount { .. } => {}
            _ => panic!("expected UnexpectedTensorCount"),
        }
    }

    #[test]
    fn decode_empty_tensor() {
        let data: [f32; 0] = [];
        let shape = [0usize; 0];
        let tensors = [(&data[..], &shape[..])];
        let conv = default_converter(YoloFormat::V8 { num_classes: 1 });
        let err = conv.decode(&tensors).unwrap_err();
        match err {
            ConverterError::EmptyTensor => {}
            _ => panic!("expected EmptyTensor"),
        }
    }
}
