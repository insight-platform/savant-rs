//! Inference output processing — turns an
//! [`OperatorInferenceOutput`] into downstream artefacts for the
//! `cars_tracking` sample.
//!
//! Scope:
//!
//! * YOLO post-processing configuration — [`build_yolo_converter`] and
//!   the [`CONFIDENCE_THRESHOLD`] / [`NMS_IOU_THRESHOLD`] /
//!   [`NMS_TOP_K`] constants.
//! * [`process_infer_output`] — the `on_inference` hook body
//!   installed on [`NvInfer`](savant_perception::templates::NvInfer)
//!   via its [`on_inference`](savant_perception::templates::NvInferBuilder::on_inference)
//!   setter.  Decodes the YOLO `output0` tensor, attaches
//!   detections, and forwards `PipelineMsg::Deliveries` downstream.
//!   Source-EOS + operator-error propagation lives in the framework
//!   defaults ([`NvInfer::default_on_source_eos`](savant_perception::templates::NvInfer::default_on_source_eos)
//!   and [`NvInfer::default_on_error`](savant_perception::templates::NvInfer::default_on_error)),
//!   so this sample-side processor only sees the inference variant.
//! * The per-frame YOLO `output0` tensor → [`RBBox`] →
//!   `VideoObject` pipeline ([`attach_detections`] +
//!   [`count_detection_objects`]).
//! * [`InferStats`] — sample-level frame / detection counters
//!   written from this module (the only place that touches tensor
//!   outputs) and read by the orchestrator for the end-of-run
//!   summary.

use deepstream_nvinfer::{OperatorInferenceOutput, SealedDeliveries};
use savant_core::converters::{NmsKind, YoloDetectionConverter, YoloFormat};
use savant_core::primitives::object::{
    IdCollisionResolutionPolicy, ObjectOperations, VideoObjectBuilder,
};
use savant_core::primitives::RBBox;
// `HashMap`/`HashSet` here must be `std::collections::*` because they
// are handed to `YoloDetectionConverter::new`, which is a boundary API
// that takes the std types.  The rest of `cars_tracking` uses
// `hashbrown`.
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::model::{vehicle_label, DETECTION_NAMESPACE, VEHICLE_CLASS_IDS, YOLO_NUM_CLASSES};
use savant_perception::supervisor::StageName;

/// Confidence threshold used by the post-processing.  Reference
/// default — the CLI supplies the runtime value.
#[allow(dead_code)]
pub const CONFIDENCE_THRESHOLD: f32 = 0.25;
/// IoU threshold used by NMS.  Reference default — the CLI supplies
/// the runtime value.
#[allow(dead_code)]
pub const NMS_IOU_THRESHOLD: f32 = 0.45;
/// Maximum detections retained after NMS.
pub const NMS_TOP_K: usize = 300;

/// Build a shared `YoloDetectionConverter` for YOLOv11 with a vehicle filter.
pub fn build_yolo_converter(conf: f32, iou: f32) -> Arc<YoloDetectionConverter> {
    let class_filter: HashSet<usize> = VEHICLE_CLASS_IDS.iter().copied().collect();
    Arc::new(YoloDetectionConverter::new(
        YoloFormat::V8 {
            num_classes: YOLO_NUM_CLASSES,
        },
        conf,
        HashMap::new(),
        Some(class_filter),
        NmsKind::ClassAgnostic {
            iou_threshold: iou,
            top_k: NMS_TOP_K,
        },
    ))
}

/// Process a single [`OperatorInferenceOutput`]: run YOLO
/// post-processing and attach detections to every frame in the
/// batch, updating [`InferStats`] along the way, then return the
/// sealed batch ready for the caller to forward.
///
/// **Pure data transform** — no sending, no routing.  The caller
/// (the `on_inference` hook body installed on
/// [`NvInfer`](savant_perception::templates::NvInfer)) owns the
/// `Router<PipelineMsg>` and is the single place that wraps the
/// returned [`SealedDeliveries`] in a [`PipelineMsg::Deliveries`]
/// envelope and emits it.  That split keeps metadata attachment
/// decoupled from downstream routing policy.
///
/// Source-EOS and operator errors are handled by the framework's
/// per-variant defaults — see
/// [`NvInfer::default_on_source_eos`](savant_perception::templates::NvInfer::default_on_source_eos)
/// and [`NvInfer::default_on_error`](savant_perception::templates::NvInfer::default_on_error) —
/// so this sample-side processor only handles the inference variant.
///
/// Returns `None` if the batch has no sealed deliveries to forward
/// (the operator can yield an empty result after metadata-only
/// errors; the caller skips the downstream send in that case).
///
/// [`PipelineMsg::Deliveries`]: savant_perception::envelopes::PipelineMsg::Deliveries
pub fn process_infer_output(
    mut inf: OperatorInferenceOutput,
    converter: &YoloDetectionConverter,
    stats: Option<&InferStats>,
    stage: &StageName,
) -> Option<SealedDeliveries> {
    let detections_before = count_detection_objects(&inf);
    attach_detections(&inf, converter);
    let detections_after = count_detection_objects(&inf);
    let new_detections = detections_after.saturating_sub(detections_before);
    let frame_count = inf.frames().len() as u64;
    log::debug!("[{stage}] attached {new_detections} detection(s) across {frame_count} frame(s)");
    if let Some(s) = stats {
        s.record_batch(frame_count, new_detections as u64);
    }
    inf.take_deliveries()
}

fn count_detection_objects(output: &deepstream_nvinfer::OperatorInferenceOutput) -> usize {
    output
        .frames()
        .iter()
        .map(|fo| {
            fo.frame
                .get_all_objects()
                .into_iter()
                .filter(|o| o.get_namespace() == DETECTION_NAMESPACE)
                .count()
        })
        .sum()
}

/// Aggregate counters tracked across the inference stage.
///
/// Cheap to clone (all fields are atomics) and cheap to tick (no locks).
#[derive(Default, Debug)]
pub struct InferStats {
    frames: std::sync::atomic::AtomicU64,
    detections: std::sync::atomic::AtomicU64,
}

impl InferStats {
    /// Create fresh zeroed counters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the outcome of one inference batch (usually one frame in
    /// the file-driven frame-by-frame sample).
    pub fn record_batch(&self, frames: u64, detections: u64) {
        use std::sync::atomic::Ordering;
        self.frames.fetch_add(frames, Ordering::Relaxed);
        self.detections.fetch_add(detections, Ordering::Relaxed);
    }

    /// Current frame count.
    pub fn frames(&self) -> u64 {
        self.frames.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Current detection count.
    pub fn detections(&self) -> u64 {
        self.detections.load(std::sync::atomic::Ordering::Relaxed)
    }
}

fn attach_detections(
    output: &deepstream_nvinfer::OperatorInferenceOutput,
    converter: &YoloDetectionConverter,
) {
    for frame_out in output.frames() {
        if frame_out.elements.is_empty() {
            log::warn!(
                "inference frame {} had no element outputs",
                frame_out.frame.get_source_id()
            );
            continue;
        }
        let elem = &frame_out.elements[0];
        let tensor = match elem.tensors.iter().find(|t| t.name == "output0") {
            Some(t) => t,
            None => {
                log::warn!(
                    "missing output0 tensor for frame {}",
                    frame_out.frame.get_source_id()
                );
                continue;
            }
        };
        let data = match tensor.to_f32_vec() {
            Ok(data) if !data.is_empty() => data,
            Ok(_) => continue,
            Err(err) => {
                log::error!(
                    "unable to read YOLO output tensor for frame {}: {err}",
                    frame_out.frame.get_source_id()
                );
                continue;
            }
        };
        let raw_shape: Vec<usize> = tensor.dims.dimensions.iter().map(|&d| d as usize).collect();
        let shape: Vec<usize> = if raw_shape.len() == 3 && raw_shape[0] == 1 {
            raw_shape[1..].to_vec()
        } else {
            raw_shape
        };

        let decoded = match converter.decode(&[(&data[..], &shape[..])]) {
            Ok(decoded) => decoded,
            Err(err) => {
                log::warn!(
                    "YOLO decode failed for frame {}: {err}",
                    frame_out.frame.get_source_id()
                );
                continue;
            }
        };

        let scaler = elem.coordinate_scaler();
        let frame = frame_out.frame.clone();
        for (class_id, detections) in decoded {
            let Some(label) = vehicle_label(class_id) else {
                continue;
            };
            for (confidence, bbox) in detections {
                let scaled = scaler.scale_rbbox(&bbox);
                let obj = match VideoObjectBuilder::default()
                    .id(0)
                    .namespace(DETECTION_NAMESPACE.into())
                    .label(label.into())
                    .detection_box(RBBox::new(
                        scaled.get_xc(),
                        scaled.get_yc(),
                        scaled.get_width(),
                        scaled.get_height(),
                        None,
                    ))
                    .confidence(Some(confidence))
                    .build()
                {
                    Ok(obj) => obj,
                    Err(err) => {
                        log::warn!("failed to build detection object: {err}");
                        continue;
                    }
                };
                if let Err(err) = frame.add_object(obj, IdCollisionResolutionPolicy::GenerateNewId)
                {
                    log::warn!("failed to attach detection object: {err}");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `build_yolo_converter` wires a class-agnostic NMS converter
    /// filtered to the sample's vehicle class ids.  The converter
    /// itself is covered by `savant_core`; this test only pins that
    /// our helper accepts the tuning knobs.
    #[test]
    fn build_yolo_converter_is_constructible() {
        let _ = build_yolo_converter(CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD);
    }
}
