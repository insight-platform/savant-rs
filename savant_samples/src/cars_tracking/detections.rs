//! YOLO post-processing for the `cars_tracking` sample.
//!
//! The nvinfer result callback runs on the operator's internal completion
//! thread.  It is forbidden to `unseal()` there — doing so would block that
//! thread and pin GPU slots for the whole time the downstream stage takes
//! to consume the frame.  Instead the callback:
//!
//! 1. Reads every output tensor while the [`OperatorInferenceOutput`] is
//!    still alive (pointers are only valid for that lifetime),
//! 2. Attaches decoded detections as `VideoObject`s onto the frame,
//! 3. Calls `take_deliveries()` and forwards the resulting
//!    [`SealedDeliveries`] through a bounded channel.
//!
//! A dedicated consumer thread (in [`crate::cars_tracking::pipeline`]) pulls
//! the sealed handles off the channel, calls `unseal()` and submits the
//! frames to the tracker — this keeps backpressure healthy and ensures GPU
//! slots are freed promptly.
//!
//! Per-source EOS / errors are only logged; hard shutdown is driven by the
//! orchestrator via `graceful_shutdown()` and sender drops.

use crossbeam::channel::{Receiver, Sender};
use deepstream_buffers::SavantIdMetaKind;
use deepstream_nvinfer::{
    BatchFormationCallback, BatchFormationResult, OperatorOutput, OperatorResultCallback, RoiKind,
    SealedDeliveries,
};
use savant_core::converters::{NmsKind, YoloDetectionConverter, YoloFormat};
use savant_core::primitives::object::{
    IdCollisionResolutionPolicy, ObjectOperations, VideoObjectBuilder,
};
use savant_core::primitives::RBBox;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::cars_tracking::model::{
    vehicle_label, DETECTION_NAMESPACE, VEHICLE_CLASS_IDS, YOLO_NUM_CLASSES,
};

/// Confidence threshold used by the post-processing.
pub const CONFIDENCE_THRESHOLD: f32 = 0.25;
/// IoU threshold used by NMS.
pub const NMS_IOU_THRESHOLD: f32 = 0.45;
/// Maximum detections retained after NMS.
pub const NMS_TOP_K: usize = 300;

/// Sender half of the nvinfer-result channel — forwards sealed deliveries
/// from the operator callback to the consumer thread.
pub type InferResultSender = Sender<SealedDeliveries>;
/// Receiver half of the nvinfer-result channel.
pub type InferResultReceiver = Receiver<SealedDeliveries>;

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

/// Build the batch-formation callback used by `NvInferBatchingOperator`.
///
/// Each slot processes a single full-frame ROI (the whole image).
pub fn build_batch_formation() -> BatchFormationCallback {
    Arc::new(|frames| {
        let ids = frames
            .iter()
            .enumerate()
            .map(|(slot, _)| SavantIdMetaKind::Frame(slot as u128))
            .collect();
        let rois = frames.iter().map(|_| RoiKind::FullFrame).collect();
        BatchFormationResult { ids, rois }
    })
}

/// Process a single [`OperatorOutput`]: attach detections on inference
/// batches and forward the resulting [`SealedDeliveries`]; log and discard
/// per-source EOS / operator errors.
///
/// Used by both the result callback (normal flow) and the orchestrator's
/// drain step (which receives a `Vec<OperatorOutput>` from
/// `graceful_shutdown` instead of having the callback fire).
pub fn process_infer_output(
    output: OperatorOutput,
    converter: &YoloDetectionConverter,
    forward: &InferResultSender,
    stats: Option<&InferStats>,
) {
    match output {
        OperatorOutput::Inference(mut inf) => {
            let detections_before = count_detection_objects(&inf);
            attach_detections(&inf, converter);
            let detections_after = count_detection_objects(&inf);
            let new_detections = detections_after.saturating_sub(detections_before);
            let frame_count = inf.frames().len() as u64;
            log::debug!(
                "[infer] attached {new_detections} detection(s) across {frame_count} frame(s)"
            );
            if let Some(s) = stats {
                s.record_batch(frame_count, new_detections as u64);
            }
            if let Some(sealed) = inf.take_deliveries() {
                drop(inf);
                if forward.send(sealed).is_err() {
                    log::warn!("nvinfer result receiver closed; dropping sealed batch");
                }
            }
        }
        OperatorOutput::Eos { source_id } => {
            log::info!("nvinfer source EOS: {source_id}");
        }
        OperatorOutput::Error(err) => {
            log::error!("nvinfer operator error: {err}");
        }
    }
}

/// Build the result callback that decodes YOLO output and forwards deliveries.
///
/// Thin wrapper over [`process_infer_output`] so the drain path
/// (`graceful_shutdown` → `Vec<OperatorOutput>`) can reuse the same logic.
pub fn build_result_callback(
    converter: Arc<YoloDetectionConverter>,
    forward: InferResultSender,
    stats: Option<Arc<InferStats>>,
) -> OperatorResultCallback {
    Box::new(move |output: OperatorOutput| {
        process_infer_output(output, converter.as_ref(), &forward, stats.as_deref())
    })
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

    /// Per-source EOS / operator errors must be logged only — pipeline
    /// shutdown is driven by the orchestrator (sender drop), so the callback
    /// must NOT push anything onto the sealed-deliveries channel.
    #[test]
    fn callback_does_not_forward_on_source_eos_or_error() {
        use deepstream_nvinfer::NvInferError;
        let (tx, rx) = crossbeam::channel::bounded::<SealedDeliveries>(1);
        let converter = build_yolo_converter(CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD);
        let mut cb = build_result_callback(converter, tx, None);

        cb(OperatorOutput::Eos {
            source_id: "cam-1".to_string(),
        });
        cb(OperatorOutput::Error(NvInferError::PipelineError(
            "synthetic".to_string(),
        )));

        assert!(rx.try_recv().is_err());
    }
}
