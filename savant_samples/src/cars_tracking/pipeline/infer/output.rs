//! Inference output processing — everything that turns a concrete
//! nvinfer [`OperatorOutput`] into downstream artefacts for the
//! `cars_tracking` sample.
//!
//! Scope:
//!
//! * YOLO post-processing configuration — [`build_yolo_converter`] and
//!   the [`CONFIDENCE_THRESHOLD`] / [`NMS_IOU_THRESHOLD`] /
//!   [`NMS_TOP_K`] constants.
//! * The [`OperatorResultCallback`] installed on
//!   [`deepstream_nvinfer::prelude::NvInferBatchingOperator`] —
//!   [`build_result_callback`] and the shared
//!   [`process_infer_output`] dispatcher used by both the callback
//!   and the drain path in [`super::spawn_infer_thread`].
//! * The per-frame YOLO `output0` tensor → [`RBBox`] →
//!   `VideoObject` pipeline ([`attach_detections`] +
//!   [`count_detection_objects`]).
//! * The downstream channel type aliases
//!   ([`InferResultSender`] / [`InferResultReceiver`]) that carry the
//!   sealed nvinfer deliveries to the tracker stage.
//! * [`InferStats`] — sample-level frame / detection counters
//!   written from this module (the only place that touches tensor
//!   outputs) and read by the orchestrator for the end-of-run
//!   summary.
//!
//! The split keeps [`super`](super) (the actor thread +
//! [`BatchFormationCallback`]) free of tensor-decoding detail, and
//! keeps this module free of operator-lifecycle concerns.  Stream
//! alignment for the in-band [`PipelineMsg::SourceEos`] sentinel is
//! documented on [`process_infer_output`].

use crossbeam::channel::{Receiver, Sender};
use deepstream_nvinfer::{OperatorOutput, OperatorResultCallback};
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
use crate::cars_tracking::message::PipelineMsg;

/// Confidence threshold used by the post-processing.
pub const CONFIDENCE_THRESHOLD: f32 = 0.25;
/// IoU threshold used by NMS.
pub const NMS_IOU_THRESHOLD: f32 = 0.45;
/// Maximum detections retained after NMS.
pub const NMS_TOP_K: usize = 300;

/// Sender half of the nvinfer-result channel — forwards [`PipelineMsg`]
/// from the operator callback + infer-thread EOS handler to the tracker
/// thread.  The infer stage emits the batched
/// [`PipelineMsg::Deliveries`] variant (the boxed payload is a
/// `deepstream_nvinfer::SealedDeliveries`); the singular `Delivery`
/// variant is never emitted on this channel.
pub type InferResultSender = Sender<PipelineMsg>;
/// Receiver half of the nvinfer-result channel.
pub type InferResultReceiver = Receiver<PipelineMsg>;

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

/// Process a single [`OperatorOutput`]: attach detections on inference
/// batches, forward the resulting [`SealedDeliveries`], and — critically
/// — forward the in-band [`PipelineMsg::SourceEos`] sentinel when the
/// operator emits [`OperatorOutput::Eos`].
///
/// Used by both the result callback (normal flow) and the orchestrator's
/// drain step (which receives a `Vec<OperatorOutput>` from
/// `graceful_shutdown` instead of having the callback fire).
///
/// `SourceEos` propagation happens **here**, not in the infer thread's
/// main loop.  The operator delivers results in per-source order and
/// emits `OperatorOutput::Eos { source_id }` strictly after the last
/// delivery for that source; forwarding the sentinel from the main
/// loop (upon receiving upstream `PipelineMsg::SourceEos`) would race
/// ahead of still-in-flight deliveries and make `SourceEos`
/// *out-of-band* on the infer->tracker channel.  Keeping the emit
/// inside this single call site guarantees stream alignment.
///
/// [`SealedDeliveries`]: deepstream_nvinfer::SealedDeliveries
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
                if forward
                    .send(PipelineMsg::Deliveries(Box::new(sealed)))
                    .is_err()
                {
                    log::warn!("nvinfer result receiver closed; dropping sealed batch");
                }
            }
        }
        OperatorOutput::Eos { source_id } => {
            // Stream-aligned propagation: the operator guarantees
            // this fires strictly after the last `Inference` output
            // for `source_id`, so the downstream receiver sees every
            // delivery *before* the sentinel.  This is the only
            // place in the stage where `SourceEos` leaves.
            log::info!("[infer/cb] OperatorOutput::Eos for source_id={source_id}; propagating");
            if forward
                .send(PipelineMsg::SourceEos {
                    source_id: source_id.clone(),
                })
                .is_err()
            {
                log::warn!("[infer/cb] downstream closed; dropping SourceEos({source_id})");
            }
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

    /// The callback is the only place that emits `PipelineMsg::SourceEos`
    /// because it is the only vantage point where "no more deliveries
    /// for this source will follow" is an invariant (the operator
    /// guarantees the `Eos` output fires strictly after the last
    /// `Inference` output for the same source_id).  Operator errors
    /// stay log-only.
    #[test]
    fn callback_forwards_source_eos_but_not_errors() {
        use deepstream_nvinfer::NvInferError;
        let (tx, rx) = crossbeam::channel::bounded::<PipelineMsg>(1);
        let converter = build_yolo_converter(CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD);
        let mut cb = build_result_callback(converter, tx, None);

        cb(OperatorOutput::Eos {
            source_id: "cam-1".to_string(),
        });
        match rx.try_recv().expect("expected SourceEos on the channel") {
            PipelineMsg::SourceEos { source_id } => assert_eq!(source_id, "cam-1"),
            PipelineMsg::Delivery(_) => panic!("unexpected Delivery on Eos"),
            PipelineMsg::Deliveries(_) => panic!("unexpected Deliveries on Eos"),
            PipelineMsg::Shutdown { .. } => panic!("unexpected Shutdown on Eos"),
        }

        cb(OperatorOutput::Error(NvInferError::PipelineError(
            "synthetic".to_string(),
        )));
        assert!(rx.try_recv().is_err());
    }
}
