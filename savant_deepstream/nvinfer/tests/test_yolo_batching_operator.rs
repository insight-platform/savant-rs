//! E2E test: [`NvInferBatchingOperator`] with mixed-resolution YOLO batches.
//!
//! Submits four batches sequentially through the batching operator, mixing
//! FullHD (1920×1080) and HD (1280×720) frames within and across batches:
//!
//! - **B1**: slot0 = FullHD, slot1 = HD   (auto-submit at max_batch_size=2)
//! - **B2**: slot0 = HD,     slot1 = FullHD (auto-submit)
//! - **B3**: slot0 = HD                   (flush)
//! - **B4**: slot0 = FullHD               (flush)
//!
//! Verifies:
//! 1. Correct slot count per batch.
//! 2. Every frame produces detections that match the YOLO ground truth.

mod common;

use common::yolo_test_utils::{
    load_ground_truth, match_detections, tensor_shape, tensor_to_f32_vec, yolo11n_properties,
};
use deepstream_buffers::{
    BufferGenerator, NvBufSurfaceMemType, SavantIdMetaKind, SurfaceView, VideoFormat,
};
use nvinfer::{
    BatchFormationResult, ModelColorFormat, ModelInputScaling, NvInferBatchingOperator,
    NvInferBatchingOperatorConfig, NvInferConfig, OperatorInferenceOutput, OperatorResultCallback,
    RoiKind,
};
use savant_core::converters::{NmsKind, YoloDetectionConverter, YoloFormat};
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
};
use savant_core::primitives::RBBox;
use serial_test::serial;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{mpsc, Arc};
use std::time::Duration;

const CONF_THRESHOLD: f32 = 0.25;
const NMS_IOU: f32 = 0.45;
const NMS_TOP_K: usize = 300;
const IOU_THRESHOLD: f32 = 0.4;
const MIN_GT_CONF: f32 = 0.4;
const MIN_MATCH_RATIO: f32 = 0.7;
const SCALING: ModelInputScaling = ModelInputScaling::Fill;

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets")
}

fn make_frame(source_id: &str, width: i64, height: i64) -> VideoFrameProxy {
    VideoFrameProxy::new(
        source_id,
        "30/1",
        width,
        height,
        VideoFrameContent::None,
        VideoFrameTranscodingMethod::Copy,
        &None,
        None,
        (1, 1_000_000_000),
        0,
        None,
        None,
    )
    .expect("build test frame")
}

fn load_image_buffer(name: &str, width: u32, height: u32) -> deepstream_buffers::SharedBuffer {
    let img_path = assets_dir().join(format!("yolo/{name}.jpg"));
    let img = image::open(&img_path)
        .unwrap_or_else(|e| panic!("failed to open {}: {e}", img_path.display()))
        .to_rgba8();
    assert_eq!(img.width(), width, "{name} width mismatch");
    assert_eq!(img.height(), height, "{name} height mismatch");
    let canvas = img.into_raw();

    let gen = BufferGenerator::builder(VideoFormat::RGBA, width, height)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(1)
        .max_buffers(1)
        .build()
        .expect("buffer generator");

    let shared = gen.acquire(Some(0)).unwrap();
    let view = SurfaceView::from_buffer(&shared, 0).unwrap();
    view.upload(&canvas, width, height, 4).expect("upload");
    drop(view);
    shared
}

struct FrameResult {
    source_id: String,
    width: i64,
    height: i64,
    detections: HashMap<usize, Vec<(f32, RBBox)>>,
}

struct BatchResult {
    num_frames: usize,
    frame_results: Vec<FrameResult>,
}

#[test]
#[serial]
fn test_yolo_batching_operator_mixed_sizes() {
    common::init();

    let gt_path = assets_dir().join("yolo/ground_truth.json");
    if !gt_path.exists() {
        eprintln!("Skipping: ground_truth.json not found");
        return;
    }
    let onnx = assets_dir().join("yolo11n.onnx");
    if !onnx.exists() {
        eprintln!("Skipping: yolo11n.onnx not found");
        return;
    }

    let gt = load_ground_truth();

    let props = yolo11n_properties();
    let nvinfer_config =
        NvInferConfig::new(props, VideoFormat::RGBA, 640, 640, ModelColorFormat::RGB)
            .scaling(SCALING);

    let operator_config = NvInferBatchingOperatorConfig {
        max_batch_size: 2,
        same_source_allowed: true,
        max_batch_wait: Duration::from_secs(5),
        nvinfer: nvinfer_config,
    };

    let converter = Arc::new(YoloDetectionConverter::new(
        YoloFormat::V8 { num_classes: 80 },
        CONF_THRESHOLD,
        HashMap::new(),
        None,
        NmsKind::ClassAgnostic {
            iou_threshold: NMS_IOU,
            top_k: NMS_TOP_K,
        },
    ));

    let (tx, rx) = mpsc::channel::<BatchResult>();

    let batch_formation: nvinfer::BatchFormationCallback = Arc::new(|frames| {
        let ids = frames
            .iter()
            .enumerate()
            .map(|(i, _)| SavantIdMetaKind::Frame(i as i64))
            .collect();
        let rois = frames.iter().map(|_| RoiKind::FullFrame).collect();
        BatchFormationResult { ids, rois }
    });

    let conv = converter.clone();
    let result_callback: OperatorResultCallback =
        Box::new(move |output: OperatorInferenceOutput| {
            let mut batch_result = BatchResult {
                num_frames: output.frames().len(),
                frame_results: Vec::new(),
            };

            for frame_out in output.frames() {
                let frame = &frame_out.frame;
                let source_id = frame.get_source_id();
                let width = frame.get_width();
                let height = frame.get_height();

                assert!(
                    !frame_out.elements.is_empty(),
                    "frame {source_id} ({width}x{height}) produced no elements"
                );

                let elem = &frame_out.elements[0];
                let tensor = elem
                    .tensors
                    .iter()
                    .find(|t| t.name == "output0")
                    .expect("missing output0 tensor");

                let data = tensor_to_f32_vec(tensor);
                let raw_shape = tensor_shape(tensor);
                let shape: Vec<usize> = if raw_shape.len() == 3 && raw_shape[0] == 1 {
                    raw_shape[1..].to_vec()
                } else {
                    raw_shape.clone()
                };

                let tensors = [(&data[..], &shape[..])];
                let detections = conv.decode(&tensors).expect("YOLO decode failed");

                let scaler = elem.coordinate_scaler();
                let mut scaled: HashMap<usize, Vec<(f32, RBBox)>> = HashMap::new();
                for (class_id, dets) in &detections {
                    for (conf, bbox) in dets {
                        let sb = scaler.scale_rbbox(bbox);
                        scaled.entry(*class_id).or_default().push((*conf, sb));
                    }
                }

                batch_result.frame_results.push(FrameResult {
                    source_id,
                    width,
                    height,
                    detections: scaled,
                });
            }

            tx.send(batch_result).unwrap();
        });

    let mut operator =
        NvInferBatchingOperator::new(operator_config, batch_formation, result_callback)
            .expect("create NvInferBatchingOperator");

    common::promote_built_engine("yolo11n.onnx", 1);

    // ── B1: FullHD + HD (fills max_batch_size=2 → auto-submit) ──────
    eprintln!("--- B1: FullHD + HD ---");
    operator
        .add_frame(
            make_frame("b1_fhd", 1920, 1080),
            load_image_buffer("barcelona_1920x1080", 1920, 1080),
        )
        .expect("B1 slot0");
    operator
        .add_frame(
            make_frame("b1_hd", 1280, 720),
            load_image_buffer("barcelona_1280x720", 1280, 720),
        )
        .expect("B1 slot1");

    // ── B2: HD + FullHD (fills → auto-submit) ───────────────────────
    eprintln!("--- B2: HD + FullHD ---");
    operator
        .add_frame(
            make_frame("b2_hd", 1280, 720),
            load_image_buffer("barcelona_1280x720", 1280, 720),
        )
        .expect("B2 slot0");
    operator
        .add_frame(
            make_frame("b2_fhd", 1920, 1080),
            load_image_buffer("barcelona_1920x1080", 1920, 1080),
        )
        .expect("B2 slot1");

    // ── B3: HD only (partial → flush) ───────────────────────────────
    eprintln!("--- B3: HD ---");
    operator
        .add_frame(
            make_frame("b3_hd", 1280, 720),
            load_image_buffer("barcelona_1280x720", 1280, 720),
        )
        .expect("B3 slot0");
    operator.flush().expect("flush B3");

    // ── B4: FullHD only (partial → flush) ───────────────────────────
    eprintln!("--- B4: FullHD ---");
    operator
        .add_frame(
            make_frame("b4_fhd", 1920, 1080),
            load_image_buffer("barcelona_1920x1080", 1920, 1080),
        )
        .expect("B4 slot0");
    operator.flush().expect("flush B4");

    // ── Collect all 4 batch results ─────────────────────────────────
    let mut results = Vec::with_capacity(4);
    for i in 0..4 {
        let r = rx
            .recv_timeout(Duration::from_secs(60))
            .unwrap_or_else(|_| panic!("timeout waiting for batch {} result", i + 1));
        results.push(r);
    }

    operator.shutdown().expect("operator shutdown");

    // ── Verify batch structure ──────────────────────────────────────
    assert_eq!(results[0].num_frames, 2, "B1 must have 2 frames");
    assert_eq!(results[1].num_frames, 2, "B2 must have 2 frames");
    assert_eq!(results[2].num_frames, 1, "B3 must have 1 frame");
    assert_eq!(results[3].num_frames, 1, "B4 must have 1 frame");

    // ── Verify detection quality per frame ──────────────────────────
    let mode_key = "fill";
    for (batch_idx, batch) in results.iter().enumerate() {
        for (frame_idx, fr) in batch.frame_results.iter().enumerate() {
            let image_name = if fr.width == 1920 && fr.height == 1080 {
                "barcelona_1920x1080"
            } else if fr.width == 1280 && fr.height == 720 {
                "barcelona_1280x720"
            } else {
                panic!(
                    "B{} frame{}: unexpected resolution {}x{}",
                    batch_idx + 1,
                    frame_idx,
                    fr.width,
                    fr.height
                );
            };

            let gt_key = format!("{image_name}_{mode_key}");
            let gt_dets = gt
                .get(&gt_key)
                .unwrap_or_else(|| panic!("missing GT key: {gt_key}"));

            let (matched, total, mismatches) =
                match_detections(&fr.detections, gt_dets, IOU_THRESHOLD, 0.0, MIN_GT_CONF);

            let ratio = if total > 0 {
                matched as f32 / total as f32
            } else {
                1.0
            };

            eprintln!(
                "  B{} frame{} [{gt_key}] ({src}): {matched}/{total} GT matched ({pct:.0}%)",
                batch_idx + 1,
                frame_idx,
                src = fr.source_id,
                pct = ratio * 100.0,
            );
            for m in &mismatches {
                eprintln!("    MISS: {m}");
            }

            assert!(
                ratio >= MIN_MATCH_RATIO,
                "B{} frame{} [{gt_key}] only {matched}/{total} ({:.0}%) matched, need {:.0}%",
                batch_idx + 1,
                frame_idx,
                ratio * 100.0,
                MIN_MATCH_RATIO * 100.0,
            );
        }
    }

    eprintln!("All 4 batches verified successfully.");
}
