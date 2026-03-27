//! Integration test: YOLO detection with converter and coordinate scaler.

mod common;

use common::yolo_test_utils::{
    load_ground_truth, match_detections, tensor_shape, tensor_to_f32_vec, yolo11n_properties,
};
use deepstream_buffers::{
    BufferGenerator, NvBufSurfaceMemType, SavantIdMetaKind, SurfaceView, UniformBatchGenerator,
    VideoFormat,
};
use nvinfer::{CoordinateScaler, ModelColorFormat, ModelInputScaling, NvInfer, NvInferConfig};
use savant_core::converters::{NmsKind, YoloDetectionConverter, YoloFormat};
use serial_test::serial;
use std::collections::HashMap;
use std::path::PathBuf;

const IOU_THRESHOLD: f32 = 0.4;
const MIN_GT_CONF: f32 = 0.4;
const CONF_THRESHOLD: f32 = 0.25;
const NMS_IOU: f32 = 0.45;
const NMS_TOP_K: usize = 300;
const MIN_MATCH_RATIO: f32 = 0.7;

const TEST_IMAGES: [(&str, u32, u32); 2] = [
    ("barcelona_1280x720", 1280, 720),
    ("barcelona_1920x1080", 1920, 1080),
];

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets")
}

fn scaling_key(scaling: ModelInputScaling) -> &'static str {
    match scaling {
        ModelInputScaling::Fill => "fill",
        ModelInputScaling::KeepAspectRatio => "keep_aspect_ratio",
        ModelInputScaling::KeepAspectRatioSymmetric => "keep_aspect_ratio_symmetric",
    }
}

fn build_yolo_engine(scaling: ModelInputScaling) -> Option<NvInfer> {
    let onnx = assets_dir().join("yolo11n.onnx");
    if !onnx.exists() {
        eprintln!("Skipping: yolo11n.onnx not found");
        return None;
    }
    let props = yolo11n_properties();
    let config = NvInferConfig::new(props, VideoFormat::RGBA, 640, 640, ModelColorFormat::RGB)
        .scaling(scaling);
    let engine = NvInfer::new(config, Box::new(|_| {})).expect("create NvInfer yolo11n");
    common::promote_built_engine("yolo11n.onnx", 1);
    Some(engine)
}

fn infer_one_image(
    engine: &NvInfer,
    converter: &YoloDetectionConverter,
    scaling: ModelInputScaling,
    image_name: &str,
    frame_w: u32,
    frame_h: u32,
    gt: &HashMap<String, Vec<common::yolo_test_utils::GtDetection>>,
) {
    let mode_key = scaling_key(scaling);
    let gt_key = format!("{image_name}_{mode_key}");
    let gt_dets = gt
        .get(&gt_key)
        .unwrap_or_else(|| panic!("missing GT key: {gt_key}"));

    common::warmup_engine(engine, frame_w, frame_h);

    let img_path = assets_dir().join(format!("yolo/{image_name}.jpg"));
    let img = image::open(&img_path)
        .unwrap_or_else(|e| panic!("failed to open {}: {e}", img_path.display()))
        .to_rgba8();
    assert_eq!(img.width(), frame_w);
    assert_eq!(img.height(), frame_h);
    let canvas = img.into_raw();

    let src_gen = BufferGenerator::builder(VideoFormat::RGBA, frame_w, frame_h)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(1)
        .max_buffers(1)
        .build()
        .expect("src generator");

    let src_shared = src_gen.acquire(Some(0)).unwrap();
    let view = SurfaceView::from_buffer(&src_shared, 0).unwrap();
    view.upload(&canvas, frame_w, frame_h, 4).expect("upload");
    drop(view);

    let batched_gen = UniformBatchGenerator::new(
        VideoFormat::RGBA,
        frame_w,
        frame_h,
        1,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .expect("batched generator");
    let config = common::platform_transform_config();
    let ids = vec![SavantIdMetaKind::Frame(0)];
    let mut batch = batched_gen.acquire_batch(config, ids).unwrap();
    let src_view = SurfaceView::from_buffer(&src_shared, 0).unwrap();
    batch.transform_slot(0, &src_view, None).unwrap();
    batch.finalize().unwrap();
    let shared = batch.shared_buffer();
    drop(batch);

    let output = engine.infer_sync(shared, None).expect("infer_sync");
    assert_eq!(
        output.num_elements(),
        1,
        "expected 1 element for full-frame"
    );

    let elem = &output.elements()[0];
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
    eprintln!(
        "  [{gt_key}] tensor shape: {raw_shape:?} -> {shape:?}, {} elements",
        data.len()
    );

    let tensors = [(&data[..], &shape[..])];
    let detections = converter.decode(&tensors).expect("decode failed");

    let total_dets: usize = detections.values().map(|v| v.len()).sum();
    eprintln!("  [{gt_key}] {total_dets} detections after converter");

    let scaler = CoordinateScaler::new(
        0.0,
        0.0,
        frame_w as f32,
        frame_h as f32,
        640.0,
        640.0,
        scaling,
    );

    let mut scaled: HashMap<usize, Vec<(f32, savant_core::primitives::RBBox)>> = HashMap::new();
    for (class_id, dets) in &detections {
        for (conf, bbox) in dets {
            let sb = scaler.scale_rbbox(bbox);
            scaled.entry(*class_id).or_default().push((*conf, sb));
        }
    }

    let (matched, total, mismatches) =
        match_detections(&scaled, gt_dets, IOU_THRESHOLD, 0.0, MIN_GT_CONF);

    eprintln!("  [{gt_key}] matched {matched}/{total} GT detections (min_conf={MIN_GT_CONF})");
    for m in &mismatches {
        eprintln!("    MISS: {m}");
    }

    let ratio = if total > 0 {
        matched as f32 / total as f32
    } else {
        1.0
    };
    assert!(
        ratio >= MIN_MATCH_RATIO,
        "[{gt_key}] only {matched}/{total} ({:.0}%) GT detections matched, need {:.0}%",
        ratio * 100.0,
        MIN_MATCH_RATIO * 100.0,
    );
}

fn run_yolo_test(scaling: ModelInputScaling) {
    common::init();

    let gt_path = assets_dir().join("yolo/ground_truth.json");
    if !gt_path.exists() {
        eprintln!("Skipping: ground_truth.json not found");
        return;
    }

    let engine = match build_yolo_engine(scaling) {
        Some(e) => e,
        None => return,
    };

    let gt = load_ground_truth();

    let converter = YoloDetectionConverter::new(
        YoloFormat::V8 { num_classes: 80 },
        CONF_THRESHOLD,
        HashMap::new(),
        None,
        NmsKind::ClassAgnostic {
            iou_threshold: NMS_IOU,
            top_k: NMS_TOP_K,
        },
    );

    for &(image_name, frame_w, frame_h) in &TEST_IMAGES {
        infer_one_image(
            &engine, &converter, scaling, image_name, frame_w, frame_h, &gt,
        );
    }
}

#[test]
#[serial]
fn test_yolo_fill() {
    run_yolo_test(ModelInputScaling::Fill);
}

#[test]
#[serial]
fn test_yolo_keep_aspect_ratio() {
    run_yolo_test(ModelInputScaling::KeepAspectRatio);
}

#[test]
#[serial]
fn test_yolo_keep_aspect_ratio_symmetric() {
    run_yolo_test(ModelInputScaling::KeepAspectRatioSymmetric);
}
