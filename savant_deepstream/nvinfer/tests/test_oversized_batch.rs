//! Oversized batch: more frames/ROIs than nvinfer `batch-size`.
//!
//! With `batch-size=1` and two frames × four ROIs each, DeepStream's
//! `gstnvinfer` must run multiple internal engine passes. These tests assert
//! all outputs arrive with correct age/gender vs ground truth.

mod common;

use common::age_gender_test_utils::{
    decode_age, decode_gender, load_face_images, place_non_overlapping,
};
use deepstream_buffers::{
    BufferGenerator, NonUniformBatch, NvBufSurfaceMemType, SavantIdMetaKind, SharedBuffer,
    SurfaceView, UniformBatchGenerator, VideoFormat,
};
use nvinfer::{BatchInferenceOutput, ModelColorFormat, NvInfer, NvInferConfig, Roi};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use savant_core::primitives::RBBox;
use serde::Deserialize;
use serial_test::serial;
use std::collections::HashMap;
use std::path::PathBuf;

const FRAME_W: u32 = 1920;
const FRAME_H: u32 = 1080;
const FRAME0_W: u32 = 800;
const FRAME0_H: u32 = 600;
const FACE_SZ: u32 = 112;
const AGE_TOLERANCE: f32 = 15.0;
const ROIS_PER_FRAME: usize = 4;
const NUM_FRAMES: u32 = 2;
const TOTAL_ROIS: usize = ROIS_PER_FRAME * NUM_FRAMES as usize;

#[derive(Debug, Deserialize)]
struct GroundTruth {
    age: f64,
    gender: String,
}

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets")
}

/// Composite `face_rgba` slices onto an `fw x fh` RGBA canvas at `placements`.
fn build_frame_canvas(
    faces: &[(String, Vec<u8>)],
    placements: &[(u32, u32)],
    fw: u32,
    fh: u32,
) -> Vec<u8> {
    assert_eq!(faces.len(), placements.len());
    let stride = fw as usize * 4;
    let mut canvas = vec![0u8; stride * fh as usize];
    for ((_, rgba), &(x, y)) in faces.iter().zip(placements) {
        for row in 0..FACE_SZ as usize {
            let dst_off = (y as usize + row) * stride + x as usize * 4;
            let src_off = row * FACE_SZ as usize * 4;
            canvas[dst_off..dst_off + FACE_SZ as usize * 4]
                .copy_from_slice(&rgba[src_off..src_off + FACE_SZ as usize * 4]);
        }
    }
    canvas
}

fn build_age_gender_engine_bs1() -> Option<NvInfer> {
    let onnx = assets_dir().join("age_gender_mobilenet_v2_dynBatch.onnx");
    if !onnx.exists() {
        eprintln!("Skipping: age_gender ONNX not found at {:?}", onnx);
        return None;
    }
    let props = common::age_gender_properties_bs1();
    let config = NvInferConfig::new(props, VideoFormat::RGBA, 112, 112, ModelColorFormat::RGB);
    let engine = NvInfer::new(config, Box::new(|_| {})).expect("create NvInfer bs1");
    common::promote_built_engine("age_gender_mobilenet_v2_dynBatch.onnx", 1);
    Some(engine)
}

fn build_age_gender_engine_bs1_flexible() -> Option<NvInfer> {
    let onnx = assets_dir().join("age_gender_mobilenet_v2_dynBatch.onnx");
    if !onnx.exists() {
        eprintln!("Skipping: age_gender ONNX not found at {:?}", onnx);
        return None;
    }
    let props = common::age_gender_properties_bs1();
    let config = NvInferConfig::new(props, VideoFormat::RGBA, 112, 112, ModelColorFormat::RGB);
    let engine = NvInfer::new(config, Box::new(|_| {})).expect("create NvInfer bs1");
    common::promote_built_engine("age_gender_mobilenet_v2_dynBatch.onnx", 1);
    Some(engine)
}

fn rois_from_placements(placements: &[(u32, u32)], id_offset: i64) -> Vec<Roi> {
    placements
        .iter()
        .enumerate()
        .map(|(i, &(x, y))| Roi {
            id: id_offset + i as i64,
            bbox: RBBox::ltwh(x as f32, y as f32, FACE_SZ as f32, FACE_SZ as f32).unwrap(),
        })
        .collect()
}

fn validate_oversized_output(
    output: &BatchInferenceOutput,
    savant_in: &[SavantIdMetaKind],
    images: &[(String, Vec<u8>)],
    gt: &HashMap<String, GroundTruth>,
) {
    assert_eq!(
        output.buffer().savant_ids(),
        savant_in,
        "SavantIdMeta must propagate through nvinfer"
    );
    assert_eq!(
        output.num_elements(),
        TOTAL_ROIS,
        "expected one ElementOutput per ROI"
    );

    let mut slot0 = 0u32;
    let mut slot1 = 0u32;
    for elem in output.elements() {
        match elem.slot_number {
            0 => slot0 += 1,
            1 => slot1 += 1,
            s => panic!("unexpected slot_number {s}"),
        }
    }
    assert_eq!(slot0, ROIS_PER_FRAME as u32, "slot 0 ROI count");
    assert_eq!(slot1, ROIS_PER_FRAME as u32, "slot 1 ROI count");

    for elem in output.elements() {
        let rid = elem.roi_id.expect("ROI path must set roi_id") as usize;
        assert!(rid < TOTAL_ROIS, "roi_id out of range: {rid}");
        let fname = &images[rid].0;
        let expected = gt.get(fname).unwrap_or_else(|| panic!("no GT for {fname}"));

        let age_tensor = elem
            .tensors
            .iter()
            .find(|t| t.name == "age")
            .unwrap_or_else(|| panic!("{fname}: missing age tensor"));
        let gender_tensor = elem
            .tensors
            .iter()
            .find(|t| t.name == "gender")
            .unwrap_or_else(|| panic!("{fname}: missing gender tensor"));

        let trt_age = decode_age(age_tensor).expect("decode age");
        let trt_gender = decode_gender(gender_tensor).expect("decode gender");
        let age_diff = (trt_age as f64 - expected.age).abs();
        assert!(
            age_diff < AGE_TOLERANCE as f64,
            "{fname}: age diff {age_diff:.1} exceeds tolerance (TRT={trt_age:.2}, GT={:.2})",
            expected.age
        );
        assert_eq!(trt_gender, expected.gender, "{fname}: gender mismatch");
    }
}

#[test]
#[serial]
fn test_oversized_uniform_batch() {
    common::init();

    let assets = assets_dir();
    let gt_path = assets.join("age_gender/ground_truth.json");
    if !gt_path.exists() {
        eprintln!("Skipping: ground_truth.json not found");
        return;
    }

    let engine = match build_age_gender_engine_bs1() {
        Some(e) => e,
        None => return,
    };
    common::warmup_engine(&engine, FRAME_W, FRAME_H);

    let gt_text = std::fs::read_to_string(&gt_path).expect("read ground_truth.json");
    let gt: HashMap<String, GroundTruth> =
        serde_json::from_str(&gt_text).expect("parse ground_truth.json");

    let images = load_face_images(&assets.join("age_gender"), FACE_SZ, FACE_SZ);
    if images.len() < TOTAL_ROIS {
        eprintln!(
            "Skipping: need at least {TOTAL_ROIS} face JPGs, found {}",
            images.len()
        );
        return;
    }
    let images: Vec<_> = images.into_iter().take(TOTAL_ROIS).collect();
    for (fname, _) in &images {
        assert!(gt.contains_key(fname), "no GT entry for {fname}");
    }

    let mut rng = SmallRng::seed_from_u64(2024);
    let p0 = place_non_overlapping(&mut rng, FRAME_W, FRAME_H, FACE_SZ, FACE_SZ, ROIS_PER_FRAME);
    let p1 = place_non_overlapping(&mut rng, FRAME_W, FRAME_H, FACE_SZ, FACE_SZ, ROIS_PER_FRAME);

    let canvas0 = build_frame_canvas(&images[0..ROIS_PER_FRAME], &p0, FRAME_W, FRAME_H);
    let canvas1 = build_frame_canvas(&images[ROIS_PER_FRAME..TOTAL_ROIS], &p1, FRAME_W, FRAME_H);

    let src_gen = BufferGenerator::builder(VideoFormat::RGBA, FRAME_W, FRAME_H)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(2)
        .max_buffers(2)
        .build()
        .expect("src generator");

    let batched_gen = UniformBatchGenerator::new(
        VideoFormat::RGBA,
        FRAME_W,
        FRAME_H,
        NUM_FRAMES,
        2,
        0,
        NvBufSurfaceMemType::Default,
    )
    .expect("batched generator");

    let config = common::platform_transform_config();
    let savant_in = vec![SavantIdMetaKind::Frame(701), SavantIdMetaKind::Frame(702)];
    let savant_clone = savant_in.clone();

    let mut batch = batched_gen.acquire_batch(config, savant_in).unwrap();

    for (slot, canvas) in [(0u32, &canvas0), (1, &canvas1)] {
        let src = src_gen.acquire(Some(slot as i64)).unwrap();
        src.with_view(0, |view| {
            view.upload(canvas, FRAME_W, FRAME_H, 4)?;
            batch.transform_slot(slot, view, None)
        })
        .unwrap();
    }

    batch.finalize().unwrap();
    let shared = batch.into_shared_buffer();

    let rois: HashMap<u32, Vec<Roi>> = [
        (0u32, rois_from_placements(&p0, 0)),
        (1u32, rois_from_placements(&p1, ROIS_PER_FRAME as i64)),
    ]
    .into();

    let output = engine
        .infer_sync(shared, Some(&rois))
        .expect("infer_sync oversized uniform");

    validate_oversized_output(&output, &savant_clone, &images, &gt);
}

#[test]
#[serial]
fn test_oversized_nonuniform_batch() {
    common::init();

    let assets = assets_dir();
    let gt_path = assets.join("age_gender/ground_truth.json");
    if !gt_path.exists() {
        eprintln!("Skipping: ground_truth.json not found");
        return;
    }

    let engine = match build_age_gender_engine_bs1_flexible() {
        Some(e) => e,
        None => return,
    };
    common::warmup_engine(&engine, FRAME_W, FRAME_H);

    let gt_text = std::fs::read_to_string(&gt_path).expect("read ground_truth.json");
    let gt: HashMap<String, GroundTruth> =
        serde_json::from_str(&gt_text).expect("parse ground_truth.json");

    let images = load_face_images(&assets.join("age_gender"), FACE_SZ, FACE_SZ);
    if images.len() < TOTAL_ROIS {
        eprintln!(
            "Skipping: need at least {TOTAL_ROIS} face JPGs, found {}",
            images.len()
        );
        return;
    }
    let images: Vec<_> = images.into_iter().take(TOTAL_ROIS).collect();
    for (fname, _) in &images {
        assert!(gt.contains_key(fname), "no GT entry for {fname}");
    }

    let mut rng = SmallRng::seed_from_u64(9001);
    let p0 = place_non_overlapping(
        &mut rng,
        FRAME0_W,
        FRAME0_H,
        FACE_SZ,
        FACE_SZ,
        ROIS_PER_FRAME,
    );
    let p1 = place_non_overlapping(&mut rng, FRAME_W, FRAME_H, FACE_SZ, FACE_SZ, ROIS_PER_FRAME);

    let canvas0 = build_frame_canvas(&images[0..ROIS_PER_FRAME], &p0, FRAME0_W, FRAME0_H);
    let canvas1 = build_frame_canvas(&images[ROIS_PER_FRAME..TOTAL_ROIS], &p1, FRAME_W, FRAME_H);

    let gen0 = BufferGenerator::builder(VideoFormat::RGBA, FRAME0_W, FRAME0_H)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(1)
        .max_buffers(1)
        .build()
        .expect("gen slot0");
    let gen1 = BufferGenerator::builder(VideoFormat::RGBA, FRAME_W, FRAME_H)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(1)
        .max_buffers(1)
        .build()
        .expect("gen slot1");

    let shared: SharedBuffer = {
        let buf0 = gen0.acquire(Some(0)).unwrap();
        let view0 = SurfaceView::from_buffer(&buf0, 0).unwrap();
        view0
            .upload(&canvas0, FRAME0_W, FRAME0_H, 4)
            .expect("upload slot0");

        let buf1 = gen1.acquire(Some(1)).unwrap();
        let view1 = SurfaceView::from_buffer(&buf1, 0).unwrap();
        view1
            .upload(&canvas1, FRAME_W, FRAME_H, 4)
            .expect("upload slot1");

        let savant_in = vec![SavantIdMetaKind::Frame(801), SavantIdMetaKind::Frame(802)];
        let mut batch = NonUniformBatch::new(0);
        batch.add(&view0).expect("add slot0");
        batch.add(&view1).expect("add slot1");
        batch.finalize(savant_in).expect("finalize nonuniform")
    };

    let savant_clone = shared.savant_ids().to_vec();

    let rois: HashMap<u32, Vec<Roi>> = [
        (0u32, rois_from_placements(&p0, 0)),
        (1u32, rois_from_placements(&p1, ROIS_PER_FRAME as i64)),
    ]
    .into();

    let output = engine
        .infer_sync(shared, Some(&rois))
        .expect("infer_sync oversized nonuniform");

    validate_oversized_output(&output, &savant_clone, &images, &gt);
}
