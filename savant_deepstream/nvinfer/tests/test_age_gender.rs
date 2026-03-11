//! Integration tests for NvInfer with the age_gender model (multi-output).

mod common;

use candle_core::{DType, Device, Tensor};
use deepstream_nvbufsurface::{
    DsNvSurfaceBufferGenerator, DsNvUniformSurfaceBufferGenerator, NvBufSurfaceMemType,
    TransformConfig, VideoFormat,
};
use nvinfer::{DataType, NvInfer, NvInferConfig, Roi};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use savant_core::primitives::RBBox;
use serde::Deserialize;
use serial_test::serial;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const FRAME_W: u32 = 1920;
const FRAME_H: u32 = 1080;
const FACE_SZ: u32 = 112;
const AGE_TOLERANCE: f32 = 15.0;

#[derive(Debug, Deserialize)]
struct GroundTruth {
    age: f64,
    gender: String,
}

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets")
}

/// Place `count` non-overlapping `w x h` rectangles on a `fw x fh` canvas.
/// Returns `(left, top)` for each placement. Panics if placement fails.
/// `gstnvinfer.cpp` applies `GST_ROUND_UP_2` to crop coordinates, shifting
/// odd left/top by 1 pixel and misaligning the crop window.  Snapping
/// placements to even pixels avoids this.
const ALIGN: u32 = 2;

fn place_non_overlapping(
    rng: &mut SmallRng,
    fw: u32,
    fh: u32,
    w: u32,
    h: u32,
    count: usize,
) -> Vec<(u32, u32)> {
    let mut placed: Vec<(u32, u32)> = Vec::with_capacity(count);
    let max_x = (fw - w) / ALIGN;
    let max_y = (fh - h) / ALIGN;
    for _ in 0..count {
        for attempt in 0..10_000 {
            let x = rng.random_range(0..=max_x) * ALIGN;
            let y = rng.random_range(0..=max_y) * ALIGN;
            let overlaps = placed
                .iter()
                .any(|&(px, py)| x < px + w && x + w > px && y < py + h && y + h > py);
            if !overlaps {
                placed.push((x, y));
                break;
            }
            assert!(attempt < 9_999, "failed to place image without overlap");
        }
    }
    placed
}

/// Build a candle tensor from a nvinfer TensorView, handling both fp16 and
/// fp32 output dtypes (nvinfer may output either depending on engine config).
fn to_candle_tensor(tv: &nvinfer::TensorView, shape: &[usize]) -> candle_core::Result<Tensor> {
    match tv.data_type {
        DataType::Half => {
            let raw: &[half::f16] = unsafe { tv.as_slice() };
            Tensor::from_slice(raw, shape, &Device::Cpu)?.to_dtype(DType::F32)
        }
        DataType::Float => {
            let raw: &[f32] = unsafe { tv.as_slice() };
            Tensor::from_slice(raw, shape, &Device::Cpu)
        }
        other => panic!("unsupported tensor dtype: {other:?}"),
    }
}

/// Decode age: weighted sum of 101 class probabilities.
fn decode_age(tensor: &nvinfer::TensorView) -> candle_core::Result<f32> {
    let probs = to_candle_tensor(tensor, &[101])?;
    let age_range = Tensor::arange(0f32, 101f32, &Device::Cpu)?;
    probs.mul(&age_range)?.sum_all()?.to_scalar::<f32>()
}

/// Decode gender: argmax over [male, female] logits.
fn decode_gender(tensor: &nvinfer::TensorView) -> candle_core::Result<String> {
    let t = to_candle_tensor(tensor, &[2])?;
    let idx = t.argmax(0)?.to_scalar::<u32>()?;
    Ok(if idx == 0 { "male" } else { "female" }.into())
}

// ---------------------------------------------------------------------------

fn make_age_gender_batch(num_frames: u32) -> gstreamer::Buffer {
    common::init();

    let src_gen = DsNvSurfaceBufferGenerator::builder(VideoFormat::RGBA, 112, 112)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(4)
        .max_buffers(4)
        .build()
        .expect("src generator");

    let batched_gen = DsNvUniformSurfaceBufferGenerator::new(
        VideoFormat::RGBA,
        112,
        112,
        16,
        2,
        0,
        NvBufSurfaceMemType::Default,
    )
    .expect("batched generator");

    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    for i in 0..num_frames {
        let src = src_gen.acquire_surface(Some(i as i64)).unwrap();
        batch.fill_slot(&src, None, Some(i as i64)).unwrap();
    }

    batch.finalize().unwrap();
    batch.as_gst_buffer().unwrap()
}

#[test]
#[serial]
fn test_multi_output_layer_names() {
    common::init();

    let onnx_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("assets/age_gender_mobilenet_v2_dynBatch.onnx");
    if !onnx_path.exists() {
        eprintln!("Skipping: age_gender model not found at {:?}", onnx_path);
        return;
    }

    let props = common::age_gender_properties();
    let config = NvInferConfig::new(props, "RGBA", 112, 112);
    let callback = Box::new(|_| {});
    let engine = NvInfer::new(config, callback).expect("create NvInfer");

    let batch = make_age_gender_batch(1);
    let output = engine.infer_sync(batch, 1, None).expect("infer_sync");

    assert_eq!(output.num_elements(), 1);
    let elem = &output.elements()[0];
    assert!(
        elem.tensors.len() >= 2,
        "age_gender model should have at least 2 output tensors, got {}",
        elem.tensors.len()
    );
}

// ---------------------------------------------------------------------------
// E2E test: real face images composited on a 1920x1080 frame
// ---------------------------------------------------------------------------

/// Build age_gender NvInfer engine with batch-size 32 (enough for 18 ROIs in
/// one pass) and frame dimensions 1920x1080.
fn age_gender_engine_1080p() -> Option<NvInfer> {
    let assets = assets_dir();
    if !assets
        .join("age_gender_mobilenet_v2_dynBatch.onnx")
        .exists()
    {
        eprintln!("Skipping: age_gender ONNX model not found");
        return None;
    }

    let mut props = common::age_gender_properties();
    props.insert("batch-size".into(), "32".into());
    props.insert(
        "model-engine-file".into(),
        assets
            .join("age_gender_mobilenet_v2_dynBatch.onnx_b32_gpu0_fp16.engine")
            .to_string_lossy()
            .into(),
    );

    let config = NvInferConfig::new(props, "RGBA", FRAME_W, FRAME_H);
    Some(NvInfer::new(config, Box::new(|_| {})).expect("create age_gender NvInfer 1080p"))
}

/// Load all face JPEGs from `assets/age_gender/`, sorted by filename.
/// Returns `(filename, RGBA pixels)` pairs.
fn load_face_images(dir: &Path) -> Vec<(String, Vec<u8>)> {
    let mut images: Vec<(String, Vec<u8>)> = Vec::new();
    for entry in std::fs::read_dir(dir).expect("read age_gender dir") {
        let entry = entry.unwrap();
        let fname = entry.file_name().to_string_lossy().to_string();
        if !fname.ends_with(".jpg") {
            continue;
        }
        let img = image::open(entry.path())
            .unwrap_or_else(|e| panic!("failed to open {fname}: {e}"))
            .to_rgba8();
        assert_eq!(img.width(), FACE_SZ, "{fname}: unexpected width");
        assert_eq!(img.height(), FACE_SZ, "{fname}: unexpected height");
        images.push((fname, img.into_raw()));
    }
    images.sort_by(|a, b| a.0.cmp(&b.0));
    images
}

#[test]
#[serial]
fn test_age_gender_e2e_real_images() {
    common::init();

    let assets = assets_dir();
    let gt_path = assets.join("age_gender/ground_truth.json");
    if !gt_path.exists() {
        eprintln!("Skipping: ground_truth.json not found (run generate_age_gender_gt.py)");
        return;
    }

    let engine = match age_gender_engine_1080p() {
        Some(e) => e,
        None => return,
    };

    // ---- Load ground truth ------------------------------------------------
    let gt_text = std::fs::read_to_string(&gt_path).expect("read ground_truth.json");
    let gt: HashMap<String, GroundTruth> =
        serde_json::from_str(&gt_text).expect("parse ground_truth.json");

    // ---- Load face images -------------------------------------------------
    let images = load_face_images(&assets.join("age_gender"));
    let num_faces = images.len();
    assert!(num_faces > 0, "no face images found");
    for (fname, _) in &images {
        assert!(gt.contains_key(fname), "no GT entry for {fname}");
    }

    // ---- Deterministic random placement -----------------------------------
    let mut rng = SmallRng::seed_from_u64(42);
    let placements = place_non_overlapping(&mut rng, FRAME_W, FRAME_H, FACE_SZ, FACE_SZ, num_faces);

    // ---- Build CPU composite canvas (RGBA, 1920x1080) ---------------------
    let stride = FRAME_W as usize * 4;
    let mut canvas = vec![0u8; stride * FRAME_H as usize];
    for ((_, rgba), &(x, y)) in images.iter().zip(&placements) {
        for row in 0..FACE_SZ as usize {
            let dst_off = (y as usize + row) * stride + x as usize * 4;
            let src_off = row * FACE_SZ as usize * 4;
            canvas[dst_off..dst_off + FACE_SZ as usize * 4]
                .copy_from_slice(&rgba[src_off..src_off + FACE_SZ as usize * 4]);
        }
    }

    // ---- Save composite PNG for sanity checking ---------------------------
    let tmp_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/tmp");
    std::fs::create_dir_all(&tmp_dir).expect("create target/tmp");
    image::save_buffer(
        tmp_dir.join("age_gender_e2e_composite.png"),
        &canvas,
        FRAME_W,
        FRAME_H,
        image::ColorType::Rgba8,
    )
    .expect("save composite PNG");
    eprintln!(
        "Composite saved to {}",
        tmp_dir.join("age_gender_e2e_composite.png").display()
    );

    // ---- Upload canvas to GPU surface -------------------------------------
    let src_gen = DsNvSurfaceBufferGenerator::builder(VideoFormat::RGBA, FRAME_W, FRAME_H)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(1)
        .max_buffers(1)
        .build()
        .expect("1080p src generator");

    let src_buf = src_gen.acquire_surface(Some(0)).unwrap();
    unsafe {
        deepstream_nvbufsurface::upload_to_surface(&src_buf, &canvas, FRAME_W, FRAME_H)
            .expect("upload_to_surface");
    }

    // ---- Create batched surface with one 1920x1080 slot -------------------
    let batched_gen = DsNvUniformSurfaceBufferGenerator::new(
        VideoFormat::RGBA,
        FRAME_W,
        FRAME_H,
        32,
        2,
        0,
        NvBufSurfaceMemType::Default,
    )
    .expect("1080p batched generator");

    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();
    batch.fill_slot(&src_buf, None, Some(0)).unwrap();
    batch.finalize().unwrap();
    let gst_buffer = batch.as_gst_buffer().unwrap();

    // ---- Build ROIs -------------------------------------------------------
    let roi_vec: Vec<Roi> = placements
        .iter()
        .enumerate()
        .map(|(i, &(x, y))| Roi {
            id: i as i64,
            bbox: RBBox::ltwh(x as f32, y as f32, FACE_SZ as f32, FACE_SZ as f32).unwrap(),
        })
        .collect();
    let rois: HashMap<u32, Vec<Roi>> = [(0u32, roi_vec)].into();

    // ---- Run inference ----------------------------------------------------
    let output = engine
        .infer_sync(gst_buffer, 1, Some(&rois))
        .expect("infer_sync");

    assert_eq!(output.batch_id(), 1);
    assert_eq!(
        output.num_elements(),
        num_faces,
        "expected one output per ROI"
    );

    // ---- Validate each ROI against ground truth ---------------------------
    let mut pass_count = 0usize;
    for (i, elem) in output.elements().iter().enumerate() {
        let fname = &images[i].0;
        let expected = &gt[fname];

        assert_eq!(elem.roi_id, Some(i as i64), "{fname}: roi_id mismatch");
        assert!(
            elem.tensors.len() >= 2,
            "{fname}: expected >= 2 output tensors, got {}",
            elem.tensors.len()
        );

        let age_tensor = elem
            .tensors
            .iter()
            .find(|t| t.name == "age")
            .unwrap_or_else(|| panic!("{fname}: missing 'age' tensor"));
        let gender_tensor = elem
            .tensors
            .iter()
            .find(|t| t.name == "gender")
            .unwrap_or_else(|| panic!("{fname}: missing 'gender' tensor"));

        let trt_age = decode_age(age_tensor).expect("decode age");
        let trt_gender = decode_gender(gender_tensor).expect("decode gender");

        let age_diff = (trt_age as f64 - expected.age).abs();
        eprintln!(
            "  {fname}: TRT age={trt_age:.1} gender={trt_gender}  |  \
             GT age={:.1} gender={}  |  age_diff={age_diff:.1}",
            expected.age, expected.gender
        );

        assert!(
            age_diff < AGE_TOLERANCE as f64,
            "{fname}: age diff {age_diff:.1} exceeds tolerance {AGE_TOLERANCE} \
             (TRT={trt_age:.2}, GT={:.2})",
            expected.age
        );
        assert_eq!(
            trt_gender, expected.gender,
            "{fname}: gender mismatch (TRT={trt_gender}, GT={})",
            expected.gender
        );
        pass_count += 1;
    }
    eprintln!("\n  All {pass_count}/{num_faces} faces passed age/gender validation.");
}

/// Run inference with two different random seeds (different placements, both
/// even-aligned) and verify that per-face TRT ages are bit-identical.
#[test]
#[serial]
fn test_age_gender_placement_independence() {
    common::init();

    let assets = assets_dir();
    let gt_path = assets.join("age_gender/ground_truth.json");
    if !gt_path.exists() {
        eprintln!("Skipping: ground_truth.json not found");
        return;
    }
    let engine = match age_gender_engine_1080p() {
        Some(e) => e,
        None => return,
    };

    let gt_text = std::fs::read_to_string(&gt_path).expect("read ground_truth.json");
    let gt: HashMap<String, GroundTruth> =
        serde_json::from_str(&gt_text).expect("parse ground_truth.json");
    let images = load_face_images(&assets.join("age_gender"));
    let num_faces = images.len();

    let run = |seed: u64, batch_id: u64| -> Vec<f32> {
        let mut rng = SmallRng::seed_from_u64(seed);
        let placements =
            place_non_overlapping(&mut rng, FRAME_W, FRAME_H, FACE_SZ, FACE_SZ, num_faces);

        let stride = FRAME_W as usize * 4;
        let mut canvas = vec![0u8; stride * FRAME_H as usize];
        for ((_, rgba), &(x, y)) in images.iter().zip(&placements) {
            for row in 0..FACE_SZ as usize {
                let dst_off = (y as usize + row) * stride + x as usize * 4;
                let src_off = row * FACE_SZ as usize * 4;
                canvas[dst_off..dst_off + FACE_SZ as usize * 4]
                    .copy_from_slice(&rgba[src_off..src_off + FACE_SZ as usize * 4]);
            }
        }

        let src_gen = DsNvSurfaceBufferGenerator::builder(VideoFormat::RGBA, FRAME_W, FRAME_H)
            .gpu_id(0)
            .mem_type(NvBufSurfaceMemType::Default)
            .min_buffers(1)
            .max_buffers(1)
            .build()
            .expect("src generator");

        let src_buf = src_gen.acquire_surface(Some(0)).unwrap();
        unsafe {
            deepstream_nvbufsurface::upload_to_surface(&src_buf, &canvas, FRAME_W, FRAME_H)
                .expect("upload_to_surface");
        }

        let batched_gen = DsNvUniformSurfaceBufferGenerator::new(
            VideoFormat::RGBA,
            FRAME_W,
            FRAME_H,
            32,
            2,
            0,
            NvBufSurfaceMemType::Default,
        )
        .expect("batched generator");
        let config = TransformConfig::default();
        let mut batch = batched_gen.acquire_batched_surface(config).unwrap();
        batch.fill_slot(&src_buf, None, Some(0)).unwrap();
        batch.finalize().unwrap();
        let gst_buffer = batch.as_gst_buffer().unwrap();

        let roi_vec: Vec<Roi> = placements
            .iter()
            .enumerate()
            .map(|(i, &(x, y))| Roi {
                id: i as i64,
                bbox: RBBox::ltwh(x as f32, y as f32, FACE_SZ as f32, FACE_SZ as f32).unwrap(),
            })
            .collect();
        let rois: HashMap<u32, Vec<Roi>> = [(0u32, roi_vec)].into();

        let output = engine
            .infer_sync(gst_buffer, batch_id, Some(&rois))
            .expect("infer_sync");

        output
            .elements()
            .iter()
            .map(|elem| {
                let age_tensor = elem.tensors.iter().find(|t| t.name == "age").unwrap();
                decode_age(age_tensor).unwrap()
            })
            .collect()
    };

    let ages_seed42 = run(42, 1);
    let ages_seed99 = run(99, 2);

    eprintln!(
        "\n  {:>16}  {:>7}  {:>7}  {:>7}  {:>6}",
        "file", "GT", "s=42", "s=99", "delta"
    );
    eprintln!("  {}", "-".repeat(58));
    for (i, (fname, _)) in images.iter().enumerate() {
        let g = gt[fname].age;
        let a1 = ages_seed42[i];
        let a2 = ages_seed99[i];
        let delta = (a1 - a2).abs();
        eprintln!("  {fname:>16}  {g:7.2}  {a1:7.2}  {a2:7.2}  {delta:6.4}",);
        assert!(
            delta < 1e-4,
            "{fname}: ages differ between seeds (s42={a1:.4}, s99={a2:.4}, delta={delta:.6})"
        );
    }
    eprintln!("\n  All {num_faces} faces: bit-identical across seeds.");
}
