//! Pixel-level crop fidelity tests using the identity_3x112x112 model.
//!
//! These tests replicate the same composite canvas + random ROI placement
//! pattern from `test_age_gender.rs` but substitute the identity model
//! (output == input).  By comparing the output tensor against the known
//! face image pixels we verify that the NvInfer crop pipeline delivers
//! correct data to the model regardless of placement position.

mod common;

use deepstream_buffers::{
    BufferGenerator, NvBufSurfaceMemType, SavantIdMetaKind, SharedBuffer, SurfaceView,
    TransformConfig, UniformBatchGenerator, VideoFormat,
};
use nvinfer::{NvInfer, NvInferConfig, Roi};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use savant_core::primitives::RBBox;
use serial_test::serial;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const FRAME_W: u32 = 1920;
const FRAME_H: u32 = 1080;
const FACE_SZ: u32 = 112;
const ALIGN: u32 = 2;

/// Per-pixel tolerance for the fp16 identity round-trip.
const PIXEL_TOL: f32 = 2.0;

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets")
}

// ---------------------------------------------------------------------------
// Helpers (same logic as test_age_gender.rs)
// ---------------------------------------------------------------------------

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

/// Convert RGBA face pixels to the expected CHW float32 RGB tensor that the
/// identity model should output (net-scale-factor=1.0, no offsets).
fn face_rgba_to_expected_rgb(rgba: &[u8], w: u32, h: u32) -> Vec<f32> {
    let npix = (w * h) as usize;
    let mut expected = vec![0.0f32; 3 * npix];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let px = y * w as usize + x;
            for c in 0..3usize {
                expected[c * npix + px] = rgba[px * 4 + c] as f32;
            }
        }
    }
    expected
}

fn identity_engine_1080p() -> Option<NvInfer> {
    let assets = assets_dir();
    if !assets.join("identity_3x112x112.onnx").exists() {
        eprintln!("Skipping: identity_3x112x112.onnx not found");
        return None;
    }

    let props = common::identity_112x112_properties();

    let config = NvInferConfig::new(props, "RGBA", FRAME_W, FRAME_H);
    let engine = NvInfer::new(config, Box::new(|_| {})).expect("create identity NvInfer 1080p");
    common::promote_built_engine("identity_3x112x112.onnx", 32);
    Some(engine)
}

// ---------------------------------------------------------------------------
// Composite builder: place face images on a 1920x1080 RGBA canvas, upload
// to GPU, wrap in a batched surface, and build matching ROIs.
// ---------------------------------------------------------------------------

struct CompositeResult {
    shared: SharedBuffer,
    rois: HashMap<u32, Vec<Roi>>,
}

fn build_composite(images: &[(String, Vec<u8>)], seed: u64) -> CompositeResult {
    let num_faces = images.len();
    let mut rng = SmallRng::seed_from_u64(seed);
    let placements = place_non_overlapping(&mut rng, FRAME_W, FRAME_H, FACE_SZ, FACE_SZ, num_faces);

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

    let src_gen = BufferGenerator::builder(VideoFormat::RGBA, FRAME_W, FRAME_H)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(1)
        .max_buffers(1)
        .build()
        .expect("1080p src generator");

    let src_shared = src_gen.acquire(Some(0)).unwrap();
    let view = SurfaceView::from_buffer(&src_shared, 0).unwrap();
    view.upload(&canvas, FRAME_W, FRAME_H, 4)
        .expect("upload_to_surface");
    drop(view);

    let batched_gen = UniformBatchGenerator::new(
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
    let ids = vec![SavantIdMetaKind::Frame(0)];
    let mut batch = batched_gen.acquire_batch(config, ids).unwrap();
    let src_view = SurfaceView::from_buffer(&src_shared, 0).unwrap();
    batch.transform_slot(0, &src_view, None).unwrap();
    batch.finalize().unwrap();
    let shared = batch.shared_buffer();

    let roi_vec: Vec<Roi> = placements
        .iter()
        .enumerate()
        .map(|(i, &(x, y))| Roi {
            id: i as i64,
            bbox: RBBox::ltwh(x as f32, y as f32, FACE_SZ as f32, FACE_SZ as f32).unwrap(),
        })
        .collect();
    let rois: HashMap<u32, Vec<Roi>> = [(0u32, roi_vec)].into();

    CompositeResult { shared, rois }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Verify that the identity model output matches the expected face pixels
/// for every ROI when placed with seed=42 (same seed as the failing
/// age_gender E2E test).
#[test]
#[serial]
fn test_roi_crop_pixel_match() {
    common::init();

    let assets = assets_dir();
    let face_dir = assets.join("age_gender");
    if !face_dir.exists() {
        eprintln!("Skipping: age_gender face directory not found");
        return;
    }

    let engine = match identity_engine_1080p() {
        Some(e) => e,
        None => return,
    };
    common::warmup_engine(&engine, FRAME_W, FRAME_H);

    let images = load_face_images(&face_dir);
    let num_faces = images.len();
    assert!(num_faces > 0, "no face images found");

    let mut rng_diag = SmallRng::seed_from_u64(42);
    let placements =
        place_non_overlapping(&mut rng_diag, FRAME_W, FRAME_H, FACE_SZ, FACE_SZ, num_faces);

    let comp = build_composite(&images, 42);
    let output = engine
        .infer_sync(comp.shared, Some(&comp.rois))
        .expect("infer_sync");

    assert_eq!(
        output.num_elements(),
        num_faces,
        "expected one output per ROI"
    );

    let npix = (FACE_SZ * FACE_SZ) as usize;
    let mut any_failed = false;

    eprintln!(
        "\n  {:>16}  {:>9}  {:>9}  {:>9}  {:>14}",
        "file", "max_err", "mean_err", "mismatches", "position"
    );
    eprintln!("  {}", "-".repeat(66));

    for (i, elem) in output.elements().iter().enumerate() {
        let fname = &images[i].0;
        let rgba = &images[i].1;

        assert_eq!(elem.roi_id, Some(i as i64), "{fname}: roi_id mismatch");
        assert!(!elem.tensors.is_empty(), "{fname}: no output tensors");

        let t = &elem.tensors[0];
        assert_eq!(
            t.dims.num_elements as usize,
            3 * npix,
            "{fname}: unexpected tensor size"
        );

        let actual: &[f32] = unsafe { t.as_slice() };
        let expected = face_rgba_to_expected_rgb(rgba, FACE_SZ, FACE_SZ);

        let mut max_err: f32 = 0.0;
        let mut sum_err: f64 = 0.0;
        let mut mismatch_count = 0usize;

        for j in 0..actual.len() {
            let err = (actual[j] - expected[j]).abs();
            if err > max_err {
                max_err = err;
            }
            sum_err += err as f64;
            if err > PIXEL_TOL {
                mismatch_count += 1;
            }
        }
        let mean_err = sum_err / actual.len() as f64;

        let (px, py) = placements[i];
        eprintln!(
            "  {fname:>16}  {max_err:9.2}  {mean_err:9.4}  {mismatch_count:>9}  ({px:>4},{py:>4})"
        );

        if max_err > PIXEL_TOL {
            eprintln!(
                "    ** FAIL: {fname} max pixel error {max_err:.2} exceeds tolerance {PIXEL_TOL}"
            );
            any_failed = true;
        }
    }

    assert!(
        !any_failed,
        "one or more ROIs exceeded pixel tolerance {PIXEL_TOL} — see table above"
    );
    eprintln!("\n  All {num_faces} ROIs passed pixel-level fidelity check.");
}

/// Run identity inference with two different seeds and verify that the
/// per-face output tensors are identical (within fp16 tolerance) regardless
/// of placement position.  Also compare each against the expected CPU
/// reference pixels.
#[test]
#[serial]
fn test_roi_crop_placement_independence() {
    common::init();

    let assets = assets_dir();
    let face_dir = assets.join("age_gender");
    if !face_dir.exists() {
        eprintln!("Skipping: age_gender face directory not found");
        return;
    }

    let engine = match identity_engine_1080p() {
        Some(e) => e,
        None => return,
    };
    common::warmup_engine(&engine, FRAME_W, FRAME_H);

    let images = load_face_images(&face_dir);
    let num_faces = images.len();
    assert!(num_faces > 0, "no face images found");

    let run = |seed: u64| -> (Vec<Vec<f32>>, Vec<(u32, u32)>) {
        let mut rng = SmallRng::seed_from_u64(seed);
        let placements =
            place_non_overlapping(&mut rng, FRAME_W, FRAME_H, FACE_SZ, FACE_SZ, num_faces);
        let comp = build_composite(&images, seed);
        let output = engine
            .infer_sync(comp.shared, Some(&comp.rois))
            .expect("infer_sync");
        assert_eq!(output.num_elements(), num_faces);
        let tensors: Vec<Vec<f32>> = output
            .elements()
            .iter()
            .map(|elem| {
                let t = &elem.tensors[0];
                let slice: &[f32] = unsafe { t.as_slice() };
                slice.to_vec()
            })
            .collect();
        (tensors, placements)
    };

    let (tensors_s42, placements_s42) = run(42);
    let (tensors_s99, placements_s99) = run(99);

    eprintln!(
        "\n  {:>16}  {:>10}  {:>10}  {:>12}  {:>14}  {:>14}",
        "file", "max_s42", "max_s99", "cross_delta", "pos_s42", "pos_s99"
    );
    eprintln!("  {}", "-".repeat(82));

    let mut any_failed = false;

    for (i, (fname, rgba)) in images.iter().enumerate() {
        let expected = face_rgba_to_expected_rgb(rgba, FACE_SZ, FACE_SZ);

        let max_err_s42 = tensors_s42[i]
            .iter()
            .zip(&expected)
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max);

        let max_err_s99 = tensors_s99[i]
            .iter()
            .zip(&expected)
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max);

        let cross_delta = tensors_s42[i]
            .iter()
            .zip(&tensors_s99[i])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        let (x42, y42) = placements_s42[i];
        let (x99, y99) = placements_s99[i];
        eprintln!(
            "  {fname:>16}  {max_err_s42:10.2}  {max_err_s99:10.2}  {cross_delta:12.4}  ({x42:>4},{y42:>4})  ({x99:>4},{y99:>4})"
        );

        if cross_delta > PIXEL_TOL {
            eprintln!(
                "    ** FAIL: {fname} cross-seed delta {cross_delta:.4} exceeds tolerance {PIXEL_TOL}"
            );
            any_failed = true;
        }
        if max_err_s42 > PIXEL_TOL {
            eprintln!(
                "    ** FAIL: {fname} seed=42 max error {max_err_s42:.2} exceeds tolerance {PIXEL_TOL}"
            );
            any_failed = true;
        }
        if max_err_s99 > PIXEL_TOL {
            eprintln!(
                "    ** FAIL: {fname} seed=99 max error {max_err_s99:.2} exceeds tolerance {PIXEL_TOL}"
            );
            any_failed = true;
        }
    }

    assert!(
        !any_failed,
        "one or more ROIs exceeded tolerance — see table above"
    );
    eprintln!(
        "\n  All {num_faces} faces: pixel-identical across seeds within tolerance {PIXEL_TOL}."
    );
}
