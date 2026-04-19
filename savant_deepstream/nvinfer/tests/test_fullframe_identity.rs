//! Full-frame identity test: pass an entire 1920x1080 random-filled canvas
//! through a FullHD identity model and compare every pixel of output vs input.
//!
//! Saves a heatmap PNG highlighting regions where the surface memory does not
//! round-trip correctly.  This directly diagnoses NvBufSurface layout issues
//! without any ROI-crop complications.

mod common;

use deepstream_buffers::{
    BufferGenerator, NvBufSurfaceMemType, SavantIdMetaKind, SharedBuffer, SurfaceView,
    TransformConfig, UniformBatchGenerator, VideoFormat,
};
use deepstream_nvinfer::{NvInfer, Roi};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use savant_core::primitives::RBBox;
use serial_test::serial;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const FRAME_W: u32 = 1920;
const FRAME_H: u32 = 1080;
const PIXEL_TOL: f32 = 2.0;

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets")
}

fn tmp_dir() -> PathBuf {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/tmp");
    std::fs::create_dir_all(&d).expect("create target/tmp");
    d
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Fill a 1920x1080 RGBA canvas with deterministic random pixel values.
fn build_random_canvas(seed: u64) -> Vec<u8> {
    let total = (FRAME_W * FRAME_H * 4) as usize;
    let mut canvas = vec![0u8; total];
    let mut rng = SmallRng::seed_from_u64(seed);
    rng.fill(canvas.as_mut_slice());
    canvas
}

fn identity_engine_fullhd() -> Option<NvInfer> {
    let assets = assets_dir();
    if !assets.join("identity_fullhd.onnx").exists() {
        eprintln!("Skipping: identity_fullhd.onnx not found");
        return None;
    }
    let props = common::identity_fullhd_properties();
    let config = deepstream_nvinfer::NvInferConfig::new(
        props,
        VideoFormat::RGBA,
        1920,
        1080,
        deepstream_nvinfer::ModelColorFormat::RGB,
    );
    let engine = NvInfer::new(config).expect("create identity FullHD NvInfer");
    common::promote_built_engine("identity_fullhd.onnx", 2);
    Some(engine)
}

/// Dump NvBufSurfaceParams from a buffer for diagnostics.
unsafe fn dump_surface_params(label: &str, buf: &gstreamer::BufferRef) {
    let surf = deepstream_buffers::extract_nvbufsurface(buf).expect("extract_nvbufsurface");
    let s = &*surf;
    eprintln!(
        "  [{label}] NvBufSurface: gpuId={}, batchSize={}, numFilled={}",
        s.gpuId, s.batchSize, s.numFilled
    );
    for i in 0..s.numFilled {
        let p = &*s.surfaceList.add(i as usize);
        eprintln!(
            "    slot {i}: width={}, height={}, pitch={}, colorFormat={}",
            p.width, p.height, p.pitch, p.colorFormat
        );
        let pp = &p.planeParams;
        eprintln!(
            "    planeParams: num_planes={}, pitch={:?}, height={:?}, width={:?}, offset={:?}",
            pp.num_planes,
            &pp.pitch[..pp.num_planes as usize],
            &pp.height[..pp.num_planes as usize],
            &pp.width[..pp.num_planes as usize],
            &pp.offset[..pp.num_planes as usize],
        );
    }
}

/// Upload `canvas` to a GPU surface and wrap in a batched buffer with a single
/// full-frame ROI.
fn canvas_to_batch_with_fullframe_roi(canvas: &[u8]) -> (SharedBuffer, HashMap<u32, Vec<Roi>>) {
    let src_gen = BufferGenerator::builder(VideoFormat::RGBA, FRAME_W, FRAME_H)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(1)
        .max_buffers(1)
        .build()
        .expect("1080p src generator");

    let src_shared = src_gen.acquire(Some(0)).unwrap();

    // Dump source surface params before upload.
    {
        let guard = src_shared.lock();
        unsafe { dump_surface_params("source (pre-upload)", guard.as_ref()) };
    }

    src_shared
        .with_view(0, |view| {
            eprintln!(
                "  [SurfaceView] width={}, height={}, pitch={}",
                view.width(),
                view.height(),
                view.pitch()
            );
            view.upload(canvas, FRAME_W, FRAME_H, 4)
        })
        .expect("upload_to_surface");

    let batched_gen = UniformBatchGenerator::new(
        VideoFormat::RGBA,
        FRAME_W,
        FRAME_H,
        2,
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

    let shared = batch.into_shared_buffer();
    {
        let guard = shared.lock();
        unsafe { dump_surface_params("batched (post-fill+finalize)", guard.as_ref()) };
    }

    let roi = Roi {
        id: 0,
        bbox: RBBox::ltwh(0.0, 0.0, FRAME_W as f32, FRAME_H as f32).unwrap(),
    };
    let rois: HashMap<u32, Vec<Roi>> = [(0u32, vec![roi])].into();

    (shared, rois)
}

// ---------------------------------------------------------------------------
// Comparison utilities
// ---------------------------------------------------------------------------

struct CompareResult {
    max_err: f32,
    mean_err: f64,
    mismatch_count: usize,
    /// Per-pixel max-across-channels absolute error, row-major 1920x1080.
    error_map: Vec<f32>,
}

fn compare_tensor_to_canvas(tensor: &[f32], canvas: &[u8]) -> CompareResult {
    let npix = (FRAME_W * FRAME_H) as usize;
    assert_eq!(tensor.len(), 3 * npix, "tensor size mismatch");

    let mut error_map = vec![0.0f32; npix];
    let mut max_err: f32 = 0.0;
    let mut sum_err: f64 = 0.0;
    let mut mismatch_count = 0usize;

    for y in 0..FRAME_H as usize {
        for x in 0..FRAME_W as usize {
            let px = y * FRAME_W as usize + x;
            let mut px_max_err: f32 = 0.0;
            for c in 0..3usize {
                let expected = canvas[px * 4 + c] as f32;
                let actual = tensor[c * npix + px];
                let err = (actual - expected).abs();
                if err > px_max_err {
                    px_max_err = err;
                }
                sum_err += err as f64;
            }
            error_map[px] = px_max_err;
            if px_max_err > max_err {
                max_err = px_max_err;
            }
            if px_max_err > PIXEL_TOL {
                mismatch_count += 1;
            }
        }
    }
    let total_values = 3 * npix;
    CompareResult {
        max_err,
        mean_err: sum_err / total_values as f64,
        mismatch_count,
        error_map,
    }
}

fn save_error_heatmap(error_map: &[f32], path: &Path, scale_max: f32) {
    let mut pixels = vec![0u8; (FRAME_W * FRAME_H) as usize];
    let inv = if scale_max > 0.0 {
        255.0 / scale_max
    } else {
        0.0
    };
    for (i, &err) in error_map.iter().enumerate() {
        pixels[i] = (err * inv).min(255.0) as u8;
    }
    image::save_buffer(path, &pixels, FRAME_W, FRAME_H, image::ColorType::L8)
        .expect("save error heatmap");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Pass a random-filled 1920x1080 canvas through the FullHD identity model as
/// one full-frame ROI.  Every output pixel should match the input within fp16
/// tolerance.  Saves a diagnostic error heatmap on failure.
#[test]
#[serial]
fn test_fullframe_identity_roundtrip() {
    common::init();

    let engine = match identity_engine_fullhd() {
        Some(e) => e,
        None => return,
    };
    common::warmup_engine(&engine, FRAME_W, FRAME_H);

    let canvas = build_random_canvas(42);

    let (shared, rois) = canvas_to_batch_with_fullframe_roi(&canvas);
    engine.submit(shared, Some(&rois)).expect("submit");
    let output = common::recv_inference(&engine);

    assert_eq!(output.num_elements(), 1, "expected one full-frame output");
    let elem = &output.elements()[0];
    assert_eq!(
        elem.slot_number, 0,
        "single-slot batch should use surface slot 0"
    );
    assert!(!elem.tensors.is_empty(), "no output tensors");
    let t = &elem.tensors[0];

    let expected_numel = 3 * FRAME_W as usize * FRAME_H as usize;
    assert_eq!(
        t.dims.num_elements as usize, expected_numel,
        "unexpected tensor size: got {}, expected {}",
        t.dims.num_elements, expected_numel
    );

    let actual: &[f32] = unsafe { t.as_slice() };
    let result = compare_tensor_to_canvas(actual, &canvas);

    eprintln!("\n  Full-frame identity round-trip (random seed=42):");
    eprintln!("    max pixel error : {:.2}", result.max_err);
    eprintln!("    mean pixel error: {:.4}", result.mean_err);
    eprintln!(
        "    mismatched pixels: {} / {} ({:.2}%)",
        result.mismatch_count,
        FRAME_W * FRAME_H,
        100.0 * result.mismatch_count as f64 / (FRAME_W * FRAME_H) as f64
    );

    if result.mismatch_count > 0 {
        let mut min_y = FRAME_H as usize;
        let mut max_y = 0usize;
        let mut min_x = FRAME_W as usize;
        let mut max_x = 0usize;
        for y in 0..FRAME_H as usize {
            for x in 0..FRAME_W as usize {
                let px = y * FRAME_W as usize + x;
                if result.error_map[px] > PIXEL_TOL {
                    if y < min_y {
                        min_y = y;
                    }
                    if y > max_y {
                        max_y = y;
                    }
                    if x < min_x {
                        min_x = x;
                    }
                    if x > max_x {
                        max_x = x;
                    }
                }
            }
        }
        eprintln!("    mismatch bounding box: x=[{min_x}..{max_x}], y=[{min_y}..{max_y}]");

        eprintln!("\n    Per-row mismatch counts (rows with errors):");
        for y in min_y..=max_y {
            let row_start = y * FRAME_W as usize;
            let row_errs: usize = result.error_map[row_start..row_start + FRAME_W as usize]
                .iter()
                .filter(|&&e| e > PIXEL_TOL)
                .count();
            if row_errs > 0 {
                eprintln!("      row {y:>4}: {row_errs:>5} bad pixels");
            }
        }
    }

    let heatmap_path = tmp_dir().join("fullframe_identity_error_heatmap.png");
    save_error_heatmap(&result.error_map, &heatmap_path, result.max_err.max(1.0));
    eprintln!("\n    Error heatmap saved to {}", heatmap_path.display());

    assert!(
        result.max_err <= PIXEL_TOL,
        "full-frame identity round-trip failed: max pixel error {:.2} exceeds tolerance {PIXEL_TOL} \
         ({} mismatched pixels) — see heatmap at {}",
        result.max_err,
        result.mismatch_count,
        heatmap_path.display()
    );
}
