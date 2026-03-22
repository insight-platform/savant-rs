//! Memory-leak stress tests for NvInfer.
//!
//! Runs many inference iterations on the identity model and asserts that
//! neither GPU (VRAM/unified) nor CPU (process RSS) memory grows unboundedly.
//!
//! The tests are `#[serial]` to avoid interference from parallel test suites
//! that allocate GPU memory.

mod common;

use deepstream_buffers::{
    BufferGenerator, NonUniformBatch, NvBufSurfaceMemType, SavantIdMetaKind, SharedBuffer,
    SurfaceView, UniformBatchGenerator, VideoFormat,
};
use nvidia_gpu_utils::{gpu_mem_used_mib, process_rss_mib};
use nvinfer::{NvInfer, NvInferConfig, Roi};
use savant_core::primitives::RBBox;
use serial_test::serial;
use std::collections::HashMap;

const STRESS_ITERATIONS: u64 = 200;
const FRAMES_PER_BATCH: u32 = 4;

/// Maximum allowed GPU memory growth in MiB over the test run.
/// Jetson GPU compute mode uses slightly more memory than VIC.
const GPU_GROWTH_LIMIT_MIB: u64 = 48;
/// Maximum allowed CPU RSS growth in MiB over the test run.
const RSS_GROWTH_LIMIT_MIB: u64 = 64;

fn make_identity_batch(num_frames: u32) -> SharedBuffer {
    common::init();

    let min_bufs = num_frames.max(4);
    let src_gen = BufferGenerator::builder(VideoFormat::RGBA, 12, 12)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(min_bufs)
        .max_buffers(min_bufs)
        .build()
        .expect("src generator");

    let batched_gen = UniformBatchGenerator::new(
        VideoFormat::RGBA,
        12,
        12,
        16,
        2,
        0,
        NvBufSurfaceMemType::Default,
    )
    .expect("batched generator");

    let config = common::platform_transform_config();
    let ids: Vec<SavantIdMetaKind> = (0..num_frames)
        .map(|i| SavantIdMetaKind::Frame(i as i64))
        .collect();
    let mut batch = batched_gen.acquire_batch(config, ids).unwrap();

    for i in 0..num_frames {
        let src_shared = src_gen.acquire(Some(i as i64)).unwrap();
        let src_view = SurfaceView::from_buffer(&src_shared, 0).unwrap();
        batch.transform_slot(i, &src_view, None).unwrap();
    }

    batch.finalize().unwrap();
    batch.shared_buffer()
}

fn identity_engine() -> Option<NvInfer> {
    let onnx = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/identity.onnx");
    if !onnx.exists() {
        eprintln!("Skipping: identity.onnx not found at {:?}", onnx);
        return None;
    }
    let props = common::identity_properties();
    let config = NvInferConfig::new(props, "RGBA", 12, 12);
    let engine = NvInfer::new(config, Box::new(|_| {})).expect("create NvInfer");
    common::promote_built_engine("identity.onnx", 16);
    Some(engine)
}

/// Run `STRESS_ITERATIONS` synchronous inferences and assert that neither GPU
/// nor CPU memory grew beyond the allowed thresholds.
#[test]
#[serial]
fn stress_no_gpu_leak() {
    common::init();
    let mut engine = match identity_engine() {
        Some(e) => e,
        None => return,
    };

    // Warm up: a few iterations to let TensorRT/CUDA settle allocations.
    for _ in 0..5 {
        let shared = make_identity_batch(FRAMES_PER_BATCH);
        let _ = engine.infer_sync(shared, None);
    }

    let gpu_before = gpu_mem_used_mib(0).expect("gpu_mem_used_mib");
    let rss_before = process_rss_mib().expect("process_rss_mib");
    eprintln!(
        "baseline: GPU = {} MiB, RSS = {} MiB ({} iterations, {} frames/batch)",
        gpu_before, rss_before, STRESS_ITERATIONS, FRAMES_PER_BATCH
    );

    for _ in 0..STRESS_ITERATIONS {
        let shared = make_identity_batch(FRAMES_PER_BATCH);
        let output = engine.infer_sync(shared, None).expect("infer_sync");
        // Consume output to prove tensors are readable, then drop.
        assert!(!output.elements().is_empty());
        drop(output);
    }

    let gpu_after = gpu_mem_used_mib(0).expect("gpu_mem_used_mib");
    let rss_after = process_rss_mib().expect("process_rss_mib");
    eprintln!(
        "after:    GPU = {} MiB (+{}), RSS = {} MiB (+{})",
        gpu_after,
        gpu_after.saturating_sub(gpu_before),
        rss_after,
        rss_after.saturating_sub(rss_before),
    );

    engine.shutdown().ok();
    drop(engine);

    let gpu_final = gpu_mem_used_mib(0).expect("gpu_mem_used_mib");
    let rss_final = process_rss_mib().expect("process_rss_mib");
    eprintln!(
        "final:    GPU = {} MiB (+{}), RSS = {} MiB (+{})",
        gpu_final,
        gpu_final.saturating_sub(gpu_before),
        rss_final,
        rss_final.saturating_sub(rss_before),
    );

    let gpu_growth = gpu_after.saturating_sub(gpu_before);
    let rss_growth = rss_after.saturating_sub(rss_before);

    assert!(
        gpu_growth <= GPU_GROWTH_LIMIT_MIB,
        "GPU memory leak detected: grew by {} MiB (limit {} MiB); before={}, after={}",
        gpu_growth,
        GPU_GROWTH_LIMIT_MIB,
        gpu_before,
        gpu_after
    );

    assert!(
        rss_growth <= RSS_GROWTH_LIMIT_MIB,
        "CPU RSS leak detected: grew by {} MiB (limit {} MiB); before={}, after={}",
        rss_growth,
        RSS_GROWTH_LIMIT_MIB,
        rss_before,
        rss_after
    );
}

/// Same stress test but with ROIs to exercise the ROI metadata code paths.
#[test]
#[serial]
fn stress_no_leak_with_rois() {
    common::init();
    let mut engine = match identity_engine() {
        Some(e) => e,
        None => return,
    };

    use nvinfer::Roi;
    use savant_core::primitives::RBBox;
    use std::collections::HashMap;

    let full_bbox = RBBox::ltwh(0.0, 0.0, 12.0, 12.0).unwrap();
    let rois: HashMap<u32, Vec<Roi>> = (0..FRAMES_PER_BATCH)
        .map(|slot| {
            (
                slot,
                vec![
                    Roi {
                        id: slot as i64 * 10,
                        bbox: full_bbox.clone(),
                    },
                    Roi {
                        id: slot as i64 * 10 + 1,
                        bbox: full_bbox.clone(),
                    },
                ],
            )
        })
        .collect();

    // Warm up.
    for _ in 0..5 {
        let shared = make_identity_batch(FRAMES_PER_BATCH);
        let _ = engine.infer_sync(shared, Some(&rois));
    }

    let gpu_before = gpu_mem_used_mib(0).expect("gpu_mem_used_mib");
    let rss_before = process_rss_mib().expect("process_rss_mib");
    eprintln!(
        "ROI baseline: GPU = {} MiB, RSS = {} MiB",
        gpu_before, rss_before
    );

    for _ in 0..STRESS_ITERATIONS {
        let shared = make_identity_batch(FRAMES_PER_BATCH);
        let output = engine
            .infer_sync(shared, Some(&rois))
            .expect("infer_sync with ROIs");
        assert!(!output.elements().is_empty());
        drop(output);
    }

    let gpu_after = gpu_mem_used_mib(0).expect("gpu_mem_used_mib");
    let rss_after = process_rss_mib().expect("process_rss_mib");
    eprintln!(
        "ROI after:    GPU = {} MiB (+{}), RSS = {} MiB (+{})",
        gpu_after,
        gpu_after.saturating_sub(gpu_before),
        rss_after,
        rss_after.saturating_sub(rss_before),
    );

    engine.shutdown().ok();
    drop(engine);

    let gpu_growth = gpu_after.saturating_sub(gpu_before);
    let rss_growth = rss_after.saturating_sub(rss_before);

    assert!(
        gpu_growth <= GPU_GROWTH_LIMIT_MIB,
        "GPU leak (ROI path): grew by {} MiB (limit {}); before={}, after={}",
        gpu_growth,
        GPU_GROWTH_LIMIT_MIB,
        gpu_before,
        gpu_after
    );

    assert!(
        rss_growth <= RSS_GROWTH_LIMIT_MIB,
        "CPU RSS leak (ROI path): grew by {} MiB (limit {}); before={}, after={}",
        rss_growth,
        RSS_GROWTH_LIMIT_MIB,
        rss_before,
        rss_after
    );
}

// ─── Non-uniform batch helpers ───────────────────────────────────────────────

fn make_nonuniform_batch_with_rois() -> (SharedBuffer, HashMap<u32, Vec<Roi>>) {
    common::init();

    let specs: &[(u32, u32, u8, i64)] = &[(24, 24, 100, 1), (36, 36, 180, 2)];

    let shared = {
        let mut batch = NonUniformBatch::new(0);

        let mut ids = Vec::new();
        for &(w, h, fill, id) in specs {
            let gen = BufferGenerator::builder(VideoFormat::RGBA, w, h)
                .gpu_id(0)
                .mem_type(NvBufSurfaceMemType::Default)
                .min_buffers(1)
                .max_buffers(1)
                .build()
                .expect("src generator");

            let src_shared = gen.acquire(Some(id)).unwrap();
            let view = SurfaceView::from_buffer(&src_shared, 0).unwrap();
            view.memset(fill).expect("memset_surface");
            batch.add(&view).unwrap();
            drop(view);
            ids.push(SavantIdMetaKind::Frame(id));
        }

        batch.finalize(ids).unwrap()
    };

    let rois: HashMap<u32, Vec<Roi>> = specs
        .iter()
        .enumerate()
        .map(|(slot, &(w, h, _, _))| {
            let bbox = RBBox::ltwh(0.0, 0.0, w as f32, h as f32).unwrap();
            (
                slot as u32,
                vec![
                    Roi {
                        id: slot as i64 * 10,
                        bbox: bbox.clone(),
                    },
                    Roi {
                        id: slot as i64 * 10 + 1,
                        bbox: bbox.clone(),
                    },
                ],
            )
        })
        .collect();

    (shared, rois)
}

fn identity_engine_flexible() -> Option<NvInfer> {
    let onnx = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/identity.onnx");
    if !onnx.exists() {
        eprintln!("Skipping: identity.onnx not found at {:?}", onnx);
        return None;
    }
    let props = common::identity_properties();
    let config = NvInferConfig::new_flexible(props, "RGBA");
    let engine = NvInfer::new(config, Box::new(|_| {})).expect("create NvInfer (flexible)");
    common::promote_built_engine("identity.onnx", 16);
    Some(engine)
}

/// Stress test for non-uniform batches with ROIs.
///
/// Non-uniform buffers use a fundamentally different ownership model
/// (system-memory header + `GstParentBufferMeta` keeping per-frame GPU
/// buffers alive), so their deallocation path is entirely separate from
/// uniform batches.
#[test]
#[serial]
fn stress_no_leak_nonuniform() {
    common::init();
    let mut engine = match identity_engine_flexible() {
        Some(e) => e,
        None => return,
    };

    // Warm up.
    for _ in 0..5 {
        let (shared, rois) = make_nonuniform_batch_with_rois();
        let _ = engine.infer_sync(shared, Some(&rois));
    }

    let gpu_before = gpu_mem_used_mib(0).expect("gpu_mem_used_mib");
    let rss_before = process_rss_mib().expect("process_rss_mib");
    eprintln!(
        "nonuniform baseline: GPU = {} MiB, RSS = {} MiB",
        gpu_before, rss_before
    );

    for _ in 0..STRESS_ITERATIONS {
        let (shared, rois) = make_nonuniform_batch_with_rois();
        let output = engine
            .infer_sync(shared, Some(&rois))
            .expect("infer_sync nonuniform");
        assert!(!output.elements().is_empty());
        drop(output);
    }

    let gpu_after = gpu_mem_used_mib(0).expect("gpu_mem_used_mib");
    let rss_after = process_rss_mib().expect("process_rss_mib");
    eprintln!(
        "nonuniform after:    GPU = {} MiB (+{}), RSS = {} MiB (+{})",
        gpu_after,
        gpu_after.saturating_sub(gpu_before),
        rss_after,
        rss_after.saturating_sub(rss_before),
    );

    engine.shutdown().ok();
    drop(engine);

    let gpu_growth = gpu_after.saturating_sub(gpu_before);
    let rss_growth = rss_after.saturating_sub(rss_before);

    assert!(
        gpu_growth <= GPU_GROWTH_LIMIT_MIB,
        "GPU leak (nonuniform): grew by {} MiB (limit {}); before={}, after={}",
        gpu_growth,
        GPU_GROWTH_LIMIT_MIB,
        gpu_before,
        gpu_after
    );

    assert!(
        rss_growth <= RSS_GROWTH_LIMIT_MIB,
        "CPU RSS leak (nonuniform): grew by {} MiB (limit {}); before={}, after={}",
        rss_growth,
        RSS_GROWTH_LIMIT_MIB,
        rss_before,
        rss_after
    );
}
