//! Memory-leak stress tests for NvInfer.
//!
//! Runs many inference iterations on the identity model and asserts that
//! neither GPU (VRAM/unified) nor CPU (process RSS) memory grows unboundedly.
//!
//! The tests are `#[serial]` to avoid interference from parallel test suites
//! that allocate GPU memory.

mod common;

use deepstream_nvbufsurface::{
    DsNvNonUniformSurfaceBuffer, DsNvSurfaceBufferGenerator, DsNvUniformSurfaceBufferGenerator,
    NvBufSurfaceMemType, TransformConfig, VideoFormat,
};
use nvidia_gpu_utils::{gpu_mem_used_mib, process_rss_mib};
use nvinfer::{NvInfer, NvInferConfig, Roi};
use savant_core::primitives::RBBox;
use serial_test::serial;
use std::collections::HashMap;

#[link(name = "cuda")]
extern "C" {
    fn cuMemsetD8_v2(dst: u64, value: u8, count: usize) -> u32;
}

const STRESS_ITERATIONS: u64 = 200;
const FRAMES_PER_BATCH: u32 = 4;

/// Maximum allowed GPU memory growth in MiB over the test run.
const GPU_GROWTH_LIMIT_MIB: u64 = 32;
/// Maximum allowed CPU RSS growth in MiB over the test run.
const RSS_GROWTH_LIMIT_MIB: u64 = 64;

fn make_identity_batch(num_frames: u32) -> gstreamer::Buffer {
    common::init();

    let src_gen = DsNvSurfaceBufferGenerator::builder(VideoFormat::RGBA, 12, 12)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(num_frames.max(4))
        .max_buffers(num_frames.max(4))
        .build()
        .expect("src generator");

    let batched_gen = DsNvUniformSurfaceBufferGenerator::new(
        VideoFormat::RGBA,
        12,
        12,
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

fn identity_engine() -> Option<NvInfer> {
    let onnx = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/identity.onnx");
    if !onnx.exists() {
        eprintln!("Skipping: identity.onnx not found at {:?}", onnx);
        return None;
    }
    let props = common::identity_properties();
    let config = NvInferConfig::new(props, "RGBA", 12, 12);
    Some(NvInfer::new(config, Box::new(|_| {})).expect("create NvInfer"))
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
    for i in 0..5 {
        let batch = make_identity_batch(FRAMES_PER_BATCH);
        let _ = engine.infer_sync(batch, i, None);
    }

    let gpu_before = gpu_mem_used_mib(0).expect("gpu_mem_used_mib");
    let rss_before = process_rss_mib().expect("process_rss_mib");
    eprintln!(
        "baseline: GPU = {} MiB, RSS = {} MiB ({} iterations, {} frames/batch)",
        gpu_before, rss_before, STRESS_ITERATIONS, FRAMES_PER_BATCH
    );

    for i in 0..STRESS_ITERATIONS {
        let batch = make_identity_batch(FRAMES_PER_BATCH);
        let output = engine.infer_sync(batch, 100 + i, None).expect("infer_sync");
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
    for i in 0..5 {
        let batch = make_identity_batch(FRAMES_PER_BATCH);
        let _ = engine.infer_sync(batch, i, Some(&rois));
    }

    let gpu_before = gpu_mem_used_mib(0).expect("gpu_mem_used_mib");
    let rss_before = process_rss_mib().expect("process_rss_mib");
    eprintln!(
        "ROI baseline: GPU = {} MiB, RSS = {} MiB",
        gpu_before, rss_before
    );

    for i in 0..STRESS_ITERATIONS {
        let batch = make_identity_batch(FRAMES_PER_BATCH);
        let output = engine
            .infer_sync(batch, 1000 + i, Some(&rois))
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

fn make_nonuniform_batch_with_rois() -> (gstreamer::Buffer, HashMap<u32, Vec<Roi>>) {
    common::init();

    let specs: &[(u32, u32, u8, i64)] = &[(24, 24, 100, 1), (36, 36, 180, 2)];

    let buffer = {
        let mut batch =
            DsNvNonUniformSurfaceBuffer::new(specs.len() as u32, 0).expect("non-uniform batch");

        for &(w, h, fill, id) in specs {
            let gen = DsNvSurfaceBufferGenerator::builder(VideoFormat::RGBA, w, h)
                .gpu_id(0)
                .mem_type(NvBufSurfaceMemType::Default)
                .min_buffers(1)
                .max_buffers(1)
                .build()
                .expect("src generator");

            let (src, data_ptr, pitch) = gen.acquire_surface_with_ptr(Some(id)).unwrap();
            let fill_size = (pitch * h) as usize;
            let ret = unsafe { cuMemsetD8_v2(data_ptr as u64, fill, fill_size) };
            assert_eq!(ret, 0, "cuMemsetD8_v2 failed");

            batch.add(&src, Some(id)).unwrap();
        }

        batch.finalize().unwrap();
        batch.as_gst_buffer().unwrap()
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

    (buffer, rois)
}

fn identity_engine_flexible() -> Option<NvInfer> {
    let onnx = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/identity.onnx");
    if !onnx.exists() {
        eprintln!("Skipping: identity.onnx not found at {:?}", onnx);
        return None;
    }
    let props = common::identity_properties();
    let config = NvInferConfig::new_flexible(props, "RGBA");
    Some(NvInfer::new(config, Box::new(|_| {})).expect("create NvInfer (flexible)"))
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
    for i in 0..5 {
        let (batch, rois) = make_nonuniform_batch_with_rois();
        let _ = engine.infer_sync(batch, i, Some(&rois));
    }

    let gpu_before = gpu_mem_used_mib(0).expect("gpu_mem_used_mib");
    let rss_before = process_rss_mib().expect("process_rss_mib");
    eprintln!(
        "nonuniform baseline: GPU = {} MiB, RSS = {} MiB",
        gpu_before, rss_before
    );

    for i in 0..STRESS_ITERATIONS {
        let (batch, rois) = make_nonuniform_batch_with_rois();
        let output = engine
            .infer_sync(batch, 2000 + i, Some(&rois))
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
