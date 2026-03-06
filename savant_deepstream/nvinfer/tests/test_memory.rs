//! Memory-leak stress tests for NvInfer.
//!
//! Runs many inference iterations on the identity model and asserts that
//! neither GPU (VRAM/unified) nor CPU (process RSS) memory grows unboundedly.
//!
//! The tests are `#[serial]` to avoid interference from parallel test suites
//! that allocate GPU memory.

mod common;

use deepstream_nvbufsurface::{
    DsNvSurfaceBufferGenerator, DsNvUniformSurfaceBufferGenerator, NvBufSurfaceMemType,
    TransformConfig, VideoFormat,
};
use nvidia_gpu_utils::{gpu_mem_used_mib, process_rss_mib};
use nvinfer::{NvInfer, NvInferConfig};
use serial_test::serial;

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

    use nvinfer::{Rect, Roi};
    use std::collections::HashMap;

    let full_rect = Rect {
        left: 0,
        top: 0,
        width: 12,
        height: 12,
    };
    let rois: HashMap<u32, Vec<Roi>> = (0..FRAMES_PER_BATCH)
        .map(|slot| {
            (
                slot,
                vec![
                    Roi {
                        id: slot as i64 * 10,
                        rect: full_rect.clone(),
                    },
                    Roi {
                        id: slot as i64 * 10 + 1,
                        rect: full_rect.clone(),
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
