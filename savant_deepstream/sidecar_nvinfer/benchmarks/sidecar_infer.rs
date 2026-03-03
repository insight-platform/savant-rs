//! Benchmarks for SidecarNvInfer with identity, age_gender, and yolo11m-seg models.
//!
//! Each (model, batch_size) pair gets its own nvinfer config so TensorRT builds
//! a dedicated engine optimised for that exact batch size.  Engine files are
//! cached in the `assets/` directory; only the first run pays the build cost.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use deepstream_nvbufsurface::{
    BatchedNvBufSurfaceGenerator, NvBufSurfaceGenerator, NvBufSurfaceMemType, TransformConfig,
    VideoFormat,
};
use sidecar_nvinfer::{SidecarConfig, SidecarNvInfer};
use std::path::{Path, PathBuf};
use std::sync::Once;

static INIT: Once = Once::new();

fn init() {
    INIT.call_once(|| {
        gstreamer::init().unwrap();
        deepstream_nvbufsurface::cuda_init(0).expect("CUDA init — is a GPU available?");
    });
}

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets")
}

/// Model descriptor parsed from a template config.
struct ModelSpec {
    /// Absolute path to ONNX file.
    onnx_path: PathBuf,
    /// Template config path (for identifying the model).
    template: PathBuf,
    /// Input dimensions: format, width, height.
    format: VideoFormat,
    width: u32,
    height: u32,
    /// Human-readable group name for Criterion.
    group_name: String,
}

/// Generate a temporary nvinfer config for a specific `batch_size`.
///
/// Rewrites:
///   * `batch-size` → the requested value
///   * `onnx-file` → absolute path
///   * `model-engine-file` → `{onnx}_b{bs}_gpu0_fp16.engine` in assets/
///   * `process-mode` → 1 (primary, required for sidecar pipeline)
fn generate_config(template: &Path, batch_size: u32) -> PathBuf {
    let dir = template.parent().unwrap();
    let content = std::fs::read_to_string(template).expect("read template config");

    let mut onnx_filename = String::new();
    let mut lines: Vec<String> = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();
        let key = trimmed.split(['=', ' ']).next().unwrap_or("");

        match key {
            "batch-size" => {
                lines.push(format!("batch-size={}", batch_size));
            }
            "onnx-file" => {
                let val = kv_value(trimmed);
                onnx_filename = val.to_string();
                let abs = dir.join(val);
                lines.push(format!("onnx-file={}", abs.display()));
            }
            "model-engine-file" => {
                // will be emitted after we know the onnx filename; skip for now
            }
            "process-mode" => {
                lines.push("process-mode=1".to_string());
            }
            _ => {
                lines.push(line.to_string());
            }
        }
    }

    let engine_name = format!("{}_b{}_gpu0_fp16.engine", onnx_filename, batch_size);
    let engine_abs = dir.join(&engine_name);
    lines.insert(
        lines
            .iter()
            .position(|l| l.starts_with("onnx-file"))
            .map(|i| i + 1)
            .unwrap_or(1),
        format!("model-engine-file={}", engine_abs.display()),
    );

    let stem = template.file_stem().unwrap().to_string_lossy();
    let tmp = std::env::temp_dir().join(format!("sidecar_bench_{}_b{}.txt", stem, batch_size));
    std::fs::write(&tmp, lines.join("\n")).unwrap();
    tmp
}

/// Extract the value after the first `=` (with optional surrounding spaces).
fn kv_value(line: &str) -> &str {
    line.split_once('=').map(|x| x.1).unwrap_or("").trim()
}

fn make_batch(format: VideoFormat, w: u32, h: u32, batch_size: u32) -> gstreamer::Buffer {
    init();

    let min_bufs = batch_size.max(4);
    let src_gen = NvBufSurfaceGenerator::builder(format, w, h)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(min_bufs)
        .max_buffers(min_bufs)
        .build()
        .expect("src generator");

    let batched_gen = BatchedNvBufSurfaceGenerator::new(
        format,
        w,
        h,
        batch_size,
        2,
        0,
        NvBufSurfaceMemType::Default,
    )
    .expect("batched generator");

    let config = TransformConfig::default();
    let mut batch = batched_gen.acquire_batched_surface(config).unwrap();

    for i in 0..batch_size {
        let src = src_gen.acquire_surface(Some(i as i64)).unwrap();
        batch.fill_slot(&src, None, Some(i as i64)).unwrap();
    }

    batch.finalize()
}

fn bench_model(c: &mut Criterion, spec: &ModelSpec, batch_sizes: &[u32]) {
    if !spec.onnx_path.exists() {
        eprintln!(
            "Skipping {} benchmarks: ONNX not found at {:?}",
            spec.group_name, spec.onnx_path
        );
        return;
    }

    let mut group = c.benchmark_group(&spec.group_name);

    for &bs in batch_sizes {
        let cfg_path = generate_config(&spec.template, bs);
        let config = SidecarConfig::new(&cfg_path, "RGBA", spec.width, spec.height);
        let sidecar =
            SidecarNvInfer::new(config, Box::new(|_| {})).expect("create sidecar for bench");

        // warm-up: one inference to ensure engine is loaded
        let warm = make_batch(spec.format, spec.width, spec.height, bs);
        let _ = sidecar.infer_sync(warm, 0);

        group.bench_with_input(BenchmarkId::new("batch", bs), &bs, |b, &bs| {
            b.iter(|| {
                let batch = make_batch(spec.format, spec.width, spec.height, bs);
                let out = sidecar.infer_sync(batch, 0).expect("infer_sync");
                std::hint::black_box(out);
            })
        });
    }

    group.finish();
}

fn bench_sync_batch_sizes(c: &mut Criterion) {
    init();
    let dir = assets_dir();

    let identity = ModelSpec {
        onnx_path: dir.join("identity.onnx"),
        template: dir.join("identity_nvinfer.txt"),
        format: VideoFormat::RGBA,
        width: 12,
        height: 12,
        group_name: "identity".into(),
    };

    let age_gender = ModelSpec {
        onnx_path: dir.join("age_gender_mobilenet_v2_dynBatch.onnx"),
        template: dir.join("age_gender_nvinfer.txt"),
        format: VideoFormat::RGBA,
        width: 112,
        height: 112,
        group_name: "age_gender".into(),
    };

    let yolo = ModelSpec {
        onnx_path: dir.join("yolo11m-seg.onnx"),
        template: dir.join("yolo11m-seg_config_savant.txt"),
        format: VideoFormat::RGBA,
        width: 640,
        height: 640,
        group_name: "yolo11m_seg".into(),
    };

    bench_model(c, &identity, &[1, 4, 16]);
    bench_model(c, &age_gender, &[1, 4, 16]);
    bench_model(c, &yolo, &[1, 4, 8]);
}

criterion_group!(benches, bench_sync_batch_sizes);
criterion_main!(benches);
