//! Benchmarks for SidecarNvInfer with identity, age_gender, and yolo11m-seg models,
//! using both **uniform** (all frames same resolution) and **non-uniform**
//! (heterogeneous frame resolutions) batched surfaces.
//!
//! Each (model, batch_size) pair gets its own nvinfer config so TensorRT builds
//! a dedicated engine optimised for that exact batch size.  Engine files are
//! cached in the `assets/` directory; only the first run pays the build cost.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use deepstream_nvbufsurface::{
    BatchedNvBufSurfaceGenerator, HeterogeneousBatch, NvBufSurfaceGenerator, NvBufSurfaceMemType,
    TransformConfig, VideoFormat,
};
use sidecar_nvinfer::{SidecarConfig, SidecarNvInfer};
use std::path::{Path, PathBuf};
use std::sync::Once;

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------

static INIT: Once = Once::new();

fn init() {
    INIT.call_once(|| {
        if std::env::var_os("GST_DEBUG_FILE").is_none() {
            std::env::set_var("GST_DEBUG_FILE", "/dev/null");
        }
        if std::env::var_os("NVDSINFERSERVER_LOG_LEVEL").is_none() {
            std::env::set_var("NVDSINFERSERVER_LOG_LEVEL", "0");
        }
        gstreamer::init().unwrap();
        deepstream_nvbufsurface::cuda_init(0).expect("CUDA init — is a GPU available?");
    });
}

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets")
}

// ---------------------------------------------------------------------------
// Config generation
// ---------------------------------------------------------------------------

/// Model descriptor parsed from a template config.
struct ModelSpec {
    onnx_path: PathBuf,
    template: PathBuf,
    format: VideoFormat,
    width: u32,
    height: u32,
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
            "model-engine-file" => {}
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

fn kv_value(line: &str) -> &str {
    line.split_once('=').map(|x| x.1).unwrap_or("").trim()
}

// ---------------------------------------------------------------------------
// Batch modes
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum BatchMode {
    Uniform,
    NonUniform,
}

impl std::fmt::Display for BatchMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchMode::Uniform => f.write_str("uniform"),
            BatchMode::NonUniform => f.write_str("nonuniform"),
        }
    }
}

const NON_UNIFORM_SCALES: [f32; 8] = [1.0, 0.75, 1.5, 0.5, 2.0, 1.25, 0.875, 1.75];

fn make_batch(
    mode: BatchMode,
    format: VideoFormat,
    base_w: u32,
    base_h: u32,
    batch_size: u32,
) -> gstreamer::Buffer {
    match mode {
        BatchMode::Uniform => make_uniform_batch(format, base_w, base_h, batch_size),
        BatchMode::NonUniform => make_nonuniform_batch(format, base_w, base_h, batch_size),
    }
}

fn make_uniform_batch(
    format: VideoFormat,
    w: u32,
    h: u32,
    batch_size: u32,
) -> gstreamer::Buffer {
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

fn make_nonuniform_batch(
    format: VideoFormat,
    base_w: u32,
    base_h: u32,
    batch_size: u32,
) -> gstreamer::Buffer {
    init();

    let mut src_bufs = Vec::with_capacity(batch_size as usize);
    for i in 0..batch_size as usize {
        let s = NON_UNIFORM_SCALES[i % NON_UNIFORM_SCALES.len()];
        let w = ((base_w as f32 * s) as u32).max(4);
        let h = ((base_h as f32 * s) as u32).max(4);

        let gen = NvBufSurfaceGenerator::builder(format, w, h)
            .gpu_id(0)
            .mem_type(NvBufSurfaceMemType::Default)
            .min_buffers(1)
            .max_buffers(1)
            .build()
            .expect("src generator for nonuniform slot");
        src_bufs.push(gen.acquire_surface(Some(i as i64)).unwrap());
    }

    let mut batch = HeterogeneousBatch::new(batch_size, 0).unwrap();
    for (i, buf) in src_bufs.iter().enumerate() {
        batch.add(buf, Some(i as i64)).unwrap();
    }

    batch.finalize()
}

// ---------------------------------------------------------------------------
// Benchmark driver
// ---------------------------------------------------------------------------

fn bench_model(c: &mut Criterion, spec: &ModelSpec, batch_sizes: &[u32], mode: BatchMode) {
    if !spec.onnx_path.exists() {
        eprintln!(
            "Skipping {} benchmarks: ONNX not found at {:?}",
            spec.group_name, spec.onnx_path
        );
        return;
    }

    let group_name = format!("{}/{}", spec.group_name, mode);
    let mut group = c.benchmark_group(&group_name);

    for &bs in batch_sizes {
        let cfg_path = generate_config(&spec.template, bs);
        let config = SidecarConfig::new(&cfg_path, "RGBA", spec.width, spec.height);
        let sidecar =
            SidecarNvInfer::new(config, Box::new(|_| {})).expect("create sidecar for bench");

        let warm = make_batch(mode, spec.format, spec.width, spec.height, bs);
        let _ = sidecar.infer_sync(warm, 0);

        group.bench_with_input(BenchmarkId::new("batch", bs), &bs, |b, &bs| {
            b.iter(|| {
                let batch = make_batch(mode, spec.format, spec.width, spec.height, bs);
                let out = sidecar.infer_sync(batch, 0).expect("infer_sync");
                std::hint::black_box(out);
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn bench_sync_batch_sizes(c: &mut Criterion) {
    init();
    let dir = assets_dir();

    let models = [
        ModelSpec {
            onnx_path: dir.join("identity.onnx"),
            template: dir.join("identity_nvinfer.txt"),
            format: VideoFormat::RGBA,
            width: 12,
            height: 12,
            group_name: "identity".into(),
        },
        ModelSpec {
            onnx_path: dir.join("age_gender_mobilenet_v2_dynBatch.onnx"),
            template: dir.join("age_gender_nvinfer.txt"),
            format: VideoFormat::RGBA,
            width: 112,
            height: 112,
            group_name: "age_gender".into(),
        },
        ModelSpec {
            onnx_path: dir.join("yolo11m-seg.onnx"),
            template: dir.join("yolo11m-seg_config_savant.txt"),
            format: VideoFormat::RGBA,
            width: 640,
            height: 640,
            group_name: "yolo11m_seg".into(),
        },
    ];

    let batch_sizes: &[&[u32]] = &[&[1, 4, 16], &[1, 4, 16], &[1, 4, 8]];
    let modes = [BatchMode::Uniform, BatchMode::NonUniform];

    for mode in modes {
        for (spec, sizes) in models.iter().zip(batch_sizes.iter()) {
            bench_model(c, spec, sizes, mode);
        }
    }
}

criterion_group!(benches, bench_sync_batch_sizes);
criterion_main!(benches);
