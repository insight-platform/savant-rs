//! Benchmarks for NvInfer with identity, age_gender, and yolo11m-seg models,
//! using both **uniform** (all frames same resolution) and **non-uniform**
//! (heterogeneous frame resolutions) batched surfaces.
//!
//! Each (model, batch_size) pair gets its own nvinfer config so TensorRT builds
//! a dedicated engine optimised for that exact batch size.  Engine files are
//! cached in the `assets/` directory; only the first run pays the build cost.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use deepstream_nvbufsurface::{
    DsNvNonUniformSurfaceBuffer, DsNvSurfaceBufferGenerator, DsNvUniformSurfaceBufferGenerator,
    NvBufSurfaceMemType, TransformConfig, VideoFormat,
};
use nvinfer::{NvInfer, NvInferConfig};
use std::collections::HashMap;
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
// Model specs
// ---------------------------------------------------------------------------

/// Model descriptor with base nvinfer properties. `batch-size` and
/// `model-engine-file` are set per batch size in bench_model.
struct ModelSpec {
    onnx_path: PathBuf,
    onnx_stem: String,
    base_properties: HashMap<String, String>,
    format: VideoFormat,
    width: u32,
    height: u32,
    group_name: String,
}

fn identity_base_properties(dir: &Path) -> HashMap<String, String> {
    let mut m = HashMap::new();
    m.insert("gpu-id".into(), "0".into());
    m.insert("gie-unique-id".into(), "1".into());
    m.insert("net-scale-factor".into(), "1.0".into());
    m.insert(
        "onnx-file".into(),
        dir.join("identity.onnx").to_string_lossy().into(),
    );
    m.insert("network-mode".into(), "2".into());
    m.insert("network-type".into(), "100".into());
    m.insert("infer-dims".into(), "3;12;12".into());
    m.insert("model-color-format".into(), "0".into());
    m
}

fn age_gender_base_properties(dir: &Path) -> HashMap<String, String> {
    let mut m = HashMap::new();
    m.insert("gpu-id".into(), "0".into());
    m.insert("gie-unique-id".into(), "2".into());
    m.insert("net-scale-factor".into(), "0.007843137254902".into());
    m.insert("offsets".into(), "127.5;127.5;127.5".into());
    m.insert(
        "onnx-file".into(),
        dir.join("age_gender_mobilenet_v2_dynBatch.onnx")
            .to_string_lossy()
            .into(),
    );
    m.insert("network-mode".into(), "2".into());
    m.insert("network-type".into(), "100".into());
    m.insert("infer-dims".into(), "3;112;112".into());
    m.insert("model-color-format".into(), "0".into());
    m
}

fn yolo_base_properties(dir: &Path) -> HashMap<String, String> {
    let mut m = HashMap::new();
    m.insert(
        "onnx-file".into(),
        dir.join("yolo11m-seg.onnx").to_string_lossy().into(),
    );
    m.insert("network-mode".into(), "2".into());
    m.insert("workspace-size".into(), "6144".into());
    m.insert("infer-dims".into(), "3;640;640".into());
    m.insert("maintain-aspect-ratio".into(), "1".into());
    m.insert("symmetric-padding".into(), "0".into());
    m.insert("net-scale-factor".into(), "0.003921569790691137".into());
    m.insert("offsets".into(), "0.0;0.0;0.0".into());
    m.insert("model-color-format".into(), "0".into());
    m.insert("output-blob-names".into(), "output0;output1".into());
    m.insert("num-detected-classes".into(), "80".into());
    m.insert("gpu-id".into(), "0".into());
    m.insert("secondary-reinfer-interval".into(), "0".into());
    m.insert("operate-on-gie-id".into(), "0".into());
    m.insert("operate-on-class-ids".into(), "0".into());
    m.insert("gie-unique-id".into(), "1".into());
    m.insert("network-type".into(), "100".into());
    // Per-class sections via dotted notation
    m.insert(
        "class-attrs-all.pre-cluster-threshold".into(),
        "10000000000.0".into(),
    );
    m.insert("class-attrs-0.pre-cluster-threshold".into(), "0.5".into());
    m.insert("class-attrs-0.nms-iou-threshold".into(), "0.5".into());
    m.insert("class-attrs-0.detected-min-h".into(), "64".into());
    m
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

fn make_uniform_batch(format: VideoFormat, w: u32, h: u32, batch_size: u32) -> gstreamer::Buffer {
    init();

    let min_bufs = batch_size.max(4);
    let src_gen = DsNvSurfaceBufferGenerator::builder(format, w, h)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(min_bufs)
        .max_buffers(min_bufs)
        .build()
        .expect("src generator");

    let batched_gen = DsNvUniformSurfaceBufferGenerator::new(
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

    batch.finalize().unwrap();
    batch.as_gst_buffer().unwrap()
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

        let gen = DsNvSurfaceBufferGenerator::builder(format, w, h)
            .gpu_id(0)
            .mem_type(NvBufSurfaceMemType::Default)
            .min_buffers(1)
            .max_buffers(1)
            .build()
            .expect("src generator for nonuniform slot");
        src_bufs.push(gen.acquire_surface(Some(i as i64)).unwrap());
    }

    let mut batch = DsNvNonUniformSurfaceBuffer::new(batch_size, 0).unwrap();
    for (i, buf) in src_bufs.iter().enumerate() {
        batch.add(buf, Some(i as i64)).unwrap();
    }

    batch.finalize().unwrap();
    batch.as_gst_buffer().unwrap()
}

// ---------------------------------------------------------------------------
// Benchmark driver
// ---------------------------------------------------------------------------

const FILL_COUNTS: &[u32] = &[1, 4, 8, 16];

fn bench_model(c: &mut Criterion, spec: &ModelSpec, batch_sizes: &[u32], mode: BatchMode) {
    if !spec.onnx_path.exists() {
        eprintln!(
            "Skipping {} benchmarks: ONNX not found at {:?}",
            spec.group_name, spec.onnx_path
        );
        return;
    }

    let dir = assets_dir();
    let group_name = format!("{}/{}", spec.group_name, mode);
    let mut group = c.benchmark_group(&group_name);

    for &bs in batch_sizes {
        let mut props = spec.base_properties.clone();
        props.insert("batch-size".into(), bs.to_string());
        props.insert(
            "model-engine-file".into(),
            dir.join(format!("{}_b{}_gpu0_fp16.engine", spec.onnx_stem, bs))
                .to_string_lossy()
                .into(),
        );

        let config = NvInferConfig::new(props, "RGBA", spec.width, spec.height);
        let engine = NvInfer::new(config, Box::new(|_| {})).expect("create NvInfer for bench");

        let warm = make_batch(mode, spec.format, spec.width, spec.height, bs);
        let _ = engine.infer_sync(warm, 0, None);

        let fills: Vec<u32> = FILL_COUNTS.iter().copied().filter(|&f| f <= bs).collect();

        for &fill in &fills {
            let id = format!("bs{}/fill{}", bs, fill);
            group.bench_with_input(BenchmarkId::new(&id, fill), &fill, |b, &fill| {
                b.iter(|| {
                    let batch = make_batch(mode, spec.format, spec.width, spec.height, fill);
                    let out = engine.infer_sync(batch, 0, None).expect("infer_sync");
                    std::hint::black_box(out);
                })
            });
        }
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
            onnx_stem: "identity.onnx".into(),
            base_properties: identity_base_properties(&dir),
            format: VideoFormat::RGBA,
            width: 12,
            height: 12,
            group_name: "identity".into(),
        },
        ModelSpec {
            onnx_path: dir.join("age_gender_mobilenet_v2_dynBatch.onnx"),
            onnx_stem: "age_gender_mobilenet_v2_dynBatch.onnx".into(),
            base_properties: age_gender_base_properties(&dir),
            format: VideoFormat::RGBA,
            width: 112,
            height: 112,
            group_name: "age_gender".into(),
        },
        ModelSpec {
            onnx_path: dir.join("yolo11m-seg.onnx"),
            onnx_stem: "yolo11m-seg.onnx".into(),
            base_properties: yolo_base_properties(&dir),
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
