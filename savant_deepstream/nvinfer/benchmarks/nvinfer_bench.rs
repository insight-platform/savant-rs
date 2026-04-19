//! Benchmarks for NvInfer with identity, age_gender, and yolo11m-seg models,
//! using both **uniform** (all frames same resolution) and **non-uniform**
//! (heterogeneous frame resolutions) batched surfaces.
//!
//! Each (model, batch_size) pair gets its own nvinfer config so TensorRT builds
//! a dedicated engine optimised for that exact batch size.  Engine files are
//! cached in `assets/engines/<platform>/`; only the first run pays the build cost.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use deepstream_buffers::{
    BufferGenerator, ComputeMode, NonUniformBatch, NvBufSurfaceMemType, SavantIdMetaKind,
    SurfaceView, TransformConfig, UniformBatchGenerator, VideoFormat,
};
use deepstream_nvinfer::{ModelColorFormat, ModelInputScaling, NvInfer, NvInferConfig};
use rand::Rng;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex, Once};
use std::time::Duration;

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
        deepstream_buffers::cuda_init(0).expect("CUDA init — is a GPU available?");
    });
}

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets")
}

fn engines_dir() -> PathBuf {
    let tag = nvidia_gpu_utils::gpu_platform_tag(0).unwrap_or_else(|_| "unknown".to_string());
    let dir = assets_dir().join("engines").join(tag);
    std::fs::create_dir_all(&dir).expect("create engine cache directory");
    dir
}

fn promote_built_engine(onnx_stem: &str, batch_size: u32) {
    let engine_name = format!("{}_b{}_gpu0_fp16.engine", onnx_stem, batch_size);
    let auto_path = assets_dir().join(&engine_name);
    let cached_path = engines_dir().join(&engine_name);
    if auto_path.exists() && !cached_path.exists() {
        std::fs::rename(&auto_path, &cached_path).unwrap_or_else(|_| {
            std::fs::copy(&auto_path, &cached_path).expect("copy engine to platform cache");
            std::fs::remove_file(&auto_path).ok();
        });
    }
}

fn platform_transform_config() -> TransformConfig {
    let mut config = TransformConfig::default();
    if cfg!(target_arch = "aarch64") {
        config.compute_mode = ComputeMode::Gpu;
    }
    config
}

fn inject_jetson_scaling(props: &mut HashMap<String, String>) {
    if cfg!(target_arch = "aarch64") {
        props.insert("scaling-compute-hw".into(), "1".into());
    }
}

/// On Jetson (shared CPU/GPU memory), cap workspace-size to avoid TensorRT
/// building degenerate engines when it cannot allocate the requested amount.
fn inject_jetson_workspace_cap(props: &mut HashMap<String, String>, max_mb: u32) {
    if cfg!(target_arch = "aarch64") {
        if let Some(current) = props.get("workspace-size") {
            if let Ok(val) = current.parse::<u32>() {
                if val > max_mb {
                    props.insert("workspace-size".into(), max_mb.to_string());
                }
            }
        }
    }
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
    model_width: u32,
    model_height: u32,
    group_name: String,
    scaling: ModelInputScaling,
}

fn identity_base_properties(dir: &Path) -> HashMap<String, String> {
    let mut m = HashMap::new();
    m.insert("gpu-id".into(), "0".into());
    m.insert("net-scale-factor".into(), "1.0".into());
    m.insert(
        "onnx-file".into(),
        dir.join("identity_3x112x112.onnx").to_string_lossy().into(),
    );
    m.insert("network-mode".into(), "2".into());
    inject_jetson_scaling(&mut m);
    m
}

fn age_gender_base_properties(dir: &Path) -> HashMap<String, String> {
    let mut m = HashMap::new();
    m.insert("gpu-id".into(), "0".into());
    m.insert("net-scale-factor".into(), "0.007843137254902".into());
    m.insert("offsets".into(), "127.5;127.5;127.5".into());
    m.insert(
        "onnx-file".into(),
        dir.join("age_gender_mobilenet_v2_dynBatch.onnx")
            .to_string_lossy()
            .into(),
    );
    m.insert("network-mode".into(), "2".into());
    inject_jetson_scaling(&mut m);
    m
}

fn yolo_base_properties(dir: &Path) -> HashMap<String, String> {
    let mut m = HashMap::new();
    m.insert(
        "onnx-file".into(),
        dir.join("yolo11m-seg.onnx").to_string_lossy().into(),
    );
    m.insert("gpu-id".into(), "0".into());
    m.insert("network-mode".into(), "2".into());
    m.insert("workspace-size".into(), "6144".into());
    m.insert("net-scale-factor".into(), "0.003921569790691137".into());
    m.insert("offsets".into(), "0.0;0.0;0.0".into());
    m.insert("output-blob-names".into(), "output0;output1".into());
    inject_jetson_scaling(&mut m);
    inject_jetson_workspace_cap(&mut m, 2048);
    m
}

fn yolo11n_base_properties(dir: &Path) -> HashMap<String, String> {
    let mut m = HashMap::new();
    m.insert(
        "onnx-file".into(),
        dir.join("yolo11n.onnx").to_string_lossy().into(),
    );
    m.insert("gpu-id".into(), "0".into());
    m.insert("network-mode".into(), "2".into());
    m.insert("workspace-size".into(), "6144".into());
    m.insert("net-scale-factor".into(), "0.003921569790691137".into());
    m.insert("offsets".into(), "0.0;0.0;0.0".into());
    m.insert("output-blob-names".into(), "output0".into());
    inject_jetson_scaling(&mut m);
    inject_jetson_workspace_cap(&mut m, 2048);
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
) -> deepstream_buffers::SharedBuffer {
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
) -> deepstream_buffers::SharedBuffer {
    init();

    let min_bufs = batch_size.max(4);
    let src_gen = BufferGenerator::builder(format, w, h)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(min_bufs)
        .max_buffers(min_bufs)
        .build()
        .expect("src generator");

    let batched_gen =
        UniformBatchGenerator::new(format, w, h, batch_size, 2, 0, NvBufSurfaceMemType::Default)
            .expect("batched generator");

    let config = platform_transform_config();
    let ids: Vec<SavantIdMetaKind> = (0..batch_size)
        .map(|i| SavantIdMetaKind::Frame(i as u128))
        .collect();
    let mut batch = batched_gen.acquire_batch(config, ids).unwrap();

    for i in 0..batch_size {
        let src = src_gen.acquire(Some(i as u128)).unwrap();
        let src_view = SurfaceView::from_buffer(&src, 0).unwrap();
        batch.transform_slot(i, &src_view, None).unwrap();
    }

    batch.finalize().unwrap();
    batch.shared_buffer()
}

fn make_nonuniform_batch(
    format: VideoFormat,
    base_w: u32,
    base_h: u32,
    batch_size: u32,
) -> deepstream_buffers::SharedBuffer {
    init();

    let mut keepalive = Vec::with_capacity(batch_size as usize);
    for i in 0..batch_size as usize {
        let s = NON_UNIFORM_SCALES[i % NON_UNIFORM_SCALES.len()];
        let w = ((base_w as f32 * s) as u32).max(4);
        let h = ((base_h as f32 * s) as u32).max(4);

        let gen = BufferGenerator::builder(format, w, h)
            .gpu_id(0)
            .mem_type(NvBufSurfaceMemType::Default)
            .min_buffers(1)
            .max_buffers(1)
            .build()
            .expect("src generator for nonuniform slot");
        let shared = gen.acquire(Some(i as u128)).unwrap();
        let view = SurfaceView::from_buffer(&shared, 0).unwrap();
        keepalive.push((shared, view));
    }

    let mut batch = NonUniformBatch::new(0);
    let mut ids = Vec::new();
    for (i, (_shared, view)) in keepalive.iter().enumerate() {
        batch.add(view).unwrap();
        ids.push(SavantIdMetaKind::Frame(i as u128));
    }

    batch.finalize(ids).unwrap()
}

// ---------------------------------------------------------------------------
// Benchmark driver
// ---------------------------------------------------------------------------

const FILL_COUNTS: &[u32] = &[1, 4, 8, 16];
const BATCHES_PER_ITER: u32 = 100;

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
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(30));

    for &bs in batch_sizes {
        let mut props = spec.base_properties.clone();
        props.insert("batch-size".into(), bs.to_string());
        props.insert(
            "model-engine-file".into(),
            engines_dir()
                .join(format!("{}_b{}_gpu0_fp16.engine", spec.onnx_stem, bs))
                .to_string_lossy()
                .into(),
        );

        let config = NvInferConfig::new(
            props,
            VideoFormat::RGBA,
            spec.model_width,
            spec.model_height,
            ModelColorFormat::RGB,
        )
        .scaling(spec.scaling);
        let engine = NvInfer::new(config, Box::new(|_| {})).expect("create NvInfer for bench");
        promote_built_engine(&spec.onnx_stem, bs);

        let warm = make_batch(mode, spec.format, spec.width, spec.height, bs);
        let _ = engine.infer_sync(warm, None);

        let fills: Vec<u32> = FILL_COUNTS.iter().copied().filter(|&f| f <= bs).collect();
        let n = BATCHES_PER_ITER;

        for &fill in &fills {
            let id = format!("bs{}/fill{}/x{}", bs, fill, n);
            group.bench_with_input(BenchmarkId::new(&id, fill), &fill, |b, &fill| {
                b.iter_batched(
                    || {
                        (0..n)
                            .map(|_| make_batch(mode, spec.format, spec.width, spec.height, fill))
                            .collect::<Vec<_>>()
                    },
                    |batches| {
                        for batch in batches {
                            let out = engine.infer_sync(batch, None).expect("infer_sync");
                            std::hint::black_box(out);
                        }
                    },
                    criterion::BatchSize::LargeInput,
                )
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
            onnx_path: dir.join("identity_3x112x112.onnx"),
            onnx_stem: "identity_3x112x112.onnx".into(),
            base_properties: identity_base_properties(&dir),
            format: VideoFormat::RGBA,
            width: 112,
            height: 112,
            model_width: 112,
            model_height: 112,
            group_name: "identity".into(),
            scaling: ModelInputScaling::Fill,
        },
        ModelSpec {
            onnx_path: dir.join("age_gender_mobilenet_v2_dynBatch.onnx"),
            onnx_stem: "age_gender_mobilenet_v2_dynBatch.onnx".into(),
            base_properties: age_gender_base_properties(&dir),
            format: VideoFormat::RGBA,
            width: 112,
            height: 112,
            model_width: 112,
            model_height: 112,
            group_name: "age_gender".into(),
            scaling: ModelInputScaling::Fill,
        },
        ModelSpec {
            onnx_path: dir.join("yolo11m-seg.onnx"),
            onnx_stem: "yolo11m-seg.onnx".into(),
            base_properties: yolo_base_properties(&dir),
            format: VideoFormat::RGBA,
            width: 640,
            height: 640,
            model_width: 640,
            model_height: 640,
            group_name: "yolo11m_seg".into(),
            scaling: ModelInputScaling::KeepAspectRatio,
        },
        ModelSpec {
            onnx_path: dir.join("yolo11n.onnx"),
            onnx_stem: "yolo11n.onnx".into(),
            base_properties: yolo11n_base_properties(&dir),
            format: VideoFormat::RGBA,
            width: 640,
            height: 640,
            model_width: 640,
            model_height: 640,
            group_name: "yolo11n".into(),
            scaling: ModelInputScaling::KeepAspectRatioSymmetric,
        },
    ];

    let batch_sizes: &[&[u32]] = &[&[1], &[1, 4, 16], &[1], &[1]];
    let modes = [BatchMode::Uniform, BatchMode::NonUniform];

    for mode in modes {
        for (spec, sizes) in models.iter().zip(batch_sizes.iter()) {
            bench_model(c, spec, sizes, mode);
        }
    }
}

// ---------------------------------------------------------------------------
// Random non-uniform batch benchmark (age-gender model, iter_batched)
// ---------------------------------------------------------------------------

const RANDOM_NONUNIFORM_BATCHES_PER_ITER: u32 = BATCHES_PER_ITER;
const RANDOM_NONUNIFORM_FRAMES: u32 = 4;
const RANDOM_SIZE_MIN: u32 = 64;
const RANDOM_SIZE_MAX_INCL: u32 = 256;
const RANDOM_SIZE_STEP: u32 = 4;

fn make_random_nonuniform_batch(rng: &mut impl Rng) -> deepstream_buffers::SharedBuffer {
    let n = RANDOM_NONUNIFORM_FRAMES;
    let steps = (RANDOM_SIZE_MAX_INCL - RANDOM_SIZE_MIN) / RANDOM_SIZE_STEP + 1;

    let mut keepalive = Vec::with_capacity(n as usize);
    for i in 0..n {
        let w = RANDOM_SIZE_MIN + rng.random_range(0..steps) * RANDOM_SIZE_STEP;
        let h = RANDOM_SIZE_MIN + rng.random_range(0..steps) * RANDOM_SIZE_STEP;

        let gen = BufferGenerator::builder(VideoFormat::RGBA, w, h)
            .gpu_id(0)
            .mem_type(NvBufSurfaceMemType::Default)
            .min_buffers(1)
            .max_buffers(1)
            .build()
            .expect("src generator for random nonuniform slot");
        let shared = gen.acquire(Some(i as u128)).unwrap();
        let view = SurfaceView::from_buffer(&shared, 0).unwrap();
        keepalive.push((shared, view));
    }

    let mut batch = NonUniformBatch::new(0);
    let mut ids = Vec::new();
    for (i, (_shared, view)) in keepalive.iter().enumerate() {
        batch.add(view).unwrap();
        ids.push(SavantIdMetaKind::Frame(i as u128));
    }

    batch.finalize(ids).unwrap()
}

fn bench_random_nonuniform_age_gender(c: &mut Criterion) {
    init();
    let dir = assets_dir();
    let onnx_path = dir.join("age_gender_mobilenet_v2_dynBatch.onnx");
    if !onnx_path.exists() {
        eprintln!(
            "Skipping random_nonuniform_age_gender benchmark: ONNX not found at {:?}",
            onnx_path
        );
        return;
    }

    let n = RANDOM_NONUNIFORM_BATCHES_PER_ITER;
    let mut group = c.benchmark_group("age_gender/random_nonuniform");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(30));

    let done_count = Arc::new(AtomicU32::new(0));
    let notify = Arc::new((Mutex::new(()), Condvar::new()));

    let cb_done = Arc::clone(&done_count);
    let cb_notify = Arc::clone(&notify);
    let callback = Box::new(move |_output| {
        cb_done.fetch_add(1, Ordering::Release);
        let (lock, cvar) = &*cb_notify;
        let _guard = lock.lock().unwrap();
        cvar.notify_one();
    });

    let mut props = age_gender_base_properties(&dir);
    props.insert("batch-size".into(), RANDOM_NONUNIFORM_FRAMES.to_string());
    props.insert(
        "model-engine-file".into(),
        engines_dir()
            .join(format!(
                "age_gender_mobilenet_v2_dynBatch.onnx_b{}_gpu0_fp16.engine",
                RANDOM_NONUNIFORM_FRAMES
            ))
            .to_string_lossy()
            .into(),
    );

    let config = NvInferConfig::new(props, VideoFormat::RGBA, 112, 112, ModelColorFormat::RGB);
    let engine = NvInfer::new(config, callback).expect("create NvInfer for random bench");
    promote_built_engine(
        "age_gender_mobilenet_v2_dynBatch.onnx",
        RANDOM_NONUNIFORM_FRAMES,
    );

    // Warmup with one batch.
    let mut rng = rand::rng();
    done_count.store(0, Ordering::Release);
    let warmup = make_random_nonuniform_batch(&mut rng);
    engine.submit(warmup, None).expect("warmup submit");
    {
        let (lock, cvar) = &*notify;
        let mut guard = lock.lock().unwrap();
        while done_count.load(Ordering::Acquire) < 1 {
            guard = cvar.wait(guard).unwrap();
        }
    }

    let id = format!("x{}_bs{}", n, RANDOM_NONUNIFORM_FRAMES);
    group.bench_function(BenchmarkId::new(&id, 0), |b| {
        b.iter_batched(
            || {
                let mut rng = rand::rng();
                (0..n)
                    .map(|_| make_random_nonuniform_batch(&mut rng))
                    .collect::<Vec<_>>()
            },
            |batches| {
                done_count.store(0, Ordering::Release);

                for batch in batches {
                    engine.submit(batch, None).expect("submit");
                }

                let (lock, cvar) = &*notify;
                let mut guard = lock.lock().unwrap();
                while done_count.load(Ordering::Acquire) < n {
                    guard = cvar.wait(guard).unwrap();
                }
            },
            criterion::BatchSize::LargeInput,
        )
    });

    group.finish();
}

fn bench_random_nonuniform_frame_size_age_gender_sync(c: &mut Criterion) {
    init();
    let dir = assets_dir();
    let onnx_path = dir.join("age_gender_mobilenet_v2_dynBatch.onnx");
    if !onnx_path.exists() {
        eprintln!(
            "Skipping random_nonuniform_age_gender_sync benchmark: ONNX not found at {:?}",
            onnx_path
        );
        return;
    }

    let n = RANDOM_NONUNIFORM_BATCHES_PER_ITER;
    let mut group = c.benchmark_group("age_gender/random_nonuniform_sync");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(30));

    let mut props = age_gender_base_properties(&dir);
    props.insert("batch-size".into(), RANDOM_NONUNIFORM_FRAMES.to_string());
    props.insert(
        "model-engine-file".into(),
        engines_dir()
            .join(format!(
                "age_gender_mobilenet_v2_dynBatch.onnx_b{}_gpu0_fp16.engine",
                RANDOM_NONUNIFORM_FRAMES
            ))
            .to_string_lossy()
            .into(),
    );

    let config = NvInferConfig::new(props, VideoFormat::RGBA, 112, 112, ModelColorFormat::RGB);
    let engine = NvInfer::new(config, Box::new(|_| {})).expect("create NvInfer for sync bench");
    promote_built_engine(
        "age_gender_mobilenet_v2_dynBatch.onnx",
        RANDOM_NONUNIFORM_FRAMES,
    );

    // Warmup.
    let mut rng = rand::rng();
    let warmup = make_random_nonuniform_batch(&mut rng);
    let _ = engine.infer_sync(warmup, None);

    let id = format!("x{}_bs{}_sync", n, RANDOM_NONUNIFORM_FRAMES);
    group.bench_function(BenchmarkId::new(&id, 0), |b| {
        b.iter_batched(
            || {
                let mut rng = rand::rng();
                (0..n)
                    .map(|_| make_random_nonuniform_batch(&mut rng))
                    .collect::<Vec<_>>()
            },
            |batches| {
                for batch in batches {
                    let out = engine.infer_sync(batch, None).expect("infer_sync");
                    std::hint::black_box(out);
                }
            },
            criterion::BatchSize::LargeInput,
        )
    });

    group.finish();
}

criterion_group!(benches, bench_sync_batch_sizes);

criterion_group!(
    bench_random_nonuniform_async,
    bench_random_nonuniform_age_gender
);

criterion_group!(
    bench_random_nonuniform_sync,
    bench_random_nonuniform_frame_size_age_gender_sync
);

criterion_main!(
    benches,
    bench_random_nonuniform_async,
    bench_random_nonuniform_sync
);
