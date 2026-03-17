//! Criterion benchmarks for the NvEncoder.
//!
//! Measures:
//! - **creation + 1 frame**: time to construct an `NvEncoder`, submit one
//!   FullHD frame, call `finish()`, and receive the encoded bitstream.
//! - **200-frame throughput**: time to encode 200 FullHD frames with a
//!   pre-created encoder (creation cost excluded from measurement).
//!
//! Codecs: H.264, HEVC, AV1 (low-latency mode) and JPEG.
//! Benchmarks that need NVENC or nvjpegenc are skipped at runtime when
//! the hardware is not available.
//!
//! Run with:
//! ```sh
//! cargo bench -p deepstream_encoders --bench encoder_bench
//! ```

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use deepstream_encoders::prelude::*;
use std::sync::Once;

static INIT: Once = Once::new();

fn ensure_init() {
    INIT.call_once(|| {
        gstreamer::init().expect("GStreamer init failed");
        cuda_init(0).expect("CUDA init failed");
    });
}

fn has_nvenc() -> bool {
    nvidia_gpu_utils::has_nvenc(0).unwrap_or(false)
}

fn has_nvjpegenc() -> bool {
    gstreamer::ElementFactory::find("nvjpegenc").is_some()
}

// ---------------------------------------------------------------------------
// Encoder configurations
// ---------------------------------------------------------------------------

fn h264_low_latency_config() -> EncoderConfig {
    let props = EncoderProperties::H264Dgpu(H264DgpuProps {
        preset: Some(DgpuPreset::P1),
        tuning_info: Some(TuningPreset::LowLatency),
        ..Default::default()
    });
    EncoderConfig::new(Codec::H264, 1920, 1080)
        .format(VideoFormat::NV12)
        .properties(props)
}

fn hevc_low_latency_config() -> EncoderConfig {
    let props = EncoderProperties::HevcDgpu(HevcDgpuProps {
        preset: Some(DgpuPreset::P1),
        tuning_info: Some(TuningPreset::LowLatency),
        ..Default::default()
    });
    EncoderConfig::new(Codec::Hevc, 1920, 1080)
        .format(VideoFormat::NV12)
        .properties(props)
}

fn av1_low_latency_config() -> EncoderConfig {
    let props = EncoderProperties::Av1Dgpu(Av1DgpuProps {
        preset: Some(DgpuPreset::P1),
        tuning_info: Some(TuningPreset::LowLatency),
        ..Default::default()
    });
    EncoderConfig::new(Codec::Av1, 1920, 1080)
        .format(VideoFormat::NV12)
        .properties(props)
}

fn jpeg_config() -> EncoderConfig {
    let props = EncoderProperties::Jpeg(JpegProps { quality: Some(85) });
    EncoderConfig::new(Codec::Jpeg, 1920, 1080)
        .format(VideoFormat::I420)
        .properties(props)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create the encoder, submit one frame, finish, return encoded output.
fn create_and_encode_one(config: &EncoderConfig) {
    let mut encoder = NvEncoder::new(config).expect("NvEncoder::new failed");

    let shared = encoder
        .generator()
        .acquire_buffer(Some(0))
        .expect("acquire_surface failed");
    let buffer = shared.into_buffer().expect("sole owner");

    encoder
        .submit_frame(buffer, 0, 0, Some(33_333_333))
        .expect("submit_frame failed");

    let frames = encoder.finish(Some(5000)).expect("finish failed");
    assert!(!frames.is_empty(), "Expected at least 1 encoded frame");
}

// ---------------------------------------------------------------------------
// Benchmarks: creation + one-frame roundtrip
// ---------------------------------------------------------------------------

fn bench_creation_plus_one_frame(c: &mut Criterion) {
    ensure_init();

    let mut group = c.benchmark_group("encoder_creation_plus_one_frame");
    group.sample_size(30);

    if has_nvenc() {
        group.bench_function("h264_low_latency", |b| {
            let config = h264_low_latency_config();
            b.iter(|| create_and_encode_one(&config));
        });

        group.bench_function("hevc_low_latency", |b| {
            let config = hevc_low_latency_config();
            b.iter(|| create_and_encode_one(&config));
        });

        group.bench_function("av1_low_latency", |b| {
            let config = av1_low_latency_config();
            b.iter(|| create_and_encode_one(&config));
        });
    } else {
        eprintln!("NVENC not available — skipping h264/hevc/av1 benchmarks");
    }

    if has_nvjpegenc() {
        group.bench_function("jpeg", |b| {
            let config = jpeg_config();
            b.iter(|| create_and_encode_one(&config));
        });
    } else {
        eprintln!("nvjpegenc not available — skipping jpeg benchmark");
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmarks: 200-frame throughput (encoder pre-created)
// ---------------------------------------------------------------------------

const THROUGHPUT_FRAMES: u64 = 200;
const FRAME_DURATION_NS: u64 = 33_333_333; // ~30 fps

/// Encode `THROUGHPUT_FRAMES` frames on a pre-created encoder, draining
/// output after each submit to keep the single-buffer pool flowing.
fn encode_n_frames(mut encoder: NvEncoder) {
    for i in 0..THROUGHPUT_FRAMES {
        let shared = encoder
            .generator()
            .acquire_buffer(Some(i as i64))
            .expect("acquire_surface failed");
        let buffer = shared.into_buffer().expect("sole owner");

        let pts_ns = i * FRAME_DURATION_NS;
        encoder
            .submit_frame(buffer, i as u128, pts_ns, Some(FRAME_DURATION_NS))
            .expect("submit_frame failed");

        while let Ok(Some(_)) = encoder.pull_encoded() {}
    }

    let remaining = encoder.finish(Some(5000)).expect("finish failed");
    let drained = remaining.len() as u64;
    assert!(drained > 0 || THROUGHPUT_FRAMES == 0);
}

fn bench_200_frame_throughput(c: &mut Criterion) {
    ensure_init();

    let mut group = c.benchmark_group("encoder_200_frame_throughput");
    group.sample_size(10);
    group.throughput(criterion::Throughput::Elements(THROUGHPUT_FRAMES));

    if has_nvenc() {
        group.bench_function("h264_low_latency", |b| {
            let config = h264_low_latency_config();
            b.iter_batched(
                || NvEncoder::new(&config).expect("NvEncoder::new failed"),
                encode_n_frames,
                BatchSize::PerIteration,
            );
        });

        group.bench_function("hevc_low_latency", |b| {
            let config = hevc_low_latency_config();
            b.iter_batched(
                || NvEncoder::new(&config).expect("NvEncoder::new failed"),
                encode_n_frames,
                BatchSize::PerIteration,
            );
        });

        group.bench_function("av1_low_latency", |b| {
            let config = av1_low_latency_config();
            b.iter_batched(
                || NvEncoder::new(&config).expect("NvEncoder::new failed"),
                encode_n_frames,
                BatchSize::PerIteration,
            );
        });
    } else {
        eprintln!("NVENC not available — skipping h264/hevc/av1 throughput benchmarks");
    }

    if has_nvjpegenc() {
        group.bench_function("jpeg", |b| {
            let config = jpeg_config();
            b.iter_batched(
                || NvEncoder::new(&config).expect("NvEncoder::new failed"),
                encode_n_frames,
                BatchSize::PerIteration,
            );
        });
    } else {
        eprintln!("nvjpegenc not available — skipping jpeg throughput benchmark");
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_creation_plus_one_frame,
    bench_200_frame_throughput
);
criterion_main!(benches);
