//! Criterion benchmarks for encoder bootstrap and single-frame roundtrip.
//!
//! Measures:
//! - **creation**: time to construct an `NvEncoder` (pipeline build, NVENC
//!   session acquisition, buffer pool creation).
//! - **creation + 1 frame**: creation plus submitting one FullHD frame,
//!   calling `finish()`, and receiving the encoded bitstream.
//!
//! Codecs: H.264, HEVC, AV1 (low-latency mode) and JPEG.
//!
//! Run with:
//! ```sh
//! cargo bench -p deepstream_encoders --bench encoder_bench
//! ```

use criterion::{criterion_group, criterion_main, Criterion};
use deepstream_encoders::properties::*;
use deepstream_encoders::{cuda_init, Codec, EncoderConfig, NvEncoder, VideoFormat};
use std::sync::Once;

static INIT: Once = Once::new();

fn ensure_init() {
    INIT.call_once(|| {
        gstreamer::init().expect("GStreamer init failed");
        cuda_init(0).expect("CUDA init failed");
    });
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
        .format(VideoFormat::NV12)
        .properties(props)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create the encoder, submit one frame, finish, return encoded output.
fn create_and_encode_one(config: &EncoderConfig) {
    let mut encoder = NvEncoder::new(config).expect("NvEncoder::new failed");

    let buffer = encoder
        .generator()
        .acquire_surface(Some(0))
        .expect("acquire_surface failed");

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

    group.bench_function("jpeg", |b| {
        let config = jpeg_config();
        b.iter(|| create_and_encode_one(&config));
    });

    group.finish();
}

criterion_group!(benches, bench_creation_plus_one_frame);
criterion_main!(benches);
