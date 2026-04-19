//! Criterion benchmarks for the channel-based [`NvEncoder`].
//!
//! Two measurements per codec:
//! - **creation + 1 frame**: construct an [`NvEncoder`], submit one
//!   FullHD frame, `graceful_shutdown`, receive the encoded bitstream.
//! - **200-frame throughput**: encode 200 FullHD frames on a pre-created
//!   encoder (creation cost excluded via `iter_batched`).
//!
//! # Platform awareness
//!
//! The benchmark does NOT rely on `target_arch = "aarch64"` as a proxy
//! for "Jetson". Instead it uses runtime probes from
//! [`nvidia_gpu_utils`]:
//!
//! - **NVENC availability** ([`nvidia_gpu_utils::has_nvenc`]) — correctly
//!   reports `false` on Jetson Orin Nano and on datacenter dGPUs without
//!   NVENC (A100, H100, B200, B300, GB200, …). Benchmarks for H.264,
//!   HEVC and AV1 are skipped at runtime when NVENC is unavailable.
//! - **nvjpegenc availability** — checked via GStreamer element factory.
//! - **Jetson kernel / model** ([`nvidia_gpu_utils::is_jetson_kernel`],
//!   [`nvidia_gpu_utils::jetson_model`]) — used for the platform summary
//!   line and to pick the matching preset when low-latency tuning
//!   differs between Jetson and dGPU.
//!
//! The RAW (`RawRgba`) and PNG codecs require only CUDA + nvvideoconvert
//! and therefore run on **every** NVIDIA GPU including A100/H100/B300
//! and Orin Nano; they broaden coverage on platforms where NVENC and
//! nvjpegenc are unavailable.
//!
//! > Note on compile-time `#[cfg(target_arch = "aarch64")]`: the core
//! > `deepstream_encoders` API exposes different *property struct types*
//! > for Jetson (`H264JetsonProps`, …) and dGPU (`H264DgpuProps`, …),
//! > so `.props(...)` call sites must select the correct type at
//! > compile time. This is structural and mirrors the decoders crate.
//! > Crates that cross-compile to aarch64 but run on an ARM dGPU server
//! > (e.g. Grace Hopper) are out of scope.
//!
//! Run with:
//! ```sh
//! cargo bench -p savant-deepstream-encoders --bench encoder_bench
//! ```

use std::sync::Once;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use deepstream_encoders::prelude::*;

// ─── Runtime platform probes ─────────────────────────────────────────

static INIT: Once = Once::new();

fn ensure_init() {
    INIT.call_once(|| {
        gstreamer::init().expect("GStreamer init failed");
        cuda_init(0).expect("CUDA init failed");
        log_platform_summary();
    });
}

fn has_nvenc() -> bool {
    nvidia_gpu_utils::has_nvenc(0).unwrap_or(false)
}

fn has_nvjpegenc() -> bool {
    gstreamer::ElementFactory::find("nvjpegenc").is_some()
}

fn is_jetson() -> bool {
    nvidia_gpu_utils::is_jetson_kernel()
}

fn platform_tag() -> String {
    nvidia_gpu_utils::gpu_platform_tag(0).unwrap_or_else(|_| "unknown".to_string())
}

fn log_platform_summary() {
    eprintln!("──────────────── encoder_bench platform summary ────────────────");
    eprintln!("  platform tag   : {}", platform_tag());
    eprintln!("  jetson kernel  : {}", is_jetson());
    if let Ok(Some(model)) = nvidia_gpu_utils::jetson_model(0) {
        eprintln!("  jetson model   : {model:?}");
    }
    eprintln!("  NVENC available: {}", has_nvenc());
    eprintln!("  nvjpegenc avail: {}", has_nvjpegenc());
    eprintln!("  NOTE: Orin Nano and datacenter dGPUs (A100/H100/B200/B300/GB200)");
    eprintln!("        do not have NVENC; h264/hevc/av1 benches will be skipped.");
    eprintln!("        RAW and PNG benches run on any CUDA-capable GPU.");
    eprintln!("────────────────────────────────────────────────────────────────");
}

// ─── Encoder configurations ──────────────────────────────────────────
//
// The `.props(...)` call site must match the compile-time type chosen
// by `deepstream_encoders::config` (H264JetsonProps on aarch64,
// H264DgpuProps on x86_64). This is a structural requirement.

fn h264_low_latency_config() -> EncoderConfig {
    #[cfg(target_arch = "aarch64")]
    let enc_cfg = H264EncoderConfig::new(1920, 1080)
        .format(VideoFormat::RGBA)
        .props(H264JetsonProps {
            preset_level: Some(JetsonPresetLevel::UltraFast),
            ..Default::default()
        });
    #[cfg(not(target_arch = "aarch64"))]
    let enc_cfg = H264EncoderConfig::new(1920, 1080)
        .format(VideoFormat::RGBA)
        .props(H264DgpuProps {
            preset: Some(DgpuPreset::P1),
            tuning_info: Some(TuningPreset::LowLatency),
            ..Default::default()
        });
    EncoderConfig::H264(enc_cfg)
}

fn hevc_low_latency_config() -> EncoderConfig {
    #[cfg(target_arch = "aarch64")]
    let enc_cfg = HevcEncoderConfig::new(1920, 1080)
        .format(VideoFormat::RGBA)
        .props(HevcJetsonProps {
            preset_level: Some(JetsonPresetLevel::UltraFast),
            ..Default::default()
        });
    #[cfg(not(target_arch = "aarch64"))]
    let enc_cfg = HevcEncoderConfig::new(1920, 1080)
        .format(VideoFormat::RGBA)
        .props(HevcDgpuProps {
            preset: Some(DgpuPreset::P1),
            tuning_info: Some(TuningPreset::LowLatency),
            ..Default::default()
        });
    EncoderConfig::Hevc(enc_cfg)
}

fn av1_low_latency_config() -> EncoderConfig {
    #[cfg(target_arch = "aarch64")]
    let enc_cfg = Av1EncoderConfig::new(1920, 1080)
        .format(VideoFormat::RGBA)
        .props(Av1JetsonProps {
            preset_level: Some(JetsonPresetLevel::UltraFast),
            ..Default::default()
        });
    #[cfg(not(target_arch = "aarch64"))]
    let enc_cfg = Av1EncoderConfig::new(1920, 1080)
        .format(VideoFormat::RGBA)
        .props(Av1DgpuProps {
            preset: Some(DgpuPreset::P1),
            tuning_info: Some(TuningPreset::LowLatency),
            ..Default::default()
        });
    EncoderConfig::Av1(enc_cfg)
}

fn jpeg_config() -> EncoderConfig {
    EncoderConfig::Jpeg(
        JpegEncoderConfig::new(1920, 1080)
            .format(VideoFormat::RGBA)
            .props(JpegProps { quality: Some(85) }),
    )
}

fn png_config() -> EncoderConfig {
    EncoderConfig::Png(
        PngEncoderConfig::new(1920, 1080)
            .format(VideoFormat::RGBA)
            .props(PngProps {
                compression_level: Some(1),
            }),
    )
}

fn raw_rgba_config() -> EncoderConfig {
    EncoderConfig::RawRgba(RawEncoderConfig::new(1920, 1080, VideoFormat::RGBA))
}

fn make_encoder(encoder_cfg: EncoderConfig) -> NvEncoder {
    let cfg = NvEncoderConfig::new(0, encoder_cfg)
        .operation_timeout(Duration::from_secs(10))
        .name("bench");
    NvEncoder::new(cfg).expect("NvEncoder::new failed")
}

// ─── Helpers ─────────────────────────────────────────────────────────

fn acquire_frame(encoder: &NvEncoder, id: u128) -> gstreamer::Buffer {
    let shared = encoder
        .generator()
        .lock()
        .acquire(Some(id))
        .expect("acquire failed");
    // Creating a SurfaceView resolves the CUDA pointer and, on Jetson,
    // performs the one-time EGL-CUDA registration for this pool slot.
    let _view = SurfaceView::from_buffer(&shared, 0).expect("SurfaceView failed");
    drop(_view);
    shared.into_buffer().expect("sole owner")
}

fn drain_graceful(encoder: &NvEncoder) -> usize {
    let mut count = 0usize;
    encoder
        .graceful_shutdown(Some(Duration::from_secs(10)), |out| {
            if let NvEncoderOutput::Frame(_) = out {
                count += 1;
            }
        })
        .expect("graceful_shutdown failed");
    count
}

fn create_and_encode_one(config: EncoderConfig) {
    let encoder = make_encoder(config);
    let buf = acquire_frame(&encoder, 0);
    encoder
        .submit_frame(buf, 0, 0, Some(33_333_333))
        .expect("submit_frame failed");
    let n = drain_graceful(&encoder);
    assert!(n > 0, "Expected at least 1 encoded frame");
}

// ─── Creation + one-frame roundtrip ──────────────────────────────────

fn bench_creation_plus_one_frame(c: &mut Criterion) {
    ensure_init();
    let mut group = c.benchmark_group("encoder_creation_plus_one_frame");
    group.sample_size(30);

    if has_nvenc() {
        group.bench_function("h264_low_latency", |b| {
            b.iter(|| create_and_encode_one(h264_low_latency_config()));
        });
        group.bench_function("hevc_low_latency", |b| {
            b.iter(|| create_and_encode_one(hevc_low_latency_config()));
        });
        group.bench_function("av1_low_latency", |b| {
            b.iter(|| create_and_encode_one(av1_low_latency_config()));
        });
    } else {
        eprintln!(
            "NVENC not available on this GPU — skipping h264/hevc/av1 benches \
             (expected on Orin Nano and on datacenter dGPUs like A100/H100/B300)"
        );
    }

    if has_nvjpegenc() {
        group.bench_function("jpeg", |b| {
            b.iter(|| create_and_encode_one(jpeg_config()));
        });
    } else {
        eprintln!("nvjpegenc not available — skipping jpeg bench");
    }

    // PNG and RAW do not require NVENC/nvjpegenc; they run on every
    // CUDA-capable GPU, including Orin Nano and A100/H100/B300.
    group.bench_function("png", |b| {
        b.iter(|| create_and_encode_one(png_config()));
    });
    group.bench_function("raw_rgba", |b| {
        b.iter(|| create_and_encode_one(raw_rgba_config()));
    });

    group.finish();
}

// ─── 200-frame throughput ────────────────────────────────────────────

const THROUGHPUT_FRAMES: u64 = 200;
const FRAME_DURATION_NS: u64 = 33_333_333;

fn encode_n_frames(encoder: NvEncoder) {
    for i in 0..THROUGHPUT_FRAMES {
        let buf = acquire_frame(&encoder, i as u128);
        let pts_ns = i * FRAME_DURATION_NS;
        encoder
            .submit_frame(buf, i as u128, pts_ns, Some(FRAME_DURATION_NS))
            .expect("submit_frame failed");
        while let Ok(Some(_)) = encoder.try_recv() {}
    }
    let drained = drain_graceful(&encoder) as u64;
    assert!(drained > 0 || THROUGHPUT_FRAMES == 0);
}

fn bench_200_frame_throughput(c: &mut Criterion) {
    ensure_init();
    let mut group = c.benchmark_group("encoder_200_frame_throughput");
    group.sample_size(10);
    group.throughput(criterion::Throughput::Elements(THROUGHPUT_FRAMES));

    if has_nvenc() {
        group.bench_function("h264_low_latency", |b| {
            b.iter_batched(
                || make_encoder(h264_low_latency_config()),
                encode_n_frames,
                BatchSize::PerIteration,
            );
        });
        group.bench_function("hevc_low_latency", |b| {
            b.iter_batched(
                || make_encoder(hevc_low_latency_config()),
                encode_n_frames,
                BatchSize::PerIteration,
            );
        });
        group.bench_function("av1_low_latency", |b| {
            b.iter_batched(
                || make_encoder(av1_low_latency_config()),
                encode_n_frames,
                BatchSize::PerIteration,
            );
        });
    } else {
        eprintln!(
            "NVENC not available on this GPU — skipping h264/hevc/av1 throughput \
             (expected on Orin Nano and on datacenter dGPUs like A100/H100/B300)"
        );
    }

    if has_nvjpegenc() {
        group.bench_function("jpeg", |b| {
            b.iter_batched(
                || make_encoder(jpeg_config()),
                encode_n_frames,
                BatchSize::PerIteration,
            );
        });
    } else {
        eprintln!("nvjpegenc not available — skipping jpeg throughput");
    }

    // PNG/RAW throughput — always available.
    group.bench_function("png", |b| {
        b.iter_batched(
            || make_encoder(png_config()),
            encode_n_frames,
            BatchSize::PerIteration,
        );
    });
    group.bench_function("raw_rgba", |b| {
        b.iter_batched(
            || make_encoder(raw_rgba_config()),
            encode_n_frames,
            BatchSize::PerIteration,
        );
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_creation_plus_one_frame,
    bench_200_frame_throughput
);
criterion_main!(benches);
