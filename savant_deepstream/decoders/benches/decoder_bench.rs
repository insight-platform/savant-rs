//! Criterion benchmarks for the NvDecoder.
//!
//! Measures HEVC and JPEG decode-to-RGBA throughput at HD, FullHD,
//! and 4K resolutions.  Each iteration decodes a batch of 100 frames,
//! including `send_eos()` and draining until the `Eos` event.
//!
//! Test bitstreams are generated at runtime with `NvEncoder` so no
//! pre-generated asset files are required.  `NvEncoder` always disables
//! B-frames internally, so the HEVC streams are I/P-only.
//!
//! Run with:
//! ```sh
//! cargo bench -p savant-deepstream-decoders --bench decoder_bench
//! ```

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use deepstream_buffers::NvBufSurfaceMemType;
use deepstream_decoders::prelude::*;
use deepstream_decoders::NvDecoderExt;
use deepstream_encoders::{
    EncoderConfig, HevcEncoderConfig, JpegEncoderConfig, NvEncoder, NvEncoderConfig,
    NvEncoderOutput,
};
use std::sync::Once;
use std::time::Duration;

static INIT: Once = Once::new();

const BENCH_FRAMES: usize = 1000;
const FRAME_DUR_NS: u64 = 33_333_333; // ~30 fps

// ---------------------------------------------------------------------------
// Init / capability probes
// ---------------------------------------------------------------------------

fn ensure_init() {
    INIT.call_once(|| {
        gstreamer::init().expect("GStreamer init failed");
        cuda_init(0).expect("CUDA init failed");
    });
}

fn has_nvenc() -> bool {
    nvidia_gpu_utils::has_nvenc(0).unwrap_or(false)
}

fn bench_rgba_pool(w: u32, h: u32) -> BufferGenerator {
    BufferGenerator::builder(VideoFormat::RGBA, w, h)
        .gpu_id(0)
        .mem_type(NvBufSurfaceMemType::Default)
        .min_buffers(4)
        .max_buffers(4)
        .build()
        .expect("RGBA pool creation failed")
}

// ---------------------------------------------------------------------------
// Bitstream generation helpers
// ---------------------------------------------------------------------------

// ── New encoder-API adapters (see test_decoder.rs for rationale) ────

fn make_encoder(enc_cfg: EncoderConfig) -> NvEncoder {
    let cfg = NvEncoderConfig::new(0, enc_cfg)
        .operation_timeout(Duration::from_secs(10))
        .name("decoder-bench-encoder");
    NvEncoder::new(cfg).expect("NvEncoder::new failed")
}

fn encoder_acquire_buffer(encoder: &NvEncoder, id: u128) -> gstreamer::Buffer {
    encoder
        .generator()
        .lock()
        .acquire(Some(id))
        .expect("acquire failed")
        .into_buffer()
        .expect("sole owner")
}

/// Encode `n` HEVC frames at the given resolution, returning one
/// Annex-B access unit per frame.
fn encode_hevc_aus(w: u32, h: u32, n: usize) -> Vec<Vec<u8>> {
    let encoder = make_encoder(EncoderConfig::Hevc(HevcEncoderConfig::new(w, h)));
    let mut aus = Vec::with_capacity(n);

    for i in 0..n {
        let buf = encoder_acquire_buffer(&encoder, i as u128);
        let pts = i as u64 * FRAME_DUR_NS;
        encoder
            .submit_frame(buf, i as u128, pts, Some(FRAME_DUR_NS))
            .expect("submit_frame failed");

        while let Ok(Some(out)) = encoder.try_recv() {
            if let NvEncoderOutput::Frame(f) = out {
                aus.push(f.data);
            }
        }
    }

    encoder
        .graceful_shutdown(Some(Duration::from_secs(5)), |out| {
            if let NvEncoderOutput::Frame(f) = out {
                aus.push(f.data);
            }
        })
        .expect("encoder graceful_shutdown failed");
    aus
}

/// Encode a single JPEG frame at the given resolution.
fn encode_jpeg_blob(w: u32, h: u32) -> Vec<u8> {
    let encoder = make_encoder(EncoderConfig::Jpeg(
        JpegEncoderConfig::new(w, h).format(VideoFormat::I420),
    ));

    let buf = encoder_acquire_buffer(&encoder, 0);
    encoder
        .submit_frame(buf, 0, 0, Some(FRAME_DUR_NS))
        .expect("submit_frame failed");

    let mut frames = Vec::new();
    encoder
        .graceful_shutdown(Some(Duration::from_secs(5)), |out| {
            if let NvEncoderOutput::Frame(f) = out {
                frames.push(f);
            }
        })
        .expect("encoder graceful_shutdown failed");
    assert!(!frames.is_empty(), "JPEG encoder produced 0 frames");
    frames.into_iter().next().unwrap().data
}

// ---------------------------------------------------------------------------
// Decode + drain helper
// ---------------------------------------------------------------------------

/// Submit all access units to `decoder`, send EOS, drain until EOS.
/// Panics if the decoded frame count does not match `expected`.
fn decode_and_drain(decoder: &NvDecoder, data: &[Vec<u8>], expected: usize) {
    for (i, au) in data.iter().enumerate() {
        let pts = i as u64 * FRAME_DUR_NS;
        decoder
            .submit_packet(au, i as u128, pts, Some(pts), Some(FRAME_DUR_NS))
            .unwrap_or_else(|e| panic!("submit_packet {i} failed: {e}"));
    }
    decoder
        .send_eos()
        .unwrap_or_else(|e| panic!("send_eos failed: {e}"));

    let mut count = 0usize;
    loop {
        match decoder.recv_timeout(Duration::from_secs(60)) {
            Ok(Some(NvDecoderOutput::Frame(_))) => count += 1,
            Ok(Some(NvDecoderOutput::Eos)) => break,
            Ok(Some(NvDecoderOutput::Error(e))) => panic!("decoder error: {e}"),
            Ok(Some(NvDecoderOutput::Event(_) | NvDecoderOutput::SourceEos { .. })) => {}
            Ok(None) => panic!("timeout after {count}/{expected} frames"),
            Err(e) => panic!("recv error: {e}"),
        }
    }
    assert_eq!(count, expected, "decoded {count} != expected {expected}");
}

// ---------------------------------------------------------------------------
// HEVC benchmark
// ---------------------------------------------------------------------------

const RESOLUTIONS: [(&str, u32, u32); 3] = [
    ("720p", 1280, 720),
    ("1080p", 1920, 1080),
    ("2160p", 3840, 2160),
];

fn bench_hevc_decode(c: &mut Criterion) {
    ensure_init();
    if !has_nvenc() {
        eprintln!("skip: NVENC not available — skipping HEVC decode benchmarks");
        return;
    }

    let mut group = c.benchmark_group("hevc_decode_1000_frames");
    group.sample_size(10);
    group.throughput(criterion::Throughput::Elements(BENCH_FRAMES as u64));

    for &(label, w, h) in &RESOLUTIONS {
        eprintln!("Generating {label} HEVC bitstream ({BENCH_FRAMES} frames)...");
        let all_aus = encode_hevc_aus(w, h, BENCH_FRAMES);
        assert!(
            all_aus.len() >= BENCH_FRAMES,
            "encoder produced {} AUs, need {BENCH_FRAMES}",
            all_aus.len()
        );
        let aus: Vec<Vec<u8>> = all_aus.into_iter().take(BENCH_FRAMES).collect();
        eprintln!("  {label}: {} AUs ready", aus.len());

        group.bench_function(label, |b| {
            b.iter_batched(
                || {
                    let data = aus.clone();
                    let config =
                        DecoderConfig::Hevc(HevcDecoderConfig::new(HevcStreamFormat::ByteStream));
                    let decoder = NvDecoder::new(
                        NvDecoderConfig::new(0, config),
                        bench_rgba_pool(w, h),
                        TransformConfig::default(),
                    )
                    .expect("NvDecoder creation failed");
                    (decoder, data)
                },
                |(decoder, data)| {
                    decode_and_drain(&decoder, &data, BENCH_FRAMES);
                },
                BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// JPEG benchmark
// ---------------------------------------------------------------------------

fn bench_jpeg_decode(c: &mut Criterion) {
    ensure_init();

    let probe_cfg = NvEncoderConfig::new(
        0,
        EncoderConfig::Jpeg(JpegEncoderConfig::new(64, 48).format(VideoFormat::I420)),
    );
    if NvEncoder::new(probe_cfg).is_err() {
        eprintln!("skip: JPEG encoding not available — skipping JPEG decode benchmarks");
        return;
    }

    let mut group = c.benchmark_group("jpeg_decode_1000_frames");
    group.sample_size(10);
    group.throughput(criterion::Throughput::Elements(BENCH_FRAMES as u64));

    for &(label, w, h) in &RESOLUTIONS {
        eprintln!("Generating {label} JPEG blob...");
        let blob = encode_jpeg_blob(w, h);
        let data: Vec<Vec<u8>> = (0..BENCH_FRAMES).map(|_| blob.clone()).collect();
        eprintln!(
            "  {label}: {} bytes, replicated to {} frames",
            blob.len(),
            data.len()
        );

        group.bench_function(label, |b| {
            b.iter_batched(
                || {
                    let data = data.clone();
                    let config = DecoderConfig::Jpeg(JpegDecoderConfig::gpu());
                    let decoder = NvDecoder::new(
                        NvDecoderConfig::new(0, config),
                        bench_rgba_pool(w, h),
                        TransformConfig::default(),
                    )
                    .expect("NvDecoder creation failed");
                    (decoder, data)
                },
                |(decoder, data)| {
                    decode_and_drain(&decoder, &data, BENCH_FRAMES);
                },
                BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_hevc_decode, bench_jpeg_decode);
criterion_main!(benches);
