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
//! cargo bench -p deepstream_decoders --bench decoder_bench
//! ```

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use deepstream_buffers::NvBufSurfaceMemType;
use deepstream_decoders::prelude::*;
use deepstream_encoders::prelude::*;
use std::sync::mpsc;
use std::sync::Once;
use std::time::Duration;

static INIT: Once = Once::new();

const BENCH_FRAMES: usize = 100;
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

fn has_nvdec() -> bool {
    gstreamer::ElementFactory::find("nvv4l2decoder").is_some()
}

fn has_nvjpegdec() -> bool {
    gstreamer::ElementFactory::find("nvjpegdec").is_some()
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

/// Encode `n` HEVC frames at the given resolution, returning one
/// Annex-B access unit per frame.
fn encode_hevc_aus(w: u32, h: u32, n: usize) -> Vec<Vec<u8>> {
    let config = EncoderConfig::new(Codec::Hevc, w, h);
    let mut encoder = NvEncoder::new(&config).expect("HEVC NvEncoder creation failed");
    let mut aus = Vec::with_capacity(n);

    for i in 0..n {
        let shared = encoder
            .generator()
            .acquire(Some(i as u128))
            .expect("acquire failed");
        let buf = shared.into_buffer().expect("into_buffer failed");
        let pts = i as u64 * FRAME_DUR_NS;
        encoder
            .submit_frame(buf, i as u128, pts, Some(FRAME_DUR_NS))
            .expect("submit_frame failed");

        while let Ok(Some(f)) = encoder.pull_encoded() {
            aus.push(f.data);
        }
    }

    let remaining = encoder.finish(Some(5000)).expect("encoder finish failed");
    aus.extend(remaining.into_iter().map(|f| f.data));
    aus
}

/// Encode a single JPEG frame at the given resolution.
fn encode_jpeg_blob(w: u32, h: u32) -> Vec<u8> {
    let config = EncoderConfig::new(Codec::Jpeg, w, h).format(VideoFormat::I420);
    let mut encoder = NvEncoder::new(&config).expect("JPEG NvEncoder creation failed");

    let shared = encoder
        .generator()
        .acquire(Some(0))
        .expect("acquire failed");
    let buf = shared.into_buffer().expect("into_buffer failed");
    encoder
        .submit_frame(buf, 0, 0, Some(FRAME_DUR_NS))
        .expect("submit_frame failed");

    let frames = encoder.finish(Some(5000)).expect("encoder finish failed");
    assert!(!frames.is_empty(), "JPEG encoder produced 0 frames");
    frames.into_iter().next().unwrap().data
}

// ---------------------------------------------------------------------------
// Decode + drain helper
// ---------------------------------------------------------------------------

/// Submit all access units to `decoder`, send EOS, drain until EOS.
/// Panics if the decoded frame count does not match `expected`.
fn decode_and_drain(
    decoder: &mut NvDecoder,
    rx: &mpsc::Receiver<DecoderEvent>,
    data: &[Vec<u8>],
    expected: usize,
) {
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
        match rx.recv_timeout(Duration::from_secs(60)) {
            Ok(DecoderEvent::Frame(_)) => count += 1,
            Ok(DecoderEvent::Eos) => break,
            Ok(DecoderEvent::Error(e)) => panic!("decoder error: {e}"),
            Ok(DecoderEvent::PipelineRestarted { reason, .. }) => {
                panic!("unexpected restart: {reason}")
            }
            Err(_) => panic!("timeout after {count}/{expected} frames"),
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
    if !has_nvenc() || !has_nvdec() {
        eprintln!("skip: NVENC or NVDEC not available — skipping HEVC decode benchmarks");
        return;
    }

    let mut group = c.benchmark_group("hevc_decode_100_frames");
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
                    let (tx, rx) = mpsc::channel();
                    let config =
                        DecoderConfig::Hevc(HevcDecoderConfig::new(HevcStreamFormat::ByteStream));
                    let decoder = NvDecoder::new(
                        0,
                        &config,
                        bench_rgba_pool(w, h),
                        TransformConfig::default(),
                        move |ev| {
                            let _ = tx.send(ev);
                        },
                    )
                    .expect("NvDecoder creation failed");
                    (decoder, rx, data)
                },
                |(mut decoder, rx, data)| {
                    decode_and_drain(&mut decoder, &rx, &data, BENCH_FRAMES);
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
    if !has_nvjpegdec() {
        eprintln!("skip: nvjpegdec not available — skipping JPEG decode benchmarks");
        return;
    }

    if NvEncoder::new(&EncoderConfig::new(Codec::Jpeg, 64, 48).format(VideoFormat::I420)).is_err() {
        eprintln!("skip: JPEG encoding not available — skipping JPEG decode benchmarks");
        return;
    }

    let mut group = c.benchmark_group("jpeg_decode_100_frames");
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
                    let (tx, rx) = mpsc::channel();
                    let config = DecoderConfig::Jpeg(JpegDecoderConfig::gpu());
                    let decoder = NvDecoder::new(
                        0,
                        &config,
                        bench_rgba_pool(w, h),
                        TransformConfig::default(),
                        move |ev| {
                            let _ = tx.send(ev);
                        },
                    )
                    .expect("NvDecoder creation failed");
                    (decoder, rx, data)
                },
                |(mut decoder, rx, data)| {
                    decode_and_drain(&mut decoder, &rx, &data, BENCH_FRAMES);
                },
                BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_hevc_decode, bench_jpeg_decode);
criterion_main!(benches);
