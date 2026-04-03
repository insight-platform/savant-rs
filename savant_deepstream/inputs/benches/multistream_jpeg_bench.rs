//! Criterion benchmark: single JPEG + EOS roundtrip through [`MultiStreamDecoder`].
//!
//! Each iteration submits **one** JPEG frame, waits for the decoded output,
//! then sends EOS and drains until `StreamStopped`.  The EOS tears down the
//! internal `NvDecoder` and buffer pool, so the next iteration must recreate
//! them from scratch.  This measures the full decoder/pool lifecycle cost per
//! frame (create → decode → drain → destroy).
//!
//! The `MultiStreamDecoder` itself is **not** recreated between iterations —
//! only the per-stream decoder/pool inside it is.
//!
//! ```sh
//! cargo bench -p deepstream_inputs --bench multistream_jpeg_bench
//! ```

use criterion::{criterion_group, criterion_main, Criterion};
use deepstream_encoders::prelude::*;
use deepstream_inputs::multistream_decoder::{
    DecoderOutput, EvictionVerdict, MultiStreamDecoder, MultiStreamDecoderConfig, StopReason,
};
use savant_core::primitives::eos::EndOfStream;
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
};
use savant_core::primitives::video_codec::VideoCodec;
use std::sync::mpsc;
use std::sync::Once;
use std::time::Duration;

static INIT: Once = Once::new();

const FRAME_DUR_NS: u64 = 33_333_333;

fn ensure_init() {
    INIT.call_once(|| {
        gstreamer::init().expect("GStreamer init failed");
        cuda_init(0).expect("CUDA init failed");
    });
}

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

fn make_jpeg_frame(source_id: &str, w: i64, h: i64, pts_ns: i64) -> VideoFrameProxy {
    VideoFrameProxy::new(
        source_id,
        (30, 1),
        w,
        h,
        VideoFrameContent::None,
        VideoFrameTranscodingMethod::Copy,
        Some(VideoCodec::Jpeg),
        Some(true),
        (1, 1_000_000_000),
        pts_ns,
        None,
        Some(FRAME_DUR_NS as i64),
    )
    .expect("VideoFrameProxy creation failed")
}

const RECV_TIMEOUT: Duration = Duration::from_secs(10);

/// Submit 1 JPEG frame → wait for Decoded → send EOS → drain until StreamStopped → join teardowns.
fn one_jpeg_eos_roundtrip(
    decoder: &MultiStreamDecoder,
    rx: &mpsc::Receiver<DecoderOutput>,
    blob: &[u8],
    w: i64,
    h: i64,
    pts_ns: &mut i64,
) {
    let source_id = "bench_jpeg";
    let timeout = Duration::from_secs(5);

    let frame = make_jpeg_frame(source_id, w, h, *pts_ns);
    *pts_ns += FRAME_DUR_NS as i64;

    decoder
        .submit(frame, Some(blob), timeout)
        .unwrap_or_else(|e| panic!("submit: {e}"));

    // Wait for the decoded frame before sending EOS.
    loop {
        match rx.recv_timeout(RECV_TIMEOUT) {
            Ok(DecoderOutput::Decoded { .. }) => break,
            Ok(DecoderOutput::StreamStarted { .. }) => {}
            Ok(DecoderOutput::Undecoded { .. }) => {}
            Ok(other) => panic!("unexpected event before Decoded: {other:?}"),
            Err(_) => panic!("timeout waiting for Decoded"),
        }
    }

    let eos = EndOfStream::new(source_id);
    decoder
        .submit_eos(&eos, timeout)
        .unwrap_or_else(|e| panic!("submit_eos: {e}"));

    // Drain remaining events until StreamStopped.
    loop {
        match rx.recv_timeout(RECV_TIMEOUT) {
            Ok(DecoderOutput::StreamStopped {
                reason: StopReason::Eos,
                ..
            }) => break,
            Ok(DecoderOutput::StreamStopped { reason, .. }) => {
                panic!("unexpected stop: {reason:?}")
            }
            Ok(DecoderOutput::Eos { .. }) | Ok(DecoderOutput::Decoded { .. }) => {}
            Ok(other) => panic!("unexpected event during drain: {other:?}"),
            Err(_) => panic!("timeout waiting for StreamStopped"),
        }
    }

    decoder.wait_for_pending_teardowns();
}

const RESOLUTIONS: [(&str, u32, u32); 3] = [
    ("720p", 1280, 720),
    ("1080p", 1920, 1080),
    ("2160p", 3840, 2160),
];

fn bench_jpeg_eos_roundtrip(c: &mut Criterion) {
    ensure_init();

    if NvEncoder::new(&EncoderConfig::new(Codec::Jpeg, 64, 48).format(VideoFormat::I420)).is_err() {
        eprintln!("SKIP: JPEG encoding not available");
        return;
    }

    let mut group = c.benchmark_group("jpeg_1frame_eos_roundtrip");
    group.sample_size(50);

    for &(label, w, h) in &RESOLUTIONS {
        eprintln!("Generating {label} JPEG blob ({w}x{h})...");
        let blob = encode_jpeg_blob(w, h);
        eprintln!("  {label}: {} bytes", blob.len());

        group.bench_function(label, |b| {
            let (tx, rx) = mpsc::channel();
            let cfg = MultiStreamDecoderConfig::new(0, 8).idle_timeout(Duration::from_secs(600));
            let decoder = MultiStreamDecoder::new(
                cfg,
                move |o| {
                    let _ = tx.send(o);
                },
                None::<fn(&str) -> EvictionVerdict>,
            );
            let mut pts_ns: i64 = 0;

            b.iter(|| {
                one_jpeg_eos_roundtrip(&decoder, &rx, &blob, w as i64, h as i64, &mut pts_ns);
            });

            let mut dec = decoder;
            dec.shutdown();
        });
    }
    group.finish();
}

criterion_group!(benches, bench_jpeg_eos_roundtrip);
criterion_main!(benches);
