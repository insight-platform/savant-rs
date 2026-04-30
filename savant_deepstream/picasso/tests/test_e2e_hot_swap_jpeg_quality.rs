//! E2E test: hot-swap JPEG quality via `set_source_spec`.
//!
//! Resolution and codec stay identical; only the JPEG `quality`
//! property changes.  The pre-fix shallow encoder-rebuild predicate
//! (`codec_differs`) compared only `width` / `height` / `codec`, so a
//! quality change was silently ignored — the running encoder kept
//! emitting frames at the *old* quality.  This test exercises the
//! deepened (full structural) comparison + the new
//! [`StreamResetReason::EncoderSpecChanged`] observability hook:
//!
//! 1. Run a window of frames at high quality (Q90).
//! 2. Swap to low quality (Q5) via `set_source_spec`.
//! 3. Run a second window of frames.
//! 4. Assert (a) `on_stream_reset` fired with
//!    [`StreamResetReason::EncoderSpecChanged`] and
//!    (b) the post-swap mean encoded byte size is meaningfully
//!    smaller than the pre-swap mean (otherwise the encoder was
//!    not actually rebuilt and Q5 frames would still be Q90-sized).

mod common;

use common::*;
use deepstream_buffers::{BufferGenerator, TransformConfig};
use deepstream_encoders::prelude::*;
use parking_lot::Mutex;
use picasso::callbacks::{OnStreamReset, StreamResetReason};
use picasso::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

const W: u32 = 640;
const H: u32 = 480;
const DUR: u64 = 33_333_333;
const FRAMES_PER_WINDOW: u64 = 10;
const Q_HIGH: u32 = 90;
const Q_LOW: u32 = 5;

/// Captures encoded byte sizes split into two windows so the test
/// can compare pre-swap (high quality) vs. post-swap (low quality)
/// means.
struct WindowedSink {
    /// Index of the swap point.  Frames received before this index
    /// land in `pre`; frames at or after this index land in `post`.
    /// Set by the test once the swap has been issued.
    swap_at: Arc<AtomicUsize>,
    /// Running counter of all encoded frames received so far —
    /// used to bucket each callback firing into pre/post.
    seen: Arc<AtomicUsize>,
    pre: Arc<Mutex<Vec<usize>>>,
    post: Arc<Mutex<Vec<usize>>>,
    eos_count: Arc<AtomicUsize>,
}

impl OnEncodedFrame for WindowedSink {
    fn call(&self, output: OutputMessage) {
        match output {
            OutputMessage::EndOfStream(_) => {
                self.eos_count.fetch_add(1, Ordering::SeqCst);
            }
            OutputMessage::VideoFrame(frame) => {
                let idx = self.seen.fetch_add(1, Ordering::SeqCst);
                let bytes = match frame.get_content().as_ref() {
                    savant_core::primitives::frame::VideoFrameContent::Internal(d) => d.len(),
                    _ => return,
                };
                let swap_at = self.swap_at.load(Ordering::SeqCst);
                if idx < swap_at {
                    self.pre.lock().push(bytes);
                } else {
                    self.post.lock().push(bytes);
                }
            }
        }
    }
}

struct ResetCollector(Arc<Mutex<Vec<(String, StreamResetReason)>>>);
impl OnStreamReset for ResetCollector {
    fn call(&self, source_id: &str, reason: StreamResetReason) {
        self.0.lock().push((source_id.to_string(), reason));
    }
}

fn jpeg_with_quality(quality: u32) -> NvEncoderConfig {
    NvEncoderConfig::new(
        0,
        EncoderConfig::Jpeg(
            JpegEncoderConfig::new(W, H)
                .format(VideoFormat::RGBA)
                .fps(30, 1)
                .props(JpegProps {
                    quality: Some(quality),
                }),
        ),
    )
    .name("picasso-jpeg-quality-swap")
}

#[test]
#[serial_test::serial]
fn e2e_hot_swap_jpeg_quality_rebuilds_encoder() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).unwrap();

    if !has_nvjpegenc() {
        eprintln!("skipping: nvjpegenc not available");
        return;
    }

    let pre: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
    let post: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
    let swap_at = Arc::new(AtomicUsize::new(usize::MAX));
    let seen = Arc::new(AtomicUsize::new(0));
    let eos_count = Arc::new(AtomicUsize::new(0));

    let resets: Arc<Mutex<Vec<(String, StreamResetReason)>>> = Arc::new(Mutex::new(Vec::new()));

    let callbacks = Callbacks {
        on_encoded_frame: Some(Arc::new(WindowedSink {
            swap_at: swap_at.clone(),
            seen: seen.clone(),
            pre: pre.clone(),
            post: post.clone(),
            eos_count: eos_count.clone(),
        })),
        on_stream_reset: Some(Arc::new(ResetCollector(resets.clone()))),
        ..Default::default()
    };

    let engine = PicassoEngine::new(
        GeneralSpec {
            idle_timeout_secs: 300,
            ..Default::default()
        },
        callbacks,
    );

    // ── Window 1 — high quality ──────────────────────────────────
    let spec1 = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(jpeg_with_quality(Q_HIGH)),
        },
        ..Default::default()
    };
    engine.set_source_spec("swap", spec1).unwrap();

    let gen = BufferGenerator::new(
        VideoFormat::RGBA,
        W,
        H,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .unwrap();

    for i in 0..FRAMES_PER_WINDOW {
        let mut frame = make_frame_sized("swap", W as i64, H as i64);
        frame.set_pts((i * DUR) as i64).unwrap();
        frame.set_duration(Some(DUR as i64)).unwrap();
        let buf = make_gpu_surface_view(&gen, i, DUR);
        engine.send_frame("swap", frame, buf, None).unwrap();
    }
    std::thread::sleep(Duration::from_secs(2));

    // Latch the swap point: every encoded frame from now on lands in `post`.
    swap_at.store(seen.load(Ordering::SeqCst), Ordering::SeqCst);

    // ── Window 2 — low quality ───────────────────────────────────
    let spec2 = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(jpeg_with_quality(Q_LOW)),
        },
        ..Default::default()
    };
    engine.set_source_spec("swap", spec2).unwrap();

    for i in FRAMES_PER_WINDOW..FRAMES_PER_WINDOW * 2 {
        let mut frame = make_frame_sized("swap", W as i64, H as i64);
        frame.set_pts((i * DUR) as i64).unwrap();
        frame.set_duration(Some(DUR as i64)).unwrap();
        let buf = make_gpu_surface_view(&gen, i, DUR);
        engine.send_frame("swap", frame, buf, None).unwrap();
    }

    engine.send_eos("swap").unwrap();
    std::thread::sleep(Duration::from_secs(2));
    engine.shutdown();

    // ── Assertions ───────────────────────────────────────────────
    let pre_sizes = pre.lock().clone();
    let post_sizes = post.lock().clone();
    assert!(
        !pre_sizes.is_empty(),
        "expected at least one pre-swap encoded frame"
    );
    assert!(
        !post_sizes.is_empty(),
        "expected at least one post-swap encoded frame"
    );

    let mean = |v: &[usize]| v.iter().copied().sum::<usize>() as f64 / v.len() as f64;
    let pre_mean = mean(&pre_sizes);
    let post_mean = mean(&post_sizes);
    log::info!(
        "pre_mean={pre_mean:.0} bytes ({} frames), post_mean={post_mean:.0} bytes ({} frames)",
        pre_sizes.len(),
        post_sizes.len(),
    );

    // Q5 JPEGs are typically <= 1/3 the size of Q90 on photographic
    // content; allow a generous margin for synthetic uniform buffers
    // by asserting at least 30% size reduction.  A still-running
    // Q90 encoder would produce post-swap frames within ~5% of
    // pre-swap, comfortably failing this bound.
    assert!(
        post_mean < pre_mean * 0.7,
        "post-swap mean ({post_mean:.0}) must be substantially smaller than pre-swap mean ({pre_mean:.0}) — encoder appears not to have been rebuilt with the new JPEG quality"
    );

    // The encoder rebuild must have been surfaced through the
    // OnStreamReset callback as `EncoderSpecChanged`.
    let resets_observed = resets.lock();
    assert!(
        resets_observed
            .iter()
            .any(|(sid, reason)| sid == "swap"
                && matches!(reason, StreamResetReason::EncoderSpecChanged)),
        "expected one EncoderSpecChanged reset for source 'swap', got {:?}",
        *resets_observed
    );

    assert_eq!(eos_count.load(Ordering::SeqCst), 1);
}
