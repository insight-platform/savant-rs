//! Benchmark for `CodecSpec::Bypass` — measures throughput of the
//! pass-through path where bboxes are transformed back to initial
//! coordinates without any GPU encoding or rendering.
//!
//! The `on_bypass_frame` callback asserts **VideoFrame sanity** on every
//! output: source_id, object count, dimensions, framerate, and that the
//! `InitialSize` transformation is preserved.
//!
//! Run with:
//!
//! ```sh
//! cargo bench -p picasso --bench bench_bypass
//! BENCH_NUM_SOURCES=8 cargo bench -p picasso --bench bench_bypass
//! ```

use deepstream_buffers::SurfaceView;
use deepstream_encoders::prelude::*;
use picasso::prelude::*;
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod, VideoFrameTransformation,
};
use savant_core::primitives::object::{
    IdCollisionResolutionPolicy, ObjectOperations, VideoObjectBuilder,
};
use savant_core::primitives::RBBox;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const NUM_FRAMES: u64 = 50_000;
const NUM_BOXES: usize = 20;
const FPS: i32 = 30;
const FRAME_DURATION_NS: u64 = 1_000_000_000 / FPS as u64;
const DEFAULT_NUM_SOURCES: usize = 4;

fn num_sources() -> usize {
    std::env::var("BENCH_NUM_SOURCES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_NUM_SOURCES)
}

fn source_id(idx: usize) -> String {
    format!("bypass-{idx}")
}

fn make_frame(source_id: &str, idx: u64) -> VideoFrameProxy {
    let f = VideoFrameProxy::new(
        source_id,
        (30, 1),
        WIDTH as i64,
        HEIGHT as i64,
        VideoFrameContent::None,
        VideoFrameTranscodingMethod::Copy,
        None,
        None,
        (1, 1_000_000_000),
        0,
        None,
        None,
    )
    .unwrap();
    let mut fm = f.clone();
    fm.add_transformation(VideoFrameTransformation::InitialSize(
        WIDTH as u64,
        HEIGHT as u64,
    ));
    fm.set_pts((idx * FRAME_DURATION_NS) as i64).unwrap();
    f
}

fn add_objects(frame: &VideoFrameProxy) {
    for i in 0..NUM_BOXES {
        let cx = 100.0 + (i as f32 * 80.0) % (WIDTH as f32 - 200.0);
        let cy = 100.0 + (i as f32 * 50.0) % (HEIGHT as f32 - 200.0);
        let obj = VideoObjectBuilder::default()
            .id(0)
            .namespace("det".to_string())
            .label("car".to_string())
            .detection_box(RBBox::new(cx, cy, 80.0, 60.0, None))
            .confidence(Some(0.92))
            .build()
            .unwrap();
        let _ = frame.add_object(obj, IdCollisionResolutionPolicy::GenerateNewId);
    }
}

fn make_buffer(gen: &BufferGenerator, idx: u64) -> SurfaceView {
    let shared = gen.acquire(Some(idx as u128)).unwrap();
    shared.set_pts_ns(idx * FRAME_DURATION_NS);
    shared.set_duration_ns(FRAME_DURATION_NS);
    SurfaceView::from_buffer(&shared, 0).unwrap()
}

/// Bypass sink that counts frames and performs sanity assertions.
struct BypassSink(Arc<AtomicUsize>);

impl OnBypassFrame for BypassSink {
    fn call(&self, output: OutputMessage) {
        match output {
            OutputMessage::VideoFrame(frame) => {
                self.0.fetch_add(1, Ordering::Relaxed);

                // Source ID must start with our prefix.
                assert!(
                    frame.get_source_id().starts_with("bypass-"),
                    "unexpected source_id: {}",
                    frame.get_source_id()
                );

                // Dimensions must match what we submitted.
                assert_eq!(
                    frame.get_width(),
                    WIDTH as i64,
                    "width mismatch on bypass output"
                );
                assert_eq!(
                    frame.get_height(),
                    HEIGHT as i64,
                    "height mismatch on bypass output"
                );

                // Framerate preserved.
                assert_eq!(frame.get_fps(), (30, 1), "fps mismatch on bypass output");

                // All objects must survive the round-trip.
                let objects = frame.get_all_objects();
                assert_eq!(
                    objects.len(),
                    NUM_BOXES,
                    "object count mismatch: expected {NUM_BOXES}, got {}",
                    objects.len()
                );

                // Every object must have a valid detection box.
                for obj in &objects {
                    let det = obj.get_detection_box();
                    assert!(det.get_width() > 0.0, "detection box width must be > 0");
                    assert!(det.get_height() > 0.0, "detection box height must be > 0");
                    assert_eq!(obj.get_namespace(), "det");
                    assert_eq!(obj.get_label(), "car");
                }

                // InitialSize transformation must be the only remaining one.
                let transforms = frame.get_transformations();
                assert_eq!(
                    transforms.len(),
                    1,
                    "bypass should leave exactly 1 transformation (InitialSize), got {}",
                    transforms.len()
                );
                match &transforms[0] {
                    VideoFrameTransformation::InitialSize(w, h) => {
                        assert_eq!(*w, WIDTH as u64);
                        assert_eq!(*h, HEIGHT as u64);
                    }
                    other => panic!("expected InitialSize, got {other:?}"),
                }
            }
            OutputMessage::EndOfStream(_) => {}
        }
    }
}

/// Accepts the EOS sentinel but panics on any real encoded data.
struct EosOnlyEncodedSink;
impl OnEncodedFrame for EosOnlyEncodedSink {
    fn call(&self, output: OutputMessage) {
        assert!(
            matches!(output, OutputMessage::EndOfStream(_)),
            "Bypass mode must only produce EOS sentinels, got a real encoded frame"
        );
    }
}

fn main() {
    gstreamer::init().unwrap();
    cuda_init(0).unwrap();

    let num_src = num_sources();
    let total_frames = NUM_FRAMES * num_src as u64;

    println!("=== Picasso Bypass Benchmark ===");
    println!(
        "Resolution: {}x{}, Frames/source: {}, Sources: {}, Total: {}",
        WIDTH, HEIGHT, NUM_FRAMES, num_src, total_frames,
    );
    println!("Objects/frame: {NUM_BOXES}");

    let bypass_count = Arc::new(AtomicUsize::new(0));

    let callbacks = Callbacks {
        on_bypass_frame: Some(Arc::new(BypassSink(bypass_count.clone()))),
        on_encoded_frame: Some(Arc::new(EosOnlyEncodedSink)),
        ..Default::default()
    };

    let general = GeneralSpec {
        idle_timeout_secs: 300,
        ..Default::default()
    };
    let mut engine = PicassoEngine::new(general, callbacks);

    let source_ids: Vec<String> = (0..num_src).map(source_id).collect();
    for sid in &source_ids {
        engine
            .set_source_spec(
                sid,
                SourceSpec {
                    codec: CodecSpec::Bypass,
                    ..Default::default()
                },
            )
            .unwrap();
    }

    let generators: Vec<BufferGenerator> = (0..num_src)
        .map(|_| {
            BufferGenerator::builder(VideoFormat::RGBA, WIDTH, HEIGHT)
                .fps(FPS, 1)
                .gpu_id(0)
                .mem_type(NvBufSurfaceMemType::Default)
                .min_buffers(32)
                .max_buffers(32)
                .build()
                .unwrap()
        })
        .collect();

    // Warm-up
    for i in 0..5u64 {
        for (s, sid) in source_ids.iter().enumerate() {
            let frame = make_frame(sid, i);
            add_objects(&frame);
            let view = make_buffer(&generators[s], i);
            engine.send_frame(sid, frame, view, None).unwrap();
        }
    }
    std::thread::sleep(std::time::Duration::from_millis(500));
    let warmup_count = bypass_count.load(Ordering::SeqCst);
    assert_eq!(
        warmup_count,
        5 * num_src,
        "warm-up should produce {expected} bypass frames, got {warmup_count}",
        expected = 5 * num_src,
    );
    bypass_count.store(0, Ordering::SeqCst);
    println!("Warm-up complete ({warmup_count} bypass frames verified)\n");

    let start = Instant::now();
    let mut submitted = 0u64;

    for i in 0..NUM_FRAMES {
        for (s, sid) in source_ids.iter().enumerate() {
            let frame = make_frame(sid, i);
            add_objects(&frame);
            let view = make_buffer(&generators[s], i);
            engine.send_frame(sid, frame, view, None).unwrap();
            submitted += 1;
        }
    }

    // Wait for all bypass callbacks to complete.
    for _ in 0..20 {
        std::thread::sleep(std::time::Duration::from_millis(250));
        if bypass_count.load(Ordering::SeqCst) as u64 >= submitted {
            break;
        }
    }

    for sid in &source_ids {
        engine.send_eos(sid).unwrap();
    }
    std::thread::sleep(std::time::Duration::from_millis(200));

    let elapsed = start.elapsed();
    let bypassed = bypass_count.load(Ordering::SeqCst);
    let fps = submitted as f64 / elapsed.as_secs_f64();

    // Final sanity: every submitted frame must have produced a bypass output.
    assert_eq!(
        bypassed as u64, submitted,
        "bypass count mismatch: submitted {submitted}, got {bypassed}"
    );

    println!("=== Results ===");
    println!("Sources:          {num_src}");
    println!("Total submitted:  {submitted}");
    println!("Total bypassed:   {bypassed}");
    println!("Wall time:        {:.2} s", elapsed.as_secs_f64());
    println!("Aggregate FPS:    {fps:.1}");
    println!("Per-source FPS:   {:.1}", fps / num_src as f64);
    println!(
        "Per-frame avg:    {:.3} ms",
        elapsed.as_secs_f64() * 1000.0 / submitted as f64
    );
    println!("RSS:              {} MB", rss_kb() / 1024);

    engine.shutdown();
}

fn rss_kb() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            return parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
        }
    }
    0
}
