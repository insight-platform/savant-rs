//! Benchmark for `CodecSpec::Drop` — measures the overhead of submitting
//! frames through PicassoEngine when all frames are discarded.
//!
//! No encoding, no rendering, no bypass.  This establishes the baseline
//! cost of engine dispatch, per-source worker wake, conditional checks,
//! and frame/object allocation.
//!
//! Run with:
//!
//! ```sh
//! cargo bench -p picasso --bench bench_drop
//! BENCH_NUM_SOURCES=8 cargo bench -p picasso --bench bench_drop
//! ```

use deepstream_encoders::prelude::*;
use deepstream_nvbufsurface::SurfaceView;
use picasso::prelude::*;
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod, VideoFrameTransformation,
};
use savant_core::primitives::object::{IdCollisionResolutionPolicy, VideoObjectBuilder};
use savant_core::primitives::RBBox;
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
    format!("drop-{idx}")
}

fn make_frame(source_id: &str, idx: u64) -> VideoFrameProxy {
    let f = VideoFrameProxy::new(
        source_id,
        "30/1",
        WIDTH as i64,
        HEIGHT as i64,
        VideoFrameContent::None,
        VideoFrameTranscodingMethod::Copy,
        &None,
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
            .build()
            .unwrap();
        let _ = frame.add_object(obj, IdCollisionResolutionPolicy::GenerateNewId);
    }
}

fn make_buffer(gen: &DsNvSurfaceBufferGenerator, idx: u64) -> gstreamer::Buffer {
    let mut buf = gen.acquire_surface(Some(idx as i64)).unwrap();
    {
        let buf_ref = buf.make_mut();
        buf_ref.set_pts(gstreamer::ClockTime::from_nseconds(idx * FRAME_DURATION_NS));
        buf_ref.set_duration(gstreamer::ClockTime::from_nseconds(FRAME_DURATION_NS));
    }
    buf
}

/// Accepts the EOS sentinel but panics on any real encoded data.
struct EosOnlyEncodedSink;
impl OnEncodedFrame for EosOnlyEncodedSink {
    fn call(&self, output: EncodedOutput) {
        assert!(
            matches!(output, EncodedOutput::EndOfStream(_)),
            "Drop mode must only produce EOS sentinels, got a real encoded frame"
        );
    }
}

/// Panics on any bypass output — Drop mode must never produce bypass frames.
struct NeverBypassSink;
impl OnBypassFrame for NeverBypassSink {
    fn call(&self, _output: EncodedOutput) {
        panic!("on_bypass_frame must never fire for CodecSpec::Drop");
    }
}

fn main() {
    gstreamer::init().unwrap();
    cuda_init(0).unwrap();

    let num_src = num_sources();
    let total_frames = NUM_FRAMES * num_src as u64;

    println!("=== Picasso Drop Benchmark ===");
    println!(
        "Resolution: {}x{}, Frames/source: {}, Sources: {}, Total: {}",
        WIDTH, HEIGHT, NUM_FRAMES, num_src, total_frames,
    );
    println!("Objects/frame: {NUM_BOXES}");

    let callbacks = Callbacks {
        on_encoded_frame: Some(Arc::new(EosOnlyEncodedSink)),
        on_bypass_frame: Some(Arc::new(NeverBypassSink)),
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
                    codec: CodecSpec::Drop,
                    ..Default::default()
                },
            )
            .unwrap();
    }

    let generators: Vec<DsNvSurfaceBufferGenerator> = (0..num_src)
        .map(|_| {
            DsNvSurfaceBufferGenerator::new(
                VideoFormat::RGBA,
                WIDTH,
                HEIGHT,
                FPS,
                1,
                0,
                NvBufSurfaceMemType::Default,
            )
            .unwrap()
        })
        .collect();

    // Warm-up
    for i in 0..5u64 {
        for (s, sid) in source_ids.iter().enumerate() {
            let frame = make_frame(sid, i);
            add_objects(&frame);
            let buf = make_buffer(&generators[s], i);
            let view = SurfaceView::from_buffer(&buf, 0).unwrap();
            engine.send_frame(sid, frame, view, None).unwrap();
        }
    }
    std::thread::sleep(std::time::Duration::from_millis(200));
    println!("Warm-up complete\n");

    let start = Instant::now();
    let mut submitted = 0u64;

    for i in 0..NUM_FRAMES {
        for (s, sid) in source_ids.iter().enumerate() {
            let frame = make_frame(sid, i);
            add_objects(&frame);
            let buf = make_buffer(&generators[s], i);
            let view = SurfaceView::from_buffer(&buf, 0).unwrap();
            engine.send_frame(sid, frame, view, None).unwrap();
            submitted += 1;
        }
    }

    // Give the workers time to process queued frames.
    std::thread::sleep(std::time::Duration::from_millis(500));

    for sid in &source_ids {
        engine.send_eos(sid).unwrap();
    }
    std::thread::sleep(std::time::Duration::from_millis(200));

    let elapsed = start.elapsed();
    let fps = submitted as f64 / elapsed.as_secs_f64();

    println!("=== Results ===");
    println!("Sources:          {num_src}");
    println!("Total submitted:  {submitted}");
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
