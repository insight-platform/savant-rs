//! Memory-leak tests for the picasso pipeline.
//!
//! These tests run many iterations of various operations and assert that
//! neither CPU (VmRSS) nor GPU memory grows unboundedly.  They are
//! intentionally marked `#[ignore]` so they don't slow down the normal
//! `cargo test` cycle — run them explicitly with:
//!
//! ```sh
//! cargo test -p picasso --test test_leak -- --ignored --nocapture
//! ```

use deepstream_encoders::prelude::*;
use deepstream_nvbufsurface::TransformConfig;
use picasso::message::WorkerMessage;
use picasso::prelude::*;
use picasso::worker::SourceWorker;
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
};
use serial_test::serial;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn rss_kb() -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            return parts[1].parse::<u64>().unwrap();
        }
    }
    panic!("VmRSS not found in /proc/self/status");
}

fn gpu_mem_mib() -> u64 {
    nvidia_gpu_utils::gpu_mem_used_mib(0).expect("gpu_mem_used_mib failed")
}

fn make_frame(source_id: &str, w: i64, h: i64) -> VideoFrameProxy {
    VideoFrameProxy::new(
        source_id,
        "30/1",
        w,
        h,
        VideoFrameContent::None,
        VideoFrameTranscodingMethod::Copy,
        &None,
        None,
        (1, 1_000_000_000),
        0,
        None,
        None,
    )
    .unwrap()
}

fn make_nvmm_buffer(
    gen: &DsNvSurfaceBufferGenerator,
    frame_id: i64,
) -> deepstream_nvbufsurface::SurfaceView {
    let mut buf = gen.acquire_surface(Some(frame_id)).unwrap();
    {
        let buf_ref = buf.make_mut();
        buf_ref.set_pts(gstreamer::ClockTime::from_nseconds(
            frame_id as u64 * 33_333_333,
        ));
        buf_ref.set_duration(gstreamer::ClockTime::from_nseconds(33_333_333));
    }
    deepstream_nvbufsurface::SurfaceView::from_buffer(&buf, 0).unwrap()
}

fn encoder_config(w: u32, h: u32) -> EncoderConfig {
    EncoderConfig::new(Codec::H264, w, h)
}

fn bypass_spec() -> SourceSpec {
    SourceSpec {
        codec: CodecSpec::Bypass,
        ..Default::default()
    }
}

struct SinkBypass(Arc<AtomicUsize>);
impl OnBypassFrame for SinkBypass {
    fn call(&self, _output: BypassOutput) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

struct SinkEncoded(Arc<AtomicUsize>);
impl OnEncodedFrame for SinkEncoded {
    fn call(&self, _: EncodedOutput) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

/// Allow `growth_kb` of RSS growth over a baseline.  We take the baseline
/// after a warm-up phase to account for one-time allocations.
const RSS_GROWTH_TOLERANCE_KB: u64 = 8_192; // 8 MiB
const GPU_GROWTH_TOLERANCE_MIB: u64 = 32; // 32 MiB

// ---------------------------------------------------------------------------
// 1. Worker lifecycle churn — CPU only
// ---------------------------------------------------------------------------

#[test]
#[ignore]
#[serial]
fn leak_worker_lifecycle_churn() {
    gstreamer::init().unwrap();

    let count = Arc::new(AtomicUsize::new(0));
    let callbacks = Arc::new(Callbacks {
        on_bypass_frame: Some(Arc::new(SinkBypass(count.clone()))),
        ..Default::default()
    });

    // Warm-up: create/destroy a few workers
    for i in 0..5 {
        let w = SourceWorker::spawn(
            format!("warmup-{i}"),
            bypass_spec(),
            callbacks.clone(),
            Duration::from_secs(60),
        );
        w.send(WorkerMessage::Frame(
            make_frame(&format!("warmup-{i}"), 320, 240),
            deepstream_nvbufsurface::SurfaceView::wrap(gstreamer::Buffer::new()),
            None,
        ))
        .unwrap();
        std::thread::sleep(Duration::from_millis(20));
        drop(w);
    }
    std::thread::sleep(Duration::from_millis(200));

    let baseline_rss = rss_kb();
    println!("worker churn: baseline RSS = {baseline_rss} kB");

    // Churn: create and destroy 200 workers, each processing 10 frames
    for i in 0..200 {
        let source = format!("churn-{i}");
        let w = SourceWorker::spawn(
            source.clone(),
            bypass_spec(),
            callbacks.clone(),
            Duration::from_secs(60),
        );
        for j in 0..10 {
            let _ = w.send(WorkerMessage::Frame(
                make_frame(&source, 320, 240),
                deepstream_nvbufsurface::SurfaceView::wrap(gstreamer::Buffer::new()),
                None,
            ));
            if j == 9 {
                let _ = w.send(WorkerMessage::Eos);
            }
        }
        std::thread::sleep(Duration::from_millis(10));
        drop(w);
    }
    std::thread::sleep(Duration::from_millis(500));

    let final_rss = rss_kb();
    let growth = final_rss.saturating_sub(baseline_rss);
    println!(
        "worker churn: final RSS = {final_rss} kB, growth = {growth} kB, \
         frames processed = {}",
        count.load(Ordering::Relaxed)
    );
    assert!(
        growth < RSS_GROWTH_TOLERANCE_KB,
        "RSS grew by {growth} kB (limit {RSS_GROWTH_TOLERANCE_KB} kB) — possible CPU memory leak"
    );
}

// ---------------------------------------------------------------------------
// 2. Sustained bypass frame processing
// ---------------------------------------------------------------------------

#[test]
#[ignore]
#[serial]
fn leak_sustained_bypass_frames() {
    gstreamer::init().unwrap();

    let count = Arc::new(AtomicUsize::new(0));
    let callbacks = Arc::new(Callbacks {
        on_bypass_frame: Some(Arc::new(SinkBypass(count.clone()))),
        ..Default::default()
    });

    let worker = SourceWorker::spawn(
        "sustained-bypass".to_string(),
        bypass_spec(),
        callbacks,
        Duration::from_secs(120),
    );

    // Warm-up
    for _ in 0..100 {
        let _ = worker.send(WorkerMessage::Frame(
            make_frame("sustained-bypass", 640, 480),
            deepstream_nvbufsurface::SurfaceView::wrap(gstreamer::Buffer::new()),
            None,
        ));
    }
    std::thread::sleep(Duration::from_millis(500));

    let baseline_rss = rss_kb();
    println!("sustained bypass: baseline RSS = {baseline_rss} kB");

    // Main phase: 5000 frames
    for _ in 0..5_000 {
        let _ = worker.send(WorkerMessage::Frame(
            make_frame("sustained-bypass", 640, 480),
            deepstream_nvbufsurface::SurfaceView::wrap(gstreamer::Buffer::new()),
            None,
        ));
    }
    // Drain the channel
    std::thread::sleep(Duration::from_secs(2));

    let final_rss = rss_kb();
    let growth = final_rss.saturating_sub(baseline_rss);
    println!(
        "sustained bypass: final RSS = {final_rss} kB, growth = {growth} kB, \
         frames = {}",
        count.load(Ordering::Relaxed)
    );
    assert!(
        growth < RSS_GROWTH_TOLERANCE_KB,
        "RSS grew by {growth} kB (limit {RSS_GROWTH_TOLERANCE_KB} kB) — possible leak"
    );

    drop(worker);
}

// ---------------------------------------------------------------------------
// 3. Engine multi-source churn — CPU only
// ---------------------------------------------------------------------------

#[test]
#[ignore]
#[serial]
fn leak_engine_multi_source_churn() {
    gstreamer::init().unwrap();

    let count = Arc::new(AtomicUsize::new(0));
    let callbacks = Callbacks {
        on_bypass_frame: Some(Arc::new(SinkBypass(count.clone()))),
        ..Default::default()
    };
    let general = GeneralSpec {
        idle_timeout_secs: 120,
    };
    let mut engine = PicassoEngine::new(general, callbacks);

    // Warm-up
    for i in 0..5 {
        let src = format!("eng-warmup-{i}");
        engine.set_source_spec(&src, bypass_spec()).unwrap();
        for _ in 0..5 {
            let _ = engine.send_frame(
                &src,
                make_frame(&src, 320, 240),
                deepstream_nvbufsurface::SurfaceView::wrap(gstreamer::Buffer::new()),
                None,
            );
        }
        std::thread::sleep(Duration::from_millis(30));
        engine.remove_source_spec(&src);
    }
    std::thread::sleep(Duration::from_millis(300));

    let baseline_rss = rss_kb();
    println!("engine churn: baseline RSS = {baseline_rss} kB");

    // Create and destroy 100 sources, each sending 20 frames
    for i in 0..100 {
        let src = format!("eng-churn-{i}");
        engine.set_source_spec(&src, bypass_spec()).unwrap();
        for _ in 0..20 {
            let _ = engine.send_frame(
                &src,
                make_frame(&src, 320, 240),
                deepstream_nvbufsurface::SurfaceView::wrap(gstreamer::Buffer::new()),
                None,
            );
        }
        std::thread::sleep(Duration::from_millis(15));
        engine.remove_source_spec(&src);
    }
    std::thread::sleep(Duration::from_secs(1));

    let final_rss = rss_kb();
    let growth = final_rss.saturating_sub(baseline_rss);
    println!(
        "engine churn: final RSS = {final_rss} kB, growth = {growth} kB, \
         frames = {}",
        count.load(Ordering::Relaxed)
    );
    assert!(
        growth < RSS_GROWTH_TOLERANCE_KB,
        "RSS grew by {growth} kB (limit {RSS_GROWTH_TOLERANCE_KB} kB) — possible leak"
    );

    engine.shutdown();
}

// ---------------------------------------------------------------------------
// 4. GPU encoder lifecycle churn
// ---------------------------------------------------------------------------

#[test]
#[ignore]
#[serial]
fn leak_gpu_encoder_lifecycle_churn() {
    gstreamer::init().unwrap();
    cuda_init(0).unwrap();

    let cfg = encoder_config(320, 240);

    // Warm-up: create/destroy a few encoders
    for _ in 0..3 {
        let mut enc = deepstream_encoders::NvEncoder::new(&cfg).unwrap();
        let buf = enc.generator().acquire_surface(Some(0)).unwrap();
        enc.submit_frame(buf, 0, 0, Some(33_333_333)).unwrap();
        let _ = enc.finish(Some(3000));
    }
    std::thread::sleep(Duration::from_millis(500));

    let baseline_gpu = gpu_mem_mib();
    let baseline_rss = rss_kb();
    println!("encoder churn: baseline GPU = {baseline_gpu} MiB, RSS = {baseline_rss} kB");

    // Churn: create/destroy 50 encoders, each processing 5 frames
    for i in 0..50 {
        let mut enc = deepstream_encoders::NvEncoder::new(&cfg).unwrap();
        for j in 0..5u128 {
            let buf = enc.generator().acquire_surface(Some(j as i64)).unwrap();
            let pts = (i as u64 * 5 + j as u64) * 33_333_333;
            enc.submit_frame(buf, j, pts, Some(33_333_333)).unwrap();
        }
        let _ = enc.finish(Some(3000));
        drop(enc);
    }
    std::thread::sleep(Duration::from_secs(1));

    let final_gpu = gpu_mem_mib();
    let final_rss = rss_kb();
    let gpu_growth = final_gpu.saturating_sub(baseline_gpu);
    let rss_growth = final_rss.saturating_sub(baseline_rss);
    println!(
        "encoder churn: final GPU = {final_gpu} MiB (+{gpu_growth}), \
         RSS = {final_rss} kB (+{rss_growth})"
    );
    assert!(
        gpu_growth < GPU_GROWTH_TOLERANCE_MIB,
        "GPU memory grew by {gpu_growth} MiB (limit {GPU_GROWTH_TOLERANCE_MIB}) — possible GPU leak"
    );
    assert!(
        rss_growth < RSS_GROWTH_TOLERANCE_KB,
        "RSS grew by {rss_growth} kB (limit {RSS_GROWTH_TOLERANCE_KB}) — possible CPU leak"
    );
}

// ---------------------------------------------------------------------------
// 5. Sustained GPU encode
// ---------------------------------------------------------------------------

#[test]
#[ignore]
#[serial]
fn leak_gpu_sustained_encode() {
    gstreamer::init().unwrap();
    cuda_init(0).unwrap();

    let cfg = encoder_config(320, 240);
    let mut enc = deepstream_encoders::NvEncoder::new(&cfg).unwrap();

    // Warm-up: 20 frames
    for i in 0..20u128 {
        let buf = enc.generator().acquire_surface(Some(i as i64)).unwrap();
        enc.submit_frame(buf, i, i as u64 * 33_333_333, Some(33_333_333))
            .unwrap();
        while let Ok(Some(_)) = enc.pull_encoded() {}
    }
    std::thread::sleep(Duration::from_millis(500));

    let baseline_gpu = gpu_mem_mib();
    let baseline_rss = rss_kb();
    println!("sustained encode: baseline GPU = {baseline_gpu} MiB, RSS = {baseline_rss} kB");

    // Main phase: 500 frames
    for i in 20..520u128 {
        let buf = enc.generator().acquire_surface(Some(i as i64)).unwrap();
        enc.submit_frame(buf, i, i as u64 * 33_333_333, Some(33_333_333))
            .unwrap();
        while let Ok(Some(_)) = enc.pull_encoded() {}
    }
    std::thread::sleep(Duration::from_millis(500));

    let final_gpu = gpu_mem_mib();
    let final_rss = rss_kb();
    let gpu_growth = final_gpu.saturating_sub(baseline_gpu);
    let rss_growth = final_rss.saturating_sub(baseline_rss);
    println!(
        "sustained encode: final GPU = {final_gpu} MiB (+{gpu_growth}), \
         RSS = {final_rss} kB (+{rss_growth})"
    );
    assert!(
        gpu_growth < GPU_GROWTH_TOLERANCE_MIB,
        "GPU memory grew by {gpu_growth} MiB (limit {GPU_GROWTH_TOLERANCE_MIB}) — possible GPU leak"
    );
    assert!(
        rss_growth < RSS_GROWTH_TOLERANCE_KB,
        "RSS grew by {rss_growth} kB (limit {RSS_GROWTH_TOLERANCE_KB}) — possible leak"
    );

    let _ = enc.finish(Some(3000));
}

// ---------------------------------------------------------------------------
// 6. GPU NvBufSurface generator acquire/release churn
// ---------------------------------------------------------------------------

#[test]
#[ignore]
#[serial]
fn leak_gpu_surface_acquire_release() {
    gstreamer::init().unwrap();
    cuda_init(0).unwrap();

    let gen = DsNvSurfaceBufferGenerator::new(
        VideoFormat::NV12,
        320,
        240,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .unwrap();

    // Warm-up
    for i in 0..10 {
        let _buf = gen.acquire_surface(Some(i)).unwrap();
    }
    std::thread::sleep(Duration::from_millis(300));

    let baseline_gpu = gpu_mem_mib();
    let baseline_rss = rss_kb();
    println!("surface churn: baseline GPU = {baseline_gpu} MiB, RSS = {baseline_rss} kB");

    // Acquire and immediately drop 2000 surfaces
    for i in 10..2_010i64 {
        let _buf = gen.acquire_surface(Some(i)).unwrap();
    }
    std::thread::sleep(Duration::from_millis(500));

    let final_gpu = gpu_mem_mib();
    let final_rss = rss_kb();
    let gpu_growth = final_gpu.saturating_sub(baseline_gpu);
    let rss_growth = final_rss.saturating_sub(baseline_rss);
    println!(
        "surface churn: final GPU = {final_gpu} MiB (+{gpu_growth}), \
         RSS = {final_rss} kB (+{rss_growth})"
    );
    assert!(
        gpu_growth < GPU_GROWTH_TOLERANCE_MIB,
        "GPU memory grew by {gpu_growth} MiB (limit {GPU_GROWTH_TOLERANCE_MIB}) — possible GPU leak"
    );
    assert!(
        rss_growth < RSS_GROWTH_TOLERANCE_KB,
        "RSS grew by {rss_growth} kB (limit {RSS_GROWTH_TOLERANCE_KB}) — possible leak"
    );
}

// ---------------------------------------------------------------------------
// 7. Full pipeline: engine with real GPU encode over many frames
// ---------------------------------------------------------------------------

#[test]
#[ignore]
#[serial]
fn leak_engine_gpu_encode_sustained() {
    let _ = env_logger::builder().is_test(true).try_init();
    gstreamer::init().unwrap();
    cuda_init(0).unwrap();

    let enc_count = Arc::new(AtomicUsize::new(0));
    let callbacks = Callbacks {
        on_encoded_frame: Some(Arc::new(SinkEncoded(enc_count.clone()))),
        ..Default::default()
    };
    let general = GeneralSpec {
        idle_timeout_secs: 120,
    };
    let mut engine = PicassoEngine::new(general, callbacks);

    let enc_cfg = encoder_config(320, 240);

    let spec = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(enc_cfg.clone()),
        },
        ..Default::default()
    };
    engine.set_source_spec("gpu-enc", spec).unwrap();

    // We need real NVMM buffers for the encoder path.
    // Create a generator to produce them.
    let src_gen = DsNvSurfaceBufferGenerator::new(
        VideoFormat::NV12,
        640,
        480,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .unwrap();

    // Warm-up: 10 frames
    for i in 0..10i64 {
        let frame = make_frame("gpu-enc", 640, 480);
        let buf = make_nvmm_buffer(&src_gen, i);
        let _ = engine.send_frame("gpu-enc", frame, buf, None);
        std::thread::sleep(Duration::from_millis(35));
    }
    std::thread::sleep(Duration::from_secs(1));

    let baseline_gpu = gpu_mem_mib();
    let baseline_rss = rss_kb();
    println!("engine GPU encode: baseline GPU = {baseline_gpu} MiB, RSS = {baseline_rss} kB");

    // Main phase: 200 frames
    for i in 10..210i64 {
        let frame = make_frame("gpu-enc", 640, 480);
        let buf = make_nvmm_buffer(&src_gen, i);
        let _ = engine.send_frame("gpu-enc", frame, buf, None);
        std::thread::sleep(Duration::from_millis(35));
    }
    std::thread::sleep(Duration::from_secs(2));

    let final_gpu = gpu_mem_mib();
    let final_rss = rss_kb();
    let gpu_growth = final_gpu.saturating_sub(baseline_gpu);
    let rss_growth = final_rss.saturating_sub(baseline_rss);
    let total_encoded = enc_count.load(Ordering::Relaxed);
    println!(
        "engine GPU encode: final GPU = {final_gpu} MiB (+{gpu_growth}), \
         RSS = {final_rss} kB (+{rss_growth}), encoded_frames = {total_encoded}"
    );
    assert!(
        gpu_growth < GPU_GROWTH_TOLERANCE_MIB,
        "GPU memory grew by {gpu_growth} MiB (limit {GPU_GROWTH_TOLERANCE_MIB}) — possible GPU leak"
    );
    assert!(
        rss_growth < RSS_GROWTH_TOLERANCE_KB,
        "RSS grew by {rss_growth} kB (limit {RSS_GROWTH_TOLERANCE_KB}) — possible leak"
    );
    assert!(
        total_encoded > 100,
        "Expected >100 encoded frames, got {total_encoded} — encode path may be broken"
    );

    engine.send_eos("gpu-enc").unwrap();
    std::thread::sleep(Duration::from_secs(1));
    engine.shutdown();
}
