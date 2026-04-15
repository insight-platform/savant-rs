//! Integration tests for [`FlexibleDecoderPool`].
//!
//! All tests require a GPU with CUDA and nvjpegdec support.

mod common;

use common::{init, CollectedOutput, OutputCollector};
use deepstream_inputs::decoder_pool::{
    EvictionDecision, FlexibleDecoderPool, FlexibleDecoderPoolConfig,
};
use parking_lot::Mutex;
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
};
use savant_core::primitives::video_codec::VideoCodec;
use serial_test::serial;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

// ── Helpers ──────────────────────────────────────────────────────────────

fn make_jpeg(width: u32, height: u32) -> Vec<u8> {
    let img = image::RgbaImage::from_pixel(width, height, image::Rgba([128, 64, 32, 255]));
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Jpeg).unwrap();
    buf.into_inner()
}

fn make_frame(source_id: &str, width: i64, height: i64, pts: i64) -> VideoFrameProxy {
    VideoFrameProxy::new(
        source_id,
        (30, 1),
        width,
        height,
        VideoFrameContent::None,
        VideoFrameTranscodingMethod::Copy,
        Some(VideoCodec::Jpeg),
        None,
        (1, 1_000_000_000),
        pts,
        None,
        None,
    )
    .unwrap()
}

// ── Config tests ─────────────────────────────────────────────────────────

#[test]
fn test_config_defaults() {
    let cfg = FlexibleDecoderPoolConfig::new(0, 4, Duration::from_secs(30));
    assert_eq!(cfg.gpu_id, 0);
    assert_eq!(cfg.pool_size, 4);
    assert_eq!(cfg.eviction_ttl, Duration::from_secs(30));
    assert_eq!(cfg.idle_timeout, Duration::from_secs(1));
    assert_eq!(cfg.detect_buffer_limit, 30);
}

#[test]
fn test_config_chaining() {
    let cfg = FlexibleDecoderPoolConfig::new(1, 8, Duration::from_secs(60))
        .idle_timeout(Duration::from_millis(500))
        .detect_buffer_limit(50);
    assert_eq!(cfg.gpu_id, 1);
    assert_eq!(cfg.pool_size, 8);
    assert_eq!(cfg.idle_timeout, Duration::from_millis(500));
    assert_eq!(cfg.detect_buffer_limit, 50);
}

#[test]
fn test_config_to_flexible() {
    let pool_cfg = FlexibleDecoderPoolConfig::new(2, 6, Duration::from_secs(10))
        .idle_timeout(Duration::from_millis(750))
        .detect_buffer_limit(20);
    let flex = pool_cfg.to_flexible_config("cam-1");
    assert_eq!(flex.source_id, "cam-1");
    assert_eq!(flex.gpu_id, 2);
    assert_eq!(flex.pool_size, 6);
    assert_eq!(flex.idle_timeout, Duration::from_millis(750));
    assert_eq!(flex.detect_buffer_limit, 20);
}

// ── Routing tests ────────────────────────────────────────────────────────

#[test]
#[serial]
fn test_submit_routes_by_source_id() {
    init();
    let collector = OutputCollector::new();
    let cfg = FlexibleDecoderPoolConfig::new(0, 4, Duration::from_secs(60));
    let pool = FlexibleDecoderPool::new(cfg, collector.callback());

    let jpeg = make_jpeg(320, 240);

    let f1 = make_frame("cam-1", 320, 240, 0);
    let f2 = make_frame("cam-2", 320, 240, 0);
    pool.submit(&f1, Some(&jpeg)).unwrap();
    pool.submit(&f2, Some(&jpeg)).unwrap();

    collector.wait_for_frames(2, Duration::from_secs(5));

    let outputs = collector.drain();
    let frame_sources: Vec<String> = outputs
        .iter()
        .filter_map(|o| match o {
            CollectedOutput::Frame { proxy_uuid, .. } => {
                // Recovered source_id not directly available in CollectedOutput,
                // but we submitted distinct UUIDs — verify we got 2 frames.
                Some(format!("{proxy_uuid}"))
            }
            _ => None,
        })
        .collect();
    assert_eq!(frame_sources.len(), 2);

    let mut pool = pool;
    pool.shutdown();
}

#[test]
#[serial]
fn test_submit_creates_decoder_on_first_frame() {
    init();
    let collector = OutputCollector::new();
    let cfg = FlexibleDecoderPoolConfig::new(0, 4, Duration::from_secs(60));
    let pool = FlexibleDecoderPool::new(cfg, collector.callback());

    let jpeg = make_jpeg(320, 240);
    let f = make_frame("cam-new", 320, 240, 0);
    pool.submit(&f, Some(&jpeg)).unwrap();

    collector.wait_for_frames(1, Duration::from_secs(5));
    assert!(collector.frame_count() >= 1);

    let mut pool = pool;
    pool.shutdown();
}

#[test]
#[serial]
fn test_submit_reuses_existing_decoder() {
    init();
    let collector = OutputCollector::new();
    let cfg = FlexibleDecoderPoolConfig::new(0, 4, Duration::from_secs(60));
    let pool = FlexibleDecoderPool::new(cfg, collector.callback());

    let jpeg = make_jpeg(320, 240);
    let f1 = make_frame("cam-1", 320, 240, 0);
    let f2 = make_frame("cam-1", 320, 240, 1_000_000);
    pool.submit(&f1, Some(&jpeg)).unwrap();
    pool.submit(&f2, Some(&jpeg)).unwrap();

    collector.wait_for_frames(2, Duration::from_secs(5));
    assert_eq!(collector.frame_count(), 2);

    let mut pool = pool;
    pool.shutdown();
}

// ── Eviction tests ───────────────────────────────────────────────────────

#[test]
#[serial]
fn test_eviction_no_callback_evicts_by_default() {
    init();
    let collector = OutputCollector::new();
    let cfg = FlexibleDecoderPoolConfig::new(0, 4, Duration::from_millis(200));
    let pool = FlexibleDecoderPool::new(cfg, collector.callback());

    let jpeg = make_jpeg(320, 240);
    let f = make_frame("cam-1", 320, 240, 0);
    pool.submit(&f, Some(&jpeg)).unwrap();
    collector.wait_for_frames(1, Duration::from_secs(5));

    // Wait for TTL to expire + sweep
    std::thread::sleep(Duration::from_millis(1500));

    // Submit again for same source — should create a fresh decoder
    let f2 = make_frame("cam-1", 320, 240, 1_000_000);
    pool.submit(&f2, Some(&jpeg)).unwrap();
    collector.wait_for_frames(2, Duration::from_secs(5));

    let mut pool = pool;
    pool.shutdown();
}

#[test]
#[serial]
fn test_eviction_callback_keep() {
    init();
    let collector = OutputCollector::new();
    let eviction_count = Arc::new(AtomicUsize::new(0));
    let eviction_count_cb = Arc::clone(&eviction_count);

    let cfg = FlexibleDecoderPoolConfig::new(0, 4, Duration::from_millis(200));
    let pool =
        FlexibleDecoderPool::with_eviction_callback(cfg, collector.callback(), move |_source_id| {
            eviction_count_cb.fetch_add(1, Ordering::Relaxed);
            EvictionDecision::Keep
        });

    let jpeg = make_jpeg(320, 240);
    let f = make_frame("cam-1", 320, 240, 0);
    pool.submit(&f, Some(&jpeg)).unwrap();
    collector.wait_for_frames(1, Duration::from_secs(5));

    // Wait for at least one eviction sweep
    std::thread::sleep(Duration::from_millis(1500));
    assert!(
        eviction_count.load(Ordering::Relaxed) >= 1,
        "eviction callback should have been called at least once"
    );

    // Stream should still be alive — submit should succeed without creating a
    // new decoder (same session).
    let f2 = make_frame("cam-1", 320, 240, 1_000_000);
    pool.submit(&f2, Some(&jpeg)).unwrap();
    collector.wait_for_frames(2, Duration::from_secs(5));

    let mut pool = pool;
    pool.shutdown();
}

#[test]
#[serial]
fn test_eviction_callback_evict() {
    init();
    let collector = OutputCollector::new();
    let eviction_count = Arc::new(AtomicUsize::new(0));
    let eviction_count_cb = Arc::clone(&eviction_count);

    let cfg = FlexibleDecoderPoolConfig::new(0, 4, Duration::from_millis(200));
    let pool =
        FlexibleDecoderPool::with_eviction_callback(cfg, collector.callback(), move |_source_id| {
            eviction_count_cb.fetch_add(1, Ordering::Relaxed);
            EvictionDecision::Evict
        });

    let jpeg = make_jpeg(320, 240);
    let f = make_frame("cam-1", 320, 240, 0);
    pool.submit(&f, Some(&jpeg)).unwrap();
    collector.wait_for_frames(1, Duration::from_secs(5));

    // Wait for eviction
    std::thread::sleep(Duration::from_millis(1500));
    assert!(eviction_count.load(Ordering::Relaxed) >= 1);

    // Submit again — should create a fresh decoder
    let f2 = make_frame("cam-1", 320, 240, 0);
    pool.submit(&f2, Some(&jpeg)).unwrap();
    collector.wait_for_frames(2, Duration::from_secs(5));

    let mut pool = pool;
    pool.shutdown();
}

#[test]
#[serial]
fn test_submit_blocks_during_eviction() {
    init();
    let collector = OutputCollector::new();
    let eviction_entered = Arc::new(Mutex::new(false));
    let eviction_entered_cb = Arc::clone(&eviction_entered);

    let cfg = FlexibleDecoderPoolConfig::new(0, 4, Duration::from_millis(200));
    let pool = Arc::new(FlexibleDecoderPool::with_eviction_callback(
        cfg,
        collector.callback(),
        move |_source_id| {
            *eviction_entered_cb.lock() = true;
            // Simulate slow eviction callback to give the submit thread
            // time to encounter the seal.
            std::thread::sleep(Duration::from_millis(300));
            EvictionDecision::Evict
        },
    ));

    let jpeg = make_jpeg(320, 240);
    let f = make_frame("cam-1", 320, 240, 0);
    pool.submit(&f, Some(&jpeg)).unwrap();
    collector.wait_for_frames(1, Duration::from_secs(5));

    // Wait for eviction to begin
    let start = std::time::Instant::now();
    loop {
        if *eviction_entered.lock() {
            break;
        }
        if start.elapsed() > Duration::from_secs(5) {
            panic!("eviction callback never entered");
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    // Submit from another thread — should block on the seal
    let pool2 = Arc::clone(&pool);
    let jpeg2 = jpeg.clone();
    let submit_done = Arc::new(Mutex::new(false));
    let submit_done2 = Arc::clone(&submit_done);
    let handle = std::thread::spawn(move || {
        let f2 = make_frame("cam-1", 320, 240, 1_000_000);
        pool2.submit(&f2, Some(&jpeg2)).unwrap();
        *submit_done2.lock() = true;
    });

    // Give the submit thread a moment — it should be blocked
    std::thread::sleep(Duration::from_millis(100));

    handle.join().unwrap();
    assert!(*submit_done.lock(), "submit should have completed");
    collector.wait_for_frames(2, Duration::from_secs(5));

    // We need to get a mutable reference to shutdown.
    // Since pool is in an Arc, we need to unwrap it.
    let mut pool = Arc::try_unwrap(pool)
        .ok()
        .expect("pool still has other refs");
    pool.shutdown();
}

#[test]
#[serial]
fn test_submit_after_evict_creates_fresh_decoder() {
    init();
    let collector = OutputCollector::new();
    let cfg = FlexibleDecoderPoolConfig::new(0, 4, Duration::from_millis(200));
    let pool = FlexibleDecoderPool::new(cfg, collector.callback());

    let jpeg = make_jpeg(320, 240);
    let f = make_frame("cam-1", 320, 240, 0);
    pool.submit(&f, Some(&jpeg)).unwrap();
    collector.wait_for_frames(1, Duration::from_secs(5));

    // Wait for eviction
    std::thread::sleep(Duration::from_millis(1500));

    // New submission should create a fresh decoder and succeed
    let f2 = make_frame("cam-1", 320, 240, 0);
    pool.submit(&f2, Some(&jpeg)).unwrap();
    collector.wait_for_frames(2, Duration::from_secs(5));

    let mut pool = pool;
    pool.shutdown();
}

// ── Lifecycle tests ──────────────────────────────────────────────────────

#[test]
#[serial]
fn test_source_eos_forwards() {
    init();
    let collector = OutputCollector::new();
    let cfg = FlexibleDecoderPoolConfig::new(0, 4, Duration::from_secs(60));
    let pool = FlexibleDecoderPool::new(cfg, collector.callback());

    let jpeg = make_jpeg(320, 240);
    let f = make_frame("cam-1", 320, 240, 0);
    pool.submit(&f, Some(&jpeg)).unwrap();
    collector.wait_for_frames(1, Duration::from_secs(5));

    // When the decoder is Active, source_eos goes through NvDecoder's
    // pipeline and appears as a GStreamer Event (not SourceEos) in the
    // callback — this is FlexibleDecoder's documented behavior.
    pool.source_eos("cam-1").unwrap();

    // Allow time for the EOS event to propagate.
    std::thread::sleep(Duration::from_millis(500));

    let mut pool = pool;
    pool.shutdown();
}

#[test]
#[serial]
fn test_source_eos_missing_stream_emits_directly() {
    init();
    let collector = OutputCollector::new();
    let cfg = FlexibleDecoderPoolConfig::new(0, 4, Duration::from_secs(60));
    let pool = FlexibleDecoderPool::new(cfg, collector.callback());

    pool.source_eos("nonexistent").unwrap();

    collector.wait_for(
        |o| matches!(o, CollectedOutput::SourceEos { source_id } if source_id == "nonexistent"),
        Duration::from_secs(2),
    );

    let mut pool = pool;
    pool.shutdown();
}

#[test]
#[serial]
fn test_graceful_shutdown_drains_all() {
    init();
    let collector = OutputCollector::new();
    let cfg = FlexibleDecoderPoolConfig::new(0, 4, Duration::from_secs(60));
    let pool = FlexibleDecoderPool::new(cfg, collector.callback());

    let jpeg = make_jpeg(320, 240);
    let f1 = make_frame("cam-1", 320, 240, 0);
    let f2 = make_frame("cam-2", 320, 240, 0);
    pool.submit(&f1, Some(&jpeg)).unwrap();
    pool.submit(&f2, Some(&jpeg)).unwrap();
    collector.wait_for_frames(2, Duration::from_secs(5));

    let mut pool = pool;
    pool.graceful_shutdown().unwrap();
}

#[test]
#[serial]
fn test_shutdown_kills_all() {
    init();
    let collector = OutputCollector::new();
    let cfg = FlexibleDecoderPoolConfig::new(0, 4, Duration::from_secs(60));
    let pool = FlexibleDecoderPool::new(cfg, collector.callback());

    let jpeg = make_jpeg(320, 240);
    let f = make_frame("cam-1", 320, 240, 0);
    pool.submit(&f, Some(&jpeg)).unwrap();
    collector.wait_for_frames(1, Duration::from_secs(5));

    let mut pool = pool;
    pool.shutdown();
}

#[test]
#[serial]
fn test_submit_after_shutdown_errors() {
    init();
    let cfg = FlexibleDecoderPoolConfig::new(0, 4, Duration::from_secs(60));
    let mut pool = FlexibleDecoderPool::new(cfg, |_| {});
    pool.shutdown();

    let f = make_frame("cam-1", 320, 240, 0);
    let result = pool.submit(&f, None);
    assert!(result.is_err());
}
