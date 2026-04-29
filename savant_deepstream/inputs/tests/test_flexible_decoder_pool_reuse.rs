//! Integration tests for the [`FlexibleDecoder`] pool-reuse strategy.
//!
//! The decoder caches its RGBA output `BufferGenerator` in a shared
//! [`PoolCacheRegistry`] and reuses it across activations whenever
//! the requested params (resolution, format, GPU id, memory type,
//! pool size, fps) are unchanged.  This test file exercises:
//!
//!   * Cache-hit on steady-state submits (same dims, same source).
//!   * Cache-replacement on dimension change.
//!   * Cache **survives** `graceful_shutdown` / `shutdown` (the
//!     cross-`source_eos` reuse contract that drives the
//!     `FlexibleDecoderPool` cars-demo-zmq optimisation).
//!   * Cache cleared on private-registry decoder drop.
//!   * Cache-hit across a worker-restart cycle (H.264 `--no-eos`
//!     PTS rebase).
//!
//! Pool identity is observed via
//! [`FlexibleDecoder::pool_cache_addr`] which returns the address
//! of the cached `Arc<Mutex<BufferGenerator>>`.  The Arc address
//! is stable for the lifetime of the cache entry, so equality
//! across submits is a reliable cache-hit signal.

mod common;

use common::*;
use deepstream_inputs::flexible_decoder::{
    FlexibleDecoder, FlexibleDecoderConfig, FlexibleDecoderOutput,
};
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrame, VideoFrameTranscodingMethod,
};
use savant_core::primitives::video_codec::VideoCodec;
use serial_test::serial;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

// ── JPEG helpers ────────────────────────────────────────────────────

fn make_jpeg(width: u32, height: u32) -> Vec<u8> {
    let img = image::RgbaImage::from_pixel(width, height, image::Rgba([128, 64, 32, 255]));
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Jpeg).unwrap();
    buf.into_inner()
}

fn jpeg_frame(source_id: &str, width: i64, height: i64, pts_ns: i64) -> VideoFrame {
    VideoFrame::new(
        source_id,
        (30, 1),
        width,
        height,
        VideoFrameContent::None,
        VideoFrameTranscodingMethod::Copy,
        Some(VideoCodec::Jpeg),
        None,
        (1, 1_000_000_000),
        pts_ns,
        None,
        None,
    )
    .expect("jpeg_frame")
}

fn dec_callback() -> impl Fn(FlexibleDecoderOutput) + Send + Sync + 'static {
    move |_out| {}
}

// ── Cache-hit / steady state ────────────────────────────────────────

#[test]
#[serial]
fn pool_cache_steady_state_keeps_same_pool() {
    init();
    let cfg = FlexibleDecoderConfig::new("src", 0, 4);
    let dec = FlexibleDecoder::new(cfg, dec_callback());

    let jpeg = make_jpeg(320, 240);

    dec.submit(&jpeg_frame("src", 320, 240, 0), Some(&jpeg))
        .expect("first submit");
    let addr_a = dec
        .pool_cache_addr()
        .expect("pool must be cached after first submit");

    for i in 1..5 {
        dec.submit(&jpeg_frame("src", 320, 240, i * 33_333_333), Some(&jpeg))
            .expect("steady-state submit");
        let addr = dec.pool_cache_addr().expect("pool still cached");
        assert_eq!(
            addr, addr_a,
            "steady-state submit #{i} replaced the pool (was {addr_a:#x}, now {addr:#x})"
        );
    }
}

// ── Cache replaced on dimension change ─────────────────────────────

#[test]
#[serial]
fn pool_cache_replaced_on_dim_change() {
    init();
    let cfg = FlexibleDecoderConfig::new("src", 0, 4);
    let dec = FlexibleDecoder::new(cfg, dec_callback());

    let jpeg_320 = make_jpeg(320, 240);
    let jpeg_640 = make_jpeg(640, 480);

    dec.submit(&jpeg_frame("src", 320, 240, 0), Some(&jpeg_320))
        .expect("submit 320x240");
    let addr_a = dec.pool_cache_addr().expect("pool cached after 320x240");

    dec.submit(&jpeg_frame("src", 640, 480, 33_333_333), Some(&jpeg_640))
        .expect("submit 640x480");
    let addr_b = dec.pool_cache_addr().expect("pool cached after 640x480");

    assert_ne!(
        addr_a, addr_b,
        "dim change must replace the cached pool (both addrs={addr_a:#x})"
    );

    dec.submit(&jpeg_frame("src", 640, 480, 66_666_666), Some(&jpeg_640))
        .expect("steady-state submit at 640x480");
    let addr_b2 = dec.pool_cache_addr().expect("pool still cached");
    assert_eq!(
        addr_b, addr_b2,
        "steady-state submit at new dims replaced the pool"
    );
}

// ── Cache survives shutdown (cross-source_eos reuse contract) ─────

/// `graceful_shutdown` no longer clears the
/// [`PoolCacheRegistry`].  The cache is owned by the registry
/// (private to this decoder in the standalone case) and survives
/// individual decoder teardown so that, when the registry is
/// shared with a parent
/// [`FlexibleDecoderPool`](deepstream_inputs::prelude::FlexibleDecoderPool),
/// a fresh decoder for the same `source_id` (typically created
/// after a `source_eos`-driven teardown) can reuse the cached
/// `BufferGenerator` and the pooled `EglCudaMeta` registrations
/// instead of rebuilding from scratch.
#[test]
#[serial]
fn pool_cache_survives_graceful_shutdown() {
    init();
    let cfg = FlexibleDecoderConfig::new("src", 0, 4);
    let dec = FlexibleDecoder::new(cfg, dec_callback());

    let jpeg = make_jpeg(320, 240);
    dec.submit(&jpeg_frame("src", 320, 240, 0), Some(&jpeg))
        .expect("first submit");
    let addr_before = dec.pool_cache_addr().expect("cache populated");

    dec.graceful_shutdown().expect("graceful_shutdown");
    let addr_after = dec
        .pool_cache_addr()
        .expect("graceful_shutdown must NOT clear the pool cache");
    assert_eq!(
        addr_before, addr_after,
        "graceful_shutdown must preserve the cached BufferGenerator address"
    );
}

#[test]
#[serial]
fn pool_cache_survives_immediate_shutdown() {
    init();
    let cfg = FlexibleDecoderConfig::new("src", 0, 4);
    let dec = FlexibleDecoder::new(cfg, dec_callback());

    let jpeg = make_jpeg(320, 240);
    dec.submit(&jpeg_frame("src", 320, 240, 0), Some(&jpeg))
        .expect("first submit");
    let addr_before = dec.pool_cache_addr().expect("cache populated");

    dec.shutdown();
    let addr_after = dec
        .pool_cache_addr()
        .expect("shutdown must NOT clear the pool cache");
    assert_eq!(
        addr_before, addr_after,
        "shutdown must preserve the cached BufferGenerator address"
    );
}

/// When the [`FlexibleDecoder`] owns its registry exclusively
/// (created via [`FlexibleDecoder::new`]), dropping the decoder
/// must drop the registry — releasing every cached
/// `BufferGenerator` after running the F2
/// [`cuda_device_synchronize`] fence.
///
/// We observe this via a [`std::sync::Weak`] handle on the
/// registry: holding a strong [`Arc`] clone here would keep the
/// registry alive past the decoder drop and mask the actual
/// release.  After dropping the decoder the weak handle must
/// fail to upgrade, which is only possible when the registry's
/// `Drop` (which calls `clear_all`) has fired.
#[test]
#[serial]
fn pool_cache_cleared_when_decoder_dropped() {
    init();
    let cfg = FlexibleDecoderConfig::new("src", 0, 4);
    let dec = FlexibleDecoder::new(cfg, dec_callback());

    let jpeg = make_jpeg(320, 240);
    dec.submit(&jpeg_frame("src", 320, 240, 0), Some(&jpeg))
        .expect("first submit");

    let weak_registry = {
        let registry = dec.pool_cache();
        assert_eq!(registry.len(), 1, "cache populated before drop");
        Arc::downgrade(&registry)
    };

    drop(dec);
    assert!(
        weak_registry.upgrade().is_none(),
        "decoder drop must release the private registry — \
         a `Weak` handle must fail to upgrade once `Drop` for \
         `PoolCacheRegistry` (which calls `clear_all`) has fired"
    );
}

// ── Cache survives worker-restart cycle (H.264 --no-eos rebase) ────

const PER_SUBMIT_BUDGET: Duration = Duration::from_secs(15);

fn submit_h264_burst(
    dec: &FlexibleDecoder,
    aus: &[AccessUnit],
    source_id: &str,
    width: i64,
    height: i64,
    pts_offset_ns: u64,
) {
    for (i, au) in aus.iter().enumerate() {
        let pts = pts_offset_ns + au.pts_ns;
        let dts = au.dts_ns.map(|d| (pts_offset_ns + d) as i64);
        let dur = au.duration_ns.map(|d| d as i64);
        let frame = make_video_frame_ns(
            source_id,
            VideoCodec::H264,
            width,
            height,
            pts as i64,
            dts,
            dur,
            None,
        );
        let start = Instant::now();
        dec.submit(&frame, Some(&au.data))
            .unwrap_or_else(|e| panic!("submit AU {i} failed: {e}"));
        let elapsed = start.elapsed();
        assert!(
            elapsed < PER_SUBMIT_BUDGET,
            "submit AU {i} blocked for {elapsed:?} (> {PER_SUBMIT_BUDGET:?})"
        );
    }
}

fn find_h264_entry(manifest: &Manifest) -> Option<&AssetEntry> {
    let platform = current_platform_tag();
    let candidates = [
        "test_h264_bt709_ip.mp4",
        "test_h264_bt601_ip.mp4",
        "test_h264_bt709_i.mp4",
        "test_h264_bt709_p.mp4",
    ];
    for name in candidates {
        if let Some(entry) = manifest.assets.iter().find(|e| e.file == name) {
            if asset_supported_on_platform(entry, &platform) {
                return Some(entry);
            }
        }
    }
    None
}

/// Reproduces the cars-demo-zmq `--no-eos` rebased-PTS pattern that
/// triggers a worker restart on every cycle boundary, and asserts the
/// `BufferGenerator` cache entry survives the restart.
///
/// On every cycle the rebased IDR trips the
/// [`StrictDecodeOrder`](savant_gstreamer::pipeline::PtsPolicy::StrictDecodeOrder)
/// feeder, the worker thread exits, and the next `submit` takes
/// `handle_active`'s worker-died branch — tearing down the
/// [`NvDecoder`](deepstream_decoders::NvDecoder), draining the
/// frame map, and falling back to `handle_idle` which calls
/// [`FlexibleDecoder::activate`] again.  Because the new frame's
/// resolution matches the cached pool, [`acquire_or_build_pool`]
/// returns the same `Arc<Mutex<BufferGenerator>>` — observed here as
/// a stable [`pool_cache_addr`].
///
/// Skipped if no H.264 asset is available on the current platform.
#[test]
#[serial]
fn pool_cache_preserved_across_worker_restart_h264() {
    init();
    let manifest = load_manifest();
    let Some(entry) = find_h264_entry(&manifest) else {
        eprintln!("no H.264 asset available for current platform — skipping");
        return;
    };

    let aus = demux_mp4_to_access_units(entry);
    assert!(!aus.is_empty(), "demuxer produced no access units");

    let cfg = FlexibleDecoderConfig::new("src", 0, 8);
    let dec = Arc::new(FlexibleDecoder::new(cfg, dec_callback()));

    submit_h264_burst(
        &dec,
        &aus,
        "src",
        entry.width as i64,
        entry.height as i64,
        0,
    );
    std::thread::sleep(Duration::from_millis(500));
    let addr_first = dec
        .pool_cache_addr()
        .expect("pool cached after first burst");

    const REBASED_CYCLES: usize = 4;
    for cycle in 1..=REBASED_CYCLES {
        submit_h264_burst(
            &dec,
            &aus,
            "src",
            entry.width as i64,
            entry.height as i64,
            0,
        );
        std::thread::sleep(Duration::from_millis(400));
        let addr = dec
            .pool_cache_addr()
            .expect("pool still cached after rebased burst");
        assert_eq!(
            addr, addr_first,
            "cycle {cycle}: pool replaced across worker-restart \
             with same dims (was {addr_first:#x}, now {addr:#x})"
        );
    }

    let _ = Arc::strong_count(&dec);
    let dec = Arc::try_unwrap(dec)
        .map_err(|_| ())
        .expect("decoder Arc has unexpected outstanding strong refs");
    let registry = dec.pool_cache();
    dec.graceful_shutdown().expect("graceful_shutdown");
    assert!(
        !registry.is_empty(),
        "graceful_shutdown must NOT clear the pool cache (registry-owned)"
    );
}

// ── Cross-`source_eos` reuse via FlexibleDecoderPool ──────────────

/// `FlexibleDecoderPool::source_eos` tears down the per-source
/// [`FlexibleDecoder`] but **must not** drop the cached
/// `BufferGenerator`.  This test simulates the cars-demo-zmq
/// teardown / restart pattern by calling
/// [`FlexibleDecoderPool::source_eos`] between two submit cycles
/// for the same `source_id` and asserting that
/// [`PoolCacheRegistry::pool_addr`] is stable across the cycle —
/// i.e. the second cycle gets a `pool_cache hit`, not a
/// `pool_cache build`.
#[test]
#[serial]
fn pool_survives_source_eos_in_flexible_decoder_pool() {
    use deepstream_inputs::prelude::{FlexibleDecoderPool, FlexibleDecoderPoolConfig};
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    init();
    let cfg = FlexibleDecoderPoolConfig::new(0, 4, Duration::from_secs(3600))
        .idle_timeout(Duration::from_secs(5));
    let frame_count = Arc::new(AtomicUsize::new(0));
    let frame_count_cb = Arc::clone(&frame_count);
    let pool = FlexibleDecoderPool::new(cfg, move |out| {
        if matches!(out, FlexibleDecoderOutput::Frame { .. }) {
            frame_count_cb.fetch_add(1, Ordering::Relaxed);
        }
    });

    let registry = pool.pool_cache();
    let jpeg = make_jpeg(320, 240);

    // Cycle 1: submit a frame for "src", read the cached pool addr.
    pool.submit(&jpeg_frame("src", 320, 240, 0), Some(&jpeg))
        .expect("submit cycle 1");
    let addr_cycle1 = registry
        .pool_addr("src")
        .expect("registry must have an entry for 'src' after first submit");

    // End-of-cycle EOS — this is the cars-demo-zmq pattern.  The
    // per-source FlexibleDecoder is drained and dropped.  The
    // cached pool entry MUST survive.
    pool.source_eos("src").expect("source_eos");
    let addr_after_eos = registry
        .pool_addr("src")
        .expect("registry entry must survive source_eos teardown");
    assert_eq!(
        addr_cycle1, addr_after_eos,
        "source_eos must NOT clear the cached pool (was {addr_cycle1:#x}, \
         after-eos {addr_after_eos:#x})"
    );

    // Cycle 2: submit again for "src".  A fresh FlexibleDecoder
    // is constructed but it inherits the shared registry, so the
    // first call to `acquire_or_build_pool` lands on the cache
    // entry from cycle 1 — `pool_cache hit`, not `pool_cache build`.
    pool.submit(&jpeg_frame("src", 320, 240, 33_333_333), Some(&jpeg))
        .expect("submit cycle 2");
    let addr_cycle2 = registry
        .pool_addr("src")
        .expect("registry entry must still be present after cycle-2 submit");
    assert_eq!(
        addr_cycle1, addr_cycle2,
        "cycle-2 decoder must reuse the cycle-1 BufferGenerator \
         (was {addr_cycle1:#x}, now {addr_cycle2:#x})"
    );

    // graceful_shutdown drains every per-source decoder and
    // clears the registry.
    let mut pool = pool;
    pool.graceful_shutdown().expect("graceful_shutdown");
    assert!(
        registry.is_empty(),
        "FlexibleDecoderPool::graceful_shutdown must clear the pool cache"
    );
}

// Suppress dead_code warnings from `common.rs` items used only by other
// integration test files in this crate.
#[allow(dead_code)]
fn _unused_imports_silencer() {
    let _ = Ordering::Relaxed;
}
