//! Tests for the per-source release sequence wired to `reset_stream`.
//!
//! Each test uses a per-source `BufferGenerator` configured with
//! `min == max == 1`. Submitting a buffer from such a pool drains it; the
//! next `try_acquire` against the same generator fails with
//! [`NvBufSurfaceError::PoolExhausted`] until the tracker releases the
//! pinned prev-frame buffer (via service batch + `PAD_DELETED`) and one
//! regular batch flows through to actually drop it. This makes the
//! pool a precise per-source pin probe.
//!
//! Run with: `cargo test -p savant-deepstream-nvtracker
//! --test test_reset_stream_release -- --test-threads=1`

mod common;

use deepstream_buffers::{
    BufferGenerator, NvBufSurfaceError, NvBufSurfaceMemType, SavantIdMetaKind, SharedBuffer,
    VideoFormat,
};
use deepstream_nvtracker::{
    default_ll_lib_path, NvTracker, NvTrackerConfig, NvTrackerOutput, Roi, TrackedFrame,
    TrackingIdResetMode,
};
use savant_core::primitives::RBBox;
use serial_test::serial;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets")
}

fn iou_config_path() -> PathBuf {
    assets_dir().join("config_tracker_IOU.yml")
}

fn rb(left: f32, top: f32, w: f32, h: f32) -> RBBox {
    RBBox::ltwh(left, top, w, h).expect("RBBox::ltwh")
}

/// Per-source single-slot generator (`min == max == 1`). The pool can
/// hand out exactly one buffer at a time, which makes the next
/// `try_acquire` a deterministic probe for "is the previous buffer
/// still pinned".
fn make_pinned_generator(w: u32, h: u32) -> BufferGenerator {
    BufferGenerator::builder(VideoFormat::RGBA, w, h)
        .gpu_id(0)
        .min_buffers(1)
        .max_buffers(1)
        .mem_type(NvBufSurfaceMemType::Default)
        .build()
        .expect("BufferGenerator::build")
}

fn make_frame(source: &str, gen: &BufferGenerator, rois: Vec<Roi>) -> TrackedFrame {
    let buf: SharedBuffer = gen.acquire(None).expect("acquire pinned buffer");
    let mut map = HashMap::new();
    if !rois.is_empty() {
        map.insert(0i32, rois);
    }
    TrackedFrame {
        source: source.to_string(),
        buffer: buf,
        rois: map,
    }
}

fn frame_ids(n: usize) -> Vec<SavantIdMetaKind> {
    (0..n)
        .map(|i| SavantIdMetaKind::Frame(i as u128 + 1))
        .collect()
}

fn try_create_tracker(stale: Option<Duration>) -> Option<NvTracker> {
    let lib = default_ll_lib_path();
    if !std::path::Path::new(&lib).is_file() {
        eprintln!("skip: missing {lib}");
        return None;
    }
    let cfg_path = iou_config_path();
    if !cfg_path.is_file() {
        eprintln!("skip: missing {}", cfg_path.display());
        return None;
    }
    let mut c = NvTrackerConfig::new(lib, cfg_path.to_string_lossy());
    c.tracking_id_reset_mode = TrackingIdResetMode::None;
    c.tracker_width = 320;
    c.tracker_height = 240;
    c.max_batch_size = 4;
    c.stale_source_after = stale;
    NvTracker::new(c).ok()
}

/// Submit a single-frame batch and drain `recv` until a `Tracking`
/// output is observed.  Service-marked outputs are filtered out by
/// `recv` itself (per `convert_output`); they never surface here.
fn submit_and_drain(tracker: &NvTracker, frame: TrackedFrame) {
    tracker.submit(&[frame], frame_ids(1)).expect("submit");
    loop {
        match tracker.recv().expect("recv") {
            NvTrackerOutput::Tracking(_) => return,
            NvTrackerOutput::Event(_) => continue,
            NvTrackerOutput::Eos { source_id } => {
                panic!("unexpected EOS for {source_id} during drain")
            }
            NvTrackerOutput::Error(e) => panic!("tracker error: {e}"),
        }
    }
}

/// Helper: probe whether the per-source pool has released its buffer.
/// Polls `try_acquire` for up to `timeout`, returning `true` on success.
fn wait_pool_free(gen: &BufferGenerator, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        match gen.try_acquire(None) {
            Ok(_buf) => return true,
            Err(NvBufSurfaceError::PoolExhausted) => {}
            Err(e) => panic!("unexpected try_acquire error: {e}"),
        }
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
#[serial]
fn test_reset_releases_pinned_prev_frame_after_one_batch() {
    common::init();
    let Some(tracker) = try_create_tracker(None) else {
        return;
    };

    // One pool per source so pin release is observable per pool.
    let gen_a = make_pinned_generator(320, 240);
    // Independent throw-away pool for the post-reset regular batch
    // that actually drops the tracker's pin.
    let gen_aux = make_pinned_generator(320, 240);

    let roi = Roi {
        id: 1,
        bbox: rb(40.0, 40.0, 80.0, 60.0),
    };

    submit_and_drain(&tracker, make_frame("cam-a", &gen_a, vec![roi.clone()]));

    // Pool is exhausted: the tracker still pins the prev-frame.
    assert!(
        matches!(
            gen_a.try_acquire(None),
            Err(NvBufSurfaceError::PoolExhausted)
        ),
        "tracker should still hold prev-frame for cam-a"
    );

    tracker.reset_stream("cam-a").expect("reset cam-a");

    // The service batch alone is not enough: the pin is released only
    // when the *next* regular batch flows through. Use a different
    // source so we don't immediately re-pin gen_a.
    submit_and_drain(&tracker, make_frame("cam-z", &gen_aux, vec![roi.clone()]));

    assert!(
        wait_pool_free(&gen_a, Duration::from_secs(3)),
        "gen_a's prev-frame pin should be released after reset_stream + one regular batch"
    );

    let _ = tracker.shutdown();
    drop((gen_a, gen_aux));
}

/// Functional isolation: resetting one source must not break tracking
/// continuity for any other live source.  Pool-level isolation is *not*
/// observable here because `nvtracker` pins the entire prev-batch
/// `NvBufSurface` (every per-source slot in it), so any new batch
/// — service or regular — implicitly releases the previous batch's pin
/// across all sources that participated in it.  We therefore assert the
/// behaviour that actually matters to callers: after `reset_stream(A)`,
/// source B can keep submitting frames and the tracker keeps emitting
/// `Tracking` outputs for B without errors or stalls.
#[test]
#[serial]
fn test_reset_isolates_target_source() {
    common::init();
    let Some(tracker) = try_create_tracker(None) else {
        return;
    };

    // Multi-buffer pools because we reuse them across several batches.
    let gen_a = BufferGenerator::builder(VideoFormat::RGBA, 320, 240)
        .gpu_id(0)
        .min_buffers(2)
        .max_buffers(4)
        .mem_type(NvBufSurfaceMemType::Default)
        .build()
        .expect("gen_a build");
    let gen_b = BufferGenerator::builder(VideoFormat::RGBA, 320, 240)
        .gpu_id(0)
        .min_buffers(2)
        .max_buffers(4)
        .mem_type(NvBufSurfaceMemType::Default)
        .build()
        .expect("gen_b build");

    let roi = Roi {
        id: 1,
        bbox: rb(40.0, 40.0, 80.0, 60.0),
    };

    // Warm up both sources so the tracker has per-source state for each.
    submit_and_drain(&tracker, make_frame("cam-a", &gen_a, vec![roi.clone()]));
    submit_and_drain(&tracker, make_frame("cam-b", &gen_b, vec![roi.clone()]));

    // Reset one of them.
    tracker.reset_stream("cam-a").expect("reset cam-a");

    // The other source must continue to flow without hangs or errors,
    // and the next cam-a submit must also succeed (i.e. the source pad
    // is fully re-armed).
    for _ in 0..3 {
        submit_and_drain(&tracker, make_frame("cam-b", &gen_b, vec![roi.clone()]));
    }
    submit_and_drain(&tracker, make_frame("cam-a", &gen_a, vec![roi.clone()]));
    submit_and_drain(&tracker, make_frame("cam-b", &gen_b, vec![roi.clone()]));

    let _ = tracker.shutdown();
    drop((gen_a, gen_b));
}

#[test]
#[serial]
fn test_service_batch_is_filtered_from_recv() {
    common::init();
    let Some(tracker) = try_create_tracker(None) else {
        return;
    };

    let gen_a = make_pinned_generator(320, 240);

    let roi = Roi {
        id: 1,
        bbox: rb(40.0, 40.0, 80.0, 60.0),
    };

    submit_and_drain(&tracker, make_frame("cam-a", &gen_a, vec![roi.clone()]));

    tracker.reset_stream("cam-a").expect("reset");

    // Drain whatever the tracker emits while the service batch is in
    // flight. With the new mechanism the service batch is filtered out
    // *inside* `convert_output`; nothing should surface during a
    // bounded window.
    let deadline = Instant::now() + Duration::from_millis(1500);
    while Instant::now() < deadline {
        match tracker.try_recv().expect("try_recv") {
            None => std::thread::sleep(Duration::from_millis(10)),
            Some(NvTrackerOutput::Event(_)) => {}
            Some(NvTrackerOutput::Tracking(t)) => {
                panic!("service batch must not surface as Tracking: {t:?}")
            }
            Some(NvTrackerOutput::Eos { source_id }) => {
                panic!("unexpected Eos for {source_id} during service drain")
            }
            Some(NvTrackerOutput::Error(e)) => panic!("tracker error: {e}"),
        }
    }

    let _ = tracker.shutdown();
    drop(gen_a);
}

#[test]
#[serial]
fn test_stale_evictor_releases_idle_source() {
    common::init();
    // Stale threshold short enough to fire well within the test window.
    let Some(tracker) = try_create_tracker(Some(Duration::from_millis(200))) else {
        return;
    };

    let gen_a = make_pinned_generator(320, 240);

    let roi = Roi {
        id: 1,
        bbox: rb(40.0, 40.0, 80.0, 60.0),
    };

    submit_and_drain(&tracker, make_frame("cam-a", &gen_a, vec![roi.clone()]));

    assert!(
        matches!(
            gen_a.try_acquire(None),
            Err(NvBufSurfaceError::PoolExhausted)
        ),
        "prev-frame must be pinned right after submit"
    );

    // The evictor only schedules `reset_stream`; the actual pin
    // release still requires one regular batch. Push from a
    // throw-away source so we don't keep cam-a's pool alive.
    let gen_aux = make_pinned_generator(320, 240);

    // Wait long enough for: stale tick + reset_stream + regular batch.
    // The evictor polls every `min(after/5, 1s)` and the tracker
    // already saw cam-a, so the next aux batch will drop the pin.
    std::thread::sleep(Duration::from_millis(500));
    submit_and_drain(&tracker, make_frame("cam-z", &gen_aux, vec![roi.clone()]));

    assert!(
        wait_pool_free(&gen_a, Duration::from_secs(3)),
        "stale evictor + one regular batch should release cam-a's pin"
    );

    let _ = tracker.shutdown();
    drop((gen_a, gen_aux));
}

#[test]
#[serial]
fn test_drop_resets_all_active_sources_without_hang() {
    common::init();
    let Some(tracker) = try_create_tracker(None) else {
        return;
    };

    let gen_a = make_pinned_generator(320, 240);
    let gen_b = make_pinned_generator(320, 240);
    let gen_c = make_pinned_generator(320, 240);

    let roi = Roi {
        id: 1,
        bbox: rb(40.0, 40.0, 80.0, 60.0),
    };

    submit_and_drain(&tracker, make_frame("cam-a", &gen_a, vec![roi.clone()]));
    submit_and_drain(&tracker, make_frame("cam-b", &gen_b, vec![roi.clone()]));
    submit_and_drain(&tracker, make_frame("cam-c", &gen_c, vec![roi.clone()]));

    let start = Instant::now();
    drop(tracker);
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_secs(15),
        "drop with 3 active sources must complete promptly under \
         leak_on_finalize=false; took {elapsed:?}"
    );

    drop((gen_a, gen_b, gen_c));
}
