//! Graceful shutdown tests (GPU + DeepStream required).

mod common;

use deepstream_buffers::{NvBufSurfaceMemType, SavantIdMetaKind, SharedBuffer, VideoFormat};
use deepstream_nvtracker::{
    default_ll_lib_path, NvTracker, NvTrackerConfig, NvTrackerError, NvTrackerOutput, Roi,
    TrackedFrame, TrackingIdResetMode,
};
use savant_core::primitives::RBBox;
use serial_test::serial;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

thread_local! {
    static GENERATORS: RefCell<HashMap<(u32, u32), deepstream_buffers::BufferGenerator>> = RefCell::new(HashMap::new());
}

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets")
}

fn iou_config_path() -> PathBuf {
    assets_dir().join("config_tracker_IOU.yml")
}

fn rb(left: f32, top: f32, w: f32, h: f32) -> RBBox {
    RBBox::ltwh(left, top, w, h).expect("RBBox::ltwh")
}

fn make_buffer(w: u32, h: u32) -> SharedBuffer {
    GENERATORS.with(|gens| {
        let mut gens = gens.borrow_mut();
        let gen = gens.entry((w, h)).or_insert_with(|| {
            deepstream_buffers::BufferGenerator::new(
                VideoFormat::RGBA,
                w,
                h,
                30,
                1,
                0,
                NvBufSurfaceMemType::Default,
            )
            .expect("BufferGenerator")
        });
        gen.acquire(None).expect("acquire buffer")
    })
}

fn make_frame(source: &str, rois: Vec<Roi>, w: u32, h: u32) -> TrackedFrame {
    let buf = make_buffer(w, h);
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

fn try_create_tracker(mode: TrackingIdResetMode) -> Option<NvTracker> {
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
    c.tracking_id_reset_mode = mode;
    c.tracker_width = 320;
    c.tracker_height = 240;
    c.max_batch_size = 4;
    match NvTracker::new(c) {
        Ok(t) => Some(t),
        Err(e) => {
            eprintln!("skip: NvTracker::new failed: {e}");
            None
        }
    }
}

fn count_tracking(outputs: &[NvTrackerOutput]) -> usize {
    outputs
        .iter()
        .filter(|o| matches!(o, NvTrackerOutput::Tracking(_)))
        .count()
}

#[test]
#[serial]
fn test_graceful_shutdown_drains_all() {
    common::init();
    let Some(tracker) = try_create_tracker(TrackingIdResetMode::None) else {
        return;
    };

    let rois = vec![Roi {
        id: 1,
        bbox: rb(40.0, 40.0, 80.0, 60.0),
    }];
    const N: usize = 3;
    for i in 0..N {
        tracker
            .submit(&[make_frame("cam-1", rois.clone(), 320, 240)], frame_ids(1))
            .unwrap_or_else(|e| panic!("submit {i}: {e}"));
    }

    let drained = tracker
        .graceful_shutdown(Duration::from_secs(30))
        .expect("graceful_shutdown");
    assert_eq!(
        count_tracking(&drained),
        N,
        "expected {N} tracking outputs, got {drained:?}"
    );
}

#[test]
#[serial]
fn test_graceful_shutdown_timeout() {
    common::init();
    let Some(tracker) = try_create_tracker(TrackingIdResetMode::None) else {
        return;
    };

    let rois = vec![Roi {
        id: 1,
        bbox: rb(40.0, 40.0, 80.0, 60.0),
    }];
    tracker
        .submit(&[make_frame("cam-1", rois, 320, 240)], frame_ids(1))
        .expect("submit");

    let drained = tracker
        .graceful_shutdown(Duration::ZERO)
        .expect("graceful_shutdown");
    assert!(
        count_tracking(&drained) < 1,
        "zero timeout should not fully drain: {drained:?}"
    );
}

#[test]
#[serial]
fn test_graceful_shutdown_submit_after_returns_shutting_down() {
    common::init();
    let Some(tracker) = try_create_tracker(TrackingIdResetMode::None) else {
        return;
    };

    let _ = tracker
        .graceful_shutdown(Duration::from_secs(30))
        .expect("graceful_shutdown");

    let rois = vec![Roi {
        id: 1,
        bbox: rb(40.0, 40.0, 80.0, 60.0),
    }];
    let err = tracker
        .submit(&[make_frame("cam-1", rois, 320, 240)], frame_ids(1))
        .expect_err("submit after shutdown");
    assert!(matches!(err, NvTrackerError::ShuttingDown));

    let structure = gstreamer::Structure::builder("test").build();
    let ev = gstreamer::event::CustomDownstream::new(structure);
    let err = tracker
        .send_event(ev)
        .expect_err("send_event after shutdown");
    assert!(matches!(err, NvTrackerError::ShuttingDown));

    let err = tracker.send_eos("s1").expect_err("send_eos after shutdown");
    assert!(matches!(err, NvTrackerError::ShuttingDown));

    let err = tracker
        .reset_stream("cam-1")
        .expect_err("reset_stream after shutdown");
    assert!(matches!(err, NvTrackerError::ShuttingDown));
}
