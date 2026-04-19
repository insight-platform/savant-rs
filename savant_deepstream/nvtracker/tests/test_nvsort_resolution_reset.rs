//! NvSORT integration tests for automatic stream state resets.
//!
//! Run with: `cargo test -p savant-deepstream-nvtracker --test test_nvsort_resolution_reset -- --test-threads=1`

mod common;

use deepstream_buffers::{
    BufferGenerator, NvBufSurfaceMemType, SavantIdMetaKind, SharedBuffer, VideoFormat,
};
use deepstream_nvtracker::{
    default_ll_lib_path, NvTracker, NvTrackerConfig, Roi, TrackedFrame, TrackingIdResetMode,
};
use savant_core::primitives::RBBox;
use serial_test::serial;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

thread_local! {
    static GENERATORS: RefCell<HashMap<(u32, u32), BufferGenerator>> = RefCell::new(HashMap::new());
}

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets")
}

fn nvsort_config_path() -> PathBuf {
    assets_dir().join("config_tracker_NvSORT.yml")
}

fn rb(left: f32, top: f32, w: f32, h: f32) -> RBBox {
    RBBox::ltwh(left, top, w, h).expect("RBBox::ltwh")
}

fn make_buffer(w: u32, h: u32) -> SharedBuffer {
    GENERATORS.with(|gens| {
        let mut gens = gens.borrow_mut();
        let gen = gens.entry((w, h)).or_insert_with(|| {
            BufferGenerator::new(
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

fn make_frame(source: &str, rois: Vec<Roi>, w: u32, h: u32, pts_ns: Option<u64>) -> TrackedFrame {
    let buf = make_buffer(w, h);
    if let Some(pts) = pts_ns {
        buf.set_pts_ns(pts);
    }
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

fn read_tracker_yaml_u32(path: &Path, key: &str) -> Option<u32> {
    let text = std::fs::read_to_string(path).ok()?;
    for line in text.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with(key) {
            continue;
        }
        let val = trimmed.split_once(':')?.1.trim();
        let no_comment = val.split('#').next()?.trim();
        if no_comment.is_empty() {
            continue;
        }
        if let Ok(parsed) = no_comment.parse::<u32>() {
            return Some(parsed);
        }
    }
    None
}

fn id_for_source(out: &deepstream_nvtracker::TrackerOutput, source: &str) -> Option<u64> {
    out.current_tracks
        .iter()
        .find(|t| t.source_id == source)
        .map(|t| t.object_id)
}

fn try_create_tracker(mode: TrackingIdResetMode) -> Option<NvTracker> {
    let lib = default_ll_lib_path();
    if !std::path::Path::new(&lib).is_file() {
        eprintln!("skip: missing {lib}");
        return None;
    }
    let cfg_path = nvsort_config_path();
    if !cfg_path.is_file() {
        eprintln!("skip: missing {}", cfg_path.display());
        return None;
    }

    let mut c = NvTrackerConfig::new(lib, cfg_path.to_string_lossy());
    c.tracking_id_reset_mode = mode;
    c.tracker_width = 640;
    c.tracker_height = 480;
    c.max_batch_size = 4;
    match NvTracker::new(c) {
        Ok(t) => Some(t),
        Err(e) => {
            eprintln!("skip: NvTracker::new failed: {e}");
            None
        }
    }
}

#[test]
#[serial]
fn test_resolution_change_resets_stream_state_nvsort() {
    common::init();
    let Some(tracker) = try_create_tracker(TrackingIdResetMode::OnStreamReset) else {
        return;
    };

    let roi_a = Roi {
        id: 1,
        bbox: rb(60.0, 60.0, 80.0, 80.0),
    };
    let roi_b = Roi {
        id: 1,
        bbox: rb(120.0, 90.0, 70.0, 70.0),
    };

    let mut id_a_before = None;
    let mut id_b_before = None;
    for i in 0..12u64 {
        let out = common::track_sync(
            &tracker,
            &[
                make_frame("cam-a", vec![roi_a.clone()], 320, 240, Some(100 + i)),
                make_frame("cam-b", vec![roi_b.clone()], 320, 240, Some(100 + i)),
            ],
            frame_ids(2),
        )
        .expect("warmup");
        id_a_before = id_for_source(&out, "cam-a");
        id_b_before = id_for_source(&out, "cam-b");
        if id_a_before.is_some() && id_b_before.is_some() {
            break;
        }
    }
    let id_a_before = id_a_before.expect("expected cam-a track after warmup");
    let id_b_before = id_b_before.expect("expected cam-b track after warmup");

    let mut cam_a_reset = false;
    let mut cam_b_preserved = false;
    for i in 0..16u64 {
        let out = common::track_sync(
            &tracker,
            &[
                make_frame("cam-a", vec![roi_a.clone()], 640, 480, Some(500 + i)),
                make_frame("cam-b", vec![roi_b.clone()], 320, 240, Some(500 + i)),
            ],
            frame_ids(2),
        )
        .expect("resolution-change");
        if let Some(id_a_after) = id_for_source(&out, "cam-a") {
            if id_a_after != id_a_before {
                cam_a_reset = true;
            }
        }
        if let Some(id_b_after) = id_for_source(&out, "cam-b") {
            if id_b_after == id_b_before {
                cam_b_preserved = true;
            }
        }
        if cam_a_reset && cam_b_preserved {
            break;
        }
    }

    assert!(
        cam_a_reset,
        "cam-a must be reset when its resolution changes"
    );
    assert!(
        cam_b_preserved,
        "cam-b must preserve tracker state when only cam-a changes resolution"
    );

    let _ = tracker.shutdown();
}

#[test]
#[serial]
fn test_nvsort_populates_shadow_and_terminated_lists() {
    common::init();
    let Some(tracker) = try_create_tracker(TrackingIdResetMode::None) else {
        return;
    };

    let cfg = nvsort_config_path();
    let probation_age = read_tracker_yaml_u32(&cfg, "probationAge").unwrap_or(5);
    let max_shadow_age = read_tracker_yaml_u32(&cfg, "maxShadowTrackingAge").unwrap_or(26);

    let roi = Roi {
        id: 1,
        bbox: rb(80.0, 80.0, 70.0, 70.0),
    };

    let mut id0 = None;
    for i in 0..=(probation_age + 8) {
        let out = common::track_sync(
            &tracker,
            &[make_frame(
                "cam-1",
                vec![roi.clone()],
                320,
                240,
                Some(i as u64 + 1),
            )],
            frame_ids(1),
        )
        .expect("warmup track");
        id0 = id_for_source(&out, "cam-1");
        if id0.is_some() {
            break;
        }
    }
    let id0 = id0.expect("expected a valid track id after warmup");

    let mut disappeared_from_current = false;
    let mut seen_shadow_for_id = false;
    let mut seen_any_shadow = false;
    let mut seen_terminated_for_id = false;
    let mut seen_any_terminated = false;
    let max_iters = max_shadow_age + probation_age + 10;
    for i in 0..=max_iters {
        let out = common::track_sync(
            &tracker,
            &[make_frame(
                "cam-1",
                vec![],
                320,
                240,
                Some(1_000 + i as u64),
            )],
            frame_ids(1),
        )
        .expect("empty-roi track");

        let in_current = out.current_tracks.iter().any(|t| t.object_id == id0);
        let in_shadow = out.shadow_tracks.iter().any(|t| t.object_id == id0);
        let in_term = out.terminated_tracks.iter().any(|t| t.object_id == id0);
        seen_any_terminated |= !out.terminated_tracks.is_empty();
        if !in_current {
            disappeared_from_current = true;
        }
        seen_any_shadow |= !out.shadow_tracks.is_empty();
        seen_shadow_for_id |= in_shadow;
        if in_term {
            seen_terminated_for_id = true;
        }
        if disappeared_from_current && seen_terminated_for_id {
            break;
        }
    }

    assert!(
        disappeared_from_current,
        "object should eventually leave current_tracks"
    );
    if seen_any_terminated {
        assert!(
            seen_terminated_for_id,
            "terminated_tracks were emitted, but expected tracked object id to appear there"
        );
    }
    if seen_any_shadow {
        assert!(
            seen_shadow_for_id,
            "shadow_tracks were emitted, but expected tracked object id to appear there"
        );
    }

    let _ = tracker.shutdown();
}

#[test]
#[serial]
fn test_pts_regression_resets_only_affected_stream_nvsort() {
    common::init();
    let Some(tracker) = try_create_tracker(TrackingIdResetMode::OnStreamReset) else {
        return;
    };

    let roi_a = Roi {
        id: 1,
        bbox: rb(50.0, 50.0, 70.0, 70.0),
    };
    let roi_b = Roi {
        id: 1,
        bbox: rb(150.0, 120.0, 70.0, 70.0),
    };

    let mut id_a_before = None;
    let mut id_b_before = None;
    for i in 0..12u64 {
        let out = common::track_sync(
            &tracker,
            &[
                make_frame("cam-a", vec![roi_a.clone()], 320, 240, Some(1_000 + i)),
                make_frame("cam-b", vec![roi_b.clone()], 320, 240, Some(1_000 + i)),
            ],
            frame_ids(2),
        )
        .expect("warmup");
        id_a_before = id_for_source(&out, "cam-a");
        id_b_before = id_for_source(&out, "cam-b");
        if id_a_before.is_some() && id_b_before.is_some() {
            break;
        }
    }
    let id_a_before = id_a_before.expect("expected cam-a id after warmup");
    let id_b_before = id_b_before.expect("expected cam-b id after warmup");

    let mut cam_a_reset = false;
    let mut cam_b_preserved = false;
    for i in 0..16u64 {
        let out = common::track_sync(
            &tracker,
            &[
                make_frame("cam-a", vec![roi_a.clone()], 320, 240, Some(900 + i)),
                make_frame("cam-b", vec![roi_b.clone()], 320, 240, Some(2_000 + i)),
            ],
            frame_ids(2),
        )
        .expect("pts-regression");
        if let Some(id_a_after) = id_for_source(&out, "cam-a") {
            if id_a_after != id_a_before {
                cam_a_reset = true;
            }
        }
        if let Some(id_b_after) = id_for_source(&out, "cam-b") {
            if id_b_after == id_b_before {
                cam_b_preserved = true;
            }
        }
        if cam_a_reset && cam_b_preserved {
            break;
        }
    }

    assert!(cam_a_reset, "cam-a must be reset when its PTS decreases");
    assert!(
        cam_b_preserved,
        "cam-b must keep state when only cam-a has PTS regression"
    );

    let _ = tracker.shutdown();
}
