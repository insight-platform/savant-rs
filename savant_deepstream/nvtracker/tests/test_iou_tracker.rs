//! IOU tracker integration tests (GPU + DeepStream required).
//!
//! Run with: `cargo test -p nvtracker --test test_iou_tracker -- --test-threads=1`

mod common;

use deepstream_buffers::{
    BufferGenerator, NvBufSurfaceMemType, SavantIdMetaKind, SharedBuffer, VideoFormat,
};
use nvtracker::{
    default_ll_lib_path, NvTracker, NvTrackerConfig, Roi, TrackState, TrackedFrame,
    TrackingIdResetMode,
};
use savant_core::primitives::RBBox;
use serial_test::serial;
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::PathBuf;

thread_local! {
    static GENERATORS: RefCell<HashMap<(u32, u32), BufferGenerator>> = RefCell::new(HashMap::new());
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

/// Acquire a single-surface NVMM buffer, reusing a cached `BufferGenerator`
/// for the given resolution.
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

/// Convenience: single-surface `TrackedFrame` with class_id 0.
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

#[test]
#[serial]
fn test_single_source_id_stability() {
    common::init();
    let Some(tracker) = try_create_tracker(TrackingIdResetMode::None) else {
        return;
    };

    let rois_frame = |dx: f32| -> Vec<Roi> {
        vec![
            Roi {
                id: 1,
                bbox: rb(40.0 + dx, 40.0, 80.0, 60.0),
            },
            Roi {
                id: 2,
                bbox: rb(180.0 + dx, 100.0, 70.0, 70.0),
            },
        ]
    };

    let out0 = common::track_sync(
        &tracker,
        &[make_frame("cam-1", rois_frame(0.0), 320, 240)],
        frame_ids(1),
    )
    .expect("track 0");
    assert_eq!(out0.current_tracks.len(), 2, "two detections");
    let id_a = out0.current_tracks[0].object_id;
    let id_b = out0.current_tracks[1].object_id;

    let out1 = common::track_sync(
        &tracker,
        &[make_frame("cam-1", rois_frame(2.0), 320, 240)],
        frame_ids(1),
    )
    .expect("track 1");
    assert_eq!(out1.current_tracks.len(), 2);
    let ids1: Vec<u64> = out1.current_tracks.iter().map(|t| t.object_id).collect();
    assert!(
        ids1.contains(&id_a) && ids1.contains(&id_b),
        "IDs should persist: {:?} vs {id_a},{id_b}",
        ids1
    );

    let out2 = common::track_sync(
        &tracker,
        &[make_frame(
            "cam-1",
            vec![Roi {
                id: 1,
                bbox: rb(44.0, 42.0, 80.0, 60.0),
            }],
            320,
            240,
        )],
        frame_ids(1),
    )
    .expect("track 2");
    assert_eq!(out2.current_tracks.len(), 1);
    assert_eq!(
        out2.current_tracks[0].object_id, id_a,
        "remaining detection should keep the first track id"
    );

    let _ = tracker.shutdown();
}

#[test]
#[serial]
fn test_multi_source_isolation() {
    common::init();
    let Some(tracker) = try_create_tracker(TrackingIdResetMode::None) else {
        return;
    };

    let roi = Roi {
        id: 1,
        bbox: rb(50.0, 50.0, 60.0, 60.0),
    };

    let out = common::track_sync(
        &tracker,
        &[
            make_frame("cam-a", vec![roi.clone()], 320, 240),
            make_frame("cam-b", vec![roi.clone()], 320, 240),
        ],
        frame_ids(2),
    )
    .expect("multi track");
    assert_eq!(out.current_tracks.len(), 2);
    let mut by_src: HashMap<String, u64> = HashMap::new();
    for t in &out.current_tracks {
        by_src.insert(t.source_id.clone(), t.object_id);
    }
    let id_a = *by_src.get("cam-a").expect("cam-a");
    let id_b = *by_src.get("cam-b").expect("cam-b");
    assert_ne!(id_a, id_b, "independent stream IDs");

    let out2 = common::track_sync(
        &tracker,
        &[
            make_frame(
                "cam-a",
                vec![Roi {
                    id: 1,
                    bbox: rb(52.0, 52.0, 60.0, 60.0),
                }],
                320,
                240,
            ),
            make_frame(
                "cam-b",
                vec![Roi {
                    id: 1,
                    bbox: rb(52.0, 52.0, 60.0, 60.0),
                }],
                320,
                240,
            ),
        ],
        frame_ids(2),
    )
    .expect("frame 2");
    let mut by2: HashMap<String, u64> = HashMap::new();
    for t in &out2.current_tracks {
        by2.insert(t.source_id.clone(), t.object_id);
    }
    assert_eq!(by2.get("cam-a"), Some(&id_a));
    assert_eq!(by2.get("cam-b"), Some(&id_b));

    let _ = tracker.shutdown();
}

#[test]
#[serial]
fn test_multi_source_nonuniform() {
    common::init();
    let Some(tracker) = try_create_tracker(TrackingIdResetMode::None) else {
        return;
    };

    let out = common::track_sync(
        &tracker,
        &[
            make_frame(
                "cam-a",
                vec![Roi {
                    id: 1,
                    bbox: rb(50.0, 50.0, 60.0, 60.0),
                }],
                320,
                240,
            ),
            make_frame(
                "cam-b",
                vec![Roi {
                    id: 1,
                    bbox: rb(100.0, 100.0, 80.0, 80.0),
                }],
                640,
                480,
            ),
        ],
        frame_ids(2),
    )
    .expect("nonuniform");
    assert_eq!(out.current_tracks.len(), 2);

    let _ = tracker.shutdown();
}

#[test]
#[serial]
fn test_same_source_multi_frame() {
    common::init();
    let Some(tracker) = try_create_tracker(TrackingIdResetMode::None) else {
        return;
    };

    let out = common::track_sync(
        &tracker,
        &[
            make_frame(
                "cam-a",
                vec![Roi {
                    id: 1,
                    bbox: rb(50.0, 50.0, 60.0, 60.0),
                }],
                320,
                240,
            ),
            make_frame(
                "cam-a",
                vec![Roi {
                    id: 1,
                    bbox: rb(55.0, 52.0, 60.0, 60.0),
                }],
                320,
                240,
            ),
        ],
        frame_ids(2),
    )
    .expect("two frames same source");
    assert!(
        !out.current_tracks.is_empty(),
        "expected at least one track from temporal batch"
    );
    let id_first = out.current_tracks[0].object_id;

    let out2 = common::track_sync(
        &tracker,
        &[
            make_frame(
                "cam-a",
                vec![Roi {
                    id: 1,
                    bbox: rb(56.0, 53.0, 60.0, 60.0),
                }],
                320,
                240,
            ),
            make_frame(
                "cam-a",
                vec![Roi {
                    id: 1,
                    bbox: rb(58.0, 54.0, 60.0, 60.0),
                }],
                320,
                240,
            ),
        ],
        frame_ids(2),
    )
    .expect("second temporal batch");
    let ids: Vec<u64> = out2
        .current_tracks
        .iter()
        .filter(|t| t.source_id == "cam-a")
        .map(|t| t.object_id)
        .collect();
    assert!(
        ids.contains(&id_first),
        "ID should remain stable across batches: {:?} contains {}",
        ids,
        id_first
    );

    let _ = tracker.shutdown();
}

#[test]
#[serial]
fn test_mixed_batch() {
    common::init();
    let Some(tracker) = try_create_tracker(TrackingIdResetMode::None) else {
        return;
    };

    let out = common::track_sync(
        &tracker,
        &[
            make_frame(
                "cam-a",
                vec![Roi {
                    id: 1,
                    bbox: rb(40.0, 40.0, 50.0, 50.0),
                }],
                320,
                240,
            ),
            make_frame(
                "cam-a",
                vec![Roi {
                    id: 1,
                    bbox: rb(45.0, 45.0, 50.0, 50.0),
                }],
                320,
                240,
            ),
            make_frame(
                "cam-b",
                vec![Roi {
                    id: 1,
                    bbox: rb(200.0, 200.0, 100.0, 100.0),
                }],
                640,
                480,
            ),
            make_frame(
                "cam-b",
                vec![Roi {
                    id: 1,
                    bbox: rb(210.0, 210.0, 100.0, 100.0),
                }],
                640,
                480,
            ),
        ],
        frame_ids(4),
    )
    .expect("mixed batch");
    assert_eq!(out.current_tracks.len(), 4, "one track per slot");

    let mut by_src: HashMap<String, Vec<u64>> = HashMap::new();
    for t in &out.current_tracks {
        by_src
            .entry(t.source_id.clone())
            .or_default()
            .push(t.object_id);
    }
    let a_ids = by_src.get("cam-a").expect("cam-a tracks");
    let b_ids = by_src.get("cam-b").expect("cam-b tracks");
    assert_eq!(a_ids.len(), 2);
    assert_eq!(b_ids.len(), 2);
    assert_ne!(a_ids[0], b_ids[0]);

    let _ = tracker.shutdown();
}

#[test]
#[serial]
fn test_reset_stream_reassigns_ids() {
    common::init();
    let Some(tracker) = try_create_tracker(TrackingIdResetMode::OnStreamReset) else {
        return;
    };

    let rois = vec![Roi {
        id: 1,
        bbox: rb(60.0, 60.0, 90.0, 90.0),
    }];

    let out0 = common::track_sync(
        &tracker,
        &[make_frame("cam-1", rois.clone(), 320, 240)],
        frame_ids(1),
    )
    .expect("t0");
    let id0 = out0.current_tracks[0].object_id;

    let out1 = common::track_sync(
        &tracker,
        &[make_frame("cam-1", rois.clone(), 320, 240)],
        frame_ids(1),
    )
    .expect("t1");
    assert_eq!(out1.current_tracks[0].object_id, id0);

    tracker.reset_stream("cam-1").expect("reset");

    let out2 = common::track_sync(
        &tracker,
        &[make_frame("cam-1", rois.clone(), 320, 240)],
        frame_ids(1),
    )
    .expect("t2");
    let id2 = out2.current_tracks[0].object_id;
    assert_ne!(
        id2, id0,
        "after stream reset, tracker should assign a new ID (mode=ON_STREAM_RESET)"
    );

    let _ = tracker.shutdown();
}

#[test]
#[serial]
fn test_reset_stream_only_affects_target() {
    common::init();
    let Some(tracker) = try_create_tracker(TrackingIdResetMode::OnStreamReset) else {
        return;
    };

    let roi = Roi {
        id: 1,
        bbox: rb(50.0, 50.0, 60.0, 60.0),
    };

    let out0 = common::track_sync(
        &tracker,
        &[
            make_frame("cam-a", vec![roi.clone()], 320, 240),
            make_frame("cam-b", vec![roi.clone()], 320, 240),
        ],
        frame_ids(2),
    )
    .expect("t0");
    let mut id_a: Option<u64> = None;
    let mut id_b: Option<u64> = None;
    for t in &out0.current_tracks {
        if t.source_id == "cam-a" {
            id_a = Some(t.object_id);
        }
        if t.source_id == "cam-b" {
            id_b = Some(t.object_id);
        }
    }
    let id_a = id_a.expect("cam-a track present");
    let id_b = id_b.expect("cam-b track present");

    tracker.reset_stream("cam-a").expect("reset cam-a");

    let out1 = common::track_sync(
        &tracker,
        &[
            make_frame("cam-a", vec![roi.clone()], 320, 240),
            make_frame("cam-b", vec![roi.clone()], 320, 240),
        ],
        frame_ids(2),
    )
    .expect("t1");
    let mut new_a: Option<u64> = None;
    let mut new_b: Option<u64> = None;
    for t in &out1.current_tracks {
        if t.source_id == "cam-a" {
            new_a = Some(t.object_id);
        }
        if t.source_id == "cam-b" {
            new_b = Some(t.object_id);
        }
    }
    let new_a = new_a.expect("cam-a track present after reset");
    let new_b = new_b.expect("cam-b track present after reset");
    assert_ne!(new_a, id_a, "cam-a should get new id");
    assert_eq!(new_b, id_b, "cam-b should keep id");

    let _ = tracker.shutdown();
}

#[test]
#[serial]
fn test_shadow_tracks_on_disappearance() {
    common::init();
    let Some(tracker) = try_create_tracker(TrackingIdResetMode::None) else {
        return;
    };

    let one = vec![Roi {
        id: 1,
        bbox: rb(80.0, 80.0, 70.0, 70.0),
    }];

    let out0 = common::track_sync(
        &tracker,
        &[make_frame("cam-1", one.clone(), 320, 240)],
        frame_ids(1),
    )
    .expect("t0");
    let id0 = out0.current_tracks[0].object_id;

    let _ = common::track_sync(
        &tracker,
        &[make_frame("cam-1", one.clone(), 320, 240)],
        frame_ids(1),
    )
    .expect("t1");

    let out2 = common::track_sync(
        &tracker,
        &[make_frame("cam-1", vec![], 320, 240)],
        frame_ids(1),
    )
    .expect("empty rois");
    let still_current: Vec<u64> = out2.current_tracks.iter().map(|t| t.object_id).collect();
    assert!(
        !still_current.contains(&id0),
        "object should not stay in current_tracks: {:?}",
        still_current
    );

    // Shadow / terminated misc lists require `enable-past-frame` (NvDCF /
    // DeepSORT modes).  The IOU tracker in DS 7.1 does NOT populate them.
    let in_shadow = out2.shadow_tracks.iter().any(|s| s.object_id == id0);
    let in_term = out2.terminated_tracks.iter().any(|s| s.object_id == id0);
    if !out2.shadow_tracks.is_empty() || !out2.terminated_tracks.is_empty() {
        assert!(
            in_shadow || in_term,
            "expected id {id0} in shadow or terminated misc lists (shadow={}, term={})",
            out2.shadow_tracks.len(),
            out2.terminated_tracks.len()
        );
    }

    let _ = tracker.shutdown();
}

#[test]
#[serial]
fn test_source_id_roundtrip() {
    common::init();
    let Some(tracker) = try_create_tracker(TrackingIdResetMode::None) else {
        return;
    };

    let sid = "my-camera-stream-42";
    let out = common::track_sync(
        &tracker,
        &[make_frame(
            sid,
            vec![Roi {
                id: 1,
                bbox: rb(40.0, 40.0, 50.0, 50.0),
            }],
            320,
            240,
        )],
        frame_ids(1),
    )
    .expect("track");
    assert_eq!(out.current_tracks[0].source_id, sid);
    let _ = tracker.shutdown();

    let Some(tracker2) = try_create_tracker(TrackingIdResetMode::None) else {
        return;
    };
    let out2 = common::track_sync(
        &tracker2,
        &[
            make_frame(
                "stream-alpha",
                vec![Roi {
                    id: 1,
                    bbox: rb(30.0, 30.0, 40.0, 40.0),
                }],
                320,
                240,
            ),
            make_frame(
                "stream-beta",
                vec![Roi {
                    id: 1,
                    bbox: rb(100.0, 100.0, 40.0, 40.0),
                }],
                320,
                240,
            ),
        ],
        frame_ids(2),
    )
    .expect("multi");
    let mut sids: Vec<String> = out2
        .current_tracks
        .iter()
        .map(|t| t.source_id.clone())
        .collect();
    sids.sort();
    assert_eq!(
        sids,
        vec!["stream-alpha".to_string(), "stream-beta".to_string()]
    );

    let _ = tracker.shutdown();
    let _ = tracker2.shutdown();
}

#[test]
#[serial]
fn test_class_id_tracking() {
    common::init();
    let Some(tracker) = try_create_tracker(TrackingIdResetMode::None) else {
        return;
    };

    let roi_a = Roi {
        id: 1,
        bbox: rb(40.0, 40.0, 80.0, 60.0),
    };
    let roi_b = Roi {
        id: 2,
        bbox: rb(180.0, 100.0, 70.0, 70.0),
    };

    let frame = TrackedFrame {
        source: "cam-1".to_string(),
        buffer: make_buffer(320, 240),
        rois: HashMap::from([(0, vec![roi_a]), (1, vec![roi_b])]),
    };

    let out = common::track_sync(&tracker, &[frame], frame_ids(1)).expect("class_id track");
    assert_eq!(out.current_tracks.len(), 2);

    let class_ids: std::collections::HashSet<i32> =
        out.current_tracks.iter().map(|t| t.class_id).collect();
    assert_eq!(
        class_ids,
        std::collections::HashSet::from([0, 1]),
        "both class_ids should be preserved"
    );

    let _ = tracker.shutdown();
}

#[test]
#[serial]
#[ignore = "diagnostic: compare nvstreammux+nvtracker vs manual path; run manually"]
fn test_mux_reference_equivalence() {}

#[test]
fn test_misc_track_state_mapping() {
    let _ = TrackState::Active;
}
