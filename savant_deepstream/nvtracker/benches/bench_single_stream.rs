//! Criterion benchmarks for single-stream tracking (NvSORT & NvDCF).
//!
//! Measures `track_sync` latency for a single camera with 10 / 25 / 50 / 100
//! still (non-moving) objects per frame.  Objects are arranged in a
//! deterministic grid on a 1920x1080 surface.
//!
//! Run with:
//! ```sh
//! cargo bench -p savant-deepstream-nvtracker --bench bench_single_stream
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use deepstream_buffers::{
    cuda_init, BufferGenerator, NvBufSurfaceMemType, SavantIdMetaKind, VideoFormat,
};
use deepstream_nvtracker::{
    default_ll_lib_path, NvTracker, NvTrackerConfig, NvTrackerOutput, Result as NvResult, Roi,
    TrackedFrame, TrackerOutput, TrackingIdResetMode,
};
use savant_core::primitives::RBBox;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Once;
use std::time::Instant;

static INIT: Once = Once::new();

fn ensure_init() {
    INIT.call_once(|| {
        let _ = env_logger::try_init();
        gstreamer::init().expect("GStreamer init failed");
        cuda_init(0).expect("CUDA init failed — GPU required");
    });
}

const W: u32 = 1920;
const H: u32 = 1080;
const BOX_SIZE: f32 = 40.0;
const WARMUP_FRAMES: usize = 10;
const VALIDATION_TOLERANCE_PX: f32 = 5.0;

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets")
}

/// Place `n` non-overlapping 40x40 bounding boxes in a grid within `w x h`.
fn generate_rois(n: usize, w: u32, h: u32) -> Vec<Roi> {
    let cols = ((w as f32) / (BOX_SIZE * 2.0)).floor() as usize;
    assert!(
        cols > 0,
        "frame too narrow for even one column of {BOX_SIZE}-wide boxes"
    );
    let mut rois = Vec::with_capacity(n);
    for i in 0..n {
        let col = i % cols;
        let row = i / cols;
        let left = BOX_SIZE * 0.5 + col as f32 * BOX_SIZE * 2.0;
        let top = BOX_SIZE * 0.5 + row as f32 * BOX_SIZE * 2.0;
        assert!(
            left + BOX_SIZE <= w as f32 && top + BOX_SIZE <= h as f32,
            "ROI {i} at ({left},{top}) exceeds {w}x{h} frame"
        );
        rois.push(Roi {
            id: i as i64,
            bbox: RBBox::ltwh(left, top, BOX_SIZE, BOX_SIZE).expect("RBBox::ltwh"),
        });
    }
    rois
}

fn make_frame(gen: &BufferGenerator, rois: &[Roi]) -> TrackedFrame {
    let buf = gen.acquire(None).expect("acquire buffer");
    let mut map = HashMap::new();
    if !rois.is_empty() {
        map.insert(0i32, rois.to_vec());
    }
    TrackedFrame {
        source: "bench-cam".to_string(),
        buffer: buf,
        rois: map,
    }
}

fn frame_ids() -> Vec<SavantIdMetaKind> {
    vec![SavantIdMetaKind::Frame(1)]
}

fn create_tracker(config_path: &std::path::Path) -> NvTracker {
    let lib = default_ll_lib_path();
    assert!(
        std::path::Path::new(&lib).is_file(),
        "missing tracker lib: {lib}"
    );
    assert!(
        config_path.is_file(),
        "missing tracker config: {}",
        config_path.display()
    );
    let mut c = NvTrackerConfig::new(lib, config_path.to_string_lossy());
    c.tracker_width = 640;
    c.tracker_height = 384;
    c.max_batch_size = 1;
    c.tracking_id_reset_mode = TrackingIdResetMode::None;
    NvTracker::new(c).expect("NvTracker::new failed")
}

fn bench_track_sync(
    tracker: &NvTracker,
    frames: &[TrackedFrame],
    ids: Vec<SavantIdMetaKind>,
) -> NvResult<TrackerOutput> {
    tracker.submit(frames, ids)?;
    loop {
        match tracker.recv()? {
            NvTrackerOutput::Tracking(t) => return Ok(t),
            NvTrackerOutput::Event(_) => continue,
            NvTrackerOutput::Eos { source_id } => {
                return Err(deepstream_nvtracker::NvTrackerError::PipelineError(
                    format!("unexpected EOS: {source_id}"),
                ));
            }
            NvTrackerOutput::Error(e) => return Err(e),
        }
    }
}

/// Push `n` warm-up frames so the tracker passes its probation threshold.
fn warm_up(tracker: &NvTracker, gen: &BufferGenerator, rois: &[Roi], n: usize) {
    for _ in 0..n {
        let _ = bench_track_sync(tracker, &[make_frame(gen, rois)], frame_ids())
            .expect("warm-up track failed");
    }
}

/// Validate that tracker output matches the expected ROIs.
///
/// Checks:
/// 1. Track count == ROI count.
/// 2. Each tracked bbox center is within `VALIDATION_TOLERANCE_PX` of the
///    nearest input ROI center.
fn validate_output(output: &TrackerOutput, expected_rois: &[Roi]) {
    assert_eq!(
        output.current_tracks.len(),
        expected_rois.len(),
        "track count ({}) != expected ROI count ({})",
        output.current_tracks.len(),
        expected_rois.len()
    );

    let roi_centers: Vec<(f32, f32)> = expected_rois
        .iter()
        .map(|r| {
            let ltwh = r.bbox.get_wrapping_bbox().as_ltwh().expect("as_ltwh");
            (ltwh.0 + ltwh.2 / 2.0, ltwh.1 + ltwh.3 / 2.0)
        })
        .collect();

    for track in &output.current_tracks {
        let tcx = track.bbox_left + track.bbox_width / 2.0;
        let tcy = track.bbox_top + track.bbox_height / 2.0;

        let min_dist = roi_centers
            .iter()
            .map(|&(rx, ry)| ((tcx - rx).powi(2) + (tcy - ry).powi(2)).sqrt())
            .fold(f32::INFINITY, f32::min);

        assert!(
            min_dist <= VALIDATION_TOLERANCE_PX,
            "tracked object (id={}) center ({tcx:.1},{tcy:.1}) has no matching ROI \
             within {VALIDATION_TOLERANCE_PX} px (nearest={min_dist:.1} px)",
            track.object_id,
        );
    }
}

/// Run a single-stream benchmark group for the given tracker backend.
fn run_single_stream_bench(c: &mut Criterion, group_name: &str, config_path: &std::path::Path) {
    ensure_init();

    let gen = BufferGenerator::new(
        VideoFormat::RGBA,
        W,
        H,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .expect("BufferGenerator");

    let mut group = c.benchmark_group(group_name);
    group.sample_size(20);

    for &n_objects in &[10usize, 25, 50, 100] {
        let rois = generate_rois(n_objects, W, H);

        let tracker = create_tracker(config_path);

        warm_up(&tracker, &gen, &rois, WARMUP_FRAMES);

        let val_output = bench_track_sync(&tracker, &[make_frame(&gen, &rois)], frame_ids())
            .expect("validation track failed");
        validate_output(&val_output, &rois);

        group.throughput(criterion::Throughput::Elements(n_objects as u64));

        group.bench_with_input(
            BenchmarkId::new("objects", n_objects),
            &n_objects,
            |b, &_n| {
                b.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        let frame = make_frame(&gen, &rois);
                        let _output = bench_track_sync(&tracker, &[frame], frame_ids())
                            .expect("track failed during benchmark");
                    }
                    start.elapsed()
                });
            },
        );

        let _ = tracker.shutdown();
    }

    group.finish();
}

fn bench_nvsort(c: &mut Criterion) {
    run_single_stream_bench(
        c,
        "nvsort_single_stream",
        &assets_dir().join("config_tracker_NvSORT.yml"),
    );
}

fn bench_nvdcf(c: &mut Criterion) {
    run_single_stream_bench(
        c,
        "nvdcf_max_perf_single_stream",
        &assets_dir().join("config_tracker_NvDCF_max_perf.yml"),
    );
}

criterion_group!(benches, bench_nvsort, bench_nvdcf);
criterion_main!(benches);
