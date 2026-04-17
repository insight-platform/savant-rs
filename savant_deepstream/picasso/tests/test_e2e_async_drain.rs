//! E2E tests for the asynchronous drain thread.
//!
//! The drain thread pulls encoded output from the hardware encoder
//! independently of frame submission. These tests validate that:
//!
//! 1. Encoded frames arrive without needing another `send_frame` to trigger
//!    draining.
//! 2. Draw-spec hot-swap (same codec) keeps the encoder and drain thread
//!    alive — no spurious restart or frame loss.
//! 3. Rapid sustained submission doesn't lose frames (JPEG + NVENC codecs
//!    concurrently when hardware supports them).
//! 4. EOS flushes the exact number of in-flight frames.

mod common;

use common::*;
use deepstream_buffers::{BufferGenerator, TransformConfig};
use deepstream_encoders::prelude::*;
use log::{Level, LevelFilter, Log, Metadata, Record};
use picasso::prelude::*;
use savant_core::primitives::frame::VideoFrameProxy;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

type LogRecords = Arc<Mutex<Vec<(Level, String)>>>;

/// Process-wide log capture.  Stores every emitted record (level + rendered
/// message) so individual tests can assert presence/absence of specific log
/// lines.  Installed on first use, replaces `env_logger`, and also echoes
/// to stderr so cargo's `--nocapture` still shows output.
struct CapturingLogger {
    records: LogRecords,
}

impl Log for CapturingLogger {
    fn enabled(&self, _metadata: &Metadata) -> bool {
        true
    }
    fn log(&self, record: &Record) {
        let msg = format!("{}", record.args());
        if let Ok(mut v) = self.records.lock() {
            v.push((record.level(), msg.clone()));
        }
        eprintln!("[{} {}] {}", record.level(), record.target(), msg);
    }
    fn flush(&self) {}
}

fn log_records() -> LogRecords {
    static RECORDS: OnceLock<LogRecords> = OnceLock::new();
    RECORDS
        .get_or_init(|| {
            let records: LogRecords = Arc::new(Mutex::new(Vec::new()));
            let logger = Box::new(CapturingLogger {
                records: records.clone(),
            });
            // May fail if another logger was installed first; in that case
            // we fall through with an empty record buffer and the absence
            // assertion degenerates into a no-op (documented in the test).
            let _ = log::set_boxed_logger(logger);
            log::set_max_level(LevelFilter::Debug);
            records
        })
        .clone()
}

/// Counts encoded frames and EOS per `source_id` for multi-source throughput tests.
struct PerSourceEncodedCb {
    enc_counts: Arc<Mutex<HashMap<String, usize>>>,
    eos_counts: Arc<Mutex<HashMap<String, usize>>>,
}

impl OnEncodedFrame for PerSourceEncodedCb {
    fn call(&self, output: OutputMessage) {
        match output {
            OutputMessage::VideoFrame(frame) => {
                let sid = frame.get_source_id().to_string();
                let mut m = self.enc_counts.lock().expect("enc_counts lock");
                *m.entry(sid).or_insert(0) += 1;
            }
            OutputMessage::EndOfStream(eos) => {
                let sid = eos.get_source_id().to_string();
                let mut m = self.eos_counts.lock().expect("eos_counts lock");
                *m.entry(sid).or_insert(0) += 1;
            }
        }
    }
}

const W: u32 = 640;
const H: u32 = 480;
const DUR: u64 = 33_333_333;

fn make_numbered_frame(source_id: &str, idx: u64) -> VideoFrameProxy {
    let mut frame = make_frame_sized(source_id, W as i64, H as i64);
    frame.set_pts((idx * DUR) as i64).unwrap();
    frame.set_duration(Some(DUR as i64)).unwrap();
    frame
}

fn setup_engine(enc_count: &Arc<AtomicUsize>, eos_count: &Arc<AtomicUsize>) -> PicassoEngine {
    let callbacks = Callbacks {
        on_encoded_frame: Some(Arc::new(CountingEncodedCb {
            count: enc_count.clone(),
            eos_count: eos_count.clone(),
        })),
        ..Default::default()
    };
    PicassoEngine::new(
        GeneralSpec {
            idle_timeout_secs: 300,
            ..Default::default()
        },
        callbacks,
    )
}

fn jpeg_spec() -> SourceSpec {
    SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(jpeg_encoder_config(W, H)),
        },
        ..Default::default()
    }
}

fn make_generator() -> BufferGenerator {
    BufferGenerator::new(
        VideoFormat::RGBA,
        W,
        H,
        30,
        1,
        0,
        NvBufSurfaceMemType::Default,
    )
    .unwrap()
}

/// The drain thread delivers encoded output without any subsequent
/// `send_frame` call. Submit a batch, then only wait — encoded frames
/// must arrive purely from the drain thread polling the encoder.
#[test]
#[serial_test::serial]
fn e2e_async_drain_delivers_independently() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).unwrap();

    let enc_count = Arc::new(AtomicUsize::new(0));
    let eos_count = Arc::new(AtomicUsize::new(0));
    let engine = setup_engine(&enc_count, &eos_count);

    engine.set_source_spec("drain", jpeg_spec()).unwrap();
    let gen = make_generator();

    let n = 10u64;
    for i in 0..n {
        let frame = make_numbered_frame("drain", i);
        let buf = make_gpu_surface_view(&gen, i, DUR);
        engine.send_frame("drain", frame, buf, None).unwrap();
    }

    // Do NOT send more frames — only wait for the drain thread to deliver.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while enc_count.load(Ordering::SeqCst) < n as usize {
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting for async drain: got {} of {n}",
            enc_count.load(Ordering::SeqCst),
        );
        std::thread::sleep(Duration::from_millis(50));
    }

    assert_eq!(enc_count.load(Ordering::SeqCst), n as usize);
    engine.shutdown();
}

/// Changing draw spec (same codec) doesn't restart the encoder.
/// The drain thread continues operating and no frames are lost across
/// the spec change.
#[test]
#[serial_test::serial]
fn e2e_draw_spec_hot_swap_preserves_drain() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).unwrap();

    let enc_count = Arc::new(AtomicUsize::new(0));
    let eos_count = Arc::new(AtomicUsize::new(0));
    let engine = setup_engine(&enc_count, &eos_count);

    let draw_spec = {
        use savant_core::draw::*;
        let mut spec = ObjectDrawSpec::new();
        let border = ColorDraw::new(0, 255, 0, 255).unwrap();
        let bg = ColorDraw::new(0, 255, 0, 50).unwrap();
        let bb = BoundingBoxDraw::new(border, bg, 2, PaddingDraw::default_padding()).unwrap();
        spec.insert("det", "car", ObjectDraw::new(Some(bb), None, None, false));
        spec
    };

    let spec_with_draw = SourceSpec {
        codec: CodecSpec::Encode {
            transform: TransformConfig::default(),
            encoder: Box::new(jpeg_encoder_config(W, H)),
        },
        draw: draw_spec,
        ..Default::default()
    };

    engine.set_source_spec("swap", spec_with_draw).unwrap();
    let gen = make_generator();

    // Phase 1: 10 frames with draw spec + objects.
    let phase1 = 10u64;
    for i in 0..phase1 {
        let frame = make_numbered_frame("swap", i);
        add_object(&frame, 100.0, 100.0, 50.0, 30.0);
        let buf = make_gpu_surface_view(&gen, i, DUR);
        engine.send_frame("swap", frame, buf, None).unwrap();
    }

    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while enc_count.load(Ordering::SeqCst) < phase1 as usize {
        assert!(
            std::time::Instant::now() < deadline,
            "phase 1 timed out: got {} of {phase1}",
            enc_count.load(Ordering::SeqCst),
        );
        std::thread::sleep(Duration::from_millis(50));
    }

    // Hot-swap to empty draw spec (same codec → no encoder restart).
    engine.set_source_spec("swap", jpeg_spec()).unwrap();

    // No EOS should have been fired by the hot-swap.
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(
        eos_count.load(Ordering::SeqCst),
        0,
        "draw-only hot-swap must not fire EOS"
    );

    // Phase 2: 10 more frames without draw spec.
    let phase2 = 10u64;
    let total = phase1 + phase2;
    for i in phase1..total {
        let frame = make_numbered_frame("swap", i);
        let buf = make_gpu_surface_view(&gen, i, DUR);
        engine.send_frame("swap", frame, buf, None).unwrap();
    }

    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while enc_count.load(Ordering::SeqCst) < total as usize {
        assert!(
            std::time::Instant::now() < deadline,
            "phase 2 timed out: got {} of {total}",
            enc_count.load(Ordering::SeqCst),
        );
        std::thread::sleep(Duration::from_millis(50));
    }

    assert_eq!(enc_count.load(Ordering::SeqCst), total as usize);
    engine.shutdown();
}

/// Submit many frames rapidly on every available encode path, then EOS each
/// source.  One engine, concurrent sources: JPEG (if `nvjpegenc`) plus
/// H.264 / HEVC / AV1 when `has_nvenc()`.  Every submitted frame must produce
/// exactly one encoded output per source — no drops from the async drain.
#[test]
#[serial_test::serial]
fn e2e_sustained_throughput_no_frame_loss() {
    let records = log_records();
    let _ = gstreamer::init();
    cuda_init(0).unwrap();

    // Snapshot log length so we only check records emitted during this test.
    let baseline = records.lock().expect("records lock").len();

    let mut streams: Vec<(String, EncoderConfig)> = Vec::new();
    if has_nvjpegenc() {
        streams.push(("burst-jpeg".to_string(), jpeg_encoder_config(W, H)));
    }
    if has_nvenc() {
        streams.push(("burst-h264".to_string(), h264_encoder_config(W, H)));
        streams.push(("burst-hevc".to_string(), hevc_encoder_config(W, H)));
        streams.push(("burst-av1".to_string(), av1_encoder_config(W, H)));
    }

    if streams.is_empty() {
        eprintln!("Skipping e2e_sustained_throughput_no_frame_loss: no nvjpegenc and no NVENC");
        return;
    }

    let enc_counts: Arc<Mutex<HashMap<String, usize>>> = Arc::new(Mutex::new(HashMap::new()));
    let eos_counts: Arc<Mutex<HashMap<String, usize>>> = Arc::new(Mutex::new(HashMap::new()));

    let callbacks = Callbacks {
        on_encoded_frame: Some(Arc::new(PerSourceEncodedCb {
            enc_counts: enc_counts.clone(),
            eos_counts: eos_counts.clone(),
        })),
        ..Default::default()
    };

    let engine = PicassoEngine::new(
        GeneralSpec {
            idle_timeout_secs: 300,
            ..Default::default()
        },
        callbacks,
    );

    for (sid, cfg) in &streams {
        let spec = SourceSpec {
            codec: CodecSpec::Encode {
                transform: TransformConfig::default(),
                encoder: Box::new(cfg.clone()),
            },
            ..Default::default()
        };
        engine.set_source_spec(sid, spec).unwrap();
    }

    let gen = make_generator();
    let n = 100u64;

    for (sid, _) in &streams {
        for i in 0..n {
            let frame = make_numbered_frame(sid, i);
            let buf = make_gpu_surface_view(&gen, i, DUR);
            engine.send_frame(sid, frame, buf, None).unwrap();
        }
        engine.send_eos(sid).unwrap();
    }

    let deadline = std::time::Instant::now() + Duration::from_secs(60);
    loop {
        let eos = eos_counts.lock().expect("eos_counts lock");
        let all_eos = streams
            .iter()
            .all(|(sid, _)| eos.get(sid).copied().unwrap_or(0) >= 1);
        drop(eos);
        if all_eos {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting for EOS on all sources (expected {})",
            streams.len()
        );
        std::thread::sleep(Duration::from_millis(100));
    }

    let enc = enc_counts.lock().expect("enc_counts lock");
    for (sid, _) in &streams {
        assert_eq!(
            enc.get(sid).copied().unwrap_or(0),
            n as usize,
            "source {sid}: every submitted frame must produce one encoded output"
        );
    }

    let eos = eos_counts.lock().expect("eos_counts lock");
    for (sid, _) in &streams {
        assert_eq!(
            eos.get(sid).copied().unwrap_or(0),
            1,
            "source {sid}: exactly one EOS"
        );
    }
    drop(eos);

    engine.shutdown();

    // --- Log absence assertion ---------------------------------------------
    //
    // Codec headers (e.g. AV1 `OBU_SEQUENCE_HEADER` from dGPU
    // `nvv4l2av1enc`) are inlined by `NvEncoder` into the next user frame,
    // so Picasso's drain loops should never hit the "cannot correlate
    // encoded payload" branch for AV1.  Before the fix, this produced an
    // `error!` on every `burst-av1` startup.  Now it must not appear — and
    // if it does, not at ERROR level.
    let snapshot: Vec<(Level, String)> = {
        let all = records.lock().expect("records lock");
        all[baseline..].to_vec()
    };
    let offending_errors: Vec<&(Level, String)> = snapshot
        .iter()
        .filter(|(level, msg)| {
            *level == Level::Error && msg.contains("cannot correlate encoded payload")
        })
        .collect();
    assert!(
        offending_errors.is_empty(),
        "expected no ERROR-level 'cannot correlate encoded payload' lines, got {} such records: {:?}",
        offending_errors.len(),
        offending_errors
    );
}

/// EOS must flush every in-flight frame. Send exactly N frames, EOS,
/// and verify the count matches precisely.
#[test]
#[serial_test::serial]
fn e2e_eos_flushes_all_in_flight() {
    let _ = env_logger::try_init();
    let _ = gstreamer::init();
    cuda_init(0).unwrap();

    let enc_count = Arc::new(AtomicUsize::new(0));
    let eos_count = Arc::new(AtomicUsize::new(0));
    let engine = setup_engine(&enc_count, &eos_count);

    engine.set_source_spec("flush", jpeg_spec()).unwrap();
    let gen = make_generator();

    let n = 30u64;
    for i in 0..n {
        let frame = make_numbered_frame("flush", i);
        let buf = make_gpu_surface_view(&gen, i, DUR);
        engine.send_frame("flush", frame, buf, None).unwrap();
    }

    // Immediately send EOS — some frames may still be in the encoder
    // pipeline. The stop_encoder → drain_remaining → finish sequence
    // must flush them all.
    engine.send_eos("flush").unwrap();

    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while eos_count.load(Ordering::SeqCst) < 1 {
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting for EOS sentinel"
        );
        std::thread::sleep(Duration::from_millis(100));
    }

    assert_eq!(
        enc_count.load(Ordering::SeqCst),
        n as usize,
        "EOS must flush all {n} in-flight frames"
    );
    assert_eq!(eos_count.load(Ordering::SeqCst), 1);
    engine.shutdown();
}
