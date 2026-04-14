//! End-to-end tests for [`bridge_savant_id_meta`] — meta propagation,
//! PTS collision, and per-PTS eviction.
//!
//! The test pipeline (`appsrc → queue → appsink`) is a passive service
//! controlled entirely from the test via three handles:
//!
//! - **`appsrc`**: push buffers directly (synchronous — sink probes fire
//!   on the caller's thread before returning).
//! - **`egress`** (`Receiver<EgressMsg>`): collect per-buffer results.
//! - **`seal`** (`Arc<ReleaseSeal>`): controls a BLOCK probe on the queue's
//!   src pad.  While sealed the queue's streaming thread is held — entries
//!   accumulate in the bridge map.  The test releases the seal when ready.

use deepstream_buffers::{
    bridge_savant_id_meta, SavantIdMeta, SavantIdMetaKind, MAX_ENTRIES_PER_PTS,
};
use gstreamer::buffer::BufferMetaForeachAction;
use gstreamer::{self as gst, prelude::*};
use savant_core::utils::release_seal::ReleaseSeal;
use std::ops::ControlFlow;
use std::sync::mpsc;
use std::sync::Arc;

fn init_gst() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let _ = env_logger::try_init();
        gst::init().unwrap();
    });
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn strip_savant_meta(buf: &mut gst::BufferRef) {
    buf.foreach_meta_mut(|mut meta| {
        if meta.downcast_ref::<SavantIdMeta>().is_some() {
            ControlFlow::Continue(BufferMetaForeachAction::Remove)
        } else {
            ControlFlow::Continue(BufferMetaForeachAction::Keep)
        }
    });
}

fn make_buffer(index: usize, pts_ns: u64) -> gst::Buffer {
    let frame_size = 16 * 16 * 3 / 2; // I420 16×16
    let mut buf = gst::Buffer::with_size(frame_size).expect("alloc buffer");
    {
        let buf_ref = buf.make_mut();
        buf_ref.set_pts(gst::ClockTime::from_nseconds(pts_ns));
        buf_ref.set_duration(gst::ClockTime::from_nseconds(33_333_333));
        SavantIdMeta::replace(buf_ref, vec![SavantIdMetaKind::Frame(index as u128)]);
    }
    buf
}

fn raw_caps() -> gst::Caps {
    gst::Caps::builder("video/x-raw")
        .field("format", "I420")
        .field("width", 16i32)
        .field("height", 16i32)
        .field("framerate", gst::Fraction::new(30, 1))
        .build()
}

// ─── Egress ─────────────────────────────────────────────────────────────────

struct EgressMsg {
    meta: Option<Vec<SavantIdMetaKind>>,
}

fn collect_frame_ids(meta: &[Vec<SavantIdMetaKind>]) -> Vec<u128> {
    let mut ids: Vec<u128> = meta
        .iter()
        .map(|v| match v[0] {
            SavantIdMetaKind::Frame(id) => id,
            _ => panic!("expected Frame variant"),
        })
        .collect();
    ids.sort();
    ids
}

fn drain_egress(rx: mpsc::Receiver<EgressMsg>) -> (u32, u32, Vec<Vec<SavantIdMetaKind>>) {
    let mut received = 0u32;
    let mut no_meta = 0u32;
    let mut meta_entries = Vec::new();
    for msg in rx {
        received += 1;
        match msg.meta {
            Some(ids) => meta_entries.push(ids),
            None => no_meta += 1,
        }
    }
    (received, no_meta, meta_entries)
}

// ─── Pipeline service ───────────────────────────────────────────────────────

struct Pipeline {
    appsrc: gstreamer_app::AppSrc,
    egress: mpsc::Receiver<EgressMsg>,
    pipeline: gst::Pipeline,
    /// Released by the BLOCK probe when it enters.  The test should wait on
    /// this after pushing the first buffer to confirm the probe is engaged
    /// before pushing the rest.
    blocked: Arc<ReleaseSeal>,
}

/// Build and start `appsrc → queue → appsink` with the bridge on `queue`.
///
/// `seal` controls a BLOCK probe on the queue's src pad.  While sealed the
/// queue's streaming thread is held — entries accumulate in the bridge map.
/// Releasing the seal removes the probe and lets all queued buffers flow.
///
/// The caller pushes buffers directly via `Pipeline::appsrc` — each
/// `push_buffer` synchronously fires the bridge's sink probe on the
/// caller's thread, guaranteeing the map is fully populated before the
/// caller releases the seal.
fn start_pipeline(seal: Arc<ReleaseSeal>, max_buffers: u32) -> Pipeline {
    init_gst();

    let pipeline = gst::Pipeline::new();
    let appsrc_elem = gst::ElementFactory::make("appsrc").build().expect("appsrc");
    let queue = gst::ElementFactory::make("queue").build().expect("queue");
    let appsink_elem = gst::ElementFactory::make("appsink")
        .build()
        .expect("appsink");

    queue.set_property("max-size-buffers", max_buffers + 8);
    queue.set_property("max-size-time", 0u64);
    queue.set_property("max-size-bytes", 0u32);

    // ── Gate: BLOCK probe on src pad (registered FIRST) ─────────────────
    let blocked = Arc::new(ReleaseSeal::new());
    let blocked_signal = blocked.clone();
    let seal_probe = seal;
    queue.static_pad("src").unwrap().add_probe(
        gst::PadProbeType::BLOCK | gst::PadProbeType::BUFFER,
        move |_pad, _info| {
            blocked_signal.release();
            seal_probe.wait();
            gst::PadProbeReturn::Remove
        },
    );

    // ── Bridge (src probe registered AFTER the gate) ────────────────────
    bridge_savant_id_meta(&queue).expect("bridge_savant_id_meta");

    // ── Strip probe on sink pad (AFTER bridge sink probe) ───────────────
    queue
        .static_pad("sink")
        .unwrap()
        .add_probe(gst::PadProbeType::BUFFER, |_pad, info| {
            if let Some(buffer) = info.buffer_mut() {
                strip_savant_meta(buffer.make_mut());
            }
            gst::PadProbeReturn::Ok
        });

    // ── Wire up ─────────────────────────────────────────────────────────
    appsrc_elem.set_property("caps", raw_caps());
    appsrc_elem.set_property_from_str("format", "time");
    appsrc_elem.set_property_from_str("stream-type", "stream");
    appsrc_elem.set_property("is-live", false);
    appsink_elem.set_property("sync", false);
    appsink_elem.set_property("emit-signals", true);

    pipeline
        .add_many([&appsrc_elem, &queue, &appsink_elem])
        .expect("add elements");
    gst::Element::link_many([&appsrc_elem, &queue, &appsink_elem]).expect("link");

    // ── Egress: appsink → channel ───────────────────────────────────────
    let (egress_tx, egress_rx) = mpsc::channel();
    let appsink = appsink_elem
        .dynamic_cast::<gstreamer_app::AppSink>()
        .unwrap();
    appsink.set_callbacks(
        gstreamer_app::AppSinkCallbacks::builder()
            .new_sample(move |sink| {
                let sample = sink.pull_sample().map_err(|_| gst::FlowError::Error)?;
                let meta = sample
                    .buffer()
                    .and_then(|b| b.meta::<SavantIdMeta>())
                    .map(|m| m.ids().to_vec());
                let _ = egress_tx.send(EgressMsg { meta });
                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );

    pipeline
        .set_state(gst::State::Playing)
        .expect("set Playing");

    let appsrc = appsrc_elem.dynamic_cast::<gstreamer_app::AppSrc>().unwrap();

    Pipeline {
        appsrc,
        egress: egress_rx,
        pipeline,
        blocked,
    }
}

impl Pipeline {
    fn push(&self, buf: gst::Buffer) {
        self.appsrc.push_buffer(buf).expect("push_buffer");
    }

    /// Block until the BLOCK probe has engaged on the queue's src pad.
    /// Call this after pushing the first buffer so the test knows the gate
    /// is actively holding before pushing the rest.
    fn wait_for_block(&self) {
        assert!(
            self.blocked.wait_timeout(std::time::Duration::from_secs(5)),
            "BLOCK probe did not engage within 5 seconds"
        );
    }

    fn finish(self) -> (u32, u32, Vec<Vec<SavantIdMetaKind>>) {
        self.appsrc.end_of_stream().unwrap();
        let bus = self.pipeline.bus().unwrap();
        for msg in bus.iter_timed(gst::ClockTime::from_seconds(10)) {
            match msg.view() {
                gst::MessageView::Eos(..) => break,
                gst::MessageView::Error(err) => {
                    panic!(
                        "Pipeline error from {:?}: {:?}",
                        err.src().map(|s| s.path_string()),
                        err.error()
                    );
                }
                _ => {}
            }
        }
        self.pipeline.set_state(gst::State::Null).unwrap();
        // Drop pipeline + appsrc so the appsink callback (which owns the
        // egress sender) is destroyed.  Only then will the egress channel
        // close, allowing drain_egress to return.
        let egress = self.egress;
        drop(self.appsrc);
        drop(self.pipeline);
        drain_egress(egress)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[test]
fn bridge_meta_unique_pts() {
    let seal = Arc::new(ReleaseSeal::new());
    seal.release(); // gate open — no accumulation needed

    let p = start_pipeline(seal, 30);
    for i in 0..30u64 {
        p.push(make_buffer(i as usize, i * 33_333_333));
    }

    let (recv, no_meta, meta_entries) = p.finish();
    assert_eq!(recv, 30, "should receive all 30 buffers");
    assert_eq!(no_meta, 0, "every buffer should have meta");
    assert_eq!(meta_entries.len(), 30);

    let frame_ids = collect_frame_ids(&meta_entries);
    let expected: Vec<u128> = (0..30).collect();
    assert_eq!(frame_ids, expected, "all frame IDs must be present");
}

#[test]
fn bridge_meta_pts_collision() {
    let seal = Arc::new(ReleaseSeal::new());
    let pts = [
        1_000_000_000u64,
        2_000_000_000,
        1_000_000_000,
        2_000_000_000,
        1_000_000_000,
        2_000_000_000,
    ];

    let p = start_pipeline(seal.clone(), pts.len() as u32);
    p.push(make_buffer(0, pts[0]));
    p.wait_for_block();
    for (i, &pts_ns) in pts.iter().enumerate().skip(1) {
        p.push(make_buffer(i, pts_ns));
    }
    // All 6 entries accumulated — release the seal.
    seal.release();

    let (recv, no_meta, meta_entries) = p.finish();
    assert_eq!(recv, 6, "should receive all 6 buffers");
    assert_eq!(no_meta, 0, "every buffer should have meta");
    assert_eq!(meta_entries.len(), 6);

    let frame_ids = collect_frame_ids(&meta_entries);
    let expected: Vec<u128> = (0..6).collect();
    assert_eq!(
        frame_ids, expected,
        "all frame IDs must survive PTS collision"
    );
}

#[test]
fn bridge_meta_all_same_pts() {
    let seal = Arc::new(ReleaseSeal::new());
    let p = start_pipeline(seal.clone(), 10);
    p.push(make_buffer(0, 42_000_000));
    p.wait_for_block();
    for i in 1..10 {
        p.push(make_buffer(i, 42_000_000));
    }
    seal.release();

    let (recv, no_meta, meta_entries) = p.finish();
    assert_eq!(recv, 10);
    assert_eq!(no_meta, 0, "every buffer should have meta");
    assert_eq!(meta_entries.len(), 10);

    let frame_ids = collect_frame_ids(&meta_entries);
    let expected: Vec<u128> = (0..10).collect();
    assert_eq!(frame_ids, expected);
}

/// Per-PTS eviction correctness is verified by the unit test
/// `bridge_map_per_pts_eviction` in `lib.rs` (directly exercises the map
/// logic without GStreamer threading).  This e2e test confirms that
/// the bridge handles a large number of same-PTS buffers gracefully.
#[test]
fn bridge_meta_many_same_pts() {
    let seal = Arc::new(ReleaseSeal::new());
    let count = MAX_ENTRIES_PER_PTS + 1; // 33

    let p = start_pipeline(seal.clone(), count as u32);
    p.push(make_buffer(0, 99_000_000_000));
    p.wait_for_block();
    for i in 1..count {
        p.push(make_buffer(i, 99_000_000_000));
    }
    seal.release();

    let (recv, _no_meta, meta_entries) = p.finish();
    assert_eq!(recv, count as u32, "should receive all {} buffers", count);

    // At least some buffers should carry meta (bridge is working).
    assert!(
        !meta_entries.is_empty(),
        "at least some buffers should have bridge-restored meta"
    );
    // Every meta entry should contain a valid Frame id.
    for ids in &meta_entries {
        assert!(matches!(ids[0], SavantIdMetaKind::Frame(_)));
    }
}
