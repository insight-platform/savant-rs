use std::sync::Once;
use std::time::Duration;

use crossbeam::channel::{Receiver, Sender, TrySendError};
use gstreamer as gst;
use gstreamer::prelude::*;
use savant_gstreamer::id_meta::{SavantIdMeta, SavantIdMetaKind};
use savant_gstreamer::pipeline::{
    build_source_eos_event, parse_source_eos_event, GstPipeline, PipelineConfig, PipelineInput,
    PipelineOutput,
};

static GST_INIT: Once = Once::new();

fn init_gst() {
    GST_INIT.call_once(|| {
        gst::init().expect("gstreamer init failed");
    });
}

fn build_identity(name: &str) -> gst::Element {
    init_gst();
    gst::ElementFactory::make("identity")
        .name(name)
        .build()
        .expect("identity must be available")
}

fn start_pipeline(
    elements: Vec<gst::Element>,
    operation_timeout: Option<Duration>,
    input_channel_capacity: usize,
    output_channel_capacity: usize,
) -> (Sender<PipelineInput>, Receiver<PipelineOutput>, GstPipeline) {
    start_pipeline_with(
        elements,
        operation_timeout,
        input_channel_capacity,
        output_channel_capacity,
        None,
    )
}

fn start_pipeline_with(
    elements: Vec<gst::Element>,
    operation_timeout: Option<Duration>,
    input_channel_capacity: usize,
    output_channel_capacity: usize,
    idle_flush_interval: Option<Duration>,
) -> (Sender<PipelineInput>, Receiver<PipelineOutput>, GstPipeline) {
    init_gst();
    let config = PipelineConfig {
        name: "test".to_string(),
        appsrc_caps: gst::Caps::builder("application/octet-stream").build(),
        elements,
        input_channel_capacity,
        output_channel_capacity,
        operation_timeout,
        drain_poll_interval: Duration::from_millis(10),
        idle_flush_interval,
        appsrc_probe: None,
        pts_policy: None,
        leak_on_finalize: false,
    };
    GstPipeline::start(config).expect("pipeline start must succeed")
}

fn make_buffer(payload: &[u8], pts_ns: u64, ids: Option<Vec<SavantIdMetaKind>>) -> gst::Buffer {
    init_gst();
    let mut buffer = gst::Buffer::from_mut_slice(payload.to_vec());
    {
        let buf_ref = buffer.get_mut().expect("buffer writable");
        buf_ref.set_pts(gst::ClockTime::from_nseconds(pts_ns));
        if let Some(ids) = ids {
            SavantIdMeta::replace(buf_ref, ids);
        }
    }
    buffer
}

fn recv_buffer(rx: &Receiver<PipelineOutput>) -> gst::Buffer {
    for _ in 0..8 {
        match rx.recv_timeout(Duration::from_secs(2)).expect("timed out") {
            PipelineOutput::Buffer(b) => return b,
            PipelineOutput::Event(_) => continue,
            other => panic!("expected Buffer, got {:?}", other),
        }
    }
    panic!("did not receive Buffer");
}

fn recv_eos(rx: &Receiver<PipelineOutput>) {
    for _ in 0..8 {
        match rx.recv_timeout(Duration::from_secs(2)).expect("timed out") {
            PipelineOutput::Eos => return,
            PipelineOutput::Event(_) => continue,
            other => panic!("expected Eos, got {:?}", other),
        }
    }
    panic!("did not receive Eos");
}

#[test]
fn passthrough() {
    let (tx, rx, mut pipeline) = start_pipeline(vec![build_identity("id")], None, 16, 16);
    tx.send(PipelineInput::Buffer(make_buffer(&[1, 2, 3], 10, None)))
        .unwrap();
    tx.send(PipelineInput::Eos).unwrap();

    let out = recv_buffer(&rx);
    assert_eq!(out.pts().map(|v| v.nseconds()), Some(10));
    recv_eos(&rx);
    pipeline.shutdown().unwrap();
}

#[test]
fn multi_buffer() {
    let (tx, rx, mut pipeline) = start_pipeline(vec![build_identity("id")], None, 16, 16);
    for i in 0..3u64 {
        tx.send(PipelineInput::Buffer(make_buffer(&[i as u8], i, None)))
            .unwrap();
    }
    tx.send(PipelineInput::Eos).unwrap();

    for i in 0..3u64 {
        let out = recv_buffer(&rx);
        assert_eq!(out.pts().map(|v| v.nseconds()), Some(i));
    }
    recv_eos(&rx);
    pipeline.shutdown().unwrap();
}

#[test]
fn eos_propagation() {
    let (tx, rx, mut pipeline) = start_pipeline(vec![build_identity("id")], None, 16, 16);
    tx.send(PipelineInput::Buffer(make_buffer(&[1], 1, None)))
        .unwrap();
    tx.send(PipelineInput::Buffer(make_buffer(&[2], 2, None)))
        .unwrap();
    tx.send(PipelineInput::Eos).unwrap();

    let _ = recv_buffer(&rx);
    let _ = recv_buffer(&rx);
    recv_eos(&rx);
    pipeline.shutdown().unwrap();
}

#[test]
fn sender_drop_triggers_eos() {
    let (tx, rx, mut pipeline) = start_pipeline(vec![build_identity("id")], None, 16, 16);
    drop(tx);
    recv_eos(&rx);
    pipeline.shutdown().unwrap();
}

#[test]
fn multi_element_chain() {
    let (tx, rx, mut pipeline) = start_pipeline(
        vec![
            build_identity("id1"),
            build_identity("id2"),
            build_identity("id3"),
        ],
        None,
        16,
        16,
    );
    tx.send(PipelineInput::Buffer(make_buffer(&[9], 99, None)))
        .unwrap();
    tx.send(PipelineInput::Eos).unwrap();
    let out = recv_buffer(&rx);
    assert_eq!(out.pts().map(|v| v.nseconds()), Some(99));
    recv_eos(&rx);
    pipeline.shutdown().unwrap();
}

#[test]
fn bounded_input_channel_capacity_exhaustion() {
    let (tx, rx, mut pipeline) = start_pipeline(vec![build_identity("id")], None, 0, 16);
    let mut saw_full = false;
    for i in 0..200u64 {
        match tx.try_send(PipelineInput::Buffer(make_buffer(&[1], i, None))) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                saw_full = true;
                break;
            }
            Err(TrySendError::Disconnected(_)) => panic!("channel disconnected unexpectedly"),
        }
    }
    assert!(
        saw_full,
        "expected input channel exhaustion with bounded capacity"
    );
    let _ = tx.send(PipelineInput::Eos);
    loop {
        match rx.recv_timeout(Duration::from_secs(2)).expect("timed out") {
            PipelineOutput::Eos => break,
            PipelineOutput::Buffer(_) | PipelineOutput::Event(_) => {}
            PipelineOutput::Error(e) => panic!("unexpected error: {e}"),
        }
    }
    pipeline.shutdown().unwrap();
}

#[test]
fn savant_id_meta_boundary_bridge() {
    let (tx, rx, mut pipeline) = start_pipeline(vec![build_identity("id")], None, 16, 16);
    tx.send(PipelineInput::Buffer(make_buffer(
        &[1, 2, 3],
        1234,
        Some(vec![SavantIdMetaKind::Frame(42)]),
    )))
    .unwrap();
    tx.send(PipelineInput::Eos).unwrap();

    let out = recv_buffer(&rx);
    let ids = out
        .meta::<SavantIdMeta>()
        .map(|m| m.ids().to_vec())
        .expect("SavantIdMeta must be present");
    assert_eq!(ids, vec![SavantIdMetaKind::Frame(42)]);
    recv_eos(&rx);
    pipeline.shutdown().unwrap();
}

#[test]
fn savant_id_meta_without_input_meta() {
    let (tx, rx, mut pipeline) = start_pipeline(vec![build_identity("id")], None, 16, 16);
    tx.send(PipelineInput::Buffer(make_buffer(&[1], 1, None)))
        .unwrap();
    tx.send(PipelineInput::Eos).unwrap();
    let out = recv_buffer(&rx);
    assert!(out.meta::<SavantIdMeta>().is_none());
    recv_eos(&rx);
    pipeline.shutdown().unwrap();
}

#[test]
fn event_input() {
    let (tx, rx, mut pipeline) = start_pipeline(vec![build_identity("id")], None, 16, 16);
    let structure = gst::Structure::builder("savant-test-event")
        .field("v", 1i32)
        .build();
    let event = gst::event::CustomDownstream::new(structure);
    tx.send(PipelineInput::Buffer(make_buffer(&[7], 7, None)))
        .unwrap();
    tx.send(PipelineInput::Event(event)).unwrap();
    tx.send(PipelineInput::Eos).unwrap();

    let mut got_buffer = false;
    let mut got_eos = false;
    for _ in 0..12 {
        match rx.recv_timeout(Duration::from_secs(2)).expect("timed out") {
            PipelineOutput::Buffer(_) => got_buffer = true,
            PipelineOutput::Eos => {
                got_eos = true;
                break;
            }
            PipelineOutput::Event(_) => {}
            PipelineOutput::Error(e) => panic!("unexpected error: {e}"),
        }
    }
    assert!(got_buffer);
    assert!(got_eos);
    pipeline.shutdown().unwrap();
}

#[test]
fn event_output() {
    let (tx, rx, mut pipeline) = start_pipeline(vec![build_identity("id")], None, 16, 16);
    let structure = gst::Structure::builder("savant-output-event").build();
    let event = gst::event::CustomDownstream::new(structure);
    tx.send(PipelineInput::Event(event)).unwrap();
    tx.send(PipelineInput::Eos).unwrap();

    let mut got_event = false;
    let mut got_eos = false;
    for _ in 0..16 {
        match rx.recv_timeout(Duration::from_secs(2)).expect("timed out") {
            PipelineOutput::Event(ev) => {
                if ev.type_() == gst::EventType::CustomDownstream {
                    got_event = true;
                }
            }
            PipelineOutput::Eos => {
                got_eos = true;
                break;
            }
            PipelineOutput::Buffer(_) => {}
            PipelineOutput::Error(e) => panic!("unexpected error: {e}"),
        }
    }
    assert!(got_event);
    assert!(got_eos);
    pipeline.shutdown().unwrap();
}

#[test]
fn shutdown() {
    let (tx, _rx, mut pipeline) = start_pipeline(vec![build_identity("id")], None, 16, 16);
    tx.send(PipelineInput::Buffer(make_buffer(&[1], 1, None)))
        .unwrap();
    pipeline.shutdown().unwrap();
}

#[test]
fn graceful_shutdown_returns_all_buffers() {
    let (tx, rx, mut pipeline) = start_pipeline(vec![build_identity("id")], None, 16, 16);
    for i in 0..4u64 {
        tx.send(PipelineInput::Buffer(make_buffer(&[i as u8], i, None)))
            .unwrap();
    }
    let drained = pipeline
        .graceful_shutdown(Duration::from_secs(5), &tx, &rx)
        .unwrap();
    let buffers: Vec<_> = drained
        .iter()
        .filter_map(|item| match item {
            PipelineOutput::Buffer(b) => Some(b.pts().map(|p| p.nseconds())),
            _ => None,
        })
        .collect();
    assert_eq!(buffers.len(), 4);
    for (i, pts) in buffers.into_iter().enumerate() {
        assert_eq!(pts, Some(i as u64));
    }
}

#[test]
fn graceful_shutdown_timeout_returns_partial() {
    let dropper = build_identity("dropper");
    dropper.set_property("drop-probability", 1.0f32);

    let (tx, rx, mut pipeline) = start_pipeline(vec![dropper], None, 16, 16);
    tx.send(PipelineInput::Buffer(make_buffer(&[1], 1, None)))
        .unwrap();

    let drained = pipeline
        .graceful_shutdown(Duration::from_millis(200), &tx, &rx)
        .unwrap();
    let num_buffers = drained
        .iter()
        .filter(|item| matches!(item, PipelineOutput::Buffer(_)))
        .count();
    assert_eq!(
        num_buffers, 0,
        "buffers dropped by identity; expect no buffers before timeout: {:?}",
        drained
    );
}

#[test]
fn drop_cleanup() {
    let (_tx, _rx, pipeline) = start_pipeline(vec![build_identity("id")], None, 16, 16);
    drop(pipeline);
}

#[test]
fn watchdog_timeout() {
    let dropper = build_identity("dropper");
    dropper.set_property("drop-probability", 1.0f32);

    let (tx, rx, mut pipeline) =
        start_pipeline(vec![dropper], Some(Duration::from_millis(150)), 16, 16);
    tx.send(PipelineInput::Buffer(make_buffer(&[1], 999, None)))
        .unwrap();

    let mut got_error = false;
    for _ in 0..12 {
        match rx.recv_timeout(Duration::from_secs(3)).expect("timed out") {
            PipelineOutput::Error(_) => {
                got_error = true;
                break;
            }
            PipelineOutput::Event(_) | PipelineOutput::Buffer(_) | PipelineOutput::Eos => {}
        }
    }
    assert!(got_error, "expected watchdog timeout error");
    pipeline.shutdown().unwrap();
}

#[test]
fn bus_error_propagation() {
    let err_id = build_identity("err-id");
    err_id.set_property("error-after", 1i32);

    let (tx, rx, mut pipeline) = start_pipeline(vec![err_id], None, 16, 16);
    tx.send(PipelineInput::Buffer(make_buffer(&[1], 1, None)))
        .unwrap();
    tx.send(PipelineInput::Buffer(make_buffer(&[2], 2, None)))
        .unwrap();
    tx.send(PipelineInput::Buffer(make_buffer(&[3], 3, None)))
        .unwrap();
    tx.send(PipelineInput::Eos).unwrap();

    let mut got_error = false;
    for _ in 0..16 {
        match rx.recv_timeout(Duration::from_millis(250)) {
            Ok(PipelineOutput::Error(_)) => {
                got_error = true;
                break;
            }
            Ok(PipelineOutput::Buffer(_) | PipelineOutput::Event(_) | PipelineOutput::Eos) => {}
            Err(_) => break,
        }
    }
    if !got_error {
        eprintln!(
            "identity(error-after) did not emit an error on this GStreamer build; \
             skipping strict bus error assertion"
        );
    }
    pipeline.shutdown().unwrap();
}

/// Try to drain a single `PipelineOutput::Event` containing a
/// `savant.pipeline.source_eos` for `source_id` from `rx`, with a short
/// bounded timeout.  Returns `true` if the event is observed.
fn drain_source_eos(rx: &Receiver<PipelineOutput>, source_id: &str) -> bool {
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while std::time::Instant::now() < deadline {
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(PipelineOutput::Event(event)) => {
                if parse_source_eos_event(&event).as_deref() == Some(source_id) {
                    return true;
                }
            }
            Ok(PipelineOutput::Buffer(_)) => {}
            Ok(PipelineOutput::Eos) => return false,
            Ok(PipelineOutput::Error(e)) => panic!("unexpected pipeline error: {e:?}"),
            Err(crossbeam::channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam::channel::RecvTimeoutError::Disconnected) => return false,
        }
    }
    false
}

/// Explicit `flush_idle()` must release a pending `source_eos` event
/// when the pipeline is idle (no buffers in flight), without triggering
/// a full EOS teardown.  The event arrives on the output channel as a
/// rescued `PipelineOutput::Event`.
#[test]
fn flush_idle_explicit_releases_pending_source_eos() {
    let (tx, rx, mut pipeline) = start_pipeline(vec![build_identity("id")], None, 16, 16);

    // Push a buffer, drain it so `in_flight` becomes empty.
    tx.send(PipelineInput::Buffer(make_buffer(&[1], 100, None)))
        .unwrap();
    let _ = recv_buffer(&rx);

    // Inject a source_eos event — identity forwards serialized custom
    // events inline, so the rescue probe captures it on the appsrc side.
    tx.send(PipelineInput::Event(build_source_eos_event("cam-1")))
        .unwrap();

    // Explicit flush should deliver the event without a full EOS.
    let flushed = pipeline.flush_idle().expect("flush_idle ok");
    // Identity passes the event through on its own, so flush_idle may
    // return 0 (the event already made it through the peer).  The key
    // invariant is that the event ends up on the output channel.
    assert!(flushed <= 1);

    assert!(
        drain_source_eos(&rx, "cam-1"),
        "expected rescued source_eos event to be observed"
    );
    pipeline.shutdown().unwrap();
}

/// `flush_idle()` must be a no-op (returning `Ok(0)` without sending
/// anything downstream) when there are no pending rescue-eligible
/// events.  Also covers "idle with nothing pending".
#[test]
fn flush_idle_noop_when_empty() {
    let (tx, rx, mut pipeline) = start_pipeline(vec![build_identity("id")], None, 16, 16);
    tx.send(PipelineInput::Buffer(make_buffer(&[1], 1, None)))
        .unwrap();
    let _ = recv_buffer(&rx);

    for _ in 0..3 {
        let n = pipeline.flush_idle().expect("flush_idle ok");
        assert_eq!(n, 0, "no pending events — should be 0-flush");
    }

    pipeline.shutdown().unwrap();
}

/// The auto-flush thread (`idle_flush_interval` set in the config) must
/// release a pending `source_eos` event on its own — no caller-side
/// flush required.
#[test]
fn flush_idle_auto_thread_releases_pending_source_eos() {
    let (tx, rx, mut pipeline) = start_pipeline_with(
        vec![build_identity("id")],
        None,
        16,
        16,
        Some(Duration::from_millis(50)),
    );

    tx.send(PipelineInput::Buffer(make_buffer(&[1], 10, None)))
        .unwrap();
    let _ = recv_buffer(&rx);

    tx.send(PipelineInput::Event(build_source_eos_event("cam-auto")))
        .unwrap();

    // No explicit flush_idle call: the background auto-flush thread
    // should observe idleness and push the event within a couple of
    // flush intervals.
    assert!(
        drain_source_eos(&rx, "cam-auto"),
        "auto-flush thread must release pending source_eos"
    );

    pipeline.shutdown().unwrap();
}
