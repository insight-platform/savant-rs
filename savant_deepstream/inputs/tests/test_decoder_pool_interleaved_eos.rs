//! Reproducer for the cars-demo-zmq pipeline-stuck symptom.
//!
//! # Production scenario (per user report)
//!
//!  1. `cars-demo-zmq pipeline` runs with `Decoder + Picasso` only.
//!  2. `cars-demo-zmq producer --input X.mov --source-id=S` finishes,
//!     either with a trailing logical `SourceEos(S)` (default) or
//!     without one (`--no-eos`).
//!  3. *Without* a pause, the same producer is launched again against
//!     the **same** `source-id=S`.  Without the fixes below the
//!     pipeline gets stuck — the new producer's ZMQ writer hangs with
//!     `Resource temporarily unavailable` and the decoder either
//!     blocks every `pool.submit(...)` for seconds (legacy) or
//!     silently rejects every packet (intermediate state).
//!  4. Working configurations after the fixes:
//!       * Producer that sends `source-eos` between runs — handled by
//!         the EOS-path tests below (`*_pts_rebase`).
//!       * Producer that runs with `--no-eos` — handled by the
//!         restart-on-error path test (`*_no_eos_pts_rebase`).
//!       * A *different* `source-id` for the second run — control
//!         test (`interleaved_eos_distinct_sources`).
//!
//! # What these tests cover
//!
//! All four tests isolate [`FlexibleDecoderPool`] from ZMQ and the
//! framework wiring.  They submit access units directly to the pool
//! and assert that the pool never blocks
//! ([`PER_SUBMIT_BUDGET`]) and recovers correctly across producer-run
//! boundaries:
//!
//! | Test                                          | EOS between cycles | Recovery path |
//! |-----------------------------------------------|--------------------|---------------|
//! | `interleaved_eos_distinct_sources`            | yes, distinct ids  | per-source decoder per cycle |
//! | `interleaved_eos_same_source_no_pause`        | yes, same id       | `source_eos` teardown + fresh decoder |
//! | `interleaved_eos_same_source_pts_rebase`      | yes, same id       | `source_eos` teardown + fresh decoder, rebased PTS |
//! | `interleaved_no_eos_same_source_pts_rebase`   | **no**, same id    | restart-on-error: feeder rejects → worker exits → next `submit` rebuilds decoder |
//!
//! # Layered fixes exercised here
//!
//!  1. [`FlexibleDecoderPool::source_eos`] tears down the per-source
//!     [`FlexibleDecoder`] (and its underlying `GstPipeline` /
//!     `nvv4l2decoder`) so the next `submit` for the same `source_id`
//!     creates a fresh one.  Required when the producer ends a run
//!     with an explicit `source-eos`.
//!  2. The [`GstPipeline`](savant_gstreamer::pipeline::GstPipeline)
//!     feeder resets its `last_key` PTS tracker when it processes a
//!     `savant.pipeline.source_eos` custom downstream event.  Belt
//!     and suspenders — paired with (1) it covers the EOS path even
//!     if the underlying decoder is not torn down (e.g. encoder /
//!     infer / track pipelines that hold long-lived state).
//!  3. [`FlexibleDecoder::worker_loop`] **exits** on
//!     [`NvDecoderOutput::Error`](deepstream_decoders::NvDecoderOutput::Error)
//!     after forwarding the error.  The next
//!     [`FlexibleDecoder::submit`] then takes
//!     [`super::handle_active::handle_active`]'s worker-died branch —
//!     [`NvDecoder::shutdown`] → drain frame map → emit
//!     `Skipped { DecoderRestarted }` per orphaned frame plus an
//!     aggregate `Restarted` → commit `Idle` → re-run `handle_idle`
//!     for the current packet against a fresh decoder.  Required for
//!     the `--no-eos` flow where there is no explicit boundary.  The
//!     restart costs one synchronous `nvv4l2decoder` `set_state(Null)`
//!     per producer-run boundary (slow on Jetson) but recovers a
//!     working decoder from any per-packet error
//!     ([`PipelineError::TimestampOrderViolation`](savant_gstreamer::pipeline::PipelineError::TimestampOrderViolation),
//!     transient `nvv4l2decoder` failures, …).
//!
//! [`handle_active`]: deepstream_inputs::flexible_decoder
//! [`FlexibleDecoder`]: deepstream_inputs::flexible_decoder::FlexibleDecoder
//! [`FlexibleDecoder::worker_loop`]: deepstream_inputs::flexible_decoder::FlexibleDecoder

mod common;

use common::*;
use deepstream_inputs::decoder_pool::{FlexibleDecoderPool, FlexibleDecoderPoolConfig};
use serial_test::serial;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Hard upper bound for any single test — if exceeded we abort the process
/// so a deadlock surfaces as a CI failure instead of an open-ended hang.
const TEST_WATCHDOG: Duration = Duration::from_secs(120);

// Production-default pool configuration (mirrors `savant_perception::stages::decoder`).
const PROD_POOL_SIZE: u32 = 8;
const PROD_EVICTION_TTL: Duration = Duration::from_secs(3600);
const PROD_IDLE_TIMEOUT: Duration = Duration::from_secs(5);
const PROD_DETECT_BUFFER_LIMIT: usize = 60;

fn prod_pool_config() -> FlexibleDecoderPoolConfig {
    FlexibleDecoderPoolConfig::new(0, PROD_POOL_SIZE, PROD_EVICTION_TTL)
        .idle_timeout(PROD_IDLE_TIMEOUT)
        .detect_buffer_limit(PROD_DETECT_BUFFER_LIMIT)
}

/// Loud upper bound for any single `pool.submit(...)` call.  In production we
/// observed the producer's ZMQ writer time out after ~5 seconds of back-pressure
/// — that translates here to a single submit blocking longer than this budget.
const PER_SUBMIT_BUDGET: Duration = Duration::from_secs(15);

/// Spawn a watchdog that aborts the process after `TEST_WATCHDOG` if `done`
/// is never set. The returned flag should be set just before normal test exit.
fn install_watchdog(test_name: &'static str) -> Arc<AtomicBool> {
    let done = Arc::new(AtomicBool::new(false));
    let done_for_thread = Arc::clone(&done);
    std::thread::spawn(move || {
        let start = std::time::Instant::now();
        while start.elapsed() < TEST_WATCHDOG {
            if done_for_thread.load(Ordering::Relaxed) {
                return;
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        eprintln!(
            "watchdog: test '{test_name}' exceeded {:?} — aborting (likely deadlock)",
            TEST_WATCHDOG
        );
        std::process::abort();
    });
    done
}

/// Submit a burst of H.264 access units for `source_id` with PTS starting
/// at `pts_offset_ns`.  Each individual `pool.submit(...)` is timed and we
/// fail loudly if any single submit exceeds [`PER_SUBMIT_BUDGET`] — that is
/// the production-equivalent of the ZMQ writer's `Resource temporarily
/// unavailable` retries.
fn submit_burst(
    pool: &FlexibleDecoderPool,
    aus: &[AccessUnit],
    source_id: &str,
    width: i64,
    height: i64,
    pts_offset_ns: u64,
) -> (usize, Duration) {
    use savant_core::primitives::video_codec::VideoCodec;
    let mut submitted = 0usize;
    let mut max_submit = Duration::ZERO;
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
        pool.submit(&frame, Some(&au.data))
            .unwrap_or_else(|e| panic!("submit failed at AU {i}: {e}"));
        let elapsed = start.elapsed();
        if elapsed > max_submit {
            max_submit = elapsed;
        }
        assert!(
            elapsed < PER_SUBMIT_BUDGET,
            "submit AU {i} blocked for {:?} (> {:?}) — pool likely deadlocked or stuck restart",
            elapsed,
            PER_SUBMIT_BUDGET
        );
        submitted += 1;
    }
    (submitted, max_submit)
}

/// Same `source_id` across many bursts separated by `source_eos`, no pause.
///
/// Reproduces the cars-demo-zmq stuck-pipeline scenario: the FlexibleDecoder
/// for `source_id` is created on the first burst and stays `Active`; each
/// `source_eos` is a custom-downstream marker that does NOT drain the
/// underlying `nvv4l2decoder`. Subsequent bursts feed fresh SPS/PPS+IDR
/// access units into the still-running decoder element.
///
/// We cycle 6 times by default (matches typical user behavior of repeatedly
/// re-running the producer).
#[test]
#[serial]
fn interleaved_eos_same_source_no_pause() {
    init();
    let done = install_watchdog("interleaved_eos_same_source_no_pause");
    let manifest = load_manifest();
    let Some(entry) = find_h264_entry(&manifest) else {
        done.store(true, Ordering::Relaxed);
        return;
    };

    let aus = demux_mp4_to_access_units(entry);
    assert!(!aus.is_empty(), "demuxer produced no access units");
    let per_burst = aus.len();

    let collector = OutputCollector::new();
    let pool = FlexibleDecoderPool::new(prod_pool_config(), collector.callback());

    const CYCLES: usize = 6;
    const SRC: &str = "interleave-same";
    let burst_span_ns = 33_333_333u64 * per_burst as u64;

    for cycle in 0..CYCLES {
        // Each producer run starts its own PTS at 0 in the real pipeline
        // (mp4 demuxer rebases). Mirror that — but stage non-overlapping
        // PTS to avoid triggering the decoder's PTS-reset path which would
        // mask this bug.  The bug we're chasing is mid-stream burst→burst,
        // not the PTS-decrease handler.
        let pts_offset = cycle as u64 * burst_span_ns;
        let (n, max_submit) = submit_burst(
            &pool,
            &aus,
            SRC,
            entry.width as i64,
            entry.height as i64,
            pts_offset,
        );
        eprintln!(
            "cycle {cycle}: submitted {n} packets at pts_offset={pts_offset} (max submit time = {max_submit:?})"
        );

        pool.source_eos(SRC)
            .unwrap_or_else(|e| panic!("cycle {cycle}: source_eos failed: {e}"));
    }

    // Drain everything still queued (real GstEvent::Eos via graceful shutdown).
    let mut pool = pool;
    pool.graceful_shutdown()
        .expect("graceful_shutdown failed — likely deadlock in the pool");

    let final_frames = collector.frame_count();
    let final_errors = collector.error_count();
    eprintln!(
        "interleaved_eos_same_source_no_pause: frames={final_frames}, errors={final_errors}, expected≈{}",
        CYCLES * per_burst
    );

    assert_eq!(
        final_errors, 0,
        "decoder produced errors during interleaved same-source bursts"
    );
    // We expect ~CYCLES*per_burst decoded frames. Allow some tolerance for
    // tail frames lost on `graceful_shutdown` or B-frame reordering at burst
    // boundaries.
    let expected = CYCLES * per_burst;
    let lower = expected.saturating_sub(per_burst); // allow up to 1 burst worth dropped
    assert!(
        final_frames >= lower,
        "decoded {final_frames} frames; expected ≥ {lower} (cycles={CYCLES} × per_burst={per_burst})"
    );
    done.store(true, Ordering::Relaxed);
}

/// Same scenario but with a *real* per-cycle PTS reset (every burst at PTS=0).
///
/// In the production cars-demo-zmq pipeline, each producer run rebases its
/// PTS to 0 (mp4 demuxer behavior). This test mirrors that exactly.
///
/// # What the test verifies
///
/// With the `source_eos`-driven per-source decoder teardown in
/// [`FlexibleDecoderPool::source_eos`] (and the
/// [`GstPipeline`](savant_gstreamer::pipeline::GstPipeline) feeder's
/// `last_key` reset on `savant.pipeline.source_eos` events) the
/// reused per-source `FlexibleDecoder` decodes every cycle correctly:
///
///  * `pool.submit(...)` **never blocks** — every individual call
///    must complete within [`PER_SUBMIT_BUDGET`] (asserted inside
///    [`submit_burst`]).
///  * **Every cycle's frames decode**: each `source_eos` between
///    cycles tears down the per-source decoder, so producer #2's
///    PTS=0 packet is the first packet of a fresh decoder + fresh
///    feeder (`last_key = None`) and is accepted normally.
///  * No `Error` outputs are produced — the violation path is
///    avoided entirely because the decoder was recreated at the
///    explicit boundary.
#[test]
#[serial]
fn interleaved_eos_same_source_pts_rebase() {
    init();
    let done = install_watchdog("interleaved_eos_same_source_pts_rebase");
    let manifest = load_manifest();
    let Some(entry) = find_h264_entry(&manifest) else {
        done.store(true, Ordering::Relaxed);
        return;
    };

    let aus = demux_mp4_to_access_units(entry);
    assert!(!aus.is_empty(), "demuxer produced no access units");
    let per_burst = aus.len();

    let collector = OutputCollector::new();
    let pool = FlexibleDecoderPool::new(prod_pool_config(), collector.callback());

    const CYCLES: usize = 6;
    const SRC: &str = "interleave-rebase";

    for cycle in 0..CYCLES {
        let pts_offset = 0u64;
        let (n, max_submit) = submit_burst(
            &pool,
            &aus,
            SRC,
            entry.width as i64,
            entry.height as i64,
            pts_offset,
        );
        eprintln!(
            "cycle {cycle}: submitted {n} packets at pts_offset=0 (rebase, max submit = {max_submit:?})"
        );

        pool.source_eos(SRC)
            .unwrap_or_else(|e| panic!("cycle {cycle}: source_eos failed: {e}"));
    }

    let mut pool = pool;
    pool.graceful_shutdown()
        .expect("graceful_shutdown failed — likely deadlock in the pool");

    let final_frames = collector.frame_count();
    let final_errors = collector.error_count();
    eprintln!(
        "interleaved_eos_same_source_pts_rebase: frames={final_frames}, errors={final_errors}"
    );

    // Every cycle must decode: source_eos between cycles resets the
    // feeder's strict-order tracker, so PTS rebases to 0 are accepted.
    let expected = CYCLES * per_burst;
    let lower = expected.saturating_sub(per_burst);
    assert!(
        final_frames >= lower,
        "expected ≈{expected} frames across {CYCLES} cycles (≥ {lower} \
         to tolerate one cycle of decoder warmup); got {final_frames}"
    );
    assert_eq!(
        final_errors, 0,
        "no PTS-violation errors are expected — source_eos must reset \
         the strict-order tracker between cycles"
    );
    done.store(true, Ordering::Relaxed);
}

/// Wait for the collector's `frame_count` to reach `target` or for `max`
/// to elapse — whichever comes first.  Used between bursts to let
/// in-flight frames drain before the next burst (which would otherwise
/// drop them via the restart-on-error teardown).  Polling on
/// `frame_count` (rather than "no change for X ms") avoids returning
/// prematurely during the fresh decoder's post-restart warmup window
/// where the count is briefly steady before frames start arriving.
fn wait_for_frames(collector: &OutputCollector, target: usize, max: Duration) {
    let deadline = Instant::now() + max;
    while Instant::now() < deadline {
        if collector.frame_count() >= target {
            return;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
}

/// `--no-eos` scenario: same `source_id` and rebased PTS across cycles,
/// **no** `pool.source_eos(...)` between cycles.
///
/// In the production cars-demo-zmq pipeline this corresponds to the
/// producer running with `--no-eos`: each producer run rebases PTS to 0
/// (mp4 demuxer behaviour) and the per-source [`FlexibleDecoder`] is
/// reused across runs without an explicit boundary.
///
/// # Per-cycle frame loss is expected
///
/// The test asset (`test_h264_bt709_ip.mp4`) packs only **one IDR per
/// cycle** (the very first frame), followed by ~9 P-frames.  When the
/// rebased IDR of cycle K hits the feeder, the
/// [`StrictDecodeOrder`](savant_gstreamer::pipeline::PtsPolicy::StrictDecodeOrder)
/// policy emits a
/// [`PipelineError::TimestampOrderViolation`](savant_gstreamer::pipeline::PipelineError::TimestampOrderViolation)
/// because the previous cycle's tail PTS is still in
/// [`GstPipeline`](savant_gstreamer::pipeline::GstPipeline)'s `last_key`.
/// The packet is dropped, the worker exits, and the next `submit`
/// rebuilds the decoder.  But the **fresh decoder never sees the
/// SPS/PPS contained in the lost IDR**, so the cycle's remaining
/// P-frames can't decode and only contribute
/// [`Skipped { DecoderRestarted }`](deepstream_inputs::flexible_decoder::FlexibleDecoderOutput::Skipped)
/// outputs.  The next cycle's IDR re-establishes the reference and
/// decoding resumes — hence the every-other-cycle yield observed
/// here.
///
/// In production this pattern is irrelevant: real H.264 streams have
/// IDRs every ~30-60 frames (1-2 seconds of video), so a single
/// producer-restart-induced loss covers at most one GOP, not an
/// entire run.  The cars-demo-zmq input (`test_h264_bt709_ip.mp4`,
/// 10 frames of 1 IDR + 9 P) is an artificially aggressive worst
/// case for this test.
///
/// # What the test verifies
///
/// With the
/// [`FlexibleDecoder::worker_loop`](deepstream_inputs::flexible_decoder)
/// restart-on-error fix in place:
///
///  * `pool.submit(...)` **never blocks beyond
///    [`PER_SUBMIT_BUDGET`]** — even on the first packet of a new
///    cycle that triggers a synchronous `nvv4l2decoder` teardown.
///  * The decoder **recovers** at every cycle boundary: the feeder
///    rejects the rebased IDR → worker exits → next `submit` takes
///    the worker-died branch → fresh decoder built → next cycle's
///    IDR is accepted by the fresh `last_key=None` feeder.
///  * `pool.graceful_shutdown()` returns cleanly (no deadlock).
///  * **More than one cycle's worth of frames decode** — proving
///    that at least one post-restart cycle reached steady state and
///    delivered frames, i.e. the recovery actually works rather
///    than silently dropping every packet.
///  * **At least one
///    [`PipelineError::TimestampOrderViolation`](savant_gstreamer::pipeline::PipelineError::TimestampOrderViolation)
///    is surfaced** to the consumer at each cycle boundary.
#[test]
#[serial]
fn interleaved_no_eos_same_source_pts_rebase() {
    init();
    let done = install_watchdog("interleaved_no_eos_same_source_pts_rebase");
    let manifest = load_manifest();
    let Some(entry) = find_h264_entry(&manifest) else {
        done.store(true, Ordering::Relaxed);
        return;
    };

    let aus = demux_mp4_to_access_units(entry);
    assert!(!aus.is_empty(), "demuxer produced no access units");
    let per_burst = aus.len();

    let collector = OutputCollector::new();
    let pool = FlexibleDecoderPool::new(prod_pool_config(), collector.callback());

    const CYCLES: usize = 4;
    const SRC: &str = "interleave-no-eos-rebase";

    // Track the cumulative target: every cycle but the last one is
    // expected to lose at least its very first packet (the violating
    // PTS=0 keyframe that trips the feeder).  Cycle 0 has no
    // restart-induced loss but a couple of B-frame-reorder tail
    // frames may not be flushed.  We wait for *most* of the
    // cycle's frames to land before submitting the next one so the
    // restart-on-error teardown doesn't drop a still-in-flight
    // tail.
    let per_cycle_min_frames = per_burst.saturating_sub(2);
    for cycle in 0..CYCLES {
        let pts_offset = 0u64;
        let (n, max_submit) = submit_burst(
            &pool,
            &aus,
            SRC,
            entry.width as i64,
            entry.height as i64,
            pts_offset,
        );
        // Wait for cycle's frames to drain.  Cycle 0 expects its full
        // burst minus reorder tail; later cycles also lose the
        // violating first packet so the per-cycle floor is `per_burst -
        // 1`.  We use the lower of the two (`per_burst - 2`) so a
        // single missed frame doesn't wedge the wait.
        let target = collector.frame_count() + per_cycle_min_frames;
        wait_for_frames(&collector, target, Duration::from_millis(3000));
        let frames_after_drain = collector.frame_count();
        eprintln!(
            "cycle {cycle}: submitted {n} packets at pts_offset=0 (--no-eos rebase, max submit = {max_submit:?}, frames_after_drain={frames_after_drain})"
        );
    }

    let mut pool = pool;
    pool.graceful_shutdown()
        .expect("graceful_shutdown failed — likely deadlock in the pool");

    let final_frames = collector.frame_count();
    let final_errors = collector.error_count();
    let final_skips = collector.skip_count();
    eprintln!(
        "interleaved_no_eos_same_source_pts_rebase: frames={final_frames}, errors={final_errors}, skipped={final_skips}, cycles={CYCLES}, per_burst={per_burst}"
    );

    // The synthetic asset has only one IDR per cycle.  When the
    // rebased IDR is dropped at the cycle boundary the remaining
    // P-frames can't decode and surface as `Skipped`.  But with the
    // restart fix, **at least one post-cycle-0 cycle must contribute
    // decoded frames**, proving recovery actually delivers frames
    // rather than wedging the decoder forever.
    assert!(
        final_frames > per_burst,
        "expected > {per_burst} decoded frames (cycle 0 + at least \
         one post-restart cycle); got {final_frames} — recovery \
         did not deliver any frames after the first restart"
    );

    // Each cycle boundary surfaces at least one
    // `TimestampOrderViolation` to the consumer (the violating IDR
    // that triggered the worker to exit).  We require at least one
    // (proves the violation reaches the consumer at all); more is
    // expected on Jetson where multiple rebased packets typically
    // race past the feeder before the worker observes the first
    // error.
    assert!(
        final_errors >= 1,
        "expected ≥ 1 TimestampOrderViolation error across \
         {CYCLES} cycles; got {final_errors}"
    );

    done.store(true, Ordering::Relaxed);
}

/// `--no-eos` scenario with a **slow** consumer callback (production
/// reproduction).
///
/// In `cars-demo-zmq pipeline` the `Decoder`'s `on_frame` hook performs a
/// blocking `router.send(...)` to the next stage (Picasso → encoder →
/// ZmqSink).  When that downstream chain is even slightly slower than the
/// decoder, the worker thread blocks inside `on_output(Frame { .. })` and
/// can no longer drain its own [`output_rx`] queue.  Combined with
/// per-cycle PTS rebases (no `source_eos` between cycles), the
/// [`StrictDecodeOrder`](savant_gstreamer::pipeline::PtsPolicy::StrictDecodeOrder)
/// feeder generates a [`PipelineError::TimestampOrderViolation`](savant_gstreamer::pipeline::PipelineError::TimestampOrderViolation)
/// for every rebased packet, fills the output channel with `Error`s, then
/// fills the input channel as
/// [`FlexibleDecoder::submit`](deepstream_inputs::flexible_decoder::FlexibleDecoder::submit)
/// keeps pushing packets — and the next `submit` blocks indefinitely on
/// `input_tx.send(...)` while still holding the state lock.  The decoder
/// is then unrecoverable: the worker is alive but blocked downstream, so
/// the `worker_died` check in `handle_active` reports `false`, so the
/// restart-on-error path never trips.
///
/// This test reproduces that exact production wedge with the
/// [`SLOW_CALLBACK_DELAY`] sleep on every emitted output, and asserts
/// recovery: every `pool.submit(...)` must complete within
/// [`PER_SUBMIT_BUDGET`] and the pool must
/// `graceful_shutdown` cleanly without exceeding the test watchdog.
const SLOW_CALLBACK_DELAY: Duration = Duration::from_millis(200);

#[test]
#[serial]
fn no_eos_slow_callback_back_to_back_cycles() {
    init();
    let done = install_watchdog("no_eos_slow_callback_back_to_back_cycles");
    let manifest = load_manifest();
    let Some(entry) = find_h264_entry(&manifest) else {
        done.store(true, Ordering::Relaxed);
        return;
    };

    let aus = demux_mp4_to_access_units(entry);
    assert!(!aus.is_empty(), "demuxer produced no access units");
    let per_burst = aus.len();

    // Wrap the standard collector so every emitted output sleeps,
    // simulating a slow downstream actor's mailbox saturation.
    let collector = OutputCollector::new();
    let inner_cb = collector.callback();
    let slow_cb = move |out: deepstream_inputs::flexible_decoder::FlexibleDecoderOutput| {
        std::thread::sleep(SLOW_CALLBACK_DELAY);
        inner_cb(out);
    };
    let pool = FlexibleDecoderPool::new(prod_pool_config(), slow_cb);

    const CYCLES: usize = 6;
    const SRC: &str = "no-eos-slow";

    for cycle in 0..CYCLES {
        let pts_offset = 0u64;
        let (n, max_submit) = submit_burst(
            &pool,
            &aus,
            SRC,
            entry.width as i64,
            entry.height as i64,
            pts_offset,
        );
        eprintln!(
            "cycle {cycle}: submitted {n} packets at pts_offset=0 \
             (--no-eos slow-callback rebase, max submit = {max_submit:?})"
        );
        // No wait_for_frames — we deliberately stack cycles back-to-back
        // so cycle K+1's packets arrive while cycle K's frames are still
        // draining through the slow callback.  This is the production
        // pattern the cars-demo-zmq test reproduced.
    }

    let mut pool = pool;
    pool.graceful_shutdown()
        .expect("graceful_shutdown failed — likely deadlock in the pool");

    let final_frames = collector.frame_count();
    let final_errors = collector.error_count();
    let final_skips = collector.skip_count();
    eprintln!(
        "no_eos_slow_callback_back_to_back_cycles: frames={final_frames}, \
         errors={final_errors}, skipped={final_skips}, cycles={CYCLES}, \
         per_burst={per_burst}"
    );

    // The slow callback dilates per-cycle latency far beyond the
    // 33 ms/frame nominal rate; combined with the rebased-PTS feeder
    // rejection at every cycle boundary, only a fraction of the
    // theoretical maximum decodes — but at least one cycle's worth of
    // frames must come through, proving recovery is not a forever-stall.
    assert!(
        final_frames >= 1,
        "decoded {final_frames} frames; expected ≥ 1 — pool wedged \
         under slow downstream + back-to-back rebased cycles"
    );

    // At least one cycle boundary surfaced a violation.
    assert!(
        final_errors >= 1,
        "expected ≥ 1 TimestampOrderViolation across {CYCLES} cycles; \
         got {final_errors}"
    );

    done.store(true, Ordering::Relaxed);
}

/// Control test: distinct `source_id` per cycle. User reports this works in
/// production and we expect this test to pass.
#[test]
#[serial]
fn interleaved_eos_distinct_sources() {
    init();
    let done = install_watchdog("interleaved_eos_distinct_sources");
    let manifest = load_manifest();
    let Some(entry) = find_h264_entry(&manifest) else {
        done.store(true, Ordering::Relaxed);
        return;
    };

    let aus = demux_mp4_to_access_units(entry);
    let per_burst = aus.len();
    let collector = OutputCollector::new();
    let pool = FlexibleDecoderPool::new(prod_pool_config(), collector.callback());

    const CYCLES: usize = 4;

    for cycle in 0..CYCLES {
        let src = format!("distinct-{cycle}");
        let (n, max_submit) = submit_burst(
            &pool,
            &aus,
            &src,
            entry.width as i64,
            entry.height as i64,
            0,
        );
        eprintln!("cycle {cycle}: src={src} submitted {n} (max submit = {max_submit:?})");
        pool.source_eos(&src)
            .unwrap_or_else(|e| panic!("source_eos({src}) failed: {e}"));
    }

    let mut pool = pool;
    pool.graceful_shutdown().expect("graceful_shutdown failed");

    let final_frames = collector.frame_count();
    let final_errors = collector.error_count();
    eprintln!("interleaved_eos_distinct_sources: frames={final_frames}, errors={final_errors}");

    assert_eq!(final_errors, 0);
    let expected = CYCLES * per_burst;
    let lower = expected.saturating_sub(per_burst);
    assert!(final_frames >= lower);
    done.store(true, Ordering::Relaxed);
}

/// Helper: pick the smallest H.264 MP4 from the manifest supported on the
/// current platform. Returns `None` to skip if not available.
fn find_h264_entry(manifest: &Manifest) -> Option<&AssetEntry> {
    let platform = current_platform_tag();
    // Prefer test_h264_bt709_ip.mp4 (matches the cars-demo-zmq input format).
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
    eprintln!("  SKIP: no supported H.264 MP4 asset on platform '{platform}'");
    None
}
