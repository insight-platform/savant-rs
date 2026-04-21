//! Blackhole actor — the pipeline terminus used when Picasso is
//! excluded via `--no-picasso`.
//!
//! The actor unseals every tracker delivery and immediately drops the
//! resulting `(VideoFrameProxy, SharedBuffer)` pairs.  Dropping the
//! `SharedBuffer` releases the GPU slot back to the decoder pool,
//! which is what keeps the pipeline flowing; without this sink the
//! tracker output channel would fill up and stall the inference /
//! decode stages.
//!
//! The blackhole also drives the pipeline-tail metrics that would
//! otherwise be driven by Picasso's
//! [`OnEncodedFrame`](picasso::prelude::OnEncodedFrame) callback:
//!
//! * [`Stats::register_frame`] on `core_stats` — the sole source for
//!   the `📊 Time-based FPS counter triggered: ...` log line; without
//!   this call FPS stays at zero with `--no-picasso`.
//! * `tail_stage` frame counter — keeps `📊 <stage>` rows
//!   symmetrical with Picasso mode.

use anyhow::{Context, Result};
use crossbeam::channel::RecvTimeoutError;
use hashbrown::HashSet;
use savant_core::pipeline::stats::{StageStats, Stats};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::decoder::handle_shutdown;
use super::tracker::TrackerResultReceiver;
use crate::cars_tracking::message::PipelineMsg;
use crate::cars_tracking::stats::tick_stage;
use crate::cars_tracking::supervisor::{ExitSender, StageExitGuard, StageName};

/// Receive poll timeout.  Drives the grace-deadline check and keeps
/// the loop responsive while there is no upstream traffic.
const BLACKHOLE_RECV_POLL: Duration = Duration::from_millis(100);

/// Spawn the blackhole actor — used when Picasso is excluded via
/// `--no-picasso`.
pub fn spawn_blackhole_thread(
    rx: TrackerResultReceiver,
    tail_stage: StageStats,
    core_stats: Arc<Stats>,
    exit_tx: ExitSender,
    name: StageName,
) -> Result<JoinHandle<Result<()>>> {
    thread::Builder::new()
        .name("cars-blackhole".into())
        .spawn(move || {
            let _exit_guard = StageExitGuard::new(name, exit_tx);
            blackhole_thread(rx, tail_stage, core_stats)
        })
        .context("spawn blackhole thread")
}

fn blackhole_thread(
    rx: TrackerResultReceiver,
    tail_stage: StageStats,
    core_stats: Arc<Stats>,
) -> Result<()> {
    log::info!("[blackhole] starting (picasso excluded) — acting as pipeline terminus");
    let mut source_ids_seen: HashSet<String> = HashSet::new();
    let mut drained: u64 = 0;
    let mut deadline: Option<Instant> = None;
    let mut break_now = false;
    // Termination contract:
    //
    // * The blackhole actor is the **terminus** of the
    //   `--no-picasso` pipeline — the symmetric counterpart to the
    //   muxer in Picasso mode.  The orchestrator joins this thread
    //   first (via `PipelineMsg::SourceEos` the loop exits naturally
    //   because the upstream channel eventually disconnects), then
    //   broadcasts `PipelineMsg::Shutdown` to every upstream stage.
    // * [`PipelineMsg::Shutdown`] — cooperative exit, honoured with
    //   the usual grace semantics.
    // * `recv_timeout(Disconnected)` — upstream dropped its sender
    //   after consuming the `Shutdown` sentinel; normal completion.
    loop {
        match rx.recv_timeout(BLACKHOLE_RECV_POLL) {
            Ok(msg @ (PipelineMsg::Delivery(_) | PipelineMsg::Deliveries(_))) => {
                // Generalized ingress: both delivery shapes are
                // normalized via [`PipelineMsg::into_pairs`].  The
                // blackhole stage simply accounts for each pair at
                // the pipeline tail and drops the buffers to release
                // GPU slots back to the decoder pool.
                let pairs = msg.into_pairs();
                for (frame, _) in pairs.iter() {
                    let sid = frame.get_source_id();
                    if source_ids_seen.insert(sid.clone()) {
                        log::info!("[blackhole] first frame for source_id={sid}");
                    }
                }
                let count = pairs.len();
                // Record the frames as they reach the pipeline tail
                // *before* dropping — this keeps the FPS counter
                // consistent even if the drop below is inlined into
                // a long-running destructor (e.g. GPU slot cleanup).
                for _ in 0..count {
                    tick_stage(&tail_stage, 1, 0);
                    core_stats.register_frame(0);
                }
                drained = drained.saturating_add(count as u64);
                // Dropping the `SharedBuffer`s releases GPU slots
                // back to the decoder pool — without this drop the
                // tracker output channel would fill up and the
                // pipeline would stall one channel-cap in.
                drop(pairs);
            }
            Ok(PipelineMsg::SourceEos { source_id }) => {
                // Terminus-side end-of-source signal.  In this
                // single-source sample the first `SourceEos` is also
                // the last — every frame has traversed the pipeline
                // end-to-end by the time we observe it.  Break so
                // the orchestrator can join us and then broadcast
                // `Shutdown` to every upstream stage.
                log::info!("[blackhole] terminus SourceEos {source_id}: exiting");
                break;
            }
            Ok(PipelineMsg::Shutdown { grace, reason }) => {
                handle_shutdown("blackhole", grace, &reason, &mut deadline, &mut break_now);
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                log::info!("[blackhole] upstream channel disconnected; exiting receive loop");
                break;
            }
        }
        // Per-iteration exit checks (see decoder.rs for rationale —
        // the grace deadline must not be gated on the `Timeout`
        // branch, otherwise a chatty upstream would keep us alive
        // past the deadline).
        if break_now {
            break;
        }
        if let Some(d) = deadline {
            if Instant::now() >= d {
                log::info!("[blackhole] grace deadline expired; exiting receive loop");
                break;
            }
        }
    }
    log::info!("[blackhole] finished drained_frames={drained}");
    Ok(())
}
