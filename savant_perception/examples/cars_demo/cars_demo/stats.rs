//! Shared per-stage stat helpers for the `cars_demo` sample actors.
//!
//! The savant_core pipeline [`Stats`](savant_core::pipeline::stats::Stats)
//! aggregates per-stage counters through [`StageStats`] handles.  Every
//! actor in this sample ticks its own [`StageStats`] on each successfully
//! processed frame; the helpers here are the minimal carrier primitives
//! each actor uses (construct, tick, read).
//!
//! Actor modules (demux / decode / infer / tracker / picasso / mux)
//! own their own behavior; this module owns nothing beyond stat
//! bookkeeping infrastructure.

use parking_lot::Mutex as PlMutex;
use savant_core::pipeline::stats::{StageLatencyStat, StageProcessingStat, StageStats};
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

/// Sample-level shared counters that don't map onto a per-stage
/// processing counter in [`savant_core::pipeline::stats::Stats`].
///
/// Per-stage frame / object counts live inside [`StageStats`] instances
/// registered with [`savant_core::pipeline::stats::Stats`] (see
/// [`crate::cars_demo::pipeline::run`]).  This struct only carries
/// the counters whose semantics are specific to this sample: demuxed
/// access-unit counts and encoded-byte totals.
#[derive(Debug, Default)]
pub struct PipelineStats {
    /// Demuxed access-unit count.
    pub demux_packets: AtomicU64,
    /// Total encoded bytes written by the muxer.
    pub encoded_bytes: AtomicU64,
}

impl PipelineStats {
    /// Allocate a fresh [`PipelineStats`] behind an [`Arc`] so every
    /// stage thread and the orchestrator can share it.
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }
}

/// Build a fresh [`StageStats`] handle bound to `name`.
pub fn make_stage(name: &str) -> StageStats {
    Arc::new(PlMutex::new((
        StageProcessingStat::new(name.to_string()),
        StageLatencyStat::new(name.to_string()),
    )))
}

/// Increment the `frame_counter` on a stage, optionally adding to the
/// `object_counter`.  Called on every successfully processed frame.
#[inline]
pub fn tick_stage(stage: &StageStats, frames: usize, objects: usize) {
    let mut guard = stage.lock();
    guard.0.frame_counter += frames;
    guard.0.object_counter += objects;
}

/// Read the total `frame_counter` value from a [`StageStats`].
#[inline]
pub fn stage_frames(stage: &StageStats) -> usize {
    stage.lock().0.frame_counter
}

/// Read the total `object_counter` value from a [`StageStats`].
#[inline]
pub fn stage_objects(stage: &StageStats) -> usize {
    stage.lock().0.object_counter
}
