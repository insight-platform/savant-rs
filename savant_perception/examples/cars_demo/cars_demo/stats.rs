//! Sample-level shared counters for the `cars_demo` sample actors.
//!
//! Per-stage frame / object / batch counters are now tracked
//! automatically by the `savant_perception` framework — every
//! `register_actor` slot gets a
//! [`StageStats`](savant_core::pipeline::stats::StageStats)
//! handle that the loop driver bumps on every inbound
//! frame-bearing envelope, and the system-wide
//! [`Stats`](savant_core::pipeline::stats::Stats) collector prints
//! a `📊` line per stage at the configured period (see
//! [`System::stats_period`](savant_perception::System::stats_period)).
//!
//! This module only carries the demo-specific atomics that don't
//! map onto a per-stage counter:
//!
//! * `demux_packets` — incremented by the demuxer's `on_packet`
//!   hook.  Sources don't get an auto-allocated `StageStats`, so
//!   per-source ingress counts live here.
//! * `encoded_bytes` — incremented by the tail's `on_encoded_frame`
//!   hook.  The framework counts encoded *frames* on the tail
//!   actor's loop driver but not their byte sizes; the byte-rate
//!   summary is sample-specific.

use std::sync::atomic::AtomicU64;
use std::sync::Arc;

/// Sample-level shared counters that don't map onto a per-stage
/// processing counter in
/// [`savant_core::pipeline::stats::Stats`].
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
