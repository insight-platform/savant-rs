//! `savant_perception` — synchronous actor framework for real-time
//! video-analytics pipelines.
//!
//! The crate is split in two layers that compose top-down:
//!
//! * **Layer A — core runtime primitives** (this module and its
//!   direct submodules).  Traits ([`Actor`], [`Handler`],
//!   [`Envelope`], [`Dispatch`], [`Source`]) plus the supporting
//!   runtime types ([`Addr`], [`Registry`], [`SharedStore`],
//!   [`Context`], [`BuildCtx`], [`OperatorSink`], [`Router`],
//!   [`ActorBuilder`], [`SourceBuilder`]) used to construct actors
//!   and wire up message dispatch.  [`Router`] is the high-level
//!   send handle templates expose to their callbacks: it wraps an
//!   optional default [`OperatorSink`] (configured via each
//!   builder's `.downstream(...)`) and adds cached name-routing via
//!   `router.send_to(&peer, msg)`.
//! * **Layer B — typed actor templates** ([`templates`]) wrapping
//!   the raw Layer A primitives with domain-specific hooks and
//!   sensible defaults so user code stays short and
//!   intent-revealing.
//!
//! End-to-end reference pipelines live under
//! [`examples/`](https://github.com/insight-platform/savant-rs/tree/main/savant_perception/examples)
//! — see the `cars-demo` example for a full decode → infer → track →
//! render → encode wiring.  Run it with
//! `cargo run -p savant-perception-framework --example cars-demo`.
//!
//! The framework is **synchronous**: every actor owns an OS thread
//! and a bounded [`crossbeam::channel`] inbox.  No async runtime,
//! no `tokio` — the heavy-lifting integrations (GStreamer,
//! NVDEC/NVENC, nvinfer, NvDCF, Picasso/Skia) are all blocking
//! APIs that would fight an async executor.
//!
//! # Streaming invariants (shared by every sample)
//!
//! 1. **No full-file materialisation.**  Decoder backed by a
//!    fixed-size NVMM pool; muxer writes each encoded access unit
//!    directly to `filesink`.
//! 2. **Backpressure via bounded channels.**  All stage boundaries
//!    use [`crossbeam::channel::bounded`] with a small capacity.
//!    A slow downstream stage blocks its upstream producer.
//! 3. **In-band EOS propagation.**  End-of-source is an in-band
//!    message (each inter-actor channel's enum has a
//!    `SourceEos { source_id }` variant) — **not** channel closure.
//! 4. **Per-frame ownership.**  Each frame traverses the pipeline
//!    as a single `(VideoFrameProxy, SharedBuffer)` tuple packaged
//!    into a sealed delivery; clones are dropped as soon as the
//!    next stage has consumed them so NVMM slots return to the
//!    pool.
//! 5. **Unseal outside callbacks.**  Operator result callbacks
//!    never call `unseal()`; they forward sealed deliveries
//!    through a bounded channel so a dedicated consumer thread
//!    unseals and submits to the next stage.

pub mod actor;
pub mod addr;
pub mod builder;
pub mod context;
pub mod envelope;
pub mod envelopes;
pub mod errors;
pub mod handler;
pub mod loop_driver;
pub mod messages;
pub mod operator_sink;
pub mod registry;
pub mod router;
pub mod shared;
pub mod shutdown;
pub mod supervisor;
pub mod system;
pub mod templates;

pub use actor::{Actor, Source};
pub use addr::Addr;
pub use builder::{ActorBuilder, SourceBuilder};
pub use context::{BuildCtx, Context, HookCtx, SourceContext};
pub use envelope::{Dispatch, Envelope, ShutdownHint};
pub use errors::ErrorAction;
pub use handler::{Flow, Handler};
pub use loop_driver::run_actor;
pub use messages::{
    RemoveSourcePayload, ResetStreamPayload, ShutdownPayload, SourceEosPayload,
    UpdateSourceSpecPayload,
};
pub use operator_sink::OperatorSink;
pub use registry::Registry;
pub use router::Router;
pub use shared::SharedStore;
pub use shutdown::{
    DefaultShutdownHandler, ShutdownAction, ShutdownCause, ShutdownCtx, ShutdownHandler,
    ShutdownWave,
};
pub use supervisor::{ExitReceiver, ExitSender, StageExit, StageExitGuard, StageKind, StageName};
pub use system::{System, SystemReport, DEFAULT_QUIESCENCE_GRACE};

/// Framework-level traits for actors and message sources, grouped
/// for convenient `use` imports.
pub mod traits {
    pub use super::actor::{Actor, Source};
    pub use super::envelope::{Dispatch, Envelope};
    pub use super::handler::{Flow, Handler};
}
