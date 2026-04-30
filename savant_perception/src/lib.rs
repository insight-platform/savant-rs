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
//!   send handle stages expose to their callbacks: it wraps an
//!   optional default [`OperatorSink`] (configured via each
//!   builder's `.downstream(...)`) and adds cached name-routing via
//!   `router.send_to(&peer, msg)`.
//! * **Layer B — typed actor stages** ([`stages`]) wrapping
//!   the raw Layer A primitives with domain-specific hooks and
//!   sensible defaults so user code stays short and
//!   intent-revealing.
//!
//! End-to-end applications can be built by composing Layer-B
//! stages with application-owned envelopes, hooks, and shutdown
//! policy.
//!
//! The framework is **synchronous**: every actor owns an OS
//! thread and a bounded [`crossbeam::channel`] inbox.  There is
//! no async runtime — the heavy-lifting integrations exposed
//! through Layer B (GStreamer, the NVIDIA hardware codecs,
//! nvinfer, NvDCF, Picasso/Skia) are all blocking APIs and are
//! a more natural fit for blocking threads than for an async
//! executor.
//!
//! # Streaming contracts
//!
//! Layer A defines a small set of contracts that every Layer-B
//! stage observes:
//!
//! 1. **Backpressure via bounded channels.**  Every actor inbox
//!    is a [`crossbeam::channel::bounded`] of small capacity, so
//!    a slow consumer naturally blocks its upstream producer.
//! 2. **In-band EOS propagation.**  End-of-source is an in-band
//!    message (each pipeline envelope has a
//!    `SourceEos { source_id }` variant), **not** channel
//!    closure.  Stages that maintain per-source state observe
//!    each EOS in order and forward an equivalent sentinel
//!    downstream so the drain sequence is preserved.
//! 3. **Cooperative shutdown.**  The supervisor reacts to a
//!    [`ShutdownCause`] (Ctrl+C or a stage exit) by invoking the
//!    installed [`ShutdownHandler`] and, on a broadcast action,
//!    sending each actor a `Shutdown { grace, reason }` envelope
//!    via [`Envelope::build_shutdown`].
//! 4. **Per-thread state, not interior mutability.**  Actors own
//!    their own state by `&mut self` for the lifetime of their
//!    thread; cross-actor state is published through
//!    [`SharedStore`] before [`System::run`] starts.

pub mod actor;
pub mod addr;
pub mod builder;
pub mod context;
pub mod envelope;
pub mod envelopes;
pub mod errors;
pub mod handler;
pub mod loop_driver;
pub mod message_ex;
pub mod messages;
pub mod operator_sink;
pub mod registry;
pub mod router;
pub mod shared;
pub mod shutdown;
pub mod stages;
pub mod supervisor;
pub mod system;

pub use actor::{Actor, Source};
pub use addr::Addr;
pub use builder::{ActorBuilder, SourceBuilder};
pub use context::{BuildCtx, Context, HookCtx, SourceContext};
pub use envelope::{Dispatch, Envelope, ShutdownHint};
pub use errors::ErrorAction;
pub use handler::{Flow, Handler};
pub use loop_driver::run_actor;
pub use message_ex::MessageExPayload;
pub use messages::{
    RemoveSourcePayload, ResetStreamPayload, ShutdownPayload, SourceEosPayload,
};
#[cfg(feature = "deepstream")]
pub use messages::UpdateSourceSpecPayload;
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
