//! Actor framework for the cars-tracking sample.
//!
//! Two layers stacked below the sample's actor modules:
//!
//! * **Layer A — core runtime primitives** (this module and its
//!   direct submodules).  Traits ([`Actor`], [`Handler`],
//!   [`Envelope`], [`Dispatch`], [`Source`]) plus the supporting
//!   runtime types ([`Addr`], [`Registry`], [`SharedStore`],
//!   [`Context`], [`BuildCtx`], [`OperatorSink`], [`Router`],
//!   [`ActorBuilder`], [`SourceBuilder`]) used to construct actors
//!   and wire up message dispatch.  [`Router`] is the high-level send
//!   handle that templates expose to their callbacks: it wraps an
//!   optional default [`OperatorSink`] (configured via each
//!   builder's `.downstream(...)`) and adds cached name-routing via
//!   `router.send_to(&peer, msg)`.
//! * **Layer B — typed actor templates** (submodules under
//!   [`actors`], added as each template lands) that wrap the raw
//!   Layer A primitives with domain-specific hooks and sensible
//!   defaults so user code stays short and intent-revealing.
//!
//! The framework is **synchronous**: every actor owns an OS thread
//! and a bounded [`crossbeam::channel`] inbox, mirroring the sample's
//! pre-existing threading model.  No async runtime, no `tokio` — the
//! heavy-lifting integrations (GStreamer, NVDEC/NVENC, nvinfer,
//! NvDCF, Picasso/Skia) are all blocking APIs that would fight an
//! async executor.
//!
//! The design documents under `savant_samples/assets/design/` remain
//! the authoritative reference for the framework's contracts and
//! invariants; rustdoc here summarises the public surface each
//! module ships.

pub mod actor;
pub mod addr;
pub mod builder;
pub mod context;
pub mod envelope;
pub mod envelopes;
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
pub use context::{BuildCtx, Context, SourceContext};
pub use envelope::{Dispatch, Envelope, ShutdownHint};
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
