//! Layer B — typed actor templates.
//!
//! Where [the core runtime](super) provides the generic trait
//! contract — [`Actor`](super::Actor), [`Source`](super::Source),
//! [`Handler<V>`](super::Handler), [`ActorBuilder`](super::ActorBuilder)
//! — the **templates** here provide opinionated, high-level
//! constructors for the concrete stage kinds that the
//! `cars_tracking` sample (and future samples) actually needs.
//!
//! Each template exposes a fluent builder whose terminal call
//! returns a Layer-A `ActorBuilder<_>` (or `SourceBuilder<_>`) that
//! the user then hands to `System::register_actor` /
//! `System::register_source`.
//!
//! # Philosophy
//!
//! * **Low noise** — the user provides only a name, capacity, and
//!   domain-specific hooks (closures or trait impls).  Everything
//!   else has a sensible default.
//! * **Progressive disclosure** — for the common case, a one-liner
//!   suffices; for the less-common case, any hook the template
//!   provides can be overridden.
//! * **No leaking abstractions** — the user never touches
//!   [`Addr`](super::Addr), [`Registry`](super::Registry), or
//!   [`Context`](super::Context) unless they explicitly want to
//!   (via a custom hook).
//!
//! See [`sink`] for the canonical example.

pub mod decoder;
pub mod mp4_demuxer;
pub mod mp4_muxer;
pub mod nvinfer;
pub mod nvtracker;
pub mod picasso;
pub mod sink;

pub use decoder::{Decoder, DecoderBuilder};
pub use mp4_demuxer::{ErrorAction, Mp4DemuxerBuilder, Mp4DemuxerSource};
pub use mp4_muxer::{Mp4Muxer, Mp4MuxerBuilder};
pub use nvinfer::{
    NvInfer, NvInferBuilder, NvInferOperatorFactory, OnErrorHook as NvInferOnErrorHook,
    OnInferenceHook, OnSourceEosHook as NvInferOnSourceEosHook,
};
pub use nvtracker::{
    NvTracker, NvTrackerBuilder, NvTrackerOperatorFactory, OnErrorHook as NvTrackerOnErrorHook,
    OnSourceEosHook as NvTrackerOnSourceEosHook, OnTrackingHook,
};
pub use picasso::{
    FramePreprocessor, Picasso, PicassoBuilder, PicassoEngineFactory, SourceSpecFactory,
};
pub use sink::{Sink, SinkBuilder};
