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
//! See [`function`] for the canonical example.

pub mod decoder;
pub mod function;
pub mod mp4_demuxer;
pub mod mp4_muxer;
pub mod nvinfer;
pub mod nvtracker;
pub mod picasso;

pub use decoder::{
    Decoder, DecoderBuilder, DecoderCommon, DecoderCommonBuilder, DecoderInbox,
    DecoderInboxBuilder, DecoderResults, DecoderResultsBuilder,
    OnStoppingHook as DecoderOnStoppingHook,
};
pub use function::{
    Function, FunctionBuilder, FunctionCommon, FunctionCommonBuilder, FunctionInbox,
    FunctionInboxBuilder, OnStartedHook as FunctionOnStartedHook,
    OnStoppingHook as FunctionOnStoppingHook,
};
pub use mp4_demuxer::{
    Mp4DemuxerBuilder, Mp4DemuxerCommon, Mp4DemuxerCommonBuilder, Mp4DemuxerResults,
    Mp4DemuxerResultsBuilder, Mp4DemuxerSource, OnStoppingHook as Mp4DemuxerOnStoppingHook,
};
pub use mp4_muxer::{
    Mp4Muxer, Mp4MuxerBuilder, Mp4MuxerCommon, Mp4MuxerCommonBuilder, Mp4MuxerInbox,
    Mp4MuxerInboxBuilder, OnStoppingHook as Mp4MuxerOnStoppingHook,
};
pub use nvinfer::{
    NvInfer, NvInferBuilder, NvInferCommon, NvInferCommonBuilder, NvInferOperatorFactory,
    NvInferResults, NvInferResultsBuilder, OnErrorHook as NvInferOnErrorHook, OnInferenceHook,
    OnSourceEosHook as NvInferOnSourceEosHook, OnStoppingHook as NvInferOnStoppingHook,
};
pub use nvtracker::{
    NvTracker, NvTrackerBuilder, NvTrackerCommon, NvTrackerCommonBuilder, NvTrackerHookCtx,
    NvTrackerOperatorFactory, NvTrackerResults, NvTrackerResultsBuilder,
    OnErrorHook as NvTrackerOnErrorHook, OnSourceEosHook as NvTrackerOnSourceEosHook,
    OnStoppingHook as NvTrackerOnStoppingHook, OnTrackingHook,
};
pub use picasso::{
    DeliveryHook as PicassoDeliveryHook, OnStoppingHook as PicassoOnStoppingHook, Picasso,
    PicassoBuilder, PicassoCommon, PicassoCommonBuilder, PicassoEngineFactory, PicassoInbox,
    PicassoInboxBuilder, SourceSpecFactory, SrcRectProvider,
};
