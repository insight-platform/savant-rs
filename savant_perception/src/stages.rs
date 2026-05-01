//! Layer B — typed actor stages.
//!
//! Where [the core runtime](super) provides the generic trait
//! contract — [`Actor`](super::Actor), [`Source`](super::Source),
//! [`Handler<V>`](super::Handler), [`ActorBuilder`](super::ActorBuilder)
//! — the **stages** here provide opinionated, high-level
//! constructors for concrete stage kinds used by video analytics
//! applications.
//!
//! Each stage exposes a fluent builder whose terminal call
//! returns a Layer-A
//! [`ActorBuilder<_>`](super::ActorBuilder) (or
//! [`SourceBuilder<_>`](super::SourceBuilder)) that the user
//! then hands to
//! [`System::register_actor`](super::System::register_actor) or
//! [`System::register_source`](super::System::register_source).
//!
//! # Philosophy
//!
//! * **Low noise** — the user provides only a name, capacity, and
//!   domain-specific hooks (closures or trait impls).  Everything
//!   else has a sensible default.
//! * **Progressive disclosure** — for the common case, a one-liner
//!   suffices; for the less-common case, any hook the stage
//!   provides can be overridden.
//! * **No leaking abstractions** — the user never touches
//!   [`Addr`](super::Addr), [`Registry`](super::Registry), or
//!   [`Context`](super::Context) unless they explicitly want to
//!   (via a custom hook).
//!
//! See [`deepstream_function`] for a minimal, dependency-free stage
//! that the same patterns are layered onto in heavier stages
//! (e.g. [`decoder`], [`nvinfer`], [`mp4_muxer`]).

pub mod bitstream_function;
#[cfg(feature = "deepstream")]
pub mod decoder;
pub mod demuxers;
#[cfg(feature = "deepstream")]
pub mod deepstream_function;
pub mod mp4_demuxer;
pub mod mp4_muxer;
#[cfg(feature = "deepstream")]
pub mod nvinfer;
#[cfg(feature = "deepstream")]
pub mod nvtracker;
#[cfg(feature = "deepstream")]
pub mod picasso;
#[cfg(feature = "deepstream")]
pub mod sorter;
pub mod uri_demuxer;
pub mod zmq_sink;
pub mod zmq_source;

pub use demuxers::demux_input::{
    looped_requester, one_shot_requester, DemuxInputRequest, InputRequester,
};

pub use bitstream_function::{
    BitstreamFunction, BitstreamFunctionBuilder, BitstreamFunctionCommon,
    BitstreamFunctionCommonBuilder, BitstreamFunctionInbox, BitstreamFunctionInboxBuilder,
    OnStartedHook as BitstreamFunctionOnStartedHook,
    OnStoppingHook as BitstreamFunctionOnStoppingHook,
};
#[cfg(feature = "deepstream")]
pub use decoder::{
    Decoder, DecoderBuilder, DecoderCommon, DecoderCommonBuilder, DecoderInbox,
    DecoderInboxBuilder, DecoderResults, DecoderResultsBuilder,
    OnStoppingHook as DecoderOnStoppingHook,
};
#[cfg(feature = "deepstream")]
pub use deepstream_function::{
    DeepStreamFunction, DeepStreamFunctionBuilder, DeepStreamFunctionCommon,
    DeepStreamFunctionCommonBuilder, DeepStreamFunctionInbox, DeepStreamFunctionInboxBuilder,
    OnStartedHook as DeepStreamFunctionOnStartedHook,
    OnStoppingHook as DeepStreamFunctionOnStoppingHook,
};
pub use mp4_demuxer::{
    Mp4DemuxerBuilder, Mp4DemuxerCommon, Mp4DemuxerCommonBuilder, Mp4DemuxerResults,
    Mp4DemuxerResultsBuilder, Mp4DemuxerSource, OnStoppingHook as Mp4DemuxerOnStoppingHook,
};
pub use mp4_muxer::{
    Mp4Muxer, Mp4MuxerBuilder, Mp4MuxerCommon, Mp4MuxerCommonBuilder, Mp4MuxerInbox,
    Mp4MuxerInboxBuilder, OnStoppingHook as Mp4MuxerOnStoppingHook,
};
#[cfg(feature = "deepstream")]
pub use nvinfer::{
    NvInfer, NvInferBuilder, NvInferCommon, NvInferCommonBuilder, NvInferOperatorFactory,
    NvInferResults, NvInferResultsBuilder, OnErrorHook as NvInferOnErrorHook, OnInferenceHook,
    OnSourceEosHook as NvInferOnSourceEosHook, OnStoppingHook as NvInferOnStoppingHook,
};
#[cfg(feature = "deepstream")]
pub use nvtracker::{
    NvTracker, NvTrackerBuilder, NvTrackerCommon, NvTrackerCommonBuilder, NvTrackerHookCtx,
    NvTrackerOperatorFactory, NvTrackerResults, NvTrackerResultsBuilder,
    OnErrorHook as NvTrackerOnErrorHook, OnSourceEosHook as NvTrackerOnSourceEosHook,
    OnStoppingHook as NvTrackerOnStoppingHook, OnTrackingHook,
};
#[cfg(feature = "deepstream")]
pub use picasso::{
    OnCropSelectHook, OnDeliveryHook as PicassoOnDeliveryHook, OnEncodedFrameHook,
    OnEncodedSourceEosHook, OnEvictionHook, OnForwardInboxDeliveryHook,
    OnForwardInboxSourceEosHook, OnGpuMatHook, OnObjectDrawSpecHook, OnRenderHook,
    OnStoppingHook as PicassoOnStoppingHook, OnStreamResetHook, Picasso, PicassoBuilder,
    PicassoCommon, PicassoCommonBuilder, PicassoEngineFactory, PicassoEngineHandle, PicassoInbox,
    PicassoInboxBuilder, PicassoResults, PicassoResultsBuilder, SourceSpecFactory,
};
#[cfg(feature = "deepstream")]
pub use sorter::{
    OnMessageHook as SorterOnMessageHook, OnSourceEosHook as SorterOnSourceEosHook,
    OnStoppingHook as SorterOnStoppingHook, OnUnregisteredHook as SorterOnUnregisteredHook,
    Sorter, SorterBuilder, SorterCommon, SorterCommonBuilder, SorterInbox, SorterInboxBuilder,
    SorterRegistration, SorterResults, SorterResultsBuilder,
};
pub use uri_demuxer::{
    OnStoppingHook as UriDemuxerOnStoppingHook, UriDemuxerBuilder, UriDemuxerCommon,
    UriDemuxerCommonBuilder, UriDemuxerResults, UriDemuxerResultsBuilder, UriDemuxerSource,
};
pub use zmq_sink::{
    OnFrameHook as ZmqSinkOnFrameHook, OnSendErrorHook as ZmqSinkOnSendErrorHook,
    OnSourceEosHook as ZmqSinkOnSourceEosHook, OnStoppingHook as ZmqSinkOnStoppingHook,
    OnWriterResultHook as ZmqSinkOnWriterResultHook, PayloadCarrier,
    TopicStrategy as ZmqSinkTopicStrategy, ZmqSink, ZmqSinkBuilder, ZmqSinkCommon,
    ZmqSinkCommonBuilder, ZmqSinkErrors, ZmqSinkErrorsBuilder, ZmqSinkInbox, ZmqSinkInboxBuilder,
};
pub use zmq_source::{
    OnMessageHook as ZmqSourceOnMessageHook, OnProtocolErrorHook as ZmqSourceOnProtocolErrorHook,
    OnReceiveErrorHook as ZmqSourceOnReceiveErrorHook,
    OnServiceMessageHook as ZmqSourceOnServiceMessageHook,
    OnSourceEosHook as ZmqSourceOnSourceEosHook, OnStoppingHook as ZmqSourceOnStoppingHook,
    ZmqSource, ZmqSourceBuilder, ZmqSourceCommon, ZmqSourceCommonBuilder, ZmqSourceResults,
    ZmqSourceResultsBuilder,
};
