//! [`Decoder`] — `EncodedMsg` → decoded-frame fan-out stage.
//!
//! A direct replacement for the handwritten decode thread in
//! `cars_tracking/pipeline/decoder.rs`.  The template:
//!
//! * Builds a [`FlexibleDecoderPool`] (one slot per `source_id` it
//!   sees) with user-supplied
//!   [`FlexibleDecoderPoolConfig`](FlexibleDecoderPoolConfig).
//! * Captures a [`Router<PipelineMsg>`] at build time (optionally
//!   with a default peer installed via
//!   [`DecoderBuilder::downstream`]) and hands it to every
//!   user-supplied variant hook below.  The template itself never
//!   calls `router.send`; user code is the single source of sends
//!   for this stage, which makes source-id-based routing (e.g.
//!   between multiple inference engines via
//!   `router.send_to(&peer, msg)`) natural.
//! * Is **stateless w.r.t. [`VideoInfo`]** — every
//!   [`EncodedMsg::Packet`](crate::framework::envelopes::EncodedMsg::Packet)
//!   carries its own stream-level
//!   [`VideoInfo`](savant_gstreamer::mp4_demuxer::VideoInfo) in-band
//!   (stamped upstream by the
//!   [`Mp4DemuxerSource`](super::mp4_demuxer::Mp4DemuxerSource)
//!   hooks), so the template constructs the decode frame on the fly
//!   without maintaining a per-`source_id` cache.  The standalone
//!   [`EncodedMsg::StreamInfo`](crate::framework::envelopes::EncodedMsg::StreamInfo)
//!   sentinel is still accepted and exposed to the user through the
//!   [`StreamInfoObserver`] hook for stream-header-level
//!   observability (stats, logging, domain signalling) but is not
//!   required for decoding.
//! * On
//!   [`EncodedMsg::SourceEos`](crate::framework::envelopes::EncodedMsg::SourceEos)
//!   kicks the per-source drain via
//!   [`FlexibleDecoderPool::source_eos`]; the downstream
//!   `on_source_eos` hook is invoked strictly after the last
//!   `on_frame` on the output callback thread — preserving
//!   stream-aligned semantics.
//! * Calls [`FlexibleDecoderPool::flush_idle`] on every idle poll
//!   ([`Actor::on_tick`]) and
//!   [`FlexibleDecoderPool::graceful_shutdown`] from
//!   [`Actor::stopping`].
//!
//! The shutdown path (in-band
//! [`EncodedMsg::Shutdown`](crate::framework::envelopes::EncodedMsg::Shutdown))
//! is handled by the loop driver via
//! [`Envelope::as_shutdown`](super::super::Envelope::as_shutdown);
//! the template adds no cooperative-exit code of its own.
//!
//! # Variant hooks
//!
//! Every [`FlexibleDecoderOutput`] variant is surfaced through a
//! user hook that receives the variant payload plus a
//! `&Router<PipelineMsg>`.  User code is the sole source of
//! `router.send(...)` / `router.send_to(...)` calls for this stage.
//!
//! # Default forwarders
//!
//! Every per-variant hook has an out-of-the-box default — if the
//! matching builder setter is not called,
//! [`build`](DecoderBuilder::build) auto-installs it:
//!
//! * [`Decoder::default_on_stream_info`] — no-op observer
//!   (stream info is already cached and logged by the template
//!   itself).
//! * [`Decoder::default_on_frame`] — forwards
//!   [`PipelineMsg::Delivery`](crate::framework::envelopes::PipelineMsg::Delivery)
//!   via `router.send(..)`.
//! * [`Decoder::default_on_source_eos`] — forwards
//!   [`PipelineMsg::SourceEos`](crate::framework::envelopes::PipelineMsg::SourceEos).
//! * [`Decoder::default_on_parameter_change`] — logs the
//!   old/new [`DecoderParameters`] at `info` level, prefixed by the
//!   stage name.
//! * [`Decoder::default_on_skipped`] — logs the
//!   [`SkipReason`] at `debug` level.
//! * [`Decoder::default_on_orphan_frame`] — logs at `debug`
//!   level.
//! * [`Decoder::default_on_decoder_event`] — no-op (GStreamer
//!   events without domain meaning are silently ignored).
//! * [`Decoder::default_on_decoder_error`] — logs at `error`
//!   level, prefixed by the stage name.
//!
//! Override any branch to add stage-specific processing (e.g.
//! detection decoding on `on_frame`, source-id-based routing via
//! `router.send_to(..)`, metrics on `on_skipped`).  The builder is
//! additive and each setter replaces exactly one branch's
//! behaviour.
//!
//! # Runtime invariant
//!
//! [`Decoder`] stores its variant hooks as non-`Option`
//! closures ([`StreamInfoObserver`] for the actor-thread observer,
//! `Arc<dyn Fn>` for the pool-callback-thread hooks).  The
//! builder's `Option<...>` fields exist purely as an internal "was
//! the setter called?" marker;
//! [`DecoderBuilder::build`] always substitutes the matching
//! `Decoder::default_on_*` before the actor value is
//! constructed.  There is therefore no runtime code path where a
//! [`FlexibleDecoderOutput`] variant is dispatched to `None` —
//! every branch is always handled, whether by a user-supplied
//! closure or by the auto-installed default.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use deepstream_inputs::flexible_decoder::DecoderConfigCallback;
use deepstream_inputs::prelude::{
    DecoderParameters, EvictionDecision, FlexibleDecoderOutput, FlexibleDecoderPool,
    FlexibleDecoderPoolConfig, SealedDelivery, SkipReason,
};
use gstreamer as gst;
use savant_core::primitives::frame::{
    VideoFrameContent, VideoFrameProxy, VideoFrameTranscodingMethod,
};
use savant_gstreamer::mp4_demuxer::{DemuxedPacket, VideoInfo};

pub use deepstream_decoders::DecoderError;
pub use deepstream_inputs::flexible_decoder::DecodedFrame;

use crate::framework::envelopes::{
    EncodedMsg, FramePayload, PacketPayload, PipelineMsg, StreamInfoPayload,
};
use crate::framework::router::Router;
use crate::framework::supervisor::StageName;
use crate::framework::{
    Actor, ActorBuilder, Context, Dispatch, Flow, Handler, ShutdownPayload, SourceEosPayload,
    StageKind,
};

/// Default decoder pool size — enough to hide NVDEC queue depth
/// without holding memory across many frames.  Matches the legacy
/// sample's `DECODER_POOL_SIZE`.
pub const DEFAULT_POOL_SIZE: u32 = 8;
/// Default eviction TTL — large enough that a long-running file
/// source is never evicted mid-run.  Matches the legacy sample.
pub const DEFAULT_EVICTION_TTL: Duration = Duration::from_secs(3600);
/// Default per-decoder idle timeout — matches the legacy sample.
pub const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(5);
/// Default detect-buffer limit — matches the legacy sample.
pub const DEFAULT_DETECT_BUFFER_LIMIT: usize = 60;
/// Fallback framerate used when the container does not advertise
/// one (per [`VideoInfo`] contract: `framerate_num == 0`).
pub const FALLBACK_FPS_NUM: i64 = 30;
/// Denominator counterpart to [`FALLBACK_FPS_NUM`].
pub const FALLBACK_FPS_DEN: i64 = 1;
/// Default inbox receive-poll cadence — idle periods at this
/// cadence cause [`FlexibleDecoderPool::flush_idle`] to run.
pub const DEFAULT_POLL_TIMEOUT: Duration = Duration::from_millis(100);

/// Observer hook fired on every
/// [`EncodedMsg::StreamInfo`](crate::framework::envelopes::EncodedMsg::StreamInfo)
/// sentinel.  Runs on the actor thread (not the pool callback
/// thread), so the hook is `FnMut` and has full access to the
/// actor's [`Context`] — e.g. for stats accounting, dump-to-log
/// invariants, or bridging into shared state via
/// [`Context::shared`](crate::framework::Context::shared).
///
/// The decoder itself no longer needs this hook to decode: every
/// [`EncodedMsg::Packet`](crate::framework::envelopes::EncodedMsg::Packet)
/// carries its own
/// [`VideoInfo`](savant_gstreamer::mp4_demuxer::VideoInfo)
/// in-band.  Use this observer purely for stream-header-level
/// observability; see [`Decoder::default_on_stream_info`]
/// for the no-op default installed when the setter is omitted.
pub type StreamInfoObserver = Box<dyn FnMut(&StreamInfoPayload, &mut Context<Decoder>) + Send>;

/// Hook fired for [`FlexibleDecoderOutput::Frame`] — receives the
/// sealed `(VideoFrameProxy, SharedBuffer)` delivery together with
/// the stage's [`Router<PipelineMsg>`] so the user decides whether
/// and where to forward it.  Runs on the pool's output callback
/// thread; must be `Fn + Send + Sync` and non-blocking.
///
/// Typical use: `router.send(PipelineMsg::Delivery(sealed))` (default
/// peer) or `router.send_to(&engine_for(&source_id), PipelineMsg::Delivery(sealed))`
/// (source-id-based dispatch between multiple inference engines).
pub type OnFrameHook = Arc<dyn Fn(SealedDelivery, &Router<PipelineMsg>) + Send + Sync + 'static>;

/// Hook fired for [`FlexibleDecoderOutput::SourceEos`] — receives
/// the `source_id` and the router.  Typical use:
/// `router.send(PipelineMsg::SourceEos { source_id: source_id.into() })`.
pub type OnDecoderSourceEosHook = Arc<dyn Fn(&str, &Router<PipelineMsg>) + Send + Sync + 'static>;

/// User-supplied eviction decision callback — invoked when a
/// per-`source_id` decoder slot exceeds
/// [`eviction_ttl`](DecoderBuilder::eviction_ttl).  Return
/// [`EvictionDecision::Keep`] to reset the TTL or
/// [`EvictionDecision::Evict`] to graceful-drain the slot.  Runs on
/// the pool's eviction sweep thread; must not block.
pub type EvictionCallback = Arc<dyn Fn(&str) -> EvictionDecision + Send + Sync + 'static>;

/// Hook fired when the underlying decoder reconfigures mid-stream
/// (codec or resolution change).  Receives the old / new parameters
/// plus the router for optional downstream signalling.  Runs on the
/// pool's output callback thread.
pub type OnParameterChangeHook = Arc<
    dyn Fn(&DecoderParameters, &DecoderParameters, &Router<PipelineMsg>) + Send + Sync + 'static,
>;

/// Hook fired for every [`FlexibleDecoderOutput::Skipped`] — reports
/// packets the decoder refused to process (see [`SkipReason`]).
/// Receives the skipped frame, the original payload (when available),
/// the reason, and the router.  Runs on the pool's output thread.
pub type OnSkippedHook = Arc<
    dyn Fn(&VideoFrameProxy, Option<&[u8]>, &SkipReason, &Router<PipelineMsg>)
        + Send
        + Sync
        + 'static,
>;

/// Hook fired for [`FlexibleDecoderOutput::OrphanFrame`] — a decoded
/// frame whose originating [`VideoFrameProxy`] could not be located
/// in the pool's frame map (typically a late delivery after a
/// reset).  Runs on the pool's output thread.
pub type OnOrphanFrameHook =
    Arc<dyn Fn(&DecodedFrame, &Router<PipelineMsg>) + Send + Sync + 'static>;

/// Hook fired for arbitrary [`gst::Event`]s captured on the decoder's
/// output — useful for bridging custom upstream events to domain
/// logic.  Runs on the pool's output thread.
pub type OnDecoderEventHook =
    Arc<dyn Fn(&gst::Event, &Router<PipelineMsg>) + Send + Sync + 'static>;

/// Hook fired on [`FlexibleDecoderOutput::Error`].  Runs on the pool's
/// output thread.  The template performs no abort bookkeeping of its
/// own; if the user wants to short-circuit further output, they can
/// call `router.default_sink().map(|s| s.abort())` from inside the
/// hook.
pub type OnDecoderErrorHook =
    Arc<dyn Fn(&DecoderError, &Router<PipelineMsg>) + Send + Sync + 'static>;

/// `EncodedMsg` → `PipelineMsg::Delivery` actor template.
///
/// Construct via [`Decoder::builder`].
///
/// All variant hooks are **always populated** at runtime — see the
/// "Runtime invariant" section of the module header for details.
pub struct Decoder {
    decoder: FlexibleDecoderPool,
    poll_timeout: Duration,
    on_stream_info: StreamInfoObserver,
    graceful_done: bool,
}

impl Decoder {
    /// Start a fluent builder for a decoder registered under `name`
    /// with inbox capacity `capacity`.
    pub fn builder(name: StageName, capacity: usize) -> DecoderBuilder {
        DecoderBuilder::new(name, capacity)
    }

    /// Default `on_stream_info` observer — a no-op.  The template
    /// itself already logs the stream-info line; decoding no longer
    /// depends on this sentinel (every
    /// [`EncodedMsg::Packet`](crate::framework::envelopes::EncodedMsg::Packet)
    /// carries its own [`VideoInfo`] in-band), so omitting
    /// `.on_stream_info(...)` simply means "don't add any
    /// user-level side-effect on top of the log line".
    pub fn default_on_stream_info(
    ) -> impl FnMut(&StreamInfoPayload, &mut Context<Decoder>) + Send + 'static {
        |_, _| {}
    }

    /// Default `on_frame` forwarder: emits
    /// [`PipelineMsg::Delivery(sealed)`](crate::framework::envelopes::PipelineMsg::Delivery)
    /// on the router.  Matches the natural pass-through behaviour
    /// of the legacy decoder thread.  Override when the hook needs
    /// to observe the frame (e.g. stats ticks) or route by
    /// `source_id` via `router.send_to(...)`.
    pub fn default_on_frame(
    ) -> impl Fn(SealedDelivery, &Router<PipelineMsg>) + Send + Sync + 'static {
        |sealed, router| {
            if !router.send(PipelineMsg::Delivery(sealed)) {
                log::warn!("decoder downstream closed; dropped a Delivery");
            }
        }
    }

    /// Default `on_source_eos` forwarder: emits
    /// [`PipelineMsg::SourceEos { source_id }`](crate::framework::envelopes::PipelineMsg::SourceEos)
    /// on the router.  The pool callback fires
    /// [`FlexibleDecoderOutput::SourceEos`] strictly after the last
    /// [`FlexibleDecoderOutput::Frame`] for the same `source_id`,
    /// so the stream-aligned "EOS after last frame" invariant is
    /// preserved without any sample-level bookkeeping.
    pub fn default_on_source_eos() -> impl Fn(&str, &Router<PipelineMsg>) + Send + Sync + 'static {
        |source_id, router| {
            if !router.send(PipelineMsg::SourceEos {
                source_id: source_id.to_string(),
            }) {
                log::warn!("decoder downstream closed; dropping SourceEos({source_id})");
            }
        }
    }

    /// Default `on_parameter_change` logger — emits a single
    /// `info!` line with the old/new [`DecoderParameters`]
    /// prefixed by the stage name.  No downstream message is
    /// forwarded (the caller can subscribe by overriding this
    /// hook).
    pub fn default_on_parameter_change(
        namespace: StageName,
    ) -> impl Fn(&DecoderParameters, &DecoderParameters, &Router<PipelineMsg>) + Send + Sync + 'static
    {
        move |old, new, _router| {
            log::info!("[{namespace}] parameter change: {old:?} -> {new:?}");
        }
    }

    /// Default `on_skipped` logger — emits a single `debug!` line
    /// with the [`SkipReason`] prefixed by the stage name.
    pub fn default_on_skipped(
        namespace: StageName,
    ) -> impl Fn(&VideoFrameProxy, Option<&[u8]>, &SkipReason, &Router<PipelineMsg>)
           + Send
           + Sync
           + 'static {
        move |_frame, _data, reason, _router| {
            log::debug!("[{namespace}] skipped: {reason:?}");
        }
    }

    /// Default `on_orphan_frame` logger — emits a single `debug!`
    /// line prefixed by the stage name.  Orphan frames usually
    /// signal a late delivery after a per-source reset and warrant
    /// observation but no forwarding.
    pub fn default_on_orphan_frame(
        namespace: StageName,
    ) -> impl Fn(&DecodedFrame, &Router<PipelineMsg>) + Send + Sync + 'static {
        move |_decoded, _router| {
            log::debug!("[{namespace}] orphan frame (source id mismatch?)");
        }
    }

    /// Default `on_decoder_event` hook — a no-op.  GStreamer events
    /// without domain meaning are silently ignored; override to
    /// bridge custom upstream events into domain logic.
    pub fn default_on_decoder_event(
    ) -> impl Fn(&gst::Event, &Router<PipelineMsg>) + Send + Sync + 'static {
        |_ev, _router| {}
    }

    /// Default `on_decoder_error` logger — emits a single `error!`
    /// line prefixed by the stage name.  The error is otherwise
    /// swallowed so one misbehaving stream does not tear the
    /// pipeline down; override to abort the default sink or
    /// propagate a fatal signal downstream.
    pub fn default_on_decoder_error(
        namespace: StageName,
    ) -> impl Fn(&DecoderError, &Router<PipelineMsg>) + Send + Sync + 'static {
        move |err, _router| {
            log::error!("[{namespace}] decoder error: {err}");
        }
    }
}

impl Actor for Decoder {
    type Msg = EncodedMsg;

    fn handle(&mut self, msg: EncodedMsg, ctx: &mut Context<Self>) -> Result<Flow> {
        msg.dispatch(self, ctx)
    }

    fn poll_timeout(&self) -> Duration {
        self.poll_timeout
    }

    fn on_tick(&mut self, ctx: &mut Context<Self>) -> Result<Flow> {
        if let Err(e) = self.decoder.flush_idle() {
            log::warn!("[{}] flush_idle failed: {e}", ctx.own_name());
        }
        Ok(Flow::Cont)
    }

    fn started(&mut self, ctx: &mut Context<Self>) -> Result<()> {
        log::info!("[{}] decoder started", ctx.own_name());
        Ok(())
    }

    fn stopping(&mut self, ctx: &mut Context<Self>) {
        if !self.graceful_done {
            if let Err(e) = self.decoder.graceful_shutdown() {
                log::warn!("[{}] graceful_shutdown failed: {e}", ctx.own_name());
            }
            self.graceful_done = true;
        }
        log::info!("[{}] decoder stopping", ctx.own_name());
    }
}

impl Handler<StreamInfoPayload> for Decoder {
    fn handle(&mut self, msg: StreamInfoPayload, ctx: &mut Context<Self>) -> Result<Flow> {
        log::info!(
            "[{}] stream info: source_id={} {}x{} @ {}/{}",
            ctx.own_name(),
            msg.source_id,
            msg.info.width,
            msg.info.height,
            msg.info.framerate_num,
            msg.info.framerate_den
        );
        (self.on_stream_info)(&msg, ctx);
        Ok(Flow::Cont)
    }
}

impl Handler<PacketPayload> for Decoder {
    fn handle(&mut self, msg: PacketPayload, _ctx: &mut Context<Self>) -> Result<Flow> {
        let PacketPayload {
            source_id,
            info,
            packet,
        } = msg;
        let frame = make_decode_frame(&source_id, &packet, &info);
        self.decoder
            .submit(&frame, Some(&packet.data))
            .map_err(|e| anyhow!("decoder submit (source_id={source_id}): {e}"))?;
        Ok(Flow::Cont)
    }
}

impl Handler<FramePayload> for Decoder {
    fn handle(&mut self, msg: FramePayload, _ctx: &mut Context<Self>) -> Result<Flow> {
        let FramePayload { frame, payload } = msg;
        let sid = frame.get_source_id();
        self.decoder
            .submit(&frame, payload.as_deref())
            .map_err(|e| anyhow!("decoder submit (source_id={sid}): {e}"))?;
        Ok(Flow::Cont)
    }
}

impl Handler<SourceEosPayload> for Decoder {
    fn handle(&mut self, msg: SourceEosPayload, ctx: &mut Context<Self>) -> Result<Flow> {
        log::info!(
            "[{}] SourceEos {}: initiating operator drain",
            ctx.own_name(),
            msg.source_id
        );
        if let Err(e) = self.decoder.source_eos(&msg.source_id) {
            log::warn!(
                "[{}] source_eos({}) failed: {e}",
                ctx.own_name(),
                msg.source_id
            );
        }
        Ok(Flow::Cont)
    }
}

/// Default no-op handler: the framework's loop driver consumes the
/// shutdown hint from [`EncodedMsg::Shutdown`](crate::framework::envelopes::EncodedMsg::Shutdown)
/// via [`Envelope::as_shutdown`](super::super::Envelope::as_shutdown),
/// so the in-band sentinel has no remaining work here.
impl Handler<ShutdownPayload> for Decoder {}

/// Fluent builder for [`Decoder`].
pub struct DecoderBuilder {
    name: StageName,
    capacity: usize,
    downstream: Option<StageName>,
    pool_config: Option<FlexibleDecoderPoolConfig>,
    gpu_id: u32,
    pool_size: u32,
    eviction_ttl: Duration,
    idle_timeout: Duration,
    detect_buffer_limit: usize,
    poll_timeout: Duration,
    on_stream_info: Option<StreamInfoObserver>,
    on_frame: Option<OnFrameHook>,
    on_source_eos: Option<OnDecoderSourceEosHook>,
    eviction_callback: Option<EvictionCallback>,
    decoder_config_callback: Option<DecoderConfigCallback>,
    on_parameter_change: Option<OnParameterChangeHook>,
    on_skipped: Option<OnSkippedHook>,
    on_orphan_frame: Option<OnOrphanFrameHook>,
    on_decoder_event: Option<OnDecoderEventHook>,
    on_decoder_error: Option<OnDecoderErrorHook>,
}

impl DecoderBuilder {
    /// Start a builder with sample-style defaults.
    pub fn new(name: StageName, capacity: usize) -> Self {
        Self {
            name,
            capacity,
            downstream: None,
            pool_config: None,
            gpu_id: 0,
            pool_size: DEFAULT_POOL_SIZE,
            eviction_ttl: DEFAULT_EVICTION_TTL,
            idle_timeout: DEFAULT_IDLE_TIMEOUT,
            detect_buffer_limit: DEFAULT_DETECT_BUFFER_LIMIT,
            poll_timeout: DEFAULT_POLL_TIMEOUT,
            on_stream_info: None,
            on_frame: None,
            on_source_eos: None,
            eviction_callback: None,
            decoder_config_callback: None,
            on_parameter_change: None,
            on_skipped: None,
            on_orphan_frame: None,
            on_decoder_event: None,
            on_decoder_error: None,
        }
    }

    /// Optional default peer installed on the
    /// [`Router<PipelineMsg>`] handed to every variant hook.  User
    /// hooks call `router.send(msg)` to route to this peer and
    /// `router.send_to(&peer, msg)` to address any other registered
    /// actor by name.
    pub fn downstream(mut self, peer: StageName) -> Self {
        self.downstream = Some(peer);
        self
    }

    /// Override the full [`FlexibleDecoderPoolConfig`].  When set,
    /// the per-field [`gpu_id`](Self::gpu_id),
    /// [`pool_size`](Self::pool_size),
    /// [`eviction_ttl`](Self::eviction_ttl),
    /// [`idle_timeout`](Self::idle_timeout), and
    /// [`detect_buffer_limit`](Self::detect_buffer_limit) knobs
    /// are ignored.
    pub fn pool_config(mut self, cfg: FlexibleDecoderPoolConfig) -> Self {
        self.pool_config = Some(cfg);
        self
    }

    /// DeepStream GPU id (default `0`).
    pub fn gpu_id(mut self, id: u32) -> Self {
        self.gpu_id = id;
        self
    }

    /// Buffer-pool slot count (default [`DEFAULT_POOL_SIZE`]).
    pub fn pool_size(mut self, n: u32) -> Self {
        self.pool_size = n;
        self
    }

    /// Eviction TTL for idle `source_id` decoders (default
    /// [`DEFAULT_EVICTION_TTL`]).
    pub fn eviction_ttl(mut self, d: Duration) -> Self {
        self.eviction_ttl = d;
        self
    }

    /// Per-decoder idle timeout (default [`DEFAULT_IDLE_TIMEOUT`]).
    pub fn idle_timeout(mut self, d: Duration) -> Self {
        self.idle_timeout = d;
        self
    }

    /// Codec-detection buffer limit (default
    /// [`DEFAULT_DETECT_BUFFER_LIMIT`]).
    pub fn detect_buffer_limit(mut self, n: usize) -> Self {
        self.detect_buffer_limit = n;
        self
    }

    /// Inbox receive-poll cadence — governs how often
    /// [`FlexibleDecoderPool::flush_idle`] is called when the inbox
    /// is quiet (default [`DEFAULT_POLL_TIMEOUT`]).
    pub fn poll_timeout(mut self, d: Duration) -> Self {
        self.poll_timeout = d;
        self
    }

    /// Override the `on_stream_info` observer — called after every
    /// [`StreamInfoPayload`] is cached, on the actor thread (with
    /// full access to the [`Context`]).  Use for stats accounting
    /// or dump-to-log invariants.
    ///
    /// Omitting this setter is equivalent to calling
    /// `.on_stream_info(Decoder::default_on_stream_info())` —
    /// a no-op; the template already logs and caches the
    /// [`StreamInfoPayload`] on its own.
    pub fn on_stream_info<F>(mut self, f: F) -> Self
    where
        F: FnMut(&StreamInfoPayload, &mut Context<Decoder>) + Send + 'static,
    {
        self.on_stream_info = Some(Box::new(f));
        self
    }

    /// Override the `on_frame` hook — fired for every
    /// [`FlexibleDecoderOutput::Frame`].  Receives the sealed
    /// `(VideoFrameProxy, SharedBuffer)` delivery and the stage's
    /// [`Router<PipelineMsg>`].  The hook decides whether to
    /// forward (via `router.send(PipelineMsg::Delivery(sealed))` for
    /// the default peer, or
    /// `router.send_to(&peer, PipelineMsg::Delivery(sealed))` for
    /// explicit name-based routing) and what (if anything) to record
    /// per frame.
    ///
    /// Runs on the pool's output callback thread; must be
    /// `Fn + Send + Sync + 'static` and non-blocking.
    ///
    /// Omitting this setter is equivalent to calling
    /// `.on_frame(Decoder::default_on_frame())`, which emits
    /// [`PipelineMsg::Delivery(sealed)`](crate::framework::envelopes::PipelineMsg::Delivery)
    /// on the default peer.
    pub fn on_frame<F>(mut self, f: F) -> Self
    where
        F: Fn(SealedDelivery, &Router<PipelineMsg>) + Send + Sync + 'static,
    {
        self.on_frame = Some(Arc::new(f));
        self
    }

    /// Override the `on_source_eos` hook — fired for
    /// [`FlexibleDecoderOutput::SourceEos`] exactly once per
    /// `source_id`, strictly after the last
    /// [`on_frame`](Self::on_frame) for that source.
    ///
    /// Omitting this setter is equivalent to calling
    /// `.on_source_eos(Decoder::default_on_source_eos())`,
    /// which emits
    /// [`PipelineMsg::SourceEos`](crate::framework::envelopes::PipelineMsg::SourceEos)
    /// on the default peer.
    pub fn on_source_eos<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &Router<PipelineMsg>) + Send + Sync + 'static,
    {
        self.on_source_eos = Some(Arc::new(f));
        self
    }

    /// Install an eviction-decision callback on the underlying pool
    /// (switches construction to
    /// [`FlexibleDecoderPool::with_eviction_callback`]).  Invoked
    /// when a per-`source_id` decoder exceeds its
    /// [`eviction_ttl`](Self::eviction_ttl); the callback chooses
    /// between [`EvictionDecision::Keep`] and
    /// [`EvictionDecision::Evict`].
    ///
    /// The callback runs on the pool's eviction sweep thread, so
    /// the closure must be `Fn + Send + Sync + 'static` and must
    /// not block on actor-thread state.
    pub fn eviction_callback<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> EvictionDecision + Send + Sync + 'static,
    {
        self.eviction_callback = Some(Arc::new(f));
        self
    }

    /// Install a decoder-config transformation callback (see
    /// [`DecoderConfigCallback`]).  Invoked on every new decoder
    /// activation (first submit on a `source_id`, or after a codec
    /// or resolution change) to let user code tweak the resolved
    /// [`DecoderConfig`] immediately before `NvDecoder::new` runs.
    ///
    /// When [`pool_config`](Self::pool_config) is also provided,
    /// that pool config wins — this hook only takes effect on
    /// the builder-constructed default config.
    pub fn decoder_config_callback<F>(mut self, f: F) -> Self
    where
        F: Fn(
                deepstream_decoders::DecoderConfig,
                &VideoFrameProxy,
            ) -> deepstream_decoders::DecoderConfig
            + Send
            + Sync
            + 'static,
    {
        self.decoder_config_callback = Some(Arc::new(f));
        self
    }

    /// Override the `on_parameter_change` hook — fired on
    /// [`FlexibleDecoderOutput::ParameterChange`].  Useful for
    /// logging resolution/codec transitions, invalidating downstream
    /// caches, or (via `router`) broadcasting a reconfiguration
    /// message.
    ///
    /// Omitting this setter is equivalent to calling
    /// `.on_parameter_change(Decoder::default_on_parameter_change(name.clone()))`,
    /// which logs the old/new parameters at `info` level.
    pub fn on_parameter_change<F>(mut self, f: F) -> Self
    where
        F: Fn(&DecoderParameters, &DecoderParameters, &Router<PipelineMsg>) + Send + Sync + 'static,
    {
        self.on_parameter_change = Some(Arc::new(f));
        self
    }

    /// Override the `on_skipped` hook — fired for every
    /// [`FlexibleDecoderOutput::Skipped`].  Reports packets the
    /// decoder refused to process (see [`SkipReason`]).
    ///
    /// Omitting this setter is equivalent to calling
    /// `.on_skipped(Decoder::default_on_skipped(name.clone()))`,
    /// which logs the skip reason at `debug` level.
    pub fn on_skipped<F>(mut self, f: F) -> Self
    where
        F: Fn(&VideoFrameProxy, Option<&[u8]>, &SkipReason, &Router<PipelineMsg>)
            + Send
            + Sync
            + 'static,
    {
        self.on_skipped = Some(Arc::new(f));
        self
    }

    /// Override the `on_orphan_frame` hook — fired for
    /// [`FlexibleDecoderOutput::OrphanFrame`], a decoded frame whose
    /// [`VideoFrameProxy`] could not be reattached.  Typical use:
    /// surface unusual reset races as a metric.
    ///
    /// Omitting this setter is equivalent to calling
    /// `.on_orphan_frame(Decoder::default_on_orphan_frame(name.clone()))`,
    /// which logs at `debug` level.
    pub fn on_orphan_frame<F>(mut self, f: F) -> Self
    where
        F: Fn(&DecodedFrame, &Router<PipelineMsg>) + Send + Sync + 'static,
    {
        self.on_orphan_frame = Some(Arc::new(f));
        self
    }

    /// Override the `on_decoder_event` hook — fired for
    /// [`FlexibleDecoderOutput::Event`], capturing arbitrary
    /// [`gst::Event`]s flowing through the decoder output.  Use to
    /// bridge custom events into domain-level hooks.
    ///
    /// Omitting this setter is equivalent to calling
    /// `.on_decoder_event(Decoder::default_on_decoder_event())`,
    /// which is a no-op (events without domain meaning are silently
    /// ignored).
    pub fn on_decoder_event<F>(mut self, f: F) -> Self
    where
        F: Fn(&gst::Event, &Router<PipelineMsg>) + Send + Sync + 'static,
    {
        self.on_decoder_event = Some(Arc::new(f));
        self
    }

    /// Override the `on_decoder_error` hook — fired for
    /// [`FlexibleDecoderOutput::Error`].  The template does **not**
    /// abort the default sink on its own; if the user wants to
    /// short-circuit further output (matching the previous "first
    /// error latches abort" semantics) they can call
    /// `router.default_sink().map(|s| s.abort())` inside the hook.
    ///
    /// Omitting this setter is equivalent to calling
    /// `.on_decoder_error(Decoder::default_on_decoder_error(name.clone()))`,
    /// which logs the error at `error` level.
    pub fn on_decoder_error<F>(mut self, f: F) -> Self
    where
        F: Fn(&DecoderError, &Router<PipelineMsg>) + Send + Sync + 'static,
    {
        self.on_decoder_error = Some(Arc::new(f));
        self
    }

    /// Finalise the template and obtain an
    /// [`ActorBuilder<Decoder>`] ready for
    /// [`System::register_actor`](super::super::System::register_actor).
    ///
    /// # Errors
    ///
    /// Currently infallible — `downstream` is optional.
    pub fn build(self) -> Result<ActorBuilder<Decoder>> {
        let DecoderBuilder {
            name,
            capacity,
            downstream,
            pool_config,
            gpu_id,
            pool_size,
            eviction_ttl,
            idle_timeout,
            detect_buffer_limit,
            poll_timeout,
            on_stream_info,
            on_frame,
            on_source_eos,
            eviction_callback,
            decoder_config_callback,
            on_parameter_change,
            on_skipped,
            on_orphan_frame,
            on_decoder_event,
            on_decoder_error,
        } = self;
        let _ = StageKind::Decoder; // ensure kind enum import stays live
        let cfg = pool_config.unwrap_or_else(|| {
            let mut cfg = FlexibleDecoderPoolConfig::new(gpu_id, pool_size, eviction_ttl)
                .idle_timeout(idle_timeout)
                .detect_buffer_limit(detect_buffer_limit);
            if let Some(cb) = decoder_config_callback.clone() {
                cfg = cfg.decoder_config_callback_arc(cb);
            }
            cfg
        });
        // Auto-install the matching `default_on_*` forwarder for
        // every hook the user did not override.  This keeps the
        // runtime-field invariant ("hooks are never None") in sync
        // with the builder's ergonomic "skip what you don't care
        // about" API.
        let on_stream_info: StreamInfoObserver =
            on_stream_info.unwrap_or_else(|| Box::new(Decoder::default_on_stream_info()));
        let on_frame: OnFrameHook =
            on_frame.unwrap_or_else(|| Arc::new(Decoder::default_on_frame()));
        let on_source_eos: OnDecoderSourceEosHook =
            on_source_eos.unwrap_or_else(|| Arc::new(Decoder::default_on_source_eos()));
        let on_parameter_change: OnParameterChangeHook = on_parameter_change
            .unwrap_or_else(|| Arc::new(Decoder::default_on_parameter_change(name.clone())));
        let on_skipped: OnSkippedHook =
            on_skipped.unwrap_or_else(|| Arc::new(Decoder::default_on_skipped(name.clone())));
        let on_orphan_frame: OnOrphanFrameHook = on_orphan_frame
            .unwrap_or_else(|| Arc::new(Decoder::default_on_orphan_frame(name.clone())));
        let on_decoder_event: OnDecoderEventHook =
            on_decoder_event.unwrap_or_else(|| Arc::new(Decoder::default_on_decoder_event()));
        let on_decoder_error: OnDecoderErrorHook = on_decoder_error
            .unwrap_or_else(|| Arc::new(Decoder::default_on_decoder_error(name.clone())));
        let hooks = VariantHooks {
            on_frame,
            on_source_eos,
            on_parameter_change,
            on_skipped,
            on_orphan_frame,
            on_decoder_event,
            on_decoder_error,
        };
        Ok(ActorBuilder::new(name.clone(), capacity)
            .poll_timeout(poll_timeout)
            .factory(move |bx| {
                let router: Router<PipelineMsg> = bx.router(downstream.as_ref())?;
                let owner = bx.own_name().clone();
                let decoder =
                    build_pool(cfg, router, owner, eviction_callback.clone(), hooks.clone());
                Ok(Decoder {
                    decoder,
                    poll_timeout,
                    on_stream_info,
                    graceful_done: false,
                })
            }))
    }
}

/// Per-output-variant hook bundle forwarded into the pool callback.
/// Every field is a non-`Option` `Arc<dyn Fn>` — the builder
/// auto-installs the matching `Decoder::default_on_*` for any
/// hook the user did not override.  The callback therefore
/// dispatches every variant unconditionally.
#[derive(Clone)]
struct VariantHooks {
    on_frame: OnFrameHook,
    on_source_eos: OnDecoderSourceEosHook,
    on_parameter_change: OnParameterChangeHook,
    on_skipped: OnSkippedHook,
    on_orphan_frame: OnOrphanFrameHook,
    on_decoder_event: OnDecoderEventHook,
    on_decoder_error: OnDecoderErrorHook,
}

/// Build the [`FlexibleDecoderPool`] with a callback that dispatches
/// every [`FlexibleDecoderOutput`] variant to the corresponding user
/// hook.  The callback never calls `router.send` itself — every send
/// site lives inside user code.
fn build_pool(
    cfg: FlexibleDecoderPoolConfig,
    router: Router<PipelineMsg>,
    owner: StageName,
    eviction_callback: Option<EvictionCallback>,
    hooks: VariantHooks,
) -> FlexibleDecoderPool {
    let owner_cb = owner.clone();
    let on_output = move |mut out: FlexibleDecoderOutput| match &out {
        FlexibleDecoderOutput::Frame { .. } => {
            if let Some(sealed) = out.take_delivery() {
                (hooks.on_frame)(sealed, &router);
            }
        }
        FlexibleDecoderOutput::ParameterChange { old, new } => {
            (hooks.on_parameter_change)(old, new, &router);
        }
        FlexibleDecoderOutput::Skipped {
            frame,
            data,
            reason,
        } => {
            (hooks.on_skipped)(frame, data.as_deref(), reason, &router);
        }
        FlexibleDecoderOutput::OrphanFrame { decoded } => {
            (hooks.on_orphan_frame)(decoded, &router);
        }
        FlexibleDecoderOutput::SourceEos { source_id } => {
            log::info!(
                "[{owner_cb}/cb] FlexibleDecoderOutput::SourceEos for source_id={source_id}"
            );
            (hooks.on_source_eos)(source_id, &router);
        }
        FlexibleDecoderOutput::Event(ev) => {
            (hooks.on_decoder_event)(ev, &router);
        }
        FlexibleDecoderOutput::Error(err) => {
            (hooks.on_decoder_error)(err, &router);
        }
    };
    if let Some(cb) = eviction_callback {
        FlexibleDecoderPool::with_eviction_callback(cfg, on_output, move |sid| cb(sid))
    } else {
        FlexibleDecoderPool::new(cfg, on_output)
    }
}

/// Build the per-packet [`VideoFrameProxy`] the decoder pool
/// consumes.  Kept `pub` so unit tests and downstream users can
/// verify the [`VideoInfo::codec`] round-trip.
pub fn make_decode_frame(
    source_id: &str,
    pkt: &DemuxedPacket,
    info: &VideoInfo,
) -> VideoFrameProxy {
    let (fps_num, fps_den) = if info.framerate_num == 0 {
        (FALLBACK_FPS_NUM, FALLBACK_FPS_DEN)
    } else {
        (info.framerate_num as i64, info.framerate_den.max(1) as i64)
    };
    VideoFrameProxy::new(
        source_id,
        (fps_num, fps_den),
        info.width as i64,
        info.height as i64,
        VideoFrameContent::None,
        VideoFrameTranscodingMethod::Copy,
        Some(info.codec),
        Some(pkt.is_keyframe),
        (1, 1_000_000_000),
        pkt.pts_ns as i64,
        pkt.dts_ns.map(|v| v as i64),
        pkt.duration_ns.map(|v| v as i64),
    )
    .expect("VideoFrameProxy::new (decode)")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::addr::Addr;
    use crate::framework::operator_sink::OperatorSink;
    use crate::framework::registry::Registry;
    use crate::framework::supervisor::{StageKind, StageName};
    use crossbeam::channel::{bounded, Receiver};
    use savant_core::primitives::video_codec::VideoCodec;

    /// Build a `Router<PipelineMsg>` whose default peer is a freshly
    /// registered `PipelineMsg` inbox.  Returns the router and the
    /// receiver so tests can observe forwarded messages.
    fn router_with_default_peer() -> (Router<PipelineMsg>, Receiver<PipelineMsg>) {
        let peer = StageName::unnamed(StageKind::Infer);
        let (tx, rx) = bounded::<PipelineMsg>(4);
        let mut reg = Registry::new();
        reg.insert::<PipelineMsg>(peer.clone(), Addr::new(peer.clone(), tx));
        let addr: Addr<PipelineMsg> = reg.get::<PipelineMsg>(&peer).unwrap();
        let owner = StageName::unnamed(StageKind::Decoder);
        let default = OperatorSink::new(owner.clone(), addr);
        let router = Router::new(owner, Arc::new(reg), Some(default));
        (router, rx)
    }

    /// `downstream` is optional now; the builder accepts the
    /// no-downstream configuration.  The resulting actor's router
    /// logs once on the first `send(msg)` and silently drops
    /// subsequent default-peer sends.
    #[test]
    fn builder_without_downstream_is_accepted() {
        let name = StageName::unnamed(StageKind::Decoder);
        let _ = Decoder::builder(name, 4)
            .build()
            .expect("no-downstream builder is accepted");
    }

    /// Runtime invariant: `build()` succeeds with only the
    /// mandatory setters (`name` + `capacity` via `builder`) and
    /// auto-installs every `default_on_*` forwarder.  The
    /// `DecoderBuilder` has no required hooks.
    #[test]
    fn builder_accepts_bare_minimum() {
        let name = StageName::unnamed(StageKind::Decoder);
        let _ = Decoder::builder(name, 4)
            .downstream(StageName::unnamed(StageKind::Infer))
            .build()
            .expect("bare-minimum builder is accepted");
    }

    #[test]
    fn make_decode_frame_forwards_info_codec() {
        let info = VideoInfo {
            codec: VideoCodec::Hevc,
            width: 1920,
            height: 1080,
            framerate_num: 30,
            framerate_den: 1,
        };
        let pkt = DemuxedPacket {
            data: Vec::new(),
            pts_ns: 0,
            dts_ns: None,
            duration_ns: None,
            is_keyframe: true,
        };
        let frame = make_decode_frame("src", &pkt, &info);
        assert_eq!(frame.get_codec(), Some(VideoCodec::Hevc));
    }

    #[test]
    fn builder_accepts_all_hooks() {
        let name = StageName::unnamed(StageKind::Decoder);
        let _ = Decoder::builder(name, 4)
            .downstream(StageName::unnamed(StageKind::Infer))
            .eviction_callback(|_| EvictionDecision::Keep)
            .decoder_config_callback(|cfg, _frame| cfg)
            .on_frame(|sealed, router| {
                let _ = router.send(PipelineMsg::Delivery(sealed));
            })
            .on_source_eos(|sid, router| {
                let _ = router.send(PipelineMsg::SourceEos {
                    source_id: sid.to_string(),
                });
            })
            .on_parameter_change(|_, _, _| {})
            .on_skipped(|_, _, _, _| {})
            .on_orphan_frame(|_, _| {})
            .on_decoder_event(|_, _| {})
            .on_decoder_error(|_, _| {})
            .on_stream_info(|_, _| {})
            .build()
            .expect("full hook configuration builds");
    }

    /// Confirms that each `default_on_*` associated function slots
    /// into the builder's generic hook bounds as-is.
    #[test]
    fn builder_accepts_default_forwarders() {
        let name = StageName::unnamed(StageKind::Decoder);
        let _ = Decoder::builder(name.clone(), 4)
            .downstream(StageName::unnamed(StageKind::Infer))
            .on_stream_info(Decoder::default_on_stream_info())
            .on_frame(Decoder::default_on_frame())
            .on_source_eos(Decoder::default_on_source_eos())
            .on_parameter_change(Decoder::default_on_parameter_change(name.clone()))
            .on_skipped(Decoder::default_on_skipped(name.clone()))
            .on_orphan_frame(Decoder::default_on_orphan_frame(name.clone()))
            .on_decoder_event(Decoder::default_on_decoder_event())
            .on_decoder_error(Decoder::default_on_decoder_error(name))
            .build()
            .expect("default forwarders build");
    }

    /// `default_on_source_eos` forwards `PipelineMsg::SourceEos` on
    /// the default peer — the contract the sample used to spell out
    /// by hand.
    #[test]
    fn default_on_source_eos_forwards_via_router() {
        let hook = Decoder::default_on_source_eos();
        let (router, rx) = router_with_default_peer();
        hook("cam-1", &router);
        match rx.try_recv().expect("SourceEos should be forwarded") {
            PipelineMsg::SourceEos { source_id } => assert_eq!(source_id, "cam-1"),
            other => panic!("expected SourceEos, got {other:?}"),
        }
        assert!(rx.try_recv().is_err(), "exactly one message forwarded");
    }

    /// `default_on_decoder_error` is log-only — it must not route
    /// anything downstream.
    #[test]
    fn default_on_decoder_error_does_not_route() {
        let hook = Decoder::default_on_decoder_error(StageName::unnamed(StageKind::Decoder));
        let (router, rx) = router_with_default_peer();
        let err = DecoderError::PipelineError("synthetic".to_string());
        hook(&err, &router);
        assert!(rx.try_recv().is_err(), "errors must not be routed");
    }

    /// `default_on_parameter_change` is log-only — it must not
    /// forward any control message downstream.
    #[test]
    fn default_on_parameter_change_does_not_route() {
        let hook = Decoder::default_on_parameter_change(StageName::unnamed(StageKind::Decoder));
        let (router, rx) = router_with_default_peer();
        let params = DecoderParameters {
            codec: VideoCodec::H264,
            width: 1920,
            height: 1080,
        };
        hook(&params, &params, &router);
        assert!(
            rx.try_recv().is_err(),
            "parameter-change log must not route"
        );
    }

    #[test]
    fn make_decode_frame_applies_fallback_fps_when_missing() {
        let info = VideoInfo {
            codec: VideoCodec::H264,
            width: 640,
            height: 480,
            framerate_num: 0,
            framerate_den: 0,
        };
        let pkt = DemuxedPacket {
            data: Vec::new(),
            pts_ns: 0,
            dts_ns: None,
            duration_ns: None,
            is_keyframe: true,
        };
        // Round-trip construction; the fallback fps arithmetic is
        // exercised inside `make_decode_frame` and any failure
        // bubbles out via the `VideoFrameProxy::new` expect.
        let _ = make_decode_frame("src", &pkt, &info);
    }
}
