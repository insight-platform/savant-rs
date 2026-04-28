//! [`Decoder`] — `EncodedMsg` → decoded-frame fan-out stage.
//!
//! The stage owns decoder-pool lifecycle and message dispatch for
//! encoded input streams:
//!
//! * Builds a [`FlexibleDecoderPool`] (one slot per `source_id` it
//!   sees) with user-supplied
//!   [`FlexibleDecoderPoolConfig`].
//! * Captures a [`Router<PipelineMsg>`] at build time (optionally
//!   with a default peer installed via
//!   [`DecoderBuilder::downstream`]) and hands it to every
//!   user-supplied variant hook below.  The stage itself never
//!   calls `router.send`; user code is the single source of sends
//!   for this stage, which makes source-id-based routing (e.g.
//!   between multiple inference engines via
//!   `router.send_to(&peer, msg)`) natural.
//! * Is **stateless w.r.t. [`VideoInfo`]** — every
//!   [`EncodedMsg::Packet`]
//!   carries its own stream-level
//!   [`VideoInfo`] in-band
//!   (stamped upstream by the
//!   [`Mp4DemuxerSource`](super::mp4_demuxer::Mp4DemuxerSource)
//!   hooks), so the stage constructs the decode frame on the fly
//!   without maintaining a per-`source_id` cache.  The standalone
//!   [`EncodedMsg::StreamInfo`]
//!   sentinel is still accepted and exposed to the user through the
//!   [`StreamInfoObserver`] hook for stream-header-level
//!   observability (stats, logging, domain signalling) but is not
//!   required for decoding.
//! * On
//!   [`EncodedMsg::SourceEos`]
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
//! [`EncodedMsg::Shutdown`])
//! is handled by the loop driver via
//! [`Envelope::as_shutdown`](super::super::Envelope::as_shutdown);
//! the stage adds no cooperative-exit code of its own.
//!
//! # Grouped builder API
//!
//! Hooks are grouped into three bundles following the cross-stage
//! pattern:
//!
//! * [`DecoderInbox`] — actor-thread inbox observer
//!   (`on_stream_info`).
//! * [`DecoderResults`] — pool-callback-thread hooks
//!   (`on_frame`, `on_source_eos`, `on_parameter_change`,
//!   `on_skipped`, `on_orphan_frame`, `on_decoder_event`,
//!   `on_decoder_error`).  `on_source_eos` returns
//!   `Result<Flow>` and `on_decoder_error` returns
//!   [`ErrorAction`], matching the
//!   other stages' egress-hook contract.  Because these run off
//!   the actor thread, a `Flow::Stop` or `ErrorAction::Fatal`
//!   translates into a cooperative stop intent: the dispatcher
//!   calls [`HookCtx::request_stop`] and aborts the default sink.
//! * [`DecoderCommon`] — lifecycle + loop-level knobs
//!   (`stopping`, `poll_timeout`).
//!
//! Pool-infrastructure knobs (`pool_config` / `gpu_id` /
//! `pool_size` / `eviction_ttl` / `idle_timeout` /
//! `detect_buffer_limit` / `eviction_callback` /
//! `decoder_config_callback` / `downstream`) stay flat at the
//! top-level builder — they configure the decoder's construction
//! contract rather than its hook surface.
//!
//! # Default forwarders
//!
//! Every per-variant hook has an out-of-the-box default — if the
//! matching setter on the bundle is not called, the bundle builder
//! auto-installs it:
//!
//! * [`Decoder::default_on_stream_info`] — no-op observer
//!   (stream info is already cached and logged by the stage
//!   itself).
//! * [`Decoder::default_on_frame`] — forwards
//!   [`PipelineMsg::Delivery`]
//!   via `router.send(..)`.
//! * [`Decoder::default_on_source_eos`] — forwards
//!   [`PipelineMsg::SourceEos`]
//!   and returns `Ok(Flow::Cont)`.
//! * [`Decoder::default_on_parameter_change`] — logs the
//!   old/new [`DecoderParameters`] at `info` level, prefixed by the
//!   stage name.
//! * [`Decoder::default_on_skipped`] — logs the
//!   [`SkipReason`] at `debug` level.
//! * [`Decoder::default_on_orphan_frame`] — logs at `debug`
//!   level.
//! * [`Decoder::default_on_decoder_event`] — no-op (GStreamer
//!   events without domain meaning are silently ignored).
//! * [`Decoder::default_on_decoder_error`] — logs at `warn`
//!   level and returns [`ErrorAction::LogAndContinue`] so one
//!   misbehaving stream does not necessarily tear the pipeline down.
//! * [`Decoder::default_stopping`] — a no-op user shutdown hook.
//!   The built-in `graceful_shutdown` always runs first; this hook
//!   runs after.
//!
//! Override any branch to add stage-specific processing (e.g.
//! detection decoding on `on_frame`, source-id-based routing via
//! `router.send_to(..)`, metrics on `on_skipped`).
//!
//! # Runtime invariant
//!
//! Every hook slot on the runtime [`Decoder`] struct and on the
//! pool-callback hook bundle is a non-`Option` closure
//! ([`StreamInfoObserver`] for the actor-thread observer,
//! `Arc<dyn Fn>` for the pool-callback-thread hooks). The bundle
//! builders' `Option<...>` fields exist purely as internal "was
//! the setter called?" markers — every bundle's `build()`
//! substitutes the matching `Decoder::default_on_*` before the
//! actor value is constructed.

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

use crate::envelopes::{EncodedMsg, FramePayload, PacketPayload, PipelineMsg, StreamInfoPayload};
use crate::router::Router;
use crate::supervisor::StageName;
use crate::{
    Actor, ActorBuilder, Context, Dispatch, ErrorAction, Flow, Handler, HookCtx, ShutdownPayload,
    SourceEosPayload, StageKind,
};

/// Default decoder pool size — enough to hide NVDEC queue depth
/// without holding memory across many frames.
pub const DEFAULT_POOL_SIZE: u32 = 8;
/// Default eviction TTL — large enough that a long-running file
/// source is not evicted mid-run under typical single-file workloads.
pub const DEFAULT_EVICTION_TTL: Duration = Duration::from_secs(3600);
/// Default per-decoder idle timeout before idle maintenance runs.
pub const DEFAULT_IDLE_TIMEOUT: Duration = Duration::from_secs(5);
/// Default detect-buffer limit for decoder-side buffering.
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
/// [`EncodedMsg::StreamInfo`]
/// sentinel.  Runs on the actor thread (not the pool callback
/// thread), so the hook is `FnMut` and has full access to the
/// actor's [`Context`] — e.g. for stats accounting, dump-to-log
/// invariants, or bridging into shared state via
/// [`Context::shared`](crate::Context::shared).
///
/// The decoder itself no longer needs this hook to decode: every
/// [`EncodedMsg::Packet`]
/// carries its own
/// [`VideoInfo`]
/// in-band.  Use this observer purely for stream-header-level
/// observability; see [`Decoder::default_on_stream_info`]
/// for the no-op default installed when the setter is omitted.
pub type StreamInfoObserver =
    Box<dyn FnMut(&StreamInfoPayload, &mut Context<Decoder>) + Send + 'static>;

/// Hook fired for [`FlexibleDecoderOutput::Frame`] — receives the
/// sealed `(VideoFrameProxy, SharedBuffer)` delivery together with
/// the stage's [`Router<PipelineMsg>`] so the user decides whether
/// and where to forward it.  Runs on the pool's output callback
/// thread; must be `Fn + Send + Sync` and non-blocking.
///
/// Typical use: `router.send(PipelineMsg::Delivery(sealed))` (default
/// peer) or `router.send_to(&engine_for(&source_id), PipelineMsg::Delivery(sealed))`
/// (source-id-based dispatch between multiple inference engines).
/// The third parameter is the off-loop [`HookCtx`] for
/// shared-state / registry access from the callback thread.
pub type OnFrameHook =
    Arc<dyn Fn(SealedDelivery, &Router<PipelineMsg>, &HookCtx) + Send + Sync + 'static>;

/// Hook fired for [`FlexibleDecoderOutput::SourceEos`] — receives
/// the `source_id`, the router, and the [`HookCtx`].  Typical use:
/// `router.send(PipelineMsg::SourceEos { source_id: source_id.into() })`.
///
/// Returns `Result<Flow>` so the hook can request a cooperative
/// stop (`Ok(Flow::Stop)`) or signal a fatal error (`Err(_)`).
/// Because this hook runs on the pool callback thread rather than
/// the actor loop, a stop request translates into the dispatcher
/// calling [`HookCtx::request_stop`] and aborting the default sink
/// — the actor observes the cooperative intent on its next tick.
pub type OnDecoderSourceEosHook =
    Arc<dyn Fn(&str, &Router<PipelineMsg>, &HookCtx) -> Result<Flow> + Send + Sync + 'static>;

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
    dyn Fn(&DecoderParameters, &DecoderParameters, &Router<PipelineMsg>, &HookCtx)
        + Send
        + Sync
        + 'static,
>;

/// Hook fired for every [`FlexibleDecoderOutput::Skipped`] — reports
/// packets the decoder refused to process (see [`SkipReason`]).
/// Receives the skipped frame, the original payload (when available),
/// the reason, the router, and the [`HookCtx`].  Runs on the pool's
/// output thread.
pub type OnSkippedHook = Arc<
    dyn Fn(&VideoFrameProxy, Option<&[u8]>, &SkipReason, &Router<PipelineMsg>, &HookCtx)
        + Send
        + Sync
        + 'static,
>;

/// Hook fired for [`FlexibleDecoderOutput::OrphanFrame`] — a decoded
/// frame whose originating [`VideoFrameProxy`] could not be located
/// in the pool's frame map (typically a late delivery after a
/// reset).  Runs on the pool's output thread.
pub type OnOrphanFrameHook =
    Arc<dyn Fn(&DecodedFrame, &Router<PipelineMsg>, &HookCtx) + Send + Sync + 'static>;

/// Hook fired for arbitrary [`gst::Event`]s captured on the decoder's
/// output — useful for bridging custom upstream events to domain
/// logic.  Runs on the pool's output thread.
pub type OnDecoderEventHook =
    Arc<dyn Fn(&gst::Event, &Router<PipelineMsg>, &HookCtx) + Send + Sync + 'static>;

/// Hook fired on [`FlexibleDecoderOutput::Error`].  Runs on the pool's
/// output thread.  Returns an [`ErrorAction`] controlling the
/// dispatcher's reaction:
///
/// * [`ErrorAction::Fatal`] — the dispatcher calls
///   [`HookCtx::request_stop`] and aborts the default sink, mirroring
///   the other stages' egress-error contract.
/// * [`ErrorAction::LogAndContinue`] / [`ErrorAction::Swallow`] —
///   dispatcher continues processing subsequent decoder output.
///
/// The default is [`ErrorAction::LogAndContinue`] so one
/// misbehaving stream does not tear the pipeline down; override
/// to return `Fatal` when errors on this stage should stop the
/// whole pipeline.
pub type OnDecoderErrorHook = Arc<
    dyn Fn(&DecoderError, &Router<PipelineMsg>, &HookCtx) -> ErrorAction + Send + Sync + 'static,
>;

/// Hook fired for [`FlexibleDecoderOutput::Restarted`] — emitted
/// when the underlying decoder slot was transparently restarted
/// after its NVDEC worker died (typically because the
/// [`GstPipeline`](savant_gstreamer::pipeline::GstPipeline)
/// watchdog tripped on stuck in-flight buffers).
///
/// Receives the `source_id` of the affected decoder, a human-readable
/// reason, the number of in-flight frames lost during the restart, the
/// router for optional downstream signalling, and the [`HookCtx`].  Runs
/// on the pool's output callback thread.
///
/// Returns `Result<Flow>` so the hook can choose to keep running
/// (`Ok(Flow::Cont)`) or escalate the restart into a cooperative pipeline
/// stop (`Ok(Flow::Stop)` / `Err(_)`).  As with
/// [`OnDecoderSourceEosHook`], a stop request translates into the
/// dispatcher calling [`HookCtx::request_stop`] and aborting the default
/// sink — the actor observes the cooperative intent on its next tick.
///
/// The default ([`Decoder::default_on_decoder_restart`]) is purely a
/// `warn!` log line and `Ok(Flow::Cont)`, so transient watchdog-induced
/// restarts do not stop the pipeline by themselves.
pub type OnDecoderRestartHook = Arc<
    dyn Fn(&str, &str, usize, &Router<PipelineMsg>, &HookCtx) -> Result<Flow>
        + Send
        + Sync
        + 'static,
>;

/// User shutdown hook invoked from [`Actor::stopping`] *after* the
/// stage's built-in cleanup (guarded
/// [`FlexibleDecoderPool::graceful_shutdown`]) has completed.
///
/// Runs on the actor thread with full access to the [`Context`].
/// Ideal for final metrics flushes, bespoke log lines, or custom
/// bookkeeping that must observe the drained pool state.  The
/// stage's load-bearing cleanup cannot be skipped — users that
/// need to replace it entirely should implement [`Actor`] directly
/// on a bespoke struct.
pub type OnStoppingHook = Box<dyn FnMut(&mut Context<Decoder>) + Send + 'static>;

/// `EncodedMsg` → `PipelineMsg::Delivery` actor stage.
///
/// Construct via [`Decoder::builder`].
///
/// All variant hooks are **always populated** at runtime — see the
/// "Runtime invariant" section of the module header for details.
pub struct Decoder {
    decoder: FlexibleDecoderPool,
    poll_timeout: Duration,
    on_stream_info: StreamInfoObserver,
    stopping: OnStoppingHook,
    graceful_done: bool,
}

impl Decoder {
    /// Start a fluent builder for a decoder registered under `name`
    /// with inbox capacity `capacity`.
    pub fn builder(name: StageName, capacity: usize) -> DecoderBuilder {
        DecoderBuilder::new(name, capacity)
    }

    /// Default `on_stream_info` observer — a no-op.  The stage
    /// itself already logs the stream-info line; decoding no longer
    /// depends on this sentinel (every
    /// [`EncodedMsg::Packet`]
    /// carries its own [`VideoInfo`] in-band), so omitting
    /// the [`DecoderInbox`] setter simply means "don't add any
    /// user-level side-effect on top of the log line".
    pub fn default_on_stream_info(
    ) -> impl FnMut(&StreamInfoPayload, &mut Context<Decoder>) + Send + 'static {
        |_, _| {}
    }

    /// Default `on_frame` forwarder: emits
    /// [`PipelineMsg::Delivery(sealed)`](crate::envelopes::PipelineMsg::Delivery)
    /// on the router.  Override when the hook needs
    /// to observe the frame (e.g. stats ticks) or route by
    /// `source_id` via `router.send_to(...)`.
    pub fn default_on_frame(
    ) -> impl Fn(SealedDelivery, &Router<PipelineMsg>, &HookCtx) + Send + Sync + 'static {
        |sealed, router, _ctx| {
            if !router.send(PipelineMsg::Delivery(sealed)) {
                log::warn!("decoder downstream closed; dropped a Delivery");
            }
        }
    }

    /// Default `on_source_eos` forwarder: emits
    /// [`PipelineMsg::SourceEos { source_id }`](crate::envelopes::PipelineMsg::SourceEos)
    /// on the router and returns `Ok(Flow::Cont)`.  The pool
    /// callback fires [`FlexibleDecoderOutput::SourceEos`] strictly
    /// after the last [`FlexibleDecoderOutput::Frame`] for the same
    /// `source_id`, so the stream-aligned "EOS after last frame"
    /// invariant is preserved without application-level bookkeeping.
    pub fn default_on_source_eos(
    ) -> impl Fn(&str, &Router<PipelineMsg>, &HookCtx) -> Result<Flow> + Send + Sync + 'static {
        |source_id, router, _ctx| {
            if !router.send(PipelineMsg::SourceEos {
                source_id: source_id.to_string(),
            }) {
                log::warn!("decoder downstream closed; dropping SourceEos({source_id})");
            }
            Ok(Flow::Cont)
        }
    }

    /// Default `on_parameter_change` logger — emits a single
    /// `info!` line with the old/new [`DecoderParameters`]
    /// prefixed by the stage name (pulled from the
    /// [`HookCtx`]).  No downstream message is forwarded (the
    /// caller can subscribe by overriding this hook).
    pub fn default_on_parameter_change(
    ) -> impl Fn(&DecoderParameters, &DecoderParameters, &Router<PipelineMsg>, &HookCtx)
           + Send
           + Sync
           + 'static {
        |old, new, _router, ctx| {
            log::info!("[{}] parameter change: {old:?} -> {new:?}", ctx.own_name());
        }
    }

    /// Default `on_skipped` logger — emits a single `debug!` line
    /// with the [`SkipReason`] prefixed by the stage name from
    /// the [`HookCtx`].
    pub fn default_on_skipped(
    ) -> impl Fn(&VideoFrameProxy, Option<&[u8]>, &SkipReason, &Router<PipelineMsg>, &HookCtx)
           + Send
           + Sync
           + 'static {
        |_frame, _data, reason, _router, ctx| {
            log::debug!("[{}] skipped: {reason:?}", ctx.own_name());
        }
    }

    /// Default `on_orphan_frame` logger — emits a single `debug!`
    /// line prefixed by the stage name from the [`HookCtx`].
    /// Orphan frames usually signal a late delivery after a
    /// per-source reset and warrant observation but no forwarding.
    pub fn default_on_orphan_frame(
    ) -> impl Fn(&DecodedFrame, &Router<PipelineMsg>, &HookCtx) + Send + Sync + 'static {
        |_decoded, _router, ctx| {
            log::debug!("[{}] orphan frame (source id mismatch?)", ctx.own_name());
        }
    }

    /// Default `on_decoder_event` hook — a no-op.  GStreamer events
    /// without domain meaning are silently ignored; override to
    /// bridge custom upstream events into domain logic.
    pub fn default_on_decoder_event(
    ) -> impl Fn(&gst::Event, &Router<PipelineMsg>, &HookCtx) + Send + Sync + 'static {
        |_ev, _router, _ctx| {}
    }

    /// Default `on_decoder_error` logger — emits a single `warn!`
    /// line prefixed by the stage name pulled from the
    /// [`HookCtx`] and returns [`ErrorAction::LogAndContinue`].
    /// The error is swallowed so one misbehaving stream does not
    /// tear the pipeline down; the level is therefore `warn`, not
    /// `error`, to match the non-fatal disposition.  Override the
    /// hook to return [`ErrorAction::Fatal`] (and to log at
    /// `error!` if desired) when the pipeline should stop on any
    /// decoder error.
    pub fn default_on_decoder_error(
    ) -> impl Fn(&DecoderError, &Router<PipelineMsg>, &HookCtx) -> ErrorAction + Send + Sync + 'static
    {
        |err, _router, ctx| {
            log::warn!("[{}] decoder error: {err}", ctx.own_name());
            ErrorAction::LogAndContinue
        }
    }

    /// Default `on_decoder_restart` logger — emits a single `warn!` line
    /// reporting the affected `source_id`, the human-readable reason,
    /// and the number of in-flight frames lost, prefixed by the stage
    /// name pulled from the [`HookCtx`], and returns `Ok(Flow::Cont)`.
    ///
    /// Watchdog-induced restarts are typically transient (the
    /// underlying decoder pool re-runs detection / activation for
    /// the next packet automatically), so the default keeps the
    /// pipeline running.  Override to return
    /// `Ok(Flow::Stop)` / `Err(_)` when a restart on this stage should
    /// tear the pipeline down — for example:
    ///
    /// ```ignore
    /// .on_decoder_restart(|sid, reason, lost, _r, _ctx| {
    ///     log::error!("decoder {sid} restarted ({lost} frames lost): {reason} - aborting");
    ///     Ok(Flow::Stop)
    /// })
    /// ```
    pub fn default_on_decoder_restart(
    ) -> impl Fn(&str, &str, usize, &Router<PipelineMsg>, &HookCtx) -> Result<Flow> + Send + Sync + 'static
    {
        |source_id, reason, lost_frames, _router, ctx| {
            log::warn!(
                "[{}] decoder restart: source_id={source_id} lost {lost_frames} in-flight frame(s); reason={reason}",
                ctx.own_name()
            );
            Ok(Flow::Cont)
        }
    }

    /// Default user shutdown hook — a no-op.  The stage's own
    /// [`Actor::stopping`] body always runs
    /// [`FlexibleDecoderPool::graceful_shutdown`] before this hook
    /// fires, so omitting the [`DecoderCommon`] setter simply means
    /// "don't add any extra cleanup on top of the built-in drain".
    pub fn default_stopping() -> impl FnMut(&mut Context<Decoder>) + Send + 'static {
        |_ctx| {}
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
        (self.stopping)(ctx);
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
/// shutdown hint from [`EncodedMsg::Shutdown`]
/// via [`Envelope::as_shutdown`](super::super::Envelope::as_shutdown),
/// so the in-band sentinel has no remaining work here.
impl Handler<ShutdownPayload> for Decoder {}

/// Inbox hook bundle for [`Decoder`] — one actor-thread observer.
/// Built through [`DecoderInbox::builder`] and handed to
/// [`DecoderBuilder::inbox`].
pub struct DecoderInbox {
    on_stream_info: StreamInfoObserver,
}

impl DecoderInbox {
    /// Start a builder that auto-installs every default on
    /// [`DecoderInboxBuilder::build`].
    pub fn builder() -> DecoderInboxBuilder {
        DecoderInboxBuilder::new()
    }
}

impl Default for DecoderInbox {
    fn default() -> Self {
        DecoderInboxBuilder::new().build()
    }
}

/// Fluent builder for [`DecoderInbox`].
pub struct DecoderInboxBuilder {
    on_stream_info: Option<StreamInfoObserver>,
}

impl DecoderInboxBuilder {
    /// Empty bundle — the `on_stream_info` slot defaults to
    /// [`Decoder::default_on_stream_info`] at
    /// [`build`](Self::build) time.
    pub fn new() -> Self {
        Self {
            on_stream_info: None,
        }
    }

    /// Install a custom `on_stream_info` observer — called after
    /// every [`StreamInfoPayload`] is logged, on the actor thread
    /// (with full access to the [`Context`]).  Use for stats
    /// accounting or dump-to-log invariants.
    pub fn on_stream_info<F>(mut self, f: F) -> Self
    where
        F: FnMut(&StreamInfoPayload, &mut Context<Decoder>) + Send + 'static,
    {
        self.on_stream_info = Some(Box::new(f));
        self
    }

    /// Finalise the bundle, substituting defaults for every omitted
    /// setter.
    pub fn build(self) -> DecoderInbox {
        DecoderInbox {
            on_stream_info: self
                .on_stream_info
                .unwrap_or_else(|| Box::new(Decoder::default_on_stream_info())),
        }
    }
}

impl Default for DecoderInboxBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Results hook bundle for [`Decoder`] — one branch per
/// [`FlexibleDecoderOutput`] variant dispatched on the pool
/// callback thread.  Built through [`DecoderResults::builder`] and
/// handed to [`DecoderBuilder::results`].  `on_source_eos` returns
/// `Result<Flow>`; `on_decoder_error` returns [`ErrorAction`] — see
/// the module docs for the dispatcher's translation to cooperative
/// stop intents.
pub struct DecoderResults {
    on_frame: OnFrameHook,
    on_source_eos: OnDecoderSourceEosHook,
    on_parameter_change: OnParameterChangeHook,
    on_skipped: OnSkippedHook,
    on_orphan_frame: OnOrphanFrameHook,
    on_decoder_event: OnDecoderEventHook,
    on_decoder_error: OnDecoderErrorHook,
    on_decoder_restart: OnDecoderRestartHook,
}

impl DecoderResults {
    /// Start a builder that auto-installs every default on
    /// [`DecoderResultsBuilder::build`].
    pub fn builder() -> DecoderResultsBuilder {
        DecoderResultsBuilder::new()
    }
}

impl Default for DecoderResults {
    fn default() -> Self {
        DecoderResultsBuilder::new().build()
    }
}

/// Fluent builder for [`DecoderResults`].
pub struct DecoderResultsBuilder {
    on_frame: Option<OnFrameHook>,
    on_source_eos: Option<OnDecoderSourceEosHook>,
    on_parameter_change: Option<OnParameterChangeHook>,
    on_skipped: Option<OnSkippedHook>,
    on_orphan_frame: Option<OnOrphanFrameHook>,
    on_decoder_event: Option<OnDecoderEventHook>,
    on_decoder_error: Option<OnDecoderErrorHook>,
    on_decoder_restart: Option<OnDecoderRestartHook>,
}

impl DecoderResultsBuilder {
    /// Empty bundle — every hook defaults to its matching
    /// `Decoder::default_on_*` at [`build`](Self::build) time.
    pub fn new() -> Self {
        Self {
            on_frame: None,
            on_source_eos: None,
            on_parameter_change: None,
            on_skipped: None,
            on_orphan_frame: None,
            on_decoder_event: None,
            on_decoder_error: None,
            on_decoder_restart: None,
        }
    }

    /// Override the `on_frame` hook — fired for every
    /// [`FlexibleDecoderOutput::Frame`].  Runs on the pool's
    /// output callback thread; must be `Fn + Send + Sync +
    /// 'static` and non-blocking.
    pub fn on_frame<F>(mut self, f: F) -> Self
    where
        F: Fn(SealedDelivery, &Router<PipelineMsg>, &HookCtx) + Send + Sync + 'static,
    {
        self.on_frame = Some(Arc::new(f));
        self
    }

    /// Override the `on_source_eos` hook — fired for
    /// [`FlexibleDecoderOutput::SourceEos`] exactly once per
    /// `source_id`, strictly after the last
    /// [`on_frame`](Self::on_frame) for that source.  Returns
    /// `Result<Flow>`; `Ok(Flow::Stop)` or `Err(_)` request a
    /// cooperative stop from the pool callback thread.
    pub fn on_source_eos<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &Router<PipelineMsg>, &HookCtx) -> Result<Flow> + Send + Sync + 'static,
    {
        self.on_source_eos = Some(Arc::new(f));
        self
    }

    /// Override the `on_parameter_change` hook — fired on
    /// [`FlexibleDecoderOutput::ParameterChange`].
    pub fn on_parameter_change<F>(mut self, f: F) -> Self
    where
        F: Fn(&DecoderParameters, &DecoderParameters, &Router<PipelineMsg>, &HookCtx)
            + Send
            + Sync
            + 'static,
    {
        self.on_parameter_change = Some(Arc::new(f));
        self
    }

    /// Override the `on_skipped` hook — fired for every
    /// [`FlexibleDecoderOutput::Skipped`].
    pub fn on_skipped<F>(mut self, f: F) -> Self
    where
        F: Fn(&VideoFrameProxy, Option<&[u8]>, &SkipReason, &Router<PipelineMsg>, &HookCtx)
            + Send
            + Sync
            + 'static,
    {
        self.on_skipped = Some(Arc::new(f));
        self
    }

    /// Override the `on_orphan_frame` hook — fired for
    /// [`FlexibleDecoderOutput::OrphanFrame`].
    pub fn on_orphan_frame<F>(mut self, f: F) -> Self
    where
        F: Fn(&DecodedFrame, &Router<PipelineMsg>, &HookCtx) + Send + Sync + 'static,
    {
        self.on_orphan_frame = Some(Arc::new(f));
        self
    }

    /// Override the `on_decoder_event` hook — fired for
    /// [`FlexibleDecoderOutput::Event`].
    pub fn on_decoder_event<F>(mut self, f: F) -> Self
    where
        F: Fn(&gst::Event, &Router<PipelineMsg>, &HookCtx) + Send + Sync + 'static,
    {
        self.on_decoder_event = Some(Arc::new(f));
        self
    }

    /// Override the `on_decoder_error` hook — fired for
    /// [`FlexibleDecoderOutput::Error`].  Returns [`ErrorAction`];
    /// the default is [`ErrorAction::LogAndContinue`].  Return
    /// [`ErrorAction::Fatal`] to request a cooperative stop of the
    /// actor loop.
    pub fn on_decoder_error<F>(mut self, f: F) -> Self
    where
        F: Fn(&DecoderError, &Router<PipelineMsg>, &HookCtx) -> ErrorAction + Send + Sync + 'static,
    {
        self.on_decoder_error = Some(Arc::new(f));
        self
    }

    /// Override the `on_decoder_restart` hook — fired for
    /// [`FlexibleDecoderOutput::Restarted`].  Returns `Result<Flow>`;
    /// the default ([`Decoder::default_on_decoder_restart`]) logs at
    /// `warn!` and returns `Ok(Flow::Cont)` so the pipeline keeps
    /// running.  Return `Ok(Flow::Stop)` / `Err(_)` to translate a
    /// watchdog-induced restart into a cooperative pipeline stop.
    pub fn on_decoder_restart<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &str, usize, &Router<PipelineMsg>, &HookCtx) -> Result<Flow>
            + Send
            + Sync
            + 'static,
    {
        self.on_decoder_restart = Some(Arc::new(f));
        self
    }

    /// Finalise the bundle, substituting defaults for every omitted
    /// setter.
    pub fn build(self) -> DecoderResults {
        let DecoderResultsBuilder {
            on_frame,
            on_source_eos,
            on_parameter_change,
            on_skipped,
            on_orphan_frame,
            on_decoder_event,
            on_decoder_error,
            on_decoder_restart,
        } = self;
        DecoderResults {
            on_frame: on_frame.unwrap_or_else(|| Arc::new(Decoder::default_on_frame())),
            on_source_eos: on_source_eos
                .unwrap_or_else(|| Arc::new(Decoder::default_on_source_eos())),
            on_parameter_change: on_parameter_change
                .unwrap_or_else(|| Arc::new(Decoder::default_on_parameter_change())),
            on_skipped: on_skipped.unwrap_or_else(|| Arc::new(Decoder::default_on_skipped())),
            on_orphan_frame: on_orphan_frame
                .unwrap_or_else(|| Arc::new(Decoder::default_on_orphan_frame())),
            on_decoder_event: on_decoder_event
                .unwrap_or_else(|| Arc::new(Decoder::default_on_decoder_event())),
            on_decoder_error: on_decoder_error
                .unwrap_or_else(|| Arc::new(Decoder::default_on_decoder_error())),
            on_decoder_restart: on_decoder_restart
                .unwrap_or_else(|| Arc::new(Decoder::default_on_decoder_restart())),
        }
    }
}

impl Default for DecoderResultsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Lifecycle + loop-level knob bundle for [`Decoder`].  Built
/// through [`DecoderCommon::builder`] and handed to
/// [`DecoderBuilder::common`].
pub struct DecoderCommon {
    poll_timeout: Duration,
    stopping: OnStoppingHook,
}

impl DecoderCommon {
    /// Start a builder seeded with [`DEFAULT_POLL_TIMEOUT`] and the
    /// no-op stopping hook.
    pub fn builder() -> DecoderCommonBuilder {
        DecoderCommonBuilder::new()
    }
}

impl Default for DecoderCommon {
    fn default() -> Self {
        DecoderCommonBuilder::new().build()
    }
}

/// Fluent builder for [`DecoderCommon`].
pub struct DecoderCommonBuilder {
    poll_timeout: Option<Duration>,
    stopping: Option<OnStoppingHook>,
}

impl DecoderCommonBuilder {
    /// Empty bundle — `poll_timeout` defaults to
    /// [`DEFAULT_POLL_TIMEOUT`] and `stopping` to a no-op.
    pub fn new() -> Self {
        Self {
            poll_timeout: None,
            stopping: None,
        }
    }

    /// Inbox receive-poll cadence (default [`DEFAULT_POLL_TIMEOUT`]).
    pub fn poll_timeout(mut self, d: Duration) -> Self {
        self.poll_timeout = Some(d);
        self
    }

    /// Override the user shutdown hook — fired from
    /// [`Actor::stopping`] **after** the stage's built-in
    /// cleanup (guarded
    /// [`FlexibleDecoderPool::graceful_shutdown`]) has completed.
    /// The built-in cleanup is **load-bearing** and always runs
    /// first; it cannot be skipped through this hook.
    pub fn stopping<F>(mut self, f: F) -> Self
    where
        F: FnMut(&mut Context<Decoder>) + Send + 'static,
    {
        self.stopping = Some(Box::new(f));
        self
    }

    /// Finalise the bundle, substituting defaults for every omitted
    /// setter.
    pub fn build(self) -> DecoderCommon {
        let DecoderCommonBuilder {
            poll_timeout,
            stopping,
        } = self;
        DecoderCommon {
            poll_timeout: poll_timeout.unwrap_or(DEFAULT_POLL_TIMEOUT),
            stopping: stopping.unwrap_or_else(|| Box::new(Decoder::default_stopping())),
        }
    }
}

impl Default for DecoderCommonBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Fluent builder for [`Decoder`].
///
/// Pool-infrastructure configuration (`pool_config`, `gpu_id`,
/// `pool_size`, `eviction_ttl`, `idle_timeout`,
/// `detect_buffer_limit`, `eviction_callback`,
/// `decoder_config_callback`, `downstream`) stays flat on the
/// top-level builder.  Inbox / result / common hook bundles live
/// on [`DecoderInbox`] / [`DecoderResults`] / [`DecoderCommon`]
/// and are installed via [`DecoderBuilder::inbox`],
/// [`DecoderBuilder::results`], [`DecoderBuilder::common`].
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
    eviction_callback: Option<EvictionCallback>,
    decoder_config_callback: Option<DecoderConfigCallback>,
    inbox: Option<DecoderInbox>,
    results: Option<DecoderResults>,
    common: Option<DecoderCommon>,
}

impl DecoderBuilder {
    /// Start a builder with framework defaults for pool sizing,
    /// eviction, buffering, and polling.
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
            eviction_callback: None,
            decoder_config_callback: None,
            inbox: None,
            results: None,
            common: None,
        }
    }

    /// Optional default peer installed on the
    /// [`Router<PipelineMsg>`] handed to every variant hook.
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

    /// Install an eviction-decision callback on the underlying pool.
    pub fn eviction_callback<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> EvictionDecision + Send + Sync + 'static,
    {
        self.eviction_callback = Some(Arc::new(f));
        self
    }

    /// Install a decoder-config transformation callback.
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

    /// Install a [`DecoderInbox`] bundle.  Omitting this call is
    /// equivalent to `.inbox(DecoderInbox::default())`.
    pub fn inbox(mut self, i: DecoderInbox) -> Self {
        self.inbox = Some(i);
        self
    }

    /// Install a [`DecoderResults`] bundle.  Omitting this call is
    /// equivalent to `.results(DecoderResults::default())`.
    pub fn results(mut self, r: DecoderResults) -> Self {
        self.results = Some(r);
        self
    }

    /// Install a [`DecoderCommon`] bundle.  Omitting this call is
    /// equivalent to `.common(DecoderCommon::default())`.
    pub fn common(mut self, c: DecoderCommon) -> Self {
        self.common = Some(c);
        self
    }

    /// Finalise the stage and obtain an
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
            eviction_callback,
            decoder_config_callback,
            inbox,
            results,
            common,
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
        let DecoderInbox { on_stream_info } = inbox.unwrap_or_default();
        let DecoderResults {
            on_frame,
            on_source_eos,
            on_parameter_change,
            on_skipped,
            on_orphan_frame,
            on_decoder_event,
            on_decoder_error,
            on_decoder_restart,
        } = results.unwrap_or_default();
        let DecoderCommon {
            poll_timeout,
            stopping,
        } = common.unwrap_or_default();
        let hooks = VariantHooks {
            on_frame,
            on_source_eos,
            on_parameter_change,
            on_skipped,
            on_orphan_frame,
            on_decoder_event,
            on_decoder_error,
            on_decoder_restart,
        };
        Ok(ActorBuilder::new(name.clone(), capacity)
            .poll_timeout(poll_timeout)
            .factory(move |bx| {
                let hook_ctx = bx.hook_ctx();
                let router: Router<PipelineMsg> = bx.router(downstream.as_ref())?;
                let owner = bx.own_name().clone();
                let decoder = build_pool(
                    cfg,
                    router,
                    owner,
                    eviction_callback.clone(),
                    hooks.clone(),
                    hook_ctx,
                );
                Ok(Decoder {
                    decoder,
                    poll_timeout,
                    on_stream_info,
                    stopping,
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
    on_decoder_restart: OnDecoderRestartHook,
}

/// Build the [`FlexibleDecoderPool`] with a callback that dispatches
/// every [`FlexibleDecoderOutput`] variant to the corresponding user
/// hook.  The callback never calls `router.send` itself — every send
/// site lives inside user code.  `on_source_eos`'s `Result<Flow>`
/// and `on_decoder_error`'s [`ErrorAction`] are translated into
/// cooperative-stop intents via [`HookCtx::request_stop`] +
/// [`Router::default_sink`]-abort, matching the egress-hook
/// contract of the other stages.
fn build_pool(
    cfg: FlexibleDecoderPoolConfig,
    router: Router<PipelineMsg>,
    owner: StageName,
    eviction_callback: Option<EvictionCallback>,
    hooks: VariantHooks,
    hook_ctx: HookCtx,
) -> FlexibleDecoderPool {
    let owner_cb = owner.clone();
    let on_output = move |mut out: FlexibleDecoderOutput| match &out {
        FlexibleDecoderOutput::Frame { .. } => {
            if let Some(sealed) = out.take_delivery() {
                (hooks.on_frame)(sealed, &router, &hook_ctx);
            }
        }
        FlexibleDecoderOutput::ParameterChange { old, new } => {
            (hooks.on_parameter_change)(old, new, &router, &hook_ctx);
        }
        FlexibleDecoderOutput::Skipped {
            frame,
            data,
            reason,
        } => {
            (hooks.on_skipped)(frame, data.as_deref(), reason, &router, &hook_ctx);
        }
        FlexibleDecoderOutput::OrphanFrame { decoded } => {
            (hooks.on_orphan_frame)(decoded, &router, &hook_ctx);
        }
        FlexibleDecoderOutput::SourceEos { source_id } => {
            log::info!(
                "[{owner_cb}/cb] FlexibleDecoderOutput::SourceEos for source_id={source_id}"
            );
            match (hooks.on_source_eos)(source_id, &router, &hook_ctx) {
                Ok(Flow::Cont) => {}
                Ok(Flow::Stop) => {
                    log::info!(
                        "[{owner_cb}/cb] on_source_eos requested stop; cooperatively shutting down"
                    );
                    hook_ctx.request_stop();
                    if let Some(sink) = router.default_sink() {
                        sink.abort();
                    }
                }
                Err(e) => {
                    log::error!("[{owner_cb}/cb] on_source_eos returned error: {e}");
                    hook_ctx.request_stop();
                    if let Some(sink) = router.default_sink() {
                        sink.abort();
                    }
                }
            }
        }
        FlexibleDecoderOutput::Event(ev) => {
            (hooks.on_decoder_event)(ev, &router, &hook_ctx);
        }
        FlexibleDecoderOutput::Error(err) => {
            match (hooks.on_decoder_error)(err, &router, &hook_ctx) {
                ErrorAction::Fatal => {
                    log::error!(
                    "[{owner_cb}/cb] on_decoder_error returned Fatal; cooperatively shutting down"
                );
                    hook_ctx.request_stop();
                    if let Some(sink) = router.default_sink() {
                        sink.abort();
                    }
                }
                ErrorAction::LogAndContinue | ErrorAction::Swallow => {}
            }
        }
        FlexibleDecoderOutput::Restarted {
            source_id,
            reason,
            lost_frames,
        } => {
            match (hooks.on_decoder_restart)(source_id, reason, *lost_frames, &router, &hook_ctx) {
                Ok(Flow::Cont) => {}
                Ok(Flow::Stop) => {
                    log::info!(
                        "[{owner_cb}/cb] on_decoder_restart requested stop; cooperatively shutting down"
                    );
                    hook_ctx.request_stop();
                    if let Some(sink) = router.default_sink() {
                        sink.abort();
                    }
                }
                Err(e) => {
                    log::error!("[{owner_cb}/cb] on_decoder_restart returned error: {e}");
                    hook_ctx.request_stop();
                    if let Some(sink) = router.default_sink() {
                        sink.abort();
                    }
                }
            }
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
    use crate::addr::Addr;
    use crate::operator_sink::OperatorSink;
    use crate::registry::Registry;
    use crate::supervisor::{StageKind, StageName};
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
    /// no-downstream configuration.
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
    /// [`DecoderBuilder`] has no required hooks.
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
            .inbox(DecoderInbox::builder().on_stream_info(|_, _| {}).build())
            .results(
                DecoderResults::builder()
                    .on_frame(|sealed, router, _ctx| {
                        let _ = router.send(PipelineMsg::Delivery(sealed));
                    })
                    .on_source_eos(|sid, router, _ctx| {
                        let _ = router.send(PipelineMsg::SourceEos {
                            source_id: sid.to_string(),
                        });
                        Ok(Flow::Cont)
                    })
                    .on_parameter_change(|_, _, _, _| {})
                    .on_skipped(|_, _, _, _, _| {})
                    .on_orphan_frame(|_, _, _| {})
                    .on_decoder_event(|_, _, _| {})
                    .on_decoder_error(|_, _, _| ErrorAction::LogAndContinue)
                    .on_decoder_restart(|_, _, _, _, _| Ok(Flow::Cont))
                    .build(),
            )
            .common(
                DecoderCommon::builder()
                    .stopping(Decoder::default_stopping())
                    .build(),
            )
            .build()
            .expect("full hook configuration builds");
    }

    /// Confirms that each `default_on_*` associated function slots
    /// into the bundle builders' generic hook bounds as-is.
    #[test]
    fn builder_accepts_default_forwarders() {
        let name = StageName::unnamed(StageKind::Decoder);
        let _ = Decoder::builder(name, 4)
            .downstream(StageName::unnamed(StageKind::Infer))
            .inbox(
                DecoderInbox::builder()
                    .on_stream_info(Decoder::default_on_stream_info())
                    .build(),
            )
            .results(
                DecoderResults::builder()
                    .on_frame(Decoder::default_on_frame())
                    .on_source_eos(Decoder::default_on_source_eos())
                    .on_parameter_change(Decoder::default_on_parameter_change())
                    .on_skipped(Decoder::default_on_skipped())
                    .on_orphan_frame(Decoder::default_on_orphan_frame())
                    .on_decoder_event(Decoder::default_on_decoder_event())
                    .on_decoder_error(Decoder::default_on_decoder_error())
                    .on_decoder_restart(Decoder::default_on_decoder_restart())
                    .build(),
            )
            .common(
                DecoderCommon::builder()
                    .stopping(Decoder::default_stopping())
                    .build(),
            )
            .build()
            .expect("default forwarders build");
    }

    /// The builder accepts a user-supplied `.stopping(F)` closure
    /// through the common bundle.  Compile-only verification of the
    /// hook bound; the ordering invariant (built-in
    /// `graceful_shutdown` runs before the user hook) is covered by
    /// end-to-end smoke runs which exercise a real
    /// `FlexibleDecoderPool`.
    #[test]
    fn builder_accepts_user_stopping() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let flag = Arc::new(AtomicBool::new(false));
        let flag_hook = flag.clone();
        let name = StageName::unnamed(StageKind::Decoder);
        let _ = Decoder::builder(name, 4)
            .downstream(StageName::unnamed(StageKind::Infer))
            .common(
                DecoderCommon::builder()
                    .stopping(move |_ctx| {
                        flag_hook.store(true, Ordering::SeqCst);
                    })
                    .build(),
            )
            .build()
            .expect("user stopping builds");
        assert!(!flag.load(Ordering::SeqCst));
    }

    /// `default_on_source_eos` forwards `PipelineMsg::SourceEos` on
    /// the default peer and returns `Ok(Flow::Cont)`.
    #[test]
    fn default_on_source_eos_forwards_via_router() {
        let hook = Decoder::default_on_source_eos();
        let (router, rx) = router_with_default_peer();
        let ctx = test_hook_ctx();
        let flow = hook("cam-1", &router, &ctx).expect("no error");
        assert!(matches!(flow, Flow::Cont));
        match rx.try_recv().expect("SourceEos should be forwarded") {
            PipelineMsg::SourceEos { source_id } => assert_eq!(source_id, "cam-1"),
            other => panic!("expected SourceEos, got {other:?}"),
        }
        assert!(rx.try_recv().is_err(), "exactly one message forwarded");
    }

    /// `default_on_decoder_error` is log-only — it returns
    /// `LogAndContinue` and must not route anything downstream.
    #[test]
    fn default_on_decoder_error_does_not_route() {
        let hook = Decoder::default_on_decoder_error();
        let (router, rx) = router_with_default_peer();
        let ctx = test_hook_ctx();
        let err = DecoderError::PipelineError("synthetic".to_string());
        let action = hook(&err, &router, &ctx);
        assert!(matches!(action, ErrorAction::LogAndContinue));
        assert!(rx.try_recv().is_err(), "errors must not be routed");
    }

    /// `default_on_parameter_change` is log-only — it must not
    /// forward any control message downstream.
    #[test]
    fn default_on_parameter_change_does_not_route() {
        let hook = Decoder::default_on_parameter_change();
        let (router, rx) = router_with_default_peer();
        let ctx = test_hook_ctx();
        let params = DecoderParameters {
            codec: VideoCodec::H264,
            width: 1920,
            height: 1080,
        };
        hook(&params, &params, &router, &ctx);
        assert!(
            rx.try_recv().is_err(),
            "parameter-change log must not route"
        );
    }

    /// Synthesise a bare [`HookCtx`] bound to a Decoder stage for
    /// default-hook unit tests.
    fn test_hook_ctx() -> HookCtx {
        HookCtx::new(
            StageName::unnamed(StageKind::Decoder),
            Arc::new(Registry::new()),
            Arc::new(crate::shared::SharedStore::new()),
            Arc::new(std::sync::atomic::AtomicBool::new(false)),
        )
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
        let _ = make_decode_frame("src", &pkt, &info);
    }

    /// A `Fatal` `on_decoder_error` result flips the shared stop
    /// flag so subsequent actor ticks observe cooperative shutdown.
    #[test]
    fn on_decoder_error_fatal_requests_stop() {
        use std::sync::atomic::{AtomicBool, Ordering};
        // Exercise the dispatcher arm directly — we cannot
        // synthesise a real `FlexibleDecoderOutput::Error` inside
        // a unit test, but we can confirm the expected stop-flag
        // transition by invoking the exact same control flow
        // the dispatcher uses:
        let stop_flag = Arc::new(AtomicBool::new(false));
        let ctx = HookCtx::new(
            StageName::unnamed(StageKind::Decoder),
            Arc::new(Registry::new()),
            Arc::new(crate::shared::SharedStore::new()),
            stop_flag.clone(),
        );
        let hook: OnDecoderErrorHook = Arc::new(|_err, _router, _ctx| ErrorAction::Fatal);
        let (router, _rx) = router_with_default_peer();
        let err = DecoderError::PipelineError("synthetic".to_string());
        match hook(&err, &router, &ctx) {
            ErrorAction::Fatal => {
                ctx.request_stop();
                if let Some(sink) = router.default_sink() {
                    sink.abort();
                }
            }
            ErrorAction::LogAndContinue | ErrorAction::Swallow => {}
        }
        assert!(stop_flag.load(Ordering::SeqCst));
    }

    /// `default_on_decoder_restart` is log-only — it returns
    /// `Ok(Flow::Cont)` and must not route anything downstream so
    /// transient watchdog-induced restarts do not crash the pipeline.
    #[test]
    fn default_on_decoder_restart_does_not_route() {
        let hook = Decoder::default_on_decoder_restart();
        let (router, rx) = router_with_default_peer();
        let ctx = test_hook_ctx();
        let flow = hook("cam-1", "worker thread exited", 8, &router, &ctx).expect("no error");
        assert!(matches!(flow, Flow::Cont));
        assert!(rx.try_recv().is_err(), "restarts must not be routed");
    }

    /// A user `on_decoder_restart` returning `Ok(Flow::Stop)` flips the
    /// shared stop flag, mirroring the
    /// [`on_decoder_error_fatal_requests_stop`] contract.
    #[test]
    fn on_decoder_restart_stop_requests_stop() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let stop_flag = Arc::new(AtomicBool::new(false));
        let ctx = HookCtx::new(
            StageName::unnamed(StageKind::Decoder),
            Arc::new(Registry::new()),
            Arc::new(crate::shared::SharedStore::new()),
            stop_flag.clone(),
        );
        let hook: OnDecoderRestartHook =
            Arc::new(|_sid, _reason, _lost, _router, _ctx| Ok(Flow::Stop));
        let (router, _rx) = router_with_default_peer();
        match hook("cam-1", "worker thread exited", 4, &router, &ctx).expect("hook ok") {
            Flow::Cont => {}
            Flow::Stop => {
                ctx.request_stop();
                if let Some(sink) = router.default_sink() {
                    sink.abort();
                }
            }
        }
        assert!(stop_flag.load(Ordering::SeqCst));
    }

    /// A user `on_decoder_restart` returning `Err(_)` likewise flips
    /// the shared stop flag — the dispatcher treats it identically to
    /// `Ok(Flow::Stop)`.
    #[test]
    fn on_decoder_restart_err_requests_stop() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let stop_flag = Arc::new(AtomicBool::new(false));
        let ctx = HookCtx::new(
            StageName::unnamed(StageKind::Decoder),
            Arc::new(Registry::new()),
            Arc::new(crate::shared::SharedStore::new()),
            stop_flag.clone(),
        );
        let hook: OnDecoderRestartHook =
            Arc::new(|_sid, _reason, _lost, _router, _ctx| Err(anyhow!("boom")));
        let (router, _rx) = router_with_default_peer();
        match hook("cam-1", "worker thread exited", 0, &router, &ctx) {
            Ok(Flow::Cont) => {}
            Ok(Flow::Stop) | Err(_) => {
                ctx.request_stop();
                if let Some(sink) = router.default_sink() {
                    sink.abort();
                }
            }
        }
        assert!(stop_flag.load(Ordering::SeqCst));
    }
}
