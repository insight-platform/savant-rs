//! [`Mp4DemuxerSource`] — an MP4 (and any other GStreamer-parsable
//! container) source that surfaces every demuxed output variant to
//! user code, along with the stage's [`Router<EncodedMsg>`].
//!
//! The stage owns the GStreamer demuxing loop and hides the
//! framework plumbing that every demux source otherwise has to
//! repeat:
//!
//! * `Arc<AtomicBool>` abort flag (becomes [`SourceContext::is_stopping`]).
//! * Manual exit-guard wiring (the framework installs it around
//!   [`Source::run`]).
//! * The first-error latch (still present, but hidden in the
//!   stage body).
//!
//! The stage itself never calls `router.send` — **every send
//! site for this stage lives in user code**.  That lets source-side
//! routing (for instance "pick a decoder pool by source_id" or
//! "broadcast the same packet to multiple decoders via
//! `router.send_to(&peer, msg)`") stay visible at the callsite.
//!
//! The user-facing surface is a fluent builder with grouped
//! hook bundles — [`Mp4DemuxerResults`] for the four demuxed-output
//! variants and [`Mp4DemuxerCommon`] for the user shutdown hook:
//!
//! ```ignore
//! sys.register_source(
//!     Mp4DemuxerSource::builder(StageName::unnamed(StageKind::Mp4Demux))
//!         .input(input_path)
//!         .source_id("cam1")
//!         .downstream(StageName::unnamed(StageKind::Decoder))
//!         .results(
//!             Mp4DemuxerResults::builder()
//!                 .on_packet(|pkt, info, source_id, router, _ctx| {
//!                     router.send(EncodedMsg::Packet {
//!                         source_id: source_id.to_string(),
//!                         info: *info,
//!                         packet: pkt,
//!                     });
//!                     Ok(())
//!                 })
//!                 .build(),
//!         )
//!         .build(),
//! )?;
//! ```
//!
//! # Default forwarders
//!
//! For the common "forward this variant to the default peer
//! unchanged" case, [`Mp4DemuxerSource`] ships four ready-made hook
//! closures:
//!
//! * [`Mp4DemuxerSource::default_on_stream_info`] — sends
//!   [`EncodedMsg::StreamInfo`].
//! * [`Mp4DemuxerSource::default_on_packet`] — sends
//!   [`EncodedMsg::Packet`].
//!   Use when the downstream decoder owns frame construction.
//! * [`Mp4DemuxerSource::default_on_packet_as_frame`] — constructs
//!   a [`VideoFrameProxy`](savant_core::primitives::frame::VideoFrameProxy)
//!   on the demuxer side (via
//!   [`make_decode_frame`](super::decoder::make_decode_frame)) and
//!   sends
//!   [`EncodedMsg::Frame`].
//!   Use when you want upstream frame construction so the decoder
//!   can stay stateless w.r.t. [`VideoInfo`].
//! * [`Mp4DemuxerSource::default_on_source_eos`] — sends
//!   [`EncodedMsg::SourceEos`].
//!
//! Pick exactly one of `default_on_packet` / `default_on_packet_as_frame`
//! per source — the two differ only in whether the
//! [`VideoFrameProxy`](savant_core::primitives::frame::VideoFrameProxy)
//! is built upstream or downstream.
//!
//! Because each one is an associated function that returns an owned
//! closure, the send still lives in user code (the user names the
//! forwarder at the call site) — the stage just factors out the
//! one-line body.  Reach for a hand-written closure when you need
//! to observe the variant, suppress it, or route via
//! `router.send_to(&peer, msg)`.
//!
//! ## Always-set invariant
//!
//! [`Mp4DemuxerSource`] stores its four variant hooks
//! (`on_stream_info`, `on_packet`, `on_source_eos`, `on_error`) as
//! non-`Option` fields.  Omitting a setter on
//! [`Mp4DemuxerBuilder`] is **equivalent to calling it with the
//! matching** `Mp4DemuxerSource::default_on_*` forwarder — the
//! builder substitutes the default in [`Mp4DemuxerBuilder::build`]
//! before constructing the source.  In particular:
//!
//! * `on_packet` defaults to
//!   [`Mp4DemuxerSource::default_on_packet_as_frame`] (upstream
//!   frame construction).  Call `.on_packet(...)` with a custom
//!   closure to opt into any other policy.
//! * `on_error` defaults to
//!   [`Mp4DemuxerSource::default_on_error`], which returns
//!   [`ErrorAction::Fatal`] for every [`Mp4DemuxerError`].
//!
//! There is therefore no runtime code path in which a demuxer
//! variant is dropped on the floor because the corresponding setter
//! was not called.  The builder's internal `Option<Hook>` fields
//! exist purely as a "was the setter called?" marker and are
//! collapsed to non-`Option` before the source is constructed.
//!
//! ```ignore
//! Mp4DemuxerSource::builder(name)
//!     .input(path).source_id("cam1")
//!     .downstream(decoder_name)
//!     .results(
//!         Mp4DemuxerResults::builder()
//!             .on_stream_info(Mp4DemuxerSource::default_on_stream_info())
//!             .on_packet(Mp4DemuxerSource::default_on_packet_as_frame())
//!             .on_source_eos(Mp4DemuxerSource::default_on_source_eos())
//!             .build(),
//!     )
//!     .build()?;
//! ```

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use parking_lot::Mutex;
use savant_gstreamer::mp4_demuxer::{
    DemuxedPacket, Mp4Demuxer, Mp4DemuxerError, Mp4DemuxerOutput, VideoInfo,
};

use crate::envelopes::EncodedMsg;
use crate::router::Router;
use crate::supervisor::StageName;
use crate::{ErrorAction, Flow, HookCtx, Source, SourceBuilder, SourceContext};

/// Closure type for `on_stream_info` — receives the freshly parsed
/// [`VideoInfo`], the `source_id`, and the stage's router.  The
/// hook decides whether to forward (typically via
/// `router.send(EncodedMsg::StreamInfo { ... })`) or drop the
/// sentinel.  Returning `Err(_)` is fatal and surfaces via
/// [`Source::run`].
pub type OnStreamInfoHook =
    Box<dyn FnMut(VideoInfo, &str, &Router<EncodedMsg>, &HookCtx) -> Result<()> + Send + 'static>;

/// Closure type for `on_packet` — receives each demuxed access
/// unit, the stream-level [`VideoInfo`] (as seen in the preceding
/// [`Mp4DemuxerOutput::StreamInfo`]), the `source_id`, and the
/// router.  Typical use:
/// `router.send(EncodedMsg::Packet { source_id: source_id.into(), info: *info, packet: pkt })`.
///
/// The [`VideoInfo`] is forwarded in-band with every
/// [`EncodedMsg::Packet`]
/// so downstream consumers (most importantly
/// [`Decoder`](super::decoder::Decoder)) never have to
/// maintain a per-source cache of stream parameters.  The same
/// `info` also unlocks upstream frame construction (for instance
/// via [`Mp4DemuxerSource::default_on_packet_as_frame`]).
/// Returning `Err(_)` is fatal.
pub type OnPacketHook = Box<
    dyn FnMut(DemuxedPacket, &VideoInfo, &str, &Router<EncodedMsg>, &HookCtx) -> Result<()>
        + Send
        + 'static,
>;

/// Closure type for `on_source_eos` — called exactly once when the
/// demuxer reaches end-of-stream, strictly after the last
/// [`OnPacketHook`] invocation.  The hook decides whether to
/// forward `EncodedMsg::SourceEos` downstream.
///
/// Return semantics — unified with every stage's
/// `on_source_eos`:
///
/// * `Ok(Flow::Cont)` — default; let the demuxer complete
///   normally.
/// * `Ok(Flow::Stop)` — request cooperative shutdown: the stage
///   flips the shared stop flag via
///   [`HookCtx::request_stop`] and aborts the default sink (if
///   any), so the GStreamer callback short-circuits on its next
///   invocation and [`Source::run`] winds down.
/// * `Err(_)` — logged at `error!` and treated as
///   `Ok(Flow::Stop)`; the first-error latch records the message
///   so [`Source::run`] surfaces it in its exit `Result`.
pub type OnSourceEosHook =
    Box<dyn FnMut(&str, &Router<EncodedMsg>, &HookCtx) -> Result<Flow> + Send + 'static>;

/// Closure type for `on_error`: classify a GStreamer pipeline
/// error.  The callback receives the structured
/// [`Mp4DemuxerError`], the stage's [`Router<EncodedMsg>`] (so
/// users can translate errors into downstream control messages
/// uniformly with the other egress error hooks), and the
/// [`HookCtx`].
///
/// The return [`ErrorAction`] drives the stage's exit
/// semantics: `Fatal` latches the error and aborts the default
/// sink, `LogAndContinue` records the error but keeps the
/// demuxer running, and `Swallow` drops the error entirely.
/// `Router<EncodedMsg>` is exposed so the hook can translate
/// errors into downstream control messages uniformly with other
/// egress hooks; default closures ignore it.
pub type OnErrorHook =
    Box<dyn FnMut(&Mp4DemuxerError, &Router<EncodedMsg>, &HookCtx) -> ErrorAction + Send + 'static>;

/// User shutdown hook fired after the demuxer has completed
/// ([`Source::run`]'s `demuxer.wait()` returned), whether
/// successfully or with an error latched.  Runs on the source
/// thread with access to the [`SourceContext`].  Use for final
/// metrics flushes, custom log lines, or bookkeeping that must
/// observe the demuxer's post-drain state.  The stage's own
/// cleanup (demuxer drop, first-error surfacing) runs *before*
/// this hook — users who need to replace that entirely should
/// implement [`Source`] directly on a bespoke struct.
pub type OnStoppingHook = Box<dyn FnMut(&SourceContext) + Send + 'static>;

/// A no-inbox [`Source`] that drives a `savant_gstreamer::Mp4Demuxer`
/// and forwards its output downstream as
/// [`EncodedMsg`].
///
/// Construct via [`Mp4DemuxerSource::builder`].
///
/// # Runtime invariant
///
/// All four variant hooks are **always populated** at runtime.  The
/// builder's `Option<...>` fields are an internal "was the setter
/// called?" marker;
/// [`Mp4DemuxerBuilder::build`](Mp4DemuxerBuilder::build) always
/// substitutes the matching `Mp4DemuxerSource::default_on_*` before
/// constructing the source.  The GStreamer callback therefore never
/// inspects an `Option` — every demuxed variant is dispatched to a
/// hook.
pub struct Mp4DemuxerSource {
    input: String,
    source_id: String,
    downstream: Option<StageName>,
    on_stream_info: OnStreamInfoHook,
    on_packet: OnPacketHook,
    on_source_eos: OnSourceEosHook,
    on_error: OnErrorHook,
    stopping: OnStoppingHook,
}

impl Mp4DemuxerSource {
    /// Start a fluent [`Mp4DemuxerBuilder`] registered under `name`.
    pub fn builder(name: StageName) -> Mp4DemuxerBuilder {
        Mp4DemuxerBuilder::new(name)
    }

    /// Default `on_stream_info` forwarder.  The returned closure
    /// sends
    /// [`EncodedMsg::StreamInfo { source_id, info }`](crate::envelopes::EncodedMsg::StreamInfo)
    /// via `router.send(...)` — the router's default peer.
    ///
    /// Use it as a one-liner when the stream-info sentinel should
    /// flow unchanged to the downstream stage:
    ///
    /// ```ignore
    /// Mp4DemuxerSource::builder(name)
    ///     .input(path)
    ///     .source_id("cam1")
    ///     .downstream(decoder_name)
    ///     .on_stream_info(Mp4DemuxerSource::default_on_stream_info())
    ///     // ...
    ///     .build()?;
    /// ```
    ///
    /// For name-based routing replace `router.send(msg)` with
    /// `router.send_to(&peer, msg)` inside a bespoke closure; for
    /// per-variant metrics wrap this forwarder in a closure that
    /// bumps counters before (or after) calling it.
    pub fn default_on_stream_info(
    ) -> impl FnMut(VideoInfo, &str, &Router<EncodedMsg>, &HookCtx) -> Result<()> + Send + 'static
    {
        |info, source_id, router, _ctx| {
            router.send(EncodedMsg::StreamInfo {
                source_id: source_id.to_string(),
                info,
            });
            Ok(())
        }
    }

    /// Default `on_packet` forwarder.  The returned closure sends
    /// [`EncodedMsg::Packet { source_id, info, packet }`](crate::envelopes::EncodedMsg::Packet)
    /// via `router.send(...)`, stamping every access unit with the
    /// stream-level [`VideoInfo`] observed on the preceding
    /// [`Mp4DemuxerOutput::StreamInfo`].
    ///
    /// Attaching the [`VideoInfo`] in-band lets downstream consumers
    /// (notably the framework's
    /// [`Decoder`](super::decoder::Decoder)) construct
    /// decode frames on the fly without maintaining a per-source
    /// [`VideoInfo`] cache — the message is self-describing.  Use
    /// [`Mp4DemuxerSource::default_on_packet_as_frame`] instead
    /// when you want the demuxer to build the
    /// [`VideoFrameProxy`](savant_core::primitives::frame::VideoFrameProxy)
    /// upstream.
    ///
    /// For per-source dispatch across multiple decoders, replace
    /// `router.send(msg)` with
    /// `router.send_to(&decoder_for(source_id), msg)` in your own
    /// closure.
    pub fn default_on_packet(
    ) -> impl FnMut(DemuxedPacket, &VideoInfo, &str, &Router<EncodedMsg>, &HookCtx) -> Result<()>
           + Send
           + 'static {
        |packet, info, source_id, router, _ctx| {
            router.send(EncodedMsg::Packet {
                source_id: source_id.to_string(),
                info: *info,
                packet,
            });
            Ok(())
        }
    }

    /// Default `on_packet` forwarder that constructs the
    /// decoder-facing [`VideoFrameProxy`](savant_core::primitives::frame::VideoFrameProxy)
    /// **on the demuxer side** (via
    /// [`make_decode_frame`](super::decoder::make_decode_frame)) and
    /// sends
    /// [`EncodedMsg::Frame { frame, payload: Some(bytes) }`](crate::envelopes::EncodedMsg::Frame).
    ///
    /// The downstream decoder consumes the variant via its
    /// [`Handler<FramePayload>`](crate::Handler) path and
    /// therefore does not need to maintain its own
    /// [`VideoInfo`]-stash — the frame arrives fully populated
    /// (codec, dims, fps, pts/dts/duration, keyframe flag all set
    /// from the demuxer's stream info + packet metadata).
    ///
    /// Swap for [`Mp4DemuxerSource::default_on_packet`] when you
    /// want the decoder to own frame construction instead.  For
    /// per-variant metrics or source-id-based fanout, wrap this
    /// forwarder in a custom closure that performs the side effect
    /// and then constructs / dispatches the frame itself, or
    /// replace `router.send(msg)` with
    /// `router.send_to(&peer, msg)`.
    pub fn default_on_packet_as_frame(
    ) -> impl FnMut(DemuxedPacket, &VideoInfo, &str, &Router<EncodedMsg>, &HookCtx) -> Result<()>
           + Send
           + 'static {
        |packet, info, source_id, router, _ctx| {
            let frame = super::decoder::make_decode_frame(source_id, &packet, info);
            let payload = Some(packet.data);
            router.send(EncodedMsg::Frame { frame, payload });
            Ok(())
        }
    }

    /// Default `on_source_eos` forwarder.  The returned closure
    /// sends
    /// [`EncodedMsg::SourceEos { source_id }`](crate::envelopes::EncodedMsg::SourceEos)
    /// via `router.send(...)` and returns `Ok(Flow::Cont)` — the
    /// source completes naturally once the underlying demuxer
    /// reports EOS.
    pub fn default_on_source_eos(
    ) -> impl FnMut(&str, &Router<EncodedMsg>, &HookCtx) -> Result<Flow> + Send + 'static {
        |source_id, router, _ctx| {
            router.send(EncodedMsg::SourceEos {
                source_id: source_id.to_string(),
            });
            Ok(Flow::Cont)
        }
    }

    /// Default `on_error` classifier.  Returns
    /// [`ErrorAction::Fatal`] for every
    /// [`Mp4DemuxerError`], latching the error and causing
    /// [`Source::run`] to return `Err`.  Override
    /// with a custom closure to downgrade specific error variants to
    /// [`ErrorAction::LogAndContinue`] or [`ErrorAction::Swallow`].
    ///
    /// The closure takes `&Router<EncodedMsg>` for API parity with
    /// the other egress error hooks (Decoder / NvInfer / NvTracker)
    /// but the default ignores it.
    pub fn default_on_error(
    ) -> impl FnMut(&Mp4DemuxerError, &Router<EncodedMsg>, &HookCtx) -> ErrorAction + Send + 'static
    {
        |_err, _router, _ctx| ErrorAction::Fatal
    }

    /// Default user shutdown hook — a no-op.  The stage's own
    /// post-drain cleanup (first-error surfacing, codec check)
    /// always runs *before* this hook fires, so omitting the
    /// [`Mp4DemuxerCommonBuilder::stopping`] setter simply means
    /// "don't add any extra cleanup on top of the built-in
    /// sequence".
    pub fn default_stopping() -> impl FnMut(&SourceContext) + Send + 'static {
        |_ctx| {}
    }
}

/// Container for the hook closures plus the last-seen
/// [`VideoInfo`].  The [`Mp4Demuxer`] callback needs `Fn + Send +
/// Sync + 'static`; our hooks are `FnMut`, so we park them behind a
/// [`parking_lot::Mutex`] that the callback takes each time it
/// fires.  Multi-threaded re-entrance is serialised by the mutex so
/// hook state remains coherent even when callbacks arrive from
/// different GStreamer threads.
///
/// The four variant hooks are non-`Option` by construction — see the
/// runtime invariant on [`Mp4DemuxerSource`].  `last_stream_info` is
/// the one genuinely optional field: it is `None` until the demuxer
/// emits its first [`Mp4DemuxerOutput::StreamInfo`] and is read by
/// every subsequent [`Mp4DemuxerOutput::Packet`] so `on_packet` hooks
/// can see the stream-level metadata without managing it themselves.
struct Mp4DemuxerHooks {
    on_stream_info: OnStreamInfoHook,
    on_packet: OnPacketHook,
    on_source_eos: OnSourceEosHook,
    on_error: OnErrorHook,
    last_stream_info: Option<VideoInfo>,
}

impl Source for Mp4DemuxerSource {
    fn run(self, ctx: SourceContext) -> Result<()> {
        let Mp4DemuxerSource {
            input,
            source_id,
            downstream,
            on_stream_info,
            on_packet,
            on_source_eos,
            on_error,
            mut stopping,
        } = self;

        let own_name = ctx.own_name().clone();
        let hook_ctx = ctx.hook_ctx();
        let router: Router<EncodedMsg> = ctx.router(downstream.as_ref())?;
        let default_sink = router.default_sink();
        let stop_flag = ctx.stop_flag();

        log::info!("[{own_name}] starting source_id={source_id} input={input}");

        // Local latch — first error seen on the pipeline bus is
        // the diagnostically useful one.
        let first_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let hooks: Arc<Mutex<Mp4DemuxerHooks>> = Arc::new(Mutex::new(Mp4DemuxerHooks {
            on_stream_info,
            on_packet,
            on_source_eos,
            on_error,
            last_stream_info: None,
        }));

        let router_cb = router.clone();
        let default_sink_cb = default_sink.clone();
        let stop_flag_cb = stop_flag.clone();
        let first_error_cb = first_error.clone();
        let hooks_cb = hooks.clone();
        let source_id_cb = source_id.clone();
        let own_name_cb = own_name.clone();
        let hook_ctx_cb = hook_ctx.clone();

        let mut demuxer = Mp4Demuxer::new_parsed(&input, move |output| {
            // Cooperative-stop / already-aborted short-circuit.  Without a
            // default peer only the stop flag matters.  The
            // `stop_flag` is shared with [`HookCtx::request_stop`],
            // so any hook (here or in an upstream stage) can ask the
            // demuxer to wind down.
            if stop_flag_cb.load(Ordering::Relaxed)
                || default_sink_cb
                    .as_ref()
                    .map(|s| s.aborted())
                    .unwrap_or(false)
            {
                return;
            }
            let mut h = hooks_cb.lock();
            match output {
                Mp4DemuxerOutput::StreamInfo(info) => {
                    log::info!(
                        "[{own_name_cb}] stream info: source_id={source_id_cb} {}x{} @ {}/{} codec={:?}",
                        info.width,
                        info.height,
                        info.framerate_num,
                        info.framerate_den,
                        info.codec
                    );
                    // Stash before running the user hook so that even
                    // if the hook fails and aborts, any residual
                    // `on_packet` already in flight can still resolve
                    // the info.  `VideoInfo: Copy`, so this is free.
                    h.last_stream_info = Some(info);
                    if let Err(e) =
                        (h.on_stream_info)(info, &source_id_cb, &router_cb, &hook_ctx_cb)
                    {
                        latch_error(&first_error_cb, format!("on_stream_info: {e}"));
                        if let Some(s) = default_sink_cb.as_ref() {
                            s.abort();
                        }
                    }
                }
                Mp4DemuxerOutput::Packet(pkt) => {
                    if let Some(info) = h.last_stream_info {
                        if let Err(e) =
                            (h.on_packet)(pkt, &info, &source_id_cb, &router_cb, &hook_ctx_cb)
                        {
                            latch_error(&first_error_cb, format!("on_packet: {e}"));
                            if let Some(s) = default_sink_cb.as_ref() {
                                s.abort();
                            }
                        }
                    } else {
                        latch_error(
                            &first_error_cb,
                            "on_packet fired before on_stream_info".to_string(),
                        );
                        if let Some(s) = default_sink_cb.as_ref() {
                            s.abort();
                        }
                    }
                }
                Mp4DemuxerOutput::Eos => {
                    log::info!("[{own_name_cb}] EOS (source_id={source_id_cb})");
                    match (h.on_source_eos)(&source_id_cb, &router_cb, &hook_ctx_cb) {
                        Ok(Flow::Cont) => {}
                        Ok(Flow::Stop) => {
                            log::info!(
                                "[{own_name_cb}] on_source_eos({source_id_cb}) requested stop"
                            );
                            hook_ctx_cb.request_stop();
                            if let Some(s) = default_sink_cb.as_ref() {
                                s.abort();
                            }
                        }
                        Err(e) => {
                            let msg = format!("on_source_eos: {e}");
                            log::error!("[{own_name_cb}] {msg}; requesting stop");
                            latch_error(&first_error_cb, msg);
                            hook_ctx_cb.request_stop();
                            if let Some(s) = default_sink_cb.as_ref() {
                                s.abort();
                            }
                        }
                    }
                }
                Mp4DemuxerOutput::Error(e) => {
                    let msg = e.to_string();
                    log::error!("[{own_name_cb}] pipeline error: {msg}");
                    let action = (h.on_error)(&e, &router_cb, &hook_ctx_cb);
                    match action {
                        ErrorAction::Fatal => {
                            latch_error(&first_error_cb, msg);
                            hook_ctx_cb.request_stop();
                            if let Some(s) = default_sink_cb.as_ref() {
                                s.abort();
                            }
                        }
                        ErrorAction::LogAndContinue => {
                            latch_error(&first_error_cb, msg);
                        }
                        ErrorAction::Swallow => {
                            // Drop on the floor — recorded in the log above.
                        }
                    }
                }
            }
        })
        .map_err(|e| anyhow!("Mp4Demuxer::new_parsed: {e}"))?;

        // Poll the demuxer's "finished" condvar with a short
        // timeout so an external stop request (Ctrl+C,
        // cooperative-stop from a hook, or a supervisor broadcast
        // that flips the shared `stop_flag`) is observed even for
        // very long files or sources that would otherwise take a
        // while to drain.  Without this, `wait()` would block on
        // the condvar until natural EOS and the source thread
        // would prevent `System::run` from joining on shutdown.
        //
        // `POLL_INTERVAL` is short enough (100 ms) to keep Ctrl+C
        // latency negligible while avoiding a busy loop.
        const POLL_INTERVAL: Duration = Duration::from_millis(100);
        let mut stopped_by_flag = false;
        while !demuxer.wait_timeout(POLL_INTERVAL) {
            if stop_flag.load(Ordering::Relaxed) {
                log::info!(
                    "[{own_name}] stop flag set; finishing MP4 demuxer (source_id={source_id})"
                );
                stopped_by_flag = true;
                break;
            }
        }
        // Tear the GStreamer pipeline down deterministically before
        // `drop(demuxer)` would otherwise run it implicitly.
        demuxer.finish();
        let codec = demuxer.detected_codec();
        log::info!(
            "[{own_name}] finished, detected_codec={codec:?} stopped_by_flag={stopped_by_flag}"
        );
        drop(demuxer);
        drop(hooks); // release the hook mutex (and its contents) last

        // User stopping hook fires AFTER the demuxer has finished
        // but BEFORE the first-error / codec-check exit path runs
        // — symmetric with how `Actor::stopping` composes on top of
        // the stage's load-bearing cleanup in other stages.
        (stopping)(&ctx);

        if let Some(err) = first_error.lock().take() {
            bail!("[{own_name}] demux error: {err}");
        }
        // A cooperative stop (Ctrl+C / supervisor broadcast) may
        // fire before the underlying source ever reported caps;
        // treat that as a clean exit and only flag the
        // "empty stream?" case when the demuxer actually finished
        // under its own steam with no codec ever seen.
        if codec.is_none() && !stopped_by_flag {
            bail!("[{own_name}] demuxer did not detect a video codec (empty stream?)");
        }
        Ok(())
    }
}

fn latch_error(slot: &Arc<Mutex<Option<String>>>, msg: String) {
    let mut g = slot.lock();
    if g.is_none() {
        *g = Some(msg);
    }
}

/// Per-variant [`Mp4DemuxerOutput`] hook bundle — one branch per
/// demuxed variant plus the error classifier.
///
/// Built through [`Mp4DemuxerResults::builder`] and handed to
/// [`Mp4DemuxerBuilder::results`].  Omitted branches auto-install
/// the matching `Mp4DemuxerSource::default_on_*` at build time, so
/// the runtime invariant "every [`Mp4DemuxerOutput`] variant is
/// always dispatched to a user-supplied or auto-installed hook"
/// holds regardless of how much the user overrides.
pub struct Mp4DemuxerResults {
    on_stream_info: OnStreamInfoHook,
    on_packet: OnPacketHook,
    on_source_eos: OnSourceEosHook,
    on_error: OnErrorHook,
}

impl Mp4DemuxerResults {
    /// Start a builder that auto-installs every default on
    /// [`Mp4DemuxerResultsBuilder::build`].
    pub fn builder() -> Mp4DemuxerResultsBuilder {
        Mp4DemuxerResultsBuilder::new()
    }
}

impl Default for Mp4DemuxerResults {
    /// Every branch wired to its matching
    /// `Mp4DemuxerSource::default_on_*`.  `on_packet` defaults to
    /// [`Mp4DemuxerSource::default_on_packet_as_frame`], i.e.
    /// upstream frame construction.
    fn default() -> Self {
        Mp4DemuxerResultsBuilder::new().build()
    }
}

/// Fluent builder for [`Mp4DemuxerResults`].
pub struct Mp4DemuxerResultsBuilder {
    on_stream_info: Option<OnStreamInfoHook>,
    on_packet: Option<OnPacketHook>,
    on_source_eos: Option<OnSourceEosHook>,
    on_error: Option<OnErrorHook>,
}

impl Mp4DemuxerResultsBuilder {
    /// Empty bundle — every hook defaults to its matching
    /// `Mp4DemuxerSource::default_*` equivalent at
    /// [`build`](Self::build) time.
    pub fn new() -> Self {
        Self {
            on_stream_info: None,
            on_packet: None,
            on_source_eos: None,
            on_error: None,
        }
    }

    /// Override the `on_stream_info` hook.  Omitting this setter is
    /// equivalent to calling
    /// `.on_stream_info(Mp4DemuxerSource::default_on_stream_info())`.
    pub fn on_stream_info<F>(mut self, f: F) -> Self
    where
        F: FnMut(VideoInfo, &str, &Router<EncodedMsg>, &HookCtx) -> Result<()> + Send + 'static,
    {
        self.on_stream_info = Some(Box::new(f));
        self
    }

    /// Override the `on_packet` hook.  Omitting this setter is
    /// equivalent to calling
    /// `.on_packet(Mp4DemuxerSource::default_on_packet_as_frame())`
    /// — the demuxer constructs a
    /// [`VideoFrameProxy`](savant_core::primitives::frame::VideoFrameProxy)
    /// upstream and sends
    /// [`EncodedMsg::Frame`].
    /// Swap to [`Mp4DemuxerSource::default_on_packet`] for
    /// downstream frame construction.
    pub fn on_packet<F>(mut self, f: F) -> Self
    where
        F: FnMut(DemuxedPacket, &VideoInfo, &str, &Router<EncodedMsg>, &HookCtx) -> Result<()>
            + Send
            + 'static,
    {
        self.on_packet = Some(Box::new(f));
        self
    }

    /// Override the `on_source_eos` hook.  Omitting this setter is
    /// equivalent to calling
    /// `.on_source_eos(Mp4DemuxerSource::default_on_source_eos())`
    /// — forward `EncodedMsg::SourceEos` downstream and return
    /// `Ok(Flow::Cont)`.  Return `Ok(Flow::Stop)` to request
    /// cooperative shutdown of the source; `Err(_)` is logged,
    /// latched, and also requests shutdown.
    pub fn on_source_eos<F>(mut self, f: F) -> Self
    where
        F: FnMut(&str, &Router<EncodedMsg>, &HookCtx) -> Result<Flow> + Send + 'static,
    {
        self.on_source_eos = Some(Box::new(f));
        self
    }

    /// Override the `on_error` classifier.  Omitting this setter is
    /// equivalent to calling
    /// `.on_error(Mp4DemuxerSource::default_on_error())` — classify
    /// every error as [`ErrorAction::Fatal`].  The closure receives
    /// the router for API parity with other egress error hooks; it
    /// is free to ignore the router.
    pub fn on_error<F>(mut self, f: F) -> Self
    where
        F: FnMut(&Mp4DemuxerError, &Router<EncodedMsg>, &HookCtx) -> ErrorAction + Send + 'static,
    {
        self.on_error = Some(Box::new(f));
        self
    }

    /// Finalise the bundle, substituting defaults for every omitted
    /// branch.
    pub fn build(self) -> Mp4DemuxerResults {
        let Mp4DemuxerResultsBuilder {
            on_stream_info,
            on_packet,
            on_source_eos,
            on_error,
        } = self;
        Mp4DemuxerResults {
            on_stream_info: on_stream_info
                .unwrap_or_else(|| Box::new(Mp4DemuxerSource::default_on_stream_info())),
            on_packet: on_packet
                .unwrap_or_else(|| Box::new(Mp4DemuxerSource::default_on_packet_as_frame())),
            on_source_eos: on_source_eos
                .unwrap_or_else(|| Box::new(Mp4DemuxerSource::default_on_source_eos())),
            on_error: on_error.unwrap_or_else(|| Box::new(Mp4DemuxerSource::default_on_error())),
        }
    }
}

impl Default for Mp4DemuxerResultsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Loop-level common knobs + user shutdown hook for
/// [`Mp4DemuxerSource`].  Built through
/// [`Mp4DemuxerCommon::builder`] and handed to
/// [`Mp4DemuxerBuilder::common`].
pub struct Mp4DemuxerCommon {
    stopping: OnStoppingHook,
}

impl Mp4DemuxerCommon {
    /// Start a builder seeded with the no-op stopping hook.
    pub fn builder() -> Mp4DemuxerCommonBuilder {
        Mp4DemuxerCommonBuilder::new()
    }
}

impl Default for Mp4DemuxerCommon {
    /// No `stopping` hook installed.
    fn default() -> Self {
        Mp4DemuxerCommonBuilder::new().build()
    }
}

/// Fluent builder for [`Mp4DemuxerCommon`].
pub struct Mp4DemuxerCommonBuilder {
    stopping: Option<OnStoppingHook>,
}

impl Mp4DemuxerCommonBuilder {
    /// Empty bundle — `stopping` defaults to a no-op.
    pub fn new() -> Self {
        Self { stopping: None }
    }

    /// Override the user shutdown hook — fired once the demuxer
    /// has finished (`demuxer.wait()` returned), **before** the
    /// first-error / codec-check exit path runs.  Runs on the
    /// source thread with access to the [`SourceContext`].  Use
    /// for final metrics flushes, bespoke log lines, or custom
    /// bookkeeping that must observe the demuxer's post-drain
    /// state.  The stage's own cleanup (demuxer drop,
    /// first-error surfacing) is **load-bearing** and always runs
    /// after this hook; users who need to replace that entirely
    /// should implement [`Source`] directly on a bespoke struct.
    pub fn stopping<F>(mut self, f: F) -> Self
    where
        F: FnMut(&SourceContext) + Send + 'static,
    {
        self.stopping = Some(Box::new(f));
        self
    }

    /// Finalise the bundle, substituting the no-op default for an
    /// omitted `stopping` setter.
    pub fn build(self) -> Mp4DemuxerCommon {
        let Mp4DemuxerCommonBuilder { stopping } = self;
        Mp4DemuxerCommon {
            stopping: stopping.unwrap_or_else(|| Box::new(Mp4DemuxerSource::default_stopping())),
        }
    }
}

impl Default for Mp4DemuxerCommonBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Fluent builder for [`Mp4DemuxerSource`].
///
/// The builder only exposes wiring-level configuration at the top
/// level (`input`, `source_id`, `downstream`).  Per-variant
/// demuxer-output hooks live on [`Mp4DemuxerResults`]; the user
/// shutdown hook lives on [`Mp4DemuxerCommon`].  Install them via
/// [`Mp4DemuxerBuilder::results`] and [`Mp4DemuxerBuilder::common`].
pub struct Mp4DemuxerBuilder {
    name: StageName,
    input: Option<String>,
    source_id: Option<String>,
    downstream: Option<StageName>,
    results: Option<Mp4DemuxerResults>,
    common: Option<Mp4DemuxerCommon>,
}

impl Mp4DemuxerBuilder {
    /// Start a builder for a demux source registered under `name`.
    ///
    /// Every per-variant hook defaults to its
    /// `Mp4DemuxerSource::default_on_*` equivalent; the user
    /// shutdown hook defaults to a no-op.  The user only needs to
    /// supply `input` and `source_id`; `downstream` / `results` /
    /// `common` are optional and composable.
    pub fn new(name: StageName) -> Self {
        Self {
            name,
            input: None,
            source_id: None,
            downstream: None,
            results: None,
            common: None,
        }
    }

    /// Required: file path or URI passed to
    /// [`Mp4Demuxer::new_parsed`].
    pub fn input(mut self, input: impl Into<String>) -> Self {
        self.input = Some(input.into());
        self
    }

    /// Required: `source_id` stamped on every
    /// [`EncodedMsg::StreamInfo`],
    /// [`EncodedMsg::Packet`],
    /// and terminal
    /// [`EncodedMsg::SourceEos`].
    pub fn source_id(mut self, id: impl Into<String>) -> Self {
        self.source_id = Some(id.into());
        self
    }

    /// Optional default peer installed on the
    /// [`Router<EncodedMsg>`] handed to every variant hook.  Hooks
    /// call `router.send(msg)` to route to this peer and
    /// `router.send_to(&peer, msg)` to address any other registered
    /// actor by name.
    pub fn downstream(mut self, peer: StageName) -> Self {
        self.downstream = Some(peer);
        self
    }

    /// Install a [`Mp4DemuxerResults`] bundle — one branch per
    /// demuxed variant.  Omitting this call is equivalent to
    /// `.results(Mp4DemuxerResults::default())`, which wires every
    /// branch to its matching `Mp4DemuxerSource::default_on_*`.
    pub fn results(mut self, r: Mp4DemuxerResults) -> Self {
        self.results = Some(r);
        self
    }

    /// Install a [`Mp4DemuxerCommon`] bundle — currently just the
    /// `stopping` hook.  Omitting this call is equivalent to
    /// `.common(Mp4DemuxerCommon::default())`.
    pub fn common(mut self, c: Mp4DemuxerCommon) -> Self {
        self.common = Some(c);
        self
    }

    /// Finalise the builder and obtain the Layer-A
    /// [`SourceBuilder<Mp4DemuxerSource>`] ready for
    /// [`System::register_source`](super::super::System::register_source).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `input` or `source_id` is missing.
    /// `downstream` is optional — omit it for a drain-only source.
    pub fn build(self) -> Result<SourceBuilder<Mp4DemuxerSource>> {
        let Mp4DemuxerBuilder {
            name,
            input,
            source_id,
            downstream,
            results,
            common,
        } = self;
        let input = input.ok_or_else(|| anyhow!("Mp4DemuxerSource: missing input"))?;
        let source_id = source_id.ok_or_else(|| anyhow!("Mp4DemuxerSource: missing source_id"))?;
        let Mp4DemuxerResults {
            on_stream_info,
            on_packet,
            on_source_eos,
            on_error,
        } = results.unwrap_or_default();
        let Mp4DemuxerCommon { stopping } = common.unwrap_or_default();
        Ok(SourceBuilder::new(name).factory(move |_bx| {
            Ok(Mp4DemuxerSource {
                input,
                source_id,
                downstream,
                on_stream_info,
                on_packet,
                on_source_eos,
                on_error,
                stopping,
            })
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::supervisor::{StageKind, StageName};

    fn err_msg(result: Result<SourceBuilder<Mp4DemuxerSource>>) -> String {
        match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected error"),
        }
    }

    #[test]
    fn builder_requires_input_and_source_id() {
        let name = StageName::unnamed(StageKind::Mp4Demux);
        assert!(err_msg(Mp4DemuxerSource::builder(name.clone()).build()).contains("missing input"));
        assert!(err_msg(
            Mp4DemuxerSource::builder(name.clone())
                .input("/tmp/x.mp4")
                .build()
        )
        .contains("missing source_id"));
    }

    /// `downstream` is optional now — a drain-only source (no
    /// default peer) still builds.  The resulting source silently
    /// drops all packets through the default `router.send(...)` path.
    #[test]
    fn builder_without_downstream_is_accepted() {
        let name = StageName::unnamed(StageKind::Mp4Demux);
        let _ = Mp4DemuxerSource::builder(name)
            .input("/tmp/x.mp4")
            .source_id("s")
            .build()
            .expect("no-downstream builder is accepted");
    }

    #[test]
    fn builder_accepts_all_hooks() {
        let name = StageName::unnamed(StageKind::Mp4Demux);
        let sb = Mp4DemuxerSource::builder(name)
            .input("/tmp/x.mp4")
            .source_id("cam1")
            .downstream(StageName::unnamed(StageKind::Decoder))
            .results(
                Mp4DemuxerResults::builder()
                    .on_stream_info(|info, sid, router, _ctx| {
                        router.send(EncodedMsg::StreamInfo {
                            source_id: sid.to_string(),
                            info,
                        });
                        Ok(())
                    })
                    .on_packet(|pkt, info, sid, router, _ctx| {
                        router.send(EncodedMsg::Packet {
                            source_id: sid.to_string(),
                            info: *info,
                            packet: pkt,
                        });
                        Ok(())
                    })
                    .on_source_eos(|sid, router, _ctx| {
                        router.send(EncodedMsg::SourceEos {
                            source_id: sid.to_string(),
                        });
                        Ok(Flow::Cont)
                    })
                    .on_error(
                        |_err: &Mp4DemuxerError, _router: &Router<EncodedMsg>, _ctx: &HookCtx| {
                            ErrorAction::Fatal
                        },
                    )
                    .build(),
            )
            .build()
            .unwrap();
        // No way to run without a real file; this is a compile-shape
        // test.  The SourceBuilder carries our factory.
        let _ = sb;
    }

    /// Confirms the `default_on_*` associated functions (including
    /// the packet→packet variant) slot into the bundle builder's
    /// generic hook bounds as-is.
    #[test]
    fn builder_accepts_default_forwarders() {
        let name = StageName::unnamed(StageKind::Mp4Demux);
        let sb = Mp4DemuxerSource::builder(name)
            .input("/tmp/x.mp4")
            .source_id("cam1")
            .downstream(StageName::unnamed(StageKind::Decoder))
            .results(
                Mp4DemuxerResults::builder()
                    .on_stream_info(Mp4DemuxerSource::default_on_stream_info())
                    .on_packet(Mp4DemuxerSource::default_on_packet())
                    .on_source_eos(Mp4DemuxerSource::default_on_source_eos())
                    .on_error(Mp4DemuxerSource::default_on_error())
                    .build(),
            )
            .build()
            .unwrap();
        let _ = sb;
    }

    /// Confirms the packet→frame forwarder slots into the bundle
    /// builder in place of `default_on_packet`.
    #[test]
    fn builder_accepts_default_on_packet_as_frame() {
        let name = StageName::unnamed(StageKind::Mp4Demux);
        let sb = Mp4DemuxerSource::builder(name)
            .input("/tmp/x.mp4")
            .source_id("cam1")
            .downstream(StageName::unnamed(StageKind::Decoder))
            .results(
                Mp4DemuxerResults::builder()
                    .on_stream_info(Mp4DemuxerSource::default_on_stream_info())
                    .on_packet(Mp4DemuxerSource::default_on_packet_as_frame())
                    .on_source_eos(Mp4DemuxerSource::default_on_source_eos())
                    .on_error(Mp4DemuxerSource::default_on_error())
                    .build(),
            )
            .build()
            .unwrap();
        let _ = sb;
    }

    /// The [`Mp4DemuxerCommon`] bundle accepts a user-supplied
    /// `.stopping(F)` closure.
    #[test]
    fn builder_accepts_user_stopping() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let flag = Arc::new(AtomicBool::new(false));
        let flag_hook = flag.clone();
        let name = StageName::unnamed(StageKind::Mp4Demux);
        let _ = Mp4DemuxerSource::builder(name)
            .input("/tmp/x.mp4")
            .source_id("cam1")
            .downstream(StageName::unnamed(StageKind::Decoder))
            .common(
                Mp4DemuxerCommon::builder()
                    .stopping(move |_ctx| {
                        flag_hook.store(true, Ordering::SeqCst);
                    })
                    .build(),
            )
            .build()
            .unwrap();
        assert!(!flag.load(Ordering::SeqCst));
    }

    /// Runtime invariant: `build()` succeeds with only the three
    /// mandatory setters (`input`, `source_id`, `downstream`) and
    /// auto-installs all four `default_on_*` forwarders plus the
    /// no-op stopping hook.  The `SourceFactory` resolves and
    /// produces a `Mp4DemuxerSource` whose hook fields are
    /// non-`Option` by construction — proving that no variant is
    /// ever dispatched to `None` at runtime.
    #[test]
    fn builder_accepts_bare_minimum() {
        use crate::context::BuildCtx;
        use crate::registry::Registry;
        use crate::shared::SharedStore;

        let name = StageName::unnamed(StageKind::Mp4Demux);
        let sb = Mp4DemuxerSource::builder(name.clone())
            .input("/tmp/x.mp4")
            .source_id("cam1")
            .downstream(StageName::unnamed(StageKind::Decoder))
            .build()
            .expect("bare-minimum builder is accepted");
        // Tear the SourceBuilder apart and actually run the factory
        // so the defaults installed in `build()` materialise into the
        // runtime source struct.  This catches any future regression
        // where a setter override path accidentally bypasses the
        // `unwrap_or_else(default_*)` chain.
        let parts = sb.into_parts();
        assert_eq!(parts.name, name);
        let reg = Arc::new(Registry::new());
        let shared = Arc::new(SharedStore::new());
        let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let bx = BuildCtx::new(&parts.name, &reg, &shared, &stop_flag);
        let src = (parts.factory)(&bx).expect("factory resolves");
        // Pattern-match proves every hook is non-`Option`; the
        // compiler would refuse if any field were wrapped in
        // `Option<_>`.  We also bind each hook to a variable to
        // silence the unused-fields lint and document intent.
        let Mp4DemuxerSource {
            input: _,
            source_id: _,
            downstream: _,
            on_stream_info: _,
            on_packet: _,
            on_source_eos: _,
            on_error: _,
            stopping: _,
        } = src;
    }
}
