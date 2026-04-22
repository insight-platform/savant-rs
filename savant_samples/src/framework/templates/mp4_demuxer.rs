//! [`Mp4DemuxerSource`] — an MP4 (and any other GStreamer-parsable
//! container) source that surfaces every demuxed output variant to
//! user code, along with the stage's [`Router<EncodedMsg>`].
//!
//! Direct replacement for
//! `cars_tracking/pipeline/mp4_demux.rs` — the template hides:
//!
//! * `Arc<AtomicBool>` abort flag (becomes [`SourceContext::is_stopping`]).
//! * Manual exit-guard wiring (the framework installs it around
//!   [`Source::run`]).
//! * The first-error latch (still present, but hidden in the
//!   template body).
//!
//! The template itself never calls `router.send` — **every send
//! site for this stage lives in user code**.  That lets source-side
//! routing (for instance "pick a decoder pool by source_id" or
//! "broadcast the same packet to multiple decoders via
//! `router.send_to(&peer, msg)`") stay visible at the callsite.
//!
//! The user-facing surface is a fluent builder:
//!
//! ```ignore
//! sys.register_source(
//!     Mp4DemuxerSource::builder(StageName::unnamed(StageKind::Mp4Demux))
//!         .input(input_path)
//!         .source_id("cam1")
//!         .downstream(StageName::unnamed(StageKind::Decoder))
//!         .on_packet(|pkt, info, source_id, router| {
//!             router.send(EncodedMsg::Packet {
//!                 source_id: source_id.to_string(),
//!                 info: *info,
//!                 packet: pkt,
//!             });
//!             Ok(())
//!         })
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
//!   [`EncodedMsg::StreamInfo`](crate::framework::envelopes::EncodedMsg::StreamInfo).
//! * [`Mp4DemuxerSource::default_on_packet`] — sends
//!   [`EncodedMsg::Packet`](crate::framework::envelopes::EncodedMsg::Packet).
//!   Use when the downstream decoder owns frame construction.
//! * [`Mp4DemuxerSource::default_on_packet_as_frame`] — constructs
//!   a [`VideoFrameProxy`](savant_core::primitives::frame::VideoFrameProxy)
//!   on the demuxer side (via
//!   [`make_decode_frame`](super::decoder::make_decode_frame)) and
//!   sends
//!   [`EncodedMsg::Frame`](crate::framework::envelopes::EncodedMsg::Frame).
//!   Use when you want upstream frame construction so the decoder
//!   can stay stateless w.r.t. [`VideoInfo`].
//! * [`Mp4DemuxerSource::default_on_source_eos`] — sends
//!   [`EncodedMsg::SourceEos`](crate::framework::envelopes::EncodedMsg::SourceEos).
//!
//! Pick exactly one of `default_on_packet` / `default_on_packet_as_frame`
//! per source — the two differ only in whether the
//! [`VideoFrameProxy`](savant_core::primitives::frame::VideoFrameProxy)
//! is built upstream or downstream.
//!
//! Because each one is an associated function that returns an owned
//! closure, the send still lives in user code (the user names the
//! forwarder at the call site) — the template just factors out the
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
//!   [`ErrorAction::Fatal`] for every
//!   [`Mp4DemuxerError`] — matching the legacy "first error
//!   latches and aborts the sink" behaviour.
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
//!     .on_stream_info(Mp4DemuxerSource::default_on_stream_info())
//!     .on_packet(Mp4DemuxerSource::default_on_packet_as_frame())
//!     .on_source_eos(Mp4DemuxerSource::default_on_source_eos())
//!     .build()?;
//! ```

use std::sync::atomic::Ordering;
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use parking_lot::Mutex;
use savant_gstreamer::mp4_demuxer::{
    DemuxedPacket, Mp4Demuxer, Mp4DemuxerError, Mp4DemuxerOutput, VideoInfo,
};

use crate::framework::envelopes::EncodedMsg;
use crate::framework::router::Router;
use crate::framework::supervisor::StageName;
use crate::framework::{Source, SourceBuilder, SourceContext};

/// Hook return for a `Mp4DemuxerOutput::Error` callback — tells the
/// template how to react to a GStreamer-level pipeline error.
///
/// All variants log the error on the spot.  They differ in whether
/// the source is allowed to continue and whether the final
/// `Source::run` return is `Ok(())` or `Err(_)`.
#[derive(Debug, Clone, Copy, Default)]
pub enum ErrorAction {
    /// Default: latch the first error, abort the downstream sink,
    /// and surface the error as `Err(_)` from [`Source::run`].
    /// Matches the legacy mp4_demux behaviour.
    #[default]
    Fatal,
    /// Log the error and continue consuming the demuxer.  The
    /// source-level `Err` is still latched — `Source::run` returns
    /// `Err(_)` on exit — but the sink is *not* aborted, giving
    /// downstream a chance to process already-queued packets.
    LogAndContinue,
    /// Log the error and swallow it completely — `Source::run`
    /// returns `Ok(())` if the demuxer drains cleanly afterwards.
    /// Use with care; intended for non-fatal GStreamer warnings
    /// that the MP4 demuxer surfaces as `Error`.
    Swallow,
}

/// Closure type for `on_stream_info` — receives the freshly parsed
/// [`VideoInfo`], the `source_id`, and the stage's router.  The
/// hook decides whether to forward (typically via
/// `router.send(EncodedMsg::StreamInfo { ... })`) or drop the
/// sentinel.  Returning `Err(_)` is fatal and surfaces via
/// [`Source::run`].
pub type OnStreamInfoHook =
    Box<dyn FnMut(VideoInfo, &str, &Router<EncodedMsg>) -> Result<()> + Send + 'static>;

/// Closure type for `on_packet` — receives each demuxed access
/// unit, the stream-level [`VideoInfo`] (as seen in the preceding
/// [`Mp4DemuxerOutput::StreamInfo`]), the `source_id`, and the
/// router.  Typical use:
/// `router.send(EncodedMsg::Packet { source_id: source_id.into(), info: *info, packet: pkt })`.
///
/// The [`VideoInfo`] is forwarded in-band with every
/// [`EncodedMsg::Packet`](crate::framework::envelopes::EncodedMsg::Packet)
/// so downstream consumers (most importantly
/// [`Decoder`](super::decoder::Decoder)) never have to
/// maintain a per-source cache of stream parameters.  The same
/// `info` also unlocks upstream frame construction (for instance
/// via [`Mp4DemuxerSource::default_on_packet_as_frame`]).
/// Returning `Err(_)` is fatal.
pub type OnPacketHook = Box<
    dyn FnMut(DemuxedPacket, &VideoInfo, &str, &Router<EncodedMsg>) -> Result<()> + Send + 'static,
>;

/// Closure type for `on_source_eos` — called exactly once when the
/// demuxer reaches end-of-stream, strictly after the last
/// [`OnPacketHook`] invocation.  The hook decides whether to
/// forward `EncodedMsg::SourceEos` downstream.
pub type OnSourceEosHook = Box<dyn FnMut(&str, &Router<EncodedMsg>) + Send + 'static>;

/// Closure type for `on_error`: classify a GStreamer pipeline
/// error.  The callback receives the structured
/// [`Mp4DemuxerError`] so user code can inspect variants (e.g.
/// distinguish a `PipelineError` with a specific `src` from an
/// `InputNotFound`) and tailor the [`ErrorAction`] accordingly.
///
/// Unlike the variant send hooks above, this one never receives a
/// router: GStreamer errors are about the source's *exit* semantics
/// (whether [`Source::run`] returns `Err`), not about forwarding
/// messages downstream.
pub type OnErrorHook = Box<dyn FnMut(&Mp4DemuxerError) -> ErrorAction + Send + 'static>;

/// A no-inbox [`Source`] that drives a `savant_gstreamer::Mp4Demuxer`
/// and forwards its output downstream as
/// [`EncodedMsg`](crate::framework::envelopes::EncodedMsg).
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
}

impl Mp4DemuxerSource {
    /// Start a fluent [`Mp4DemuxerBuilder`] registered under `name`.
    pub fn builder(name: StageName) -> Mp4DemuxerBuilder {
        Mp4DemuxerBuilder::new(name)
    }

    /// Default `on_stream_info` forwarder.  The returned closure
    /// sends
    /// [`EncodedMsg::StreamInfo { source_id, info }`](crate::framework::envelopes::EncodedMsg::StreamInfo)
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
    ) -> impl FnMut(VideoInfo, &str, &Router<EncodedMsg>) -> Result<()> + Send + 'static {
        |info, source_id, router| {
            router.send(EncodedMsg::StreamInfo {
                source_id: source_id.to_string(),
                info,
            });
            Ok(())
        }
    }

    /// Default `on_packet` forwarder.  The returned closure sends
    /// [`EncodedMsg::Packet { source_id, info, packet }`](crate::framework::envelopes::EncodedMsg::Packet)
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
    ) -> impl FnMut(DemuxedPacket, &VideoInfo, &str, &Router<EncodedMsg>) -> Result<()> + Send + 'static
    {
        |packet, info, source_id, router| {
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
    /// [`EncodedMsg::Frame { frame, payload: Some(bytes) }`](crate::framework::envelopes::EncodedMsg::Frame).
    ///
    /// The downstream decoder consumes the variant via its
    /// [`Handler<FramePayload>`](crate::framework::Handler) path and
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
    ) -> impl FnMut(DemuxedPacket, &VideoInfo, &str, &Router<EncodedMsg>) -> Result<()> + Send + 'static
    {
        |packet, info, source_id, router| {
            let frame = super::decoder::make_decode_frame(source_id, &packet, info);
            let payload = Some(packet.data);
            router.send(EncodedMsg::Frame { frame, payload });
            Ok(())
        }
    }

    /// Default `on_source_eos` forwarder.  The returned closure
    /// sends
    /// [`EncodedMsg::SourceEos { source_id }`](crate::framework::envelopes::EncodedMsg::SourceEos)
    /// via `router.send(...)`.
    pub fn default_on_source_eos() -> impl FnMut(&str, &Router<EncodedMsg>) + Send + 'static {
        |source_id, router| {
            router.send(EncodedMsg::SourceEos {
                source_id: source_id.to_string(),
            });
        }
    }

    /// Default `on_error` classifier.  Returns
    /// [`ErrorAction::Fatal`] for every
    /// [`Mp4DemuxerError`] — matches the legacy "first error latches,
    /// sink aborts, `Source::run` returns `Err`" behaviour.  Override
    /// with a custom closure to downgrade specific error variants to
    /// [`ErrorAction::LogAndContinue`] or [`ErrorAction::Swallow`].
    pub fn default_on_error() -> impl FnMut(&Mp4DemuxerError) -> ErrorAction + Send + 'static {
        |_err| ErrorAction::Fatal
    }
}

/// Container for the hook closures plus the last-seen
/// [`VideoInfo`].  The [`Mp4Demuxer`] callback needs `Fn + Send +
/// Sync + 'static`; our hooks are `FnMut`, so we park them behind a
/// [`parking_lot::Mutex`] that the callback takes each time it
/// fires.  Multi-threaded re-entrance is serialised by the mutex —
/// matching the legacy demux actor's use of an `AtomicBool` +
/// `Mutex<Option<String>>` for its own internal state.
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
        } = self;

        let own_name = ctx.own_name().clone();
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

        let demuxer = Mp4Demuxer::new_parsed(&input, move |output| {
            // Cooperative-stop / already-aborted short-circuit —
            // matches the legacy `aborted` flag semantics.  Without a
            // default peer only the stop flag matters.
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
                    if let Err(e) = (h.on_stream_info)(info, &source_id_cb, &router_cb) {
                        latch_error(&first_error_cb, format!("on_stream_info: {e}"));
                        if let Some(s) = default_sink_cb.as_ref() {
                            s.abort();
                        }
                    }
                }
                Mp4DemuxerOutput::Packet(pkt) => {
                    if let Some(info) = h.last_stream_info {
                        if let Err(e) = (h.on_packet)(pkt, &info, &source_id_cb, &router_cb) {
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
                    (h.on_source_eos)(&source_id_cb, &router_cb);
                }
                Mp4DemuxerOutput::Error(e) => {
                    let msg = e.to_string();
                    log::error!("[{own_name_cb}] pipeline error: {msg}");
                    let action = (h.on_error)(&e);
                    match action {
                        ErrorAction::Fatal => {
                            latch_error(&first_error_cb, msg);
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

        demuxer.wait();
        let codec = demuxer.detected_codec();
        log::info!("[{own_name}] finished, detected_codec={codec:?}");
        drop(demuxer);
        drop(hooks); // release the hook mutex (and its contents) last

        if let Some(err) = first_error.lock().take() {
            bail!("[{own_name}] demux error: {err}");
        }
        if codec.is_none() {
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

/// Fluent builder for [`Mp4DemuxerSource`].
pub struct Mp4DemuxerBuilder {
    name: StageName,
    input: Option<String>,
    source_id: Option<String>,
    downstream: Option<StageName>,
    on_stream_info: Option<OnStreamInfoHook>,
    on_packet: Option<OnPacketHook>,
    on_source_eos: Option<OnSourceEosHook>,
    on_error: Option<OnErrorHook>,
}

impl Mp4DemuxerBuilder {
    /// Start a builder for a demux source registered under `name`.
    pub fn new(name: StageName) -> Self {
        Self {
            name,
            input: None,
            source_id: None,
            downstream: None,
            on_stream_info: None,
            on_packet: None,
            on_source_eos: None,
            on_error: None,
        }
    }

    /// Required: file path or URI passed to
    /// [`Mp4Demuxer::new_parsed`].
    pub fn input(mut self, input: impl Into<String>) -> Self {
        self.input = Some(input.into());
        self
    }

    /// Required: `source_id` stamped on every
    /// [`EncodedMsg::StreamInfo`](crate::framework::envelopes::EncodedMsg::StreamInfo),
    /// [`EncodedMsg::Packet`](crate::framework::envelopes::EncodedMsg::Packet),
    /// and terminal
    /// [`EncodedMsg::SourceEos`](crate::framework::envelopes::EncodedMsg::SourceEos).
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

    /// Hook fired for [`Mp4DemuxerOutput::StreamInfo`].  Receives the
    /// parsed [`VideoInfo`], the `source_id`, and the router.  The
    /// hook is responsible for forwarding downstream (typically
    /// `router.send(EncodedMsg::StreamInfo { ... })`).  Returning
    /// `Err(_)` latches the error and aborts the default sink.
    ///
    /// Omitting this setter is equivalent to calling
    /// `.on_stream_info(Mp4DemuxerSource::default_on_stream_info())`.
    pub fn on_stream_info<F>(mut self, f: F) -> Self
    where
        F: FnMut(VideoInfo, &str, &Router<EncodedMsg>) -> Result<()> + Send + 'static,
    {
        self.on_stream_info = Some(Box::new(f));
        self
    }

    /// Hook fired for every [`Mp4DemuxerOutput::Packet`].  Receives
    /// the access unit, the stream-level [`VideoInfo`] (as observed
    /// from the preceding [`Mp4DemuxerOutput::StreamInfo`]), the
    /// `source_id`, and the router.  The hook typically does
    /// `router.send(EncodedMsg::Packet { source_id: source_id.into(),
    /// info: *info, packet: pkt })` — which lets the downstream
    /// decoder construct frames on the fly without caching per-source
    /// stream parameters.  To build the decode frame on the demuxer
    /// side instead, call
    /// [`make_decode_frame`](super::decoder::make_decode_frame)
    /// and emit `EncodedMsg::Frame` (see
    /// [`Mp4DemuxerSource::default_on_packet_as_frame`]).
    /// Returning `Err(_)` is fatal.
    ///
    /// Omitting this setter is equivalent to calling
    /// `.on_packet(Mp4DemuxerSource::default_on_packet_as_frame())`
    /// — i.e. the demuxer constructs a
    /// [`VideoFrameProxy`](savant_core::primitives::frame::VideoFrameProxy)
    /// upstream and sends
    /// [`EncodedMsg::Frame`](crate::framework::envelopes::EncodedMsg::Frame).
    pub fn on_packet<F>(mut self, f: F) -> Self
    where
        F: FnMut(DemuxedPacket, &VideoInfo, &str, &Router<EncodedMsg>) -> Result<()>
            + Send
            + 'static,
    {
        self.on_packet = Some(Box::new(f));
        self
    }

    /// Hook fired exactly once on [`Mp4DemuxerOutput::Eos`], strictly
    /// after the last [`on_packet`](Self::on_packet) call.  Decides
    /// whether to forward `EncodedMsg::SourceEos` downstream.
    ///
    /// Omitting this setter is equivalent to calling
    /// `.on_source_eos(Mp4DemuxerSource::default_on_source_eos())`.
    pub fn on_source_eos<F>(mut self, f: F) -> Self
    where
        F: FnMut(&str, &Router<EncodedMsg>) + Send + 'static,
    {
        self.on_source_eos = Some(Box::new(f));
        self
    }

    /// Hook that classifies a GStreamer pipeline error; see
    /// [`ErrorAction`].  The callback receives the structured
    /// [`Mp4DemuxerError`] so user code can pattern-match on the
    /// specific failure mode.
    ///
    /// Omitting this setter is equivalent to calling
    /// `.on_error(Mp4DemuxerSource::default_on_error())`, which
    /// returns [`ErrorAction::Fatal`] for every variant — matching
    /// the legacy mp4_demux behaviour.
    pub fn on_error<F>(mut self, f: F) -> Self
    where
        F: FnMut(&Mp4DemuxerError) -> ErrorAction + Send + 'static,
    {
        self.on_error = Some(Box::new(f));
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
            on_stream_info,
            on_packet,
            on_source_eos,
            on_error,
        } = self;
        let input = input.ok_or_else(|| anyhow!("Mp4DemuxerSource: missing input"))?;
        let source_id = source_id.ok_or_else(|| anyhow!("Mp4DemuxerSource: missing source_id"))?;
        // Auto-install the matching default forwarder for every hook
        // the user did not override.  This keeps the runtime-field
        // invariant ("hooks are never None") in sync with the
        // builder's ergonomic "skip what you don't care about" API.
        let on_stream_info: OnStreamInfoHook =
            on_stream_info.unwrap_or_else(|| Box::new(Mp4DemuxerSource::default_on_stream_info()));
        let on_packet: OnPacketHook =
            on_packet.unwrap_or_else(|| Box::new(Mp4DemuxerSource::default_on_packet_as_frame()));
        let on_source_eos: OnSourceEosHook =
            on_source_eos.unwrap_or_else(|| Box::new(Mp4DemuxerSource::default_on_source_eos()));
        let on_error: OnErrorHook =
            on_error.unwrap_or_else(|| Box::new(Mp4DemuxerSource::default_on_error()));
        Ok(SourceBuilder::new(name).factory(move |_bx| {
            Ok(Mp4DemuxerSource {
                input,
                source_id,
                downstream,
                on_stream_info,
                on_packet,
                on_source_eos,
                on_error,
            })
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::supervisor::{StageKind, StageName};

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
            .on_stream_info(|info, sid, router| {
                router.send(EncodedMsg::StreamInfo {
                    source_id: sid.to_string(),
                    info,
                });
                Ok(())
            })
            .on_packet(|pkt, info, sid, router| {
                router.send(EncodedMsg::Packet {
                    source_id: sid.to_string(),
                    info: *info,
                    packet: pkt,
                });
                Ok(())
            })
            .on_source_eos(|sid, router| {
                router.send(EncodedMsg::SourceEos {
                    source_id: sid.to_string(),
                });
            })
            .on_error(|_err: &Mp4DemuxerError| ErrorAction::Fatal)
            .build()
            .unwrap();
        // No way to run without a real file; this is a compile-shape
        // test.  The SourceBuilder carries our factory.
        let _ = sb;
    }

    /// Confirms the `default_on_*` associated functions (including
    /// the packet→packet variant) slot into the builder's generic
    /// hook bounds as-is.
    #[test]
    fn builder_accepts_default_forwarders() {
        let name = StageName::unnamed(StageKind::Mp4Demux);
        let sb = Mp4DemuxerSource::builder(name)
            .input("/tmp/x.mp4")
            .source_id("cam1")
            .downstream(StageName::unnamed(StageKind::Decoder))
            .on_stream_info(Mp4DemuxerSource::default_on_stream_info())
            .on_packet(Mp4DemuxerSource::default_on_packet())
            .on_source_eos(Mp4DemuxerSource::default_on_source_eos())
            .on_error(|_err: &Mp4DemuxerError| ErrorAction::Fatal)
            .build()
            .unwrap();
        let _ = sb;
    }

    /// Confirms the packet→frame forwarder slots into the builder
    /// in place of `default_on_packet`.
    #[test]
    fn builder_accepts_default_on_packet_as_frame() {
        let name = StageName::unnamed(StageKind::Mp4Demux);
        let sb = Mp4DemuxerSource::builder(name)
            .input("/tmp/x.mp4")
            .source_id("cam1")
            .downstream(StageName::unnamed(StageKind::Decoder))
            .on_stream_info(Mp4DemuxerSource::default_on_stream_info())
            .on_packet(Mp4DemuxerSource::default_on_packet_as_frame())
            .on_source_eos(Mp4DemuxerSource::default_on_source_eos())
            .on_error(|_err: &Mp4DemuxerError| ErrorAction::Fatal)
            .build()
            .unwrap();
        let _ = sb;
    }

    /// Runtime invariant: `build()` succeeds with only the three
    /// mandatory setters (`input`, `source_id`, `downstream`) and
    /// auto-installs all four `default_on_*` forwarders.  The
    /// `SourceFactory` resolves and produces a `Mp4DemuxerSource`
    /// whose hook fields are non-`Option` by construction — proving
    /// that no variant is ever dispatched to `None` at runtime.
    #[test]
    fn builder_accepts_bare_minimum() {
        use crate::framework::context::BuildCtx;
        use crate::framework::registry::Registry;
        use crate::framework::shared::SharedStore;

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
        // `unwrap_or_else(default_on_*)` chain.
        let parts = sb.into_parts();
        assert_eq!(parts.name, name);
        let reg = Arc::new(Registry::new());
        let shared = SharedStore::new();
        let bx = BuildCtx::new(&parts.name, &reg, &shared);
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
        } = src;
    }
}
