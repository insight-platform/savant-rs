//! [`UriDemuxerSource`] — a URI-based source (file://, http://,
//! rtsp://, rtmp://, hls://, …) that surfaces every demuxed output
//! variant to user code, along with the stage's
//! [`Router<EncodedMsg>`].
//!
//! API-symmetric with
//! [`Mp4DemuxerSource`](super::mp4_demuxer::Mp4DemuxerSource): the
//! two stages differ only in the underlying demuxer driver
//! ([`savant_gstreamer::uri_demuxer::UriDemuxer`] vs
//! [`savant_gstreamer::mp4_demuxer::Mp4Demuxer`]) and in the error
//! type surfaced by the `on_error` classifier. Every envelope
//! variant ([`EncodedMsg::StreamInfo`], [`EncodedMsg::Packet`],
//! [`EncodedMsg::Frame`], [`EncodedMsg::SourceEos`]) is identical;
//! `DemuxedPacket` / `VideoInfo` / `VideoCodec` are shared types
//! (see [`savant_gstreamer::demux`]).
//!
//! Compared to [`Mp4DemuxerSource`](super::mp4_demuxer::Mp4DemuxerSource),
//! [`UriDemuxerSource`] additionally exposes:
//!
//! * `.parsed(bool)` — forwarded to
//!   [`UriDemuxerConfig::with_parsed`]. Defaults to `true`.
//! * `.bin_properties(Vec<(String, PropertyValue)>)` — applied to
//!   the top-level `urisourcebin` element
//!   (e.g. `buffer-size`, `use-buffering`, `connection-speed`).
//! * `.source_properties(Vec<(String, PropertyValue)>)` — applied
//!   to the dynamically-created inner source element via the
//!   `source-setup` signal (e.g. `rtspsrc.latency`,
//!   `souphttpsrc.user-agent`).
//!
//! Like [`Mp4DemuxerSource`](super::mp4_demuxer::Mp4DemuxerSource),
//! the URI demuxer is driven by a user-supplied [`InputRequester`]
//! callback the framework asks before each input run (and again
//! after every underlying demuxer exits).  The callback returns a
//! [`DemuxInputRequest`] (see the
//! [demux_input](super::demuxers::demux_input) module docs); use
//! [`UriDemuxerBuilder::one_shot`] for the common single-URI case
//! and [`UriDemuxerBuilder::looped`] for an infinite replay of the
//! same URI.
//!
//! ```ignore
//! sys.register_source(
//!     UriDemuxerSource::builder(StageName::unnamed(StageKind::UriDemux))
//!         .one_shot("rtsp://cam/stream", "cam1")
//!         .downstream(StageName::unnamed(StageKind::Decoder))
//!         .source_properties(vec![
//!             ("latency".into(), PropertyValue::U64(200)),
//!         ])
//!         .results(
//!             UriDemuxerResults::builder()
//!                 .on_packet(|uri, source_id, info, pkt, router, _ctx| {
//!                     log::debug!("packet from {uri}");
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
//! Mirror the mp4 stage:
//!
//! * [`UriDemuxerSource::default_on_stream_info`]
//! * [`UriDemuxerSource::default_on_packet`]
//! * [`UriDemuxerSource::default_on_packet_as_frame`]
//! * [`UriDemuxerSource::default_on_source_eos`]
//! * [`UriDemuxerSource::default_on_error`]
//! * [`UriDemuxerSource::default_stopping`]
//!
//! ## Always-set invariant
//!
//! As with [`Mp4DemuxerSource`](super::mp4_demuxer::Mp4DemuxerSource),
//! the four variant hooks (`on_stream_info`, `on_packet`,
//! `on_source_eos`, `on_error`) are stored as non-`Option` fields.
//! Omitted setters on [`UriDemuxerBuilder`] are substituted with the
//! matching `UriDemuxerSource::default_on_*` in
//! [`UriDemuxerBuilder::build`] — so no runtime code path can drop a
//! demuxer variant on the floor.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Result};
use parking_lot::Mutex;
use savant_gstreamer::uri_demuxer::{
    PropertyValue, UriDemuxer, UriDemuxerConfig, UriDemuxerError, UriDemuxerOutput,
};
// `DemuxedPacket` and `VideoInfo` are shared between demuxers —
// import them through the mp4 stage's re-exports so both
// stages stay pinned to the same type identity.
use savant_gstreamer::mp4_demuxer::{DemuxedPacket, VideoInfo};

use crate::envelopes::EncodedMsg;
use crate::router::Router;
use crate::stages::demuxers::demux_input::{DemuxInputRequest, InputRequester};
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

/// Closure type for `on_packet` — receives the URI of the run
/// currently in progress, the `source_id`, the stream-level
/// [`VideoInfo`] (as seen in the preceding
/// [`UriDemuxerOutput::StreamInfo`]), the demuxed access unit,
/// and the router.  Typical use:
/// `router.send(EncodedMsg::Packet { source_id: source_id.into(), info: *info, packet: pkt })`.
///
/// The leading `uri` argument is the same string the
/// [`InputRequester`] returned in the [`DemuxInputRequest::Run`]
/// that started this run, so multi-URI pipelines can route per
/// URI without threading the value through external state.
///
/// The [`VideoInfo`] is forwarded in-band with every
/// [`EncodedMsg::Packet`]
/// so downstream consumers (most importantly
/// [`Decoder`](super::decoder::Decoder)) never have to
/// maintain a per-source cache of stream parameters.  The same
/// `info` also unlocks upstream frame construction (for instance
/// via [`UriDemuxerSource::default_on_packet_as_frame`]).
/// Returning `Err(_)` is fatal.
pub type OnPacketHook = Box<
    dyn FnMut(
            &str,
            &str,
            &VideoInfo,
            DemuxedPacket,
            &Router<EncodedMsg>,
            &HookCtx,
        ) -> Result<()>
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
/// [`UriDemuxerError`], the stage's [`Router<EncodedMsg>`] (so
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
    Box<dyn FnMut(&UriDemuxerError, &Router<EncodedMsg>, &HookCtx) -> ErrorAction + Send + 'static>;

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

/// A no-inbox [`Source`] that drives a
/// [`savant_gstreamer::uri_demuxer::UriDemuxer`] and forwards its
/// output downstream as
/// [`EncodedMsg`].
///
/// Construct via [`UriDemuxerSource::builder`].
///
/// # Runtime invariant
///
/// All four variant hooks are **always populated** at runtime.  The
/// builder's `Option<...>` fields are an internal "was the setter
/// called?" marker;
/// [`UriDemuxerBuilder::build`](UriDemuxerBuilder::build) always
/// substitutes the matching `UriDemuxerSource::default_on_*` before
/// constructing the source.  The GStreamer callback therefore never
/// inspects an `Option` — every demuxed variant is dispatched to a
/// hook.
pub struct UriDemuxerSource {
    request_input: InputRequester,
    downstream: Option<StageName>,
    parsed: bool,
    bin_properties: Vec<(String, PropertyValue)>,
    source_properties: Vec<(String, PropertyValue)>,
    on_stream_info: OnStreamInfoHook,
    on_packet: OnPacketHook,
    on_source_eos: OnSourceEosHook,
    on_error: OnErrorHook,
    stopping: OnStoppingHook,
}

impl UriDemuxerSource {
    /// Start a fluent [`UriDemuxerBuilder`] registered under `name`.
    pub fn builder(name: StageName) -> UriDemuxerBuilder {
        UriDemuxerBuilder::new(name)
    }

    /// Default `on_stream_info` forwarder.  The returned closure
    /// sends
    /// [`EncodedMsg::StreamInfo { source_id, info }`](crate::envelopes::EncodedMsg::StreamInfo)
    /// via `router.send(...)` — the router's default peer.
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
    /// [`UriDemuxerOutput::StreamInfo`].  The leading `uri`
    /// argument (URI of the current run) is ignored by the
    /// default forwarder.
    ///
    /// Attaching the [`VideoInfo`] in-band lets downstream consumers
    /// (notably the framework's
    /// [`Decoder`](super::decoder::Decoder)) construct
    /// decode frames on the fly without maintaining a per-source
    /// [`VideoInfo`] cache — the message is self-describing.  Use
    /// [`UriDemuxerSource::default_on_packet_as_frame`] instead
    /// when you want the demuxer to build the
    /// [`VideoFrame`](savant_core::primitives::frame::VideoFrame)
    /// upstream.
    pub fn default_on_packet() -> impl FnMut(
        &str,
        &str,
        &VideoInfo,
        DemuxedPacket,
        &Router<EncodedMsg>,
        &HookCtx,
    ) -> Result<()>
           + Send
           + 'static {
        |_uri, source_id, info, packet, router, _ctx| {
            router.send(EncodedMsg::Packet {
                source_id: source_id.to_string(),
                info: *info,
                packet,
            });
            Ok(())
        }
    }

    /// Default `on_packet` forwarder that constructs the
    /// decoder-facing [`VideoFrame`](savant_core::primitives::frame::VideoFrame)
    /// **on the demuxer side** (via
    /// [`make_decode_frame`](super::decoder::make_decode_frame)) and
    /// sends
    /// [`EncodedMsg::Frame { frame, payload: Some(bytes) }`](crate::envelopes::EncodedMsg::Frame).
    /// The leading `uri` argument (URI of the current run) is
    /// ignored.
    ///
    /// Swap for [`UriDemuxerSource::default_on_packet`] when you
    /// want the decoder to own frame construction instead.
    pub fn default_on_packet_as_frame() -> impl FnMut(
        &str,
        &str,
        &VideoInfo,
        DemuxedPacket,
        &Router<EncodedMsg>,
        &HookCtx,
    ) -> Result<()>
           + Send
           + 'static {
        |_uri, source_id, info, packet, router, _ctx| {
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
    /// [`UriDemuxerError`].  Override with a custom closure to
    /// downgrade specific error variants to
    /// [`ErrorAction::LogAndContinue`] or [`ErrorAction::Swallow`].
    pub fn default_on_error(
    ) -> impl FnMut(&UriDemuxerError, &Router<EncodedMsg>, &HookCtx) -> ErrorAction + Send + 'static
    {
        |_err, _router, _ctx| ErrorAction::Fatal
    }

    /// Default user shutdown hook — a no-op.
    pub fn default_stopping() -> impl FnMut(&SourceContext) + Send + 'static {
        |_ctx| {}
    }
}

/// Container for the hook closures plus the last-seen
/// [`VideoInfo`].  The [`UriDemuxer`] callback needs `Fn + Send +
/// Sync + 'static`; our hooks are `FnMut`, so we park them behind a
/// [`parking_lot::Mutex`] that the callback takes each time it
/// fires.
///
/// The four variant hooks are non-`Option` by construction — see the
/// runtime invariant on [`UriDemuxerSource`].  `last_stream_info` is
/// the one genuinely optional field: it is `None` until the demuxer
/// emits its first [`UriDemuxerOutput::StreamInfo`] and is read by
/// every subsequent [`UriDemuxerOutput::Packet`] so `on_packet`
/// hooks can see the stream-level metadata without managing it
/// themselves.
struct UriDemuxerHooks {
    on_stream_info: OnStreamInfoHook,
    on_packet: OnPacketHook,
    on_source_eos: OnSourceEosHook,
    on_error: OnErrorHook,
    last_stream_info: Option<VideoInfo>,
}

/// Polling cadence used while waiting for the underlying demuxer
/// to finish *and* during [`DemuxInputRequest::Idle`] sleeps.
const POLL_INTERVAL: Duration = Duration::from_millis(100);

impl Source for UriDemuxerSource {
    fn run(self, ctx: SourceContext) -> Result<()> {
        let UriDemuxerSource {
            mut request_input,
            downstream,
            parsed,
            bin_properties,
            source_properties,
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

        // Hooks live for the lifetime of the source — see the
        // comment on the mp4 stage: shared across every per-input
        // run via `Arc<Mutex<...>>`.
        let hooks: Arc<Mutex<UriDemuxerHooks>> = Arc::new(Mutex::new(UriDemuxerHooks {
            on_stream_info,
            on_packet,
            on_source_eos,
            on_error,
            last_stream_info: None,
        }));

        let outcome = (|| -> Result<()> {
            loop {
                if stop_flag.load(Ordering::Relaxed) {
                    log::info!("[{own_name}] stop flag set; not requesting next input");
                    return Ok(());
                }
                match request_input(&hook_ctx) {
                    DemuxInputRequest::Stop => {
                        log::info!("[{own_name}] input requester returned Stop");
                        return Ok(());
                    }
                    DemuxInputRequest::Idle(d) => {
                        log::info!("[{own_name}] input requester returned Idle({d:?})");
                        if !idle_sleep(d, &stop_flag) {
                            return Ok(());
                        }
                    }
                    DemuxInputRequest::Run { input, source_id } => {
                        run_one(
                            &input,
                            &source_id,
                            &own_name,
                            &router,
                            default_sink.as_ref(),
                            &stop_flag,
                            &hook_ctx,
                            &hooks,
                            parsed,
                            &bin_properties,
                            &source_properties,
                        )?;
                    }
                }
            }
        })();

        drop(hooks);

        // User stopping hook fires AFTER the run loop ends.
        (stopping)(&ctx);

        outcome
    }
}

/// Cooperative sleep used by [`DemuxInputRequest::Idle`].  Returns
/// `true` if the full duration elapsed and the requester should be
/// re-invoked, `false` if `stop_flag` flipped during the sleep.
fn idle_sleep(d: Duration, stop_flag: &Arc<std::sync::atomic::AtomicBool>) -> bool {
    let until = Instant::now() + d;
    while Instant::now() < until {
        if stop_flag.load(Ordering::Relaxed) {
            return false;
        }
        let remaining = until.saturating_duration_since(Instant::now());
        std::thread::sleep(remaining.min(POLL_INTERVAL));
    }
    !stop_flag.load(Ordering::Relaxed)
}

/// Drive one [`UriDemuxer`] from construction to drain for the
/// supplied `(input, source_id)` pair.  See the mirror function on
/// the mp4 stage for the full contract; the only differences are
/// the underlying demuxer crate and the per-build URI config
/// (`parsed`, `bin_properties`, `source_properties`).
#[allow(clippy::too_many_arguments)]
fn run_one(
    input: &str,
    source_id: &str,
    own_name: &StageName,
    router: &Router<EncodedMsg>,
    default_sink: Option<&crate::OperatorSink<EncodedMsg>>,
    stop_flag: &Arc<std::sync::atomic::AtomicBool>,
    hook_ctx: &HookCtx,
    hooks: &Arc<Mutex<UriDemuxerHooks>>,
    parsed: bool,
    bin_properties: &[(String, PropertyValue)],
    source_properties: &[(String, PropertyValue)],
) -> Result<()> {
    log::info!("[{own_name}] starting source_id={source_id} uri={input} parsed={parsed}");

    // Reset stream-info stash so a stale value from a previous run
    // can never be observed by this run's `on_packet`.
    hooks.lock().last_stream_info = None;

    // Per-iteration error latch.
    let first_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

    let router_cb = router.clone();
    let default_sink_cb = default_sink.cloned();
    let stop_flag_cb = stop_flag.clone();
    let first_error_cb = first_error.clone();
    let hooks_cb = hooks.clone();
    let source_id_cb = source_id.to_string();
    let input_cb = input.to_string();
    let own_name_cb = own_name.clone();
    let hook_ctx_cb = hook_ctx.clone();

    let mut cfg = UriDemuxerConfig::new(input).with_parsed(parsed);
    cfg.bin_properties = bin_properties.to_vec();
    cfg.source_properties = source_properties.to_vec();

    let mut demuxer = UriDemuxer::new(cfg, move |output| {
        // Cooperative-stop / already-aborted short-circuit.
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
            UriDemuxerOutput::StreamInfo(info) => {
                log::info!(
                    "[{own_name_cb}] stream info: source_id={source_id_cb} {}x{} @ {}/{} codec={:?}",
                    info.width,
                    info.height,
                    info.framerate_num,
                    info.framerate_den,
                    info.codec
                );
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
            UriDemuxerOutput::Packet(pkt) => {
                if let Some(info) = h.last_stream_info {
                    if let Err(e) = (h.on_packet)(
                        &input_cb,
                        &source_id_cb,
                        &info,
                        pkt,
                        &router_cb,
                        &hook_ctx_cb,
                    ) {
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
            UriDemuxerOutput::Eos => {
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
            UriDemuxerOutput::Error(e) => {
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
    .map_err(|e| anyhow!("UriDemuxer::new: {e}"))?;

    let mut stopped_by_flag = false;
    while !demuxer.wait_timeout(POLL_INTERVAL) {
        if stop_flag.load(Ordering::Relaxed) {
            log::info!("[{own_name}] stop flag set; finishing URI demuxer (source_id={source_id})");
            stopped_by_flag = true;
            break;
        }
    }
    demuxer.finish();
    let codec = demuxer.detected_codec();
    log::info!(
        "[{own_name}] finished, detected_codec={codec:?} stopped_by_flag={stopped_by_flag}"
    );
    drop(demuxer);

    if let Some(err) = first_error.lock().take() {
        bail!("[{own_name}] demux error: {err}");
    }
    if codec.is_none() && !stopped_by_flag {
        bail!("[{own_name}] demuxer did not detect a video codec (empty stream?)");
    }
    Ok(())
}

fn latch_error(slot: &Arc<Mutex<Option<String>>>, msg: String) {
    let mut g = slot.lock();
    if g.is_none() {
        *g = Some(msg);
    }
}

/// Per-variant [`UriDemuxerOutput`] hook bundle — one branch per
/// demuxed variant plus the error classifier.
///
/// Built through [`UriDemuxerResults::builder`] and handed to
/// [`UriDemuxerBuilder::results`].  Omitted branches auto-install
/// the matching `UriDemuxerSource::default_on_*` at build time.
pub struct UriDemuxerResults {
    on_stream_info: OnStreamInfoHook,
    on_packet: OnPacketHook,
    on_source_eos: OnSourceEosHook,
    on_error: OnErrorHook,
}

impl UriDemuxerResults {
    /// Start a builder that auto-installs every default on
    /// [`UriDemuxerResultsBuilder::build`].
    pub fn builder() -> UriDemuxerResultsBuilder {
        UriDemuxerResultsBuilder::new()
    }
}

impl Default for UriDemuxerResults {
    /// Every branch wired to its matching
    /// `UriDemuxerSource::default_on_*`.  `on_packet` defaults to
    /// [`UriDemuxerSource::default_on_packet_as_frame`], i.e.
    /// upstream frame construction.
    fn default() -> Self {
        UriDemuxerResultsBuilder::new().build()
    }
}

/// Fluent builder for [`UriDemuxerResults`].
pub struct UriDemuxerResultsBuilder {
    on_stream_info: Option<OnStreamInfoHook>,
    on_packet: Option<OnPacketHook>,
    on_source_eos: Option<OnSourceEosHook>,
    on_error: Option<OnErrorHook>,
}

impl UriDemuxerResultsBuilder {
    /// Empty bundle — every hook defaults to its matching
    /// `UriDemuxerSource::default_*` equivalent at
    /// [`build`](Self::build) time.
    pub fn new() -> Self {
        Self {
            on_stream_info: None,
            on_packet: None,
            on_source_eos: None,
            on_error: None,
        }
    }

    /// Override the `on_stream_info` hook.
    pub fn on_stream_info<F>(mut self, f: F) -> Self
    where
        F: FnMut(VideoInfo, &str, &Router<EncodedMsg>, &HookCtx) -> Result<()> + Send + 'static,
    {
        self.on_stream_info = Some(Box::new(f));
        self
    }

    /// Override the `on_packet` hook.  Omitting this setter is
    /// equivalent to calling
    /// `.on_packet(UriDemuxerSource::default_on_packet_as_frame())`.
    ///
    /// Closure signature: `(uri, source_id, info, pkt, router, ctx)`.
    /// `uri` is the URI returned by the [`InputRequester`] for the
    /// run currently in progress.
    pub fn on_packet<F>(mut self, f: F) -> Self
    where
        F: FnMut(
                &str,
                &str,
                &VideoInfo,
                DemuxedPacket,
                &Router<EncodedMsg>,
                &HookCtx,
            ) -> Result<()>
            + Send
            + 'static,
    {
        self.on_packet = Some(Box::new(f));
        self
    }

    /// Override the `on_source_eos` hook.
    pub fn on_source_eos<F>(mut self, f: F) -> Self
    where
        F: FnMut(&str, &Router<EncodedMsg>, &HookCtx) -> Result<Flow> + Send + 'static,
    {
        self.on_source_eos = Some(Box::new(f));
        self
    }

    /// Override the `on_error` classifier.
    pub fn on_error<F>(mut self, f: F) -> Self
    where
        F: FnMut(&UriDemuxerError, &Router<EncodedMsg>, &HookCtx) -> ErrorAction + Send + 'static,
    {
        self.on_error = Some(Box::new(f));
        self
    }

    /// Finalise the bundle, substituting defaults for every omitted
    /// branch.
    pub fn build(self) -> UriDemuxerResults {
        let UriDemuxerResultsBuilder {
            on_stream_info,
            on_packet,
            on_source_eos,
            on_error,
        } = self;
        UriDemuxerResults {
            on_stream_info: on_stream_info
                .unwrap_or_else(|| Box::new(UriDemuxerSource::default_on_stream_info())),
            on_packet: on_packet
                .unwrap_or_else(|| Box::new(UriDemuxerSource::default_on_packet_as_frame())),
            on_source_eos: on_source_eos
                .unwrap_or_else(|| Box::new(UriDemuxerSource::default_on_source_eos())),
            on_error: on_error.unwrap_or_else(|| Box::new(UriDemuxerSource::default_on_error())),
        }
    }
}

impl Default for UriDemuxerResultsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Loop-level common knobs + user shutdown hook for
/// [`UriDemuxerSource`].  Built through
/// [`UriDemuxerCommon::builder`] and handed to
/// [`UriDemuxerBuilder::common`].
pub struct UriDemuxerCommon {
    stopping: OnStoppingHook,
}

impl UriDemuxerCommon {
    /// Start a builder seeded with the no-op stopping hook.
    pub fn builder() -> UriDemuxerCommonBuilder {
        UriDemuxerCommonBuilder::new()
    }
}

impl Default for UriDemuxerCommon {
    fn default() -> Self {
        UriDemuxerCommonBuilder::new().build()
    }
}

/// Fluent builder for [`UriDemuxerCommon`].
pub struct UriDemuxerCommonBuilder {
    stopping: Option<OnStoppingHook>,
}

impl UriDemuxerCommonBuilder {
    /// Empty bundle — `stopping` defaults to a no-op.
    pub fn new() -> Self {
        Self { stopping: None }
    }

    /// Override the user shutdown hook.
    pub fn stopping<F>(mut self, f: F) -> Self
    where
        F: FnMut(&SourceContext) + Send + 'static,
    {
        self.stopping = Some(Box::new(f));
        self
    }

    /// Finalise the bundle.
    pub fn build(self) -> UriDemuxerCommon {
        let UriDemuxerCommonBuilder { stopping } = self;
        UriDemuxerCommon {
            stopping: stopping.unwrap_or_else(|| Box::new(UriDemuxerSource::default_stopping())),
        }
    }
}

impl Default for UriDemuxerCommonBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Fluent builder for [`UriDemuxerSource`].
///
/// The builder exposes wiring-level configuration at the top level
/// (`input` and its `one_shot` / `looped` shortcuts, plus
/// `downstream`, `parsed`, `bin_properties`,
/// `source_properties`).  Per-variant demuxer-output hooks live on
/// [`UriDemuxerResults`]; the user shutdown hook lives on
/// [`UriDemuxerCommon`].
pub struct UriDemuxerBuilder {
    name: StageName,
    request_input: Option<InputRequester>,
    downstream: Option<StageName>,
    parsed: bool,
    bin_properties: Vec<(String, PropertyValue)>,
    source_properties: Vec<(String, PropertyValue)>,
    results: Option<UriDemuxerResults>,
    common: Option<UriDemuxerCommon>,
}

impl UriDemuxerBuilder {
    /// Start a builder for a URI demux source registered under
    /// `name`.  `parsed` defaults to `true` (matches
    /// [`UriDemuxerConfig::new`]); bin/source property vectors
    /// default to empty.
    pub fn new(name: StageName) -> Self {
        Self {
            name,
            request_input: None,
            downstream: None,
            parsed: true,
            bin_properties: Vec::new(),
            source_properties: Vec::new(),
            results: None,
            common: None,
        }
    }

    /// Install the [`InputRequester`] driving this source.
    ///
    /// The framework calls `f` once at startup and again each time
    /// the underlying demuxer finishes a clean drain.  See
    /// [`DemuxInputRequest`] and the
    /// [`Mp4DemuxerSource`](super::mp4_demuxer::Mp4DemuxerSource)
    /// module docs for the full loop semantics — the URI source
    /// behaves identically.
    ///
    /// For the common single-URI case use the
    /// [`one_shot`](Self::one_shot) shortcut; for an infinite
    /// replay of the same URI use [`looped`](Self::looped).
    pub fn input<F>(mut self, f: F) -> Self
    where
        F: FnMut(&HookCtx) -> DemuxInputRequest + Send + 'static,
    {
        self.request_input = Some(Box::new(f));
        self
    }

    /// Convenience for the single-URI case: emit one
    /// [`DemuxInputRequest::Run`] for `(input, source_id)` then
    /// [`DemuxInputRequest::Stop`] on every subsequent invocation.
    ///
    /// Equivalent to
    /// `.input(crate::stages::demuxers::demux_input::one_shot_requester(input, source_id))`.
    pub fn one_shot(
        mut self,
        input: impl Into<String>,
        source_id: impl Into<String>,
    ) -> Self {
        self.request_input = Some(crate::stages::demuxers::demux_input::one_shot_requester(
            input, source_id,
        ));
        self
    }

    /// Convenience for an infinitely-looping single-URI source:
    /// emit [`DemuxInputRequest::Run`] for the same
    /// `(input, source_id)` pair on every invocation.  The source
    /// only exits when `stop_flag` flips.
    ///
    /// Equivalent to
    /// `.input(crate::stages::demuxers::demux_input::looped_requester(input, source_id))`.
    pub fn looped(
        mut self,
        input: impl Into<String>,
        source_id: impl Into<String>,
    ) -> Self {
        self.request_input = Some(crate::stages::demuxers::demux_input::looped_requester(
            input, source_id,
        ));
        self
    }

    /// Optional default peer installed on the
    /// [`Router<EncodedMsg>`] handed to every variant hook.
    pub fn downstream(mut self, peer: StageName) -> Self {
        self.downstream = Some(peer);
        self
    }

    /// Whether to request a parsed (Annex-B / AU-aligned) elementary
    /// stream from the underlying
    /// [`savant_gstreamer::uri_demuxer::UriDemuxer`].  Forwarded to
    /// [`UriDemuxerConfig::with_parsed`].  Defaults to `true`.
    pub fn parsed(mut self, parsed: bool) -> Self {
        self.parsed = parsed;
        self
    }

    /// Properties applied to the top-level `urisourcebin`
    /// (e.g. `buffer-size`, `use-buffering`, `connection-speed`).
    /// Replaces any previously-accumulated bin-property set.
    pub fn bin_properties(mut self, props: Vec<(String, PropertyValue)>) -> Self {
        self.bin_properties = props;
        self
    }

    /// Properties applied to the dynamically-created inner source
    /// element via the `source-setup` signal
    /// (e.g. `rtspsrc.latency`, `souphttpsrc.user-agent`).  Replaces
    /// any previously-accumulated source-property set.
    pub fn source_properties(mut self, props: Vec<(String, PropertyValue)>) -> Self {
        self.source_properties = props;
        self
    }

    /// Install a [`UriDemuxerResults`] bundle — one branch per
    /// demuxed variant.
    pub fn results(mut self, r: UriDemuxerResults) -> Self {
        self.results = Some(r);
        self
    }

    /// Install a [`UriDemuxerCommon`] bundle.
    pub fn common(mut self, c: UriDemuxerCommon) -> Self {
        self.common = Some(c);
        self
    }

    /// Finalise the builder and obtain the Layer-A
    /// [`SourceBuilder<UriDemuxerSource>`] ready for
    /// [`System::register_source`](super::super::System::register_source).
    ///
    /// # Errors
    ///
    /// Returns `Err` if no [`InputRequester`] has been installed
    /// (via [`input`](Self::input), [`one_shot`](Self::one_shot),
    /// or [`looped`](Self::looped)).  `downstream` is optional —
    /// omit it for a drain-only source.
    pub fn build(self) -> Result<SourceBuilder<UriDemuxerSource>> {
        let UriDemuxerBuilder {
            name,
            request_input,
            downstream,
            parsed,
            bin_properties,
            source_properties,
            results,
            common,
        } = self;
        let request_input = request_input.ok_or_else(|| {
            anyhow!(
                "UriDemuxerSource: missing input (call .input(...), .one_shot(...), or .looped(...))"
            )
        })?;
        let UriDemuxerResults {
            on_stream_info,
            on_packet,
            on_source_eos,
            on_error,
        } = results.unwrap_or_default();
        let UriDemuxerCommon { stopping } = common.unwrap_or_default();
        Ok(SourceBuilder::new(name).factory(move |_bx| {
            Ok(UriDemuxerSource {
                request_input,
                downstream,
                parsed,
                bin_properties,
                source_properties,
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

    fn err_msg(result: Result<SourceBuilder<UriDemuxerSource>>) -> String {
        match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected error"),
        }
    }

    #[test]
    fn builder_requires_input() {
        let name = StageName::unnamed(StageKind::UriDemux);
        assert!(err_msg(UriDemuxerSource::builder(name).build()).contains("missing input"));
    }

    /// `downstream` is optional — a drain-only source (no default
    /// peer) still builds.
    #[test]
    fn builder_without_downstream_is_accepted() {
        let name = StageName::unnamed(StageKind::UriDemux);
        let _ = UriDemuxerSource::builder(name)
            .one_shot("file:///tmp/x.mp4", "s")
            .build()
            .expect("no-downstream builder is accepted");
    }

    #[test]
    fn builder_accepts_all_hooks() {
        let name = StageName::unnamed(StageKind::UriDemux);
        let sb = UriDemuxerSource::builder(name)
            .one_shot("file:///tmp/x.mp4", "cam1")
            .downstream(StageName::unnamed(StageKind::Decoder))
            .parsed(true)
            .bin_properties(vec![("buffer-size".into(), PropertyValue::I64(8192))])
            .source_properties(vec![("latency".into(), PropertyValue::U64(200))])
            .results(
                UriDemuxerResults::builder()
                    .on_stream_info(|info, sid, router, _ctx| {
                        router.send(EncodedMsg::StreamInfo {
                            source_id: sid.to_string(),
                            info,
                        });
                        Ok(())
                    })
                    .on_packet(|uri, sid, info, pkt, router, _ctx| {
                        let _ = uri;
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
                        |_err: &UriDemuxerError, _router: &Router<EncodedMsg>, _ctx: &HookCtx| {
                            ErrorAction::Fatal
                        },
                    )
                    .build(),
            )
            .build()
            .unwrap();
        let _ = sb;
    }

    /// Confirms the `default_on_*` associated functions (including
    /// the packet→packet variant) slot into the bundle builder's
    /// generic hook bounds as-is.
    #[test]
    fn builder_accepts_default_forwarders() {
        let name = StageName::unnamed(StageKind::UriDemux);
        let sb = UriDemuxerSource::builder(name)
            .one_shot("file:///tmp/x.mp4", "cam1")
            .downstream(StageName::unnamed(StageKind::Decoder))
            .results(
                UriDemuxerResults::builder()
                    .on_stream_info(UriDemuxerSource::default_on_stream_info())
                    .on_packet(UriDemuxerSource::default_on_packet())
                    .on_source_eos(UriDemuxerSource::default_on_source_eos())
                    .on_error(UriDemuxerSource::default_on_error())
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
        let name = StageName::unnamed(StageKind::UriDemux);
        let sb = UriDemuxerSource::builder(name)
            .one_shot("file:///tmp/x.mp4", "cam1")
            .downstream(StageName::unnamed(StageKind::Decoder))
            .results(
                UriDemuxerResults::builder()
                    .on_stream_info(UriDemuxerSource::default_on_stream_info())
                    .on_packet(UriDemuxerSource::default_on_packet_as_frame())
                    .on_source_eos(UriDemuxerSource::default_on_source_eos())
                    .on_error(UriDemuxerSource::default_on_error())
                    .build(),
            )
            .build()
            .unwrap();
        let _ = sb;
    }

    /// The [`UriDemuxerCommon`] bundle accepts a user-supplied
    /// `.stopping(F)` closure.
    #[test]
    fn builder_accepts_user_stopping() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let flag = Arc::new(AtomicBool::new(false));
        let flag_hook = flag.clone();
        let name = StageName::unnamed(StageKind::UriDemux);
        let _ = UriDemuxerSource::builder(name)
            .one_shot("file:///tmp/x.mp4", "cam1")
            .downstream(StageName::unnamed(StageKind::Decoder))
            .common(
                UriDemuxerCommon::builder()
                    .stopping(move |_ctx| {
                        flag_hook.store(true, Ordering::SeqCst);
                    })
                    .build(),
            )
            .build()
            .unwrap();
        assert!(!flag.load(Ordering::SeqCst));
    }

    /// Runtime invariant: `build()` succeeds with only the
    /// mandatory setters (`one_shot` + `downstream`) and
    /// auto-installs all four `default_on_*` forwarders plus the
    /// no-op stopping hook, defaulting `parsed = true` and empty
    /// property vectors.
    #[test]
    fn builder_accepts_bare_minimum() {
        use crate::context::BuildCtx;
        use crate::registry::Registry;
        use crate::shared::SharedStore;

        let name = StageName::unnamed(StageKind::UriDemux);
        let sb = UriDemuxerSource::builder(name.clone())
            .one_shot("file:///tmp/x.mp4", "cam1")
            .downstream(StageName::unnamed(StageKind::Decoder))
            .build()
            .expect("bare-minimum builder is accepted");
        let parts = sb.into_parts();
        assert_eq!(parts.name, name);
        let reg = Arc::new(Registry::new());
        let shared = Arc::new(SharedStore::new());
        let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let bx = BuildCtx::new(&parts.name, &reg, &shared, &stop_flag);
        let src = (parts.factory)(&bx).expect("factory resolves");
        // Pattern-match proves every hook is non-`Option`.
        let UriDemuxerSource {
            request_input: _,
            downstream: _,
            parsed,
            bin_properties,
            source_properties,
            on_stream_info: _,
            on_packet: _,
            on_source_eos: _,
            on_error: _,
            stopping: _,
        } = src;
        assert!(parsed, "parsed defaults to true");
        assert!(bin_properties.is_empty());
        assert!(source_properties.is_empty());
    }

    /// The custom [`input`] setter accepts an arbitrary `FnMut`
    /// returning a [`DemuxInputRequest`].
    #[test]
    fn builder_accepts_custom_input() {
        use crate::context::BuildCtx;
        use crate::registry::Registry;
        use crate::shared::SharedStore;

        let name = StageName::unnamed(StageKind::UriDemux);
        let sb = UriDemuxerSource::builder(name)
            .input(|_ctx| DemuxInputRequest::Idle(Duration::from_millis(50)))
            .downstream(StageName::unnamed(StageKind::Decoder))
            .build()
            .unwrap();
        let parts = sb.into_parts();
        let reg = Arc::new(Registry::new());
        let shared = Arc::new(SharedStore::new());
        let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let bx = BuildCtx::new(&parts.name, &reg, &shared, &stop_flag);
        let mut src = (parts.factory)(&bx).unwrap();
        let hc = bx.hook_ctx();
        match (src.request_input)(&hc) {
            DemuxInputRequest::Idle(d) => assert_eq!(d, Duration::from_millis(50)),
            other => panic!("expected Idle, got {other:?}"),
        }
    }

    /// `looped` always returns the same `Run`.
    #[test]
    fn looped_yields_run_repeatedly() {
        use crate::context::BuildCtx;
        use crate::registry::Registry;
        use crate::shared::SharedStore;

        let name = StageName::unnamed(StageKind::UriDemux);
        let sb = UriDemuxerSource::builder(name)
            .looped("rtsp://cam/stream", "loop")
            .build()
            .unwrap();
        let parts = sb.into_parts();
        let reg = Arc::new(Registry::new());
        let shared = Arc::new(SharedStore::new());
        let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let bx = BuildCtx::new(&parts.name, &reg, &shared, &stop_flag);
        let mut src = (parts.factory)(&bx).unwrap();
        let hc = bx.hook_ctx();
        for _ in 0..3 {
            assert!(matches!(
                (src.request_input)(&hc),
                DemuxInputRequest::Run { .. }
            ));
        }
    }

    /// `idle_sleep` honours the shared stop flag.
    #[test]
    fn idle_sleep_short_circuits_on_stop_flag() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let flag = Arc::new(AtomicBool::new(false));
        let flag_for_sleep = flag.clone();
        let handle = std::thread::spawn(move || {
            idle_sleep(Duration::from_secs(60), &flag_for_sleep)
        });
        std::thread::sleep(Duration::from_millis(50));
        flag.store(true, Ordering::Relaxed);
        let elapsed_ok = handle.join().expect("thread joins");
        assert!(!elapsed_ok, "stop_flag must short-circuit the sleep");
    }
}
