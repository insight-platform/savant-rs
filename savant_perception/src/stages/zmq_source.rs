//! [`ZmqSource`] — ZeroMQ ingress source for [`EncodedMsg`].
//!
//! The stage wraps [`savant_core::transport::zeromq::NonBlockingReader`]
//! and polls it in a cooperative loop.  The runtime dispatches every
//! `ReaderResult::Message` by [`Message`] kind to one of three
//! orthogonal hooks; each hook is the sole drop/send authority for
//! its kind (it either calls `router.send(...)` or it does not):
//!
//! * `on_message`         — fired only for [`Message::is_video_frame`] —
//!   the default forwards the frame as [`EncodedMsg::Frame`] with
//!   multipart-first carrier resolution.
//! * `on_source_eos`      — fired only for [`Message::is_end_of_stream`] —
//!   the default forwards [`EncodedMsg::SourceEos`] and returns
//!   [`Flow::Cont`] (ZMQ is *always* multi-stream; per-subcommand
//!   single-stream behaviour is layered on top by the binary).
//! * `on_service_message` — fired for every other [`Message`] kind
//!   (`UserData`, `VideoFrameUpdate`, `VideoFrameBatch`, `Shutdown`,
//!   `Unknown`).  The default logs at `debug` and drops; users
//!   override to forward custom protobufs.
//!
//! Protocol/receive errors are classified via [`ErrorAction`].
//!
//! Runtime invariant: all hook slots on [`ZmqSource`] are non-`Option` boxed
//! closures; omitted builder setters are substituted with defaults at build
//! time.

use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use savant_core::message::Message;
use savant_core::primitives::frame::VideoFrameContent;
use savant_core::transport::zeromq::{NonBlockingReader, ReaderConfig, ReaderResult};

use crate::envelopes::EncodedMsg;
use crate::router::Router;
use crate::supervisor::StageName;

use opentelemetry::trace::TraceContextExt;
use crate::{ErrorAction, Flow, HookCtx, Source, SourceBuilder, SourceContext};

/// Default maximum in-flight receive results buffered by
/// [`NonBlockingReader`].
pub const DEFAULT_RESULTS_QUEUE_SIZE: usize = 16;

/// Default source-loop poll cadence.
pub const DEFAULT_POLL_TIMEOUT: Duration = Duration::from_millis(100);

/// Hook fired only for video-frame messages.
///
/// The hook is the drop/send authority for the frame: call
/// `router.send(EncodedMsg::Frame { .. })` to forward, or simply
/// return without sending to drop.  EOS messages flow through
/// [`OnSourceEosHook`]; non-frame, non-EOS kinds flow through
/// [`OnServiceMessageHook`].
pub type OnMessageHook = Box<
    dyn FnMut(&HookCtx, &Router<EncodedMsg>, String, Box<Message>, Vec<Vec<u8>>) -> Result<()>
        + Send
        + 'static,
>;

/// Hook fired for non-frame, non-EOS service-message kinds
/// (`UserData`, `VideoFrameUpdate`, `VideoFrameBatch`, `Shutdown`,
/// `Unknown`).
///
/// Same drop/send contract as [`OnMessageHook`]: the hook is the
/// sole authority — call `router.send(...)` to forward, or return
/// without sending to drop.  The default implementation logs at
/// `debug` and drops the message.
pub type OnServiceMessageHook = Box<
    dyn FnMut(&HookCtx, &Router<EncodedMsg>, String, Box<Message>, Vec<Vec<u8>>) -> Result<()>
        + Send
        + 'static,
>;

/// Protocol-level reader-result classifier for non-message variants.
pub type OnProtocolErrorHook =
    Box<dyn FnMut(&HookCtx, &ReaderResult) -> ErrorAction + Send + 'static>;

/// Reader receive-error classifier.
pub type OnReceiveErrorHook =
    Box<dyn FnMut(&HookCtx, &anyhow::Error) -> ErrorAction + Send + 'static>;

/// Wire-EOS hook with stop/continue control.
pub type OnSourceEosHook =
    Box<dyn FnMut(&HookCtx, &Router<EncodedMsg>, &str) -> Result<Flow> + Send + 'static>;

/// User stopping hook fired after reader shutdown.
pub type OnStoppingHook = Box<dyn FnMut(&SourceContext) + Send + 'static>;

/// No-inbox source that receives ZeroMQ messages and emits [`EncodedMsg`].
pub struct ZmqSource {
    config: ReaderConfig,
    results_queue_size: usize,
    downstream: Option<StageName>,
    on_message: OnMessageHook,
    on_service_message: OnServiceMessageHook,
    on_protocol_error: OnProtocolErrorHook,
    on_receive_error: OnReceiveErrorHook,
    on_source_eos: OnSourceEosHook,
    poll_timeout: Duration,
    on_stopping: OnStoppingHook,
}

impl ZmqSource {
    /// Start a fluent builder for stage `name`.
    pub fn builder(name: StageName) -> ZmqSourceBuilder {
        ZmqSourceBuilder::new(name)
    }

    /// Default frame-message hook: forwards the wire video frame as
    /// [`EncodedMsg::Frame`], applying multipart-first carrier
    /// resolution.
    ///
    /// The runtime guarantees this hook is invoked only when
    /// [`Message::is_video_frame`] is `true`.
    ///
    /// # Cross-process trace continuation
    ///
    /// The wire [`Message`]'s `span_context` (a W3C-style
    /// [`PropagatedContext`](savant_core::otlp::PropagatedContext)
    /// populated by an upstream sink via the same hook on the
    /// other end of the wire) is extracted into an
    /// [`opentelemetry::Context`] and installed on the
    /// reconstructed [`VideoFrame`].  The frame's RAII span guard
    /// then bounds the local trace tree exactly as if the frame had
    /// been produced in-process — no special handling required by
    /// downstream stages.
    ///
    /// Frames whose inbound message carried no propagated context
    /// arrive with `otel_ctx = None` and remain untraced locally;
    /// the framework's stage / callback span sites collapse to
    /// no-ops on the unsampled path.
    #[allow(clippy::type_complexity)]
    pub fn default_on_message(
    ) -> impl FnMut(&HookCtx, &Router<EncodedMsg>, String, Box<Message>, Vec<Vec<u8>>) -> Result<()>
           + Send
           + 'static {
        |_ctx, router, _topic, message, data| {
            let mut frame = message
                .as_video_frame()
                .ok_or_else(|| anyhow!("ZmqSource::default_on_message: not a video_frame"))?;
            // Cross-process trace continuation: lift the wire span
            // context onto the frame so the local trace tree
            // continues from the producer side.
            let propagated = message.get_span_context();
            if !propagated.0.is_empty() {
                let extracted = propagated.extract();
                if extracted.span().span_context().is_valid() {
                    frame.set_otel_ctx(extracted);
                }
            }
            let payload = if let Some(first) = data.into_iter().next() {
                if first.is_empty() {
                    None
                } else {
                    // Explicit multipart payload wins; clear frame
                    // content to avoid duplicated carriers.
                    frame.set_content(VideoFrameContent::None);
                    Some(first)
                }
            } else {
                None
            };
            router.send(EncodedMsg::frame(frame, payload));
            Ok(())
        }
    }

    /// Default `on_service_message` hook: log at `debug` and drop.
    ///
    /// Override to forward `UserData`, `VideoFrameUpdate`,
    /// `VideoFrameBatch`, `Shutdown`, or `Unknown` kinds onto the
    /// framework as custom envelopes.
    #[allow(clippy::type_complexity)]
    pub fn default_on_service_message(
    ) -> impl FnMut(&HookCtx, &Router<EncodedMsg>, String, Box<Message>, Vec<Vec<u8>>) -> Result<()>
           + Send
           + 'static {
        |_ctx, _router, topic, _message, _data| {
            log::debug!("ignoring non-frame/non-EOS message kind, topic={topic}");
            Ok(())
        }
    }

    /// Default protocol-error classifier: warn-and-continue.
    pub fn default_on_protocol_error(
    ) -> impl FnMut(&HookCtx, &ReaderResult) -> ErrorAction + Send + 'static {
        |_ctx, _result| ErrorAction::LogAndContinue
    }

    /// Default receive-error classifier: fatal.
    pub fn default_on_receive_error(
    ) -> impl FnMut(&HookCtx, &anyhow::Error) -> ErrorAction + Send + 'static {
        |_ctx, _error| ErrorAction::Fatal
    }

    /// Default source-EOS forwarder:
    ///
    /// * forwards [`EncodedMsg::SourceEos`] for `source_id` to
    ///   the downstream router;
    /// * returns [`Flow::Cont`] so the source keeps reading. ZMQ
    ///   is inherently multi-producer: another publisher may
    ///   still be sending into this endpoint, and a per-source
    ///   EOS is not a reason to tear down the source actor.
    ///
    /// # Overriding
    ///
    /// Returning [`Flow::Stop`] does **not** stop a single
    /// stream — it terminates this `ZmqSource` actor. The thread
    /// breaks out of its receive loop, shuts the ZMQ reader
    /// down, runs the user
    /// [`stopping`](ZmqSourceCommonBuilder::stopping) hook, and
    /// exits. The exit raises a
    /// [`ShutdownCause::StageExit`](super::super::shutdown::ShutdownCause::StageExit)
    /// in the supervisor; what happens to the rest of the
    /// system is then up to the application's
    /// [`ShutdownHandler`](super::super::shutdown::ShutdownHandler)
    /// (the default broadcasts a graceful shutdown to every
    /// actor).
    ///
    /// An override that returns `Stop` is responsible for
    /// forwarding [`EncodedMsg::SourceEos`] to the router
    /// *before* returning, otherwise downstream stages will
    /// never observe a per-source EOS for `source_id` (only the
    /// system-level shutdown sentinel, if any).
    pub fn default_on_source_eos(
    ) -> impl FnMut(&HookCtx, &Router<EncodedMsg>, &str) -> Result<Flow> + Send + 'static {
        |_ctx, router, source_id| {
            router.send(EncodedMsg::source_eos(source_id));
            Ok(Flow::Cont)
        }
    }

    /// Default user stopping hook: no-op.
    pub fn default_on_stopping() -> impl FnMut(&SourceContext) + Send + 'static {
        |_ctx| {}
    }
}

impl Source for ZmqSource {
    fn run(self, ctx: SourceContext) -> Result<()> {
        let ZmqSource {
            config,
            results_queue_size,
            downstream,
            on_message,
            on_service_message,
            on_protocol_error,
            on_receive_error,
            on_source_eos,
            poll_timeout,
            on_stopping,
        } = self;

        let mut on_message = on_message;
        let mut on_service_message = on_service_message;
        let mut on_protocol_error = on_protocol_error;
        let mut on_receive_error = on_receive_error;
        let mut on_source_eos = on_source_eos;
        let mut on_stopping = on_stopping;

        let own_name = ctx.own_name().clone();
        let hook_ctx = ctx.hook_ctx();
        let router: Router<EncodedMsg> = ctx.router(downstream.as_ref())?;
        let stop_flag = ctx.stop_flag();

        log::info!(
            "[{own_name}] starting ZMQ source endpoint={}",
            config.endpoint()
        );

        let mut reader = NonBlockingReader::new(&config, results_queue_size)
            .map_err(|e| anyhow!("NonBlockingReader::new: {e}"))?;
        reader
            .start()
            .map_err(|e| anyhow!("NonBlockingReader::start: {e}"))?;

        let mut latched: Option<String> = None;

        'outer: loop {
            if stop_flag.load(Ordering::Relaxed) {
                log::info!("[{own_name}] stop flag set; exiting receive loop");
                break;
            }

            match reader.try_receive() {
                None => {
                    std::thread::sleep(poll_timeout / 4);
                    continue;
                }
                Some(Err(e)) => {
                    log::error!("[{own_name}] reader.receive: {e}");
                    // No frame on a receive error — no span.
                    let action = on_receive_error(&hook_ctx, &e);
                    match action {
                        ErrorAction::Fatal => {
                            latched.get_or_insert_with(|| e.to_string());
                            break 'outer;
                        }
                        ErrorAction::LogAndContinue => {
                            latched.get_or_insert_with(|| e.to_string());
                        }
                        ErrorAction::Swallow => {}
                    }
                }
                Some(Ok(ReaderResult::Timeout)) => continue,
                Some(Ok(ReaderResult::Message {
                    message,
                    topic,
                    routing_id: _,
                    data,
                })) => {
                    let topic_str = String::from_utf8_lossy(&topic).to_string();
                    let msg = *message;

                    if msg.is_video_frame() {
                        let boxed = Box::new(msg);
                        // Source-side metrics: count one outbound
                        // frame per video-frame message read off
                        // the wire.
                        hook_ctx.stage_metrics().record_message(1, 0, 0);
                        // The user hook materialises a `VideoFrame`
                        // from the wire message; before that call
                        // there is no frame context to parent
                        // against, so no span here.  The hook's
                        // default impl calls `frame.set_otel_ctx`
                        // with the extracted W3C parent — downstream
                        // stages take it from there.
                        if let Err(e) = on_message(&hook_ctx, &router, topic_str, boxed, data) {
                            log::error!("[{own_name}] on_message: {e}");
                            latched.get_or_insert_with(|| format!("on_message: {e}"));
                            break 'outer;
                        }
                    } else if msg.is_end_of_stream() {
                        let source_id = msg
                            .as_end_of_stream()
                            .map(|eos| eos.source_id.clone())
                            .unwrap_or(topic_str);
                        // No frame on a SourceEos — no span.
                        match on_source_eos(&hook_ctx, &router, &source_id) {
                            Ok(Flow::Cont) => {}
                            Ok(Flow::Stop) => {
                                log::info!(
                                    "[{own_name}] on_source_eos requested stop ({source_id})"
                                );
                                break 'outer;
                            }
                            Err(e) => {
                                log::error!("[{own_name}] on_source_eos: {e}");
                                latched.get_or_insert_with(|| format!("on_source_eos: {e}"));
                                break 'outer;
                            }
                        }
                    } else {
                        let boxed = Box::new(msg);
                        // Service messages don't carry a video frame — no span.
                        if let Err(e) =
                            on_service_message(&hook_ctx, &router, topic_str, boxed, data)
                        {
                            log::error!("[{own_name}] on_service_message: {e}");
                            latched.get_or_insert_with(|| format!("on_service_message: {e}"));
                            break 'outer;
                        }
                    }
                }
                Some(Ok(other)) => {
                    log::warn!("[{own_name}] protocol error: {other:?}");
                    // Protocol error has no frame — no span.
                    let action = on_protocol_error(&hook_ctx, &other);
                    match action {
                        ErrorAction::Fatal => {
                            latched.get_or_insert_with(|| format!("protocol error: {other:?}"));
                            break 'outer;
                        }
                        ErrorAction::LogAndContinue => {
                            latched.get_or_insert_with(|| format!("protocol error: {other:?}"));
                        }
                        ErrorAction::Swallow => {}
                    }
                }
            }
        }

        if let Err(e) = reader.shutdown() {
            log::warn!("[{own_name}] reader.shutdown: {e}");
        }
        drop(reader);

        on_stopping(&ctx);

        if let Some(err) = latched {
            bail!("[{own_name}] zmq source error: {err}");
        }
        Ok(())
    }
}

/// Result-path hook bundle for [`ZmqSource`].
pub struct ZmqSourceResults {
    on_message: OnMessageHook,
    on_service_message: OnServiceMessageHook,
    on_protocol_error: OnProtocolErrorHook,
    on_receive_error: OnReceiveErrorHook,
    on_source_eos: OnSourceEosHook,
}

impl ZmqSourceResults {
    /// Start a builder that installs defaults for omitted setters.
    pub fn builder() -> ZmqSourceResultsBuilder {
        ZmqSourceResultsBuilder::new()
    }
}

impl Default for ZmqSourceResults {
    fn default() -> Self {
        ZmqSourceResultsBuilder::new().build()
    }
}

/// Fluent builder for [`ZmqSourceResults`].
pub struct ZmqSourceResultsBuilder {
    on_message: Option<OnMessageHook>,
    on_service_message: Option<OnServiceMessageHook>,
    on_protocol_error: Option<OnProtocolErrorHook>,
    on_receive_error: Option<OnReceiveErrorHook>,
    on_source_eos: Option<OnSourceEosHook>,
}

impl ZmqSourceResultsBuilder {
    /// Create an empty results bundle builder.
    pub fn new() -> Self {
        Self {
            on_message: None,
            on_service_message: None,
            on_protocol_error: None,
            on_receive_error: None,
            on_source_eos: None,
        }
    }

    /// Override `on_message` (fired only for video-frame messages).
    ///
    /// The hook is the drop/send authority for the frame: call
    /// `router.send(EncodedMsg::Frame { .. })` to forward, or simply
    /// return without sending to drop.
    pub fn on_message<F>(mut self, f: F) -> Self
    where
        F: FnMut(&HookCtx, &Router<EncodedMsg>, String, Box<Message>, Vec<Vec<u8>>) -> Result<()>
            + Send
            + 'static,
    {
        self.on_message = Some(Box::new(f));
        self
    }

    /// Override `on_service_message` (fired only for non-frame, non-EOS
    /// kinds).
    pub fn on_service_message<F>(mut self, f: F) -> Self
    where
        F: FnMut(&HookCtx, &Router<EncodedMsg>, String, Box<Message>, Vec<Vec<u8>>) -> Result<()>
            + Send
            + 'static,
    {
        self.on_service_message = Some(Box::new(f));
        self
    }

    /// Override protocol-error classifier.
    pub fn on_protocol_error<F>(mut self, f: F) -> Self
    where
        F: FnMut(&HookCtx, &ReaderResult) -> ErrorAction + Send + 'static,
    {
        self.on_protocol_error = Some(Box::new(f));
        self
    }

    /// Override receive-error classifier.
    pub fn on_receive_error<F>(mut self, f: F) -> Self
    where
        F: FnMut(&HookCtx, &anyhow::Error) -> ErrorAction + Send + 'static,
    {
        self.on_receive_error = Some(Box::new(f));
        self
    }

    /// Override source-EOS hook.
    pub fn on_source_eos<F>(mut self, f: F) -> Self
    where
        F: FnMut(&HookCtx, &Router<EncodedMsg>, &str) -> Result<Flow> + Send + 'static,
    {
        self.on_source_eos = Some(Box::new(f));
        self
    }

    /// Finalise with default substitutions.
    pub fn build(self) -> ZmqSourceResults {
        let ZmqSourceResultsBuilder {
            on_message,
            on_service_message,
            on_protocol_error,
            on_receive_error,
            on_source_eos,
        } = self;
        ZmqSourceResults {
            on_message: on_message.unwrap_or_else(|| Box::new(ZmqSource::default_on_message())),
            on_service_message: on_service_message
                .unwrap_or_else(|| Box::new(ZmqSource::default_on_service_message())),
            on_protocol_error: on_protocol_error
                .unwrap_or_else(|| Box::new(ZmqSource::default_on_protocol_error())),
            on_receive_error: on_receive_error
                .unwrap_or_else(|| Box::new(ZmqSource::default_on_receive_error())),
            on_source_eos: on_source_eos
                .unwrap_or_else(|| Box::new(ZmqSource::default_on_source_eos())),
        }
    }
}

impl Default for ZmqSourceResultsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Common loop knobs for [`ZmqSource`].
pub struct ZmqSourceCommon {
    poll_timeout: Duration,
    on_stopping: OnStoppingHook,
}

impl ZmqSourceCommon {
    /// Start a builder seeded with defaults.
    pub fn builder() -> ZmqSourceCommonBuilder {
        ZmqSourceCommonBuilder::new()
    }
}

impl Default for ZmqSourceCommon {
    fn default() -> Self {
        ZmqSourceCommonBuilder::new().build()
    }
}

/// Fluent builder for [`ZmqSourceCommon`].
pub struct ZmqSourceCommonBuilder {
    poll_timeout: Option<Duration>,
    on_stopping: Option<OnStoppingHook>,
}

impl ZmqSourceCommonBuilder {
    /// Create an empty common-bundle builder.
    pub fn new() -> Self {
        Self {
            poll_timeout: None,
            on_stopping: None,
        }
    }

    /// Override receive poll timeout.
    pub fn poll_timeout(mut self, timeout: Duration) -> Self {
        self.poll_timeout = Some(timeout);
        self
    }

    /// Override user stopping hook.
    pub fn on_stopping<F>(mut self, f: F) -> Self
    where
        F: FnMut(&SourceContext) + Send + 'static,
    {
        self.on_stopping = Some(Box::new(f));
        self
    }

    /// Finalise with default substitutions.
    pub fn build(self) -> ZmqSourceCommon {
        let ZmqSourceCommonBuilder {
            poll_timeout,
            on_stopping,
        } = self;
        ZmqSourceCommon {
            poll_timeout: poll_timeout.unwrap_or(DEFAULT_POLL_TIMEOUT),
            on_stopping: on_stopping.unwrap_or_else(|| Box::new(ZmqSource::default_on_stopping())),
        }
    }
}

impl Default for ZmqSourceCommonBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Fluent builder for [`ZmqSource`].
pub struct ZmqSourceBuilder {
    name: StageName,
    config: Option<ReaderConfig>,
    results_queue_size: usize,
    downstream: Option<StageName>,
    results: Option<ZmqSourceResults>,
    common: Option<ZmqSourceCommon>,
}

impl ZmqSourceBuilder {
    /// Create a builder for stage `name`.
    pub fn new(name: StageName) -> Self {
        Self {
            name,
            config: None,
            results_queue_size: DEFAULT_RESULTS_QUEUE_SIZE,
            downstream: None,
            results: None,
            common: None,
        }
    }

    /// Set explicit reader config.
    pub fn config(mut self, cfg: ReaderConfig) -> Self {
        self.config = Some(cfg);
        self
    }

    /// Shortcut for default reader config from `url`.
    pub fn url(mut self, url: &str) -> Result<Self> {
        let cfg = ReaderConfig::new().url(url)?.build()?;
        self.config = Some(cfg);
        Ok(self)
    }

    /// Set maximum in-flight reader results.
    pub fn results_queue_size(mut self, n: usize) -> Self {
        self.results_queue_size = n;
        self
    }

    /// Set default downstream peer.
    pub fn downstream(mut self, peer: StageName) -> Self {
        self.downstream = Some(peer);
        self
    }

    /// Install result-hook bundle.
    pub fn results(mut self, r: ZmqSourceResults) -> Self {
        self.results = Some(r);
        self
    }

    /// Install common loop-hook bundle.
    pub fn common(mut self, c: ZmqSourceCommon) -> Self {
        self.common = Some(c);
        self
    }

    /// Finalise into [`SourceBuilder<ZmqSource>`].
    ///
    /// # Errors
    ///
    /// Returns `Err` when reader config is missing.
    pub fn build(self) -> Result<SourceBuilder<ZmqSource>> {
        let ZmqSourceBuilder {
            name,
            config,
            results_queue_size,
            downstream,
            results,
            common,
        } = self;

        let config = config.ok_or_else(|| anyhow!("ZmqSource: missing config"))?;
        let ZmqSourceResults {
            on_message,
            on_service_message,
            on_protocol_error,
            on_receive_error,
            on_source_eos,
        } = results.unwrap_or_default();
        let ZmqSourceCommon {
            poll_timeout,
            on_stopping,
        } = common.unwrap_or_default();

        Ok(SourceBuilder::new(name).factory(move |_bx| {
            Ok(ZmqSource {
                config,
                results_queue_size,
                downstream,
                on_message,
                on_service_message,
                on_protocol_error,
                on_receive_error,
                on_source_eos,
                poll_timeout,
                on_stopping,
            })
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::supervisor::{StageKind, StageName};

    #[test]
    fn builder_requires_config() {
        let name = StageName::unnamed(StageKind::BitstreamSource);
        let err = ZmqSource::builder(name)
            .build()
            .err()
            .expect("missing config");
        assert!(err.to_string().contains("missing config"));
    }

    #[test]
    fn builder_accepts_url_shortcut() {
        let name = StageName::unnamed(StageKind::BitstreamSource);
        let sb = ZmqSource::builder(name)
            .url("router+bind:ipc:///tmp/savant_zmq_source_test")
            .expect("url accepted")
            .build();
        assert!(sb.is_ok());
    }

    #[test]
    fn builder_accepts_all_hooks() {
        let name = StageName::unnamed(StageKind::BitstreamSource);
        let cfg = ReaderConfig::new()
            .url("router+bind:ipc:///tmp/savant_zmq_source_hooks")
            .unwrap()
            .build()
            .unwrap();
        let _ = ZmqSource::builder(name)
            .config(cfg)
            .results_queue_size(8)
            .downstream(StageName::unnamed(StageKind::Decoder))
            .results(
                ZmqSourceResults::builder()
                    .on_message(|_topic, _message, _data, _router, _ctx| Ok(()))
                    .on_service_message(|_topic, _message, _data, _router, _ctx| Ok(()))
                    .on_protocol_error(|_res, _ctx| ErrorAction::Swallow)
                    .on_receive_error(|_err, _ctx| ErrorAction::LogAndContinue)
                    .on_source_eos(|_sid, _router, _ctx| Ok(Flow::Cont))
                    .build(),
            )
            .common(
                ZmqSourceCommon::builder()
                    .poll_timeout(Duration::from_millis(25))
                    .on_stopping(ZmqSource::default_on_stopping())
                    .build(),
            )
            .build()
            .unwrap();
    }

    #[test]
    fn builder_accepts_default_forwarders() {
        let name = StageName::unnamed(StageKind::BitstreamSource);
        let cfg = ReaderConfig::new()
            .url("router+bind:ipc:///tmp/savant_zmq_source_defaults")
            .unwrap()
            .build()
            .unwrap();
        let _ = ZmqSource::builder(name)
            .config(cfg)
            .downstream(StageName::unnamed(StageKind::Decoder))
            .results(
                ZmqSourceResults::builder()
                    .on_message(ZmqSource::default_on_message())
                    .on_service_message(ZmqSource::default_on_service_message())
                    .on_protocol_error(ZmqSource::default_on_protocol_error())
                    .on_receive_error(ZmqSource::default_on_receive_error())
                    .on_source_eos(ZmqSource::default_on_source_eos())
                    .build(),
            )
            .common(
                ZmqSourceCommon::builder()
                    .poll_timeout(DEFAULT_POLL_TIMEOUT)
                    .on_stopping(ZmqSource::default_on_stopping())
                    .build(),
            )
            .build()
            .unwrap();
    }

    #[test]
    fn builder_without_downstream_is_accepted() {
        let name = StageName::unnamed(StageKind::BitstreamSource);
        let cfg = ReaderConfig::new()
            .url("router+bind:ipc:///tmp/savant_zmq_source_no_downstream")
            .unwrap()
            .build()
            .unwrap();
        let _ = ZmqSource::builder(name).config(cfg).build().unwrap();
    }

    #[test]
    fn runtime_invariant_hooks_non_optional() {
        use crate::context::BuildCtx;
        use crate::registry::Registry;
        use crate::shared::SharedStore;
        use std::sync::Arc;

        let name = StageName::unnamed(StageKind::BitstreamSource);
        let cfg = ReaderConfig::new()
            .url("router+bind:ipc:///tmp/savant_zmq_source_invariant")
            .unwrap()
            .build()
            .unwrap();
        let sb = ZmqSource::builder(name.clone())
            .config(cfg)
            .build()
            .unwrap();
        let parts = sb.into_parts();
        assert_eq!(parts.name, name);
        let reg = Arc::new(Registry::new());
        let shared = Arc::new(SharedStore::new());
        let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let pn: Arc<str> = Arc::from("test");
        let sm = crate::stage_metrics::StageMetrics::new(parts.name.to_string());
        let bx = BuildCtx::new(&parts.name, &pn, &reg, &shared, &stop_flag, &sm);
        let src = (parts.factory)(&bx).expect("factory resolves");
        let ZmqSource {
            config: _,
            results_queue_size: _,
            downstream: _,
            on_message: _,
            on_service_message: _,
            on_protocol_error: _,
            on_receive_error: _,
            on_source_eos: _,
            poll_timeout: _,
            on_stopping: _,
        } = src;
    }
}
