//! [`ZmqSink`] — `EncodedMsg` → ZeroMQ terminus actor.
//!
//! The stage wraps [`NonBlockingWriter`] and turns inbox
//! [`EncodedMsg`] values into wire [`Message`] values:
//!
//! * [`EncodedMsg::Frame`]   → [`Message::video_frame`] + optional
//!   multipart payload (configurable via [`PayloadCarrier`]).
//! * [`EncodedMsg::SourceEos`] → `writer.send_eos(source_id)`.
//! * [`EncodedMsg::StreamInfo`] / [`EncodedMsg::Packet`] are accepted
//!   and ignored at `debug` level for protocol forward-compatibility.
//!
//! ZMQ is treated as pure transport: the sink **never initiates
//! messages on its own state changes** (e.g. no synthetic EOS on
//! shutdown).  All wire traffic is driven by inbox events.
//!
//! Hook setters are grouped into three bundles following the
//! cross-stage pattern used by [`Mp4Muxer`](super::mp4_muxer):
//!
//! * [`ZmqSinkInbox`]  — inbox hooks (`on_source_eos`, `on_frame`).
//!   `on_frame` is a per-frame observer; the sink itself is a pure
//!   transport — every accepted inbox frame is forwarded on the
//!   wire.  Drop decisions belong upstream.
//! * [`ZmqSinkErrors`] — writer/send error classifiers
//! * [`ZmqSinkCommon`] — loop knobs + topic strategy + stopping
//!
//! Runtime invariant: all hook slots on [`ZmqSink`] are non-`Option`
//! boxed closures; omitted bundle setters are replaced by defaults at
//! build time.

use std::time::Duration;

use anyhow::{anyhow, bail, Result};
use savant_core::message::Message;
use savant_core::primitives::frame::VideoFrameContent;
use savant_core::transport::zeromq::{NonBlockingWriter, WriterConfig, WriterResult};

use crate::envelopes::{EncodedMsg, FramePayload, PacketPayload, StreamInfoPayload};
use crate::supervisor::StageName;
use crate::{
    Actor, ActorBuilder, Context, Dispatch, ErrorAction, Flow, Handler, ShutdownPayload,
    SourceEosPayload,
};

/// Default inbox receive-poll cadence.
pub const DEFAULT_POLL_TIMEOUT: Duration = Duration::from_millis(100);

/// Default maximum in-flight write operations for
/// [`NonBlockingWriter`].
pub const DEFAULT_INFLIGHT_MESSAGES: usize = 16;

/// Carrier policy for encoded frame payloads on the wire.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum PayloadCarrier {
    /// Put encoded bytes in an extra multipart segment and clear
    /// frame content before serializing the frame message.
    #[default]
    Multipart,
    /// Put encoded bytes into
    /// [`VideoFrameContent::Internal`] and send no extra multipart
    /// segments.
    Internal,
}

/// Source-EOS hook: return [`Flow::Stop`] to terminate, or
/// [`Flow::Cont`] to keep processing further sources.
pub type OnSourceEosHook = Box<dyn FnMut(&mut Context<ZmqSink>, &str) -> Result<Flow> + Send + 'static>;

/// Frame observer hook called before message adaptation/sending.
pub type OnFrameHook =
    Box<dyn FnMut(&mut Context<ZmqSink>, &FramePayload) -> Result<()> + Send + 'static>;

/// Writer-result classifier hook.
pub type OnWriterResultHook =
    Box<dyn FnMut(&mut Context<ZmqSink>, &WriterResult) -> ErrorAction + Send + 'static>;

/// Send-error classifier hook.
pub type OnSendErrorHook =
    Box<dyn FnMut(&mut Context<ZmqSink>, &anyhow::Error) -> ErrorAction + Send + 'static>;

/// User shutdown hook fired from [`Actor::stopping`] after built-in
/// writer cleanup completes.
pub type OnStoppingHook = Box<dyn FnMut(&mut Context<ZmqSink>) + Send + 'static>;

/// Topic strategy for per-frame wire publish topic.
pub type TopicStrategy = Box<dyn FnMut(&FramePayload) -> String + Send + 'static>;

/// `EncodedMsg` → ZeroMQ sink actor.
pub struct ZmqSink {
    config: WriterConfig,
    inflight: usize,
    payload_carrier: PayloadCarrier,
    writer: Option<NonBlockingWriter>,
    poll_timeout: Duration,
    on_source_eos: OnSourceEosHook,
    on_frame: OnFrameHook,
    on_writer_result: OnWriterResultHook,
    on_send_error: OnSendErrorHook,
    topic_strategy: TopicStrategy,
    on_stopping: OnStoppingHook,
    finalised: bool,
}

impl ZmqSink {
    /// Start a fluent builder for a sink registered under `name`
    /// with inbox capacity `capacity`.
    pub fn builder(name: StageName, capacity: usize) -> ZmqSinkBuilder {
        ZmqSinkBuilder::new(name, capacity)
    }

    /// Default `on_source_eos` hook — log and continue.
    ///
    /// ZMQ is always multi-stream by default: a single per-stream EOS
    /// is forwarded over the wire (by the runtime), but the actor
    /// keeps running.  Applications that want single-stream
    /// termination can return [`Flow::Stop`] from an override.
    pub fn default_on_source_eos(
    ) -> impl FnMut(&mut Context<ZmqSink>, &str) -> Result<Flow> + Send + 'static {
        |ctx, source_id| {
            log::info!(
                "[{}] SourceEos {source_id}: continuing (multi-stream)",
                ctx.own_name()
            );
            Ok(Flow::Cont)
        }
    }

    /// Default frame observer — no-op.
    pub fn default_on_frame(
    ) -> impl FnMut(&mut Context<ZmqSink>, &FramePayload) -> Result<()> + Send + 'static {
        |_ctx, _payload| Ok(())
    }

    /// Default writer-result classifier.
    ///
    /// `Ack` and `Success` are treated as non-errors (`Swallow`).
    /// Timeout variants are treated as recoverable
    /// (`LogAndContinue`).
    pub fn default_on_writer_result(
    ) -> impl FnMut(&mut Context<ZmqSink>, &WriterResult) -> ErrorAction + Send + 'static {
        |_ctx, result| match result {
            WriterResult::Ack { .. } | WriterResult::Success { .. } => ErrorAction::Swallow,
            WriterResult::SendTimeout | WriterResult::AckTimeout(_) => ErrorAction::LogAndContinue,
        }
    }

    /// Default send-error classifier — fatal.
    pub fn default_on_send_error(
    ) -> impl FnMut(&mut Context<ZmqSink>, &anyhow::Error) -> ErrorAction + Send + 'static {
        |_ctx, _error| ErrorAction::Fatal
    }

    /// Default stopping hook — no-op.
    pub fn default_on_stopping() -> impl FnMut(&mut Context<ZmqSink>) + Send + 'static {
        |_ctx| {}
    }

    /// Default per-frame topic strategy — source id.
    pub fn default_topic_strategy() -> impl FnMut(&FramePayload) -> String + Send + 'static {
        |payload| payload.frame.get_source_id()
    }

    fn classify_send_error(&mut self, err: anyhow::Error, ctx: &mut Context<Self>) -> Result<Flow> {
        match (self.on_send_error)(ctx, &err) {
            ErrorAction::Fatal => Err(anyhow!("fatal send error: {err}")),
            ErrorAction::LogAndContinue => {
                log::warn!("[{}] send error: {err}", ctx.own_name());
                Ok(Flow::Cont)
            }
            ErrorAction::Swallow => Ok(Flow::Cont),
        }
    }

    fn classify_writer_result(
        &mut self,
        wr: WriterResult,
        ctx: &mut Context<Self>,
    ) -> Result<Flow> {
        match (self.on_writer_result)(ctx, &wr) {
            ErrorAction::Fatal => Err(anyhow!("fatal writer result: {wr:?}")),
            ErrorAction::LogAndContinue => {
                log::warn!(
                    "[{}] writer result classified as recoverable: {wr:?}",
                    ctx.own_name()
                );
                Ok(Flow::Cont)
            }
            ErrorAction::Swallow => Ok(Flow::Cont),
        }
    }
}

impl Actor for ZmqSink {
    type Msg = EncodedMsg;

    fn handle(&mut self, msg: EncodedMsg, ctx: &mut Context<Self>) -> Result<Flow> {
        msg.dispatch(self, ctx)
    }

    fn poll_timeout(&self) -> Duration {
        self.poll_timeout
    }

    fn started(&mut self, ctx: &mut Context<Self>) -> Result<()> {
        log::info!(
            "[{}] starting endpoint={} inflight={} carrier={:?}",
            ctx.own_name(),
            self.config.endpoint(),
            self.inflight,
            self.payload_carrier
        );
        let mut writer = NonBlockingWriter::new(&self.config, self.inflight)?;
        writer.start()?;
        self.writer = Some(writer);
        Ok(())
    }

    fn stopping(&mut self, ctx: &mut Context<Self>) {
        // ZMQ is pure transport: do NOT initiate any wire traffic
        // on shutdown.  Just close the writer and run the user
        // stopping hook.
        if !self.finalised {
            if let Some(mut writer) = self.writer.take() {
                if let Err(e) = writer.shutdown() {
                    log::warn!("[{}] writer.shutdown failed: {e}", ctx.own_name());
                }
            }
            self.finalised = true;
        }
        (self.on_stopping)(ctx);
    }
}

impl Handler<FramePayload> for ZmqSink {
    fn handle(&mut self, msg: FramePayload, ctx: &mut Context<Self>) -> Result<Flow> {
        (self.on_frame)(ctx, &msg)?;

        let topic = (self.topic_strategy)(&msg);

        let payload_bytes = msg.payload.clone();
        let frame_internal_bytes = match msg.frame.get_content().as_ref() {
            VideoFrameContent::Internal(bytes) => Some(bytes.clone()),
            _ => None,
        };
        let Some(bitstream) = payload_bytes.or(frame_internal_bytes) else {
            log::warn!(
                "[{}] dropping frame source_id={} because no bitstream is available",
                ctx.own_name(),
                msg.frame.get_source_id()
            );
            return Ok(Flow::Cont);
        };

        let mut frame_for_message = msg.frame.clone();
        let mut extra_parts: Vec<Vec<u8>> = Vec::new();

        match self.payload_carrier {
            PayloadCarrier::Multipart => {
                frame_for_message.set_content(VideoFrameContent::None);
                extra_parts.push(bitstream);
            }
            PayloadCarrier::Internal => {
                frame_for_message.set_content(VideoFrameContent::Internal(bitstream));
            }
        }

        let message = Message::video_frame(&frame_for_message);

        let Some(writer) = self.writer.as_ref() else {
            bail!(
                "[{}] frame received before writer was started",
                ctx.own_name()
            );
        };

        let extra_part_refs = extra_parts.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let op = match writer.send_message(&topic, &message, &extra_part_refs) {
            Ok(op) => op,
            Err(e) => return self.classify_send_error(e, ctx),
        };

        let wr = match op.get() {
            Ok(wr) => wr,
            Err(e) => return self.classify_send_error(e, ctx),
        };

        self.classify_writer_result(wr, ctx)
    }
}

impl Handler<SourceEosPayload> for ZmqSink {
    fn handle(&mut self, msg: SourceEosPayload, ctx: &mut Context<Self>) -> Result<Flow> {
        let flow = (self.on_source_eos)(ctx, &msg.source_id)?;
        if let Some(writer) = self.writer.as_ref() {
            let op = match writer.send_eos(&msg.source_id) {
                Ok(op) => op,
                Err(e) => return self.classify_send_error(e, ctx),
            };
            let wr = match op.get() {
                Ok(wr) => wr,
                Err(e) => return self.classify_send_error(e, ctx),
            };
            self.classify_writer_result(wr, ctx)?;
        }
        Ok(flow)
    }
}

/// Non-fatal: stream-info sentinel is not part of the wire publish
/// path for this sink.
impl Handler<StreamInfoPayload> for ZmqSink {
    fn handle(&mut self, msg: StreamInfoPayload, ctx: &mut Context<Self>) -> Result<Flow> {
        log::debug!(
            "[{}] ignoring StreamInfo source_id={}",
            ctx.own_name(),
            msg.source_id
        );
        Ok(Flow::Cont)
    }
}

/// Non-fatal: packet sentinel is not part of the wire publish path
/// for this sink.
impl Handler<PacketPayload> for ZmqSink {
    fn handle(&mut self, msg: PacketPayload, ctx: &mut Context<Self>) -> Result<Flow> {
        log::debug!(
            "[{}] ignoring Packet source_id={}",
            ctx.own_name(),
            msg.source_id
        );
        Ok(Flow::Cont)
    }
}

/// Default no-op — the loop driver consumes the shutdown hint via
/// `Envelope::as_shutdown`.
impl Handler<ShutdownPayload> for ZmqSink {}

/// Inbox hook bundle for [`ZmqSink`].
pub struct ZmqSinkInbox {
    on_source_eos: OnSourceEosHook,
    on_frame: OnFrameHook,
}

impl ZmqSinkInbox {
    /// Start a builder that installs defaults for omitted setters.
    pub fn builder() -> ZmqSinkInboxBuilder {
        ZmqSinkInboxBuilder::new()
    }
}

impl Default for ZmqSinkInbox {
    fn default() -> Self {
        ZmqSinkInboxBuilder::new().build()
    }
}

/// Fluent builder for [`ZmqSinkInbox`].
pub struct ZmqSinkInboxBuilder {
    on_source_eos: Option<OnSourceEosHook>,
    on_frame: Option<OnFrameHook>,
}

impl ZmqSinkInboxBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self {
            on_source_eos: None,
            on_frame: None,
        }
    }

    /// Install a custom source-EOS hook.
    pub fn on_source_eos<F>(mut self, f: F) -> Self
    where
        F: FnMut(&mut Context<ZmqSink>, &str) -> Result<Flow> + Send + 'static,
    {
        self.on_source_eos = Some(Box::new(f));
        self
    }

    /// Install a custom frame observer hook.
    pub fn on_frame<F>(mut self, f: F) -> Self
    where
        F: FnMut(&mut Context<ZmqSink>, &FramePayload) -> Result<()> + Send + 'static,
    {
        self.on_frame = Some(Box::new(f));
        self
    }

    /// Finalise the bundle with default substitutions.
    pub fn build(self) -> ZmqSinkInbox {
        let ZmqSinkInboxBuilder {
            on_source_eos,
            on_frame,
        } = self;
        ZmqSinkInbox {
            on_source_eos: on_source_eos
                .unwrap_or_else(|| Box::new(ZmqSink::default_on_source_eos())),
            on_frame: on_frame.unwrap_or_else(|| Box::new(ZmqSink::default_on_frame())),
        }
    }
}

impl Default for ZmqSinkInboxBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Error hook bundle for [`ZmqSink`].
pub struct ZmqSinkErrors {
    on_writer_result: OnWriterResultHook,
    on_send_error: OnSendErrorHook,
}

impl ZmqSinkErrors {
    /// Start a builder that installs defaults for omitted setters.
    pub fn builder() -> ZmqSinkErrorsBuilder {
        ZmqSinkErrorsBuilder::new()
    }
}

impl Default for ZmqSinkErrors {
    fn default() -> Self {
        ZmqSinkErrorsBuilder::new().build()
    }
}

/// Fluent builder for [`ZmqSinkErrors`].
pub struct ZmqSinkErrorsBuilder {
    on_writer_result: Option<OnWriterResultHook>,
    on_send_error: Option<OnSendErrorHook>,
}

impl ZmqSinkErrorsBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self {
            on_writer_result: None,
            on_send_error: None,
        }
    }

    /// Install a custom writer-result classifier.
    pub fn on_writer_result<F>(mut self, f: F) -> Self
    where
        F: FnMut(&mut Context<ZmqSink>, &WriterResult) -> ErrorAction + Send + 'static,
    {
        self.on_writer_result = Some(Box::new(f));
        self
    }

    /// Install a custom send-error classifier.
    pub fn on_send_error<F>(mut self, f: F) -> Self
    where
        F: FnMut(&mut Context<ZmqSink>, &anyhow::Error) -> ErrorAction + Send + 'static,
    {
        self.on_send_error = Some(Box::new(f));
        self
    }

    /// Finalise the bundle with default substitutions.
    pub fn build(self) -> ZmqSinkErrors {
        let ZmqSinkErrorsBuilder {
            on_writer_result,
            on_send_error,
        } = self;
        ZmqSinkErrors {
            on_writer_result: on_writer_result
                .unwrap_or_else(|| Box::new(ZmqSink::default_on_writer_result())),
            on_send_error: on_send_error
                .unwrap_or_else(|| Box::new(ZmqSink::default_on_send_error())),
        }
    }
}

impl Default for ZmqSinkErrorsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Loop-level common knobs and shaping hooks for [`ZmqSink`].
pub struct ZmqSinkCommon {
    poll_timeout: Duration,
    on_stopping: OnStoppingHook,
    topic_strategy: TopicStrategy,
}

impl ZmqSinkCommon {
    /// Start a builder seeded with defaults.
    pub fn builder() -> ZmqSinkCommonBuilder {
        ZmqSinkCommonBuilder::new()
    }
}

impl Default for ZmqSinkCommon {
    fn default() -> Self {
        ZmqSinkCommonBuilder::new().build()
    }
}

/// Fluent builder for [`ZmqSinkCommon`].
pub struct ZmqSinkCommonBuilder {
    poll_timeout: Option<Duration>,
    on_stopping: Option<OnStoppingHook>,
    topic_strategy: Option<TopicStrategy>,
}

impl ZmqSinkCommonBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self {
            poll_timeout: None,
            on_stopping: None,
            topic_strategy: None,
        }
    }

    /// Override inbox receive poll timeout.
    pub fn poll_timeout(mut self, d: Duration) -> Self {
        self.poll_timeout = Some(d);
        self
    }

    /// Override stopping hook.
    pub fn on_stopping<F>(mut self, f: F) -> Self
    where
        F: FnMut(&mut Context<ZmqSink>) + Send + 'static,
    {
        self.on_stopping = Some(Box::new(f));
        self
    }

    /// Override topic strategy.
    pub fn topic_strategy<F>(mut self, f: F) -> Self
    where
        F: FnMut(&FramePayload) -> String + Send + 'static,
    {
        self.topic_strategy = Some(Box::new(f));
        self
    }

    /// Finalise the bundle with default substitutions.
    pub fn build(self) -> ZmqSinkCommon {
        let ZmqSinkCommonBuilder {
            poll_timeout,
            on_stopping,
            topic_strategy,
        } = self;
        ZmqSinkCommon {
            poll_timeout: poll_timeout.unwrap_or(DEFAULT_POLL_TIMEOUT),
            on_stopping: on_stopping.unwrap_or_else(|| Box::new(ZmqSink::default_on_stopping())),
            topic_strategy: topic_strategy
                .unwrap_or_else(|| Box::new(ZmqSink::default_topic_strategy())),
        }
    }
}

impl Default for ZmqSinkCommonBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Fluent builder for [`ZmqSink`].
pub struct ZmqSinkBuilder {
    name: StageName,
    capacity: usize,
    config: Option<WriterConfig>,
    inflight: usize,
    payload_carrier: PayloadCarrier,
    inbox: Option<ZmqSinkInbox>,
    errors: Option<ZmqSinkErrors>,
    common: Option<ZmqSinkCommon>,
}

impl ZmqSinkBuilder {
    /// Create a builder for stage `name` with inbox capacity
    /// `capacity`.
    pub fn new(name: StageName, capacity: usize) -> Self {
        Self {
            name,
            capacity,
            config: None,
            inflight: DEFAULT_INFLIGHT_MESSAGES,
            payload_carrier: PayloadCarrier::default(),
            inbox: None,
            errors: None,
            common: None,
        }
    }

    /// Set explicit writer config (required unless [`Self::url`] is
    /// used).
    pub fn config(mut self, cfg: WriterConfig) -> Self {
        self.config = Some(cfg);
        self
    }

    /// Shortcut for default writer config from `url`.
    pub fn url(mut self, url: &str) -> Result<Self> {
        let cfg = WriterConfig::new().url(url)?.build()?;
        self.config = Some(cfg);
        Ok(self)
    }

    /// Set writer max in-flight operations.
    pub fn inflight(mut self, n: usize) -> Self {
        self.inflight = n;
        self
    }

    /// Select payload carrier mode.
    pub fn payload_carrier(mut self, c: PayloadCarrier) -> Self {
        self.payload_carrier = c;
        self
    }

    /// Install inbox hook bundle.
    pub fn inbox(mut self, i: ZmqSinkInbox) -> Self {
        self.inbox = Some(i);
        self
    }

    /// Install error hook bundle.
    pub fn errors(mut self, e: ZmqSinkErrors) -> Self {
        self.errors = Some(e);
        self
    }

    /// Install common loop/hook bundle.
    pub fn common(mut self, c: ZmqSinkCommon) -> Self {
        self.common = Some(c);
        self
    }

    /// Finalise into [`ActorBuilder<ZmqSink>`].
    ///
    /// # Errors
    ///
    /// Returns `Err` when writer config is missing.
    pub fn build(self) -> Result<ActorBuilder<ZmqSink>> {
        let ZmqSinkBuilder {
            name,
            capacity,
            config,
            inflight,
            payload_carrier,
            inbox,
            errors,
            common,
        } = self;

        let config = config.ok_or_else(|| anyhow!("ZmqSink: missing config"))?;
        let ZmqSinkInbox {
            on_source_eos,
            on_frame,
        } = inbox.unwrap_or_default();
        let ZmqSinkErrors {
            on_writer_result,
            on_send_error,
        } = errors.unwrap_or_default();
        let ZmqSinkCommon {
            poll_timeout,
            on_stopping,
            topic_strategy,
        } = common.unwrap_or_default();

        Ok(ActorBuilder::new(name, capacity)
            .poll_timeout(poll_timeout)
            .factory(move |_bx| {
                Ok(ZmqSink {
                    config,
                    inflight,
                    payload_carrier,
                    writer: None,
                    poll_timeout,
                    on_source_eos,
                    on_frame,
                    on_writer_result,
                    on_send_error,
                    topic_strategy,
                    on_stopping,
                    finalised: false,
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
        let name = StageName::unnamed(StageKind::BitstreamSink);
        let err = ZmqSink::builder(name, 4)
            .build()
            .err()
            .expect("missing config");
        assert!(err.to_string().contains("missing config"));
    }

    #[test]
    fn builder_accepts_url_shortcut() {
        let name = StageName::unnamed(StageKind::BitstreamSink);
        let ab = ZmqSink::builder(name, 4)
            .url("dealer+connect:tcp://127.0.0.1:65199")
            .expect("url config")
            .build();
        assert!(ab.is_ok());
    }

    #[test]
    fn builder_accepts_all_hooks() {
        let name = StageName::unnamed(StageKind::BitstreamSink);
        let cfg = WriterConfig::new()
            .url("dealer+connect:tcp://127.0.0.1:65199")
            .unwrap()
            .build()
            .unwrap();
        let _ = ZmqSink::builder(name, 4)
            .config(cfg)
            .inflight(8)
            .payload_carrier(PayloadCarrier::Internal)
            .inbox(
                ZmqSinkInbox::builder()
                    .on_source_eos(|_ctx, _sid| Ok(Flow::Cont))
                    .on_frame(|_ctx, _payload| Ok(()))
                    .build(),
            )
            .errors(
                ZmqSinkErrors::builder()
                    .on_writer_result(|_ctx, _wr| ErrorAction::LogAndContinue)
                    .on_send_error(|_ctx, _err| ErrorAction::Swallow)
                    .build(),
            )
            .common(
                ZmqSinkCommon::builder()
                    .poll_timeout(Duration::from_millis(25))
                    .on_stopping(ZmqSink::default_on_stopping())
                    .topic_strategy(ZmqSink::default_topic_strategy())
                    .build(),
            )
            .build()
            .unwrap();
    }

    #[test]
    fn builder_accepts_default_hooks() {
        let name = StageName::unnamed(StageKind::BitstreamSink);
        let cfg = WriterConfig::new()
            .url("dealer+connect:tcp://127.0.0.1:65199")
            .unwrap()
            .build()
            .unwrap();
        let _ = ZmqSink::builder(name, 4)
            .config(cfg)
            .inbox(
                ZmqSinkInbox::builder()
                    .on_source_eos(ZmqSink::default_on_source_eos())
                    .on_frame(ZmqSink::default_on_frame())
                    .build(),
            )
            .errors(
                ZmqSinkErrors::builder()
                    .on_writer_result(ZmqSink::default_on_writer_result())
                    .on_send_error(ZmqSink::default_on_send_error())
                    .build(),
            )
            .common(
                ZmqSinkCommon::builder()
                    .poll_timeout(DEFAULT_POLL_TIMEOUT)
                    .on_stopping(ZmqSink::default_on_stopping())
                    .topic_strategy(ZmqSink::default_topic_strategy())
                    .build(),
            )
            .build()
            .unwrap();
    }

    #[test]
    fn runtime_invariant_hooks_non_optional() {
        use crate::context::BuildCtx;
        use crate::registry::Registry;
        use crate::shared::SharedStore;
        use std::sync::Arc;

        let name = StageName::unnamed(StageKind::BitstreamSink);
        let cfg = WriterConfig::new()
            .url("dealer+connect:tcp://127.0.0.1:65199")
            .unwrap()
            .build()
            .unwrap();
        let ab = ZmqSink::builder(name.clone(), 4)
            .config(cfg)
            .build()
            .unwrap();
        let parts = ab.into_parts();
        let reg = Arc::new(Registry::new());
        let shared = Arc::new(SharedStore::new());
        let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let bx = BuildCtx::new(&parts.name, &reg, &shared, &stop_flag);
        let sink = (parts.factory)(&bx).expect("factory resolves");
        let ZmqSink {
            on_source_eos: _,
            on_frame: _,
            on_writer_result: _,
            on_send_error: _,
            topic_strategy: _,
            on_stopping: _,
            ..
        } = sink;
    }
}
