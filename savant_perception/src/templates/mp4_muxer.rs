//! [`Mp4Muxer`] — `EncodedMsg` → MP4 file terminus.
//!
//! A direct replacement for the handwritten mux thread in
//! `cars_tracking/pipeline/mp4_mux.rs`.  The template:
//!
//! * Constructs [`Mp4Muxer`] on [`Actor::started`] (so the GStreamer
//!   pipeline initialisation runs on the worker thread).
//! * Maps [`EncodedMsg::Frame`] to [`Mp4Muxer::push`], reading
//!   `pts` / `dts` / `duration` off the pre-built frame and
//!   following the standard "payload `Some(bytes)` first, else
//!   [`VideoFrameContent::Internal`] fallback" contract.
//! * Stops the receive loop on [`EncodedMsg::SourceEos`] (default:
//!   first-EOS terminus semantics; configurable via the inbox
//!   bundle for multi-source muxers).
//! * Finalises the `moov` atom on [`Actor::stopping`] via
//!   [`Mp4Muxer::finish`].
//! * Ignores [`EncodedMsg::StreamInfo`] / [`EncodedMsg::Packet`]
//!   at the `debug` log level — those variants are not part of
//!   the picasso → mux contract but left non-fatal for
//!   forward-compatibility.
//!
//! # Grouped builder API
//!
//! Hooks are grouped into two bundles following the cross-template
//! pattern:
//!
//! * [`Mp4MuxerInbox`] — inbox hooks (`on_frame`,
//!   `on_source_eos`).
//! * [`Mp4MuxerCommon`] — loop-level knobs (`poll_timeout`) plus
//!   the user `stopping` hook.
//!
//! ```ignore
//! Mp4Muxer::builder(name, 16)
//!     .output("/tmp/out.mp4")
//!     .framerate(30, 1)
//!     .inbox(
//!         Mp4MuxerInbox::builder()
//!             .on_frame(|payload, ctx| { /* ... */ Ok(()) })
//!             .build(),
//!     )
//!     .common(Mp4MuxerCommon::builder().build())
//!     .build()?;
//! ```
//!
//! The in-band [`EncodedMsg::Shutdown`] sentinel is handled by the
//! loop driver via [`Envelope::as_shutdown`](super::super::Envelope::as_shutdown)
//! — the template does not add its own cooperative-exit code.
//!
//! # Runtime invariant
//!
//! Every hook slot on the runtime [`Mp4Muxer`] struct is a
//! non-`Option` boxed closure — the bundle builders always
//! substitute the matching default when the user omits a setter.
//! In particular `on_frame` (previously `Option<FrameObserver>`)
//! is now always installed with a no-op default, eliminating the
//! `Option<_>` wrapper at runtime.

use std::time::Duration;

use anyhow::{anyhow, Result};
use savant_core::primitives::frame::VideoFrameContent;
use savant_core::primitives::video_codec::VideoCodec;
use savant_gstreamer::mp4_muxer::Mp4Muxer as GstMp4Muxer;

use crate::envelopes::{EncodedMsg, FramePayload, PacketPayload, StreamInfoPayload};
use crate::supervisor::StageName;
use crate::{
    Actor, ActorBuilder, Context, Dispatch, Flow, Handler, ShutdownPayload, SourceEosPayload,
};

/// Default inbox receive-poll cadence.
pub const DEFAULT_POLL_TIMEOUT: Duration = Duration::from_millis(100);

/// Closure type for `on_source_eos`: decide whether an EOS should
/// terminate the muxer.  Default: [`Flow::Stop`] on the first EOS.
pub type EosHook = Box<dyn FnMut(&str, &mut Context<Mp4Muxer>) -> Result<Flow> + Send + 'static>;

/// Closure type for `on_frame` observer — runs before the frame is
/// pushed to the muxer.  Returning `Err` aborts the loop.  Always
/// installed at runtime; the default is a no-op.
pub type FrameObserver =
    Box<dyn FnMut(&FramePayload, &mut Context<Mp4Muxer>) -> Result<()> + Send + 'static>;

/// User shutdown hook invoked from [`Actor::stopping`] *after* the
/// template's built-in cleanup (guarded [`GstMp4Muxer::finish`] on
/// the owned muxer) has completed.
///
/// Runs on the actor thread with full access to the [`Context`].
/// Ideal for final metrics flushes, bespoke log lines, or custom
/// bookkeeping that must observe the finalised MP4 file.  The
/// built-in finish is **load-bearing** — without it the MP4 moov
/// atom is never written.  Users who need to replace the finish
/// step entirely should implement [`Actor`] directly on a bespoke
/// struct.
pub type OnStoppingHook = Box<dyn FnMut(&mut Context<Mp4Muxer>) + Send + 'static>;

/// `EncodedMsg` → MP4 terminus actor.
///
/// Construct via [`Mp4Muxer::builder`].  Hook slots are non-`Option`
/// by construction — see the runtime invariant in the module docs.
pub struct Mp4Muxer {
    output: String,
    codec: VideoCodec,
    fps_num: i32,
    fps_den: i32,
    muxer: Option<GstMp4Muxer>,
    poll_timeout: Duration,
    on_source_eos: EosHook,
    on_frame: FrameObserver,
    stopping: OnStoppingHook,
    finalised: bool,
}

impl Mp4Muxer {
    /// Start a fluent builder for a muxer registered under `name`
    /// with inbox capacity `capacity`.
    pub fn builder(name: StageName, capacity: usize) -> Mp4MuxerBuilder {
        Mp4MuxerBuilder::new(name, capacity)
    }

    /// Default `on_source_eos` hook — logs at `info` level and
    /// returns `Ok(Flow::Stop)`, finalising the file on the first
    /// EOS.  Matches legacy single-source terminus behaviour.
    pub fn default_on_source_eos(
    ) -> impl FnMut(&str, &mut Context<Mp4Muxer>) -> Result<Flow> + Send + 'static {
        |source_id, ctx| {
            log::info!("[{}] SourceEos {source_id}: finalising", ctx.own_name());
            Ok(Flow::Stop)
        }
    }

    /// Default `on_frame` observer — a no-op.  Installed
    /// automatically when the user omits the bundle setter so the
    /// runtime struct never needs to inspect an `Option`.
    pub fn default_on_frame(
    ) -> impl FnMut(&FramePayload, &mut Context<Mp4Muxer>) -> Result<()> + Send + 'static {
        |_payload, _ctx| Ok(())
    }

    /// Default user shutdown hook — a no-op.  The template's own
    /// [`Actor::stopping`] body always runs the guarded
    /// [`GstMp4Muxer::finish`] before this hook fires, so omitting
    /// the bundle setter simply means "don't add any extra cleanup
    /// on top of the built-in finalisation".
    pub fn default_stopping() -> impl FnMut(&mut Context<Mp4Muxer>) + Send + 'static {
        |_ctx| {}
    }
}

impl Actor for Mp4Muxer {
    type Msg = EncodedMsg;

    fn handle(&mut self, msg: EncodedMsg, ctx: &mut Context<Self>) -> Result<Flow> {
        msg.dispatch(self, ctx)
    }

    fn poll_timeout(&self) -> Duration {
        self.poll_timeout
    }

    fn started(&mut self, ctx: &mut Context<Self>) -> Result<()> {
        log::info!(
            "[{}] starting output={} fps={}/{}",
            ctx.own_name(),
            self.output,
            self.fps_num,
            self.fps_den
        );
        let muxer = GstMp4Muxer::new(self.codec, &self.output, self.fps_num, self.fps_den)
            .map_err(|e| anyhow!("Mp4Muxer::new: {e}"))?;
        self.muxer = Some(muxer);
        Ok(())
    }

    fn stopping(&mut self, ctx: &mut Context<Self>) {
        if !self.finalised {
            if let Some(mut muxer) = self.muxer.take() {
                if let Err(e) = muxer.finish() {
                    log::warn!("[{}] Mp4Muxer::finish failed: {e}", ctx.own_name());
                } else {
                    log::info!("[{}] finished: {}", ctx.own_name(), self.output);
                }
            }
            self.finalised = true;
        }
        (self.stopping)(ctx);
    }
}

impl Handler<FramePayload> for Mp4Muxer {
    fn handle(&mut self, msg: FramePayload, ctx: &mut Context<Self>) -> Result<Flow> {
        (self.on_frame)(&msg, ctx)?;
        let FramePayload { frame, payload } = msg;
        let Some(muxer) = self.muxer.as_mut() else {
            return Err(anyhow!(
                "[{}] frame received before muxer was built",
                ctx.own_name()
            ));
        };
        let pts_ns = frame.get_pts().max(0) as u64;
        let dts_ns = frame.get_dts().map(|v| v.max(0) as u64);
        let duration_ns = frame.get_duration().map(|v| v.max(0) as u64);
        let content = frame.get_content();
        let data: &[u8] = match payload.as_deref() {
            Some(bytes) => bytes,
            None => match content.as_ref() {
                VideoFrameContent::Internal(bytes) => bytes.as_slice(),
                other => {
                    log::error!(
                        "[{}] frame source_id={} has no payload and non-Internal content: {:?}",
                        ctx.own_name(),
                        frame.get_source_id(),
                        std::mem::discriminant(other)
                    );
                    return Ok(Flow::Cont);
                }
            },
        };
        muxer
            .push(data, pts_ns, dts_ns, duration_ns)
            .map_err(|e| anyhow!("mux push: {e}"))?;
        Ok(Flow::Cont)
    }
}

impl Handler<SourceEosPayload> for Mp4Muxer {
    fn handle(&mut self, msg: SourceEosPayload, ctx: &mut Context<Self>) -> Result<Flow> {
        (self.on_source_eos)(&msg.source_id, ctx)
    }
}

/// Non-fatal: stream-info sentinels are not part of the picasso →
/// mux contract, but we accept and log them to keep the protocol
/// forward-compatible.
impl Handler<StreamInfoPayload> for Mp4Muxer {
    fn handle(&mut self, msg: StreamInfoPayload, ctx: &mut Context<Self>) -> Result<Flow> {
        log::debug!(
            "[{}] ignoring StreamInfo source_id={}",
            ctx.own_name(),
            msg.source_id
        );
        Ok(Flow::Cont)
    }
}

/// Non-fatal: pre-decode packet sentinels are not part of the picasso → mux contract.
impl Handler<PacketPayload> for Mp4Muxer {
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
/// [`Envelope::as_shutdown`](super::super::Envelope::as_shutdown).
impl Handler<ShutdownPayload> for Mp4Muxer {}

/// Inbox hook bundle for [`Mp4Muxer`] — one branch per inbox
/// message kind the template interprets beyond the load-bearing
/// push path.  Built through [`Mp4MuxerInbox::builder`] and handed
/// to [`Mp4MuxerBuilder::inbox`].  Omitted branches auto-install
/// the matching `Mp4Muxer::default_on_*` at build time.
pub struct Mp4MuxerInbox {
    on_source_eos: EosHook,
    on_frame: FrameObserver,
}

impl Mp4MuxerInbox {
    /// Start a builder that auto-installs every default on
    /// [`Mp4MuxerInboxBuilder::build`].
    pub fn builder() -> Mp4MuxerInboxBuilder {
        Mp4MuxerInboxBuilder::new()
    }
}

impl Default for Mp4MuxerInbox {
    fn default() -> Self {
        Mp4MuxerInboxBuilder::new().build()
    }
}

/// Fluent builder for [`Mp4MuxerInbox`].
pub struct Mp4MuxerInboxBuilder {
    on_source_eos: Option<EosHook>,
    on_frame: Option<FrameObserver>,
}

impl Mp4MuxerInboxBuilder {
    /// Empty bundle — every hook defaults to its matching
    /// `Mp4Muxer::default_on_*` at [`build`](Self::build) time.
    pub fn new() -> Self {
        Self {
            on_source_eos: None,
            on_frame: None,
        }
    }

    /// Install a custom source-EOS hook.  Return [`Flow::Stop`] to
    /// finalise the file on that EOS (default: first EOS stops);
    /// [`Flow::Cont`] to keep accepting frames from other sources
    /// (multi-source mux).
    pub fn on_source_eos<F>(mut self, f: F) -> Self
    where
        F: FnMut(&str, &mut Context<Mp4Muxer>) -> Result<Flow> + Send + 'static,
    {
        self.on_source_eos = Some(Box::new(f));
        self
    }

    /// Install a frame observer called once per incoming frame
    /// *before* it is pushed to the muxer.  Useful for stats or
    /// tee patterns.  Returning `Err` aborts the loop.
    pub fn on_frame<F>(mut self, f: F) -> Self
    where
        F: FnMut(&FramePayload, &mut Context<Mp4Muxer>) -> Result<()> + Send + 'static,
    {
        self.on_frame = Some(Box::new(f));
        self
    }

    /// Finalise the bundle, substituting defaults for every omitted
    /// setter.
    pub fn build(self) -> Mp4MuxerInbox {
        let Mp4MuxerInboxBuilder {
            on_source_eos,
            on_frame,
        } = self;
        Mp4MuxerInbox {
            on_source_eos: on_source_eos
                .unwrap_or_else(|| Box::new(Mp4Muxer::default_on_source_eos())),
            on_frame: on_frame.unwrap_or_else(|| Box::new(Mp4Muxer::default_on_frame())),
        }
    }
}

impl Default for Mp4MuxerInboxBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Loop-level common knobs + user shutdown hook for [`Mp4Muxer`].
/// Built through [`Mp4MuxerCommon::builder`] and handed to
/// [`Mp4MuxerBuilder::common`].
pub struct Mp4MuxerCommon {
    poll_timeout: Duration,
    stopping: OnStoppingHook,
}

impl Mp4MuxerCommon {
    /// Start a builder seeded with [`DEFAULT_POLL_TIMEOUT`] and the
    /// no-op stopping hook.
    pub fn builder() -> Mp4MuxerCommonBuilder {
        Mp4MuxerCommonBuilder::new()
    }
}

impl Default for Mp4MuxerCommon {
    fn default() -> Self {
        Mp4MuxerCommonBuilder::new().build()
    }
}

/// Fluent builder for [`Mp4MuxerCommon`].
pub struct Mp4MuxerCommonBuilder {
    poll_timeout: Option<Duration>,
    stopping: Option<OnStoppingHook>,
}

impl Mp4MuxerCommonBuilder {
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
    /// [`Actor::stopping`] **after** the template's built-in
    /// cleanup (guarded [`GstMp4Muxer::finish`]) has completed.
    /// The built-in `finish` step is **load-bearing** (without it
    /// the MP4 moov atom is never written) and always runs first;
    /// it cannot be skipped through this hook.
    pub fn stopping<F>(mut self, f: F) -> Self
    where
        F: FnMut(&mut Context<Mp4Muxer>) + Send + 'static,
    {
        self.stopping = Some(Box::new(f));
        self
    }

    /// Finalise the bundle, substituting defaults for every omitted
    /// setter.
    pub fn build(self) -> Mp4MuxerCommon {
        let Mp4MuxerCommonBuilder {
            poll_timeout,
            stopping,
        } = self;
        Mp4MuxerCommon {
            poll_timeout: poll_timeout.unwrap_or(DEFAULT_POLL_TIMEOUT),
            stopping: stopping.unwrap_or_else(|| Box::new(Mp4Muxer::default_stopping())),
        }
    }
}

impl Default for Mp4MuxerCommonBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Fluent builder for [`Mp4Muxer`].
///
/// The builder only exposes wiring-level configuration at the top
/// level (`output`, `codec`, `framerate`).  Inbox hooks live on
/// [`Mp4MuxerInbox`]; loop-level knobs and the user shutdown hook
/// live on [`Mp4MuxerCommon`].  Install them via
/// [`Mp4MuxerBuilder::inbox`] and [`Mp4MuxerBuilder::common`].
pub struct Mp4MuxerBuilder {
    name: StageName,
    capacity: usize,
    output: Option<String>,
    codec: VideoCodec,
    fps_num: i32,
    fps_den: i32,
    inbox: Option<Mp4MuxerInbox>,
    common: Option<Mp4MuxerCommon>,
}

impl Mp4MuxerBuilder {
    /// Start a builder with H.264 + 30 fps defaults.
    pub fn new(name: StageName, capacity: usize) -> Self {
        Self {
            name,
            capacity,
            output: None,
            codec: VideoCodec::H264,
            fps_num: 30,
            fps_den: 1,
            inbox: None,
            common: None,
        }
    }

    /// Required: output file path.
    pub fn output(mut self, path: impl Into<String>) -> Self {
        self.output = Some(path.into());
        self
    }

    /// Override the output codec (default [`VideoCodec::H264`]).
    pub fn codec(mut self, codec: VideoCodec) -> Self {
        self.codec = codec;
        self
    }

    /// Override the target framerate (default 30/1).
    pub fn framerate(mut self, num: i32, den: i32) -> Self {
        self.fps_num = num;
        self.fps_den = den;
        self
    }

    /// Install a [`Mp4MuxerInbox`] bundle.  Omitting this call is
    /// equivalent to `.inbox(Mp4MuxerInbox::default())`, which
    /// wires every slot to the matching `Mp4Muxer::default_on_*`.
    pub fn inbox(mut self, i: Mp4MuxerInbox) -> Self {
        self.inbox = Some(i);
        self
    }

    /// Install a [`Mp4MuxerCommon`] bundle.  Omitting this call is
    /// equivalent to `.common(Mp4MuxerCommon::default())`.
    pub fn common(mut self, c: Mp4MuxerCommon) -> Self {
        self.common = Some(c);
        self
    }

    /// Finalise the template and obtain the Layer-A
    /// [`ActorBuilder<Mp4Muxer>`].
    ///
    /// # Errors
    ///
    /// Returns `Err` if `output` is missing.
    pub fn build(self) -> Result<ActorBuilder<Mp4Muxer>> {
        let Mp4MuxerBuilder {
            name,
            capacity,
            output,
            codec,
            fps_num,
            fps_den,
            inbox,
            common,
        } = self;
        let output = output.ok_or_else(|| anyhow!("Mp4Muxer: missing output"))?;
        let Mp4MuxerInbox {
            on_source_eos,
            on_frame,
        } = inbox.unwrap_or_default();
        let Mp4MuxerCommon {
            poll_timeout,
            stopping,
        } = common.unwrap_or_default();
        Ok(ActorBuilder::new(name, capacity)
            .poll_timeout(poll_timeout)
            .factory(move |_bx| {
                Ok(Mp4Muxer {
                    output,
                    codec,
                    fps_num,
                    fps_den,
                    muxer: None,
                    poll_timeout,
                    on_source_eos,
                    on_frame,
                    stopping,
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
    fn builder_requires_output() {
        let name = StageName::unnamed(StageKind::Mp4Mux);
        let err = Mp4Muxer::builder(name, 4)
            .build()
            .err()
            .expect("missing output");
        assert!(err.to_string().contains("missing output"));
    }

    #[test]
    fn builder_accepts_all_hooks() {
        let name = StageName::unnamed(StageKind::Mp4Mux);
        let _ = Mp4Muxer::builder(name, 4)
            .output("/tmp/out.mp4")
            .codec(VideoCodec::Hevc)
            .framerate(25, 1)
            .inbox(
                Mp4MuxerInbox::builder()
                    .on_source_eos(|_sid, _ctx| Ok(Flow::Stop))
                    .on_frame(|_pl, _ctx| Ok(()))
                    .build(),
            )
            .common(
                Mp4MuxerCommon::builder()
                    .poll_timeout(Duration::from_millis(50))
                    .stopping(Mp4Muxer::default_stopping())
                    .build(),
            )
            .build()
            .unwrap();
    }

    /// The [`Mp4MuxerCommon`] bundle accepts a user-supplied
    /// `.stopping(F)` closure.  Compile-only verification of the
    /// hook bound; the ordering invariant (built-in `finish` runs
    /// before the user hook) is covered by the `cars-demo` smoke
    /// run which exercises a real [`GstMp4Muxer`].
    #[test]
    fn builder_accepts_user_stopping() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        let flag = Arc::new(AtomicBool::new(false));
        let flag_hook = flag.clone();
        let name = StageName::unnamed(StageKind::Mp4Mux);
        let _ = Mp4Muxer::builder(name, 4)
            .output("/tmp/out.mp4")
            .common(
                Mp4MuxerCommon::builder()
                    .stopping(move |_ctx| {
                        flag_hook.store(true, Ordering::SeqCst);
                    })
                    .build(),
            )
            .build()
            .unwrap();
        assert!(!flag.load(Ordering::SeqCst));
    }

    /// Runtime invariant: the `on_frame` slot is no longer
    /// `Option<FrameObserver>`.  A minimal builder without any
    /// `.inbox(...)` call still produces a runtime [`Mp4Muxer`]
    /// whose `on_frame` is populated with the no-op default.
    #[test]
    fn runtime_invariant_on_frame_non_optional() {
        use crate::context::BuildCtx;
        use crate::registry::Registry;
        use crate::shared::SharedStore;
        use std::sync::Arc;

        let name = StageName::unnamed(StageKind::Mp4Mux);
        let ab = Mp4Muxer::builder(name.clone(), 4)
            .output("/tmp/out.mp4")
            .build()
            .unwrap();
        let parts = ab.into_parts();
        let reg = Arc::new(Registry::new());
        let shared = Arc::new(SharedStore::new());
        let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let bx = BuildCtx::new(&parts.name, &reg, &shared, &stop_flag);
        let mx = (parts.factory)(&bx).expect("factory resolves");
        let Mp4Muxer {
            on_frame: _,
            on_source_eos: _,
            stopping: _,
            ..
        } = mx;
    }
}
