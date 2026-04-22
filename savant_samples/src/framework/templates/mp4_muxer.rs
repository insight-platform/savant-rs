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
//!   first-EOS terminus semantics; configurable via
//!   [`Mp4MuxerBuilder::on_source_eos`] for multi-source muxers).
//! * Finalises the `moov` atom on [`Actor::stopping`] via
//!   [`Mp4Muxer::finish`].
//! * Ignores [`EncodedMsg::StreamInfo`] / [`EncodedMsg::Packet`]
//!   at the `debug` log level — those variants are not part of
//!   the picasso → mux contract but left non-fatal for
//!   forward-compatibility.
//!
//! The in-band [`EncodedMsg::Shutdown`] sentinel is handled by the
//! loop driver via [`Envelope::as_shutdown`](super::super::Envelope::as_shutdown)
//! — the template does not add its own cooperative-exit code.

use std::time::Duration;

use anyhow::{anyhow, Result};
use savant_core::primitives::frame::VideoFrameContent;
use savant_core::primitives::video_codec::VideoCodec;
use savant_gstreamer::mp4_muxer::Mp4Muxer as GstMp4Muxer;

use crate::framework::envelopes::{EncodedMsg, FramePayload, PacketPayload, StreamInfoPayload};
use crate::framework::supervisor::StageName;
use crate::framework::{
    Actor, ActorBuilder, Context, Dispatch, Flow, Handler, ShutdownPayload, SourceEosPayload,
};

/// Default inbox receive-poll cadence.
pub const DEFAULT_POLL_TIMEOUT: Duration = Duration::from_millis(100);

/// Closure type for `on_source_eos`: decide whether an EOS should
/// terminate the muxer.  Default: [`Flow::Stop`] on the first EOS.
pub type EosHook = Box<dyn FnMut(&str, &mut Context<Mp4Muxer>) -> Result<Flow> + Send>;

/// Closure type for `on_frame` observer — runs before the frame is
/// pushed to the muxer.  Returning `Err` aborts the loop.
pub type FrameObserver = Box<dyn FnMut(&FramePayload, &mut Context<Mp4Muxer>) -> Result<()> + Send>;

/// `EncodedMsg` → MP4 terminus actor.
///
/// Construct via [`Mp4Muxer::builder`].
pub struct Mp4Muxer {
    output: String,
    codec: VideoCodec,
    fps_num: i32,
    fps_den: i32,
    muxer: Option<GstMp4Muxer>,
    poll_timeout: Duration,
    on_source_eos: EosHook,
    on_frame: Option<FrameObserver>,
    finalised: bool,
}

impl Mp4Muxer {
    /// Start a fluent builder for a muxer registered under `name`
    /// with inbox capacity `capacity`.
    pub fn builder(name: StageName, capacity: usize) -> Mp4MuxerBuilder {
        Mp4MuxerBuilder::new(name, capacity)
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
    }
}

impl Handler<FramePayload> for Mp4Muxer {
    fn handle(&mut self, msg: FramePayload, ctx: &mut Context<Self>) -> Result<Flow> {
        if let Some(hook) = self.on_frame.as_mut() {
            hook(&msg, ctx)?;
        }
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

/// Fluent builder for [`Mp4Muxer`].
pub struct Mp4MuxerBuilder {
    name: StageName,
    capacity: usize,
    output: Option<String>,
    codec: VideoCodec,
    fps_num: i32,
    fps_den: i32,
    poll_timeout: Duration,
    on_source_eos: EosHook,
    on_frame: Option<FrameObserver>,
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
            poll_timeout: DEFAULT_POLL_TIMEOUT,
            on_source_eos: Box::new(default_first_eos_stops),
            on_frame: None,
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

    /// Inbox receive-poll cadence (default [`DEFAULT_POLL_TIMEOUT`]).
    pub fn poll_timeout(mut self, d: Duration) -> Self {
        self.poll_timeout = d;
        self
    }

    /// Install a custom source-EOS hook.  Return [`Flow::Stop`] to
    /// finalise the file on that EOS (default: first EOS stops);
    /// [`Flow::Cont`] to keep accepting frames from other sources
    /// (multi-source mux).
    pub fn on_source_eos<F>(mut self, f: F) -> Self
    where
        F: FnMut(&str, &mut Context<Mp4Muxer>) -> Result<Flow> + Send + 'static,
    {
        self.on_source_eos = Box::new(f);
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
            poll_timeout,
            on_source_eos,
            on_frame,
        } = self;
        let output = output.ok_or_else(|| anyhow!("Mp4Muxer: missing output"))?;
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
                    finalised: false,
                })
            }))
    }
}

fn default_first_eos_stops(source_id: &str, ctx: &mut Context<Mp4Muxer>) -> Result<Flow> {
    log::info!("[{}] SourceEos {source_id}: finalising", ctx.own_name());
    Ok(Flow::Stop)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::supervisor::{StageKind, StageName};

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
            .poll_timeout(Duration::from_millis(50))
            .on_source_eos(|_sid, _ctx| Ok(Flow::Stop))
            .on_frame(|_pl, _ctx| Ok(()))
            .build()
            .unwrap();
    }
}
