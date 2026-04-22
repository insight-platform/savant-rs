//! [`Picasso`] — `PipelineMsg` → `EncodedMsg` rendering +
//! encoding stage wrapping a user-supplied
//! [`PicassoEngine`](picasso::prelude::PicassoEngine).
//!
//! Direct replacement for the handwritten picasso thread in
//! `cars_tracking/pipeline/picasso.rs`.  The template:
//!
//! * Captures a [`Router<EncodedMsg>`] at build time (optionally
//!   with a default peer installed via
//!   [`PicassoBuilder::downstream`]) and hands it to the
//!   user-supplied [`PicassoEngineFactory`].  The template itself
//!   never calls `router.send`; the user's
//!   [`OnEncodedFrame`](picasso::prelude::OnEncodedFrame) impl
//!   (installed on the engine via the factory) is the single send
//!   site for encoded output.
//! * Accepts the engine via factory (required): the factory receives
//!   the wired [`Router<EncodedMsg>`] so the user can build their
//!   own `OnEncodedFrame` and plug it into
//!   [`picasso::prelude::Callbacks::builder`].
//! * Accepts a *source-spec factory* that turns a per-source
//!   `(width, height, fps)` tuple into a
//!   [`SourceSpec`](picasso::prelude::SourceSpec) the first time a
//!   given `source_id` appears.
//! * Optional per-frame preprocessor for overlays
//!   (`on_frame_preprocess`) — same hook point as the legacy
//!   sample's `attach_frame_id_overlay`.
//! * On [`PipelineMsg::SourceEos`] calls
//!   [`PicassoEngine::send_eos`](picasso::prelude::PicassoEngine::send_eos)
//!   — the downstream
//!   [`EncodedMsg::SourceEos`](crate::framework::envelopes::EncodedMsg::SourceEos)
//!   is emitted by the user's own `OnEncodedFrame` callback on
//!   [`OutputMessage::EndOfStream`](picasso::prelude::OutputMessage::EndOfStream),
//!   preserving stream alignment.
//! * On [`Actor::stopping`] calls
//!   [`PicassoEngine::shutdown`](picasso::prelude::PicassoEngine::shutdown).

use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use deepstream_buffers::{Rect, SurfaceView};
use hashbrown::HashSet;
use picasso::prelude::{PicassoEngine, SourceSpec};
use savant_core::primitives::frame::VideoFrameProxy;

use crate::framework::envelopes::{BatchDelivery, EncodedMsg, PipelineMsg, SingleDelivery};
use crate::framework::router::Router;
use crate::framework::supervisor::StageName;
use crate::framework::{
    Actor, ActorBuilder, BuildCtx, Context, Dispatch, Flow, Handler, RemoveSourcePayload,
    ShutdownPayload, SourceEosPayload, UpdateSourceSpecPayload,
};

/// Default inbox receive-poll cadence.
pub const DEFAULT_POLL_TIMEOUT: Duration = Duration::from_millis(100);

/// Source-spec factory — builds a
/// [`SourceSpec`](picasso::prelude::SourceSpec) for a
/// previously-unseen `source_id` the first time the picasso stage
/// sees a frame for it.  Invoked exactly once per `source_id` per
/// template instance.
pub type SourceSpecFactory =
    Box<dyn FnMut(&str, u32, u32, i32, i32) -> Result<SourceSpec> + Send + 'static>;

/// Per-frame preprocessor — called once per incoming
/// `(frame, _buffer)` pair before the frame is fed to
/// [`PicassoEngine::send_frame`](picasso::prelude::PicassoEngine::send_frame).
///
/// `frame_counter` is a monotonically-increasing counter across
/// all sources; use it as an overlay badge or correlate with
/// external stats.
pub type FramePreprocessor =
    Box<dyn FnMut(&VideoFrameProxy, u64, &mut Context<Picasso>) -> Result<()> + Send + 'static>;

/// Engine factory — called in phase 2 of `System::build`.  Receives
/// the already-wired [`Router<EncodedMsg>`] so the user can build
/// whatever [`OnEncodedFrame`](picasso::prelude::OnEncodedFrame)
/// impl they want (typical: one that calls `router.send(...)` for
/// `OutputMessage::VideoFrame` and `OutputMessage::EndOfStream`).
/// Typical implementation:
///
/// ```ignore
/// struct MySink { router: Router<EncodedMsg> }
/// impl OnEncodedFrame for MySink {
///     fn call(&self, output: OutputMessage) {
///         match output {
///             OutputMessage::VideoFrame(frame) => {
///                 let _ = self.router.send(EncodedMsg::Frame { frame, payload: None });
///             }
///             OutputMessage::EndOfStream(eos) => {
///                 let _ = self.router.send(EncodedMsg::SourceEos { source_id: eos.source_id });
///             }
///         }
///     }
/// }
///
/// .engine_factory(|_bx, router| {
///     let callbacks = Callbacks::builder()
///         .on_encoded_frame(MySink { router })
///         .build();
///     let general = GeneralSpec::builder().name("demo").build();
///     Ok(Arc::new(PicassoEngine::new(general, callbacks)))
/// })
/// ```
pub type PicassoEngineFactory =
    Box<dyn FnOnce(&BuildCtx, Router<EncodedMsg>) -> Result<Arc<PicassoEngine>> + Send>;

/// Optional per-frame `src_rect` provider.  Returning
/// `Some(rect)` forwards the rectangle to
/// [`PicassoEngine::send_frame`](picasso::prelude::PicassoEngine::send_frame);
/// returning `None` leaves the existing `None` default in place
/// (use the whole source surface).
pub type SrcRectProvider =
    Arc<dyn Fn(&str, &VideoFrameProxy) -> Option<Rect> + Send + Sync + 'static>;

/// `PipelineMsg` → `EncodedMsg` actor wrapping a
/// [`PicassoEngine`](picasso::prelude::PicassoEngine).
pub struct Picasso {
    engine: Arc<PicassoEngine>,
    source_spec: SourceSpecFactory,
    preprocess: Option<FramePreprocessor>,
    src_rect: Option<SrcRectProvider>,
    registered: HashSet<String>,
    frame_counter: u64,
    poll_timeout: Duration,
    shutdown_done: bool,
}

impl Picasso {
    /// Start a fluent builder for a picasso actor registered under
    /// `name` with inbox capacity `capacity`.
    pub fn builder(name: StageName, capacity: usize) -> PicassoBuilder {
        PicassoBuilder::new(name, capacity)
    }
}

impl Actor for Picasso {
    type Msg = PipelineMsg;

    fn handle(&mut self, msg: PipelineMsg, ctx: &mut Context<Self>) -> Result<Flow> {
        msg.dispatch(self, ctx)
    }

    fn poll_timeout(&self) -> Duration {
        self.poll_timeout
    }

    fn started(&mut self, ctx: &mut Context<Self>) -> Result<()> {
        log::info!("[{}] picasso started", ctx.own_name());
        Ok(())
    }

    fn stopping(&mut self, ctx: &mut Context<Self>) {
        if !self.shutdown_done {
            self.engine.shutdown();
            self.shutdown_done = true;
        }
        log::info!("[{}] picasso stopping", ctx.own_name());
    }
}

impl Handler<SingleDelivery> for Picasso {
    fn handle(&mut self, msg: SingleDelivery, ctx: &mut Context<Self>) -> Result<Flow> {
        let pairs = PipelineMsg::Delivery(msg.0).into_pairs();
        self.send_pairs(pairs, ctx)
    }
}

impl Handler<BatchDelivery> for Picasso {
    fn handle(&mut self, msg: BatchDelivery, ctx: &mut Context<Self>) -> Result<Flow> {
        let pairs = msg.0.unseal();
        self.send_pairs(pairs, ctx)
    }
}

impl Handler<SourceEosPayload> for Picasso {
    fn handle(&mut self, msg: SourceEosPayload, ctx: &mut Context<Self>) -> Result<Flow> {
        log::info!(
            "[{}] SourceEos {}: flushing picasso",
            ctx.own_name(),
            msg.source_id
        );
        if let Err(e) = self.engine.send_eos(&msg.source_id) {
            log::warn!("[{}] send_eos({}): {e}", ctx.own_name(), msg.source_id);
        }
        Ok(Flow::Cont)
    }
}

impl Handler<ShutdownPayload> for Picasso {}

impl Handler<RemoveSourcePayload> for Picasso {
    /// Tear down the Picasso worker for `source_id` via
    /// [`PicassoEngine::remove_source_spec`].  The local
    /// `registered` set is also cleared so the next delivery for the
    /// same `source_id` re-triggers the `source_spec_factory`.
    fn handle(&mut self, msg: RemoveSourcePayload, ctx: &mut Context<Self>) -> Result<Flow> {
        log::info!(
            "[{}] remove_source_spec({}) dispatched",
            ctx.own_name(),
            msg.source_id
        );
        self.engine.remove_source_spec(&msg.source_id);
        self.registered.remove(&msg.source_id);
        Ok(Flow::Cont)
    }
}

impl Handler<UpdateSourceSpecPayload> for Picasso {
    /// Install (or hot-swap) the [`SourceSpec`] bound to
    /// `source_id` via [`PicassoEngine::set_source_spec`].  Also
    /// marks the `source_id` as registered so the next delivery
    /// does not re-run the `source_spec_factory`.
    fn handle(&mut self, msg: UpdateSourceSpecPayload, ctx: &mut Context<Self>) -> Result<Flow> {
        log::info!(
            "[{}] set_source_spec({}) dispatched",
            ctx.own_name(),
            msg.source_id
        );
        let UpdateSourceSpecPayload { source_id, spec } = msg;
        self.engine
            .set_source_spec(&source_id, *spec)
            .map_err(|e| anyhow!("set_source_spec({source_id}): {e}"))?;
        self.registered.insert(source_id);
        Ok(Flow::Cont)
    }
}

impl Picasso {
    fn send_pairs(
        &mut self,
        pairs: Vec<(VideoFrameProxy, deepstream_buffers::SharedBuffer)>,
        ctx: &mut Context<Self>,
    ) -> Result<Flow> {
        for (frame, buffer) in pairs {
            let sid = frame.get_source_id();
            if self.registered.insert(sid.clone()) {
                let w = frame.get_width().max(1) as u32;
                let h = frame.get_height().max(1) as u32;
                let (fps_num, fps_den) = frame.get_fps();
                log::info!(
                    "[{}] registering source_id={sid} {w}x{h} fps={fps_num}/{fps_den}",
                    ctx.own_name()
                );
                let spec = (self.source_spec)(&sid, w, h, fps_num as i32, fps_den as i32)?;
                self.engine
                    .set_source_spec(&sid, spec)
                    .map_err(|e| anyhow!("set_source_spec({sid}): {e}"))?;
            }
            if let Some(pre) = self.preprocess.as_mut() {
                if let Err(e) = pre(&frame, self.frame_counter, ctx) {
                    log::warn!("[{}] frame preprocess failed: {e}", ctx.own_name());
                }
            }
            self.frame_counter = self.frame_counter.wrapping_add(1);

            let view = SurfaceView::from_buffer(&buffer, 0)
                .map_err(|e| anyhow!("SurfaceView::from_buffer: {e}"))?;
            let src_rect = self.src_rect.as_ref().and_then(|hook| hook(&sid, &frame));
            if let Err(e) = self.engine.send_frame(&sid, frame, view, src_rect) {
                log::error!("[{}] send_frame failed: {e}", ctx.own_name());
                return Err(anyhow!("picasso send_frame: {e}"));
            }
        }
        Ok(Flow::Cont)
    }
}

/// Fluent builder for [`Picasso`].
pub struct PicassoBuilder {
    name: StageName,
    capacity: usize,
    downstream: Option<StageName>,
    poll_timeout: Duration,
    engine_factory: Option<PicassoEngineFactory>,
    source_spec: Option<SourceSpecFactory>,
    preprocess: Option<FramePreprocessor>,
    src_rect: Option<SrcRectProvider>,
}

impl PicassoBuilder {
    /// Start a builder with sample-style poll defaults.
    pub fn new(name: StageName, capacity: usize) -> Self {
        Self {
            name,
            capacity,
            downstream: None,
            poll_timeout: DEFAULT_POLL_TIMEOUT,
            engine_factory: None,
            source_spec: None,
            preprocess: None,
            src_rect: None,
        }
    }

    /// Optional default peer installed on the
    /// [`Router<EncodedMsg>`] handed to the engine factory.  The
    /// user's `OnEncodedFrame` calls `router.send(msg)` to route to
    /// this peer and `router.send_to(&peer, msg)` to address any
    /// other registered actor by name.
    pub fn downstream(mut self, peer: StageName) -> Self {
        self.downstream = Some(peer);
        self
    }

    /// Required: engine factory.  See [`PicassoEngineFactory`].
    pub fn engine_factory<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&BuildCtx, Router<EncodedMsg>) -> Result<Arc<PicassoEngine>> + Send + 'static,
    {
        self.engine_factory = Some(Box::new(f));
        self
    }

    /// Required: source-spec factory.  See [`SourceSpecFactory`].
    pub fn source_spec_factory<F>(mut self, f: F) -> Self
    where
        F: FnMut(&str, u32, u32, i32, i32) -> Result<SourceSpec> + Send + 'static,
    {
        self.source_spec = Some(Box::new(f));
        self
    }

    /// Optional: per-frame preprocessor (overlays, counters, …).
    pub fn on_frame_preprocess<F>(mut self, f: F) -> Self
    where
        F: FnMut(&VideoFrameProxy, u64, &mut Context<Picasso>) -> Result<()> + Send + 'static,
    {
        self.preprocess = Some(Box::new(f));
        self
    }

    /// Optional: per-frame `src_rect` provider.  Returning
    /// `Some(rect)` feeds the rectangle to
    /// [`PicassoEngine::send_frame`]; returning `None` leaves the
    /// default `None` behaviour (use the whole source surface).
    pub fn src_rect_for<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &VideoFrameProxy) -> Option<Rect> + Send + Sync + 'static,
    {
        self.src_rect = Some(Arc::new(f));
        self
    }

    /// Inbox receive-poll cadence (default [`DEFAULT_POLL_TIMEOUT`]).
    pub fn poll_timeout(mut self, d: Duration) -> Self {
        self.poll_timeout = d;
        self
    }

    /// Finalise the template.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `source_spec_factory` or `engine_factory` is
    /// missing.  `downstream` is optional — callers that route
    /// exclusively via [`Router::send_to`] may omit it.
    pub fn build(self) -> Result<ActorBuilder<Picasso>> {
        let PicassoBuilder {
            name,
            capacity,
            downstream,
            poll_timeout,
            engine_factory,
            source_spec,
            preprocess,
            src_rect,
        } = self;
        let source_spec =
            source_spec.ok_or_else(|| anyhow!("Picasso: missing source_spec_factory"))?;
        let engine_factory =
            engine_factory.ok_or_else(|| anyhow!("Picasso: missing engine_factory"))?;
        Ok(ActorBuilder::new(name, capacity)
            .poll_timeout(poll_timeout)
            .factory(move |bx| {
                let router: Router<EncodedMsg> = bx.router(downstream.as_ref())?;
                let engine = (engine_factory)(bx, router)?;
                Ok(Picasso {
                    engine,
                    source_spec,
                    preprocess,
                    src_rect,
                    registered: HashSet::new(),
                    frame_counter: 0,
                    poll_timeout,
                    shutdown_done: false,
                })
            }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::supervisor::{StageKind, StageName};
    use picasso::prelude::{Callbacks, GeneralSpec, OnEncodedFrame, OutputMessage};

    fn dummy_spec(_sid: &str, _w: u32, _h: u32, _fn_: i32, _fd: i32) -> Result<SourceSpec> {
        Ok(SourceSpec::default())
    }

    /// Smallest `OnEncodedFrame` impl that owns a router and does
    /// its own sends — the shape users are expected to provide.
    struct TestSink {
        router: Router<EncodedMsg>,
    }

    impl OnEncodedFrame for TestSink {
        fn call(&self, output: OutputMessage) {
            match output {
                OutputMessage::VideoFrame(frame) => {
                    let _ = self.router.send(EncodedMsg::Frame {
                        frame,
                        payload: None,
                    });
                }
                OutputMessage::EndOfStream(eos) => {
                    let _ = self.router.send(EncodedMsg::SourceEos {
                        source_id: eos.source_id,
                    });
                }
            }
        }
    }

    fn make_engine_factory() -> PicassoEngineFactory {
        Box::new(|_bx, router| {
            let callbacks = Callbacks::builder()
                .on_encoded_frame(TestSink { router })
                .build();
            let general = GeneralSpec::builder().name("demo").build();
            Ok(Arc::new(PicassoEngine::new(general, callbacks)))
        })
    }

    #[test]
    fn builder_requires_source_spec_and_engine_factory() {
        let name = StageName::unnamed(StageKind::Picasso);
        let err = Picasso::builder(name.clone(), 4).build().err().unwrap();
        assert!(err.to_string().contains("missing source_spec_factory"));

        let err = Picasso::builder(name, 4)
            .source_spec_factory(dummy_spec)
            .build()
            .err()
            .unwrap();
        assert!(err.to_string().contains("missing engine_factory"));
    }

    /// `downstream` is optional — the builder accepts the
    /// no-downstream configuration.  The resulting actor's router
    /// logs once the first time the user's `OnEncodedFrame` tries
    /// to forward via the default peer and silently drops subsequent
    /// default-peer sends.
    #[test]
    fn builder_without_downstream_is_accepted() {
        let name = StageName::unnamed(StageKind::Picasso);
        let _ = Picasso::builder(name, 4)
            .source_spec_factory(dummy_spec)
            .engine_factory(make_engine_factory())
            .build()
            .unwrap();
    }

    #[test]
    fn picasso_implements_source_lifecycle_handlers() {
        fn assert_remove<H: Handler<RemoveSourcePayload>>() {}
        fn assert_update<H: Handler<UpdateSourceSpecPayload>>() {}
        assert_remove::<Picasso>();
        assert_update::<Picasso>();

        let remove = RemoveSourcePayload {
            source_id: "cam-0".to_string(),
        };
        assert_eq!(remove.source_id, "cam-0");

        let update = UpdateSourceSpecPayload {
            source_id: "cam-1".to_string(),
            spec: Box::new(SourceSpec::default()),
        };
        assert_eq!(update.source_id, "cam-1");
    }

    #[test]
    fn builder_accepts_full_config() {
        let name = StageName::unnamed(StageKind::Picasso);
        let _ = Picasso::builder(name, 4)
            .downstream(StageName::unnamed(StageKind::Mp4Mux))
            .engine_factory(make_engine_factory())
            .source_spec_factory(dummy_spec)
            .src_rect_for(|_sid, _frame| None)
            .on_frame_preprocess(|_f, _cnt, _ctx| Ok(()))
            .poll_timeout(Duration::from_millis(50))
            .build()
            .unwrap();
    }
}
