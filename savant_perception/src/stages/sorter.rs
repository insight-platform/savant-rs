//! [`Sorter`] — `PipelineMsg` → `PipelineMsg` reorder buffer that
//! restores per-source frame ordering after a fan-out / fan-in
//! section.
//!
//! # The problem
//!
//! Pipelines that fan a per-frame stream out across multiple
//! parallel routes (e.g. one route per object class for inference,
//! one route per region of interest, …) and fan it back in tend to
//! reorder frames: route A may finish frame 17 before route B
//! finishes frame 16, so the converged stream observes
//! `… 15, 17, 16, 18, …`.  Most downstream consumers assume
//! per-source ordering — encoders, muxers, and analytics that
//! depend on frame timestamps all break under reordering.
//!
//! # Contract
//!
//! `Sorter` lives at the fan-in point and reorders frames per
//! source.  Two parties cooperate:
//!
//! * The **registrar** — the upstream stage that fans the stream
//!   out — registers the *intended* order **before** routing each
//!   frame down a fan-out branch.  Registrations are sent through
//!   the sorter's normal inbox using
//!   [`PipelineMsg::message_ex`](crate::envelopes::PipelineMsg::message_ex)
//!   carrying a [`SorterRegistration`] payload.  Convenience
//!   constructors [`Sorter::register_frame`] and
//!   [`Sorter::register_eos`] mint the envelopes.
//! * The **fan-out branches** — convert the frames and send them
//!   to the sorter's inbox in whatever order they happen to
//!   complete.  The sorter unseals each delivery on arrival,
//!   matches it against the per-source registration queue, and
//!   forwards frames downstream **in registration order**.  Any
//!   stage on a route that decides not to deliver a previously-
//!   announced uuid (e.g. a filter that drops a frame) emits
//!   [`Sorter::unregister_frame`] to splice the uuid out of the
//!   per-source queue — without it, the sorter would block its
//!   drain forever waiting for the abandoned frame.
//!
//! Ingress [`PipelineMsg::SourceEos`] sentinels are silently
//! discarded — the sorter's source-EOS contract is driven entirely
//! by [`SorterRegistration::Eos`] entries the registrar emits in
//! the right place in the registration sequence.
//!
//! # Per-source state machine
//!
//! For every source the sorter keeps:
//!
//! * `expected: VecDeque<Slot>` — registrations in arrival order.
//!   Each slot is either `Frame { uuid }` or `Eos`.  The
//!   registrar pushes onto the back; the sorter pops from the
//!   front.
//! * `expected_uuids: HashSet<u128>` — fast membership lookup so
//!   an arriving frame can be classified `O(1)`.
//! * `pending: HashMap<u128, (VideoFrame, SharedBuffer)>` —
//!   frames received but waiting for the matching slot to reach
//!   the head of `expected`.
//!
//! On every input the sorter re-runs the drain loop:
//! pop slots from the front while the head is either
//! `Frame { uuid }` whose pair is already in `pending` (emit it via
//! the [`SorterResults::on_message`] hook), or `Eos` (fire
//! [`SorterResults::on_source_eos`] and discard the per-source
//! state).  Stop as soon as the head is a frame slot whose pair
//! has not yet arrived.
//!
//! Frames that arrive whose uuid is not in `expected_uuids` are
//! forwarded to the [`SorterInbox::unregistered`] hook (default:
//! `warn!` + drop) — the registrar is required to publish every
//! uuid before sending the corresponding frame down a fan-out
//! branch, so unregistered traffic indicates either a bug in the
//! registrar or noise from an unrelated sender.  A pair that
//! arrives after its slot has been retracted via
//! [`SorterRegistration::Unregister`] is treated the same way:
//! the uuid is no longer in `expected_uuids`, so the
//! `unregistered` hook fires.
//!
//! # Batches
//!
//! Both [`PipelineMsg::Delivery`] and
//! [`PipelineMsg::Deliveries`] are accepted as input but **batch
//! boundaries are dissolved**: each `(frame, buffer)` pair is
//! matched against `expected` independently, and the default
//! [`SorterResults::on_message`] hook re-seals each pair as a
//! fresh single-frame [`PipelineMsg::Delivery`] before forwarding
//! it downstream.  Batches produced by nvinfer / nvtracker are a
//! convenience artefact of those operators; the sorter does not
//! preserve them.
//!
//! # Grouped builder API
//!
//! The stage exposes three hook bundles matching the cross-stage
//! grouped-builder pattern:
//!
//! * [`SorterInbox`] — ingress concerns
//!   ([`unregistered`](SorterInboxBuilder::unregistered)).
//! * [`SorterResults`] — egress concerns
//!   ([`on_message`](SorterResultsBuilder::on_message),
//!   [`on_source_eos`](SorterResultsBuilder::on_source_eos)).
//! * [`SorterCommon`] — lifecycle + loop knobs
//!   (`stopping`, `poll_timeout`).
//!
//! ```ignore
//! use savant_perception::stages::Sorter;
//! use savant_perception::supervisor::{StageKind, StageName};
//!
//! Sorter::builder(StageName::unnamed(StageKind::Sorter), 16)
//!     .downstream(picasso_name)
//!     .build();
//! ```
//!
//! Every hook has a sensible default (forward via `router.send`,
//! warn-and-drop unregistered, log on EOS), so the minimal builder
//! above is a complete reorder buffer that requires zero user
//! code.
//!
//! # Runtime invariant
//!
//! Every hook slot on the runtime [`Sorter`] struct is a
//! non-`Option` boxed closure: when the user does not install a
//! hook, the bundle builders substitute the matching
//! `Sorter::default_*` at build time.  The actor body never
//! branches on whether a hook is present.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use deepstream_buffers::sealed::Sealed;
use deepstream_buffers::SharedBuffer;
use savant_core::primitives::frame::VideoFrame;
use savant_core::utils::release_seal::ReleaseSeal;

use crate::envelopes::{BatchDelivery, PipelineMsg, SingleDelivery};
use crate::message_ex::MessageExPayload;
use crate::router::Router;
use crate::supervisor::StageName;
use crate::{
    Actor, ActorBuilder, Context, Dispatch, Flow, Handler, ShutdownPayload, SourceEosPayload,
};

/// Default inbox receive-poll cadence.
pub const DEFAULT_POLL_TIMEOUT: Duration = Duration::from_millis(100);

/// Registration payload carried via
/// [`PipelineMsg::message_ex`](crate::envelopes::PipelineMsg::message_ex)
/// to declare the intended per-source ordering at the sorter.  The
/// registrar emits one of these *before* routing the corresponding
/// frame down a fan-out branch (or *before* the registrar wants the
/// sorter to fire its EOS hook); any stage on the route may emit
/// [`SorterRegistration::Unregister`] to retract a previously
/// announced uuid that will not arrive.
///
/// Construct via [`Sorter::register_frame`] / [`Sorter::register_eos`] /
/// [`Sorter::unregister_frame`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SorterRegistration {
    /// Register a frame uuid in the per-source order queue.  The
    /// sorter will hold the frame in its `pending` map until this
    /// slot reaches the head of `expected`, then emit it via
    /// [`SorterResults::on_message`].
    Frame {
        /// Source id this frame belongs to.
        source_id: String,
        /// Frame uuid (matches
        /// [`VideoFrame::get_uuid_u128`](savant_core::primitives::frame::VideoFrame::get_uuid_u128)).
        uuid: u128,
    },
    /// Retract a previously-announced [`Frame`](Self::Frame) slot.
    /// Used by route stages that decide *not* to deliver a frame
    /// the registrar already published — e.g. an inference stage
    /// that drops a frame on a confidence threshold, or a tracker
    /// that times the frame out before it reaches the sorter.
    /// Without this retraction the sorter would block its
    /// per-source drain forever waiting for the abandoned uuid.
    ///
    /// On receipt the sorter:
    ///
    /// * splices the matching `Frame { uuid }` slot out of the
    ///   per-source `expected` queue (preserving the relative
    ///   order of every other slot);
    /// * drops any pending `(frame, buffer)` pair that has already
    ///   arrived under that uuid — the upstream raced its own
    ///   `Unregister` against the delivery, and the sorter
    ///   honours the retraction;
    /// * re-runs the drain pass so any subsequent slot whose pair
    ///   was waiting on the now-removed head can flush downstream.
    ///
    /// Unregistering a uuid that is not currently in `expected` is
    /// idempotent: the sorter logs at `warn` level and continues.
    /// Unregistering an [`Eos`](Self::Eos) slot is not supported —
    /// EOS slots carry no uuid and are addressed by source id
    /// alone; if the route can no longer guarantee a stream's
    /// completion, simply do not emit a `Register::Eos` for it.
    Unregister {
        /// Source id this retraction belongs to.
        source_id: String,
        /// Frame uuid being retracted (must match a previous
        /// [`Frame`](Self::Frame) registration).
        uuid: u128,
    },
    /// Register an end-of-stream sentinel for the source.  The
    /// sorter will fire [`SorterResults::on_source_eos`] when this
    /// slot reaches the head of `expected`, then discard the
    /// per-source state.
    Eos {
        /// Source id this EOS belongs to.
        source_id: String,
    },
}

/// Egress hook fired once for every frame the sorter releases in
/// registration order.  Receives the unsealed
/// `(frame, buffer)` pair, the stage's [`Router<PipelineMsg>`],
/// and the actor's [`Context`].
///
/// Default: re-seal the pair as a fresh [`PipelineMsg::Delivery`]
/// (with an immediately-released [`ReleaseSeal`]) and forward via
/// `router.send(msg)`.  Override only if you need to bypass the
/// re-seal (e.g. accumulate into a custom batch envelope or
/// emit through a different channel).
pub type OnMessageHook = Box<
    dyn FnMut(VideoFrame, SharedBuffer, &Router<PipelineMsg>, &mut Context<Sorter>) -> Result<Flow>
        + Send
        + 'static,
>;

/// Egress hook fired once for every
/// [`SorterRegistration::Eos`] when its slot reaches the head of
/// `expected`.  Receives the source id, the router, and the
/// actor's [`Context`].
///
/// Default: forward [`PipelineMsg::SourceEos`] via
/// `router.send(msg)` and return [`Flow::Cont`] so the sorter
/// stays alive across an arbitrary number of per-source drains.
pub type OnSourceEosHook = Box<
    dyn FnMut(&str, &Router<PipelineMsg>, &mut Context<Sorter>) -> Result<Flow> + Send + 'static,
>;

/// Ingress hook fired when a frame whose uuid is **not** in
/// `expected_uuids` arrives.  Receives the unsealed
/// `(frame, buffer)` pair, the router, and the actor's
/// [`Context`].
///
/// Default: emit a single `warn!` line and drop the pair.  The
/// implicit invariant is that the registrar publishes every uuid
/// **before** the corresponding frame can be in flight; an
/// unregistered arrival is therefore a bug or noise from an
/// unrelated sender.
pub type UnregisteredHook = Box<
    dyn FnMut(VideoFrame, SharedBuffer, &Router<PipelineMsg>, &mut Context<Sorter>) -> Result<()>
        + Send
        + 'static,
>;

/// User shutdown hook invoked from [`Actor::stopping`].  The
/// stage has no load-bearing built-in cleanup, so this hook's body
/// is the only work done on stop (after a single `info!` log line
/// from the stage itself).  Default: no-op.
pub type OnStoppingHook = Box<dyn FnMut(&mut Context<Sorter>) + Send + 'static>;

/// Per-source state — the registration queue, fast membership set,
/// and pending-pairs map.  Created lazily on the first arrival or
/// registration for a given `source_id`; destroyed when the
/// source's [`SorterRegistration::Eos`] slot is drained.
struct PerSource {
    /// Registrations in arrival order — the sorter pops from the
    /// front when releasing frames downstream.
    expected: VecDeque<Slot>,
    /// Mirror of every `Frame { uuid }` slot currently in
    /// `expected`, used for `O(1)` "is this uuid registered?"
    /// classification of arriving frames.
    expected_uuids: HashSet<u128>,
    /// Frames received whose registration slot has not yet
    /// reached the head of `expected`.
    pending: HashMap<u128, (VideoFrame, SharedBuffer)>,
}

impl PerSource {
    fn new() -> Self {
        Self {
            expected: VecDeque::new(),
            expected_uuids: HashSet::new(),
            pending: HashMap::new(),
        }
    }
}

/// Single registration slot in the per-source `expected` queue.
#[derive(Clone)]
enum Slot {
    Frame { uuid: u128 },
    Eos,
}

/// One ready-to-emit item popped from `expected` during a drain
/// pass.  Lifted out of the per-source borrow so the egress hook
/// can run without aliasing `&mut self`.
enum Ready {
    Frame(VideoFrame, SharedBuffer),
    Eos,
}

/// `PipelineMsg` → `PipelineMsg` reorder buffer.  See the module
/// docs for the full contract.  Hook slots are non-`Option` by
/// construction — see the runtime invariant in the module docs.
pub struct Sorter {
    router: Router<PipelineMsg>,
    on_message: OnMessageHook,
    on_source_eos: OnSourceEosHook,
    unregistered: UnregisteredHook,
    stopping: OnStoppingHook,
    poll_timeout: Duration,
    per_source: HashMap<String, PerSource>,
}

impl Sorter {
    /// Start a fluent builder for a sorter registered under `name`
    /// with inbox capacity `capacity`.
    pub fn builder(name: StageName, capacity: usize) -> SorterBuilder {
        SorterBuilder::new(name, capacity)
    }

    /// Sender-side sugar: construct a [`PipelineMsg`] that
    /// registers `uuid` in the per-source order queue at the
    /// sorter.
    pub fn register_frame(source_id: impl Into<String>, uuid: u128) -> PipelineMsg {
        PipelineMsg::message_ex(SorterRegistration::Frame {
            source_id: source_id.into(),
            uuid,
        })
    }

    /// Sender-side sugar: construct a [`PipelineMsg`] that
    /// registers an EOS sentinel in the per-source order queue at
    /// the sorter.
    pub fn register_eos(source_id: impl Into<String>) -> PipelineMsg {
        PipelineMsg::message_ex(SorterRegistration::Eos {
            source_id: source_id.into(),
        })
    }

    /// Sender-side sugar: construct a [`PipelineMsg`] that
    /// retracts a previously-registered frame uuid.  Use from any
    /// stage on the route that decides not to deliver a frame the
    /// registrar already announced — see
    /// [`SorterRegistration::Unregister`] for the full contract.
    pub fn unregister_frame(source_id: impl Into<String>, uuid: u128) -> PipelineMsg {
        PipelineMsg::message_ex(SorterRegistration::Unregister {
            source_id: source_id.into(),
            uuid,
        })
    }

    /// Default `on_message` hook — re-seal the pair as a fresh
    /// single-frame [`PipelineMsg::Delivery`] (the new
    /// [`ReleaseSeal`] is released immediately) and forward via
    /// `router.send(msg)`.
    pub fn default_on_message(
    ) -> impl FnMut(VideoFrame, SharedBuffer, &Router<PipelineMsg>, &mut Context<Sorter>) -> Result<Flow>
           + Send
           + 'static {
        |frame, buffer, router, ctx| {
            let seal = Arc::new(ReleaseSeal::new());
            let sealed = Sealed::new((frame, buffer), Arc::clone(&seal));
            seal.release();
            if !router.send(PipelineMsg::Delivery(sealed)) {
                log::warn!(
                    "[{}] downstream closed; dropping ordered frame",
                    ctx.own_name()
                );
            }
            Ok(Flow::Cont)
        }
    }

    /// Default `on_source_eos` hook — forward
    /// [`PipelineMsg::SourceEos`] via `router.send(msg)` and
    /// return [`Flow::Cont`].
    pub fn default_on_source_eos(
    ) -> impl FnMut(&str, &Router<PipelineMsg>, &mut Context<Sorter>) -> Result<Flow> + Send + 'static
    {
        |source_id, router, ctx| {
            log::info!(
                "[{}] SourceEos {source_id}: forwarding (registered)",
                ctx.own_name()
            );
            if !router.send(PipelineMsg::SourceEos {
                source_id: source_id.to_string(),
            }) {
                log::warn!(
                    "[{}] downstream closed; dropping SourceEos({source_id})",
                    ctx.own_name()
                );
            }
            Ok(Flow::Cont)
        }
    }

    /// Default `unregistered` hook — log a single `warn!` line and
    /// drop the pair.  The pair's `SharedBuffer` is released via
    /// the normal `Arc` decrement when the closure body returns.
    pub fn default_unregistered(
    ) -> impl FnMut(VideoFrame, SharedBuffer, &Router<PipelineMsg>, &mut Context<Sorter>) -> Result<()>
           + Send
           + 'static {
        |frame, _buffer, _router, ctx| {
            log::warn!(
                "[{}] dropped unregistered frame source_id={} uuid={}",
                ctx.own_name(),
                frame.get_source_id(),
                frame.get_uuid_as_string(),
            );
            Ok(())
        }
    }

    /// Default user shutdown hook — no-op.  The stage emits a
    /// single `info!` log line of its own before invoking this
    /// hook.
    pub fn default_stopping() -> impl FnMut(&mut Context<Sorter>) + Send + 'static {
        |_ctx| {}
    }

    fn entry_mut(&mut self, source_id: &str) -> &mut PerSource {
        if !self.per_source.contains_key(source_id) {
            self.per_source
                .insert(source_id.to_string(), PerSource::new());
        }
        self.per_source.get_mut(source_id).expect("just inserted")
    }

    /// Splice the `Frame { uuid }` slot out of the per-source
    /// `expected` queue and drop any pending pair already received
    /// under that uuid.  Returns `true` if the uuid was actually
    /// found in the queue (caller can use the boolean to decide
    /// whether to log a "no-op unregister" diagnostic).
    fn splice_uuid(&mut self, source_id: &str, uuid: u128) -> bool {
        let Some(entry) = self.per_source.get_mut(source_id) else {
            return false;
        };
        if !entry.expected_uuids.remove(&uuid) {
            return false;
        }
        entry
            .expected
            .retain(|slot| !matches!(slot, Slot::Frame { uuid: u } if *u == uuid));
        // Drop any pending pair: the route has retracted the
        // delivery promise, so the frame must not flow downstream
        // even if it has already arrived.
        let _ = entry.pending.remove(&uuid);
        true
    }

    /// Pop a single ready slot from `expected` (front), removing
    /// the matching pair from `pending` if it is a frame slot.
    /// Returns `None` if the head slot is a frame whose pair has
    /// not yet arrived (or `expected` is empty / the source is
    /// unknown).  Also drops the per-source entry once an `Eos`
    /// slot is popped.
    fn pop_ready(&mut self, source_id: &str) -> Option<Ready> {
        let entry = self.per_source.get_mut(source_id)?;
        let head = entry.expected.front()?;
        match head {
            Slot::Frame { uuid } => {
                let uuid = *uuid;
                let pair = entry.pending.remove(&uuid)?;
                entry.expected.pop_front();
                entry.expected_uuids.remove(&uuid);
                Some(Ready::Frame(pair.0, pair.1))
            }
            Slot::Eos => {
                entry.expected.pop_front();
                let drop_entry = entry.expected.is_empty() && entry.pending.is_empty();
                if drop_entry {
                    self.per_source.remove(source_id);
                }
                Some(Ready::Eos)
            }
        }
    }

    /// Drain `expected` in order for `source_id`, firing the
    /// matching egress hook for each slot.  Stops at the first
    /// frame slot whose pair has not arrived yet, or as soon as a
    /// hook returns [`Flow::Stop`].
    fn drain(&mut self, source_id: &str, ctx: &mut Context<Self>) -> Result<Flow> {
        loop {
            let Some(ready) = self.pop_ready(source_id) else {
                return Ok(Flow::Cont);
            };
            let flow = match ready {
                Ready::Frame(frame, buffer) => {
                    (self.on_message)(frame, buffer, &self.router, ctx)?
                }
                Ready::Eos => (self.on_source_eos)(source_id, &self.router, ctx)?,
            };
            if matches!(flow, Flow::Stop) {
                return Ok(Flow::Stop);
            }
        }
    }

    /// Match an incoming `(frame, buffer)` pair against the
    /// per-source state machine — stash in `pending` if the uuid
    /// is registered, otherwise route to the `unregistered` hook.
    fn ingest_pair(
        &mut self,
        frame: VideoFrame,
        buffer: SharedBuffer,
        ctx: &mut Context<Self>,
    ) -> Result<Flow> {
        let source_id = frame.get_source_id();
        let uuid = frame.get_uuid_u128();
        let registered = self
            .per_source
            .get(&source_id)
            .map(|e| e.expected_uuids.contains(&uuid))
            .unwrap_or(false);
        if !registered {
            (self.unregistered)(frame, buffer, &self.router, ctx)?;
            return Ok(Flow::Cont);
        }
        let entry = self.entry_mut(&source_id);
        entry.pending.insert(uuid, (frame, buffer));
        self.drain(&source_id, ctx)
    }
}

impl Actor for Sorter {
    type Msg = PipelineMsg;

    fn handle(&mut self, msg: PipelineMsg, ctx: &mut Context<Self>) -> Result<Flow> {
        msg.dispatch(self, ctx)
    }

    fn poll_timeout(&self) -> Duration {
        self.poll_timeout
    }

    fn started(&mut self, ctx: &mut Context<Self>) -> Result<()> {
        log::info!("[{}] sorter started", ctx.own_name());
        Ok(())
    }

    fn stopping(&mut self, ctx: &mut Context<Self>) {
        log::info!("[{}] sorter stopping", ctx.own_name());
        (self.stopping)(ctx);
    }

    fn handle_message_ex(
        &mut self,
        msg: MessageExPayload,
        ctx: &mut Context<Self>,
    ) -> Result<Flow> {
        match msg.downcast::<SorterRegistration>() {
            Ok(reg) => match *reg {
                SorterRegistration::Frame { source_id, uuid } => {
                    let entry = self.entry_mut(&source_id);
                    if entry.expected_uuids.insert(uuid) {
                        entry.expected.push_back(Slot::Frame { uuid });
                    } else {
                        log::warn!(
                            "[{}] duplicate registration source_id={source_id} uuid={uuid}; ignoring",
                            ctx.own_name()
                        );
                    }
                    self.drain(&source_id, ctx)
                }
                SorterRegistration::Unregister { source_id, uuid } => {
                    if !self.splice_uuid(&source_id, uuid) {
                        log::warn!(
                            "[{}] unregister: source_id={source_id} uuid={uuid} not in expected; ignoring",
                            ctx.own_name()
                        );
                    }
                    self.drain(&source_id, ctx)
                }
                SorterRegistration::Eos { source_id } => {
                    let entry = self.entry_mut(&source_id);
                    entry.expected.push_back(Slot::Eos);
                    self.drain(&source_id, ctx)
                }
            },
            Err(payload) => {
                log::debug!(
                    "[{}] MessageEx({}) dropped (not a SorterRegistration)",
                    ctx.own_name(),
                    payload.type_name,
                );
                Ok(Flow::Cont)
            }
        }
    }
}

impl Handler<SingleDelivery> for Sorter {
    fn handle(&mut self, msg: SingleDelivery, ctx: &mut Context<Self>) -> Result<Flow> {
        let pairs = PipelineMsg::Delivery(msg.0).into_pairs();
        for (frame, buffer) in pairs {
            if matches!(self.ingest_pair(frame, buffer, ctx)?, Flow::Stop) {
                return Ok(Flow::Stop);
            }
        }
        Ok(Flow::Cont)
    }
}

impl Handler<BatchDelivery> for Sorter {
    fn handle(&mut self, msg: BatchDelivery, ctx: &mut Context<Self>) -> Result<Flow> {
        let pairs = msg.0.unseal();
        for (frame, buffer) in pairs {
            if matches!(self.ingest_pair(frame, buffer, ctx)?, Flow::Stop) {
                return Ok(Flow::Stop);
            }
        }
        Ok(Flow::Cont)
    }
}

/// Ingress source-EOS sentinels are silently discarded — the
/// sorter's per-source EOS contract is driven by
/// [`SorterRegistration::Eos`] entries the registrar emits in the
/// right place in the registration sequence.
impl Handler<SourceEosPayload> for Sorter {
    fn handle(&mut self, _msg: SourceEosPayload, _ctx: &mut Context<Self>) -> Result<Flow> {
        Ok(Flow::Cont)
    }
}

impl Handler<ShutdownPayload> for Sorter {}

/// Ingress hook bundle for [`Sorter`].
pub struct SorterInbox {
    unregistered: UnregisteredHook,
}

impl SorterInbox {
    /// Start a builder that auto-installs every default on
    /// [`SorterInboxBuilder::build`].
    pub fn builder() -> SorterInboxBuilder {
        SorterInboxBuilder::new()
    }
}

impl Default for SorterInbox {
    fn default() -> Self {
        SorterInboxBuilder::new().build()
    }
}

/// Fluent builder for [`SorterInbox`].
pub struct SorterInboxBuilder {
    unregistered: Option<UnregisteredHook>,
}

impl SorterInboxBuilder {
    /// Empty bundle — every hook defaults to the matching
    /// `Sorter::default_*` at [`build`](Self::build) time.
    pub fn new() -> Self {
        Self { unregistered: None }
    }

    /// Override the `unregistered` hook.  Default: warn-log + drop.
    pub fn unregistered<F>(mut self, f: F) -> Self
    where
        F: FnMut(
                VideoFrame,
                SharedBuffer,
                &Router<PipelineMsg>,
                &mut Context<Sorter>,
            ) -> Result<()>
            + Send
            + 'static,
    {
        self.unregistered = Some(Box::new(f));
        self
    }

    /// Finalise the bundle, substituting defaults for every
    /// omitted setter.
    pub fn build(self) -> SorterInbox {
        let SorterInboxBuilder { unregistered } = self;
        SorterInbox {
            unregistered: unregistered.unwrap_or_else(|| Box::new(Sorter::default_unregistered())),
        }
    }
}

impl Default for SorterInboxBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Egress hook bundle for [`Sorter`].
pub struct SorterResults {
    on_message: OnMessageHook,
    on_source_eos: OnSourceEosHook,
}

impl SorterResults {
    /// Start a builder that auto-installs every default on
    /// [`SorterResultsBuilder::build`].
    pub fn builder() -> SorterResultsBuilder {
        SorterResultsBuilder::new()
    }
}

impl Default for SorterResults {
    fn default() -> Self {
        SorterResultsBuilder::new().build()
    }
}

/// Fluent builder for [`SorterResults`].
pub struct SorterResultsBuilder {
    on_message: Option<OnMessageHook>,
    on_source_eos: Option<OnSourceEosHook>,
}

impl SorterResultsBuilder {
    /// Empty bundle — every hook defaults to the matching
    /// `Sorter::default_*` at [`build`](Self::build) time.
    pub fn new() -> Self {
        Self {
            on_message: None,
            on_source_eos: None,
        }
    }

    /// Override the `on_message` hook.  Default: re-seal the pair
    /// as a fresh [`PipelineMsg::Delivery`] and forward via
    /// `router.send(msg)`.
    pub fn on_message<F>(mut self, f: F) -> Self
    where
        F: FnMut(
                VideoFrame,
                SharedBuffer,
                &Router<PipelineMsg>,
                &mut Context<Sorter>,
            ) -> Result<Flow>
            + Send
            + 'static,
    {
        self.on_message = Some(Box::new(f));
        self
    }

    /// Override the `on_source_eos` hook.  Default: forward
    /// [`PipelineMsg::SourceEos`] via `router.send(msg)` and
    /// return [`Flow::Cont`].
    pub fn on_source_eos<F>(mut self, f: F) -> Self
    where
        F: FnMut(&str, &Router<PipelineMsg>, &mut Context<Sorter>) -> Result<Flow>
            + Send
            + 'static,
    {
        self.on_source_eos = Some(Box::new(f));
        self
    }

    /// Finalise the bundle, substituting defaults for every
    /// omitted setter.
    pub fn build(self) -> SorterResults {
        let SorterResultsBuilder {
            on_message,
            on_source_eos,
        } = self;
        SorterResults {
            on_message: on_message.unwrap_or_else(|| Box::new(Sorter::default_on_message())),
            on_source_eos: on_source_eos
                .unwrap_or_else(|| Box::new(Sorter::default_on_source_eos())),
        }
    }
}

impl Default for SorterResultsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Lifecycle + loop knob bundle for [`Sorter`].
pub struct SorterCommon {
    poll_timeout: Duration,
    stopping: OnStoppingHook,
}

impl SorterCommon {
    /// Start a builder seeded with [`DEFAULT_POLL_TIMEOUT`] and
    /// the no-op stopping hook.
    pub fn builder() -> SorterCommonBuilder {
        SorterCommonBuilder::new()
    }
}

impl Default for SorterCommon {
    fn default() -> Self {
        SorterCommonBuilder::new().build()
    }
}

/// Fluent builder for [`SorterCommon`].
pub struct SorterCommonBuilder {
    poll_timeout: Option<Duration>,
    stopping: Option<OnStoppingHook>,
}

impl SorterCommonBuilder {
    /// Empty bundle — `poll_timeout` defaults to
    /// [`DEFAULT_POLL_TIMEOUT`] and `stopping` to a no-op.
    pub fn new() -> Self {
        Self {
            poll_timeout: None,
            stopping: None,
        }
    }

    /// Inbox receive-poll cadence (default
    /// [`DEFAULT_POLL_TIMEOUT`]).
    pub fn poll_timeout(mut self, d: Duration) -> Self {
        self.poll_timeout = Some(d);
        self
    }

    /// Override the user shutdown hook — runs on the actor thread
    /// after the stage's own `info!` log line.  Default: no-op.
    pub fn stopping<F>(mut self, f: F) -> Self
    where
        F: FnMut(&mut Context<Sorter>) + Send + 'static,
    {
        self.stopping = Some(Box::new(f));
        self
    }

    /// Finalise the bundle, substituting defaults for every
    /// omitted setter.
    pub fn build(self) -> SorterCommon {
        let SorterCommonBuilder {
            poll_timeout,
            stopping,
        } = self;
        SorterCommon {
            poll_timeout: poll_timeout.unwrap_or(DEFAULT_POLL_TIMEOUT),
            stopping: stopping.unwrap_or_else(|| Box::new(Sorter::default_stopping())),
        }
    }
}

impl Default for SorterCommonBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Fluent builder for [`Sorter`].
pub struct SorterBuilder {
    name: StageName,
    capacity: usize,
    downstream: Option<StageName>,
    inbox: Option<SorterInbox>,
    results: Option<SorterResults>,
    common: Option<SorterCommon>,
}

impl SorterBuilder {
    /// Start a builder for a sorter actor registered under `name`
    /// with inbox capacity `capacity`.
    pub fn new(name: StageName, capacity: usize) -> Self {
        Self {
            name,
            capacity,
            downstream: None,
            inbox: None,
            results: None,
            common: None,
        }
    }

    /// Optional default peer for `router.send(msg)` inside the
    /// per-variant hooks.  When omitted, hooks must use
    /// `router.send_to(&peer, msg)`.
    pub fn downstream(mut self, peer: StageName) -> Self {
        self.downstream = Some(peer);
        self
    }

    /// Install a [`SorterInbox`] bundle.  Omitting this call is
    /// equivalent to `.inbox(SorterInbox::default())`, which
    /// wires every slot to the matching `Sorter::default_*`.
    pub fn inbox(mut self, i: SorterInbox) -> Self {
        self.inbox = Some(i);
        self
    }

    /// Install a [`SorterResults`] bundle.  Omitting this call is
    /// equivalent to `.results(SorterResults::default())`.
    pub fn results(mut self, r: SorterResults) -> Self {
        self.results = Some(r);
        self
    }

    /// Install a [`SorterCommon`] bundle.  Omitting this call is
    /// equivalent to `.common(SorterCommon::default())`.
    pub fn common(mut self, c: SorterCommon) -> Self {
        self.common = Some(c);
        self
    }

    /// Finalise the stage and obtain the Layer-A
    /// [`ActorBuilder<Sorter>`] suitable for
    /// [`System::register_actor`](super::super::System::register_actor).
    pub fn build(self) -> ActorBuilder<Sorter> {
        let SorterBuilder {
            name,
            capacity,
            downstream,
            inbox,
            results,
            common,
        } = self;
        let SorterInbox { unregistered } = inbox.unwrap_or_default();
        let SorterResults {
            on_message,
            on_source_eos,
        } = results.unwrap_or_default();
        let SorterCommon {
            poll_timeout,
            stopping,
        } = common.unwrap_or_default();
        ActorBuilder::new(name, capacity)
            .poll_timeout(poll_timeout)
            .factory(move |bx| {
                let router: Router<PipelineMsg> = bx.router(downstream.as_ref())?;
                Ok(Sorter {
                    router,
                    on_message,
                    on_source_eos,
                    unregistered,
                    stopping,
                    poll_timeout,
                    per_source: HashMap::new(),
                })
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::addr::Addr;
    use crate::registry::Registry;
    use crate::shared::SharedStore;
    use crate::supervisor::{StageKind, StageName};
    use crossbeam::channel::{bounded, Receiver};
    use std::sync::atomic::AtomicBool;
    use std::sync::Arc;

    /// Build a freestanding `Sorter` plus a `Receiver<PipelineMsg>`
    /// that catches everything its router forwards.
    fn make_sorter() -> (Sorter, Receiver<PipelineMsg>) {
        let downstream = StageName::unnamed(StageKind::DeepStreamFunction);
        let mut reg = Registry::new();
        let (tx, rx) = bounded::<PipelineMsg>(64);
        reg.insert::<PipelineMsg>(downstream.clone(), Addr::new(downstream.clone(), tx));
        let reg = Arc::new(reg);
        let shared = Arc::new(SharedStore::new());
        let stop_flag = Arc::new(AtomicBool::new(false));
        let name = StageName::unnamed(StageKind::Sorter);
        let bx = crate::context::BuildCtx::new(&name, &reg, &shared, &stop_flag);

        let parts = Sorter::builder(name.clone(), 16)
            .downstream(downstream)
            .build()
            .into_parts();
        let sorter = (parts.factory)(&bx).expect("factory resolves");
        (sorter, rx)
    }

    fn make_ctx() -> Context<Sorter> {
        Context::new(
            StageName::unnamed(StageKind::Sorter),
            Arc::new(Registry::new()),
            Arc::new(SharedStore::new()),
            Arc::new(AtomicBool::new(false)),
        )
    }

    /// Reach in past the public API to seed `expected` for tests
    /// without minting MessageEx envelopes.
    fn register_frame_direct(sorter: &mut Sorter, source_id: &str, uuid: u128) {
        let entry = sorter.entry_mut(source_id);
        entry.expected_uuids.insert(uuid);
        entry.expected.push_back(Slot::Frame { uuid });
    }

    fn register_eos_direct(sorter: &mut Sorter, source_id: &str) {
        let entry = sorter.entry_mut(source_id);
        entry.expected.push_back(Slot::Eos);
    }

    #[test]
    fn register_frame_constructor_round_trips_through_message_ex() {
        let msg = Sorter::register_frame("cam-0", 42u128);
        let payload = match msg {
            PipelineMsg::MessageEx(p) => p,
            other => panic!("expected MessageEx, got {other:?}"),
        };
        let reg = payload
            .downcast::<SorterRegistration>()
            .expect("downcast must succeed");
        assert_eq!(
            *reg,
            SorterRegistration::Frame {
                source_id: "cam-0".into(),
                uuid: 42,
            }
        );
    }

    #[test]
    fn register_eos_constructor_round_trips_through_message_ex() {
        let msg = Sorter::register_eos("cam-1");
        let payload = match msg {
            PipelineMsg::MessageEx(p) => p,
            other => panic!("expected MessageEx, got {other:?}"),
        };
        let reg = payload
            .downcast::<SorterRegistration>()
            .expect("downcast must succeed");
        assert_eq!(
            *reg,
            SorterRegistration::Eos {
                source_id: "cam-1".into(),
            }
        );
    }

    #[test]
    fn builder_minimal_config_is_accepted() {
        let _ = Sorter::builder(StageName::unnamed(StageKind::Sorter), 4).build();
    }

    #[test]
    fn builder_accepts_full_config() {
        let _ = Sorter::builder(StageName::unnamed(StageKind::Sorter), 4)
            .downstream(StageName::unnamed(StageKind::Render))
            .inbox(SorterInbox::builder().build())
            .results(SorterResults::builder().build())
            .common(SorterCommon::builder().build())
            .build();
    }

    /// Runtime invariant: every hook slot is populated with a
    /// non-`Option` boxed closure after the factory runs.
    #[test]
    fn runtime_invariant_all_hooks_populated() {
        let (sorter, _rx) = make_sorter();
        let Sorter {
            on_message: _,
            on_source_eos: _,
            unregistered: _,
            stopping: _,
            per_source: _,
            ..
        } = sorter;
    }

    /// EOS slot drains via `on_source_eos`, forwarding
    /// [`PipelineMsg::SourceEos`] downstream and dropping the
    /// per-source state.
    #[test]
    fn registered_eos_drains_downstream() {
        let (mut sorter, rx) = make_sorter();
        let mut ctx = make_ctx();

        register_eos_direct(&mut sorter, "cam-0");
        let flow = sorter.drain("cam-0", &mut ctx).unwrap();
        assert!(matches!(flow, Flow::Cont));

        match rx.try_recv().expect("EOS must reach downstream") {
            PipelineMsg::SourceEos { source_id } => assert_eq!(source_id, "cam-0"),
            other => panic!("expected SourceEos, got {other:?}"),
        }
        assert!(
            !sorter.per_source.contains_key("cam-0"),
            "per-source state must be dropped after EOS"
        );
    }

    /// Ingress [`PipelineMsg::SourceEos`] is silently discarded —
    /// the sorter's EOS contract is driven by registrations.
    #[test]
    fn ingress_source_eos_is_silently_discarded() {
        let (mut sorter, rx) = make_sorter();
        let mut ctx = make_ctx();
        let _ = <Sorter as Handler<SourceEosPayload>>::handle(
            &mut sorter,
            SourceEosPayload {
                source_id: "cam-0".into(),
            },
            &mut ctx,
        )
        .unwrap();
        assert!(
            rx.try_recv().is_err(),
            "ingress SourceEos must not reach downstream"
        );
    }

    /// `handle_message_ex` consumes a [`SorterRegistration::Eos`]
    /// payload and drives the EOS through the downstream router.
    #[test]
    fn handle_message_ex_consumes_eos_registration() {
        let (mut sorter, rx) = make_sorter();
        let mut ctx = make_ctx();
        let payload = match Sorter::register_eos("cam-7") {
            PipelineMsg::MessageEx(p) => p,
            other => panic!("expected MessageEx, got {other:?}"),
        };
        sorter.handle_message_ex(payload, &mut ctx).unwrap();
        match rx.try_recv().expect("EOS must reach downstream") {
            PipelineMsg::SourceEos { source_id } => assert_eq!(source_id, "cam-7"),
            other => panic!("expected SourceEos, got {other:?}"),
        }
    }

    /// `handle_message_ex` ignores foreign payload types instead of
    /// erroring out.
    #[test]
    fn handle_message_ex_ignores_unknown_payload() {
        let (mut sorter, _rx) = make_sorter();
        let mut ctx = make_ctx();
        let payload = MessageExPayload::new("totally unrelated");
        let flow = sorter.handle_message_ex(payload, &mut ctx).unwrap();
        assert!(matches!(flow, Flow::Cont));
    }

    /// Registering a uuid twice for the same source is logged and
    /// otherwise treated as a no-op — neither slot duplicates the
    /// expected queue.
    #[test]
    fn duplicate_frame_registration_is_ignored() {
        let (mut sorter, _rx) = make_sorter();
        let mut ctx = make_ctx();

        let p1 = match Sorter::register_frame("cam-0", 11u128) {
            PipelineMsg::MessageEx(p) => p,
            _ => unreachable!(),
        };
        let p2 = match Sorter::register_frame("cam-0", 11u128) {
            PipelineMsg::MessageEx(p) => p,
            _ => unreachable!(),
        };
        sorter.handle_message_ex(p1, &mut ctx).unwrap();
        sorter.handle_message_ex(p2, &mut ctx).unwrap();

        let entry = sorter.per_source.get("cam-0").expect("entry must exist");
        assert_eq!(entry.expected.len(), 1, "duplicate must not be queued");
        assert_eq!(entry.expected_uuids.len(), 1);
    }

    /// `unregister_frame` constructor round-trips through
    /// MessageEx into the matching enum variant.
    #[test]
    fn unregister_frame_constructor_round_trips_through_message_ex() {
        let msg = Sorter::unregister_frame("cam-0", 99u128);
        let payload = match msg {
            PipelineMsg::MessageEx(p) => p,
            other => panic!("expected MessageEx, got {other:?}"),
        };
        let reg = payload
            .downcast::<SorterRegistration>()
            .expect("downcast must succeed");
        assert_eq!(
            *reg,
            SorterRegistration::Unregister {
                source_id: "cam-0".into(),
                uuid: 99,
            }
        );
    }

    /// `Unregister` splices the matching slot out of `expected`,
    /// removing it from `expected_uuids` so a later arrival of
    /// the abandoned uuid is treated as unregistered.
    #[test]
    fn unregister_splices_uuid_from_expected_queue() {
        let (mut sorter, _rx) = make_sorter();
        let mut ctx = make_ctx();

        register_frame_direct(&mut sorter, "cam-0", 1);
        register_frame_direct(&mut sorter, "cam-0", 2);
        register_frame_direct(&mut sorter, "cam-0", 3);

        let payload = match Sorter::unregister_frame("cam-0", 2u128) {
            PipelineMsg::MessageEx(p) => p,
            _ => unreachable!(),
        };
        sorter.handle_message_ex(payload, &mut ctx).unwrap();

        let entry = sorter.per_source.get("cam-0").expect("entry must exist");
        assert_eq!(entry.expected.len(), 2, "spliced slot must be removed");
        assert!(!entry.expected_uuids.contains(&2));
        assert!(entry.expected_uuids.contains(&1));
        assert!(entry.expected_uuids.contains(&3));
        // Order of the remaining slots must be preserved.
        match (
            entry.expected.front().expect("first remains"),
            entry.expected.back().expect("third remains"),
        ) {
            (Slot::Frame { uuid: u1 }, Slot::Frame { uuid: u3 }) => {
                assert_eq!(*u1, 1);
                assert_eq!(*u3, 3);
            }
            _ => panic!("expected Frame slots only"),
        }
    }

    /// `Unregister` of an unknown source / uuid is a no-op (logs
    /// at warn level and returns Cont).  No per-source state is
    /// created on the side, and the drain is still attempted.
    #[test]
    fn unregister_of_unknown_uuid_is_noop() {
        let (mut sorter, _rx) = make_sorter();
        let mut ctx = make_ctx();

        let payload = match Sorter::unregister_frame("ghost", 42u128) {
            PipelineMsg::MessageEx(p) => p,
            _ => unreachable!(),
        };
        let flow = sorter.handle_message_ex(payload, &mut ctx).unwrap();
        assert!(matches!(flow, Flow::Cont));
        assert!(
            !sorter.per_source.contains_key("ghost"),
            "unregister must not auto-create per-source state"
        );
    }

    /// After the head `Frame` slot is unregistered, the next slot's
    /// already-pending pair drains immediately without waiting for
    /// the retracted uuid's frame to arrive.
    #[test]
    fn unregister_unblocks_pending_drain_for_next_slot() {
        let (mut sorter, rx) = make_sorter();
        let mut ctx = make_ctx();

        // Queue: [F1, EOS]; F1 will be retracted, then EOS drains.
        register_frame_direct(&mut sorter, "cam-0", 1);
        register_eos_direct(&mut sorter, "cam-0");

        let payload = match Sorter::unregister_frame("cam-0", 1u128) {
            PipelineMsg::MessageEx(p) => p,
            _ => unreachable!(),
        };
        sorter.handle_message_ex(payload, &mut ctx).unwrap();

        // The retraction's drain pass advanced past the now-empty
        // slot to fire the EOS hook downstream.
        match rx.try_recv().expect("EOS must reach downstream") {
            PipelineMsg::SourceEos { source_id } => assert_eq!(source_id, "cam-0"),
            other => panic!("expected SourceEos, got {other:?}"),
        }
        assert!(
            !sorter.per_source.contains_key("cam-0"),
            "EOS must drop per-source state"
        );
    }

    /// The [`SorterRegistration`] enum survives the standard
    /// derives that consumers rely on (clone + PartialEq).
    #[test]
    fn sorter_registration_supports_clone_and_eq() {
        let f1 = SorterRegistration::Frame {
            source_id: "x".into(),
            uuid: 1,
        };
        let f2 = f1.clone();
        assert_eq!(f1, f2);

        let e1 = SorterRegistration::Eos {
            source_id: "x".into(),
        };
        let e2 = e1.clone();
        assert_eq!(e1, e2);

        assert_ne!(
            SorterRegistration::Eos {
                source_id: "a".into()
            },
            SorterRegistration::Eos {
                source_id: "b".into()
            }
        );
    }

    /// Registering frame uuids in order and then "delivering" those
    /// frames (via the public `ingest_pair` path) drains them in
    /// registration order — even when the deliveries arrive
    /// reversed.  Uses synthetic registrations so we don't have to
    /// build real `VideoFrame`s in unit tests; instead this test
    /// directly exercises the `pop_ready` state machine.
    #[test]
    fn pop_ready_releases_frames_in_registered_order() {
        let (mut sorter, _rx) = make_sorter();

        register_frame_direct(&mut sorter, "cam-0", 1);
        register_frame_direct(&mut sorter, "cam-0", 2);
        register_frame_direct(&mut sorter, "cam-0", 3);

        // Without any pending pairs, pop_ready must return None
        // (head is a frame slot whose pair has not arrived yet).
        assert!(matches!(sorter.pop_ready("cam-0"), None));

        // Slots remain intact in registration order.
        let entry = sorter.per_source.get("cam-0").unwrap();
        assert_eq!(entry.expected.len(), 3);
        assert_eq!(entry.expected_uuids.len(), 3);
    }
}
