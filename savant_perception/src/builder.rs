//! [`ActorBuilder`] and [`SourceBuilder`] — the registration
//! primitives consumed by
//! [`System::register_actor`](super::system::System::register_actor)
//! and
//! [`System::register_source`](super::system::System::register_source).
//!
//! A builder is **configuration + factory**: it carries the
//! actor's [`StageName`], inbox capacity, and lifecycle knobs
//! plus a `FnOnce(&BuildCtx) -> Result<A>` factory closure.  The
//! builder is the single hand-off point from user code into the
//! framework — once it is handed to `register_actor` /
//! `register_source` the framework owns the full construction
//! and thread-spawn path.
//!
//! These types do no I/O.  They are pure value constructors:
//!
//! * `register_actor` / `register_source` (called eagerly from
//!   user code) allocate each actor's bounded inbox channel and
//!   publish its [`Addr<M>`](super::addr::Addr) under its
//!   `StageName`.
//! * [`System::run`](super::system::System::run) (called once,
//!   when the topology is complete) invokes each captured
//!   factory against a [`BuildCtx`] populated with the registry
//!   and shared store, then spawns one OS thread per actor /
//!   source.

use std::time::Duration;

use anyhow::Result;

use super::actor::{Actor, Source};
use super::context::BuildCtx;
use super::supervisor::StageName;

/// Factory closure handed to an [`ActorBuilder`].  Invoked once
/// by [`System::run`](super::system::System::run), after every
/// peer's [`Addr`](super::addr::Addr) is already published in
/// the registry, so the factory can resolve peers eagerly and
/// produce a ready-to-run actor value by return.
pub type ActorFactory<A> = Box<dyn FnOnce(&BuildCtx) -> Result<A> + Send>;

/// Factory closure handed to a [`SourceBuilder`].  Same call-site
/// as [`ActorFactory`] — invoked once by `System::run` against a
/// fully-populated [`BuildCtx`], returns the source value.
pub type SourceFactory<S> = Box<dyn FnOnce(&BuildCtx) -> Result<S> + Send>;

/// Builder for an [`Actor`] — configures its [`StageName`],
/// inbox capacity, and lifecycle knobs, then carries the factory
/// closure that
/// [`System::run`](super::system::System::run) will invoke to
/// instantiate the actor.
///
/// Builders are intended to be constructed once via a fluent
/// API — typically from a higher-level *stage* (Layer B)
/// that sets opinionated defaults and only exposes the
/// actor-specific hooks to user code — but the raw surface here
/// is also available for free-form actors.
pub struct ActorBuilder<A: Actor> {
    name: StageName,
    inbox_capacity: usize,
    poll_timeout: Option<Duration>,
    factory: Option<ActorFactory<A>>,
}

impl<A: Actor> ActorBuilder<A> {
    /// Start a builder for an actor registered under `name` with
    /// inbox capacity `inbox_capacity`.
    ///
    /// The capacity bounds the per-peer queue depth and is the
    /// application's primary backpressure knob — small values keep
    /// memory footprint tight but throttle faster.
    pub fn new(name: StageName, inbox_capacity: usize) -> Self {
        Self {
            name,
            inbox_capacity,
            poll_timeout: None,
            factory: None,
        }
    }

    /// Override the actor's [`poll_timeout`](Actor::poll_timeout)
    /// for this instance.  When unset, the actor's trait default
    /// is used.
    pub fn poll_timeout(mut self, d: Duration) -> Self {
        self.poll_timeout = Some(d);
        self
    }

    /// Install the factory closure that
    /// [`System::run`](super::system::System::run) will invoke
    /// to instantiate the actor.
    ///
    /// The factory receives a [`BuildCtx`] with the fully-populated
    /// registry and shared store — resolve peers, read config,
    /// then return the actor value.
    pub fn factory<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&BuildCtx) -> Result<A> + Send + 'static,
    {
        self.factory = Some(Box::new(f));
        self
    }

    /// The actor's registered name.
    pub fn name(&self) -> &StageName {
        &self.name
    }

    /// The configured inbox capacity.
    pub fn inbox_capacity(&self) -> usize {
        self.inbox_capacity
    }

    /// The per-instance poll-timeout override, if any.
    pub fn poll_timeout_override(&self) -> Option<Duration> {
        self.poll_timeout
    }

    /// Internal: drain the builder into its parts.
    /// [`System::register_actor`](super::system::System::register_actor)
    /// uses these to allocate the inbox channel and capture the
    /// factory; [`System::run`](super::system::System::run) then
    /// invokes the factory before spawning the actor's thread.
    #[allow(dead_code, reason = "called from System::register_actor")]
    pub(crate) fn into_parts(self) -> ActorBuilderParts<A> {
        ActorBuilderParts {
            name: self.name,
            inbox_capacity: self.inbox_capacity,
            poll_timeout: self.poll_timeout,
            factory: self
                .factory
                .expect("ActorBuilder::factory(...) must be called before registration"),
        }
    }
}

#[allow(
    dead_code,
    reason = "fields read from System::register_actor / System::run"
)]
pub(crate) struct ActorBuilderParts<A: Actor> {
    pub(crate) name: StageName,
    pub(crate) inbox_capacity: usize,
    pub(crate) poll_timeout: Option<Duration>,
    pub(crate) factory: ActorFactory<A>,
}

/// Builder for a [`Source`] — no inbox capacity, just a name and a
/// factory.  Sources are no-inbox producers that push onto peer
/// `Addr<M>` handles obtained from the [`BuildCtx`].
pub struct SourceBuilder<S: Source> {
    name: StageName,
    factory: Option<SourceFactory<S>>,
}

impl<S: Source> SourceBuilder<S> {
    /// Start a builder for a source registered under `name`.
    pub fn new(name: StageName) -> Self {
        Self {
            name,
            factory: None,
        }
    }

    /// Install the factory closure.
    pub fn factory<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&BuildCtx) -> Result<S> + Send + 'static,
    {
        self.factory = Some(Box::new(f));
        self
    }

    /// The source's registered name.
    pub fn name(&self) -> &StageName {
        &self.name
    }

    /// Internal: drain the builder into its parts.
    /// [`System::register_source`](super::system::System::register_source)
    /// captures these; [`System::run`](super::system::System::run)
    /// invokes the factory and spawns the source's thread.
    #[allow(dead_code, reason = "called from System::register_source")]
    pub(crate) fn into_parts(self) -> SourceBuilderParts<S> {
        SourceBuilderParts {
            name: self.name,
            factory: self
                .factory
                .expect("SourceBuilder::factory(...) must be called before registration"),
        }
    }
}

#[allow(
    dead_code,
    reason = "fields read from System::register_source / System::run"
)]
pub(crate) struct SourceBuilderParts<S: Source> {
    pub(crate) name: StageName,
    pub(crate) factory: SourceFactory<S>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{BuildCtx, Context, SourceContext};
    use crate::handler::Flow;
    use crate::registry::Registry;
    use crate::shared::SharedStore;
    use crate::supervisor::StageKind;

    struct TestMsg;
    impl super::super::envelope::Envelope for TestMsg {
        fn as_shutdown(&self) -> Option<super::super::envelope::ShutdownHint<'_>> {
            None
        }
    }

    struct TestActor;
    impl Actor for TestActor {
        type Msg = TestMsg;
        fn handle(&mut self, _msg: Self::Msg, _ctx: &mut Context<Self>) -> Result<Flow> {
            Ok(Flow::Cont)
        }
    }

    #[test]
    fn builder_records_config_and_factory() {
        let name = StageName::unnamed(StageKind::Infer);
        let b: ActorBuilder<TestActor> = ActorBuilder::new(name.clone(), 4)
            .poll_timeout(Duration::from_millis(50))
            .factory(|_bx: &BuildCtx| Ok(TestActor));
        assert_eq!(b.name(), &name);
        assert_eq!(b.inbox_capacity(), 4);
        assert_eq!(b.poll_timeout_override(), Some(Duration::from_millis(50)));
        let parts = b.into_parts();
        assert_eq!(parts.inbox_capacity, 4);
        let reg = std::sync::Arc::new(Registry::new());
        let shared = std::sync::Arc::new(SharedStore::new());
        let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let bx = BuildCtx::new(&parts.name, &reg, &shared, &stop_flag);
        let actor = (parts.factory)(&bx).unwrap();
        let _ = actor;
    }

    #[test]
    #[should_panic(expected = "factory(...) must be called")]
    fn missing_factory_panics_on_into_parts() {
        let name = StageName::unnamed(StageKind::Infer);
        let b: ActorBuilder<TestActor> = ActorBuilder::new(name, 4);
        let _ = b.into_parts();
    }

    struct TestSource;
    impl Source for TestSource {
        fn run(self, _ctx: SourceContext) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn source_builder_round_trip() {
        let name = StageName::unnamed(StageKind::Mp4Demux);
        let b: SourceBuilder<TestSource> =
            SourceBuilder::new(name.clone()).factory(|_| Ok(TestSource));
        assert_eq!(b.name(), &name);
        let parts = b.into_parts();
        let reg = std::sync::Arc::new(Registry::new());
        let shared = std::sync::Arc::new(SharedStore::new());
        let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let bx = BuildCtx::new(&parts.name, &reg, &shared, &stop_flag);
        let s = (parts.factory)(&bx).unwrap();
        s.run(SourceContext::new(parts.name, reg, shared, stop_flag))
            .unwrap();
    }
}
