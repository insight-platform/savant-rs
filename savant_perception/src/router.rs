//! [`Router<M>`] — name-routed send abstraction layered over
//! [`OperatorSink<M>`](super::operator_sink::OperatorSink).
//!
//! Every actor / source that forwards messages downstream holds a
//! `Router<M>` rather than a raw [`OperatorSink`].  The router
//! combines two capabilities:
//!
//! * An **optional default peer** — installed at build time from the
//!   builder's `.downstream(name)` call.  `router.send(msg)` routes
//!   to that peer without any runtime name lookup.
//! * **Name-based routing** — `router.send_to(&peer, msg)` resolves
//!   any registered peer by [`StageName`](super::supervisor::StageName)
//!   at call time, caching the resolved [`OperatorSink<M>`] so that
//!   subsequent sends to the same peer avoid the registry lookup.
//!
//! The router is the canonical send handle across templates.  Every
//! template's internal `send` sites, every operator-result-callback
//! argument that used to be `&OperatorSink<M>`, and every
//! construction-time factory that previously took a single peer now
//! uses `Router<M>`.
//!
//! # Cloning
//!
//! `Router<M>` is cheap to clone (`Arc<_>` under the hood) and every
//! clone shares the cache and the default sink's abort flag.  Hand
//! clones into operator-result callbacks freely — the same "first
//! failed send flips the aborted flag and warns once" contract from
//! [`OperatorSink`] carries over through the default peer.
//!
//! # No default peer
//!
//! A router built without a default silently drops any `send(msg)`
//! call after logging **once** — symmetrical with `OperatorSink`'s
//! abort behaviour.  `send_to(&peer, msg)` works regardless of
//! whether a default is installed; a missing default only matters
//! when user code calls the defaulted `send`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;

use super::envelope::Envelope;
use super::operator_sink::OperatorSink;
use super::registry::Registry;
use super::supervisor::StageName;

struct RouterInner<M: Envelope> {
    owner: StageName,
    registry: Arc<Registry>,
    default: Option<OperatorSink<M>>,
    cache: Mutex<HashMap<StageName, OperatorSink<M>>>,
    /// One-shot latch: the first time `send` is called without a
    /// default peer, log a `warn!`; subsequent calls short-circuit
    /// silently.  Mirrors the abort flag on [`OperatorSink`] so the
    /// callback thread never spams logs.
    warned_missing_default: AtomicBool,
}

/// Name-routed send handle.  See module docs for the full contract.
pub struct Router<M: Envelope>(Arc<RouterInner<M>>);

impl<M: Envelope> Router<M> {
    /// Construct a router owned by `owner` that draws on `registry`
    /// for name resolution.  `default` is the optional pre-bound
    /// sink used by [`send`](Router::send).
    ///
    /// Internal: user code should obtain routers via
    /// [`BuildCtx::router`](super::context::BuildCtx::router),
    /// [`Context::router`](super::context::Context::router), or
    /// [`SourceContext::router`](super::context::SourceContext::router).
    #[allow(dead_code, reason = "consumed by System in Component 2")]
    pub(crate) fn new(
        owner: StageName,
        registry: Arc<Registry>,
        default: Option<OperatorSink<M>>,
    ) -> Self {
        Self(Arc::new(RouterInner {
            owner,
            registry,
            default,
            cache: Mutex::new(HashMap::new()),
            warned_missing_default: AtomicBool::new(false),
        }))
    }

    /// This router's owning stage — the actor / source that sends
    /// through it.  Used for log attribution.
    pub fn owner(&self) -> &StageName {
        &self.0.owner
    }

    /// Whether a default peer was installed at build time.
    pub fn has_default(&self) -> bool {
        self.0.default.is_some()
    }

    /// Stage name of the default peer, if any.
    pub fn default_peer(&self) -> Option<&StageName> {
        self.0.default.as_ref().map(|s| s.peer())
    }

    /// Send to the default peer.
    ///
    /// Returns `true` on a successful send, `false` if:
    /// * no default peer was installed (logged once via `warn!`
    ///   before the latch flips);
    /// * the default peer's inbox is closed or has already been
    ///   flagged as aborted (see [`OperatorSink::send`]).
    pub fn send(&self, msg: M) -> bool {
        match self.0.default.as_ref() {
            Some(sink) => sink.send(msg),
            None => {
                if !self.0.warned_missing_default.swap(true, Ordering::Relaxed) {
                    log::warn!(
                        "[{}] router.send(...) called but no default peer was configured \
                         (builder's .downstream() was not called); dropping message and \
                         subsequent sends without a default are silent",
                        self.0.owner
                    );
                }
                false
            }
        }
    }

    /// Send to a peer resolved by name at call time.
    ///
    /// The resolved [`OperatorSink<M>`] is cached on first use and
    /// reused on every subsequent call for the same `peer`, so the
    /// registry lookup cost is paid once per distinct target.
    ///
    /// Returns:
    /// * `Ok(true)`  — the send succeeded.
    /// * `Ok(false)` — the cached sink for `peer` has been aborted
    ///   (inbox closed, observed by some earlier call).
    /// * `Err(_)`    — `peer` is not registered, or the registered
    ///   entry has a different envelope type than `M`.
    pub fn send_to(&self, peer: &StageName, msg: M) -> Result<bool> {
        let sink = self.resolve(peer)?;
        Ok(sink.send(msg))
    }

    /// Resolve (or reuse) the cached [`OperatorSink<M>`] for `peer`.
    ///
    /// Exposed so callers that need the sink handle itself (for
    /// example to observe `aborted`) can obtain it without calling
    /// `send_to`.
    pub fn sink_for(&self, peer: &StageName) -> Result<OperatorSink<M>> {
        self.resolve(peer)
    }

    /// Clone of the default peer's [`OperatorSink<M>`], if any.
    ///
    /// Use this when you need the raw `abort` / `aborted` primitives
    /// on the default peer (for example in a cooperative GStreamer
    /// callback that latches early-exit state).  Regular sends should
    /// go through [`Router::send`].
    pub fn default_sink(&self) -> Option<OperatorSink<M>> {
        self.0.default.clone()
    }

    fn resolve(&self, peer: &StageName) -> Result<OperatorSink<M>> {
        {
            let cache = self.0.cache.lock().expect("router cache poisoned");
            if let Some(sink) = cache.get(peer) {
                return Ok(sink.clone());
            }
        }
        let addr = self.0.registry.get::<M>(peer)?;
        let sink = OperatorSink::new(self.0.owner.clone(), addr);
        let mut cache = self.0.cache.lock().expect("router cache poisoned");
        let entry = cache.entry(peer.clone()).or_insert(sink);
        Ok(entry.clone())
    }
}

impl<M: Envelope> Clone for Router<M> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<M: Envelope> std::fmt::Debug for Router<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Router")
            .field("owner", &self.0.owner)
            .field("default_peer", &self.default_peer())
            .field(
                "cached_peers",
                &self
                    .0
                    .cache
                    .lock()
                    .ok()
                    .map(|c| c.len())
                    .unwrap_or_default(),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::addr::Addr;
    use crate::envelope::ShutdownHint;
    use crate::supervisor::StageKind;
    use crossbeam::channel::{bounded, Receiver};

    #[derive(Debug, PartialEq, Eq)]
    struct MsgA(u32);
    impl Envelope for MsgA {
        fn as_shutdown(&self) -> Option<ShutdownHint<'_>> {
            None
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    struct MsgB;
    impl Envelope for MsgB {
        fn as_shutdown(&self) -> Option<ShutdownHint<'_>> {
            None
        }
    }

    fn owner_name() -> StageName {
        StageName::unnamed(StageKind::Infer)
    }

    fn register<M: Envelope>(
        reg: &mut Registry,
        name: StageName,
        cap: usize,
    ) -> (StageName, Receiver<M>) {
        let (tx, rx) = bounded::<M>(cap);
        reg.insert::<M>(name.clone(), Addr::new(name.clone(), tx));
        (name, rx)
    }

    #[test]
    fn send_uses_default_peer() {
        let mut reg = Registry::new();
        let (tracker, rx) = register::<MsgA>(&mut reg, StageName::unnamed(StageKind::Tracker), 2);
        let addr: Addr<MsgA> = reg.get::<MsgA>(&tracker).unwrap();
        let default = OperatorSink::new(owner_name(), addr);
        let r = Router::new(owner_name(), Arc::new(reg), Some(default));
        assert!(r.has_default());
        assert_eq!(r.default_peer(), Some(&tracker));
        assert!(r.send(MsgA(1)));
        assert_eq!(rx.try_recv().unwrap(), MsgA(1));
    }

    #[test]
    fn send_without_default_warns_once_and_returns_false() {
        let reg = Registry::new();
        let r: Router<MsgA> = Router::new(owner_name(), Arc::new(reg), None);
        assert!(!r.has_default());
        assert!(!r.send(MsgA(1)));
        assert!(!r.send(MsgA(2)), "subsequent sends also return false");
        assert!(
            r.0.warned_missing_default.load(Ordering::Relaxed),
            "warn latch must be set after first send"
        );
    }

    #[test]
    fn send_to_resolves_and_caches() {
        let mut reg = Registry::new();
        let (picasso, rx1) = register::<MsgA>(&mut reg, StageName::unnamed(StageKind::Picasso), 2);
        let (blackhole, rx2) =
            register::<MsgA>(&mut reg, StageName::unnamed(StageKind::Function), 2);
        let r: Router<MsgA> = Router::new(owner_name(), Arc::new(reg), None);

        assert!(r.send_to(&picasso, MsgA(1)).unwrap());
        assert_eq!(rx1.try_recv().unwrap(), MsgA(1));
        assert!(r.send_to(&blackhole, MsgA(2)).unwrap());
        assert_eq!(rx2.try_recv().unwrap(), MsgA(2));

        let cache = r.0.cache.lock().unwrap();
        assert!(cache.contains_key(&picasso));
        assert!(cache.contains_key(&blackhole));
        assert_eq!(cache.len(), 2);
        drop(cache);

        assert!(r.send_to(&picasso, MsgA(3)).unwrap());
        assert_eq!(rx1.try_recv().unwrap(), MsgA(3));
        assert_eq!(
            r.0.cache.lock().unwrap().len(),
            2,
            "cache size unchanged on repeat send_to"
        );
    }

    #[test]
    fn send_to_unknown_peer_errors() {
        let reg = Registry::new();
        let r: Router<MsgA> = Router::new(owner_name(), Arc::new(reg), None);
        let err = r
            .send_to(&StageName::unnamed(StageKind::Picasso), MsgA(1))
            .unwrap_err();
        assert!(err.to_string().contains("no stage"));
    }

    #[test]
    fn send_to_wrong_envelope_type_errors() {
        let mut reg = Registry::new();
        let (name, _rx) = register::<MsgB>(&mut reg, StageName::unnamed(StageKind::Picasso), 1);
        let r: Router<MsgA> = Router::new(owner_name(), Arc::new(reg), None);
        let err = r.send_to(&name, MsgA(1)).unwrap_err();
        assert!(err.to_string().contains("different message type"));
    }

    #[test]
    fn clones_share_cache_and_abort_state() {
        let mut reg = Registry::new();
        let (tracker, rx) = register::<MsgA>(&mut reg, StageName::unnamed(StageKind::Tracker), 2);
        let addr: Addr<MsgA> = reg.get::<MsgA>(&tracker).unwrap();
        let default = OperatorSink::new(owner_name(), addr);
        let r = Router::new(owner_name(), Arc::new(reg), Some(default));
        let r2 = r.clone();

        drop(rx);
        assert!(!r.send(MsgA(1)), "send must fail when receiver dropped");
        assert!(
            !r2.send(MsgA(2)),
            "clone observes the abort flag latched by the sibling"
        );
    }

    #[test]
    fn clone_shares_resolved_cache() {
        let mut reg = Registry::new();
        let (picasso, rx) = register::<MsgA>(&mut reg, StageName::unnamed(StageKind::Picasso), 2);
        let r: Router<MsgA> = Router::new(owner_name(), Arc::new(reg), None);
        let r2 = r.clone();

        r.send_to(&picasso, MsgA(1)).unwrap();
        assert_eq!(rx.try_recv().unwrap(), MsgA(1));
        assert_eq!(
            r2.0.cache.lock().unwrap().len(),
            1,
            "clone sees the same cache populated by the sibling"
        );
    }
}
