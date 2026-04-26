//! [`OperatorSink<M>`] — typed adapter for operator result
//! callbacks.
//!
//! Several wrapped operators (nvinfer, nvtracker, Picasso,
//! FlexibleDecoderPool) surface output through callback threads
//! owned by the operator itself.  Those callbacks need to forward
//! payloads *into* the next actor's inbox, but also need to behave
//! correctly when the inbox is closed — specifically:
//!
//! * Log **once** per aborted stream.
//! * Treat a drop as a soft error: the producer callback stays
//!   healthy, only the destination flipping the `aborted` flag.
//! * Be cheap to clone into every callback invocation (callbacks
//!   are hot-path, sometimes per-frame).
//!
//! `OperatorSink<M>` wraps an [`Addr<M>`] with an `Arc<AtomicBool>`
//! aborted flag and exposes a [`send`](OperatorSink::send) helper
//! that encapsulates the "send or flip aborted + warn" logic in
//! one place.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::addr::Addr;
use super::envelope::Envelope;
use super::supervisor::StageName;

/// Callback-friendly adapter around an [`Addr<M>`].
///
/// Hand one to every operator result callback that forwards into
/// `peer`.  The first failed [`send`](OperatorSink::send) flips
/// the aborted flag and emits a single `warn!` log record; every
/// subsequent send short-circuits silently so the callback thread
/// stays responsive but never spams logs.
///
/// Cloning an `OperatorSink` is cheap — it's a pair of
/// `Arc` clones and a `StageName` clone (the `StageName` itself
/// holds a `Cow<'static, str>` so the common case clones a
/// refcount-free static pointer).
pub struct OperatorSink<M: Envelope> {
    owner: StageName,
    addr: Addr<M>,
    aborted: Arc<AtomicBool>,
}

impl<M: Envelope> OperatorSink<M> {
    /// Construct a sink owned by `owner` that forwards into
    /// `addr`.  `owner` is used for log-record attribution; `addr`
    /// targets the peer.
    ///
    /// Internal: user code should obtain sinks via
    /// [`BuildCtx::sink`](super::context::BuildCtx::sink) or
    /// [`Context::sink`](super::context::Context::sink).
    #[allow(
        dead_code,
        reason = "called by BuildCtx/Context when handing a sink to user code"
    )]
    pub(crate) fn new(owner: StageName, addr: Addr<M>) -> Self {
        Self {
            owner,
            addr,
            aborted: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Peer this sink forwards into.
    pub fn peer(&self) -> &StageName {
        self.addr.name()
    }

    /// Owner stage — the actor this sink was built for.
    pub fn owner(&self) -> &StageName {
        &self.owner
    }

    /// Whether the destination inbox has been observed closed by
    /// any earlier `send` on **any clone** of this sink.
    pub fn aborted(&self) -> bool {
        self.aborted.load(Ordering::Relaxed)
    }

    /// Attempt a blocking `send` on the inbox.
    ///
    /// Returns `true` on success, `false` if the sink has already
    /// been aborted (no attempt made) or if the send itself fails
    /// (first-time failure flips the aborted flag and emits a
    /// `warn!` log).
    pub fn send(&self, msg: M) -> bool {
        if self.aborted.load(Ordering::Relaxed) {
            return false;
        }
        if self.addr.send(msg).is_err() {
            if !self.aborted.swap(true, Ordering::Relaxed) {
                log::warn!(
                    "[{}] operator sink to [{}] aborted (inbox closed); dropping future messages silently",
                    self.owner,
                    self.addr.name()
                );
            }
            return false;
        }
        true
    }

    /// Force the aborted flag without attempting a send — useful
    /// when an operator callback observes a local error and wants
    /// to stop forwarding without waiting for a `send` failure.
    pub fn abort(&self) {
        if !self.aborted.swap(true, Ordering::Relaxed) {
            log::warn!(
                "[{}] operator sink to [{}] aborted explicitly",
                self.owner,
                self.addr.name()
            );
        }
    }
}

impl<M: Envelope> Clone for OperatorSink<M> {
    fn clone(&self) -> Self {
        Self {
            owner: self.owner.clone(),
            addr: self.addr.clone(),
            aborted: self.aborted.clone(),
        }
    }
}

impl<M: Envelope> std::fmt::Debug for OperatorSink<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OperatorSink")
            .field("owner", &self.owner)
            .field("peer", self.addr.name())
            .field("aborted", &self.aborted())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::envelope::ShutdownHint;
    use crate::supervisor::StageKind;
    use crossbeam::channel::bounded;

    struct Ping;
    impl Envelope for Ping {
        fn as_shutdown(&self) -> Option<ShutdownHint<'_>> {
            None
        }
    }

    fn sink(cap: usize) -> (OperatorSink<Ping>, crossbeam::channel::Receiver<Ping>) {
        let (tx, rx) = bounded::<Ping>(cap);
        let addr = Addr::new(StageName::unnamed(StageKind::Tracker), tx);
        (
            OperatorSink::new(StageName::unnamed(StageKind::Infer), addr),
            rx,
        )
    }

    #[test]
    fn successful_send_returns_true() {
        let (s, rx) = sink(1);
        assert!(s.send(Ping));
        assert!(rx.try_recv().is_ok());
    }

    #[test]
    fn aborted_send_returns_false_and_does_not_panic() {
        let (s, rx) = sink(1);
        drop(rx);
        assert!(!s.send(Ping));
        assert!(s.aborted(), "first failed send must flip the aborted flag");
        assert!(
            !s.send(Ping),
            "subsequent sends on aborted sink short-circuit"
        );
    }

    #[test]
    fn explicit_abort_flips_flag_without_send() {
        let (s, _rx) = sink(1);
        assert!(!s.aborted());
        s.abort();
        assert!(s.aborted());
    }

    #[test]
    fn clones_share_aborted_flag() {
        let (s, rx) = sink(1);
        let clone = s.clone();
        drop(rx);
        s.send(Ping);
        assert!(clone.aborted(), "aborted flag is shared across clones");
    }

    #[test]
    fn peer_and_owner_are_distinguishable() {
        let (s, _rx) = sink(1);
        assert_eq!(s.owner().kind, StageKind::Infer);
        assert_eq!(s.peer().kind, StageKind::Tracker);
    }
}
