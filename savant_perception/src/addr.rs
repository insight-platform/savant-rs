//! [`Addr<M>`] — typed, clonable handle to an actor's inbox.
//!
//! Every actor allocates exactly one bounded
//! [`crossbeam::channel`] inbox of its envelope type when it is
//! registered with [`System`](super::system::System).  The
//! registry publishes the sending half as an `Addr<M>` keyed by
//! the actor's [`StageName`]; peer
//! actors look it up in their
//! [`BuildCtx`](super::context::BuildCtx) /
//! [`Context`](super::context::Context) at construction time or
//! at runtime.
//!
//! An `Addr<M>` is a cheap (two `Arc`-sized fields) clone: duplicate
//! it freely and hand clones to callback threads, child tasks, or
//! fan-out helpers.  All clones share the same underlying channel —
//! one clone dropping does **not** close the channel while other
//! senders (including receivers) remain alive.

use crossbeam::channel::{SendError, Sender, TrySendError};

use super::envelope::Envelope;
use super::supervisor::StageName;

/// Typed handle to an actor's inbox.
///
/// Holds a cloneable [`Sender`] plus
/// the peer's [`StageName`] for log records and introspection.
/// `send` / `try_send` push onto the underlying bounded channel;
/// [`name`](Addr::name) returns the peer identity.
pub struct Addr<M: Envelope> {
    name: StageName,
    tx: Sender<M>,
}

impl<M: Envelope> Addr<M> {
    /// Internal constructor — the framework builds these inside
    /// [`System::register_actor`](super::system::System::register_actor)
    /// when the actor's inbox channel is allocated.  User code
    /// should always obtain `Addr`s via
    /// [`BuildCtx::addr`](super::context::BuildCtx::addr) or
    /// [`Context::resolve`](super::context::Context::resolve).
    #[allow(dead_code, reason = "constructed by System on actor registration")]
    pub(crate) fn new(name: StageName, tx: Sender<M>) -> Self {
        Self { name, tx }
    }

    /// Stage name of the peer this address routes to.
    pub fn name(&self) -> &StageName {
        &self.name
    }

    /// Blocking send.  Returns `Err(SendError(msg))` iff every
    /// receiver has dropped — in which case the peer has already
    /// exited and the caller typically wants to flip an aborted
    /// flag (see [`OperatorSink`](super::operator_sink::OperatorSink)).
    pub fn send(&self, msg: M) -> Result<(), SendError<M>> {
        self.tx.send(msg)
    }

    /// Non-blocking send.  Returns `Err(TrySendError::Full(msg))`
    /// if the inbox is at capacity, `Err(TrySendError::Disconnected(msg))`
    /// if the peer has exited.
    pub fn try_send(&self, msg: M) -> Result<(), TrySendError<M>> {
        self.tx.try_send(msg)
    }

    /// Whether every receiver on the underlying channel has
    /// dropped.  A disconnected peer can only be spotted by a
    /// failed `send` / `try_send` (the sender side has no cheap
    /// is-disconnected check in crossbeam) — this helper routes
    /// through an empty `try_send` of a best-effort probe, but
    /// because constructing a probe value is impossible in
    /// generic `M`, we instead expose the sender's `len`/capacity
    /// semantics via [`Addr::capacity`] and recommend treating a
    /// failed `send` as the canonical signal.
    pub fn capacity(&self) -> Option<usize> {
        self.tx.capacity()
    }

    /// Current number of queued messages on the inbox.
    pub fn queue_len(&self) -> usize {
        self.tx.len()
    }
}

impl<M: Envelope> Clone for Addr<M> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            tx: self.tx.clone(),
        }
    }
}

impl<M: Envelope> std::fmt::Debug for Addr<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Addr")
            .field("peer", &self.name)
            .field(
                "queue",
                &format_args!(
                    "{}/{}",
                    self.tx.len(),
                    self.tx
                        .capacity()
                        .map(|c| c.to_string())
                        .unwrap_or_else(|| "∞".into())
                ),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::supervisor::StageKind;
    use crate::{Envelope, ShutdownHint};
    use crossbeam::channel::bounded;

    struct Ping;
    impl Envelope for Ping {
        fn as_shutdown(&self) -> Option<ShutdownHint<'_>> {
            None
        }
    }

    #[test]
    fn send_and_clone_round_trip() {
        let (tx, rx) = bounded::<Ping>(2);
        let a = Addr::new(StageName::unnamed(StageKind::Infer), tx);
        let b = a.clone();
        a.send(Ping).unwrap();
        b.send(Ping).unwrap();
        assert!(rx.try_recv().is_ok());
        assert!(rx.try_recv().is_ok());
        assert_eq!(a.name().kind, StageKind::Infer);
        assert_eq!(b.name().kind, StageKind::Infer);
    }

    #[test]
    fn try_send_reports_full_when_bounded() {
        let (tx, _rx) = bounded::<Ping>(1);
        let a = Addr::new(StageName::unnamed(StageKind::Infer), tx);
        a.try_send(Ping).unwrap();
        match a.try_send(Ping) {
            Err(crossbeam::channel::TrySendError::Full(_)) => {}
            other => panic!("expected Full, got {other:?}"),
        }
    }

    #[test]
    fn send_reports_err_when_receiver_dropped() {
        let (tx, rx) = bounded::<Ping>(1);
        let a = Addr::new(StageName::unnamed(StageKind::Infer), tx);
        drop(rx);
        assert!(a.send(Ping).is_err());
    }

    #[test]
    fn capacity_and_queue_len_reflect_backing_channel() {
        let (tx, _rx) = bounded::<Ping>(3);
        let a = Addr::new(StageName::unnamed(StageKind::Infer), tx);
        assert_eq!(a.capacity(), Some(3));
        assert_eq!(a.queue_len(), 0);
        a.try_send(Ping).unwrap();
        assert_eq!(a.queue_len(), 1);
    }
}
