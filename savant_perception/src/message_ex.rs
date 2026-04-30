//! [`MessageExPayload`] — type-erased payload for the
//! `MessageEx` variant of [`PipelineMsg`](crate::envelopes::PipelineMsg)
//! and [`EncodedMsg`](crate::envelopes::EncodedMsg).
//!
//! `MessageEx` is the framework's extensibility escape hatch: a
//! single envelope variant that carries an arbitrary
//! receiver-specific payload type without forcing the envelope to
//! be parametric and without forcing every actor on the channel
//! to agree on the concrete type.  The payload is moved
//! (`Send`-only — single owner, single receiver) and erased
//! behind `Box<dyn Any + Send>`; the receiver downcasts to the
//! concrete `T` it expects.
//!
//! # Send / receive shape
//!
//! ```ignore
//! // Sender side — typed `T` boxed into the variant:
//! addr.send(EncodedMsg::message_ex(MyType { /* … */ }))?;
//!
//! // Receiver side — override `Actor::handle_message_ex` and
//! // downcast inside:
//! impl Actor for MyStage {
//!     type Msg = EncodedMsg;
//!     fn handle(&mut self, msg: EncodedMsg, ctx: &mut Context<Self>) -> Result<Flow> {
//!         msg.dispatch(self, ctx)
//!     }
//!     fn handle_message_ex(
//!         &mut self,
//!         msg: MessageExPayload,
//!         _ctx: &mut Context<Self>,
//!     ) -> Result<Flow> {
//!         match msg.downcast::<MyType>() {
//!             Ok(t) => self.handle_my_type(*t),
//!             Err(p) => log::warn!("unexpected MessageEx({})", p.type_name),
//!         }
//!         Ok(Flow::Cont)
//!     }
//! }
//! ```
//!
//! Stages that don't care about `MessageEx` need no changes —
//! the default [`Actor::handle_message_ex`](crate::Actor::handle_message_ex)
//! body drops the payload and emits a single `debug!` line.
//!
//! # Why type-erased
//!
//! The framework's channels are typed (`Sender<M>` per actor),
//! and existing pipelines route a single envelope type
//! (`PipelineMsg` or `EncodedMsg`) through every stage on a bus.
//! Adding a typed parameter (`PipelineMsg<C>`) would cascade
//! through every signature in the codebase and force every
//! pipeline to agree on a single `C`.  Type erasure preserves the
//! one-envelope-fits-all property at the cost of one
//! [`TypeId`](std::any::TypeId) check at the receiver.
//!
//! [`type_name`](MessageExPayload::type_name) is captured at
//! construction so logs can record what crossed the wire without
//! forcing the receiver to attempt a downcast.

use std::any::Any;
use std::fmt;

/// Type-erased payload for the `MessageEx` envelope variant.
///
/// Construct with [`MessageExPayload::new`]; recover the concrete
/// `T` with [`downcast`](Self::downcast).  The captured
/// [`type_name`](Self::type_name) is intended for diagnostics
/// only — it is not stable across compiler versions.
pub struct MessageExPayload {
    /// Pretty-printable name of the boxed `T`, captured via
    /// [`std::any::type_name::<T>`] at construction.  Suitable
    /// for `debug!` / `warn!` log lines; **not** a stable wire
    /// identifier.
    pub type_name: &'static str,
    value: Box<dyn Any + Send>,
}

impl MessageExPayload {
    /// Box `value` into a [`MessageExPayload`], capturing
    /// `std::any::type_name::<T>()` for diagnostics.
    pub fn new<T: Any + Send>(value: T) -> Self {
        Self {
            type_name: std::any::type_name::<T>(),
            value: Box::new(value),
        }
    }

    /// Attempt to recover the concrete `T` from this payload.
    ///
    /// * `Ok(Box<T>)` — the payload's underlying type matched and
    ///   ownership has been transferred to the caller.
    /// * `Err(self)` — the payload's underlying type did not
    ///   match `T`; the original `MessageExPayload` is returned
    ///   intact so the caller can inspect
    ///   [`type_name`](Self::type_name) and either retry with a
    ///   different `T`, forward, or drop.
    pub fn downcast<T: Any>(self) -> Result<Box<T>, Self> {
        let MessageExPayload { type_name, value } = self;
        match value.downcast::<T>() {
            Ok(boxed) => Ok(boxed),
            Err(value) => Err(Self { type_name, value }),
        }
    }
}

impl fmt::Debug for MessageExPayload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MessageExPayload")
            .field("type_name", &self.type_name)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct MyType {
        value: u32,
    }

    #[test]
    fn round_trip_via_downcast() {
        let payload = MessageExPayload::new(MyType { value: 42 });
        assert!(payload.type_name.contains("MyType"));
        let recovered = payload.downcast::<MyType>().expect("downcast must succeed");
        assert_eq!(*recovered, MyType { value: 42 });
    }

    #[test]
    fn downcast_wrong_type_returns_payload_intact() {
        let payload = MessageExPayload::new(MyType { value: 7 });
        let err = payload
            .downcast::<u64>()
            .err()
            .expect("u64 must not match MyType");
        assert!(err.type_name.contains("MyType"));
        // Caller can recover by trying the correct type next.
        let recovered = err.downcast::<MyType>().expect("retry must succeed");
        assert_eq!(recovered.value, 7);
    }

    /// Debug formatting prints the captured type name without
    /// touching the boxed value (which has no Debug bound).
    #[test]
    fn debug_includes_type_name() {
        let p = MessageExPayload::new(MyType { value: 1 });
        let s = format!("{p:?}");
        assert!(s.contains("MessageExPayload"));
        assert!(s.contains("MyType"));
    }
}
