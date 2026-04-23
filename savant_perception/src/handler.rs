//! [`Flow`] + [`Handler<M>`] — the per-variant handler trait that
//! Actix-style dispatch routes each envelope variant to.
//!
//! An actor implements [`Handler<V>`] for each envelope variant it
//! wants to react to.  The default method body drops the message
//! and returns [`Flow::Cont`], so opting *in* to a variant you
//! don't need to do anything special for is a zero-body impl:
//!
//! ```ignore
//! impl Handler<StreamInfo> for Decoder {}
//! ```
//!
//! (The receiver still gets ownership of `StreamInfo`, but the
//! default body simply drops it.)
//!
//! For variants you *do* care about, override `handle`:
//!
//! ```ignore
//! impl Handler<Delivery> for NvInfer {
//!     fn handle(&mut self, msg: Delivery, ctx: &mut Context<Self>) -> Result<Flow> {
//!         self.operator.add_frame(msg.frame, msg.buffer)?;
//!         Ok(Flow::Cont)
//!     }
//! }
//! ```
//!
//! The dispatch glue is generated once per envelope in a
//! [`Dispatch`](super::envelope::Dispatch) impl.

/// Result of a single [`Handler::handle`] call (or
/// [`Actor::on_tick`](super::actor::Actor::on_tick)) — tells the
/// receive-loop driver whether to keep running or break out and
/// enter the [`stopping`](super::actor::Actor::stopping) phase.
///
/// An error (`Err(_)` returned from the handler) short-circuits the
/// loop with the error as its result; the driver still invokes
/// `stopping` and the [`StageExitGuard`](super::supervisor::StageExitGuard)
/// still fires on its `Drop`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    /// Continue the receive loop: fetch the next envelope, fire a
    /// tick on timeout, etc.
    Cont,
    /// Break the receive loop after this call.  Used by terminal
    /// actors (e.g. the muxer finalising on
    /// [`EncodedMsg::SourceEos`](crate::envelopes::EncodedMsg::SourceEos))
    /// and by free-form actors that want to exit mid-batch.
    Stop,
}

/// Per-envelope-variant handler trait, implemented once per
/// `(Actor, V)` pair.
///
/// The default method body drops the message and returns
/// [`Flow::Cont`], so opting in to a variant with "observe and
/// continue" semantics is a zero-body impl — see the module docs
/// for an example.
///
/// `V` is the **payload type** of an envelope variant — e.g.
/// [`Delivery`](crate::envelopes::PipelineMsg), the
/// individual `SealedDeliveries` batch, a per-source `SourceEos`
/// struct, etc.  The [`Dispatch`](super::envelope::Dispatch) impl
/// for the parent envelope takes care of destructuring the
/// envelope variant and handing its payload to this trait.
pub trait Handler<V>: super::actor::Actor {
    /// Handle one message of variant `V`.
    ///
    /// The default body drops the message and returns
    /// [`Flow::Cont`]; override for domain behaviour.  Returning
    /// `Ok(Flow::Stop)` or `Err(_)` exits the receive loop.
    fn handle(&mut self, msg: V, ctx: &mut super::context::Context<Self>) -> anyhow::Result<Flow> {
        let _ = (msg, ctx);
        Ok(Flow::Cont)
    }
}

#[cfg(test)]
mod tests {
    use super::Flow;

    #[test]
    fn flow_equality_is_derived() {
        assert_eq!(Flow::Cont, Flow::Cont);
        assert_ne!(Flow::Cont, Flow::Stop);
    }
}
