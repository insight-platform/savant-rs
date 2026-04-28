//! [`Envelope`] and [`Dispatch`] — the type-level contract every
//! inter-actor message enum must satisfy.
//!
//! An **envelope** is the enum that actually travels through an
//! [`Addr<M>`](super::addr::Addr): the receiver's inbox sees it, the
//! receive-loop inspects it for a shutdown hint, and then it's
//! either handed to the user's [`Handler<V>`](super::handler::Handler)
//! impls (via [`Dispatch`]) or consumed directly by a free-form
//! actor's [`handle`](super::actor::Actor::handle) override.
//!
//! The [`ShutdownHint`] returned by
//! [`Envelope::as_shutdown`] is the framework's **only** peek at
//! the envelope's contents — it tells the loop driver whether the
//! current message is a cooperative shutdown sentinel so the driver
//! can arm a deadline or an immediate break without any
//! domain-specific knowledge.
//!
//! [`Dispatch`] is the Actix-style message-dispatch glue.  It is
//! implemented once per envelope (typically alongside the envelope
//! definition) and routes each variant to the corresponding
//! `Handler<V>` impl on the actor, keeping
//! [`Actor::handle`](super::actor::Actor::handle) a single
//! boilerplate-free line.

use std::borrow::Cow;
use std::time::Duration;

/// Per-message hint surfaced to the receive-loop driver so it can
/// honour cooperative-shutdown semantics without knowing the shape
/// of the envelope.
///
/// The two cases mirror the supervisor's two shutdown shapes:
///
/// * [`ShutdownHint::Graceful`] — arm a deadline of `grace` (if
///   `Some`) *or* break after the current message (if `None`);
///   the earliest of any existing deadline wins.
/// * [`ShutdownHint::Abort`] — break immediately after the
///   current message. Reserved for applications that ship a
///   hard-stop sentinel alongside the graceful one.
///
/// Both variants include a human-readable `reason` used purely for
/// logs.
#[derive(Debug, Clone, Copy)]
pub enum ShutdownHint<'a> {
    /// Cooperative stop.  With `grace = None`, exit after this
    /// message is handled.  With `grace = Some(d)`, keep running
    /// until the loop observes `now >= deadline`.
    Graceful {
        /// Grace period (or `None` to exit immediately after
        /// this message).
        grace: Option<Duration>,
        /// Log reason.
        reason: &'a str,
    },
    /// Hard stop — break immediately after this message.
    Abort {
        /// Log reason.
        reason: &'a str,
    },
}

/// Trait every envelope (the enum carried on an actor's inbox) must
/// implement so the receive-loop can honour cooperative-shutdown
/// semantics.
///
/// A single method — [`Envelope::as_shutdown`] — returns `Some(hint)`
/// iff the envelope *is* a shutdown sentinel (the loop driver
/// reacts to the hint) or `None` otherwise (the loop driver hands
/// the envelope to [`Actor::handle`](super::actor::Actor::handle)
/// untouched).
///
/// # Implementation sketch
///
/// ```
/// use std::borrow::Cow;
/// use std::time::Duration;
/// use savant_perception::{Envelope, ShutdownHint};
///
/// enum MyMsg {
///     Work(u32),
///     Shutdown { grace: Option<Duration>, reason: Cow<'static, str> },
/// }
///
/// impl Envelope for MyMsg {
///     fn as_shutdown(&self) -> Option<ShutdownHint<'_>> {
///         match self {
///             MyMsg::Shutdown { grace, reason } => {
///                 Some(ShutdownHint::Graceful { grace: *grace, reason })
///             }
///             _ => None,
///         }
///     }
/// }
/// ```
pub trait Envelope: Send + 'static {
    /// Classify this envelope for the receive-loop driver.
    ///
    /// Return `None` for a payload message (delivery / batch /
    /// per-source EOS / stream info / …) — the driver hands the
    /// envelope straight to the actor.
    ///
    /// Return `Some(ShutdownHint)` for a cooperative-shutdown
    /// sentinel — the driver arms its deadline / break-now state
    /// accordingly and then (still) forwards the envelope to
    /// [`Actor::handle`](super::actor::Actor::handle) so free-form
    /// actors get a chance to observe the sentinel directly.
    fn as_shutdown(&self) -> Option<ShutdownHint<'_>>;

    /// Construct a cooperative-shutdown envelope of this type.
    ///
    /// Called by [`System`](super::system::System)'s supervisor
    /// to broadcast an in-band `Shutdown` sentinel onto every
    /// actor's inbox when the user-installed or default
    /// [`ShutdownHandler`](super::shutdown::ShutdownHandler)
    /// resolves to an action that broadcasts.
    ///
    /// Return `None` to signal that this envelope does not carry
    /// a shutdown sentinel — the supervisor will fall back to the
    /// shared stop flag (checked by
    /// [`Context::should_quit`](super::context::Context::should_quit)).
    /// Most pipeline envelopes carry a dedicated `Shutdown`
    /// variant so they can guarantee FIFO drain; providing this
    /// method enables that variant.
    fn build_shutdown(grace: Option<Duration>, reason: Cow<'static, str>) -> Option<Self>
    where
        Self: Sized,
    {
        let _ = (grace, reason);
        None
    }
}

/// Message-dispatch glue: routes each variant of an envelope enum
/// to the appropriate [`Handler<V>`](super::handler::Handler) impl
/// on the actor.
///
/// Implemented once per envelope (typically right next to the
/// envelope definition) so every actor that handles the envelope
/// shares a single, reviewed dispatch table.  Actors opt in to
/// variants by writing `impl Handler<Variant> for MyActor {}`; the
/// default method body on [`Handler`](super::handler::Handler)
/// drops the message and continues so "I don't care about this
/// variant" requires no method body — only an empty impl.
///
/// Actors that prefer to handle every variant inline can bypass
/// this machinery entirely by overriding
/// [`Actor::handle`](super::actor::Actor::handle) and pattern-
/// matching the envelope themselves.
pub trait Dispatch<A>: Envelope
where
    A: super::actor::Actor<Msg = Self>,
{
    /// Route `self` to the `Handler<V>` impl for the matching
    /// envelope variant, returning the resulting
    /// [`Flow`](super::handler::Flow).
    fn dispatch(
        self,
        actor: &mut A,
        ctx: &mut super::context::Context<A>,
    ) -> anyhow::Result<super::handler::Flow>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    /// Minimal envelope used by framework unit tests.
    enum TestMsg {
        Ping,
        Shutdown {
            grace: Option<Duration>,
            reason: Cow<'static, str>,
        },
    }

    impl Envelope for TestMsg {
        fn as_shutdown(&self) -> Option<ShutdownHint<'_>> {
            match self {
                TestMsg::Shutdown { grace, reason } => Some(ShutdownHint::Graceful {
                    grace: *grace,
                    reason,
                }),
                _ => None,
            }
        }
    }

    #[test]
    fn payload_variant_has_no_shutdown_hint() {
        assert!(TestMsg::Ping.as_shutdown().is_none());
    }

    #[test]
    fn shutdown_variant_surfaces_graceful_hint() {
        let msg = TestMsg::Shutdown {
            grace: Some(Duration::from_millis(50)),
            reason: Cow::Borrowed("bye"),
        };
        match msg.as_shutdown().expect("shutdown variant must hint") {
            ShutdownHint::Graceful { grace, reason } => {
                assert_eq!(grace, Some(Duration::from_millis(50)));
                assert_eq!(reason, "bye");
            }
            other => panic!("expected Graceful, got {other:?}"),
        }
    }
}
